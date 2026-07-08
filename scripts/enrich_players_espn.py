"""
Enrich the players table with ESPN Cricinfo bio data via the cricdata SDK.

cricsheet identifies players by an initials+surname key (e.g. "AJ Finch"),
which has no direct ESPN player ID. This script:

  1. Searches ESPN by surname for each player.
  2. Filters candidates to those matching country + first-initial.
  3. Accepts an automated match only when exactly one candidate survives —
     ambiguous/unmatched players are logged, never guessed.
  4. Fetches player_bio() for matched players and fills only NULL columns
     on `players` (full_name, batting_style, bowling_style, date_of_birth,
     country) — never overwrites existing values.

Writes a crosswalk table `player_espn_map` for auditability, and a JSONL
cache of raw ESPN responses so a re-run doesn't re-fetch anything.

Usage:
  python scripts/enrich_players_espn.py --limit 100          # pilot
  python scripts/enrich_players_espn.py                      # full run
  python scripts/enrich_players_espn.py --report-only        # just print stats
"""

import datetime
import json
import sys
import time
from pathlib import Path

import click
from sqlalchemy import text, Column, Integer, String, Float
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
from src.db.schema import get_engine, Base, Player

from cricdata import CricinfoClient

CACHE_DIR = Path(__file__).parent.parent / "data" / "espn_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_CACHE = CACHE_DIR / "search_cache.jsonl"
BIO_CACHE = CACHE_DIR / "bio_cache.jsonl"
REQUEST_DELAY_S = 0.6
MAX_RETRIES = 4


def _with_retry(fn, *args, **kwargs):
    """Retry on transient HTTP errors (e.g. 503 rate-limit) with backoff."""
    delay = 2.0
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs), None
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay *= 2
    return None, last_err


# ---------------------------------------------------------------------------
# Crosswalk table
# ---------------------------------------------------------------------------

class PlayerEspnMap(Base):
    __tablename__ = "player_espn_map"
    player_id = Column(Integer, primary_key=True)
    espn_id = Column(String, nullable=True)
    status = Column(String, nullable=False)   # matched | ambiguous | not_found | error
    candidates = Column(Integer, default=0)
    reason = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

SURNAME_PARTICLES = {"de", "van", "der", "du", "al", "abu", "bin", "el"}


def split_key(cricsheet_key: str):
    """'DAS Gunaratne' -> ('DAS', 'Gunaratne'); handles 'van der Merwe' etc."""
    # strip cricsheet disambiguator suffixes: "Mohammad Nawaz (3)" -> "Mohammad Nawaz"
    key = cricsheet_key.strip()
    if key.endswith(")") and "(" in key:
        key = key[: key.rindex("(")].strip()
    parts = key.split()
    if len(parts) < 2:
        return "", cricsheet_key
    initials = parts[0]
    surname_parts = parts[1:]
    # pull any lowercase-led surname particles into the surname
    while len(surname_parts) > 1 and surname_parts[0].lower() in SURNAME_PARTICLES:
        break  # already part of surname_parts; nothing to hoist from initials side
    surname = " ".join(surname_parts)
    return initials, surname


def load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["key"]] = rec["value"]
    return cache


def append_cache(path: Path, key: str, value):
    with open(path, "a") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")


def parse_espn_dob(s: str | None) -> datetime.date | None:
    """ESPN displayDOB is 'DD/MM/YYYY'."""
    if not s:
        return None
    try:
        return datetime.datetime.strptime(s, "%d/%m/%Y").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _cached_search(ci: CricinfoClient, cache: dict, query: str, limit: int):
    cache_key = query.lower() if limit == 25 else f"{query.lower()}|{limit}"
    if cache_key in cache:
        return cache[cache_key], None
    results, err = _with_retry(ci.search_players, query, limit=limit)
    if err:
        return None, err
    append_cache(SEARCH_CACHE, cache_key, results)
    cache[cache_key] = results
    time.sleep(REQUEST_DELAY_S)
    return results, None


def _given_name_initials(full_name: str, surname: str) -> str:
    """'Dwayne John Bravo' with surname 'Bravo' -> 'DJ'."""
    fn = full_name.strip()
    if fn.lower().endswith(surname.lower()):
        fn = fn[: len(fn) - len(surname)].strip()
    return "".join(w[0].upper() for w in fn.split() if w[:1].isalpha())


def _disambiguate_by_bio(ci, bio_cache, candidates, initials, surname):
    """
    Break an ambiguous candidate set by comparing each candidate's full-name
    given-name initials (from their bio) against the cricsheet initials.
    'SK Raina' -> Suresh Kumar Raina ('SK') beats Suryansh Raina ('S').
    Returns the sole exact match, or None if still ambiguous.
    """
    want = "".join(c.upper() for c in initials if c.isalpha())
    hits = []
    for c in candidates:
        cid = str(c["id"])
        if cid in bio_cache:
            bio = bio_cache[cid]
        else:
            bio, err = _with_retry(ci.player_bio, cid)
            if err:
                continue
            append_cache(BIO_CACHE, cid, bio)
            bio_cache[cid] = bio
            time.sleep(REQUEST_DELAY_S)
        full = bio.get("fullName") or bio.get("displayName") or ""
        if _given_name_initials(full, surname) == want:
            hits.append(c)
    return hits[0] if len(hits) == 1 else None


def find_match(ci, cache, bio_cache, initials, surname, country):
    results, err = _cached_search(ci, cache, surname, 25)
    if err:
        return None, "error", 0, f"search failed: {err}"

    match = _filter_candidates(results or [], initials, surname, country)
    if match[1] == "matched":
        return match[:4]

    # Fallback 1: full-name keys ("Mandeep Singh") — search the whole name,
    # common surnames bury the right player below the top-25.
    if any(c.islower() for c in initials):
        results, err = _cached_search(ci, cache, f"{initials} {surname}", 25)
    # Fallback 2: initials keys with a common surname — widen the net.
    else:
        results, err = _cached_search(ci, cache, surname, 75)

    if not err:
        fb = _filter_candidates(results or [], initials, surname, country)
        if fb[1] == "matched":
            return fb[:4]
        if fb[1] == "ambiguous" and match[1] != "ambiguous":
            match = fb

    # Ambiguity tie-break: fetch candidate bios, match full-name initials.
    if match[1] == "ambiguous" and not any(c.islower() for c in initials):
        winner = _disambiguate_by_bio(ci, bio_cache, match[4], initials, surname)
        if winner:
            return winner["id"], "matched", 1, "bio full-name initials"

    return match[:4]


def _filter_candidates(results: list, initials: str, surname: str, country: str | None):
    if not results:
        return None, "not_found", 0, "no search results", []

    # cricsheet initials carry ALL given names ("LRPL Taylor"), but ESPN
    # displays the commonly-used one ("Ross Taylor") — so the display first
    # name's initial may match ANY of the cricsheet initials, not just the first.
    # Keys like "Mohammad Nawaz" carry a full first name, not initials — there
    # the display first name must match it as a whole word, not per-letter.
    is_full_first_name = any(c.islower() for c in initials)
    initial_set = (set() if is_full_first_name
                   else {c.lower() for c in initials if c.isalpha()}) if initials else set()

    candidates = []
    for r in results:
        display = r.get("displayName", "")
        name_parts = display.split()
        if not name_parts:
            continue
        # last name-part(s) must match surname (case-insensitive, allow multi-word)
        if not display.lower().endswith(surname.lower()):
            continue
        teams = [t["core"]["displayName"] for t in r.get("teamRelationships", [])
                 if t.get("type") == "team"]
        if country and country not in teams:
            continue
        if initial_set and name_parts[0][0].lower() not in initial_set:
            continue
        if is_full_first_name and name_parts[0].lower() != initials.lower():
            continue
        candidates.append(r)

    if len(candidates) == 1:
        return candidates[0]["id"], "matched", 1, "surname+country+any-initial", candidates
    if len(candidates) == 0:
        return None, "not_found", 0, "no candidate survived filtering", []
    return None, "ambiguous", len(candidates), "multiple candidates after filtering", candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--limit", type=int, default=None, help="Only process first N unmatched players")
@click.option("--report-only", is_flag=True, help="Print current match-status breakdown and exit")
@click.option("--recheck-failures", is_flag=True,
              help="Re-run all ambiguous/not_found/error rows too (applies new overrides "
                   "and current matching logic); leaves matched rows untouched")
def main(limit, report_only, recheck_failures):
    engine = get_engine(DB_PATH)
    Base.metadata.create_all(engine, tables=[PlayerEspnMap.__table__])
    session = Session(engine)

    if report_only:
        rows = session.execute(text(
            "SELECT status, COUNT(*) FROM player_espn_map GROUP BY status"
        )).fetchall()
        total = session.query(Player).count()
        print(f"players: {total}")
        for status, cnt in rows:
            print(f"  {status}: {cnt}")
        return

    # In normal runs only transient errors are re-run; --recheck-failures also
    # re-runs ambiguous/not_found so newly-added overrides and improved matching
    # logic get a second chance at players an earlier pass couldn't resolve.
    redo = ("'ambiguous', 'not_found', 'error'" if recheck_failures else "'error'")
    done = {r[0] for r in session.execute(text(
        f"SELECT player_id FROM player_espn_map WHERE status NOT IN ({redo})"
    ))}
    retryable = {r[0] for r in session.execute(text(
        f"SELECT player_id FROM player_espn_map WHERE status IN ({redo})"
    ))}
    if retryable:
        session.execute(text(f"DELETE FROM player_espn_map WHERE status IN ({redo})"))
        session.commit()

    players = session.query(Player).filter(~Player.id.in_(done)).all() if done else session.query(Player).all()
    if limit:
        players = players[:limit]

    reruns = "re-running failures" if recheck_failures else "retrying after transient errors"
    print(f"Processing {len(players)} players ({len(done)} already done, "
          f"{len(retryable)} {reruns})…")

    ci = CricinfoClient()
    search_cache = load_cache(SEARCH_CACHE)
    bio_cache = load_cache(BIO_CACHE)

    # Manual overrides for players automated matching can't resolve
    # (nicknames like "Rassie", surnames too common for search, etc.)
    overrides_path = Path(__file__).parent.parent / "data" / "espn_overrides.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}

    matched = ambiguous = not_found = errors = 0

    for i, p in enumerate(players, 1):
        initials, surname = split_key(p.cricsheet_key)
        if not surname:
            session.add(PlayerEspnMap(player_id=p.id, status="not_found", reason="unparseable key"))
            not_found += 1
            continue

        if p.cricsheet_key in overrides:
            espn_id, status, n_cand, reason = overrides[p.cricsheet_key], "matched", 1, "manual override"
        else:
            espn_id, status, n_cand, reason = find_match(
                ci, search_cache, bio_cache, initials, surname, p.country)

        if status == "matched":
            bio_key = str(espn_id)
            if bio_key in bio_cache:
                bio = bio_cache[bio_key]
            else:
                bio, err = _with_retry(ci.player_bio, espn_id)
                if err:
                    status, reason = "error", f"bio fetch failed: {err}"
                else:
                    append_cache(BIO_CACHE, bio_key, bio)
                    bio_cache[bio_key] = bio
                    time.sleep(REQUEST_DELAY_S)

            if bio:
                bat_styles = bio.get("batStyle") or []
                bowl_styles = bio.get("bowlStyle") or []
                dob = parse_espn_dob(bio.get("displayDOB"))
                team_name = (bio.get("team") or {}).get("displayName")

                if not p.full_name and (bio.get("fullName") or bio.get("displayName")):
                    p.full_name = bio.get("fullName") or bio.get("displayName")
                if not p.batting_style and bat_styles:
                    p.batting_style = bat_styles[0].get("description")
                if not p.bowling_style and bowl_styles:
                    p.bowling_style = bowl_styles[0].get("description")
                if not p.date_of_birth and dob:
                    p.date_of_birth = dob
                if not p.country and team_name:
                    p.country = team_name
                matched += 1

        elif status == "ambiguous":
            ambiguous += 1
        elif status == "error":
            errors += 1
        else:
            not_found += 1

        session.add(PlayerEspnMap(
            player_id=p.id, espn_id=str(espn_id) if espn_id else None,
            status=status, candidates=n_cand, reason=reason,
        ))

        if i % 25 == 0:
            session.commit()
            print(f"  [{i}/{len(players)}] matched={matched} ambiguous={ambiguous} "
                  f"not_found={not_found} errors={errors}")

    session.commit()
    print(f"\nDone. matched={matched} ambiguous={ambiguous} not_found={not_found} errors={errors}")
    session.close()


if __name__ == "__main__":
    main()
