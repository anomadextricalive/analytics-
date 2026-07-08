"""
Crawl cross-format international career stats via the cricdata SDK.

The analytical DB is T20-ONLY (see README) — ratings are never format-mixed.
This script ADDS cross-format career *context* alongside that T20 focus:
Statsguru-backed cumulative career lines (t20i / odi / test) pulled per player
via CricinfoClient.player_career_stats(espn_id, fmt, stat_type).

For every matched player in `player_espn_map` it fetches batting + bowling
summaries per format, normalises the key numbers into `player_career_intl`,
and records crawl state in `player_career_status`.

Idempotent resume — like enrich_players_espn.py:
  - players with status='done' are skipped on restart (never re-fetched)
  - status='error' rows are retried on restart
  - commits in batches, so killing + relaunching never loses data

Usage:
  python scripts/crawl_career_intl.py --limit 20        # pilot
  python scripts/crawl_career_intl.py                   # full run
  python scripts/crawl_career_intl.py --formats t20i,odi,test
  python scripts/crawl_career_intl.py --recheck         # re-run error rows too
  python scripts/crawl_career_intl.py --report-only     # just print stats
"""

import datetime
import json
import sys
import time
from pathlib import Path

import click
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
from src.db.schema import get_engine

from cricdata import CricinfoClient

REQUEST_DELAY_S = 0.5
MAX_RETRIES = 4
BATCH = 25
DEFAULT_FORMATS = ("t20i", "odi", "test")
STAT_TYPES = ("batting", "bowling")


def _with_retry(fn, *args, **kwargs):
    """Retry on transient HTTP errors (e.g. 503 rate-limit) with backoff."""
    delay = 2.0
    last_err = None
    for _ in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs), None
        except Exception as e:  # noqa: BLE001 — SDK raises varied HTTP errors
            last_err = e
            time.sleep(delay)
            delay *= 2
    return None, last_err


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS player_career_intl (
        player_id  INTEGER NOT NULL,
        espn_id    TEXT,
        fmt        TEXT NOT NULL,
        stat_type  TEXT NOT NULL,
        span       TEXT,
        mat        INTEGER,
        inns       INTEGER,
        runs       INTEGER,
        ave        REAL,
        sr         REAL,
        bf         INTEGER,
        hs         TEXT,
        hundreds   INTEGER,
        fifties    INTEGER,
        ducks      INTEGER,
        fours      INTEGER,
        sixes      INTEGER,
        no         INTEGER,
        wkts       INTEGER,
        econ       REAL,
        bbi        TEXT,
        overs      REAL,
        mdns       INTEGER,
        four_w     INTEGER,
        five_w     INTEGER,
        raw_json   TEXT,
        PRIMARY KEY (player_id, fmt, stat_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_career_status (
        player_id     INTEGER PRIMARY KEY,
        espn_id       TEXT,
        status        TEXT NOT NULL,   -- done | error
        formats_found TEXT,            -- csv of fmts with mat>0
        reason        TEXT,
        updated_at    TEXT
    )
    """,
]


def _ensure_schema(engine):
    with engine.begin() as conn:
        for ddl in _DDL:
            conn.execute(text(ddl))


# ---------------------------------------------------------------------------
# Parsing helpers — Statsguru summary values are strings ("48.69", "122*", "-")
# ---------------------------------------------------------------------------

def _num(v, cast=float):
    if v is None:
        return None
    s = str(v).strip().replace("*", "")
    if s in ("", "-", "–"):
        return None
    try:
        return cast(s)
    except (ValueError, TypeError):
        return None


def _int(v):
    return _num(v, int)


def _bat_row(player_id, espn_id, fmt, summary):
    return {
        "player_id": player_id, "espn_id": espn_id, "fmt": fmt, "stat_type": "batting",
        "span": summary.get("Span"),
        "mat": _int(summary.get("Mat")), "inns": _int(summary.get("Inns")),
        "runs": _int(summary.get("Runs")), "ave": _num(summary.get("Ave")),
        "sr": _num(summary.get("SR")), "bf": _int(summary.get("BF")),
        "hs": (summary.get("HS") or None), "hundreds": _int(summary.get("100")),
        "fifties": _int(summary.get("50")), "ducks": _int(summary.get("0")),
        "fours": _int(summary.get("4s")), "sixes": _int(summary.get("6s")),
        "no": _int(summary.get("NO")),
        "wkts": None, "econ": None, "bbi": None, "overs": None,
        "mdns": None, "four_w": None, "five_w": None,
        "raw_json": json.dumps(summary),
    }


def _bowl_row(player_id, espn_id, fmt, summary):
    return {
        "player_id": player_id, "espn_id": espn_id, "fmt": fmt, "stat_type": "bowling",
        "span": summary.get("Span"),
        "mat": _int(summary.get("Mat")), "inns": _int(summary.get("Inns")),
        "runs": _int(summary.get("Runs")), "ave": _num(summary.get("Ave")),
        "sr": _num(summary.get("SR")), "bf": None,
        "hs": None, "hundreds": None, "fifties": None, "ducks": None,
        "fours": None, "sixes": None, "no": None,
        "wkts": _int(summary.get("Wkts")), "econ": _num(summary.get("Econ")),
        "bbi": (summary.get("BBI") or None), "overs": _num(summary.get("Overs")),
        "mdns": _int(summary.get("Mdns")), "four_w": _int(summary.get("4")),
        "five_w": _int(summary.get("5")),
        "raw_json": json.dumps(summary),
    }


_UPSERT = text("""
    INSERT INTO player_career_intl
      (player_id, espn_id, fmt, stat_type, span, mat, inns, runs, ave, sr, bf,
       hs, hundreds, fifties, ducks, fours, sixes, no,
       wkts, econ, bbi, overs, mdns, four_w, five_w, raw_json)
    VALUES
      (:player_id, :espn_id, :fmt, :stat_type, :span, :mat, :inns, :runs, :ave, :sr, :bf,
       :hs, :hundreds, :fifties, :ducks, :fours, :sixes, :no,
       :wkts, :econ, :bbi, :overs, :mdns, :four_w, :five_w, :raw_json)
    ON CONFLICT(player_id, fmt, stat_type) DO UPDATE SET
       span=excluded.span, mat=excluded.mat, inns=excluded.inns, runs=excluded.runs,
       ave=excluded.ave, sr=excluded.sr, bf=excluded.bf, hs=excluded.hs,
       hundreds=excluded.hundreds, fifties=excluded.fifties, ducks=excluded.ducks,
       fours=excluded.fours, sixes=excluded.sixes, no=excluded.no, wkts=excluded.wkts,
       econ=excluded.econ, bbi=excluded.bbi, overs=excluded.overs, mdns=excluded.mdns,
       four_w=excluded.four_w, five_w=excluded.five_w, raw_json=excluded.raw_json
""")

_UPSERT_STATUS = text("""
    INSERT INTO player_career_status (player_id, espn_id, status, formats_found, reason, updated_at)
    VALUES (:player_id, :espn_id, :status, :formats_found, :reason, :updated_at)
    ON CONFLICT(player_id) DO UPDATE SET
       espn_id=excluded.espn_id, status=excluded.status,
       formats_found=excluded.formats_found, reason=excluded.reason,
       updated_at=excluded.updated_at
""")


def _report(engine):
    with engine.connect() as conn:
        st = conn.execute(text(
            "SELECT status, COUNT(*) FROM player_career_status GROUP BY status"
        )).fetchall()
        rows = conn.execute(text(
            "SELECT fmt, stat_type, COUNT(*) FROM player_career_intl "
            "WHERE mat > 0 GROUP BY fmt, stat_type ORDER BY fmt, stat_type"
        )).fetchall()
    click.echo("── crawl status ──")
    for s, n in st:
        click.echo(f"  {s:8s} {n}")
    click.echo("── career rows (mat>0) ──")
    for fmt, stype, n in rows:
        click.echo(f"  {fmt:5s} {stype:8s} {n}")


@click.command()
@click.option("--limit", type=int, default=None, help="Only process N players (pilot).")
@click.option("--formats", default=",".join(DEFAULT_FORMATS),
              help="Comma-separated formats: t20i,odi,test,fc,lista,t20")
@click.option("--recheck", is_flag=True, help="Re-run error rows too (not just unseen).")
@click.option("--report-only", is_flag=True, help="Print stats and exit.")
def main(limit, formats, recheck, report_only):
    engine = get_engine(str(DB_PATH))
    _ensure_schema(engine)

    if report_only:
        _report(engine)
        return

    fmts = [f.strip() for f in formats.split(",") if f.strip()]
    client = CricinfoClient()

    # Candidate players: matched in the ESPN crosswalk, not yet done.
    skip_clause = "status='done'" if not recheck else "1=0"
    with engine.connect() as conn:
        done = {r[0] for r in conn.execute(text(
            f"SELECT player_id FROM player_career_status WHERE {skip_clause}"
        )).fetchall()}
        players = conn.execute(text(
            "SELECT player_id, espn_id FROM player_espn_map "
            "WHERE status='matched' AND espn_id IS NOT NULL"
        )).fetchall()

    todo = [(pid, eid) for pid, eid in players if pid not in done]
    if limit:
        todo = todo[:limit]

    click.echo(f"players to crawl: {len(todo)} "
               f"({len(done)} already done) · formats={fmts}")

    pending, done_ct, err_ct = [], 0, 0
    now = datetime.datetime.now().isoformat(timespec="seconds")

    def _flush():
        nonlocal pending
        if not pending:
            return
        with engine.begin() as conn:
            for rows, status_row in pending:
                for row in rows:
                    conn.execute(_UPSERT, row)
                conn.execute(_UPSERT_STATUS, status_row)
        pending = []

    for i, (pid, eid) in enumerate(todo, 1):
        rows, found, err = [], [], None
        for fmt in fmts:
            for stype in STAT_TYPES:
                res, e = _with_retry(client.player_career_stats, eid,
                                     fmt=fmt, stat_type=stype)
                time.sleep(REQUEST_DELAY_S)
                if e is not None or not isinstance(res, dict):
                    err = str(e)[:200] if e else "no-dict-response"
                    continue
                summary = res.get("summary") or {}
                row = (_bat_row if stype == "batting" else _bowl_row)(
                    pid, eid, fmt, summary)
                if row["mat"] and row["mat"] > 0:
                    rows.append(row)
                    if fmt not in found:
                        found.append(fmt)

        status = "error" if (err and not rows) else "done"
        if status == "error":
            err_ct += 1
        else:
            done_ct += 1
        status_row = {
            "player_id": pid, "espn_id": eid, "status": status,
            "formats_found": ",".join(found), "reason": err, "updated_at": now,
        }
        pending.append((rows, status_row))

        if i % BATCH == 0:
            _flush()
            click.echo(f"  [{i}/{len(todo)}] done={done_ct} err={err_ct}")

    _flush()
    click.echo(f"finished · done={done_ct} err={err_ct}")
    _report(engine)


if __name__ == "__main__":
    main()
