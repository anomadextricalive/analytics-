"""
Build a tournament-metadata table (`tournaments`) in the cricket analytics DB.

One row per distinct `matches.tournament` code. Aggregate stats (seasons,
match/venue/team counts) are derived from `matches`; display_name / country /
region are curated via a hardcoded map below.

This is ADDITIVE ONLY: it creates `tournaments` (if absent) and upserts rows.
It never drops, alters, or writes to any other table. A separate crawl may be
writing to the same SQLite file, so we set a long busy_timeout, keep the write
in one short transaction, and retry on "database is locked".

Usage:
  python scripts/enrich_tournaments.py                # build / refresh table
  python scripts/enrich_tournaments.py --report-only  # print table, no write
"""

import datetime
import sys
import time
from pathlib import Path

import click
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH  # noqa: E402
from src.db.schema import get_engine  # noqa: E402

BUSY_TIMEOUT_MS = 60000
MAX_RETRIES = 5

# Curated metadata. New codes not present here are still inserted with
# display_name = code and country/region = "Unknown" (with a warning).
CURATED = {
    "t20i_male": ("Men's T20 Internationals", "International", "International"),
    "ipl":       ("Indian Premier League",    "India",         "Asia"),
    "bbl":       ("Big Bash League",           "Australia",     "Oceania"),
    "cpl":       ("Caribbean Premier League",  "West Indies",   "Americas"),
    "psl":       ("Pakistan Super League",     "Pakistan",      "Asia"),
    "lpl":       ("Lanka Premier League",      "Sri Lanka",     "Asia"),
    "msl":       ("Mzansi Super League",       "South Africa",  "Africa"),
    "etpl":      ("European T20 Premier League", "Europe",      "Europe"),
    "t20_blast": ("Vitality T20 Blast",        "England",       "Europe"),
    "sma":       ("Syed Mushtaq Ali Trophy",   "India",         "Asia"),
    "bpl":       ("Bangladesh Premier League", "Bangladesh",    "Asia"),
    "csa_t20c":  ("CSA T20 Challenge",         "South Africa",  "Africa"),
    "super_smash": ("Super Smash",             "New Zealand",   "Oceania"),
    "hundred_male": ("The Hundred (Men)",      "England",       "Europe"),
    "ilt20":     ("International League T20",  "UAE",           "Asia"),
    "sa20":      ("SA20",                      "South Africa",  "Africa"),
    "mct":       ("Major Clubs T20 Tournament", "Sri Lanka",    "Asia"),
    "ipt":       ("Inter-Provincial T20 Trophy", "Ireland",     "Europe"),
    "mlc":       ("Major League Cricket",      "USA",           "Americas"),
    "npl":       ("Nepal Premier League",      "Nepal",         "Asia"),
    "eca_euro_cup": ("ECA European Cup",       "Europe",        "Europe"),
    # Cricbuzz-sourced (2026-10)
    "tnpl": ('Tamil Nadu Premier League', 'India', 'Asia'),
    "maharaja_trophy": ('Maharaja Trophy KSCA T20', 'India', 'Asia'),
    "dpl": ('Delhi Premier League', 'India', 'Asia'),
    "mppl": ('Madhya Pradesh Premier League', 'India', 'Asia'),
    "kpl": ('Karnataka Premier League', 'India', 'Asia'),
    "t20_mumbai": ('T20 Mumbai League', 'India', 'Asia'),
    "uppl": ('UP T20 League', 'India', 'Asia'),
    "kerala_cricket_league": ('Kerala Cricket League', 'India', 'Asia'),
    "tg20": ('TG20 League', 'India', 'Asia'),
    "mpl": ('Maharashtra Premier League', 'India', 'Asia'),
    "sher_e_punjab_t20_league": ('Sher-E-Punjab T20 League', 'India', 'Asia'),
    "apl_andhra": ('Andhra Premier League', 'India', 'Asia'),
    "pondicherry_premier_league": ('Pondicherry Premier League', 'India', 'Asia'),
    "rajasthan_premier_league": ('Rajasthan Premier League', 'India', 'Asia'),
    "saurashtra_premier_league": ('Saurashtra Premier League', 'India', 'Asia'),
    "chhattisgarh_cricket_premier_league": ('Chhattisgarh Cricket Premier League', 'India', 'Asia'),
    "haryana_premier_league": ('Haryana Premier League', 'India', 'Asia'),
    "rajputana_premier_league": ('Rajputana Premier League', 'India', 'Asia'),
    "gt20_canada": ('Global T20 Canada', 'Canada', 'Americas'),
    "csa_provincial_t20": ('CSA Provincial T20 Cup', 'South Africa', 'Africa'),
    "global_super_league": ('Global Super League', 'West Indies', 'Americas'),
    "apl_afg": ('Afghanistan Premier League', 'UAE', 'Asia'),
    "bangabandhu_t20_cup": ('Bangabandhu T20 Cup', 'Bangladesh', 'Asia'),
    "hk_t20_blitz": ('Hong Kong T20 Blitz', 'Hong Kong', 'Asia'),
    "super_provincial_t20_tournament": ('Super Provincial T20', 'Sri Lanka', 'Asia'),
    "emirates_t20_trophy": ('Emirates T20 Trophy', 'UAE', 'Asia'),
    "uae_summer_t20_bash": ('UAE Summer T20 Bash', 'UAE', 'Asia'),
    "t20_spring_challenge": ('T20 Spring Challenge', 'UAE', 'Asia'),
    "t10_abu_dhabi": ('Abu Dhabi T10', 'UAE', 'Asia'),
    "t10_zim_afro": ('Zim Afro T10', 'Zimbabwe', 'Africa'),
    "t10_lanka": ('Lanka T10 Super League', 'Sri Lanka', 'Asia'),
    "t10_qatar": ('Qatar T10 League', 'Qatar', 'Asia'),
    "legends_llc": ('Legends League Cricket', 'International', 'International'),
    "legends_road_safety": ('Road Safety World Series', 'International', 'International'),
    "legends_wcl": ('World Championship of Legends', 'International', 'International'),
    "legends_legends_cricket_trophy": ('Legends Cricket Trophy', 'International', 'International'),
    "legends_world_legends_pro_t20_league": ('World Legends Pro T20 League', 'International', 'International'),
    "legends_international_masters_league_t20": ('International Masters League T20', 'International', 'International'),
    "legends_asian_legends_league": ('Asian Legends League', 'International', 'International'),
}

FORMAT = "T20"   # this DB is T20-only
GENDER = "male"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS tournaments (
  code          TEXT PRIMARY KEY,
  display_name  TEXT,
  country       TEXT,
  region        TEXT,
  format        TEXT,
  gender        TEXT,
  first_season  TEXT,
  last_season   TEXT,
  num_seasons   INTEGER,
  num_matches   INTEGER,
  num_teams     INTEGER,
  num_venues    INTEGER,
  updated_at    TEXT
)
"""

UPSERT_SQL = """
INSERT INTO tournaments (
  code, display_name, country, region, format, gender,
  first_season, last_season, num_seasons, num_matches,
  num_teams, num_venues, updated_at
) VALUES (
  :code, :display_name, :country, :region, :format, :gender,
  :first_season, :last_season, :num_seasons, :num_matches,
  :num_teams, :num_venues, :updated_at
)
ON CONFLICT(code) DO UPDATE SET
  display_name = excluded.display_name,
  country      = excluded.country,
  region       = excluded.region,
  format       = excluded.format,
  gender       = excluded.gender,
  first_season = excluded.first_season,
  last_season  = excluded.last_season,
  num_seasons  = excluded.num_seasons,
  num_matches  = excluded.num_matches,
  num_teams    = excluded.num_teams,
  num_venues   = excluded.num_venues,
  updated_at   = excluded.updated_at
"""


def _set_busy_timeout(conn):
    conn.execute(text(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}"))


def compute_rows(engine):
    """Read-only: derive one metadata dict per tournament code from `matches`."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    rows = []
    with engine.connect() as conn:
        _set_busy_timeout(conn)
        codes = [
            r[0]
            for r in conn.execute(
                text("SELECT DISTINCT tournament FROM matches "
                     "WHERE tournament IS NOT NULL ORDER BY tournament")
            )
        ]
        for code in codes:
            agg = conn.execute(
                text(
                    "SELECT MIN(season), MAX(season), "
                    "COUNT(DISTINCT season), COUNT(*), "
                    "COUNT(DISTINCT venue_id) "
                    "FROM matches WHERE tournament = :t"
                ),
                {"t": code},
            ).one()
            num_teams = conn.execute(
                text(
                    "SELECT COUNT(*) FROM ("
                    "  SELECT team1_id AS t FROM matches WHERE tournament = :t"
                    "  UNION"
                    "  SELECT team2_id FROM matches WHERE tournament = :t"
                    ")"
                ),
                {"t": code},
            ).scalar()

            if code in CURATED:
                display_name, country, region = CURATED[code]
            else:
                display_name, country, region = code, "Unknown", "Unknown"
                click.echo(
                    f"WARNING: unmapped tournament code '{code}' — inserted with "
                    f"display_name='{code}', country/region='Unknown'. Curate later.",
                    err=True,
                )

            rows.append({
                "code": code,
                "display_name": display_name,
                "country": country,
                "region": region,
                "format": FORMAT,
                "gender": GENDER,
                "first_season": agg[0],
                "last_season": agg[1],
                "num_seasons": agg[2],
                "num_matches": agg[3],
                "num_teams": num_teams,
                "num_venues": agg[4],
                "updated_at": now,
            })
    return rows


def write_rows(engine, rows):
    """One short transaction, retried with backoff on 'database is locked'."""
    delay = 1.0
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with engine.begin() as conn:
                _set_busy_timeout(conn)
                conn.execute(text(CREATE_SQL))
                for row in rows:
                    conn.execute(text(UPSERT_SQL), row)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            if "locked" in str(e).lower() and attempt < MAX_RETRIES:
                click.echo(
                    f"database is locked (attempt {attempt}/{MAX_RETRIES}), "
                    f"retrying in {delay:.1f}s...",
                    err=True,
                )
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise last_err


def print_report(engine):
    with engine.connect() as conn:
        _set_busy_timeout(conn)
        exists = conn.execute(
            text("SELECT COUNT(*) FROM sqlite_master "
                 "WHERE type='table' AND name='tournaments'")
        ).scalar()
        if not exists:
            click.echo("tournaments table does not exist yet.")
            return
        result = conn.execute(text(
            "SELECT code, display_name, country, region, first_season, "
            "last_season, num_seasons, num_matches, num_teams, num_venues "
            "FROM tournaments ORDER BY num_matches DESC"
        ))
        cols = result.keys()
        click.echo("  ".join(cols))
        n = 0
        for r in result:
            n += 1
            click.echo("  ".join(str(v) for v in r))
        click.echo(f"({n} rows)")


@click.command()
@click.option("--report-only", is_flag=True,
              help="Print the tournaments table contents without writing.")
def main(report_only):
    engine = get_engine(DB_PATH)
    if report_only:
        print_report(engine)
        return

    rows = compute_rows(engine)
    write_rows(engine, rows)
    click.echo(f"Upserted {len(rows)} tournament rows into `tournaments`.")
    print_report(engine)


if __name__ == "__main__":
    main()
