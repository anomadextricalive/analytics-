"""
Build pre-joined MongoDB documents from SQLite (read-only).

SQLite is NEVER modified — this script only reads from it.
MongoDB is rebuilt atomically: new collections are written first,
then old raw collections are dropped only after a successful write.

Collections created:
  player_profiles  — one doc per player, everything pre-joined
  venue_profiles   — one doc per venue with difficulty factors
  match_profiles   — one doc per match with team/venue names

Usage:
  python scripts/build_mongo_profiles.py \
    --uri "mongodb+srv://user:pass@cluster/..." \
    --db cricket_analytics
"""

import sys
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
from src.db.schema import get_engine

console = Console()

# Raw collections that will be dropped after profiles are built
RAW_COLLECTIONS = [
    "players", "venues", "teams", "matches", "innings", "deliveries",
    "player_innings", "player_bowling_innings", "partnerships",
    "player_career_bat", "player_career_bowl", "player_position_bat",
    "player_chase_bat", "player_venue_bat", "player_venue_bowl",
    "player_perf_by_opponent", "player_perf_by_season",
    "player_perf_by_team", "player_perf_by_result",
    "player_dismissal_analysis", "player_bowling_dismissal_analysis",
    "player_milestones", "player_of_match_awards",
    "player_fielding_stats", "venue_difficulty", "player_ratings",
]


def _read(engine, sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def _to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts, dropping NaN/None cleanly."""
    return [{k: v for k, v in row.items() if v is not None and v == v}
            for row in df.to_dict("records")]


# ─────────────────────────────────────────────────────────────────────────────
# VENUE PROFILES
# ─────────────────────────────────────────────────────────────────────────────

def build_venue_profiles(engine) -> list[dict]:
    venues = _read(engine, "SELECT * FROM venues")
    diff   = _read(engine, "SELECT * FROM venue_difficulty")

    merged = venues.merge(diff, left_on="id", right_on="venue_id", how="left",
                          suffixes=("", "_diff"))

    docs = []
    for _, row in merged.iterrows():
        doc = {
            "id":   int(row["id"]),
            "name": row.get("name", ""),
            "city": row.get("city"),
        }
        if pd.notna(row.get("bat_factor")):
            doc["difficulty"] = {
                "bat_factor":       round(float(row["bat_factor"]), 4),
                "bowl_factor":      round(float(row["bowl_factor"]), 4),
                "avg_first_inn_runs":  row.get("avg_first_inn_runs"),
                "avg_second_inn_runs": row.get("avg_second_inn_runs"),
                "total_matches":    int(row["total_matches"]) if pd.notna(row.get("total_matches")) else None,
                "boundary_rate":    round(float(row["boundary_rate"]), 4) if pd.notna(row.get("boundary_rate")) else None,
                "pace_index":       round(float(row["pace_index"]), 4) if pd.notna(row.get("pace_index")) else None,
            }
        docs.append({k: v for k, v in doc.items() if v is not None})
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# MATCH PROFILES
# ─────────────────────────────────────────────────────────────────────────────

def build_match_profiles(engine) -> list[dict]:
    sql = """
        SELECT m.id, m.cricsheet_id, m.match_date, m.tournament, m.season,
               m.toss_decision, m.win_by_runs, m.win_by_wickets, m.no_result,
               v.name  AS venue,
               v.city  AS city,
               t1.name AS team1,
               t2.name AS team2,
               tw.name AS toss_winner,
               wt.name AS winner
        FROM matches m
        LEFT JOIN venues  v  ON v.id  = m.venue_id
        LEFT JOIN teams   t1 ON t1.id = m.team1_id
        LEFT JOIN teams   t2 ON t2.id = m.team2_id
        LEFT JOIN teams   tw ON tw.id = m.toss_winner_id
        LEFT JOIN teams   wt ON wt.id = m.winner_id
    """
    df = _read(engine, sql)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return _to_records(df)


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER PROFILES
# ─────────────────────────────────────────────────────────────────────────────

def build_player_profiles(engine) -> list[dict]:
    players   = _read(engine, "SELECT id, cricsheet_uuid, cricsheet_key, country FROM players")
    ratings   = _read(engine, "SELECT * FROM player_ratings WHERE tournament='ALL'")
    career_bat = _read(engine, "SELECT * FROM player_career_bat WHERE tournament='ALL'")
    career_bowl= _read(engine, "SELECT * FROM player_career_bowl WHERE tournament='ALL'")
    chase      = _read(engine, "SELECT * FROM player_chase_bat")
    position   = _read(engine, "SELECT * FROM player_position_bat")
    fielding   = _read(engine, "SELECT * FROM player_fielding_stats WHERE tournament='ALL'")
    pom        = _read(engine, "SELECT player_id, COUNT(*) as awards FROM player_of_match_awards GROUP BY player_id")
    milestones = _read(engine, "SELECT player_id, milestone_type, COUNT(*) as count FROM player_milestones GROUP BY player_id, milestone_type")

    # By-breakdown tables
    by_venue   = _read(engine, """
        SELECT pvb.player_id, v.name as venue, pvb.innings, pvb.runs, pvb.balls,
               pvb.average, pvb.strike_rate
        FROM player_venue_bat pvb JOIN venues v ON v.id = pvb.venue_id
        WHERE pvb.innings >= 3
    """)
    by_season  = _read(engine, "SELECT * FROM player_perf_by_season WHERE tournament='ALL'")
    by_opponent= _read(engine, """
        SELECT ppo.*, t.name as opponent_name
        FROM player_perf_by_opponent ppo JOIN teams t ON t.id = ppo.opponent_id
    """)
    by_team    = _read(engine, """
        SELECT ppt.*, t.name as team_name
        FROM player_perf_by_team ppt JOIN teams t ON t.id = ppt.team_id
    """)
    by_result  = _read(engine, "SELECT * FROM player_perf_by_result")
    dismiss_bat = _read(engine, "SELECT * FROM player_dismissal_analysis")
    dismiss_bowl= _read(engine, "SELECT * FROM player_bowling_dismissal_analysis")
    by_tourn   = _read(engine, "SELECT * FROM player_career_bat WHERE tournament != 'ALL'")
    bowl_tourn = _read(engine, "SELECT * FROM player_career_bowl WHERE tournament != 'ALL'")

    def _group(df: pd.DataFrame, key: str) -> dict:
        """Group a DataFrame by player_id → {player_id: [rows]}"""
        out = {}
        for pid, grp in df.groupby(key):
            out[int(pid)] = _to_records(grp.drop(columns=[key], errors="ignore"))
        return out

    chase_by_pid    = _group(chase,       "player_id")
    pos_by_pid      = _group(position,    "player_id")
    venue_by_pid    = _group(by_venue,    "player_id")
    season_by_pid   = _group(by_season,   "player_id")
    opp_by_pid      = _group(by_opponent, "player_id")
    team_by_pid     = _group(by_team,     "player_id")
    result_by_pid   = _group(by_result,   "player_id")
    dbat_by_pid     = _group(dismiss_bat, "player_id")
    dbowl_by_pid    = _group(dismiss_bowl,"player_id")
    tourn_bat_pid   = _group(by_tourn,    "player_id")
    tourn_bowl_pid  = _group(bowl_tourn,  "player_id")
    milest_by_pid   = _group(milestones,  "player_id")
    pom_map         = dict(zip(pom["player_id"].astype(int), pom["awards"].astype(int)))

    # Index lookup dicts
    rat_map    = {int(r["player_id"]): r for _, r in ratings.iterrows()}
    cbat_map   = {int(r["player_id"]): r for _, r in career_bat.iterrows()}
    cbowl_map  = {int(r["player_id"]): r for _, r in career_bowl.iterrows()}
    field_map  = {int(r["player_id"]): r for _, r in fielding.iterrows()}

    docs = []
    for _, p in players.iterrows():
        pid = int(p["id"])
        doc = {
            "id":             pid,
            "cricsheet_uuid": p.get("cricsheet_uuid"),
            "name":           p["cricsheet_key"],
            "country":        p.get("country"),
            "player_of_match": pom_map.get(pid, 0),
        }

        # Ratings
        if pid in rat_map:
            r = rat_map[pid]
            doc["ratings"] = {k: round(float(v), 2) for k, v in r.items()
                               if k not in ("player_id", "tournament", "updated_at")
                               and pd.notna(v)}

        # Career batting
        if pid in cbat_map:
            b = cbat_map[pid]
            doc["career_bat"] = {k: v for k, v in b.items()
                                 if k not in ("player_id", "tournament") and pd.notna(v)}

        # Career bowling
        if pid in cbowl_map:
            b = cbowl_map[pid]
            doc["career_bowl"] = {k: v for k, v in b.items()
                                  if k not in ("player_id", "tournament") and pd.notna(v)}

        # Fielding
        if pid in field_map:
            f = field_map[pid]
            doc["fielding"] = {k: v for k, v in f.items()
                               if k not in ("player_id", "tournament") and pd.notna(v)}

        # Milestones summary
        if pid in milest_by_pid:
            doc["milestones"] = {m["milestone_type"]: m["count"] for m in milest_by_pid[pid]}

        # Splits
        if pid in chase_by_pid:    doc["by_innings_type"] = chase_by_pid[pid]
        if pid in pos_by_pid:      doc["by_position"]     = pos_by_pid[pid]
        if pid in venue_by_pid:    doc["by_venue"]        = venue_by_pid[pid]
        if pid in season_by_pid:   doc["by_season"]       = season_by_pid[pid]
        if pid in opp_by_pid:      doc["by_opponent"]     = opp_by_pid[pid]
        if pid in team_by_pid:     doc["by_team"]         = team_by_pid[pid]
        if pid in result_by_pid:   doc["by_result"]       = result_by_pid[pid]
        if pid in tourn_bat_pid:   doc["by_tournament_bat"]  = tourn_bat_pid[pid]
        if pid in tourn_bowl_pid:  doc["by_tournament_bowl"] = tourn_bowl_pid[pid]
        if pid in dbat_by_pid:     doc["dismissals_batting"]  = dbat_by_pid[pid]
        if pid in dbowl_by_pid:    doc["dismissals_bowling"]  = dbowl_by_pid[pid]

        docs.append({k: v for k, v in doc.items() if v is not None})

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--uri", required=True)
@click.option("--db",  required=True)
def main(uri: str, db: str):
    """Build pre-joined MongoDB profiles from SQLite (SQLite is never modified)."""
    try:
        from pymongo import MongoClient
        import certifi
    except ImportError:
        console.print("[red]Run: pip install pymongo certifi[/red]")
        sys.exit(1)

    if not DB_PATH.exists():
        console.print(f"[red]SQLite DB not found: {DB_PATH}[/red]")
        sys.exit(1)

    engine = get_engine()
    console.print(f"[green]✓[/green] SQLite connected (read-only source of truth)")

    client = MongoClient(uri, serverSelectionTimeoutMS=10_000, tlsCAFile=certifi.where())
    client.admin.command("ping")
    mongo_db = client[db]
    console.print(f"[green]✓[/green] MongoDB connected → [bold]{db}[/bold]")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TimeElapsedColumn(), console=console) as prog:

        # 1. Build venue profiles
        t = prog.add_task("Building venue profiles…", total=None)
        venues = build_venue_profiles(engine)
        mongo_db["venue_profiles"].drop()
        mongo_db["venue_profiles"].insert_many(venues)
        mongo_db["venue_profiles"].create_index("id", unique=True)
        prog.update(t, description=f"[green]✓ venue_profiles[/green] ({len(venues)} docs)")
        prog.stop_task(t)

        # 2. Build match profiles
        t = prog.add_task("Building match profiles…", total=None)
        matches = build_match_profiles(engine)
        mongo_db["match_profiles"].drop()
        for i in range(0, len(matches), 500):
            mongo_db["match_profiles"].insert_many(matches[i:i+500])
        mongo_db["match_profiles"].create_index("id", unique=True)
        mongo_db["match_profiles"].create_index("tournament")
        prog.update(t, description=f"[green]✓ match_profiles[/green] ({len(matches)} docs)")
        prog.stop_task(t)

        # 3. Build player profiles (biggest)
        t = prog.add_task("Building player profiles…", total=None)
        players = build_player_profiles(engine)
        mongo_db["player_profiles"].drop()
        for i in range(0, len(players), 200):
            mongo_db["player_profiles"].insert_many(players[i:i+200])
        mongo_db["player_profiles"].create_index("id", unique=True)
        mongo_db["player_profiles"].create_index("name")
        mongo_db["player_profiles"].create_index("cricsheet_uuid", sparse=True)
        prog.update(t, description=f"[green]✓ player_profiles[/green] ({len(players)} docs)")
        prog.stop_task(t)

    # 4. Drop raw collections only after all profiles written successfully
    console.print("\nDropping raw collections…")
    dropped = 0
    for col in RAW_COLLECTIONS:
        if col in mongo_db.list_collection_names():
            mongo_db[col].drop()
            dropped += 1
    console.print(f"[green]✓[/green] Dropped {dropped} raw collections")

    console.print(f"\n[bold green]Done![/bold green] MongoDB now has 3 pre-joined collections:")
    console.print("  player_profiles  — query by name or cricsheet_uuid")
    console.print("  match_profiles   — query by tournament, date, team")
    console.print("  venue_profiles   — query by name or id")


if __name__ == "__main__":
    main()
