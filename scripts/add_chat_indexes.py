"""
One-time migration: add indexes to cricket.db that speed up the
queries the insight engine commonly generates.

Run once:  python scripts/add_chat_indexes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import DB_PATH

INDEXES = [
    # deliveries — largest table (1.3M rows), queried most
    "CREATE INDEX IF NOT EXISTS idx_del_batter   ON deliveries(batter_id)",
    "CREATE INDEX IF NOT EXISTS idx_del_bowler   ON deliveries(bowler_id)",
    "CREATE INDEX IF NOT EXISTS idx_del_innings  ON deliveries(innings_id)",
    "CREATE INDEX IF NOT EXISTS idx_del_phase    ON deliveries(phase)",
    # matches
    "CREATE INDEX IF NOT EXISTS idx_match_date       ON matches(match_date)",
    "CREATE INDEX IF NOT EXISTS idx_match_tournament ON matches(tournament)",
    "CREATE INDEX IF NOT EXISTS idx_match_season     ON matches(season)",
    "CREATE INDEX IF NOT EXISTS idx_match_venue      ON matches(venue_id)",
    # player_innings
    "CREATE INDEX IF NOT EXISTS idx_pi_batter  ON player_innings(batter_id)",
    "CREATE INDEX IF NOT EXISTS idx_pi_match   ON player_innings(match_id)",
    # player_bowling_innings
    "CREATE INDEX IF NOT EXISTS idx_pbi_bowler ON player_bowling_innings(bowler_id)",
    # player_perf_by_season
    "CREATE INDEX IF NOT EXISTS idx_pps_player ON player_perf_by_season(player_id)",
    "CREATE INDEX IF NOT EXISTS idx_pps_season ON player_perf_by_season(season)",
]

import sqlite3
conn = sqlite3.connect(str(DB_PATH))
conn.execute("PRAGMA journal_mode=WAL")
for idx_sql in INDEXES:
    print(f"  {idx_sql[:60]}…")
    conn.execute(idx_sql)
conn.commit()
conn.close()
print(f"\nDone — {len(INDEXES)} indexes added/verified on {DB_PATH}")
