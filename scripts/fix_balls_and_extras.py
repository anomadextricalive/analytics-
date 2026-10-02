"""Recompute ball-level derived data from `deliveries` (source of truth).

Fixes three parser conventions that were wrong for every tournament:
  1. Bowler runs_conceded dropped wides + no-ball runs (only legal balls were counted).
     Correct: bat runs + wides + no-balls (byes, leg byes, penalty are not the bowler's).
  2. Batter balls_faced dropped no-balls. Correct: every delivery except a wide.
  3. Innings length hard-coded to 120 balls (req-rate, phases). Now per match:
     The Hundred = 100 balls (5-ball sets), T10 = 60 balls, T20 = overs*6.

Phase grids (over_number is 1-based; for The Hundred it is the 5-ball set):
  T20      powerplay 1-6   middle 7-15  death 16-20
  T10      powerplay 1-2   middle 3-8   death 9-10
  Hundred  powerplay 1-5   middle 6-15  death 16-20   (25 / 50 / 25 balls)

Idempotent. Usage: python scripts/fix_balls_and_extras.py [db_path]
Always back the DB up first.
"""
import sqlite3
import sys
import time

DB = sys.argv[1] if len(sys.argv) > 1 else "data/cricket.db"
con = sqlite3.connect(DB)
con.execute("PRAGMA journal_mode=WAL")
cur = con.cursor()


def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def one(sql):
    return cur.execute(sql).fetchone()[0]


def invariant():
    """innings total = bowler-charged runs + byes + leg byes + penalty, per innings."""
    return cur.execute("""
        SELECT COUNT(*) FROM (
          SELECT i.id, i.total_runs AS tot,
                 SUM(d.bat_runs + d.wide + d.no_ball + d.bye + d.leg_bye + d.penalty) AS rebuilt
          FROM innings i JOIN deliveries d ON d.innings_id = i.id
          GROUP BY i.id) WHERE tot != rebuilt""").fetchone()[0]


step("BEFORE: bowler runs in player_bowling_innings = %d" % one("SELECT SUM(runs_conceded) FROM player_bowling_innings"))
step("BEFORE: batter balls in player_innings = %d" % one("SELECT SUM(balls_faced) FROM player_innings"))
step("innings where bat+wide+nb+bye+lb+pen != innings total: %d" % invariant())

# ---- 1. per-match config -------------------------------------------------
cols = [r[1] for r in cur.execute("PRAGMA table_info(matches)")]
if "balls_limit" not in cols:
    cur.execute("ALTER TABLE matches ADD COLUMN balls_limit INTEGER")
cur.execute("""
    UPDATE matches SET balls_limit = CASE
        WHEN tournament = 'hundred_male' THEN 100
        WHEN tournament LIKE 't10%'      THEN 60
        ELSE MIN(COALESCE(CAST(json_extract(raw_meta, '$.overs') AS INTEGER), 20), 20) * 6
    END""")
step("matches.balls_limit set: %s" % cur.execute(
    "SELECT balls_limit, COUNT(*) FROM matches GROUP BY 1").fetchall())

cur.execute("DROP TABLE IF EXISTS _inn_cfg")
cur.execute("""
    CREATE TABLE _inn_cfg AS
    SELECT i.id AS innings_id, i.target, m.balls_limit,
           CASE WHEN m.tournament = 'hundred_male' THEN 5
                WHEN m.tournament LIKE 't10%'      THEN 2 ELSE 6 END AS pp_hi,
           CASE WHEN m.tournament = 'hundred_male' THEN 15
                WHEN m.tournament LIKE 't10%'      THEN 8 ELSE 15 END AS mid_hi
    FROM innings i JOIN matches m ON m.id = i.match_id""")
cur.execute("CREATE UNIQUE INDEX _ix_inn_cfg ON _inn_cfg(innings_id)")

# ---- 2. phase -------------------------------------------------------------
step("recomputing deliveries.phase ...")
cur.execute("""
    UPDATE deliveries SET phase = (
        SELECT CASE WHEN deliveries.over_number <= c.pp_hi  THEN 0
                    WHEN deliveries.over_number <= c.mid_hi THEN 1 ELSE 2 END
        FROM _inn_cfg c WHERE c.innings_id = deliveries.innings_id)""")

# ---- 3. running rates (2nd innings) ----------------------------------------
step("recomputing req_rate_at_ball / crr_at_ball ...")
cur.execute("DROP TABLE IF EXISTS _rate")
cur.execute("""
    CREATE TABLE _rate AS
    SELECT id,
           SUM(total_runs) OVER w AS runs,
           SUM(CASE WHEN wide = 0 AND no_ball = 0 THEN 1 ELSE 0 END) OVER w AS balls
    FROM deliveries
    WHERE innings_id IN (SELECT innings_id FROM _inn_cfg c JOIN innings i ON i.id = c.innings_id
                         WHERE i.innings_number = 2 AND c.target IS NOT NULL)
    WINDOW w AS (PARTITION BY innings_id ORDER BY ball_number)""")
cur.execute("CREATE UNIQUE INDEX _ix_rate ON _rate(id)")
cur.execute("""
    UPDATE deliveries SET
      req_rate_at_ball = (SELECT ROUND(MAX(0, c.target - r.runs) * 1.0
                              / MAX(1, c.balls_limit - r.balls) * 6, 2)
                          FROM _rate r, _inn_cfg c
                          WHERE r.id = deliveries.id AND c.innings_id = deliveries.innings_id),
      crr_at_ball = (SELECT ROUND(r.runs * 1.0 / MAX(1, r.balls) * 6, 2)
                     FROM _rate r WHERE r.id = deliveries.id)
    WHERE id IN (SELECT id FROM _rate)""")
cur.execute("UPDATE innings SET required_rr_start = ROUND(target * 6.0 / (SELECT balls_limit FROM matches m WHERE m.id = innings.match_id), 3) WHERE innings_number = 2 AND target IS NOT NULL")
cur.execute("""
    UPDATE player_innings SET required_rr_start =
      (SELECT i.required_rr_start FROM innings i WHERE i.id = player_innings.innings_id)
    WHERE is_chase = 1""")

# ---- 4. batting aggregates --------------------------------------------------
step("recomputing player_innings balls_faced + phase splits ...")
cur.execute("DROP TABLE IF EXISTS _bat")
cur.execute("""
    CREATE TABLE _bat AS
    SELECT innings_id, batter_id,
      SUM(bat_runs) runs,
      SUM(wide = 0) balls,
      SUM(CASE WHEN phase = 0 THEN bat_runs END) pp_r, SUM(phase = 0 AND wide = 0) pp_b,
      SUM(CASE WHEN phase = 1 THEN bat_runs END) mid_r, SUM(phase = 1 AND wide = 0) mid_b,
      SUM(CASE WHEN phase = 2 THEN bat_runs END) death_r, SUM(phase = 2 AND wide = 0) death_b
    FROM deliveries GROUP BY innings_id, batter_id""")
cur.execute("CREATE UNIQUE INDEX _ix_bat ON _bat(innings_id, batter_id)")
cur.execute("""
    UPDATE player_innings SET
      runs = (SELECT runs FROM _bat b WHERE b.innings_id = player_innings.innings_id AND b.batter_id = player_innings.batter_id),
      balls_faced = (SELECT balls FROM _bat b WHERE b.innings_id = player_innings.innings_id AND b.batter_id = player_innings.batter_id),
      pp_runs = COALESCE((SELECT pp_r FROM _bat b WHERE b.innings_id = player_innings.innings_id AND b.batter_id = player_innings.batter_id), 0),
      pp_balls = (SELECT pp_b FROM _bat b WHERE b.innings_id = player_innings.innings_id AND b.batter_id = player_innings.batter_id),
      mid_runs = COALESCE((SELECT mid_r FROM _bat b WHERE b.innings_id = player_innings.innings_id AND b.batter_id = player_innings.batter_id), 0),
      mid_balls = (SELECT mid_b FROM _bat b WHERE b.innings_id = player_innings.innings_id AND b.batter_id = player_innings.batter_id),
      death_runs = COALESCE((SELECT death_r FROM _bat b WHERE b.innings_id = player_innings.innings_id AND b.batter_id = player_innings.batter_id), 0),
      death_balls = (SELECT death_b FROM _bat b WHERE b.innings_id = player_innings.innings_id AND b.batter_id = player_innings.batter_id)
    WHERE EXISTS (SELECT 1 FROM _bat b WHERE b.innings_id = player_innings.innings_id AND b.batter_id = player_innings.batter_id)""")

# ---- 5. bowling aggregates --------------------------------------------------
step("recomputing player_bowling_innings runs + phase splits ...")
cur.execute("DROP TABLE IF EXISTS _bowl")
cur.execute("""
    CREATE TABLE _bowl AS
    SELECT innings_id, bowler_id,
      SUM(wide = 0 AND no_ball = 0) balls,
      SUM(bat_runs + wide + no_ball) runs,
      SUM(CASE WHEN phase = 0 THEN bat_runs + wide + no_ball END) pp_r, SUM(phase = 0 AND wide = 0 AND no_ball = 0) pp_b,
      SUM(CASE WHEN phase = 1 THEN bat_runs + wide + no_ball END) mid_r, SUM(phase = 1 AND wide = 0 AND no_ball = 0) mid_b,
      SUM(CASE WHEN phase = 2 THEN bat_runs + wide + no_ball END) death_r, SUM(phase = 2 AND wide = 0 AND no_ball = 0) death_b
    FROM deliveries GROUP BY innings_id, bowler_id""")
cur.execute("CREATE UNIQUE INDEX _ix_bowl ON _bowl(innings_id, bowler_id)")
sub = lambda col: ("(SELECT %s FROM _bowl b WHERE b.innings_id = player_bowling_innings.innings_id "
                   "AND b.bowler_id = player_bowling_innings.bowler_id)" % col)
cur.execute(f"""
    UPDATE player_bowling_innings SET
      balls_bowled = {sub('balls')}, runs_conceded = {sub('runs')},
      pp_balls = {sub('pp_b')}, pp_runs = COALESCE({sub('pp_r')}, 0),
      mid_balls = {sub('mid_b')}, mid_runs = COALESCE({sub('mid_r')}, 0),
      death_balls = {sub('death_b')}, death_runs = COALESCE({sub('death_r')}, 0)
    WHERE EXISTS (SELECT 1 FROM _bowl b WHERE b.innings_id = player_bowling_innings.innings_id
                  AND b.bowler_id = player_bowling_innings.bowler_id)""")

for t in ("_inn_cfg", "_rate", "_bat", "_bowl"):
    cur.execute(f"DROP TABLE {t}")
con.commit()

step("AFTER: bowler runs in player_bowling_innings = %d" % one("SELECT SUM(runs_conceded) FROM player_bowling_innings"))
step("AFTER: batter balls in player_innings = %d" % one("SELECT SUM(balls_faced) FROM player_innings"))
step("AFTER: bowler runs sum == deliveries bat+wide+nb: %s" % (
    one("SELECT SUM(runs_conceded) FROM player_bowling_innings") ==
    one("SELECT SUM(bat_runs + wide + no_ball) FROM deliveries")))
con.execute("VACUUM") if "--vacuum" in sys.argv else None
con.close()
step("done. Now rerun: pipeline.py metrics / ratings / enrich_tournaments / enrich")
