"""League-scoped versions of the Player Explorer deep-dive queries.

Used when a single tournament is chosen in the sidebar. Everything is computed from the match-level tables
(player_innings, player_bowling_innings, deliveries) joined to matches, so each number is for that league only.
All values travel as bound parameters (:pid, :t); nothing is string-built from user input.
`run(q, **params)` is the dashboard's sql() helper.
"""
import numpy as np
import pandas as pd

_BAT_AVG = "ROUND(SUM(pi.runs) * 1.0 / NULLIF(COUNT(*) - SUM(pi.not_out), 0), 2)"
_BAT_SR = "ROUND(SUM(pi.runs) * 100.0 / NULLIF(SUM(pi.balls_faced), 0), 2)"
_T20I_TEAMS = """(SELECT team1_id FROM matches WHERE tournament = 't20i_male'
                  UNION SELECT team2_id FROM matches WHERE tournament = 't20i_male')"""


def seasons(run, pid, t):
    return run("""SELECT season, bat_innings AS innings, bat_runs AS runs, bat_average AS average, bat_sr AS sr
                  FROM player_perf_by_season WHERE player_id = :pid AND tournament = :t ORDER BY season""", pid=pid, t=t)


def phase_row(run, pid, t):
    return run("""SELECT pp_sr, mid_sr, death_sr, adj_strike_rate AS overall_sr, pp_runs, pp_balls, mid_runs, mid_balls,
                         death_runs, death_balls FROM player_career_bat WHERE player_id = :pid AND tournament = :t""", pid=pid, t=t)


def positions(run, pid, t):
    return run(f"""SELECT pi.batting_position AS position, COUNT(*) AS innings, {_BAT_AVG} AS average, {_BAT_SR} AS strike_rate,
                          ROUND(SUM(pi.pp_runs) * 100.0 / NULLIF(SUM(pi.pp_balls), 0), 2) AS pp_sr,
                          ROUND(SUM(pi.mid_runs) * 100.0 / NULLIF(SUM(pi.mid_balls), 0), 2) AS mid_sr,
                          ROUND(SUM(pi.death_runs) * 100.0 / NULLIF(SUM(pi.death_balls), 0), 2) AS death_sr
                   FROM player_innings pi JOIN matches m ON m.id = pi.match_id
                   WHERE pi.batter_id = :pid AND m.tournament = :t AND pi.balls_faced > 0
                   GROUP BY pi.batting_position ORDER BY pi.batting_position""", pid=pid, t=t)


def by_opponent(run, pid, t, team_type="all"):
    cond = ""
    if team_type == "countries": cond = f"AND tm.id IN {_T20I_TEAMS}"
    elif team_type == "clubs": cond = f"AND tm.id NOT IN {_T20I_TEAMS}"
    bat = run(f"""SELECT tm.name AS opponent, COUNT(*) AS inn, SUM(pi.runs) AS runs, {_BAT_AVG} AS avg, {_BAT_SR} AS sr,
                         SUM(CASE WHEN pi.runs >= 50 AND pi.runs < 100 THEN 1 ELSE 0 END) AS fifties,
                         SUM(CASE WHEN pi.runs >= 100 THEN 1 ELSE 0 END) AS hundreds,
                         SUM(CASE WHEN pi.runs = 0 AND pi.not_out = 0 THEN 1 ELSE 0 END) AS ducks
                  FROM player_innings pi JOIN innings i ON i.id = pi.innings_id JOIN matches m ON m.id = pi.match_id
                  JOIN teams tm ON tm.id = i.bowling_team_id
                  WHERE pi.batter_id = :pid AND m.tournament = :t AND pi.balls_faced > 0 {cond} GROUP BY tm.id""", pid=pid, t=t)
    bowl = run(f"""SELECT tm.name AS opponent, SUM(pb.wickets) AS wkts,
                          ROUND(SUM(pb.runs_conceded) * 6.0 / NULLIF(SUM(pb.balls_bowled), 0), 2) AS econ
                   FROM player_bowling_innings pb JOIN innings i ON i.id = pb.innings_id JOIN matches m ON m.id = pb.match_id
                   JOIN teams tm ON tm.id = i.batting_team_id
                   WHERE pb.bowler_id = :pid AND m.tournament = :t AND pb.balls_bowled > 0 {cond} GROUP BY tm.id""", pid=pid, t=t)
    if bat.empty and bowl.empty:
        return pd.DataFrame(columns=["opponent", "inn", "runs", "avg", "sr", "fifties", "hundreds", "ducks", "wkts", "econ"])
    df = bat.merge(bowl, on="opponent", how="outer")
    df["inn"] = df["inn"].fillna(0).astype(int)
    return df.sort_values("inn", ascending=False).reset_index(drop=True)


def vs_bowler(run, pid, t):
    return run("""SELECT p.cricsheet_key AS bowler, p.country, COUNT(*) AS balls, SUM(d.bat_runs) AS runs,
                         SUM(CASE WHEN d.is_wicket = 1 AND d.player_out_id = :pid THEN 1 ELSE 0 END) AS dismissals,
                         ROUND(SUM(d.bat_runs) * 100.0 / COUNT(*), 1) AS sr,
                         ROUND(SUM(d.bat_runs) * 1.0 / NULLIF(SUM(CASE WHEN d.is_wicket = 1 AND d.player_out_id = :pid THEN 1 ELSE 0 END), 0), 1) AS avg
                  FROM deliveries d JOIN players p ON p.id = d.bowler_id JOIN innings i ON i.id = d.innings_id JOIN matches m ON m.id = i.match_id
                  WHERE d.batter_id = :pid AND d.wide = 0 AND m.tournament = :t
                  GROUP BY d.bowler_id HAVING balls >= 6 ORDER BY balls DESC LIMIT 50""", pid=pid, t=t)


def venues(run, pid, t):
    return run(f"""SELECT v.name AS venue, COUNT(*) AS innings, SUM(pi.runs) AS runs, {_BAT_AVG} AS average, {_BAT_SR} AS strike_rate,
                          COALESCE(vd.bat_factor, 1.0) AS bat_factor
                   FROM player_innings pi JOIN matches m ON m.id = pi.match_id JOIN venues v ON v.id = m.venue_id
                   LEFT JOIN venue_difficulty vd ON vd.venue_id = m.venue_id
                   WHERE pi.batter_id = :pid AND m.tournament = :t AND pi.balls_faced > 0
                   GROUP BY m.venue_id HAVING COUNT(*) >= 2 ORDER BY COUNT(*) DESC""", pid=pid, t=t)


def milestones(run, pid, t):
    return run("""SELECT pm.milestone_type, pm.value, pm.match_date, pm.tournament, v.name AS venue, tm.name AS opposition
                  FROM player_milestones pm LEFT JOIN venues v ON v.id = pm.venue_id LEFT JOIN teams tm ON tm.id = pm.opposition_id
                  WHERE pm.player_id = :pid AND pm.tournament = :t ORDER BY pm.match_date DESC""", pid=pid, t=t)


def dismissal_bat(run, pid, t):
    return run("""SELECT d.wicket_kind AS dismissal_kind, COUNT(*) AS count,
                         ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
                  FROM deliveries d JOIN innings i ON i.id = d.innings_id JOIN matches m ON m.id = i.match_id
                  WHERE d.player_out_id = :pid AND d.is_wicket = 1 AND d.wicket_kind IS NOT NULL AND m.tournament = :t
                  GROUP BY d.wicket_kind ORDER BY count DESC""", pid=pid, t=t)


def dismissal_bowl(run, pid, t):
    return run("""SELECT d.wicket_kind AS dismissal_kind, COUNT(*) AS count,
                         ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
                  FROM deliveries d JOIN innings i ON i.id = d.innings_id JOIN matches m ON m.id = i.match_id
                  WHERE d.bowler_id = :pid AND d.is_wicket = 1 AND m.tournament = :t
                    AND d.wicket_kind NOT IN ('run out', 'obstructing the field', 'retired hurt', 'retired out', 'timed out', 'handled the ball')
                  GROUP BY d.wicket_kind ORDER BY count DESC""", pid=pid, t=t)


def dismissal_heatmap(run, pid, t):
    return run("""SELECT d.wicket_kind AS dismissal,
                         CASE d.phase WHEN 0 THEN 'Powerplay' WHEN 1 THEN 'Middle' ELSE 'Death' END AS phase, COUNT(*) AS count
                  FROM deliveries d JOIN innings i ON i.id = d.innings_id JOIN matches m ON m.id = i.match_id
                  WHERE d.player_out_id = :pid AND d.is_wicket = 1 AND d.wicket_kind IS NOT NULL AND m.tournament = :t
                  GROUP BY d.wicket_kind, d.phase ORDER BY d.phase, count DESC""", pid=pid, t=t)


def recent_innings(run, pid, t):
    return run("""SELECT pi.runs, pi.balls_faced, pi.not_out, pi.dismissal_kind, m.match_date, m.season, tm.name AS opposition
                  FROM player_innings pi JOIN matches m ON m.id = pi.match_id JOIN innings i ON i.id = pi.innings_id
                  LEFT JOIN teams tm ON tm.id = i.bowling_team_id
                  WHERE pi.batter_id = :pid AND pi.balls_faced > 0 AND m.tournament = :t ORDER BY m.match_date ASC""", pid=pid, t=t)


def form(run, pid, t):
    """Same definitions as src/analytics/similarity.py (mean runs per innings), but only this league's innings."""
    raw = recent_innings(run, pid, t)
    cols = ["avg_5", "avg_10", "avg_20", "sr_5", "sr_10", "sr_20", "career_avg", "career_sr", "cv",
            "breakout_flag", "breakout_delta", "innings_total"]
    if raw.empty:
        return pd.DataFrame(columns=cols)
    runs, balls, n = raw["runs"].values.astype(float), raw["balls_faced"].values.astype(float), len(raw)
    career_avg = float(np.mean(runs))
    career_sr = float(runs.sum() / balls.sum() * 100) if balls.sum() > 0 else None
    cv = float(np.std(runs) / career_avg * 100) if career_avg > 0 else None
    avg = lambda w: float(np.mean(runs[-w:])) if n >= 3 else None
    sr = lambda w: float(runs[-w:].sum() / balls[-w:].sum() * 100) if n >= 3 and balls[-w:].sum() > 0 else None
    avg10 = avg(10)
    return pd.DataFrame([{"avg_5": avg(5), "avg_10": avg10, "avg_20": avg(20), "sr_5": sr(5), "sr_10": sr(10), "sr_20": sr(20),
                          "career_avg": career_avg, "career_sr": career_sr, "cv": cv,
                          "breakout_flag": bool(avg10 and career_avg and avg10 > career_avg * 1.2 and n >= 10),
                          "breakout_delta": (avg10 - career_avg) if avg10 else None, "innings_total": n}])
