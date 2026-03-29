"""
Compute and materialise all aggregated metric tables from raw delivery data.

Tables populated:
  PlayerCareerBat, PlayerCareerBowl
  PlayerPositionBat
  PlayerChaseBat
  PlayerVenueBat, PlayerVenueBowl
  players (career tallies)
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import MIN_BAT_INNINGS, MIN_BOWL_INNINGS
from src.db.schema import (
    Player, PlayerCareerBat, PlayerCareerBowl,
    PlayerPositionBat, PlayerChaseBat,
    PlayerVenueBat, PlayerVenueBowl,
    VenueDifficulty,
    PlayerPerformanceByOpponent, PlayerPerformanceBySeason,
    PlayerPerformanceByTeam, PlayerPerformanceByResult,
    PlayerDismissalAnalysis, PlayerBowlingDismissalAnalysis,
    PlayerFieldingStats, PlayerMilestone,
)


# ---------------------------------------------------------------------------
# Helper: safe division
# ---------------------------------------------------------------------------

def _safe_div(num, den, default=None):
    if den and den > 0:
        return round(num / den, 2)
    return default


def _avg(runs, innings, not_outs):
    denom = innings - not_outs
    return _safe_div(runs, denom)


def _sr(runs, balls):
    return _safe_div(runs * 100, balls)


# ---------------------------------------------------------------------------
# Load raw per-innings data
# ---------------------------------------------------------------------------

def _load_bat_raw(session: Session) -> pd.DataFrame:
    sql = text("""
        SELECT
            pi.batter_id,
            pi.innings_id,
            pi.match_id,
            pi.batting_position,
            pi.runs,
            pi.balls_faced,
            pi.fours,
            pi.sixes,
            pi.not_out,
            pi.dismissal_kind,
            pi.pp_runs,  pi.pp_balls,
            pi.mid_runs, pi.mid_balls,
            pi.death_runs, pi.death_balls,
            pi.is_chase,
            pi.chase_won,
            pi.required_rr_start,
            m.tournament,
            m.venue_id,
            m.season,
            i.batting_team_id  AS team_id,
            i.bowling_team_id  AS opponent_id,
            m.winner_id,
            m.no_result,
            i.innings_number
        FROM player_innings pi
        JOIN matches m  ON m.id  = pi.match_id
        JOIN innings i  ON i.id  = pi.innings_id
    """)
    return pd.read_sql(sql, session.bind)


def _load_bowl_raw(session: Session) -> pd.DataFrame:
    sql = text("""
        SELECT
            pbi.bowler_id,
            pbi.innings_id,
            pbi.balls_bowled,
            pbi.runs_conceded,
            pbi.wickets,
            pbi.dot_balls,
            pbi.pp_balls,  pbi.pp_runs,  pbi.pp_wickets,
            pbi.mid_balls, pbi.mid_runs, pbi.mid_wickets,
            pbi.death_balls, pbi.death_runs, pbi.death_wickets,
            m.tournament,
            m.venue_id,
            m.season,
            i.batting_team_id  AS opponent_id,
            i.bowling_team_id  AS team_id,
            m.winner_id,
            m.no_result
        FROM player_bowling_innings pbi
        JOIN matches m ON m.id = pbi.match_id
        JOIN innings i ON i.id = pbi.innings_id
    """)
    return pd.read_sql(sql, session.bind)


def _load_venue_factors(session: Session) -> dict[int, float]:
    rows = session.query(VenueDifficulty).all()
    return {r.venue_id: r.bat_factor for r in rows}


# ---------------------------------------------------------------------------
# Career batting aggregate
# ---------------------------------------------------------------------------

def _career_bat(df: pd.DataFrame, venue_factors: dict) -> pd.DataFrame:
    """
    Aggregate by player + tournament (and 'ALL').
    Returns rows ready for PlayerCareerBat.
    """
    frames = []
    for key, grp in [("ALL", df)] + list(df.groupby("tournament")):
        if isinstance(key, tuple):
            key = key[0]
        g = grp.copy()
        g["adj_runs"] = g.apply(
            lambda r: r["runs"] / venue_factors.get(r["venue_id"], 1.0), axis=1
        )

        agg = (
            g.groupby("batter_id")
            .agg(
                innings    = ("innings_id", "count"),
                not_outs   = ("not_out",    "sum"),
                runs       = ("runs",       "sum"),
                balls      = ("balls_faced","sum"),
                fours      = ("fours",      "sum"),
                sixes      = ("sixes",      "sum"),
                hs         = ("runs",       "max"),
                adj_runs   = ("adj_runs",   "sum"),
                pp_runs    = ("pp_runs",    "sum"),
                pp_balls   = ("pp_balls",   "sum"),
                mid_runs   = ("mid_runs",   "sum"),
                mid_balls  = ("mid_balls",  "sum"),
                death_runs = ("death_runs", "sum"),
                death_balls= ("death_balls","sum"),
            )
            .reset_index()
        )
        agg["fifties"]  = (g.groupby("batter_id")["runs"].apply(
            lambda x: ((x >= 50) & (x < 100)).sum()
        )).values if len(agg) > 0 else 0
        agg["hundreds"] = (g.groupby("batter_id")["runs"].apply(
            lambda x: (x >= 100).sum()
        )).values if len(agg) > 0 else 0

        denom = (agg["innings"] - agg["not_outs"]).clip(lower=1)
        agg["average"]       = (agg["runs"]     / denom).round(2)
        agg["strike_rate"]   = (agg["runs"] * 100 / agg["balls"].clip(lower=1)).round(2)
        agg["adj_average"]   = (agg["adj_runs"] / denom).round(2)
        agg["adj_sr"]        = agg["strike_rate"]   # no phase-level venue adj for now
        agg["pp_sr"]         = (agg["pp_runs"]    * 100 / agg["pp_balls"].clip(lower=1)).round(2)
        agg["mid_sr"]        = (agg["mid_runs"]   * 100 / agg["mid_balls"].clip(lower=1)).round(2)
        agg["death_sr"]      = (agg["death_runs"] * 100 / agg["death_balls"].clip(lower=1)).round(2)
        agg["tournament"]    = key
        frames.append(agg)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Career bowling aggregate
# ---------------------------------------------------------------------------

def _career_bowl(df: pd.DataFrame, venue_factors: dict) -> pd.DataFrame:
    frames = []
    for key, grp in [("ALL", df)] + list(df.groupby("tournament")):
        if isinstance(key, tuple):
            key = key[0]
        g = grp.copy()
        # adj: runs conceded adjusted for venue bat factor (easy pitch → discount)
        g["adj_runs"] = g.apply(
            lambda r: r["runs_conceded"] / venue_factors.get(r["venue_id"], 1.0), axis=1
        )

        agg = (
            g.groupby("bowler_id")
            .agg(
                innings     = ("innings_id",    "count"),
                balls       = ("balls_bowled",  "sum"),
                runs        = ("runs_conceded", "sum"),
                wickets     = ("wickets",       "sum"),
                dot_balls   = ("dot_balls",     "sum"),
                adj_runs    = ("adj_runs",      "sum"),
                pp_balls    = ("pp_balls",      "sum"),
                pp_runs     = ("pp_runs",       "sum"),
                pp_wickets  = ("pp_wickets",    "sum"),
                mid_balls   = ("mid_balls",     "sum"),
                mid_runs    = ("mid_runs",      "sum"),
                mid_wickets = ("mid_wickets",   "sum"),
                death_balls = ("death_balls",   "sum"),
                death_runs  = ("death_runs",    "sum"),
                death_wickets= ("death_wickets","sum"),
            )
            .reset_index()
        )

        agg["economy"]       = (agg["runs"]     * 6 / agg["balls"].clip(lower=1)).round(2)
        agg["average"]       = (agg["runs"]     / agg["wickets"].clip(lower=1)).round(2)
        agg["strike_rate"]   = (agg["balls"]    / agg["wickets"].clip(lower=1)).round(2)
        agg["dot_pct"]       = (agg["dot_balls"]* 100 / agg["balls"].clip(lower=1)).round(2)
        agg["adj_economy"]   = (agg["adj_runs"] * 6 / agg["balls"].clip(lower=1)).round(2)
        agg["pp_economy"]    = (agg["pp_runs"]  * 6 / agg["pp_balls"].clip(lower=1)).round(2)
        agg["mid_economy"]   = (agg["mid_runs"] * 6 / agg["mid_balls"].clip(lower=1)).round(2)
        agg["death_economy"] = (agg["death_runs"]* 6/ agg["death_balls"].clip(lower=1)).round(2)
        agg["tournament"]    = key
        frames.append(agg)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Chase / first innings split
# ---------------------------------------------------------------------------

def _chase_bat(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, is_chase), g in df.groupby(["batter_id", "is_chase"]):
        inn_type = "chase" if is_chase else "first"
        innings  = len(g)
        not_outs = g["not_out"].sum()
        runs     = g["runs"].sum()
        balls    = g["balls_faced"].sum()
        denom    = max(1, innings - not_outs)

        # High pressure: chases with req RR > 10 at start
        if is_chase:
            hp = g[g["required_rr_start"].fillna(0) > 10]
            hp_inn   = len(hp)
            hp_balls = hp["balls_faced"].sum()
            hp_runs  = hp["runs"].sum()
            hp_sr    = round(hp_runs * 100 / max(1, hp_balls), 2)
            # finishes: not out in winning chase
            finishes = g[(g["not_out"]) & (g["chase_won"] == True)].shape[0]
        else:
            hp_inn = hp_sr = finishes = None

        rows.append({
            "player_id":           pid,
            "innings_type":        inn_type,
            "innings":             innings,
            "not_outs":            not_outs,
            "runs":                runs,
            "balls":               balls,
            "average":             round(runs / denom, 2),
            "strike_rate":         round(runs * 100 / max(1, balls), 2),
            "high_pressure_innings": hp_inn,
            "high_pressure_sr":    hp_sr,
            "finishes":            finishes,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Position split
# ---------------------------------------------------------------------------

def _position_bat(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, pos), g in df.groupby(["batter_id", "batting_position"]):
        if pos is None:
            continue
        inn     = len(g)
        no      = g["not_out"].sum()
        runs    = g["runs"].sum()
        balls   = g["balls_faced"].sum()
        denom   = max(1, inn - no)
        rows.append({
            "player_id":    pid,
            "position":     int(pos),
            "innings":      inn,
            "not_outs":     no,
            "runs":         runs,
            "balls":        balls,
            "average":      round(runs / denom, 2),
            "strike_rate":  round(runs * 100 / max(1, balls), 2),
            "pp_sr":        round(g["pp_runs"].sum() * 100 / max(1, g["pp_balls"].sum()), 2),
            "mid_sr":       round(g["mid_runs"].sum() * 100 / max(1, g["mid_balls"].sum()), 2),
            "death_sr":     round(g["death_runs"].sum() * 100 / max(1, g["death_balls"].sum()), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Venue split
# ---------------------------------------------------------------------------

def _venue_bat(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, vid), g in df.groupby(["batter_id", "venue_id"]):
        inn  = len(g); no = g["not_out"].sum()
        runs = g["runs"].sum(); balls = g["balls_faced"].sum()
        rows.append({
            "player_id":  pid, "venue_id": vid,
            "innings":    inn, "not_outs": no,
            "runs":       runs, "balls": balls,
            "average":    round(runs / max(1, inn - no), 2),
            "strike_rate": round(runs * 100 / max(1, balls), 2),
        })
    return pd.DataFrame(rows)


def _venue_bowl(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, vid), g in df.groupby(["bowler_id", "venue_id"]):
        balls   = g["balls_bowled"].sum()
        runs    = g["runs_conceded"].sum()
        wickets = g["wickets"].sum()
        rows.append({
            "player_id":   pid, "venue_id": vid,
            "innings":     len(g), "balls": balls,
            "runs":        runs, "wickets": wickets,
            "economy":     round(runs * 6 / max(1, balls), 2),
            "average":     round(runs / max(1, wickets), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DB write helpers
# ---------------------------------------------------------------------------

def _safe_kwargs(model, row: dict) -> dict:
    """Return only the keys that exist as columns on *model*."""
    valid = set(model.__table__.columns.keys())
    return {k: v for k, v in row.items() if k in valid and not (isinstance(v, float) and __import__('math').isnan(v))}


# ---------------------------------------------------------------------------
# New Howstat-style breakdown computations
# ---------------------------------------------------------------------------

def _agg_bat(g: pd.DataFrame) -> dict:
    inn  = len(g)
    no   = int(g["not_out"].sum())
    runs = int(g["runs"].sum())
    balls= int(g["balls_faced"].sum())
    hs   = int(g["runs"].max()) if inn > 0 else 0
    denom = max(1, inn - no)
    return {
        "bat_innings": inn, "bat_not_outs": no, "bat_runs": runs,
        "bat_balls": balls, "bat_hs": hs,
        "bat_average":  round(runs / denom, 2),
        "bat_sr":       round(runs * 100 / max(1, balls), 2),
        "bat_fifties":  int(((g["runs"] >= 50) & (g["runs"] < 100)).sum()),
        "bat_hundreds": int((g["runs"] >= 100).sum()),
        "bat_ducks":    int(((g["runs"] == 0) & (~g["not_out"])).sum()),
    }


def _agg_bowl(g: pd.DataFrame) -> dict:
    inn   = len(g)
    balls = int(g["balls_bowled"].sum())
    runs  = int(g["runs_conceded"].sum())
    wkts  = int(g["wickets"].sum())
    return {
        "bowl_innings": inn, "bowl_balls": balls,
        "bowl_runs": runs, "bowl_wickets": wkts,
        "bowl_economy": round(runs * 6 / max(1, balls), 2),
        "bowl_average": round(runs / max(1, wkts), 2),
    }


def _perf_by_opponent(bat_df: pd.DataFrame, bowl_df: pd.DataFrame) -> list[dict]:
    rows = []
    all_ids = set(bat_df["batter_id"].unique()) | set(bowl_df["bowler_id"].unique())
    for pid in all_ids:
        bat_p  = bat_df[bat_df["batter_id"]  == pid]
        bowl_p = bowl_df[bowl_df["bowler_id"] == pid]
        opps   = set(bat_p["opponent_id"].dropna().unique()) | \
                 set(bowl_p["opponent_id"].dropna().unique())
        for opp in opps:
            if pd.isna(opp):
                continue
            ba = _agg_bat(bat_p[bat_p["opponent_id"] == opp])
            bo = _agg_bowl(bowl_p[bowl_p["opponent_id"] == opp])
            rows.append({"player_id": int(pid), "opponent_id": int(opp), **ba, **bo})
    return rows


def _perf_by_season(bat_df: pd.DataFrame, bowl_df: pd.DataFrame) -> list[dict]:
    rows = []
    for (pid, season, tourn), g_bat in bat_df.groupby(
            ["batter_id", "season", "tournament"]):
        g_bowl = bowl_df[(bowl_df["bowler_id"] == pid) &
                         (bowl_df["season"]    == season) &
                         (bowl_df["tournament"] == tourn)]
        ba = _agg_bat(g_bat)
        bo = _agg_bowl(g_bowl)
        rows.append({"player_id": int(pid), "season": str(season),
                     "tournament": tourn, **ba, **bo})
    return rows


def _perf_by_team(bat_df: pd.DataFrame, bowl_df: pd.DataFrame) -> list[dict]:
    rows = []
    all_ids = set(bat_df["batter_id"].unique()) | set(bowl_df["bowler_id"].unique())
    for pid in all_ids:
        bat_p  = bat_df[bat_df["batter_id"]  == pid]
        bowl_p = bowl_df[bowl_df["bowler_id"] == pid]
        teams  = set(bat_p["team_id"].dropna().unique()) | \
                 set(bowl_p["team_id"].dropna().unique())
        for tid in teams:
            if pd.isna(tid):
                continue
            ba = _agg_bat(bat_p[bat_p["team_id"] == tid])
            bo = _agg_bowl(bowl_p[bowl_p["team_id"] == tid])
            rows.append({"player_id": int(pid), "team_id": int(tid), **ba, **bo})
    return rows


def _perf_by_result(bat_df: pd.DataFrame, bowl_df: pd.DataFrame) -> list[dict]:
    """Split by won / lost / no_result from the batting player's perspective."""
    def _result(row):
        if row.get("no_result"):
            return "no_result"
        wid = row.get("winner_id")
        tid = row.get("team_id")
        if pd.isna(wid) or pd.isna(tid):
            return "no_result"
        return "won" if int(wid) == int(tid) else "lost"

    bat_df = bat_df.copy()
    bat_df["result"] = bat_df.apply(_result, axis=1)
    bowl_df = bowl_df.copy()
    bowl_df["result"] = bowl_df.apply(_result, axis=1)

    rows = []
    all_ids = set(bat_df["batter_id"].unique()) | set(bowl_df["bowler_id"].unique())
    for pid in all_ids:
        for res in ("won", "lost", "no_result"):
            ba = _agg_bat(bat_df[(bat_df["batter_id"]  == pid) & (bat_df["result"]  == res)])
            bo = _agg_bowl(bowl_df[(bowl_df["bowler_id"] == pid) & (bowl_df["result"] == res)])
            if ba["bat_innings"] == 0 and bo["bowl_innings"] == 0:
                continue
            rows.append({"player_id": int(pid), "result": res, **ba, **bo})
    return rows


def _dismissal_analysis_bat(bat_df: pd.DataFrame) -> list[dict]:
    """How each batter was dismissed."""
    dismissed = bat_df[~bat_df["not_out"] & bat_df["dismissal_kind"].notna()]
    rows = []
    for pid, g in dismissed.groupby("batter_id"):
        total = len(g)
        for kind, cnt in g["dismissal_kind"].value_counts().items():
            rows.append({
                "player_id":      int(pid),
                "dismissal_kind": str(kind),
                "count":          int(cnt),
                "pct":            round(cnt * 100 / total, 1),
            })
    return rows


def _dismissal_analysis_bowl(session: Session) -> list[dict]:
    """How each bowler took their wickets — from the deliveries table."""
    sql = text("""
        SELECT
            d.bowler_id AS player_id,
            d.wicket_kind AS dismissal_kind,
            COUNT(*) AS cnt
        FROM deliveries d
        WHERE d.is_wicket = 1
          AND d.wicket_kind NOT IN ('run out','retired hurt','retired out',
                                    'obstructing the field','timed out',
                                    'handled the ball')
          AND d.wicket_kind IS NOT NULL
        GROUP BY d.bowler_id, d.wicket_kind
    """)
    df = pd.read_sql(sql, session.bind)
    totals = df.groupby("player_id")["cnt"].transform("sum")
    df["pct"] = (df["cnt"] * 100 / totals).round(1)
    return df.rename(columns={"cnt": "count"}).to_dict(orient="records")


def _fielding_stats(session: Session) -> list[dict]:
    """
    Count fielding catches from deliveries.
    A catch is a wicket where wicket_kind = 'caught' or 'caught and bowled'
    and the player_out is NOT the bowler (caught) or IS the bowler (c&b).
    We approximate by crediting any fielder listed in the raw JSON, but since
    cricsheet JSON doesn't always name fielders in a dedicated column, we count
    all 'caught' dismissals per innings and attribute them to fielders via
    a separate fielder query.
    """
    sql = text("""
        SELECT
            d.player_out_id,
            d.innings_id,
            d.wicket_kind,
            d.bowler_id
        FROM deliveries d
        WHERE d.is_wicket = 1
          AND d.wicket_kind IN ('caught', 'caught and bowled', 'stumped')
    """)
    df = pd.read_sql(sql, session.bind)
    # For 'caught and bowled', the bowler is the fielder
    # For 'caught', we don't yet have the fielder name from the parser
    # We count these as team catches (can be improved with fielder parsing)
    # For now: count bowler catches from 'caught and bowled'
    cb = df[df["wicket_kind"] == "caught and bowled"]
    rows = []
    for pid, g in cb.groupby("bowler_id"):
        total = len(g)
        # max catches in a single innings
        max_inn = g.groupby("innings_id").size().max() if total > 0 else 0
        rows.append({
            "player_id": int(pid),
            "tournament": "ALL",
            "catches": int(total),
            "most_catches_inn": int(max_inn),
        })
    return rows


def _compute_median(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.median())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def rebuild_all_metrics(session: Session):
    """Rebuild every aggregate table from scratch."""
    print("Loading raw batting data…")
    bat_df  = _load_bat_raw(session)
    print("Loading raw bowling data…")
    bowl_df = _load_bowl_raw(session)
    print(f"  {len(bat_df)} batting innings, {len(bowl_df)} bowling innings")

    venue_factors = _load_venue_factors(session)

    print("Computing career batting…")
    career_bat_df = _career_bat(bat_df, venue_factors)
    session.query(PlayerCareerBat).delete()
    for _, row in career_bat_df.iterrows():
        r = row.to_dict()
        pid = int(r["batter_id"])
        tourn = r["tournament"]
        # Compute extras: median, ducks, thirties, times opened, top-scored
        subset = bat_df[bat_df["batter_id"] == pid]
        if tourn != "ALL":
            subset = subset[subset["tournament"] == tourn]
        median_s = _compute_median(subset["runs"])
        ducks    = int(((subset["runs"] == 0) & (~subset["not_out"])).sum())
        thirties = int(((subset["runs"] >= 30) & (subset["runs"] < 50)).sum())
        opened   = int((subset["batting_position"] <= 2).sum()) if "batting_position" in subset else 0

        session.add(PlayerCareerBat(
            player_id     = pid,
            tournament    = tourn,
            innings       = int(r["innings"]),
            not_outs      = int(r.get("not_outs", 0)),
            runs          = int(r["runs"]),
            balls         = int(r["balls"]),
            hs            = int(r["hs"]),
            thirties      = thirties,
            fifties       = int(r.get("fifties", 0)),
            hundreds      = int(r.get("hundreds", 0)),
            ducks         = ducks,
            fours         = int(r.get("fours", 0)),
            sixes         = int(r.get("sixes", 0)),
            median_score  = median_s,
            times_opened  = opened,
            average       = r["average"],
            strike_rate   = r["strike_rate"],
            pp_sr         = r["pp_sr"],
            mid_sr        = r["mid_sr"],
            death_sr      = r["death_sr"],
            adj_average   = r["adj_average"],
            adj_strike_rate = r["adj_sr"],
        ))

    print("Computing career bowling…")
    career_bowl_df = _career_bowl(bowl_df, venue_factors)
    session.query(PlayerCareerBowl).delete()
    for _, row in career_bowl_df.iterrows():
        r = row.to_dict()
        session.add(PlayerCareerBowl(
            player_id   = int(r["bowler_id"]),
            tournament  = r["tournament"],
            innings     = int(r["innings"]),
            balls       = int(r["balls"]),
            runs        = int(r["runs"]),
            wickets     = int(r["wickets"]),
            dot_balls   = int(r["dot_balls"]),
            economy     = r["economy"],
            average     = r["average"],
            strike_rate = r["strike_rate"],
            dot_pct     = r["dot_pct"],
            pp_economy  = r["pp_economy"],
            mid_economy = r["mid_economy"],
            death_economy = r["death_economy"],
            adj_economy = r["adj_economy"],
        ))

    print("Computing chase splits…")
    chase_df = _chase_bat(bat_df)
    session.query(PlayerChaseBat).delete()
    for _, row in chase_df.iterrows():
        r = row.to_dict()
        session.add(PlayerChaseBat(**{k: (None if pd.isna(v) else v)
                                      for k, v in r.items()}))

    print("Computing position splits…")
    pos_df = _position_bat(bat_df)
    session.query(PlayerPositionBat).delete()
    for _, row in pos_df.iterrows():
        r = row.to_dict()
        session.add(PlayerPositionBat(**r))

    print("Computing venue batting splits…")
    vb_df = _venue_bat(bat_df)
    session.query(PlayerVenueBat).delete()
    for _, row in vb_df.iterrows():
        r = row.to_dict()
        session.add(PlayerVenueBat(**r))

    print("Computing venue bowling splits…")
    vbowl_df = _venue_bowl(bowl_df)
    session.query(PlayerVenueBowl).delete()
    for _, row in vbowl_df.iterrows():
        r = row.to_dict()
        session.add(PlayerVenueBowl(**r))

    print("Computing performance by opponent…")
    session.query(PlayerPerformanceByOpponent).delete()
    for row in _perf_by_opponent(bat_df, bowl_df):
        session.add(PlayerPerformanceByOpponent(**_safe_kwargs(PlayerPerformanceByOpponent, row)))

    print("Computing performance by season…")
    session.query(PlayerPerformanceBySeason).delete()
    for row in _perf_by_season(bat_df, bowl_df):
        session.add(PlayerPerformanceBySeason(**_safe_kwargs(PlayerPerformanceBySeason, row)))

    print("Computing performance by team…")
    session.query(PlayerPerformanceByTeam).delete()
    for row in _perf_by_team(bat_df, bowl_df):
        session.add(PlayerPerformanceByTeam(**_safe_kwargs(PlayerPerformanceByTeam, row)))

    print("Computing performance by match result…")
    session.query(PlayerPerformanceByResult).delete()
    for row in _perf_by_result(bat_df, bowl_df):
        session.add(PlayerPerformanceByResult(**_safe_kwargs(PlayerPerformanceByResult, row)))

    print("Computing batting dismissal analysis…")
    session.query(PlayerDismissalAnalysis).delete()
    for row in _dismissal_analysis_bat(bat_df):
        session.add(PlayerDismissalAnalysis(**row))

    print("Computing bowling dismissal analysis…")
    session.query(PlayerBowlingDismissalAnalysis).delete()
    for row in _dismissal_analysis_bowl(session):
        session.add(PlayerBowlingDismissalAnalysis(**row))

    print("Computing fielding stats…")
    session.query(PlayerFieldingStats).delete()
    for row in _fielding_stats(session):
        session.add(PlayerFieldingStats(**row))

    session.commit()
    print("All metric tables rebuilt.")
