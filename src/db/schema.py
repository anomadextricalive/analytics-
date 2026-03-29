"""
SQLAlchemy ORM models — exhaustive schema that captures every piece of
metadata cricsheet provides plus derived aggregates.

Table hierarchy:
  players ──┐
  venues  ──┤── matches ── innings ── deliveries
  teams   ──┘
                        └── player_innings (bat)
                        └── player_bowling_innings (bowl)
                        └── partnerships
  player_career_bat      (aggregated)
  player_career_bowl     (aggregated)
  player_phase_bat       (per phase: PP/middle/death)
  player_phase_bowl
  player_position_bat    (by batting position 1-11)
  player_venue_bat       (per ground)
  player_venue_bowl
  player_chase_bat       (chase vs first innings)
  venue_difficulty       (computed pitch factor)
"""

from datetime import date
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean,
    Date, ForeignKey, UniqueConstraint, Index, Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Core reference tables
# ---------------------------------------------------------------------------

class Player(Base):
    __tablename__ = "players"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    cricsheet_key   = Column(String, unique=True, nullable=False)  # e.g. "V Kohli"
    full_name       = Column(String)
    country         = Column(String)
    batting_style   = Column(String)   # right/left
    bowling_style   = Column(String)

    # populated from cricinfo/manual enrichment when available
    date_of_birth   = Column(Date)
    debut_date      = Column(Date)

    # Raw career tallies (all T20) ----------------------------------------
    # Batting
    t20_bat_innings     = Column(Integer, default=0)
    t20_bat_runs        = Column(Integer, default=0)
    t20_bat_balls       = Column(Integer, default=0)
    t20_bat_not_outs    = Column(Integer, default=0)
    t20_bat_fours       = Column(Integer, default=0)
    t20_bat_sixes       = Column(Integer, default=0)
    t20_bat_thirties    = Column(Integer, default=0)   # 30–49 runs
    t20_bat_fifties     = Column(Integer, default=0)
    t20_bat_hundreds    = Column(Integer, default=0)
    t20_bat_ducks       = Column(Integer, default=0)
    t20_bat_hs          = Column(Integer, default=0)
    t20_bat_median      = Column(Float)                # median score
    t20_bat_times_opened = Column(Integer, default=0)  # innings as opener
    t20_bat_top_scored  = Column(Integer, default=0)   # times top scorer in innings

    # Bowling
    t20_bowl_innings    = Column(Integer, default=0)
    t20_bowl_balls      = Column(Integer, default=0)
    t20_bowl_runs       = Column(Integer, default=0)
    t20_bowl_wickets    = Column(Integer, default=0)
    t20_bowl_maidens    = Column(Integer, default=0)
    t20_bowl_bbi_runs   = Column(Integer)
    t20_bowl_bbi_wkts   = Column(Integer)
    t20_bowl_four_wkt_hauls = Column(Integer, default=0)  # 4+ wicket hauls

    # Fielding
    t20_field_catches         = Column(Integer, default=0)
    t20_field_most_catches_inn = Column(Integer, default=0)  # best in a single innings

    # Milestones
    t20_player_of_match       = Column(Integer, default=0)

    # relationships
    bat_innings     = relationship("PlayerInnings",        back_populates="batter")
    bowl_innings    = relationship("PlayerBowlingInnings", back_populates="bowler")


class Venue(Base):
    __tablename__ = "venues"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    name            = Column(String, unique=True, nullable=False)
    city            = Column(String)
    country         = Column(String)

    # Pitch character (populated by pitch regression)
    avg_first_inn_score     = Column(Float)
    avg_second_inn_score    = Column(Float)
    avg_first_inn_wickets   = Column(Float)
    pace_index              = Column(Float)   # 0-1: spin-favoring → pace-favoring
    boundary_ease           = Column(Float)   # relative boundary % vs global avg
    venue_bat_factor        = Column(Float, default=1.0)   # >1 = batter-friendly
    venue_bowl_factor       = Column(Float, default=1.0)   # >1 = bowler-friendly

    matches         = relationship("Match", back_populates="venue")


class Team(Base):
    __tablename__ = "teams"

    id      = Column(Integer, primary_key=True, autoincrement=True)
    name    = Column(String, unique=True, nullable=False)


# ---------------------------------------------------------------------------
# Match and innings
# ---------------------------------------------------------------------------

class Match(Base):
    __tablename__ = "matches"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    cricsheet_id    = Column(String, unique=True, nullable=False)  # filename stem

    match_date      = Column(Date)
    tournament      = Column(String)    # e.g. "IPL", "T20I", "PSL"
    gender          = Column(String)    # male / female
    match_type      = Column(String)    # always T20 here
    season          = Column(String)

    venue_id        = Column(Integer, ForeignKey("venues.id"))
    team1_id        = Column(Integer, ForeignKey("teams.id"))
    team2_id        = Column(Integer, ForeignKey("teams.id"))

    toss_winner_id  = Column(Integer, ForeignKey("teams.id"))
    toss_decision   = Column(String)    # bat / field

    winner_id       = Column(Integer, ForeignKey("teams.id"))
    win_by_runs     = Column(Integer)
    win_by_wickets  = Column(Integer)
    no_result       = Column(Boolean, default=False)

    # derived: did the team batting first win?
    chasing_team_won = Column(Boolean)

    # raw JSON blob for any extra cricsheet fields
    raw_meta        = Column(Text)

    venue           = relationship("Venue",  back_populates="matches")
    team1           = relationship("Team",   foreign_keys=[team1_id])
    team2           = relationship("Team",   foreign_keys=[team2_id])
    winner          = relationship("Team",   foreign_keys=[winner_id])
    innings         = relationship("Innings", back_populates="match",
                                   order_by="Innings.innings_number")


class Innings(Base):
    __tablename__ = "innings"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    match_id        = Column(Integer, ForeignKey("matches.id"), nullable=False)
    innings_number  = Column(Integer, nullable=False)   # 1 or 2
    batting_team_id = Column(Integer, ForeignKey("teams.id"))
    bowling_team_id = Column(Integer, ForeignKey("teams.id"))

    target          = Column(Integer)   # only for 2nd innings
    total_runs      = Column(Integer)
    total_wickets   = Column(Integer)
    total_balls     = Column(Integer)
    extras          = Column(Integer)

    # did batting team successfully chase? (2nd innings only)
    chase_successful = Column(Boolean)
    required_rr_start = Column(Float)   # target / (120 balls)

    match           = relationship("Match",   back_populates="innings")
    batting_team    = relationship("Team",    foreign_keys=[batting_team_id])
    bowling_team    = relationship("Team",    foreign_keys=[bowling_team_id])
    deliveries      = relationship("Delivery", back_populates="innings",
                                   order_by="Delivery.ball_number")
    player_innings  = relationship("PlayerInnings",        back_populates="innings")
    bowl_innings    = relationship("PlayerBowlingInnings", back_populates="innings")
    partnerships    = relationship("Partnership",          back_populates="innings")

    __table_args__ = (
        UniqueConstraint("match_id", "innings_number"),
    )


# ---------------------------------------------------------------------------
# Ball-by-ball
# ---------------------------------------------------------------------------

class Delivery(Base):
    """One row per legal + illegal delivery."""
    __tablename__ = "deliveries"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    innings_id      = Column(Integer, ForeignKey("innings.id"), nullable=False)

    # position in innings
    over_number     = Column(Integer, nullable=False)    # 1-based
    ball_in_over    = Column(Integer, nullable=False)    # 1-based (legal only)
    ball_number     = Column(Integer, nullable=False)    # global sort key

    # phase (0=powerplay, 1=middle, 2=death) — derived
    phase           = Column(Integer)

    batter_id       = Column(Integer, ForeignKey("players.id"))
    non_striker_id  = Column(Integer, ForeignKey("players.id"))
    bowler_id       = Column(Integer, ForeignKey("players.id"))

    # runs
    bat_runs        = Column(Integer, default=0)
    extras          = Column(Integer, default=0)
    total_runs      = Column(Integer, default=0)
    is_boundary_4   = Column(Boolean, default=False)
    is_boundary_6   = Column(Boolean, default=False)
    is_dot          = Column(Boolean, default=False)

    # extras breakdown
    wide            = Column(Integer, default=0)
    no_ball         = Column(Integer, default=0)
    bye             = Column(Integer, default=0)
    leg_bye         = Column(Integer, default=0)
    penalty         = Column(Integer, default=0)

    # wicket
    is_wicket       = Column(Boolean, default=False)
    wicket_kind     = Column(String)   # caught, bowled, lbw, run out, ...
    player_out_id   = Column(Integer, ForeignKey("players.id"))

    # chase pressure context
    req_rate_at_ball = Column(Float)   # required run rate at THIS ball
    crr_at_ball      = Column(Float)   # current run rate

    innings         = relationship("Innings", back_populates="deliveries")

    __table_args__ = (
        Index("ix_del_innings_ball", "innings_id", "ball_number"),
        Index("ix_del_batter",  "batter_id"),
        Index("ix_del_bowler",  "bowler_id"),
    )


# ---------------------------------------------------------------------------
# Per-innings aggregates (denormalised for fast queries)
# ---------------------------------------------------------------------------

class PlayerInnings(Base):
    """One row per batter per innings."""
    __tablename__ = "player_innings"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    innings_id      = Column(Integer, ForeignKey("innings.id"), nullable=False)
    match_id        = Column(Integer, ForeignKey("matches.id"), nullable=False)
    batter_id       = Column(Integer, ForeignKey("players.id"), nullable=False)
    team_id         = Column(Integer, ForeignKey("teams.id"))

    batting_position = Column(Integer)   # 1=opener, 2=opener, 3..11

    runs            = Column(Integer, default=0)
    balls_faced     = Column(Integer, default=0)
    fours           = Column(Integer, default=0)
    sixes           = Column(Integer, default=0)
    not_out         = Column(Boolean, default=False)
    dismissal_kind  = Column(String)

    # phase breakdown
    pp_runs         = Column(Integer, default=0)
    pp_balls        = Column(Integer, default=0)
    mid_runs        = Column(Integer, default=0)
    mid_balls       = Column(Integer, default=0)
    death_runs      = Column(Integer, default=0)
    death_balls     = Column(Integer, default=0)

    # context
    is_chase        = Column(Boolean, default=False)
    chase_won       = Column(Boolean)    # did batting team win (2nd inn only)
    required_rr_start = Column(Float)

    innings         = relationship("Innings", back_populates="player_innings")
    batter          = relationship("Player",  back_populates="bat_innings")

    __table_args__ = (
        UniqueConstraint("innings_id", "batter_id"),
        Index("ix_pi_batter", "batter_id"),
    )


class PlayerBowlingInnings(Base):
    """One row per bowler per innings."""
    __tablename__ = "player_bowling_innings"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    innings_id      = Column(Integer, ForeignKey("innings.id"), nullable=False)
    match_id        = Column(Integer, ForeignKey("matches.id"), nullable=False)
    bowler_id       = Column(Integer, ForeignKey("players.id"), nullable=False)
    team_id         = Column(Integer, ForeignKey("teams.id"))

    overs_bowled    = Column(Float, default=0.0)
    balls_bowled    = Column(Integer, default=0)
    runs_conceded   = Column(Integer, default=0)
    wickets         = Column(Integer, default=0)
    maidens         = Column(Integer, default=0)
    dot_balls       = Column(Integer, default=0)
    wides           = Column(Integer, default=0)
    no_balls        = Column(Integer, default=0)
    boundaries_con  = Column(Integer, default=0)   # 4s + 6s conceded

    # phase breakdown
    pp_balls        = Column(Integer, default=0)
    pp_runs         = Column(Integer, default=0)
    pp_wickets      = Column(Integer, default=0)
    mid_balls       = Column(Integer, default=0)
    mid_runs        = Column(Integer, default=0)
    mid_wickets     = Column(Integer, default=0)
    death_balls     = Column(Integer, default=0)
    death_runs      = Column(Integer, default=0)
    death_wickets   = Column(Integer, default=0)

    innings         = relationship("Innings", back_populates="bowl_innings")
    bowler          = relationship("Player",  back_populates="bowl_innings")

    __table_args__ = (
        UniqueConstraint("innings_id", "bowler_id"),
        Index("ix_pbi_bowler", "bowler_id"),
    )


class Partnership(Base):
    """Run partnership between two batters in an innings."""
    __tablename__ = "partnerships"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    innings_id      = Column(Integer, ForeignKey("innings.id"), nullable=False)
    match_id        = Column(Integer, ForeignKey("matches.id"), nullable=False)
    wicket_number   = Column(Integer, nullable=False)   # 1st, 2nd, ... 10th

    batter1_id      = Column(Integer, ForeignKey("players.id"))
    batter2_id      = Column(Integer, ForeignKey("players.id"))

    runs            = Column(Integer, default=0)
    balls           = Column(Integer, default=0)
    batter1_runs    = Column(Integer, default=0)
    batter2_runs    = Column(Integer, default=0)

    innings         = relationship("Innings", back_populates="partnerships")


# ---------------------------------------------------------------------------
# Derived / aggregated stats tables (materialised views in SQL land)
# ---------------------------------------------------------------------------

class PlayerCareerBat(Base):
    """Full career batting aggregate per player (all T20, rebuilt on demand)."""
    __tablename__ = "player_career_bat"

    player_id       = Column(Integer, ForeignKey("players.id"), primary_key=True)
    tournament      = Column(String,  primary_key=True)   # "ALL" or specific

    innings         = Column(Integer, default=0)
    not_outs        = Column(Integer, default=0)
    runs            = Column(Integer, default=0)
    balls           = Column(Integer, default=0)
    hs              = Column(Integer, default=0)
    thirties        = Column(Integer, default=0)   # scores 30-49
    fifties         = Column(Integer, default=0)
    hundreds        = Column(Integer, default=0)
    ducks           = Column(Integer, default=0)
    fours           = Column(Integer, default=0)
    sixes           = Column(Integer, default=0)
    median_score    = Column(Float)
    times_opened    = Column(Integer, default=0)
    top_scored      = Column(Integer, default=0)   # times top scorer in innings

    average         = Column(Float)
    strike_rate     = Column(Float)

    # phase SRs
    pp_sr           = Column(Float)
    mid_sr          = Column(Float)
    death_sr        = Column(Float)

    # adjusted for venue difficulty
    adj_average     = Column(Float)
    adj_strike_rate = Column(Float)


class PlayerCareerBowl(Base):
    __tablename__ = "player_career_bowl"

    player_id       = Column(Integer, ForeignKey("players.id"), primary_key=True)
    tournament      = Column(String,  primary_key=True)

    innings         = Column(Integer, default=0)
    balls           = Column(Integer, default=0)
    runs            = Column(Integer, default=0)
    wickets         = Column(Integer, default=0)
    dot_balls       = Column(Integer, default=0)
    economy         = Column(Float)
    average         = Column(Float)
    strike_rate     = Column(Float)
    dot_pct         = Column(Float)

    pp_economy      = Column(Float)
    mid_economy     = Column(Float)
    death_economy   = Column(Float)

    adj_economy     = Column(Float)


class PlayerPositionBat(Base):
    """Batting stats split by batting position."""
    __tablename__ = "player_position_bat"

    player_id       = Column(Integer, ForeignKey("players.id"), primary_key=True)
    position        = Column(Integer, primary_key=True)   # 1–11

    innings         = Column(Integer, default=0)
    not_outs        = Column(Integer, default=0)
    runs            = Column(Integer, default=0)
    balls           = Column(Integer, default=0)
    average         = Column(Float)
    strike_rate     = Column(Float)
    pp_sr           = Column(Float)
    mid_sr          = Column(Float)
    death_sr        = Column(Float)


class PlayerChaseBat(Base):
    """Chase vs first-innings batting split."""
    __tablename__ = "player_chase_bat"

    player_id       = Column(Integer, ForeignKey("players.id"), primary_key=True)
    innings_type    = Column(String, primary_key=True)   # "chase" | "first"

    innings         = Column(Integer, default=0)
    not_outs        = Column(Integer, default=0)
    runs            = Column(Integer, default=0)
    balls           = Column(Integer, default=0)
    average         = Column(Float)
    strike_rate     = Column(Float)

    # chase-specific pressure metrics
    high_pressure_innings = Column(Integer)  # chases with RRR > 10
    high_pressure_sr      = Column(Float)
    finishes              = Column(Integer)  # not out in winning chase


class PlayerVenueBat(Base):
    """Batting performance broken down by venue."""
    __tablename__ = "player_venue_bat"

    player_id       = Column(Integer, ForeignKey("players.id"), primary_key=True)
    venue_id        = Column(Integer, ForeignKey("venues.id"), primary_key=True)

    innings         = Column(Integer, default=0)
    not_outs        = Column(Integer, default=0)
    runs            = Column(Integer, default=0)
    balls           = Column(Integer, default=0)
    average         = Column(Float)
    strike_rate     = Column(Float)


class PlayerVenueBowl(Base):
    __tablename__ = "player_venue_bowl"

    player_id       = Column(Integer, ForeignKey("players.id"), primary_key=True)
    venue_id        = Column(Integer, ForeignKey("venues.id"), primary_key=True)

    innings         = Column(Integer, default=0)
    balls           = Column(Integer, default=0)
    runs            = Column(Integer, default=0)
    wickets         = Column(Integer, default=0)
    economy         = Column(Float)
    average         = Column(Float)


# ---------------------------------------------------------------------------
# Howstat-style performance breakdowns
# ---------------------------------------------------------------------------

class PlayerPerformanceByOpponent(Base):
    """Batting + bowling stats broken down by opponent team."""
    __tablename__ = "player_perf_by_opponent"

    player_id       = Column(Integer, ForeignKey("players.id"), primary_key=True)
    opponent_id     = Column(Integer, ForeignKey("teams.id"),   primary_key=True)

    # Batting
    bat_innings     = Column(Integer, default=0)
    bat_not_outs    = Column(Integer, default=0)
    bat_runs        = Column(Integer, default=0)
    bat_balls       = Column(Integer, default=0)
    bat_hs          = Column(Integer, default=0)
    bat_average     = Column(Float)
    bat_sr          = Column(Float)
    bat_fifties     = Column(Integer, default=0)
    bat_hundreds    = Column(Integer, default=0)
    bat_ducks       = Column(Integer, default=0)

    # Bowling
    bowl_innings    = Column(Integer, default=0)
    bowl_balls      = Column(Integer, default=0)
    bowl_runs       = Column(Integer, default=0)
    bowl_wickets    = Column(Integer, default=0)
    bowl_economy    = Column(Float)
    bowl_average    = Column(Float)
    bowl_sr         = Column(Float)


class PlayerPerformanceBySeason(Base):
    """Batting + bowling stats broken down by season/year."""
    __tablename__ = "player_perf_by_season"

    player_id       = Column(Integer, ForeignKey("players.id"), primary_key=True)
    season          = Column(String,  primary_key=True)   # e.g. "2023" or "2023/24"
    tournament      = Column(String,  primary_key=True, default="ALL")

    bat_innings     = Column(Integer, default=0)
    bat_not_outs    = Column(Integer, default=0)
    bat_runs        = Column(Integer, default=0)
    bat_balls       = Column(Integer, default=0)
    bat_hs          = Column(Integer, default=0)
    bat_average     = Column(Float)
    bat_sr          = Column(Float)
    bat_fifties     = Column(Integer, default=0)
    bat_hundreds    = Column(Integer, default=0)
    bat_ducks       = Column(Integer, default=0)

    bowl_innings    = Column(Integer, default=0)
    bowl_balls      = Column(Integer, default=0)
    bowl_runs       = Column(Integer, default=0)
    bowl_wickets    = Column(Integer, default=0)
    bowl_economy    = Column(Float)
    bowl_average    = Column(Float)


class PlayerPerformanceByTeam(Base):
    """Stats for when the player represented a specific team."""
    __tablename__ = "player_perf_by_team"

    player_id       = Column(Integer, ForeignKey("players.id"), primary_key=True)
    team_id         = Column(Integer, ForeignKey("teams.id"),   primary_key=True)

    bat_innings     = Column(Integer, default=0)
    bat_not_outs    = Column(Integer, default=0)
    bat_runs        = Column(Integer, default=0)
    bat_balls       = Column(Integer, default=0)
    bat_hs          = Column(Integer, default=0)
    bat_average     = Column(Float)
    bat_sr          = Column(Float)
    bat_fifties     = Column(Integer, default=0)
    bat_hundreds    = Column(Integer, default=0)
    bat_ducks       = Column(Integer, default=0)

    bowl_innings    = Column(Integer, default=0)
    bowl_balls      = Column(Integer, default=0)
    bowl_runs       = Column(Integer, default=0)
    bowl_wickets    = Column(Integer, default=0)
    bowl_economy    = Column(Float)
    bowl_average    = Column(Float)


class PlayerPerformanceByResult(Base):
    """Batting/bowling stats split by match result (won / lost / no result)."""
    __tablename__ = "player_perf_by_result"

    player_id       = Column(Integer, ForeignKey("players.id"), primary_key=True)
    result          = Column(String,  primary_key=True)   # "won" | "lost" | "no_result"

    bat_innings     = Column(Integer, default=0)
    bat_not_outs    = Column(Integer, default=0)
    bat_runs        = Column(Integer, default=0)
    bat_balls       = Column(Integer, default=0)
    bat_hs          = Column(Integer, default=0)
    bat_average     = Column(Float)
    bat_sr          = Column(Float)
    bat_fifties     = Column(Integer, default=0)
    bat_ducks       = Column(Integer, default=0)

    bowl_innings    = Column(Integer, default=0)
    bowl_balls      = Column(Integer, default=0)
    bowl_runs       = Column(Integer, default=0)
    bowl_wickets    = Column(Integer, default=0)
    bowl_economy    = Column(Float)


class PlayerDismissalAnalysis(Base):
    """How the player was dismissed — batting mode of dismissal breakdown."""
    __tablename__ = "player_dismissal_analysis"

    player_id       = Column(Integer, ForeignKey("players.id"), primary_key=True)
    dismissal_kind  = Column(String,  primary_key=True)
    # e.g.: caught, bowled, lbw, run out, stumped, hit wicket, obstructing, retired

    count           = Column(Integer, default=0)
    pct             = Column(Float)    # % of dismissals


class PlayerBowlingDismissalAnalysis(Base):
    """How the bowler took wickets — mode of dismissal breakdown."""
    __tablename__ = "player_bowling_dismissal_analysis"

    player_id       = Column(Integer, ForeignKey("players.id"), primary_key=True)
    dismissal_kind  = Column(String,  primary_key=True)

    count           = Column(Integer, default=0)
    pct             = Column(Float)


class PlayerMilestone(Base):
    """
    Individual milestone innings/spells — every 50, 100, and 4-wkt haul.
    One row per event so you can list them all (like howstat's milestone page).
    """
    __tablename__ = "player_milestones"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player_id       = Column(Integer, ForeignKey("players.id"), nullable=False)
    match_id        = Column(Integer, ForeignKey("matches.id"))
    innings_id      = Column(Integer, ForeignKey("innings.id"))

    milestone_type  = Column(String, nullable=False)
    # "fifty" | "hundred" | "thirty" | "duck" | "four_wicket_haul" | "five_wicket_haul"
    # | "player_of_match"

    value           = Column(Integer)   # runs scored / wickets taken
    opposition_id   = Column(Integer, ForeignKey("teams.id"))
    venue_id        = Column(Integer, ForeignKey("venues.id"))
    match_date      = Column(Date)
    tournament      = Column(String)

    __table_args__ = (
        Index("ix_milestone_player", "player_id", "milestone_type"),
    )


class PlayerFieldingStats(Base):
    """Fielding statistics per player (catches, run-outs assisted, stumpings)."""
    __tablename__ = "player_fielding_stats"

    player_id           = Column(Integer, ForeignKey("players.id"), primary_key=True)
    tournament          = Column(String,  primary_key=True, default="ALL")

    catches             = Column(Integer, default=0)
    run_outs_direct     = Column(Integer, default=0)
    run_outs_assisted   = Column(Integer, default=0)
    stumpings           = Column(Integer, default=0)
    most_catches_inn    = Column(Integer, default=0)   # max in a single innings


class PlayerOfMatchAward(Base):
    """One row per Player of the Match award."""
    __tablename__ = "player_of_match_awards"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player_id       = Column(Integer, ForeignKey("players.id"), nullable=False)
    match_id        = Column(Integer, ForeignKey("matches.id"))
    match_date      = Column(Date)
    tournament      = Column(String)
    opposition_id   = Column(Integer, ForeignKey("teams.id"))
    venue_id        = Column(Integer, ForeignKey("venues.id"))

    __table_args__ = (
        Index("ix_pom_player", "player_id"),
    )


class VenueDifficulty(Base):
    """
    Computed pitch/venue difficulty factor.
    venue_bat_factor > 1 means batting is easier here than average.
    """
    __tablename__ = "venue_difficulty"

    venue_id        = Column(Integer, ForeignKey("venues.id"), primary_key=True)

    total_matches       = Column(Integer)
    avg_first_inn_runs  = Column(Float)
    avg_second_inn_runs = Column(Float)
    avg_total_runs      = Column(Float)
    avg_wickets_per_inn = Column(Float)

    # shrinkage-adjusted factors
    bat_factor          = Column(Float, default=1.0)   # vs global average
    bowl_factor         = Column(Float, default=1.0)
    pace_index          = Column(Float)    # fraction of wickets to pace
    spin_index          = Column(Float)
    boundary_rate       = Column(Float)    # boundaries per 6 balls

    # Bayesian credible interval
    bat_factor_lo       = Column(Float)
    bat_factor_hi       = Column(Float)


# ---------------------------------------------------------------------------
# Player rating output table
# ---------------------------------------------------------------------------

class PlayerRating(Base):
    """
    Final composite ratings, updated after each analytics run.
    All ratings are normalised to a 0-100 scale.
    """
    __tablename__ = "player_ratings"

    player_id       = Column(Integer, ForeignKey("players.id"), primary_key=True)
    tournament      = Column(String,  primary_key=True)

    # Overall
    bat_rating      = Column(Float)
    bowl_rating     = Column(Float)
    overall_rating  = Column(Float)

    # Specialisation scores
    opener_score    = Column(Float)
    finisher_score  = Column(Float)
    anchor_score    = Column(Float)
    pp_bat_score    = Column(Float)
    death_bat_score = Column(Float)
    chase_score     = Column(Float)

    pp_bowl_score   = Column(Float)
    mid_bowl_score  = Column(Float)
    death_bowl_score = Column(Float)

    # Venue-adjusted
    adj_bat_rating  = Column(Float)
    adj_bowl_rating = Column(Float)

    updated_at      = Column(Date)


def get_engine(db_path=None):
    from config import DB_PATH
    path = db_path or DB_PATH
    return create_engine(f"sqlite:///{path}", echo=False)


def init_db(db_path=None):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine
