"""
Fast cricsheet JSON parser — bulk inserts, in-memory caches, WAL mode.

Speed improvements over naive ORM approach:
  - Player/team/venue resolved from in-memory dicts (zero DB round trips per delivery)
  - Deliveries, PlayerInnings, PlayerBowlingInnings collected into lists and
    bulk-inserted with session.bulk_insert_mappings()
  - SQLite WAL + synchronous=OFF + cache_size=64MB pragmas
  - Commit every 500 matches instead of per-match
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session
from tqdm import tqdm
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import YEAR_FROM, YEAR_TO, PHASES
from src.db.schema import (
    Player, Venue, Team, Match, Innings,
    Delivery, PlayerInnings, PlayerBowlingInnings, Partnership,
    PlayerMilestone, PlayerOfMatchAward,
)

console = Console()

BATCH = 500   # commit every N matches


# ---------------------------------------------------------------------------
# Phase helper
# ---------------------------------------------------------------------------

def _phase(over_1: int) -> int:
    pp_lo, pp_hi   = PHASES["powerplay"]
    mid_lo, mid_hi = PHASES["middle"]
    if pp_lo <= over_1 <= pp_hi:
        return 0
    if mid_lo <= over_1 <= mid_hi:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Global in-memory registry (rebuilt once per ingest run)
# ---------------------------------------------------------------------------

class Registry:
    """In-memory id caches so we never query the DB for lookups."""

    def __init__(self, session: Session):
        self.session = session
        self.players: dict[str, int] = {}   # cricsheet_key → id
        self.teams:   dict[str, int] = {}   # name → id
        self.venues:  dict[str, int] = {}   # name → id
        self._load()

    def _load(self):
        for p in self.session.query(Player).all():
            self.players[p.cricsheet_key] = p.id
        for t in self.session.query(Team).all():
            self.teams[t.name] = t.id
        for v in self.session.query(Venue).all():
            self.venues[v.name] = v.id

    # -- players --
    def player_id(self, name: str) -> int:
        if name not in self.players:
            p = Player(cricsheet_key=name)
            self.session.add(p)
            self.session.flush()
            self.players[name] = p.id
        return self.players[name]

    # -- teams --
    def team_id(self, name: str) -> int:
        if name not in self.teams:
            t = Team(name=name)
            self.session.add(t)
            self.session.flush()
            self.teams[name] = t.id
        return self.teams[name]

    # -- venues --
    def venue_id(self, name: str, city: str | None = None) -> int:
        if name not in self.venues:
            v = Venue(name=name, city=city)
            self.session.add(v)
            self.session.flush()
            self.venues[name] = v.id
        return self.venues[name]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class MatchParser:
    def __init__(self, session: Session, registry: Registry):
        self.s   = session
        self.reg = registry

        # Bulk buffers flushed every BATCH matches
        self._buf_deliveries:   list[dict] = []
        self._buf_bat_innings:  list[dict] = []
        self._buf_bowl_innings: list[dict] = []
        self._buf_milestones:   list[dict] = []
        self._buf_pom:          list[dict] = []

    # ------------------------------------------------------------------
    def flush_buffers(self):
        s = self.s
        if self._buf_deliveries:
            s.bulk_insert_mappings(Delivery,         self._buf_deliveries);   self._buf_deliveries.clear()
        if self._buf_bat_innings:
            s.bulk_insert_mappings(PlayerInnings,    self._buf_bat_innings);  self._buf_bat_innings.clear()
        if self._buf_bowl_innings:
            s.bulk_insert_mappings(PlayerBowlingInnings, self._buf_bowl_innings); self._buf_bowl_innings.clear()
        if self._buf_milestones:
            s.bulk_insert_mappings(PlayerMilestone,  self._buf_milestones);   self._buf_milestones.clear()
        if self._buf_pom:
            s.bulk_insert_mappings(PlayerOfMatchAward, self._buf_pom);        self._buf_pom.clear()

    # ------------------------------------------------------------------
    def parse_file(self, path: Path, tournament: str) -> bool:
        try:
            raw = json.loads(path.read_bytes())
        except Exception:
            return False

        info = raw.get("info", {})
        dates = info.get("dates", [])
        if not dates:
            return False
        match_date = datetime.strptime(dates[0], "%Y-%m-%d").date()
        if not (YEAR_FROM <= match_date.year <= YEAR_TO):
            return False

        existing = self.s.query(Match.id).filter_by(cricsheet_id=path.stem).first()
        if existing:
            return False

        reg = self.reg

        # ---- venue / teams ----
        venue_name  = info.get("venue", "Unknown")
        venue_city  = info.get("city")
        vid = reg.venue_id(venue_name, venue_city)

        teams_raw = info.get("teams", [])
        t1id = reg.team_id(teams_raw[0]) if len(teams_raw) > 0 else None
        t2id = reg.team_id(teams_raw[1]) if len(teams_raw) > 1 else None

        toss      = info.get("toss", {})
        toss_name = toss.get("winner")
        toss_dec  = toss.get("decision", "")
        toss_tid  = reg.team_id(toss_name) if toss_name else None

        outcome      = info.get("outcome", {})
        winner_name  = outcome.get("winner")
        winner_tid   = reg.team_id(winner_name) if winner_name else None

        by = outcome.get("by", {})
        win_runs = by.get("runs")
        win_wkts = by.get("wickets")

        if toss_name and toss_dec == "bat":
            batting_first = toss_name
        elif toss_name and toss_dec == "field":
            batting_first = (teams_raw[1] if teams_raw and teams_raw[0] == toss_name
                             else (teams_raw[0] if teams_raw else None))
        else:
            batting_first = teams_raw[0] if teams_raw else None

        chasing_won = None
        if winner_name and batting_first:
            chasing_won = (winner_name != batting_first)

        match = Match(
            cricsheet_id     = path.stem,
            match_date       = match_date,
            tournament       = tournament,
            gender           = info.get("gender", "male"),
            match_type       = info.get("match_type", "T20"),
            season           = str(info.get("season", match_date.year)),
            venue_id         = vid,
            team1_id         = t1id,
            team2_id         = t2id,
            toss_winner_id   = toss_tid,
            toss_decision    = toss_dec,
            winner_id        = winner_tid,
            win_by_runs      = win_runs,
            win_by_wickets   = win_wkts,
            no_result        = "no result" in str(outcome).lower(),
            chasing_team_won = chasing_won,
            raw_meta         = json.dumps(info, default=str),
        )
        self.s.add(match)
        self.s.flush()

        # Player of Match
        for pom_name in info.get("player_of_match", []):
            self._buf_pom.append({
                "player_id":   reg.player_id(pom_name),
                "match_id":    match.id,
                "match_date":  match_date,
                "tournament":  tournament,
                "venue_id":    vid,
            })

        self._parse_innings(raw.get("innings", []), match, teams_raw,
                            match_date, tournament, vid)
        return True

    # ------------------------------------------------------------------
    def _parse_innings(self, innings_data, match, teams_raw,
                       match_date, tournament, vid):
        reg = self.reg

        for inn_idx, inn_data in enumerate(innings_data):
            inn_number   = inn_idx + 1
            batting_name = inn_data.get("team")
            bowling_name = next(
                (t for t in teams_raw if t != batting_name), None
            ) if batting_name else None

            bat_tid  = reg.team_id(batting_name)  if batting_name else None
            bowl_tid = reg.team_id(bowling_name)  if bowling_name else None

            target_obj = inn_data.get("target", {})
            target_val = target_obj.get("runs") if isinstance(target_obj, dict) else None
            req_rr_start = (target_val / 20.0) if target_val else None

            innings = Innings(
                match_id        = match.id,
                innings_number  = inn_number,
                batting_team_id = bat_tid,
                bowling_team_id = bowl_tid,
                target          = target_val,
            )
            self.s.add(innings)
            self.s.flush()

            # -- per-batter / per-bowler accumulators (in Python, not DB) --
            bat_acc:  dict[str, dict] = {}
            bowl_acc: dict[str, dict] = {}

            total_runs = total_wickets = total_balls = extras_total = 0
            ball_seq = 0

            for over_obj in inn_data.get("overs", []):
                over_0 = over_obj["over"]
                over_1 = over_0 + 1
                ph = _phase(over_1)
                legal_in_over = 0

                for d in over_obj.get("deliveries", []):
                    ball_seq += 1

                    batter_name   = d.get("batter", "")
                    non_s_name    = d.get("non_striker", "")
                    bowler_name   = d.get("bowler", "")

                    runs_obj  = d.get("runs", {})
                    bat_r     = runs_obj.get("batter", 0)
                    ext_r     = runs_obj.get("extras", 0)
                    total_r   = runs_obj.get("total", bat_r + ext_r)

                    extras_obj = d.get("extras", {})
                    wide_r  = extras_obj.get("wides",   0)
                    nb_r    = extras_obj.get("noballs", 0)
                    bye_r   = extras_obj.get("byes",    0)
                    lb_r    = extras_obj.get("legbyes", 0)
                    pen_r   = extras_obj.get("penalty", 0)

                    is_legal = (wide_r == 0 and nb_r == 0)
                    if is_legal:
                        legal_in_over += 1
                        total_balls   += 1

                    is_4   = (bat_r == 4)
                    is_6   = (bat_r == 6)
                    is_dot = is_legal and bat_r == 0 and ext_r == 0

                    wickets   = d.get("wickets", [])
                    is_wicket = len(wickets) > 0
                    player_out = wickets[0].get("player_out") if is_wicket else None
                    wkt_kind   = wickets[0].get("kind")       if is_wicket else None
                    bowler_wkt = is_wicket and wkt_kind not in (
                        "run out", "obstructing the field", "retired hurt",
                        "retired out", "timed out", "handled the ball",
                    )

                    total_runs    += total_r
                    extras_total  += ext_r
                    if is_wicket:
                        total_wickets += 1

                    if inn_number == 2 and target_val:
                        balls_rem = max(1, 120 - total_balls)
                        runs_need = max(0, target_val - total_runs)
                        req_r = round((runs_need / balls_rem) * 6, 2)
                        crr   = round((total_runs / max(1, total_balls)) * 6, 2)
                    else:
                        req_r = crr = None

                    # Resolve IDs from cache (zero DB hits)
                    batter_id   = reg.player_id(batter_name)
                    non_s_id    = reg.player_id(non_s_name)
                    bowler_id   = reg.player_id(bowler_name)
                    out_pid     = reg.player_id(player_out) if player_out else None

                    self._buf_deliveries.append({
                        "innings_id":       innings.id,
                        "over_number":      over_1,
                        "ball_in_over":     legal_in_over if is_legal else 0,
                        "ball_number":      ball_seq,
                        "phase":            ph,
                        "batter_id":        batter_id,
                        "non_striker_id":   non_s_id,
                        "bowler_id":        bowler_id,
                        "bat_runs":         bat_r,
                        "extras":           ext_r,
                        "total_runs":       total_r,
                        "is_boundary_4":    is_4,
                        "is_boundary_6":    is_6,
                        "is_dot":           is_dot,
                        "wide":             wide_r,
                        "no_ball":          nb_r,
                        "bye":              bye_r,
                        "leg_bye":          lb_r,
                        "penalty":          pen_r,
                        "is_wicket":        is_wicket,
                        "wicket_kind":      wkt_kind,
                        "player_out_id":    out_pid,
                        "req_rate_at_ball": req_r,
                        "crr_at_ball":      crr,
                    })

                    # -- accumulate batter --
                    if batter_name not in bat_acc:
                        bat_acc[batter_name] = {
                            "pid": batter_id, "runs": 0, "balls": 0,
                            "fours": 0, "sixes": 0, "not_out": True,
                            "dismissal": None, "position": len(bat_acc) + 1,
                            "pp_r": 0, "pp_b": 0, "mid_r": 0, "mid_b": 0,
                            "death_r": 0, "death_b": 0,
                        }
                    ba = bat_acc[batter_name]
                    ba["runs"] += bat_r
                    if is_4: ba["fours"] += 1
                    if is_6: ba["sixes"] += 1
                    if is_legal:
                        ba["balls"] += 1
                        if ph == 0:   ba["pp_r"]    += bat_r; ba["pp_b"]    += 1
                        elif ph == 1: ba["mid_r"]   += bat_r; ba["mid_b"]   += 1
                        else:         ba["death_r"] += bat_r; ba["death_b"] += 1

                    if is_wicket and player_out and player_out in bat_acc:
                        bat_acc[player_out]["not_out"]   = False
                        bat_acc[player_out]["dismissal"] = wkt_kind

                    # -- accumulate bowler --
                    if bowler_name not in bowl_acc:
                        bowl_acc[bowler_name] = {
                            "pid": bowler_id, "balls": 0, "runs": 0,
                            "wkts": 0, "dots": 0, "wides": 0, "nbs": 0,
                            "boundaries": 0,
                            "pp_b": 0, "pp_r": 0, "pp_w": 0,
                            "mid_b": 0, "mid_r": 0, "mid_w": 0,
                            "death_b": 0, "death_r": 0, "death_w": 0,
                        }
                    bo = bowl_acc[bowler_name]
                    if is_legal:
                        bo["balls"] += 1
                        bowl_r = total_r - bye_r - lb_r
                        bo["runs"]  += bowl_r
                        if ph == 0:
                            bo["pp_b"] += 1; bo["pp_r"] += bowl_r
                        elif ph == 1:
                            bo["mid_b"] += 1; bo["mid_r"] += bowl_r
                        else:
                            bo["death_b"] += 1; bo["death_r"] += bowl_r
                    bo["wides"] += wide_r
                    bo["nbs"]   += nb_r
                    if is_dot:        bo["dots"]       += 1
                    if is_4 or is_6:  bo["boundaries"] += 1
                    if bowler_wkt:
                        bo["wkts"] += 1
                        if ph == 0:   bo["pp_w"]    += 1
                        elif ph == 1: bo["mid_w"]   += 1
                        else:         bo["death_w"] += 1

            # Finalise innings row
            innings.total_runs    = total_runs
            innings.total_wickets = total_wickets
            innings.total_balls   = total_balls
            innings.extras        = extras_total
            is_chase = (inn_number == 2)
            if is_chase and target_val:
                innings.chase_successful  = (total_wickets < 10 and
                                             total_runs >= target_val)
                innings.required_rr_start = req_rr_start
            self.s.flush()

            chase_won = innings.chase_successful if is_chase else None

            # Milestones — detect top scorer
            inn_runs_vals = [ba["runs"] for ba in bat_acc.values()]
            top_score     = max(inn_runs_vals) if inn_runs_vals else 0

            # -- Write PlayerInnings (buffered) --
            for name, ba in bat_acc.items():
                r = ba["runs"]
                milestone = None
                if r >= 100:     milestone = "hundred"
                elif r >= 50:    milestone = "fifty"
                elif r >= 30:    milestone = "thirty"
                elif r == 0 and not ba["not_out"]: milestone = "duck"
                if milestone:
                    self._buf_milestones.append({
                        "player_id":      ba["pid"],
                        "match_id":       match.id,
                        "innings_id":     innings.id,
                        "milestone_type": milestone,
                        "value":          r,
                        "venue_id":       vid,
                        "match_date":     match_date,
                        "tournament":     tournament,
                    })

                self._buf_bat_innings.append({
                    "innings_id":        innings.id,
                    "match_id":          match.id,
                    "batter_id":         ba["pid"],
                    "team_id":           bat_tid,
                    "batting_position":  ba["position"],
                    "runs":              ba["runs"],
                    "balls_faced":       ba["balls"],
                    "fours":             ba["fours"],
                    "sixes":             ba["sixes"],
                    "not_out":           ba["not_out"],
                    "dismissal_kind":    ba["dismissal"],
                    "pp_runs":           ba["pp_r"],
                    "pp_balls":          ba["pp_b"],
                    "mid_runs":          ba["mid_r"],
                    "mid_balls":         ba["mid_b"],
                    "death_runs":        ba["death_r"],
                    "death_balls":       ba["death_b"],
                    "is_chase":          is_chase,
                    "chase_won":         chase_won,
                    "required_rr_start": req_rr_start,
                })

            # 4-wicket haul milestones
            for name, bo in bowl_acc.items():
                if bo["wkts"] >= 4:
                    self._buf_milestones.append({
                        "player_id":      bo["pid"],
                        "match_id":       match.id,
                        "innings_id":     innings.id,
                        "milestone_type": "four_wicket_haul" if bo["wkts"] == 4 else "five_wicket_haul",
                        "value":          bo["wkts"],
                        "venue_id":       vid,
                        "match_date":     match_date,
                        "tournament":     tournament,
                    })

                self._buf_bowl_innings.append({
                    "innings_id":     innings.id,
                    "match_id":       match.id,
                    "bowler_id":      bo["pid"],
                    "team_id":        bowl_tid,
                    "balls_bowled":   bo["balls"],
                    "overs_bowled":   round(bo["balls"] / 6, 1),
                    "runs_conceded":  bo["runs"],
                    "wickets":        bo["wkts"],
                    "dot_balls":      bo["dots"],
                    "wides":          bo["wides"],
                    "no_balls":       bo["nbs"],
                    "boundaries_con": bo["boundaries"],
                    "pp_balls":       bo["pp_b"],  "pp_runs":    bo["pp_r"],  "pp_wickets":    bo["pp_w"],
                    "mid_balls":      bo["mid_b"], "mid_runs":   bo["mid_r"], "mid_wickets":   bo["mid_w"],
                    "death_balls":    bo["death_b"],"death_runs": bo["death_r"],"death_wickets": bo["death_w"],
                })


# ---------------------------------------------------------------------------
# Batch ingestion entry point
# ---------------------------------------------------------------------------

def _configure_sqlite(session: Session):
    """Apply SQLite pragmas for maximum write speed."""
    conn = session.connection()
    conn.execute(text("PRAGMA journal_mode=WAL"))
    conn.execute(text("PRAGMA synchronous=NORMAL"))
    conn.execute(text("PRAGMA cache_size=-65536"))   # 64 MB
    conn.execute(text("PRAGMA temp_store=MEMORY"))
    conn.execute(text("PRAGMA mmap_size=268435456")) # 256 MB


def ingest_directory(session: Session, directory: Path,
                     tournament: str, verbose: bool = False) -> dict:
    """Ingest all JSON files in *directory*. Returns counts."""
    _configure_sqlite(session)

    files    = sorted(directory.glob("*.json"))
    inserted = skipped = errors = 0

    registry = Registry(session)
    parser   = MatchParser(session, registry)

    for i, f in enumerate(tqdm(files, desc=tournament, unit="match")):
        try:
            ok = parser.parse_file(f, tournament)
            if ok:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            errors += 1
            if verbose:
                console.print(f"[red]error[/red] {f.name}: {e}")
            session.rollback()
            # rebuild registry after rollback
            registry = Registry(session)
            parser   = MatchParser(session, registry)
            continue

        if inserted % BATCH == 0 and inserted > 0:
            parser.flush_buffers()
            session.commit()

    # Final flush
    parser.flush_buffers()
    session.commit()

    return {"inserted": inserted, "skipped": skipped, "errors": errors}
