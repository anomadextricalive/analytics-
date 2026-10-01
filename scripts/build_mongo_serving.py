"""
Publish ready-to-render documents to a separate MongoDB database for the web app.

SQLite stays the source of truth; this script only reads it. Each collection is written to
`<name>__staging`, its count is checked, then the live collection is renamed to `<name>__prev`
(replacing any older __prev) and the staging one takes its name. Nothing else is dropped.

Collections:
  players        one doc per player: bio, aliases, career per tournament, ratings, fielding, innings log
  matches        one doc per match: teams, venue, toss, result, both scorecards
  venues         one doc per venue: metadata + difficulty factors
  tournaments    one doc per competition
  leaderboards   top batters / bowlers / fielders per tournament (and ALL core)
  innings_balls  one doc per innings with every delivery (compact arrays)

Usage:
  python scripts/build_mongo_serving.py --dry-run          # build + report sizes only
  python scripts/build_mongo_serving.py --db cricket_serving
URI is read from ~/.cricket_mongo_uri.
"""
import argparse
import datetime
import math
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH  # noqa: E402


def clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, datetime.date) and not isinstance(v, datetime.datetime):
        return datetime.datetime(v.year, v.month, v.day)
    return v


def rows(con, sql, *args):
    cur = con.execute(sql, args)
    cols = [c[0] for c in cur.description]
    for r in cur:
        yield {c: clean(v) for c, v in zip(cols, r) if v is not None}


def build(con):
    t0 = time.time()
    team = dict(con.execute("SELECT id, name FROM teams"))
    venue = {r["id"]: r for r in rows(con, "SELECT * FROM venues")}
    pname = dict(con.execute("SELECT id, cricsheet_key FROM players"))
    tour_name = dict(con.execute("SELECT code, display_name FROM tournaments"))

    # ---------- per-player pieces ----------
    aliases = defaultdict(list)
    for pid, a in con.execute("SELECT player_id, alias FROM player_aliases"):
        aliases[pid].append(a)
    by_pid = lambda sql: _group(rows(con, sql), "player_id")
    bat = by_pid("SELECT * FROM player_career_bat")
    bowl = by_pid("SELECT * FROM player_career_bowl")
    rat = by_pid("SELECT * FROM player_ratings")
    fld = by_pid("SELECT * FROM player_fielding_stats")
    pom = defaultdict(int)
    for pid, n in con.execute("SELECT player_id, count(*) FROM player_of_match_awards GROUP BY 1"):
        pom[pid] = n

    bat_log = defaultdict(list)
    for r in rows(con, """
            SELECT pi.batter_id pid, m.id match_id, m.match_date date, m.tournament, m.season,
                   pi.batting_position pos, pi.runs, pi.balls_faced balls, pi.fours, pi.sixes, pi.not_out,
                   pi.dismissal_kind how_out, pi.is_chase, i.batting_team_id team_id, i.bowling_team_id opp_id, m.venue_id
            FROM player_innings pi JOIN matches m ON m.id = pi.match_id JOIN innings i ON i.id = pi.innings_id
            WHERE pi.balls_faced > 0 ORDER BY m.match_date"""):
        pid = r.pop("pid")
        r["team"] = team.get(r.pop("team_id")); r["opp"] = team.get(r.pop("opp_id"))
        r["venue"] = venue.get(r.pop("venue_id"), {}).get("name")
        bat_log[pid].append(r)
    bowl_log = defaultdict(list)
    for r in rows(con, """
            SELECT pb.bowler_id pid, m.id match_id, m.match_date date, m.tournament, m.season,
                   pb.balls_bowled balls, pb.runs_conceded runs, pb.wickets, pb.dot_balls dots, pb.wides, pb.no_balls,
                   i.bowling_team_id team_id, i.batting_team_id opp_id, m.venue_id
            FROM player_bowling_innings pb JOIN matches m ON m.id = pb.match_id JOIN innings i ON i.id = pb.innings_id
            WHERE pb.balls_bowled > 0 ORDER BY m.match_date"""):
        pid = r.pop("pid")
        r["team"] = team.get(r.pop("team_id")); r["opp"] = team.get(r.pop("opp_id"))
        r["venue"] = venue.get(r.pop("venue_id"), {}).get("name")
        bowl_log[pid].append(r)

    players = []
    for p in rows(con, "SELECT * FROM players"):
        pid = p["id"]
        if pid not in bat_log and pid not in bowl_log:
            continue
        doc = {"_id": pid, "key": p.get("cricsheet_key"), "name": p.get("full_name") or p.get("cricsheet_key"),
               "full_name": p.get("full_name"), "country": p.get("country"), "role": p.get("player_role"),
               "batting_style": p.get("batting_style"), "bowling_style": p.get("bowling_style"),
               "dob": clean(datetime.date.fromisoformat(p["date_of_birth"])) if p.get("date_of_birth") else None,
               "aliases": sorted(set(aliases.get(pid, []) + [p.get("cricsheet_key")])),
               "tournaments": sorted({x["tournament"] for x in bat_log.get(pid, []) + bowl_log.get(pid, [])}),
               "player_of_match": pom.get(pid, 0),
               "career_bat": _by_tour(bat.get(pid)), "career_bowl": _by_tour(bowl.get(pid)),
               "ratings": _by_tour(rat.get(pid)), "fielding": _by_tour(fld.get(pid)),
               "batting_innings": bat_log.get(pid, []), "bowling_innings": bowl_log.get(pid, [])}
        players.append({k: v for k, v in doc.items() if v not in (None, [], {})})

    # ---------- matches with scorecards ----------
    bat_by_inn, bowl_by_inn = defaultdict(list), defaultdict(list)
    for r in rows(con, "SELECT innings_id, batter_id, batting_position pos, runs, balls_faced balls, fours, sixes, not_out, dismissal_kind how_out FROM player_innings ORDER BY batting_position"):
        r["player"] = pname.get(r["batter_id"]); bat_by_inn[r.pop("innings_id")].append(r)
    for r in rows(con, "SELECT innings_id, bowler_id, balls_bowled balls, runs_conceded runs, wickets, maidens, dot_balls dots, wides, no_balls FROM player_bowling_innings"):
        r["player"] = pname.get(r["bowler_id"]); bowl_by_inn[r.pop("innings_id")].append(r)
    inns = defaultdict(list)
    for r in rows(con, "SELECT * FROM innings ORDER BY innings_number"):
        r["batting_team"] = team.get(r.pop("batting_team_id")); r["bowling_team"] = team.get(r.pop("bowling_team_id"))
        r["batting"] = bat_by_inn.get(r["id"], []); r["bowling"] = bowl_by_inn.get(r["id"], [])
        inns[r.pop("match_id")].append(r)
    pom_by_match = defaultdict(list)
    for mid, pid in con.execute("SELECT match_id, player_id FROM player_of_match_awards"):
        pom_by_match[mid].append(pname.get(pid))
    matches = []
    for m in rows(con, "SELECT id, cricsheet_id, match_date, tournament, season, match_type, venue_id, team1_id, team2_id, toss_winner_id, toss_decision, winner_id, win_by_runs, win_by_wickets, no_result FROM matches"):
        doc = {"_id": m["id"], "source": "cricbuzz" if str(m.get("cricsheet_id", "")).startswith("cb") else "cricsheet",
               "source_id": m.get("cricsheet_id"), "date": clean(datetime.date.fromisoformat(m["match_date"])),
               "tournament": m["tournament"], "tournament_name": tour_name.get(m["tournament"]), "season": m.get("season"),
               "format": m.get("match_type"), "venue": venue.get(m.get("venue_id"), {}).get("name"), "venue_id": m.get("venue_id"),
               "teams": [team.get(m.get("team1_id")), team.get(m.get("team2_id"))],
               "toss": {"winner": team.get(m.get("toss_winner_id")), "decision": m.get("toss_decision")},
               "winner": team.get(m.get("winner_id")), "by_runs": m.get("win_by_runs"), "by_wickets": m.get("win_by_wickets"),
               "no_result": bool(m.get("no_result")), "player_of_match": pom_by_match.get(m["id"], []),
               "innings": inns.get(m["id"], [])}
        matches.append({k: v for k, v in doc.items() if v not in (None, [], {})})

    # ---------- venues, tournaments, leaderboards ----------
    vdiff = {r["venue_id"]: r for r in rows(con, "SELECT * FROM venue_difficulty")}
    venues = []
    for vid, v in venue.items():
        d = dict(v); d["_id"] = d.pop("id"); d["difficulty"] = {k: x for k, x in vdiff.get(vid, {}).items() if k != "venue_id"}
        venues.append({k: x for k, x in d.items() if x not in (None, {}, [])})
    tournaments = [{"_id": r.pop("code"), **r} for r in rows(con, "SELECT * FROM tournaments")]

    lb = []
    for tour in ["ALL"] + [t["_id"] for t in tournaments]:
        tb = list(rows(con, "SELECT player_id, runs, innings, average, strike_rate, hs FROM player_career_bat WHERE tournament=? ORDER BY runs DESC LIMIT 50", tour))
        tw = list(rows(con, "SELECT player_id, wickets, innings, economy, average, strike_rate FROM player_career_bowl WHERE tournament=? ORDER BY wickets DESC LIMIT 50", tour))
        tf = list(rows(con, "SELECT player_id, catches, stumpings, run_outs_direct, run_outs_assisted FROM player_fielding_stats WHERE tournament=? ORDER BY catches + stumpings DESC LIMIT 50", tour))
        for lst in (tb, tw, tf):
            for r in lst: r["player"] = pname.get(r["player_id"])
        lb.append({"_id": tour, "most_runs": tb, "most_wickets": tw, "best_fielders": tf})

    # ---------- ball-by-ball, one doc per innings ----------
    def innings_balls():
        cols = ["over_number", "ball_in_over", "batter_id", "non_striker_id", "bowler_id", "bat_runs", "extras",
                "wide", "no_ball", "is_wicket", "wicket_kind", "player_out_id", "fielder_id"]
        cur = con.execute("""SELECT innings_id, over_number, ball_in_over, batter_id, non_striker_id, bowler_id, bat_runs,
                                    extras, wide, no_ball, is_wicket, wicket_kind, player_out_id, fielder_id
                             FROM deliveries ORDER BY innings_id, ball_number""")
        cur_id, balls = None, []
        for r in cur:
            if r[0] != cur_id:
                if cur_id is not None:
                    yield {"_id": cur_id, "cols": cols, "balls": balls}
                cur_id, balls = r[0], []
            balls.append(list(r[1:]))
        if cur_id is not None:
            yield {"_id": cur_id, "cols": cols, "balls": balls}

    print(f"built documents in {time.time() - t0:.0f}s", flush=True)
    return {"players": players, "matches": matches, "venues": venues, "tournaments": tournaments,
            "leaderboards": lb}, innings_balls


def _group(it, key):
    out = defaultdict(list)
    for r in it:
        out[r.pop(key)].append(r)
    return out


def _by_tour(lst):
    return {r.pop("tournament"): r for r in (lst or [])} or None


def publish(mdb, name, docs_iter, batch=1000):
    stg = f"{name}__staging"
    mdb[stg].drop()
    n, buf = 0, []
    for d in docs_iter:
        buf.append(d)
        if len(buf) >= batch:
            mdb[stg].insert_many(buf, ordered=False); n += len(buf); buf = []
    if buf:
        mdb[stg].insert_many(buf, ordered=False); n += len(buf)
    got = mdb[stg].count_documents({})
    if got != n:
        raise SystemExit(f"{name}: staging has {got}, expected {n}; live collection untouched")
    names = set(mdb.list_collection_names())
    if name in names:
        mdb[name].rename(f"{name}__prev", dropTarget=True)   # keep the previous version
    mdb[stg].rename(name)
    print(f"  {name}: {n} docs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="cricket_serving")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cols, balls_gen = build(con)

    import bson
    for name, docs in cols.items():
        sizes = [len(bson.encode(d)) for d in docs]
        print(f"  {name:13s} {len(docs):6d} docs  {sum(sizes)/1e6:7.1f} MB  largest {max(sizes)/1e3:7.1f} KB", flush=True)
    if a.dry_run:
        n = tot = big = 0
        for d in balls_gen():
            s = len(bson.encode(d)); n += 1; tot += s; big = max(big, s)
        print(f"  innings_balls {n:6d} docs  {tot/1e6:7.1f} MB  largest {big/1e3:7.1f} KB")
        return

    import certifi
    from pymongo import MongoClient
    uri = open(Path.home() / ".cricket_mongo_uri").read().strip()
    mdb = MongoClient(uri, serverSelectionTimeoutMS=15000, tlsCAFile=certifi.where())[a.db]
    for name, docs in cols.items():
        publish(mdb, name, iter(docs), batch=200 if name == "players" else 1000)
    publish(mdb, "innings_balls", balls_gen(), batch=500)
    mdb["players"].create_index("key")
    mdb["players"].create_index("tournaments")
    mdb["matches"].create_index([("tournament", 1), ("date", -1)])
    mdb["matches"].create_index("teams")
    print("published to", a.db)


if __name__ == "__main__":
    main()
