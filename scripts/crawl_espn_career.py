"""ESPN Cricinfo career snapshots keyed by ESPN id (the player-page headline numbers).

One row per (espn_id, fmt, stat_type) in `espn_career`:
  status 'ok'    -> ESPN returned a career line (mat > 0)
         'none'  -> ESPN answered, player has no matches in that format/discipline
         'error' -> call failed after retries; retried on next run (never counted as 'none')
Resumable. Highest-volume DB players first. Existing player_career_intl t20i rows are copied (same ESPN id)
instead of re-fetched. Usage: python scripts/crawl_espn_career.py [db] [--formats t20,t20i] [--limit N] [--workers 3]
"""
import datetime, json, sqlite3, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from cricdata import CricinfoClient

args = [a for a in sys.argv[1:] if not a.startswith("--")]
def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
DB = args[0] if args else "data/cricket.db"
FMTS = [f for f in opt("--formats", "t20,t20i").split(",") if f]
LIMIT = int(opt("--limit", 0)); WORKERS = int(opt("--workers", 3))
ci = CricinfoClient(timeout=60); lock = threading.Lock()

con = sqlite3.connect(DB, check_same_thread=False, timeout=60); cur = con.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS espn_career (
    espn_id TEXT NOT NULL, fmt TEXT NOT NULL, stat_type TEXT NOT NULL, status TEXT NOT NULL,
    span TEXT, mat INTEGER, inns INTEGER, runs INTEGER, bf INTEGER, hs TEXT, ave REAL, sr REAL,
    hundreds INTEGER, fifties INTEGER, fours INTEGER, sixes INTEGER,
    overs REAL, mdns INTEGER, wkts INTEGER, econ REAL, bbi TEXT, four_w INTEGER, five_w INTEGER,
    balls INTEGER, nos INTEGER, ducks INTEGER,
    fetched_at TEXT, source TEXT, raw_json TEXT, PRIMARY KEY (espn_id, fmt, stat_type))""")

def num(v, cast=int):
    try: return cast(str(v).replace(",", ""))
    except Exception: return None

def overs_to_balls(o):
    try:
        w, _, b = str(o).partition("."); return int(w) * 6 + int(b or 0)
    except Exception: return None

def parse(stat, s):
    if stat == "batting":
        return dict(span=s.get("Span"), mat=num(s.get("Mat")), inns=num(s.get("Inns")), runs=num(s.get("Runs")), bf=num(s.get("BF")),
                    hs=s.get("HS"), ave=num(s.get("Ave"), float), sr=num(s.get("SR"), float), hundreds=num(s.get("100")), nos=num(s.get("NO")), ducks=num(s.get("0")),
                    fifties=num(s.get("50")), fours=num(s.get("4s")), sixes=num(s.get("6s")))
    return dict(span=s.get("Span"), mat=num(s.get("Mat")), inns=num(s.get("Inns")), runs=num(s.get("Runs")), overs=num(s.get("Overs"), float),
                mdns=num(s.get("Mdns")), wkts=num(s.get("Wkts")), ave=num(s.get("Ave"), float), sr=num(s.get("SR"), float),
                econ=num(s.get("Econ"), float), bbi=s.get("BBI"), four_w=num(s.get("4")), five_w=num(s.get("5")),
                balls=num(s.get("Balls")) or overs_to_balls(s.get("Overs")))

# copy existing t20i rows (verified by the same ESPN id)
if "t20i" in FMTS:
    cur.execute("""INSERT OR IGNORE INTO espn_career (espn_id, fmt, stat_type, status, span, mat, inns, runs, bf, hs, ave, sr, hundreds, fifties, fours, sixes,
        overs, mdns, wkts, econ, bbi, four_w, five_w, fetched_at, source, raw_json)
        SELECT CAST(c.espn_id AS TEXT), c.fmt, c.stat_type, 'ok', c.span, c.mat, c.inns, c.runs, c.bf, c.hs, c.ave, c.sr, c.hundreds, c.fifties, c.fours, c.sixes,
        c.overs, c.mdns, c.wkts, c.econ, c.bbi, c.four_w, c.five_w, NULL, 'copied from player_career_intl', c.raw_json
        FROM player_career_intl c JOIN player_espn_map m ON m.player_id=c.player_id AND CAST(m.espn_id AS TEXT)=CAST(c.espn_id AS TEXT)
        WHERE c.fmt='t20i' AND m.status='matched'""")
    con.commit()

targets = cur.execute("""SELECT CAST(m.espn_id AS TEXT), COALESCE((SELECT COUNT(*) FROM player_innings pi WHERE pi.batter_id=m.player_id),0)
    + COALESCE((SELECT COUNT(*) FROM player_bowling_innings pb WHERE pb.bowler_id=m.player_id),0) AS n
    FROM player_espn_map m WHERE m.status='matched' AND m.espn_id IS NOT NULL ORDER BY n DESC""").fetchall()
have = {(r[0], r[1], r[2]) for r in cur.execute("SELECT espn_id, fmt, stat_type FROM espn_career WHERE status IN ('ok','none')")}
jobs = [(eid, f, st) for eid, _ in targets for f in FMTS for st in ("batting", "bowling") if (eid, f, st) not in have]
if LIMIT: jobs = jobs[:LIMIT]
print(len(targets), "players;", len(jobs), "calls to make", flush=True)

def work(job):
    eid, f, st = job
    for a in range(12):
        try:
            res = ci.player_career_stats(eid, fmt=f, stat_type=st); break
        except Exception: res = None; time.sleep(0.3 + 0.1 * a)
    time.sleep(0.1)
    return job, res

done = 0
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for (eid, f, st), res in pool.map(work, jobs):
        now = datetime.datetime.now().isoformat(timespec="seconds")
        src = f"stats.espncricinfo.com player {eid} class {f} {st}"
        if not isinstance(res, dict):
            cur.execute("INSERT OR REPLACE INTO espn_career (espn_id,fmt,stat_type,status,fetched_at,source) VALUES (?,?,?,?,?,?)", (eid, f, st, "error", now, src))
        else:
            s = res.get("summary") or {}; p = parse(st, s)
            status = "ok" if (p.get("mat") or 0) > 0 else "none"
            cols = ["espn_id", "fmt", "stat_type", "status", "fetched_at", "source", "raw_json"] + [k for k, v in p.items()]
            vals = [eid, f, st, status, now, src, json.dumps(s)] + [v for v in p.values()]
            cur.execute(f"INSERT OR REPLACE INTO espn_career ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})", vals)
        con.commit(); done += 1
        if done % 100 == 0: print(done, "/", len(jobs), flush=True)
con.commit()
print(cur.execute("SELECT fmt, stat_type, status, COUNT(*) FROM espn_career GROUP BY 1,2,3").fetchall())
