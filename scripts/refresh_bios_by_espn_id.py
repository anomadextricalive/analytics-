"""Pull ESPN Cricinfo bios by the (now exact) ESPN id.
  --overwrite-conflicts : replace bio fields for players whose id was corrected (data/espn_register_conflicts.csv,
                          minus data/espn_register_review_keep_old.csv). Old values saved in players_bio__prev.
  default               : fill only NULL fields for matched players with no full_name yet.
Cache: data/espn_bio_cache.jsonl (resumable).  Usage: python scripts/refresh_bios_by_espn_id.py [db] [--overwrite-conflicts] [--limit N]
"""
import csv, datetime, json, os, sqlite3, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
lock = threading.Lock()
from cricdata import CricinfoClient

args = [a for a in sys.argv[1:] if not a.startswith("--")]
DB = args[0] if args else "data/cricket.db"
OVER = "--overwrite-conflicts" in sys.argv
FETCH_ONLY = "--fetch-only" in sys.argv   # only fill the cache (no DB writes) so other writers are never blocked
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
CACHE = "data/espn_bio_cache.jsonl"
cache = {}
if os.path.exists(CACHE):
    for line in open(CACHE):
        d = json.loads(line); cache[d["id"]] = d["bio"]
cf = open(CACHE, "a")
ci = CricinfoClient(timeout=60)
con = sqlite3.connect(DB, timeout=60); cur = con.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS players_bio__prev AS SELECT id, full_name, batting_style, bowling_style, date_of_birth, country, 'snapshot' AS note FROM players WHERE 0""")

def fetch(eid):
    if eid in cache: return cache[eid]
    for a in range(8):
        try:
            b = ci.player_bio(eid); break
        except Exception: time.sleep(0.6 * (a + 1)); b = None
    with lock:
        cache[eid] = b
        cf.write(json.dumps({"id": eid, "bio": b}, default=str) + "\n"); cf.flush()
    time.sleep(0.15)
    return b

def dob(s):
    try: m, d, y = s.split("/"); return datetime.date(int(y), int(m), int(d)).isoformat()
    except Exception: return None

if OVER:
    skip = {r["player_id"] for r in csv.DictReader(open("data/espn_register_review_keep_old.csv"))}
    ids = [int(r["player_id"]) for r in csv.DictReader(open("data/espn_register_conflicts.csv")) if r["player_id"] not in skip]
else:
    ids = [r[0] for r in cur.execute("""SELECT p.id FROM players p JOIN player_espn_map m ON m.player_id=p.id
        WHERE m.status='matched' AND (p.full_name IS NULL OR p.date_of_birth IS NULL)""")]
if LIMIT: ids = ids[:LIMIT]
print(len(ids), "players", "(overwrite)" if OVER else "(fill-only)", flush=True)
done = fail = 0
eids = {pid: str(cur.execute("SELECT espn_id FROM player_espn_map WHERE player_id=?", (pid,)).fetchone()[0]) for pid in ids}
pool = ThreadPoolExecutor(max_workers=3)
futs = {pid: pool.submit(fetch, eids[pid]) for pid in ids}   # fetched in parallel, applied in order
for pid in ids:
    b = futs[pid].result()
    if FETCH_ONLY:
        done += 1
        if done % 100 == 0: print(done, "fetched", flush=True)
        continue
    if not b: fail += 1; continue
    bat = (b.get("batStyle") or [{}])[0].get("description"); bowl = (b.get("bowlStyle") or [{}])[0].get("description")
    vals = {"full_name": b.get("fullName") or b.get("displayName"), "batting_style": bat, "bowling_style": bowl,
            "date_of_birth": dob(b.get("displayDOB") or ""), "country": (b.get("team") or {}).get("displayName")}
    if OVER:
        cur.execute("INSERT INTO players_bio__prev SELECT id, full_name, batting_style, bowling_style, date_of_birth, country, 'before register fix' FROM players WHERE id=?", (pid,))
        cur.execute("UPDATE players SET full_name=?, batting_style=?, bowling_style=?, date_of_birth=?, country=COALESCE(?,country) WHERE id=?",
                    (vals["full_name"], vals["batting_style"], vals["bowling_style"], vals["date_of_birth"], vals["country"], pid))
    else:
        for k, v in vals.items():
            if v: cur.execute(f"UPDATE players SET {k}=? WHERE id=? AND {k} IS NULL", (v, pid))
    done += 1
    con.commit()
    if done % 50 == 0: print(done, "...", flush=True)
con.commit(); print("done", done, "failed", fail)
