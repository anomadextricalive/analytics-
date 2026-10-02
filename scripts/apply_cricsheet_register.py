"""Link players to ESPN Cricinfo ids using Cricsheet's own register (data/cricsheet_people.csv,
from https://cricsheet.org/register/people.csv). Exact, no name matching.

- Old mapping is kept in player_espn_map__prev (created once).
- Register id overrides the old fuzzy id. Every override is written to
  data/espn_register_conflicts.csv so it can be reviewed.
- Alternate ids (key_cricinfo_2/_3) are stored in player_espn_alt_ids.
Usage: python scripts/apply_cricsheet_register.py [db] [--dry-run]
"""
import csv, sqlite3, sys, collections

DB = next((a for a in sys.argv[1:] if not a.startswith("--")), "data/cricket.db")
DRY = "--dry-run" in sys.argv
reg = {r["identifier"]: r for r in csv.DictReader(open("data/cricsheet_people.csv"))}
con = sqlite3.connect(DB)
cur = con.cursor()
if not DRY:
    cur.execute("CREATE TABLE IF NOT EXISTS player_espn_map__prev AS SELECT * FROM player_espn_map")
    cur.execute("""CREATE TABLE IF NOT EXISTS player_espn_alt_ids (
        player_id INTEGER, espn_id TEXT, source TEXT, PRIMARY KEY (player_id, espn_id))""")
import os
keep_old = set()
if os.path.exists("data/espn_register_review_keep_old.csv"):   # register id failed the ESPN Cricinfo name check, old one passed
    keep_old = {int(r["player_id"]) for r in csv.DictReader(open("data/espn_register_review_keep_old.csv"))}
old = {r[0]: r for r in cur.execute("SELECT player_id, espn_id, status FROM player_espn_map")}
players = cur.execute("SELECT id, cricsheet_uuid, cricsheet_key, full_name FROM players WHERE cricsheet_uuid IS NOT NULL").fetchall()
stats = collections.Counter(); conflicts = []; upserts = []; alts = []
for pid, uuid, key, full in players:
    r = reg.get(uuid)
    if not r or not r["key_cricinfo"]:
        stats["no register id"] += 1; continue
    new = r["key_cricinfo"]
    if pid in keep_old:
        stats["kept old (review)"] += 1; continue
    cur_row = old.get(pid)
    cur_id = str(cur_row[1]) if cur_row and cur_row[1] else None
    if cur_id == new: stats["agree"] += 1; reason = "cricsheet_register (agrees)"
    elif cur_id is None: stats["new link"] += 1; reason = "cricsheet_register (new)"
    elif cur_id in {r["key_cricinfo_2"], r["key_cricinfo_3"]}:
        stats["old id was register ALT profile"] += 1; reason = f"cricsheet_register (old id {cur_id} is alt profile)"
    else:
        stats["OVERRIDE fuzzy id"] += 1; reason = f"cricsheet_register (replaced fuzzy {cur_id})"
        conflicts.append([pid, key, full, cur_id, new, cur_row[2] if cur_row else ""])
    upserts.append((pid, new, "matched", reason))
    for k in ("key_cricinfo_2", "key_cricinfo_3"):
        if r[k]: alts.append((pid, r[k], k))
print(dict(stats), "| alt ids:", len(alts))
with open("data/espn_register_conflicts.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["player_id", "cricsheet_key", "full_name", "old_fuzzy_id", "register_id", "old_status"]); w.writerows(conflicts)
if not DRY:
    cur.executemany("""INSERT INTO player_espn_map (player_id, espn_id, status, candidates, reason)
        VALUES (?, ?, ?, NULL, ?) ON CONFLICT(player_id) DO UPDATE SET espn_id=excluded.espn_id,
        status=excluded.status, reason=excluded.reason""", upserts)
    cur.executemany("INSERT OR IGNORE INTO player_espn_alt_ids VALUES (?,?,?)", alts)
    con.commit()
    # one ESPN id -> more than one player = possible duplicate people
    d = cur.execute("""SELECT espn_id, COUNT(*) c, group_concat(player_id) FROM player_espn_map
                       WHERE status='matched' GROUP BY espn_id HAVING c>1""").fetchall()
    with open("data/espn_shared_ids.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["espn_id", "n_players", "player_ids"]); w.writerows(d)
    print("ESPN ids shared by >1 player:", len(d))
con.close()
