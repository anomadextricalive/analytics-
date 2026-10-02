"""Merge Statsguru bulk-crawl rows (~/etpl2026/statsguru/bulk.db) into espn_career (fmt='t20'), keyed by ESPN id.
Checks first (written to ~/etpl2026/statsguru/checks_*.csv):
  1. every pass has all pages 'ok' and 200 rows/page (last page may be shorter)
  2. same player across passes must agree on Mat/Inns/Runs (batting passes) -> disagreements listed
  3. player-set differences between passes -> listed (never silently zero-filled)
  4. --overrides FILE: per-player live career lines (espn_id -> {name, batting, bowling}) replace the bulk rows for players whose
     bulk pages were fetched at different moments (they played a match mid-crawl, or debuted). Applied before the checks.
Usage: python scripts/merge_statsguru_bulk.py [db] [--check-only] [--overrides FILE]"""
import csv, json, sqlite3, sys, collections
OVR = sys.argv[sys.argv.index("--overrides") + 1] if "--overrides" in sys.argv else None
args = [a for a in sys.argv[1:] if not a.startswith("--") and a != OVR]
DB = args[0] if args else "data/cricket.db"; CHECK = "--check-only" in sys.argv
BULK = "/Users/anomadextricalive/etpl2026/statsguru/bulk.db"; OUT = "/Users/anomadextricalive/etpl2026/statsguru/"
b = sqlite3.connect(BULK)
print("jobs:", b.execute("SELECT pass, status, COUNT(*), SUM(rows) FROM jobs GROUP BY 1,2").fetchall())
tp = dict(b.execute("SELECT pass, MAX(total_pages) FROM jobs GROUP BY 1").fetchall())
okp = dict(b.execute("SELECT pass, COUNT(*) FROM jobs WHERE status='ok' GROUP BY 1").fetchall())
incomplete = {p: (okp.get(p, 0), tp[p]) for p in tp if okp.get(p, 0) != tp[p]}
if incomplete: print("INCOMPLETE passes (ok pages, expected):", incomplete)
P = collections.defaultdict(dict)   # pass -> espn_id -> cells
dups = 0
for ps, eid, cells in b.execute("SELECT pass, espn_id, cells FROM rows ORDER BY page, pos"):
    if eid in P[ps]: dups += 1
    P[ps][eid] = json.loads(cells)
print({k: len(v) for k, v in P.items()}, "| duplicate ids within a pass:", dups)
if OVR:
    for eid, o in json.load(open(OVR)).items():
        bt, bw = o.get("batting"), o.get("bowling")
        if isinstance(bt, dict):
            for ps in ("bat_default", "bat_bf", "bat_fours"): P[ps][eid] = {**bt, "Player": o["name"]}
        if isinstance(bw, dict):
            if not bw.get("Balls") and str(bw.get("Overs", "")).replace(".", "").isdigit(): w, _, x = str(bw["Overs"]).partition("."); bw = {**bw, "Balls": str(int(w) * 6 + int(x or 0))}
            P["bowl_default"][eid] = {**bw, "Player": o["name"]}
    print("overrides applied:", len(json.load(open(OVR))))
# 2. cross-pass agreement
bad = []
for eid, c in P["bat_default"].items():
    for other in ("bat_bf", "bat_fours"):
        o = P[other].get(eid)
        if o:
            for k in ("Mat", "Inns", "Runs", "NO", "HS", "100", "50", "0"):
                if k in c and k in o and c[k] != o[k]: bad.append((eid, c["Player"], other, k, c[k], o[k]))
with open(OUT + "checks_cross_pass_disagreements.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["espn_id", "player", "pass", "field", "default", "other"]); w.writerows(bad)
print("cross-pass field disagreements:", len(bad))
# 3. player-set differences
d = set(P["bat_default"]); 
for other in ("bat_bf", "bat_fours"):
    s = set(P[other]); print(f"{other}: only-in-default {len(d - s)}, only-in-{other} {len(s - d)}")
    with open(OUT + f"checks_playerset_{other}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["espn_id", "player", "where"])
        for e in d - s: w.writerow([e, P['bat_default'][e]['Player'], "default only"])
        for e in s - d: w.writerow([e, P[other][e]['Player'], other + " only"])
if CHECK or incomplete: sys.exit(0)

def n(v, cast=int):
    try: return cast(str(v).replace(",", "").replace("*", ""))
    except Exception: return None
con = sqlite3.connect(DB, timeout=60); cur = con.cursor()
fa = dict(b.execute("SELECT pass, MAX(fetched_at) FROM jobs GROUP BY 1").fetchall())
cnt = 0
for eid, c in P["bat_default"].items():
    bf, fo = P["bat_bf"].get(eid, {}), P["bat_fours"].get(eid, {})
    miss = [p for p, x in (("bat_bf", bf), ("bat_fours", fo)) if not x]
    cur.execute("""INSERT OR REPLACE INTO espn_career (espn_id, fmt, stat_type, status, span, mat, inns, runs, bf, hs, ave, sr, hundreds, fifties, fours, sixes, nos, ducks, fetched_at, source, raw_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (eid, "t20", "batting", "ok", c.get("Span"), n(c.get("Mat")), n(c.get("Inns")), n(c.get("Runs")), n(bf.get("BF")), c.get("HS"), n(c.get("Ave"), float),
         n(bf.get("SR"), float), n(c.get("100")), n(c.get("50")), n(fo.get("4s")), n(fo.get("6s")), n(c.get("NO")), n(c.get("0")), fa["bat_default"],
         "statsguru bulk class6 bat_default" + ("+bf" if bf else "") + ("+fours" if fo else ""), json.dumps({"default": c, "bf": bf, "fours": fo, "missing_passes": miss})))
    cnt += 1
for eid, c in P["bowl_default"].items():
    cur.execute("""INSERT OR REPLACE INTO espn_career (espn_id, fmt, stat_type, status, span, mat, inns, runs, balls, wkts, bbi, ave, econ, sr, four_w, five_w, fetched_at, source, raw_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (eid, "t20", "bowling", "ok", c.get("Span"), n(c.get("Mat")), n(c.get("Inns")), n(c.get("Runs")), n(c.get("Balls")), n(c.get("Wkts")), c.get("BBI"),
         n(c.get("Ave"), float), n(c.get("Econ"), float), n(c.get("SR"), float), n(c.get("4")), n(c.get("5")), fa["bowl_default"], "statsguru bulk class6 bowl_default", json.dumps(c)))
    cnt += 1
con.commit(); print("espn_career rows written:", cnt)
print(cur.execute("SELECT fmt, stat_type, status, COUNT(*) FROM espn_career GROUP BY 1,2,3").fetchall())
linked = cur.execute("SELECT COUNT(*) FROM player_espn_map m JOIN espn_career e ON e.espn_id=CAST(m.espn_id AS TEXT) AND e.fmt='t20' AND e.stat_type='batting' WHERE m.status='matched'").fetchone()[0]
print("linked players with a t20 batting line:", linked, "of", cur.execute("SELECT COUNT(*) FROM player_espn_map WHERE status='matched'").fetchone()[0])
