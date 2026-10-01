"""
Backfill deliveries.fielder_id / fielder2_id from the original match JSON, without re-ingesting.

Adds the two columns if the DB predates them, then for every match whose JSON is found in the
given source directories (searched recursively for <cricsheet_id>.json), walks deliveries in the
same order the parser numbered them and writes the fielders of each wicket.
Only fielder columns are written; nothing is deleted.

Usage:
  python scripts/backfill_fielders.py <json_dir> [<json_dir> ...]
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH  # noqa: E402


def main(dirs: list[str]):
    con = sqlite3.connect(DB_PATH)
    cols = {r[1] for r in con.execute("PRAGMA table_info(deliveries)")}
    for c in ("fielder_id", "fielder2_id"):
        if c not in cols:
            con.execute(f"ALTER TABLE deliveries ADD COLUMN {c} INTEGER REFERENCES players(id)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_deliv_inn_ball ON deliveries(innings_id, ball_number)")
    con.commit()

    files = {}
    for d in dirs:
        for f in Path(d).rglob("*.json"):
            files.setdefault(f.stem, f)

    by_uuid = dict(con.execute("SELECT cricsheet_uuid, id FROM players WHERE cricsheet_uuid IS NOT NULL"))
    by_key = dict(con.execute("SELECT cricsheet_key, id FROM players"))
    matches = con.execute("SELECT id, cricsheet_id FROM matches").fetchall()

    found = updated = unresolved = 0
    for mid, csid in matches:
        f = files.get(csid)
        if not f:
            continue
        found += 1
        js = json.loads(f.read_bytes())
        people = js.get("info", {}).get("registry", {}).get("people", {})
        inn_ids = dict(con.execute("SELECT innings_number, id FROM innings WHERE match_id=?", (mid,)))

        def pid(name):
            return by_uuid.get(people.get(name)) or by_key.get(name)

        rows = []
        for n, inn in enumerate(js.get("innings", []), 1):
            iid = inn_ids.get(n)
            if iid is None:
                continue
            seq = 0
            for ov in inn.get("overs", []):
                for d in ov.get("deliveries", []):
                    seq += 1
                    w = (d.get("wickets") or [None])[0]
                    if not w:
                        continue
                    if w.get("kind") == "caught and bowled":
                        f1, f2 = pid(d.get("bowler")), None
                    else:
                        names = [x.get("name") for x in (w.get("fielders") or []) if x.get("name")]
                        if not names:
                            continue
                        f1 = pid(names[0])
                        f2 = pid(names[1]) if len(names) > 1 else None
                        if f1 is None:
                            unresolved += 1
                    rows.append((f1, f2, iid, seq))
        con.executemany("UPDATE deliveries SET fielder_id=?, fielder2_id=? WHERE innings_id=? AND ball_number=?", rows)
        updated += len(rows)
        if found % 1000 == 0:
            con.commit()
            print(f"  {found} matches…", flush=True)
    con.commit()
    print(f"matches with JSON: {found}/{len(matches)} | wicket rows updated: {updated} | fielder names unresolved: {unresolved}")


if __name__ == "__main__":
    main(sys.argv[1:])
