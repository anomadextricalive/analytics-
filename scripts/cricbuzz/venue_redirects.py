"""
Point matches at a venue's canonical row when Cricbuzz used a different spelling of a ground the DB
already has. Hand-checked pairs only. Venue rows are left in place (nothing deleted); the applied
mapping is written to data/cricbuzz/venue_redirects_applied.csv.

Usage (from repo root):  python scripts/cricbuzz/venue_redirects.py
"""
import csv
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import DB_PATH  # noqa: E402

# (name Cricbuzz used, canonical name already in the DB); prefixes are matched with LIKE '<prefix>%'
REDIRECTS = [
    ("M.Chinnaswamy Stadium, Bengaluru", "M Chinnaswamy Stadium"),
    ("MA Chidambaram Stadium, Chennai", "MA Chidambaram Stadium, Chepauk"),
    ("Sharjah Cricket Stadium, Sharjah", "Sharjah Cricket Stadium"),
    ("Harare Sports Club, Harare", "Harare Sports Club"),
    ("Pallekele International Cricket Stadium, Pallekele", "Pallekele International Cricket Stadium"),
    ("Holkar Stadium, Indore", "Holkar Stadium"),
    ("Rajiv Gandhi International Stadium, Hyderabad", "Rajiv Gandhi International Stadium, Uppal"),
    ("Shere Bangla National Stadium, Dhaka", "Shere Bangla National Stadium, Mirpur"),
    ("Mission Road Ground, Mong Kok", "Mission Road Ground, Mong Kok, Hong Kong"),
    ("Dr DY Patil Sports Academy, Navi Mumbai", "Dr DY Patil Sports Academy, Mumbai"),
    ("R.Premadasa Stadium, Colombo", "R Premadasa Stadium, Colombo"),
    ("Rangiri Dambulla International Stadium, Dambulla", "Rangiri Dambulla International Stadium"),
    ("Barabati Stadium, Cuttack", "Barabati Stadium"),
    ("Green Park, Kanpur", "Green Park"),
    ("Lalabhai Contractor Stadium, Surat", "Lalbhai Contractor Stadium"),
    ("ICC Academy Ground No 2, Dubai", "ICC Academy Ground No 2"),
    ("ICC Academy Ground, Dubai", "ICC Academy, Dubai"),
    ("Al Amerat Cricket Ground (Ministry Turf 1), Al Amerat", "Al Amerat Cricket Ground Oman Cricket (Ministry Turf 1)"),
    ("Maharaja Yadavindra Singh International Cricket Stadium, ", "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur"),
    ("Srikantadatta Narasimha Raja Wadeyar Ground, Mysore", "Srikantadatta Narasimha Raja Wadiyar Ground, Mysore"),
]


def main():
    con = sqlite3.connect(DB_PATH)
    applied = []
    for src, dst in REDIRECTS:
        to = con.execute("SELECT id FROM venues WHERE name = ?", (dst,)).fetchone()
        if not to:
            print(f"  skip (canonical missing): {dst}")
            continue
        frm = con.execute("SELECT id, name FROM venues WHERE name LIKE ? AND id != ?", (src + "%", to[0])).fetchall()
        frm = [f for f in frm if f[1].strip() == src or src.endswith(", ")]
        for fid, fname in frm:
            n = con.execute("UPDATE matches SET venue_id = ? WHERE venue_id = ?", (to[0], fid)).rowcount
            applied.append({"from_id": fid, "from_name": fname, "to_id": to[0], "to_name": dst, "matches_moved": n})
    con.commit()
    out = ROOT / "data" / "cricbuzz" / "venue_redirects_applied.csv"
    prev = list(csv.DictReader(open(out))) if out.exists() else []
    applied = prev + [a for a in applied if a["matches_moved"]]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["from_id", "from_name", "to_id", "to_name", "matches_moved"])
        w.writeheader(); w.writerows(applied)
    print(f"redirected {sum(int(a['matches_moved']) for a in applied)} matches across {len(applied)} venues -> {out}")


if __name__ == "__main__":
    main()
