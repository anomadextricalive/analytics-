"""
Build the `player_aliases` table: every name form a player can be searched by.

Sources per player:
  - cricsheet key            ("BJ Currie")
  - ESPN full name           ("Bradley James Currie")
  - first + last of full name ("Bradley Currie")
  - first initial + surname  ("B Currie")
  - Cricbuzz names            (data/cricbuzz/identity_decisions.csv, when present)

The table is derived and rebuilt from scratch on each run; no other table is touched.

Usage:
  python scripts/build_player_aliases.py
"""

import csv
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH  # noqa: E402

CRICBUZZ_IDS = Path(__file__).parent.parent / "data" / "cricbuzz" / "identity_decisions.csv"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", (s or "").lower().replace("-", " "))).strip()


def name_forms(key: str, full_name: str | None) -> list[tuple[str, str]]:
    forms = [(key, "cricsheet_key")]
    if full_name:
        forms.append((full_name, "full_name"))
        parts = full_name.split()
        if len(parts) >= 2:
            forms.append((f"{parts[0]} {parts[-1]}", "first_last"))
            forms.append((f"{parts[0][0]} {parts[-1]}", "initial_last"))
            # multi-word surnames: "Hendrik Erasmus van der Dussen" -> "Hendrik van der Dussen"
            for i, p in enumerate(parts[1:], 1):
                if p[:1].islower():
                    forms.append((f"{parts[0]} {' '.join(parts[i:])}", "first_last"))
                    forms.append((f"{parts[0][0]} {' '.join(parts[i:])}", "initial_last"))
                    break
    kp = key.split()
    if len(kp) >= 2 and len(kp[0]) <= 3 and kp[0].isupper():
        forms.append((f"{kp[0][0]} {' '.join(kp[1:])}", "initial_last"))
    return forms


def main():
    con = sqlite3.connect(DB_PATH)
    players = con.execute("SELECT id, cricsheet_key, full_name FROM players").fetchall()
    id_of_key = {k: pid for pid, k, _ in players}

    rows = set()
    for pid, key, full in players:
        for alias, src in name_forms(key, full):
            rows.add((pid, alias, norm(alias), src))

    if CRICBUZZ_IDS.exists():
        for r in csv.DictReader(open(CRICBUZZ_IDS)):
            pid = id_of_key.get(r["key"])
            if pid and r["cricbuzz_name"]:
                rows.add((pid, r["cricbuzz_name"], norm(r["cricbuzz_name"]), "cricbuzz"))

    # keep one row per (player, normalised alias)
    dedup = {}
    for pid, alias, n, src in sorted(rows):
        if n:
            dedup.setdefault((pid, n), (pid, alias, n, src))

    con.execute("""CREATE TABLE IF NOT EXISTS player_aliases (
                     player_id INTEGER NOT NULL, alias TEXT NOT NULL,
                     alias_norm TEXT NOT NULL, source TEXT)""")
    con.execute("DELETE FROM player_aliases")
    con.executemany("INSERT INTO player_aliases VALUES (?,?,?,?)", dedup.values())
    con.execute("CREATE INDEX IF NOT EXISTS ix_alias_norm ON player_aliases(alias_norm)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_alias_player ON player_aliases(player_id)")
    con.commit()
    print(f"player_aliases: {len(dedup)} aliases for {len({r[0] for r in dedup.values()})} players")


if __name__ == "__main__":
    main()
