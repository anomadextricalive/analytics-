"""
Fill empty player bio fields from scraped Cricbuzz profiles. Never overwrites a value that is
already set (ESPN data wins). Bowling style is mapped onto the repo's 8-bucket vocabulary.

Usage (from repo root):
  python scripts/cricbuzz/apply_profiles.py <profiles_dir> <cb_player_map.json>
"""
import datetime
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import DB_PATH  # noqa: E402

BOWL = [  # first matching fragment wins (lower-case)
    ("left-arm chinaman", "Left-arm wrist-spin"), ("left-arm wrist", "Left-arm wrist-spin"),
    ("left-arm orthodox", "Slow left-arm orthodox"), ("slow left-arm", "Slow left-arm orthodox"),
    ("left-arm fast-medium", "Left-arm fast-medium"), ("left-arm medium", "Left-arm fast-medium"),
    ("left-arm fast", "Left-arm fast"),
    ("offbreak", "Right-arm off-break"), ("off-break", "Right-arm off-break"), ("off break", "Right-arm off-break"),
    ("legbreak", "Right-arm leg-break googly"), ("leg-break", "Right-arm leg-break googly"), ("googly", "Right-arm leg-break googly"),
    ("right-arm fast-medium", "Right-arm fast-medium"), ("right-arm medium", "Right-arm fast-medium"),
    ("right-arm fast", "Right-arm fast"),
]
BAT = {"right handed bat": "Right-hand bat", "left handed bat": "Left-hand bat"}
ROLE = {"batsman": "Batter", "batter": "Batter", "bowler": "Bowler", "batting allrounder": "Batting All-rounder",
        "bowling allrounder": "Bowling All-rounder", "wk-batsman": "Wicketkeeper Batter", "wk-batter": "Wicketkeeper Batter"}


def bowl_style(s):
    s = (s or "").lower().split(",")[0].strip()
    return next((v for k, v in BOWL if k in s), None) if s else None


def dob(s):
    try:
        return datetime.datetime.strptime(s, "%B %d, %Y").date().isoformat()
    except Exception:
        return None


def main(profiles_dir, map_path):
    cbmap = json.load(open(map_path))
    con = sqlite3.connect(DB_PATH)
    id_of = dict(con.execute("SELECT cricsheet_key, id FROM players"))
    fields = {"full_name": 0, "date_of_birth": 0, "batting_style": 0, "bowling_style": 0, "country": 0, "player_role": 0}
    seen = 0
    for f in Path(profiles_dir).glob("*.json"):
        p = json.load(open(f))
        pid = id_of.get(cbmap.get(str(p.get("id"))))
        if not pid or p.get("err"):
            continue
        seen += 1
        vals = {
            "full_name": p.get("fullName") or p.get("name"),
            "date_of_birth": dob(p.get("DoBFormat")),
            "batting_style": BAT.get((p.get("bat") or "").lower()),
            "bowling_style": bowl_style(p.get("bowl")),
            "country": p.get("intlTeam") or None,
            "player_role": ROLE.get((p.get("role") or "").lower()),
        }
        for col, v in vals.items():
            if v:
                n = con.execute(f"UPDATE players SET {col} = ? WHERE id = ? AND ({col} IS NULL OR {col} = '')",
                                (v, pid)).rowcount
                fields[col] += n
    con.commit()
    print(f"profiles matched to players: {seen}; fields filled (only where empty): {fields}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
