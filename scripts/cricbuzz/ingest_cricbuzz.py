"""Ingest Cricbuzz-converted cricsheet JSON (one dir per tournament code) into the clone DB."""
import sys
from pathlib import Path
sys.path.insert(0, '.')
from sqlalchemy.orm import Session
from sqlalchemy import text
from config import DB_PATH
from src.db.schema import init_db
from src.ingest.parser import ingest_directory

src = Path(sys.argv[1])
s = Session(init_db(DB_PATH)); tot = 0
for d in sorted(p for p in src.iterdir() if p.is_dir()):
    r = ingest_directory(s, d, tournament=d.name); s.commit(); tot += r['inserted']
    print(f"{d.name:42s} {r}", flush=True)
s.execute(text("UPDATE matches SET match_type='T10' WHERE substr(tournament, 1, 4) = 't10_'")); s.commit()
print('inserted total', tot, flush=True)
