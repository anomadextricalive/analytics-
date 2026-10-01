"""Sync cricket.db (SQLite) -> MongoDB cluster. Staging collection + rename swap, keeps raw collections.
Usage (from the analytics clone):  python ~/etpl2026/sync_mongo.py [--skip deliveries] [--only t1 t2]
URI is read from ~/.cricket_mongo_uri (never pass on the command line)."""
import sys, os, time, datetime, argparse
from pathlib import Path
sys.path.insert(0, os.getcwd())
import certifi
from pymongo import MongoClient
from sqlalchemy import inspect, text
from config import DB_PATH
from src.db.schema import get_engine

ap = argparse.ArgumentParser(); ap.add_argument('--skip', nargs='*', default=[]); ap.add_argument('--only', nargs='*', default=[])
ap.add_argument('--no-profiles', action='store_true'); ap.add_argument('--db', default='cricket_analytics')
a = ap.parse_args()
uri = open(os.path.expanduser('~/.cricket_mongo_uri')).read().strip()
mdb = MongoClient(uri, serverSelectionTimeoutMS=15000, tlsCAFile=certifi.where())[a.db]
eng = get_engine(DB_PATH)
PROFILES = {'player_profiles', 'venue_profiles', 'match_profiles'}
sqlite_tables = set(inspect(eng).get_table_names())
existing = set(mdb.list_collection_names()) - PROFILES
EXTRA = {'player_career_intl', 'player_career_status', 'player_vs_bowler_style', 'tournaments', 'player_phase_bat', 'player_phase_bowl'}
tables = sorted((existing | EXTRA) & sqlite_tables)
if a.only: tables = [t for t in tables if t in a.only]
tables = [t for t in tables if t not in a.skip]
print('tables:', tables, flush=True)

def coerce(v):
    if isinstance(v, datetime.date) and not isinstance(v, datetime.datetime): return datetime.datetime(v.year, v.month, v.day)
    return v

def swap(stg, name):
    if name in mdb.list_collection_names():
        mdb[stg].rename(name, dropTarget=True)
    else:
        mdb[stg].rename(name)

CH = 10000
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
for t in tables:
    t0 = time.time(); stg = f'{t}__new'; mdb[stg].drop(); n = 0
    with eng.connect().execution_options(stream_results=True) as conn, ThreadPoolExecutor(6) as ex:
        res = conn.execute(text(f'SELECT * FROM {t}')); cols = list(res.keys())
        pending = set()
        while True:
            rows = res.fetchmany(CH)
            if not rows: break
            docs = [{c: coerce(v) for c, v in zip(cols, r)} for r in rows]; n += len(docs)
            pending.add(ex.submit(mdb[stg].insert_many, docs, ordered=False))
            if len(pending) >= 12:                     # bound memory: at most 12 batches in flight
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done: fut.result()
        for fut in pending: fut.result()
    if mdb[stg].estimated_document_count() != n:
        raise SystemExit(f'{t}: count mismatch {mdb[stg].estimated_document_count()} vs {n}, live collection untouched')
    swap(stg, t)
    print(f'{t}: {n} rows ({round(time.time()-t0)}s)', flush=True)

if not a.no_profiles:
    from scripts.build_mongo_profiles import build_venue_profiles, build_match_profiles, build_player_profiles
    for name, fn, ch in [('venue_profiles', build_venue_profiles, 500), ('match_profiles', build_match_profiles, 500), ('player_profiles', build_player_profiles, 200)]:
        t0 = time.time(); docs = fn(eng); stg = f'{name}__new'; mdb[stg].drop()
        for i in range(0, len(docs), ch): mdb[stg].insert_many(docs[i:i+ch])
        mdb[stg].create_index('id', unique=True); swap(stg, name)
        print(f'{name}: {len(docs)} docs ({round(time.time()-t0)}s)', flush=True)
print('DONE', flush=True)
