"""Scrape completed matches of Cricbuzz series into compact per-match JSON.

Usage: python cb_scrape.py series_list.csv   (columns: series_id, code)
Output: matches/<code>/<matchId>.json  with keys: series_id, code, info (matchInfo from series page),
        header (matchHeader), innings [{bat, bowl, extras}], commentary {inningsId: [items]}
Resumable: existing match files are skipped.
"""
import csv, json, os, re, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36"
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'matches')
dec = json.JSONDecoder()
log_lock = threading.Lock()


def get(url, tries=4):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            return urllib.request.urlopen(req, timeout=40).read().decode('utf-8', 'ignore')
        except Exception:
            time.sleep(3 * (a + 1))
    return None


def objs_after(s, key):
    """Yield (pos, obj) for every JSON object following `key` in s."""
    for m in re.finditer(re.escape(key), s):
        try:
            yield m.start(), dec.raw_decode(s[m.end():])[0]
        except Exception:
            continue


def log(msg):
    with log_lock:
        with open(os.path.join(BASE, 'scrape.log'), 'a') as fh:
            fh.write(msg + '\n')


import datetime
sys.path.insert(0, BASE)
from cb_dedupe import Existing
EXISTING = Existing(os.path.expanduser('~/etpl2026/analytics-clone/data/cricket.db'))


def in_db(mi):
    try:
        dt = datetime.datetime.fromtimestamp(int(mi['startDate']) / 1000, datetime.timezone.utc).date()
        return EXISTING.has(dt, mi['team1']['teamName'], mi['team2']['teamName'])
    except Exception:
        return False


def series_matches(sid):
    s = get(f'https://www.cricbuzz.com/cricket-series/{sid}/x/matches')
    if not s:
        return []
    s = s.replace('\\"', '"')
    out = {}
    for _, o in objs_after(s, '{"matchInfo":'):
        mi = o.get('matchInfo', o)
        if mi.get('seriesId') == sid:
            out[mi['matchId']] = mi
    return list(out.values())


def scrape_match(sid, code, mi):
    mid = mi['matchId']
    d = os.path.join(OUT, code)
    path = os.path.join(d, f'{mid}.json')
    if os.path.exists(path):
        return 'cached'
    s = get(f'https://www.cricbuzz.com/live-cricket-scorecard/{mid}/x')
    if not s:
        log(f'{mid} scorecard fetch failed'); return 'fail'
    s = s.replace('\\"', '"')
    header = next((o for _, o in objs_after(s, '"matchHeader":')), None)
    pos = [m.start() for m in re.finditer(r'"batTeamDetails":', s)]
    innings = []
    for n, p in enumerate(pos):
        seg = s[p:(pos[n + 1] if n + 1 < len(pos) else len(s))]
        bat = next((o for _, o in objs_after(seg, '"batTeamDetails":')), None)
        bowl = next((o for _, o in objs_after(seg, '"bowlTeamDetails":')), None)
        ext = next((o for _, o in objs_after(seg, '"extrasData":')), None)
        inn_id = re.search(r'"inningsId":(\d+)', s[max(0, p - 400):p])
        innings.append({'bat': bat, 'bowl': bowl, 'extras': ext,
                        'inningsId': int(inn_id.group(1)) if inn_id else n + 1})
    commentary = {}
    for i in range(1, len(innings) + 1):
        c = get(f'https://www.cricbuzz.com/api/mcenter/{mid}/full-commentary/{i}')
        try:
            j = json.loads(c) if c else {}
        except Exception:
            j = {}
        for inn in j.get('commentary', []):
            if inn.get('inningsId') == i:
                commentary[i] = inn.get('commentaryList', [])
        time.sleep(0.2)
    os.makedirs(d, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump({'series_id': sid, 'code': code, 'info': mi, 'header': header,
                   'innings': innings, 'commentary': commentary}, fh)
    return 'ok'


def run_series(row):
    sid, code = int(row['series_id']), row['code']
    ms = [m for m in series_matches(sid) if str(m.get('state', '')).lower() == 'complete']
    n = {'ok': 0, 'cached': 0, 'fail': 0, 'in_db': 0}
    for mi in ms:
        if in_db(mi):
            n['in_db'] += 1; continue
        r = scrape_match(sid, code, mi)
        n[r] += 1
        time.sleep(0.3)
    log(f'series {sid} {code}: {len(ms)} complete {n}')
    return n


if __name__ == '__main__':
    rows = list(csv.DictReader(open(sys.argv[1])))
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run_series, rows))
    log('DONE')
    print('DONE')
