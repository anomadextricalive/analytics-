"""Catalogue every Cricbuzz series id: name, dates, per-format match counts. Resumable."""
import re, json, sys, time, threading, urllib.request, os
from concurrent.futures import ThreadPoolExecutor
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36"
OUT = os.path.expanduser('~/etpl2026/cricbuzz_all/series_catalogue.jsonl')
lo, hi = int(sys.argv[1]), int(sys.argv[2])
done = set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)['id'])
lock = threading.Lock(); fh = open(OUT, 'a')
dec = json.JSONDecoder()
def probe(i):
    if i in done: return
    for attempt in range(3):
        try:
            req = urllib.request.Request(f'https://www.cricbuzz.com/cricket-series/{i}/x/matches', headers={'User-Agent': UA})
            s = urllib.request.urlopen(req, timeout=30).read().decode('utf-8', 'ignore').replace('\\"', '"')
            break
        except Exception as e:
            time.sleep(2 * (attempt + 1)); s = None
    rec = {'id': i}
    if s:
        t = re.search(r'<title>([^<]*)', s); rec['title'] = t.group(1) if t else ''
        fm = {}; dates = []; names = set()
        for m in re.finditer(r'"matchInfo":\{"matchId":(\d+),"seriesId":%d,"seriesName":"([^"]*)","matchDesc":"([^"]*)","matchFormat":"(\w+)","startDate":"(\d+)"' % i, s):
            fm[m.group(4)] = fm.get(m.group(4), 0) + 1; dates.append(int(m.group(5))); names.add(m.group(2))
        rec['formats'] = fm; rec['names'] = sorted(names)
        if dates: rec['first'] = min(dates); rec['last'] = max(dates)
    else: rec['err'] = True
    with lock:
        fh.write(json.dumps(rec) + '\n'); fh.flush()
    time.sleep(0.25)
with ThreadPoolExecutor(4) as ex: list(ex.map(probe, range(lo, hi + 1)))
print('DONE', flush=True)
