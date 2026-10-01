"""Convert scraped Cricbuzz matches (cb_scrape.py output) into cricsheet-style JSON.

Usage: python cb_to_cricsheet.py <matches_dir> <out_dir> <cricket.db>
  - skips matches already in the DB (same team pair within +-1 day)
  - skips innings > 2 (super overs) and matches that fail validation
  - player identity: Cricbuzz id map (cb_player_map.json) -> exact full name -> exact key ->
    unique surname + matching first name (only when the DB player has a full_name) -> new player
Writes report.csv (one row per match) next to this script.
"""
import csv, datetime, difflib, html, json, os, re, sqlite3, sys, collections
from pathlib import Path

BASE = Path(__file__).parent
SRC, OUT, DB = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
MAP_PATH = BASE / 'cb_player_map.v5.json'
SERIES_CODE_OVERRIDE = {2138: 'slpl', 10350: 'pondicherry_premier_league'}          # Sri Lanka Premier League 2012 is not the LPL

db = sqlite3.connect(DB)
norm = lambda s: re.sub(r'[^a-z ]', '', (s or '').lower().replace('-', ' ')).strip()

# ---------------- existing matches (for dedupe) ----------------
sys.path.insert(0, str(BASE))
from cb_dedupe import Existing
_existing = Existing(DB)
already_in_db = _existing.has

# ---------------- player identity ----------------
# Decided once per Cricbuzz player from ALL their matches (pass 1 collects contexts), with plausibility checks.
from cb_dedupe import place_tokens as team_tokens
P = db.execute("select id, cricsheet_key, cricsheet_uuid, full_name, country from players").fetchall()
keys = {p[1] for p in P}
uuid_of = {p[1]: p[2] for p in P}
key_of_id = {p[0]: p[1] for p in P}
full_of = {p[1]: p[3] for p in P}
country_of = {p[1]: p[4] for p in P}
by_full = collections.defaultdict(list)
by_key = collections.defaultdict(list)
by_sur = collections.defaultdict(list)
for pid, k, u, f, c in P:
    if f: by_full[norm(f)].append(k)
    by_key[norm(k)].append(k)
    if norm(k).split():
        by_sur[norm(k).split()[-1]].append(k)
ctx = collections.defaultdict(set)            # key -> {(team token, year)}
years = collections.defaultdict(set)          # key -> {year}
t20i_nations = collections.defaultdict(set)   # key -> {national teams played T20Is for}
for pid, team, d, tour in db.execute("""
        select pi.batter_id, t.name, substr(m.match_date,1,4), m.tournament from player_innings pi
        join innings i on i.id=pi.innings_id join teams t on t.id=i.batting_team_id join matches m on m.id=pi.match_id
        union select pb.bowler_id, t.name, substr(m.match_date,1,4), m.tournament from player_bowling_innings pb
        join innings i on i.id=pb.innings_id join teams t on t.id=i.bowling_team_id join matches m on m.id=pb.match_id"""):
    k = key_of_id[pid]; y = int(d)
    years[k].add(y)
    for tok in team_tokens(team): ctx[k].add((tok, y))
    if tour in ('t20i_male', 't20_wc_male'): t20i_nations[k].add(team)
reg = {}
if (BASE / 'people.csv').exists():
    for r in csv.DictReader(open(BASE / 'people.csv')):
        if r.get('key_cricbuzz'): reg[r['key_cricbuzz']] = r['identifier']
key_of_uuid = {u: k for k, u in uuid_of.items() if u}
NICK = {'sam': 'samuel', 'will': 'william', 'bill': 'william', 'tom': 'thomas', 'ben': 'benjamin', 'josh': 'joshua',
        'chris': 'christopher', 'matt': 'matthew', 'mike': 'michael', 'dan': 'daniel', 'danny': 'daniel', 'nick': 'nicholas',
        'alex': 'alexander', 'andy': 'andrew', 'rob': 'robert', 'bob': 'robert', 'jim': 'james', 'jimmy': 'james',
        'joe': 'joseph', 'jon': 'jonathan', 'jonny': 'jonathan', 'pat': 'patrick', 'tim': 'timothy', 'zak': 'zachary',
        'zac': 'zachary', 'jake': 'jacob', 'harry': 'henry', 'freddie': 'frederick', 'fred': 'frederick', 'ollie': 'oliver',
        'olly': 'oliver', 'charlie': 'charles', 'dave': 'david', 'steve': 'steven', 'greg': 'gregory', 'ed': 'edward',
        'faf': 'francois', 'rassie': 'hendrik', 'jos': 'joseph', 'stephen': 'steven', 'phil': 'philip'}
# leagues restricted to Indian domestic players
INDIA_DOMESTIC = {'tnpl', 'kpl', 'dpl', 'mppl', 'uppl', 'mpl', 'apl_andhra', 'pondicherry_premier_league', 'rajasthan_premier_league',
                  'saurashtra_premier_league', 'chhattisgarh_cricket_premier_league', 'haryana_premier_league',
                  'rajputana_premier_league', 'sher_e_punjab_t20_league', 'tg20', 't20_mumbai', 'maharaja_trophy',
                  'kerala_cricket_league', 'sma'}

def first_ok(cb_first, k):
    """Cricbuzz first name compatible with DB key k: exact, nickname, or initial (no loose prefixes)."""
    kt = norm(k).split()
    f = full_of.get(k)
    if f and norm(f).split():
        ff = norm(f).split()[0]
        if ff == cb_first or NICK.get(cb_first) == ff:
            return True
        if len(cb_first) == 1:
            return ff[0] == cb_first
        # cricsheet key spelled with full first name ("Faf du Plessis" style) also counts
        return len(kt) >= 2 and kt[0] == cb_first
    if len(kt) >= 2 and len(kt[0]) <= 3:            # initials key like "SM Curran"
        return kt[0][0] == cb_first[0]
    return len(kt) >= 2 and (kt[0] == cb_first or NICK.get(cb_first) == kt[0])


def loose_first_ok(cb_first, k):
    """Looser first-name check, used only when the surname is unique in the DB."""
    f = full_of.get(k); kt = norm(k).split()
    if not f or not norm(f).split():
        return len(kt) >= 2 and (kt[0][0] == cb_first[0] if len(kt[0]) <= 3 else kt[0] == cb_first)
    ft = norm(f).split()
    if cb_first in ft[:-1] or NICK.get(cb_first) in ft:
        return True
    ff = ft[0]
    if len(cb_first) >= 3 and (ff.startswith(cb_first) or cb_first.startswith(ff)):
        return True
    return difflib.SequenceMatcher(None, cb_first, ff).ratio() >= 0.85

class CBPlayer:
    def __init__(self):
        self.names = collections.Counter(); self.ctx = set(); self.years = set(); self.tours = set()

CB = collections.defaultdict(CBPlayer)

def collect(cbid, name, team, year, tour):
    if not cbid: return
    p = CB[str(cbid)]; p.names[name] += 1; p.years.add(year); p.tours.add(tour)
    for tok in team_tokens(team): p.ctx.add((tok, year))

def overlap(p, k):
    return sum(1 for tok, y in p.ctx if any(t == tok and abs(yy - y) <= 2 for t, yy in ctx.get(k, ())))

def plausible(p, k):
    ys = years.get(k)
    if ys and p.years and (min(p.years) > max(ys) + 6 or max(p.years) < min(ys) - 6):
        return False                                # careers far apart
    if p.tours & INDIA_DOMESTIC:
        if country_of.get(k) and country_of[k] != 'India':
            return False
        if any('india' not in n.lower() for n in t20i_nations.get(k, ())):
            return False                            # played T20Is for another nation
    return True

def decide(cbid, p):
    name = p.names.most_common(1)[0][0]
    n = norm(name); t = n.split()
    if cbid in reg and reg[cbid] in key_of_uuid:
        return key_of_uuid[reg[cbid]], 'register'
    exact = set(by_full.get(n, [])) | set(by_key.get(n, []))
    exact = [k for k in exact if plausible(p, k)]
    if len(exact) == 1:
        return exact[0], 'exact_name'
    if len(exact) > 1:
        best = sorted(exact, key=lambda k: -overlap(p, k))
        if overlap(p, best[0]) > overlap(p, best[1]):
            return best[0], 'exact_name+team'
    if not t:
        return None, 'new'
    sur = t[-1]; first = t[0] if len(t) > 1 else None
    cands = [k for k in by_sur.get(sur, []) if (first is None or first_ok(first, k)) and plausible(p, k)]
    if cands:
        scored = sorted(((overlap(p, k), k) for k in cands), reverse=True)
        if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            return scored[0][1], 'team_context'
        if first and len(cands) == 1 and full_of.get(cands[0]) and len(first) > 1:
            return cands[0], 'first+surname'
    if first and len(first) > 1:                    # surname shared by exactly one DB player
        same = [k for k in by_sur.get(sur, []) if len(norm(k).split()) >= 2]
        if len(same) == 1 and loose_first_ok(first, same[0]) and plausible(p, same[0]):
            return same[0], 'unique_surname'
    if first:                                       # one-word cricsheet key ("Hazratullah")
        mono = [k for k in set(by_key.get(first, []) + by_key.get(sur, []))
                if len(norm(k).split()) == 1 and plausible(p, k) and years.get(k)]
        if len(mono) == 1 and overlap(p, mono[0]) > 0:     # one-word keys only with a shared team
            return mono[0], 'mononym+team'
    return None, 'new'

cbmap, stats, decisions = {}, collections.Counter(), []
used_keys = set()

def resolve_all():
    for cbid, p in CB.items():
        key, how = decide(cbid, p)
        name = p.names.most_common(1)[0][0]
        if not key:
            key = name if (name not in keys and name not in used_keys) else f'{name} (cb{cbid})'
        cbmap[cbid] = key; used_keys.add(key); stats[how] += 1
        decisions.append({'cricbuzz_id': cbid, 'cricbuzz_name': name, 'key': key, 'rule': how,
                          'cb_years': f'{min(p.years)}-{max(p.years)}', 'cb_tournaments': ' '.join(sorted(p.tours)),
                          'db_full_name': full_of.get(key) or '', 'db_years': f'{min(years[key])}-{max(years[key])}' if years.get(key) else ''})

def resolve(cbid, name, team=None, year=None):
    cbid = str(cbid)
    if cbid in cbmap and cbid != '0':
        return cbmap[cbid]
    stats['unmapped'] += 1
    return name if name not in keys else f'{name} (cb-noid)'


# ---------------- commentary parsing ----------------
BALL_RE = re.compile(r'^(.+?) to (.+?), (.*)$')
KIND = [('Caught&Bowled', 'caught and bowled'), ('Caught', 'caught'), ('Bowled', 'bowled'), ('Lbw', 'lbw'),
        ('Stumped', 'stumped'), ('Run Out', 'run out'), ('Hit Wicket', 'hit wicket'), ('Hit wicket', 'hit wicket'),
        ('Retired', 'retired hurt'), ('Obstructing', 'obstructing the field'), ('Handled', 'handled the ball'),
        ('Timed', 'timed out')]

def runs_of(s):
    m = re.match(r'^(\d+) runs?$', s)
    if m: return int(m.group(1))
    return {'no run': 0, 'FOUR': 4, 'SIX': 6}.get(s)

def parse_ball(t):
    d = dict(bat=0, wide=0, nb=0, bye=0, lb=0, wkt=None, unknown=False)
    t = t.strip()
    mo = re.search(r'(?:^|, )(out .*)$', t)
    if mo and not t.startswith('out '):
        pre = t[:mo.start()].rstrip(', ')
        d = parse_ball(pre) if pre else d
        d['wkt'] = mo.group(1); return d
    if t.startswith('out '):
        d['wkt'] = t; return d
    if t.startswith('THATS OUT'):
        d['short_wkt'] = True; return d
    parts = [p.strip() for p in t.split(',')]
    head = parts[0]
    if head == 'wide': d['wide'] = 1; return d
    m = re.match(r'^(\d+) wides$', head)
    if m: d['wide'] = int(m.group(1)); return d
    if head == 'no ball':
        d['nb'] = 1
        r = runs_of(parts[1]) if len(parts) > 1 else None
        if r is not None: d['bat'] = r
        return d
    if head in ('leg byes', 'byes'):
        r = runs_of(parts[1]) if len(parts) > 1 else None
        d['lb' if head == 'leg byes' else 'bye'] = r if r is not None else 1; return d
    r = runs_of(head)
    if r is not None: d['bat'] = r; return d
    d['unknown'] = True
    return d

def clean_text(c):
    t = c.get('commText', '')
    for v in (c.get('commentaryFormats') or {}).values():
        for fid, val in zip(v.get('formatId', []), v.get('formatValue', [])):
            t = t.replace(fid, val)
    return html.unescape(re.sub('<[^>]+>', '', t))

def name_lookup(short, cands):
    s = short.strip()
    if s in cands: return s
    for test in (lambda c: c.endswith(' ' + s), lambda c: c.split()[-1] == s.split()[-1], lambda c: s.lower() in c.lower()):
        m = [c for c in cands if test(c)]
        if len(m) == 1: return m[0]
    return None

def build_innings(inn, comm):
    """Return (deliveries list, issues) for one innings; names are Cricbuzz full names."""
    bat = list(inn['bat']['batsmenData'].values())
    bowl = list(inn['bowl']['bowlersData'].values())
    id2bat = {b['batId']: b['batName'] for b in bat}
    id2bowl = {b['bowlerId']: b['bowlName'] for b in bowl}
    batters = [b['batName'] for b in bat]
    bowlers = [b['bowlName'] for b in bowl]
    issues = []
    lines = sorted([c for c in comm if c.get('commText')], key=lambda c: c.get('timestamp', 0))
    items = []
    for c in lines:
        t = clean_text(c)
        if 'Local Time' in t or 'comes into the attack' in t or 'back into the attack' in t:
            continue
        if not c.get('overNumber'):
            continue                     # non-delivery line (e.g. duplicate "THATS OUT!!" banner)
        m = BALL_RE.match(t.strip())
        if not m: continue
        bowler_s, batter_s, rest = m.groups()
        p = parse_ball(rest)
        if p.get('short_wkt'):
            p['wkt'] = rest
        if c.get('totalRuns') is not None:   # exact runs off this ball, trust it over text parsing
            diff = int(c['totalRuns']) - (p['bat'] + p['wide'] + (1 if p['nb'] else 0) + p['bye'] + p['lb'])
            if diff:
                if p['wide']: p['wide'] += diff
                elif p['lb']: p['lb'] += diff
                elif p['bye']: p['bye'] += diff
                else: p['bat'] += diff
                p['exact'] = True
        sid = (c.get('batsmanStriker') or {}).get('batId') or 0
        bid = (c.get('bowlerStriker') or {}).get('bowlId') or 0
        p['striker'] = id2bat.get(sid) or name_lookup(batter_s, batters)
        p['bowler'] = id2bowl.get(bid) or name_lookup(bowler_s, bowlers)
        p['bowler_s'], p['batter_s'], p['ts'], p['rest'] = bowler_s, batter_s, c.get('timestamp', 0), rest
        p['over'] = c.get('overNumber')
        items.append(p)
    # Cricbuzz sometimes re-posts a ball (correction): keep the latest legal ball per over.ball,
    final, seen_legal = [], {}
    for p in items:
        if p['wide'] or p['nb']:
            final.append(p)              # consecutive wides share over.ball and text, keep all
        else:
            if p['over'] in seen_legal:
                final[seen_legal[p['over']]] = None
            seen_legal[p['over']] = len(final); final.append(p)
    final = [p for p in final if p is not None]
    crease = batters[:2]; appeared = set(crease); last_seen = {}
    def next_new():
        return next((b for b in batters if b not in appeared), '')
    out = []
    legal = 0
    for p in final:
        if p['unknown']:
            issues.append(f"unparsed: {p['rest'][:40]}")
        striker, bowler = p['striker'], p['bowler']
        if not striker or not bowler:
            issues.append('unresolved name'); striker = striker or p['batter_s']; bowler = bowler or p['bowler_s']
        if striker not in crease:
            old = min(crease, key=lambda b: last_seen.get(b, -1)) if crease else ''
            crease = [striker if b == old else b for b in crease] if crease else [striker, '']
            appeared.add(striker)
        last_seen[striker] = len(out)
        non_striker = next((b for b in crease if b != striker), '') or striker
        is_legal = not (p['wide'] or p['nb'])
        ex = {k2: v for k2, v in (('wides', p['wide']), ('noballs', 1 if p['nb'] else 0), ('byes', p['bye']), ('legbyes', p['lb'])) if v}
        et = sum(ex.values())
        dl = {'batter': striker, 'bowler': bowler, 'non_striker': non_striker,
              'runs': {'batter': p['bat'], 'extras': et, 'total': p['bat'] + et}}
        if ex: dl['extras'] = ex
        if p.get('wkt'):
            kind = next((k2 for pat, k2 in KIND if pat in p['wkt']), 'caught')
            outn = striker
            if kind == 'run out':
                mo = re.match(r'^out (.+?) Run Out!!', p['wkt'])
                if mo: outn = name_lookup(mo.group(1), batters) or striker
            dl['wickets'] = [{'player_out': outn, 'kind': kind}]
        dl['_legal'], dl['_runout'] = is_legal, bool(p.get('wkt') and 'Run Out' in p['wkt'])
        if p.get('exact'): dl['_exact'] = True
        out.append(dl)
        crossed = ((p['bat'] + p['bye'] + p['lb']) % 2 == 1) if not p['wide'] else ((p['wide'] - 1) % 2 == 1)
        pair = [non_striker, striker] if crossed else [striker, non_striker]
        if is_legal:
            legal += 1
            if legal % 6 == 0: pair = [pair[1], pair[0]]
        crease = pair
        if dl.get('wickets'):
            o = dl['wickets'][0]['player_out']
            if o in crease:
                nb = next_new(); crease[crease.index(o)] = nb
                if nb: appeared.add(nb)
    # reconcile run-out balls with scorecard per-batter runs
    sc = {b['batName']: int(b.get('runs') or 0) for b in bat}
    built = collections.Counter()
    for d in out: built[d['batter']] += d['runs']['batter']
    for d in out:
        if d['_runout'] and not d.get('_exact'):
            add = max(0, min(sc.get(d['batter'], 0) - built[d['batter']], 3))
            if add:
                d['runs']['batter'] += add; d['runs']['total'] += add; built[d['batter']] += add
    return out, issues

def overs_to_balls(o):
    o = float(o or 0); return int(o) * 6 + round((o - int(o)) * 10)

def local_date(ts, tz):
    m = re.match(r'([+-])(\d\d):(\d\d)', tz or '+00:00')
    off = (1 if m.group(1) == '+' else -1) * (int(m.group(2)) * 60 + int(m.group(3))) if m else 0
    return (datetime.datetime.fromtimestamp(int(ts) / 1000, datetime.timezone.utc) + datetime.timedelta(minutes=off)).date().isoformat()

# ---------------- pass 1: collect every Cricbuzz player's teams/years/leagues ----------------
for f in sorted(SRC.glob('*/*.json')):
    d = json.loads(f.read_text()); h = d.get('header')
    if not h or not d.get('innings'): continue
    code = SERIES_CODE_OVERRIDE.get(int(d['series_id']), d['code'])
    t = {h['team1']['id']: h['team1']['name'], h['team2']['id']: h['team2']['name']}
    yr = int(local_date(h['matchStartTimestamp'], (h.get('venue') or {}).get('timezone'))[:4])
    for inn in d['innings']:
        if not inn.get('bat') or not inn.get('bowl'): continue
        bt = t.get(inn['bat'].get('batTeamId')) or inn['bat'].get('batTeamName')
        wt = t.get(inn['bowl'].get('bowlTeamId')) or inn['bowl'].get('bowlTeamName')
        for b in inn['bat']['batsmenData'].values(): collect(b['batId'], b['batName'], bt, yr, code)
        for b in inn['bowl']['bowlersData'].values(): collect(b['bowlerId'], b['bowlName'], wt, yr, code)
    for pm in h.get('playersOfTheMatch') or []:
        collect(pm['id'], pm['fullName'], '', yr, code)
resolve_all()
with open(BASE / 'identity_decisions.v5.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(decisions[0])); w.writeheader(); w.writerows(decisions)

# ---------------- main ----------------
report = []
for f in sorted(SRC.glob('*/*.json')):
    d = json.loads(f.read_text())
    code = SERIES_CODE_OVERRIDE.get(int(d['series_id']), d['code'])
    mid = d['info']['matchId']
    row = {'match': mid, 'code': code, 'series': d['series_id'], 'status': '', 'note': ''}
    try:
        h = d['header']
        if not h or not d['innings']:
            row['status'] = 'skip:no-scorecard'; report.append(row); continue
        t = {h['team1']['id']: h['team1']['name'], h['team2']['id']: h['team2']['name']}
        date = local_date(h['matchStartTimestamp'], (h.get('venue') or {}).get('timezone'))
        row['date'] = date; row['teams'] = ' v '.join(t.values())
        if already_in_db(date, *t.values()):
            row['status'] = 'skip:already-in-db'; report.append(row); continue
        inns = sorted([i for i in d['innings'] if i['bat'] and i['bowl']], key=lambda i: i['inningsId'])
        inns = [i for i in inns if i['inningsId'] <= 2]
        if len(d['innings']) > 2: row['note'] += 'super-over dropped; '
        built, problems = [], []
        for inn in inns:
            comm = d['commentary'].get(str(inn['inningsId'])) or d['commentary'].get(inn['inningsId']) or []
            if not comm:
                problems.append(f"inn{inn['inningsId']} no commentary"); continue
            dels, iss = build_innings(inn, comm)
            sc_runs = sum(int(b.get('runs') or 0) for b in inn['bat']['batsmenData'].values()) + int((inn['extras'] or {}).get('total') or 0)
            sc_balls = sum(overs_to_balls(b.get('overs')) for b in inn['bowl']['bowlersData'].values())
            b_runs = sum(x['runs']['total'] for x in dels); b_balls = sum(1 for x in dels if x['_legal'])
            sc_wk = sum(1 for b in inn['bat']['batsmenData'].values()
                        if (b.get('outDesc') or '').strip() and (b.get('outDesc') or '').strip().lower() not in ('not out', 'batting')
                        and 'retd' not in (b.get('outDesc') or '').lower() and 'retired' not in (b.get('outDesc') or '').lower())
            b_wk = sum(1 for x in dels for w in x.get('wickets', []) if not w['kind'].startswith('retired'))
            if abs(b_runs - sc_runs) > 3 or abs(b_balls - sc_balls) > 2 or b_wk != sc_wk:
                problems.append(f"inn{inn['inningsId']} runs {b_runs}/{sc_runs} balls {b_balls}/{sc_balls} wkts {b_wk}/{sc_wk}")
            elif b_runs != sc_runs:
                row['note'] += f"inn{inn['inningsId']} off by {sc_runs - b_runs}; "
            built.append((inn, dels))
            row['note'] += ('; '.join(sorted(set(iss)))[:120] + '; ') if iss else ''
        if problems or not built:
            row['status'] = 'skip:' + ' | '.join(problems or ['no innings']); report.append(row); continue
        # identities
        idmap = {}
        yr = int(date[:4])
        for inn, _ in built:
            bt = t.get(inn['bat'].get('batTeamId')) or inn['bat'].get('batTeamName')
            wt = t.get(inn['bowl'].get('bowlTeamId')) or inn['bowl'].get('bowlTeamName')
            for b in inn['bat']['batsmenData'].values(): idmap[b['batName']] = resolve(b['batId'], b['batName'], bt, yr)
            for b in inn['bowl']['bowlersData'].values(): idmap[b['bowlName']] = resolve(b['bowlerId'], b['bowlName'], wt, yr)
        K = lambda n: idmap.get(n) or resolve(0, n)
        innings_out, first_total = [], None
        for n, (inn, dels) in enumerate(built, 1):
            overs, cur, leg = [], [], 0
            for x in dels:
                legal = x.pop('_legal'); x.pop('_runout'); x.pop('_exact', None)
                for fld in ('batter', 'bowler', 'non_striker'): x[fld] = K(x[fld])
                for w in x.get('wickets', []): w['player_out'] = K(w['player_out'])
                cur.append(x)
                if legal:
                    leg += 1
                    if leg % 6 == 0: overs.append(cur); cur = []
            if cur: overs.append(cur)
            team = t.get(inn['bat'].get('batTeamId')) or inn['bat'].get('batTeamName')
            io = {'team': team, 'overs': [{'over': i, 'deliveries': ov} for i, ov in enumerate(overs)]}
            runs = sum(x['runs']['total'] for ov in overs for x in ov)
            if n == 2 and first_total is not None:
                rt = (h.get('revisedTarget') or {}).get('runs') or (h.get('revisedTarget') or {}).get('target')
                io['target'] = {'runs': int(rt) if rt else first_total + 1}
            else:
                first_total = runs
            innings_out.append(io)
        res = h.get('result') or {}
        if res.get('winningteamId') in t and res.get('resultType') == 'win':
            outcome = {'winner': t[res['winningteamId']],
                       'by': {('runs' if res.get('winByRuns') else 'wickets'): int(res.get('winningMargin') or 0)}}
        elif res.get('resultType') == 'tie' or 'tied' in str(h.get('status', '')).lower():
            outcome = {'result': 'tie'}
            if res.get('winningteamId') in t: outcome['eliminator'] = t[res['winningteamId']]
        else:
            outcome = {'result': 'no result'}
        toss = h.get('tossResults') or {}
        people = {}
        for key in set(idmap.values()):
            if uuid_of.get(key): people[key] = uuid_of[key]
        venue = h.get('venue') or {}
        js = {'meta': {'data_version': 'cricbuzz-scrape', 'source': f'cricbuzz.com match {mid}'},
              'info': {'match_type': 'T20', 'gender': 'male', 'dates': [date], 'season': date[:4],
                       'venue': f"{venue.get('name')}, {venue.get('city')}" if venue.get('city') else venue.get('name', 'Unknown'),
                       'city': venue.get('city'), 'teams': list(t.values()),
                       'toss': {'winner': t.get(toss.get('tossWinnerId')), 'decision': {'Bowling': 'field', 'Batting': 'bat'}.get(toss.get('decision'))},
                       'outcome': outcome,
                       'player_of_match': [resolve(p['id'], p['fullName']) for p in h.get('playersOfTheMatch') or []],
                       'event': {'name': d['info'].get('seriesName'), 'match_number': d['info'].get('matchDesc')},
                       'cricbuzz_format': d['info'].get('matchFormat'),
                       'registry': {'people': people}},
              'innings': innings_out}
        o = OUT / code; o.mkdir(parents=True, exist_ok=True)
        (o / f'cb{mid}.json').write_text(json.dumps(js))
        row['status'] = 'ok'
    except Exception as e:
        row['status'] = f'error:{type(e).__name__}:{str(e)[:80]}'
    report.append(row)

MAP_PATH.write_text(json.dumps(cbmap))
with open(BASE / 'report.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['match', 'code', 'series', 'date', 'teams', 'status', 'note']); w.writeheader(); w.writerows(report)
c = collections.Counter(r['status'].split(':')[0] + (':' + r['status'].split(':')[1] if r['status'].startswith('skip') else '') for r in report)
print('matches', len(report), dict(c))
print('identity', dict(stats))
