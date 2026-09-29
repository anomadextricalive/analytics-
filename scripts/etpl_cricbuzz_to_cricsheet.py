"""Convert scraped Cricbuzz ETPL 2026 data (commentary + scorecards) into cricsheet-style JSON."""
import csv, json, re, sys, sqlite3, collections, datetime
from pathlib import Path

SRC = Path(sys.argv[1])            # scratchpad with csv files
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
DB = sys.argv[3]

matches = {int(r['id']): r for r in json.load(open(SRC / 'matches.json'))}
info = {int(r['match']): r for r in csv.DictReader(open(SRC / 'match_info.csv'))}
bat = collections.defaultdict(list)
for r in csv.DictReader(open(SRC / 'bat.csv')):
    bat[(int(r['match']), int(r['innings']))].append(r)
bowl = collections.defaultdict(list)
for r in csv.DictReader(open(SRC / 'bowl.csv')):
    bowl[(int(r['match']), int(r['innings']))].append(r)
com = collections.defaultdict(list)
for r in csv.DictReader(open(SRC / 'commentary.csv')):
    com[(int(r['match']), int(r['innings']))].append(r)

# ---------- player identity resolution against existing DB ----------
db = sqlite3.connect(DB)
P = db.execute("select id,cricsheet_key,cricsheet_uuid,full_name,country,t20_bat_innings,t20_bowl_innings from players").fetchall()
last_played = dict(db.execute("""
  select p, max(d) from (
    select pi.batter_id p, m.match_date d from player_innings pi join matches m on m.id=pi.match_id
    union all select pb.bowler_id, m.match_date from player_bowling_innings pb join matches m on m.id=pb.match_id
  ) group by p""").fetchall())
by_full = collections.defaultdict(list)
for p in P:
    if p[3]: by_full[p[3].lower()].append(p)

def norm(s): return re.sub(r'[^a-z ]', '', s.lower().replace('-', ' '))

def resolve_player(name, team):
    """Return (existing_player_row|None, confidence, note)."""
    nl = name.lower()
    ex = by_full.get(nl, [])
    if len(ex) == 1: return ex[0], 'full_name', ''
    toks = name.split()
    sur = toks[-1].lower(); first = toks[0][0].lower()
    cands = [p for p in P if norm(p[1]).split() and norm(p[1]).split()[-1] == norm(sur) and p[1][0].lower() == first]
    # multi-word surnames (van der Merwe, de Leede, du Plessis)
    if not cands and len(toks) > 2:
        sur2 = ' '.join(toks[-2:]).lower()
        cands = [p for p in P if norm(p[1]).endswith(norm(sur2)) and p[1][0].lower() == first]
    if not cands and len(toks) > 3:
        sur3 = ' '.join(toks[-3:]).lower()
        cands = [p for p in P if norm(p[1]).endswith(norm(sur3)) and p[1][0].lower() == first]
    if len(cands) == 1:
        lp = last_played.get(cands[0][0])
        return cands[0], ('surname+initial' if lp and lp >= '2025-01-01' else 'surname+initial-stale'), f'last_played={lp}'
    if len(cands) > 1:
        recent = [c for c in cands if last_played.get(c[0], '') >= '2025-06-01']
        if len(recent) == 1: return recent[0], 'ambiguous-recent', f'{len(cands)} cands'
        return None, 'ambiguous', '; '.join(f'{c[1]}({c[4]},{last_played.get(c[0])})' for c in cands)
    return None, 'new', ''

# roster (full names) per team from scorecards
teams_players = collections.defaultdict(set)
for (mid, inn), rows in bat.items():
    for r in rows: teams_players[r['team']].add(r['batName'])
for (mid, inn), rows in bowl.items():
    for r in rows: teams_players[r['team']].add(r['bowlName'])

manual = json.load(open(SRC / 'player_overrides.json')) if (SRC / 'player_overrides.json').exists() else {}
ident = {}   # full name -> dict(key, uuid, status, note)
review = []
for team, names in sorted(teams_players.items()):
    for n in sorted(names):
        if n in ident: continue
        if n in manual:
            m = manual[n]
            if m.get('new'):
                ident[n] = dict(key=n, uuid=None, status='override-new', note=m.get('why', ''))
            else:
                row = next(p for p in P if p[1] == m['key'])
                ident[n] = dict(key=row[1], uuid=row[2], status='override', note=m.get('why', ''))
            continue
        row, conf, note = resolve_player(n, team)
        if row: ident[n] = dict(key=row[1], uuid=row[2], status=conf, note=note, matched_full=row[3], country=row[4])
        else: ident[n] = dict(key=n, uuid=None, status=conf, note=note)
        review.append(dict(team=team, cricbuzz_name=n, db_key=ident[n]['key'], db_full=ident[n].get('matched_full'), status=ident[n]['status'], note=ident[n]['note']))
with open(SRC / 'player_mapping_review.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(review[0])); w.writeheader(); w.writerows(review)

# ---------- commentary -> deliveries ----------
BALL_RE = re.compile(r'^(.+?) to (.+?), (.*)$')
def short_to_full(short, cands):
    s = short.strip()
    if s in cands: return s
    m = [c for c in cands if c.endswith(' ' + s)]
    if len(m) == 1: return m[0]
    m = [c for c in cands if c.split()[-1] == s.split()[-1]]
    if len(m) == 1: return m[0]
    m = [c for c in cands if s.lower() in c.lower()]
    if len(m) == 1: return m[0]
    return None

def parse_ball(rest):
    """Return dict(bat, wides, noballs, byes, legbyes, wicket(kind,out) ...) from text after 'Bowler to Batter, '"""
    d = dict(bat=0, wide=0, nb=0, bye=0, lb=0, wkt=None, unknown=False)
    t = rest.strip()
    mo = re.search(r'(?:^|, )(out .*)$', t)
    if mo and not t.startswith('out '):
        # extras + wicket on same delivery, e.g. "no ball, out X Run Out!! ..." / "wide, out X Stumped!!"
        d = parse_ball(t[:mo.start()].rstrip(', ')) if t[:mo.start()].strip() else d
        d['wkt'] = mo.group(1); return d
    if t.startswith('out '):
        d['wkt'] = t; return d
    if t.startswith('THATS OUT'):
        d['wkt_short'] = t; return d
    parts = [p.strip() for p in t.split(',')]
    head = parts[0]
    def runs(s):
        m = re.match(r'^(\d+) runs?$', s)
        if m: return int(m.group(1))
        if s == 'no run': return 0
        if s == 'FOUR': return 4
        if s == 'SIX': return 6
        return None
    if head == 'wide': d['wide'] = 1; return d
    m = re.match(r'^(\d+) wides$', head)
    if m: d['wide'] = int(m.group(1)); return d
    if head == 'no ball':
        d['nb'] = 1
        if len(parts) > 1:
            r = runs(parts[1])
            if r is not None: d['bat'] = r
        return d
    if head == 'leg byes':
        r = runs(parts[1]) if len(parts) > 1 else None
        d['lb'] = r if r is not None else 1; return d
    if head == 'byes':
        r = runs(parts[1]) if len(parts) > 1 else None
        d['bye'] = r if r is not None else 1; return d
    r = runs(head)
    if r is not None: d['bat'] = r; return d
    d['unknown'] = True
    return d

KIND = [('Caught&Bowled', 'caught and bowled'), ('Caught', 'caught'), ('Bowled', 'bowled'), ('Lbw', 'lbw'),
        ('Stumped', 'stumped'), ('Run Out', 'run out'), ('Hit wicket', 'hit wicket'), ('Hit Wicket', 'hit wicket'),
        ('Retired', 'retired hurt'), ('Obstructing', 'obstructing the field'), ('Handled', 'handled the ball'), ('Timed', 'timed out')]

issues = []
TEAM_CANON = {'Glasgow Cosmics': 'Glasgow Cosmic', 'GGC': 'Glasgow Cosmic', 'RDD': 'Rotterdam Dockers', 'ECR': 'Edinburgh Castle Rockers',
              'BFW': 'Belfast Wolves', 'ADF': 'Amsterdam Flames', 'DBG': 'Dublin Guardians'}
def canon(n): return TEAM_CANON.get(n, n)
def build_innings(mid, inn, team_bat, team_bowl):
    brows = bat[(mid, inn)]; wrows = bowl[(mid, inn)]
    batters = [r['batName'] for r in brows]            # batting order (incl. did-not-bat)
    bowlers = [r['bowlName'] for r in wrows]
    lines = [r for r in com[(mid, inn)] if r['text']]
    lines.sort(key=lambda r: int(r['ts']))
    # openers
    openers = batters[:2]
    crease = list(openers)
    appeared = set(openers)
    last_seen = {}
    def next_new():
        for b in batters:
            if b not in appeared: return b
        return ''
    dels = []
    for r in lines:
        m = BALL_RE.match(r['text'])
        if not m or 'comes into the attack' in r['text'] or 'is back into the attack' in r['text'] or 'Local Time' in r['text']: continue
        bowler_s, batter_s, rest = m.groups()
        p = parse_ball(rest)
        if p.get('wkt_short'):
            p['_short_only'] = True
        p['bowler_s'] = bowler_s; p['batter_s'] = batter_s; p['rest'] = rest; p['over'] = r['over']; p['ball'] = r['ball']; p['ts'] = int(r['ts'])
        dels.append(p)
    # drop 'THATS OUT' duplicates when an 'out ...' line follows for same bowler/batter within the next few lines
    final = []
    for i, p in enumerate(dels):
        if p.get('_short_only'):
            dup = any(q.get('wkt') and q['bowler_s'] == p['bowler_s'] and q['batter_s'] == p['batter_s'] and 0 < q['ts'] - p['ts'] < 20000 or
                      (q.get('wkt') and q['bowler_s'] == p['bowler_s'] and q['batter_s'] == p['batter_s'] and 0 <= p['ts'] - q['ts'] < 20000)
                      for q in dels[max(0, i-3):i+4] if q is not p)
            if dup: continue
            issues.append((mid, inn, 'short-only wicket line', p['rest']))
            p['wkt'] = p['rest']
        final.append(p)
    out_deliveries = []
    legal = 0
    for p in final:
        bowler = short_to_full(p['bowler_s'], bowlers)
        striker = short_to_full(p['batter_s'], batters)
        if not bowler: issues.append((mid, inn, 'bowler unresolved', p['bowler_s'])); bowler = p['bowler_s']
        if not striker: issues.append((mid, inn, 'batter unresolved', p['batter_s'])); striker = p['batter_s']
        # crease bookkeeping
        if striker not in crease:
            issues.append((mid, inn, 'striker not in crease', f"{striker} vs {crease} @ {p['over']}"))
            # resync: striker replaces the crease member seen least recently (retired/unknown)
            old = min(crease, key=lambda b: last_seen.get(b, -1))
            crease = [striker if b == old else b for b in crease]
            appeared.add(striker)
        last_seen[striker] = len(out_deliveries)
        non_striker = crease[0] if crease[1] == striker else crease[1]
        legal_ball = not (p['wide'] or p['nb'])
        dl = {'batter': striker, 'bowler': bowler, 'non_striker': non_striker, 'runs': {'batter': p['bat'], 'extras': 0, 'total': 0}}
        ex = {}
        if p['wide']: ex['wides'] = p['wide']
        if p['nb']: ex['noballs'] = 1
        if p['bye']: ex['byes'] = p['bye']
        if p['lb']: ex['legbyes'] = p['lb']
        extras_total = sum(ex.values())
        dl['runs']['extras'] = extras_total
        dl['runs']['total'] = p['bat'] + extras_total
        if ex: dl['extras'] = ex
        if p.get('wkt'):
            t = p['wkt']
            kind = next((k for pat, k in KIND if pat in t), None)
            out_name = striker
            if kind == 'run out':
                mo = re.match(r'^out (.+?) Run Out!!', t)
                if mo:
                    out_name = short_to_full(mo.group(1), batters) or striker
            dl['wickets'] = [{'player_out': out_name, 'kind': kind or 'caught'}]
            dl['_runout'] = (kind == 'run out')
        dl['_legal'] = legal_ball
        out_deliveries.append(dl)
        # runs crossing
        crossed = (p['bat'] + p['bye'] + p['lb']) % 2 == 1 if not p['wide'] else ((p['wide'] - 1) % 2 == 1)
        if p['nb'] and p['bat'] % 2 == 1: crossed = True
        pair = [striker, non_striker]
        if crossed: pair = [non_striker, striker]
        if legal_ball:
            legal += 1
            if legal % 6 == 0: pair = [pair[1], pair[0]]
        # after swap: crease order irrelevant; we store as set-like list
        crease = pair
        if p.get('wkt'):
            outn = dl['wickets'][0]['player_out']
            if outn in crease:
                idx = crease.index(outn)
                nb_ = next_new()
                crease[idx] = nb_
                if nb_: appeared.add(nb_)
    # group into overs by legal-ball count
    overs = []; cur = []; leg = 0
    for dl in out_deliveries:
        cur.append(dl)
        if dl['_legal']:
            leg += 1
            if leg % 6 == 0:
                overs.append(cur); cur = []
    if cur: overs.append(cur)
    return overs, batters, bowlers

def totals(overs):
    runs = wk = balls = 0
    for ov in overs:
        for d in ov:
            runs += d['runs']['total']; wk += len(d.get('wickets', [])); balls += 1 if d['_legal'] else 0
    return runs, wk, balls

def parse_score(s):
    m = re.match(r'(\d+)/(\w+) \(([\d.]+)\)', s.strip().split(' ')[0] + ' ' + s.strip().split(' ')[1]) if s else None
    return m

summary = []
for mid, mrow in sorted(matches.items()):
    if (mid, 1) not in bat: continue    # abandoned, no play
    inf = info[mid]
    m = {}
    mh = json.loads(inf['revisedTarget']) if inf.get('revisedTarget') else {}
    # innings teams from bat.csv
    inn_teams = {}
    for inn in (1, 2, 3, 4):
        if (mid, inn) in bat: inn_teams[inn] = canon(bat[(mid, inn)][0]['team'])
    t1, t2 = mrow['t1'].title(), mrow['t2'].title()
    team_names = sorted(set(inn_teams.values()))
    innings_out = []
    all_names = set()
    first_total = None
    for inn in sorted(inn_teams):
        tb = inn_teams[inn]; tw = next(t for t in team_names if t != tb)
        overs, batters, bowlers = build_innings(mid, inn, tb, tw)
        # scorecard truth
        truth = mrow['s1'] if tb.upper() == mrow['t1'].upper().replace('GLASGOW COSMICS','GLASGOW COSMIC') else mrow['s2']
        mt = re.match(r'(\d+)/(\w+) \(([\d.]+)\)', truth.strip())
        t_runs = int(mt.group(1)) if mt else None
        # reconcile run-out balls (runs unknown in commentary) using each batter's scorecard runs
        sc_runs = {r['batName']: int(r['runs'] or 0) for r in bat[(mid, inn)]}
        built = collections.Counter()
        for ov in overs:
            for d in ov: built[d['batter']] += d['runs']['batter']
        for ov in overs:
            for d in ov:
                if d.get('_runout'):
                    deficit = sc_runs.get(d['batter'], 0) - built[d['batter']]
                    add = max(0, min(deficit, 3))
                    if add:
                        d['runs']['batter'] += add; d['runs']['total'] += add; built[d['batter']] += add
        runs, wk, balls = totals(overs)
        summary.append(dict(match=mid, innings=inn, team=tb, scorecard=truth, built_runs=runs, built_wkts=wk, built_balls=balls, ok=(t_runs == runs)))
        for ov in overs:
            for d in ov:
                d.pop('_legal', None); d.pop('_runout', None)
                all_names.update([d['batter'], d['bowler'], d['non_striker']])
                if d['non_striker'] == '': d['non_striker'] = d['batter']
        inn_obj = {'team': tb, 'overs': [{'over': i, 'deliveries': ov} for i, ov in enumerate(overs)]}
        if inn == 2:
            tgt = first_total + 1
            inn_obj['target'] = {'runs': tgt, 'overs': 20}
        else:
            first_total = runs
        innings_out.append(inn_obj)
    people = {}
    for n in all_names:
        if n in ident and ident[n]['uuid']: people[ident[n]['key']] = ident[n]['uuid']
    # translate names to db keys
    for io in innings_out:
        for ov in io['overs']:
            for d in ov['deliveries']:
                for f in ('batter', 'bowler', 'non_striker'):
                    d[f] = ident.get(d[f], {'key': d[f]})['key']
                for w in d.get('wickets', []):
                    w['player_out'] = ident.get(w['player_out'], {'key': w['player_out']})['key']
    toss_dec = {'Bowling': 'field', 'Batting': 'bat'}.get(inf['toss_decision'])
    outcome = {}
    st = inf['result']
    mo = re.match(r'(.+?) won by (\d+) (runs?|wkts?)', st)
    if mo:
        outcome = {'winner': canon(mo.group(1)), 'by': {('runs' if mo.group(3).startswith('run') else 'wickets'): int(mo.group(2))}}
    else:
        outcome = {'result': 'no result'}
    venue = {'The Village': 'The Village, Malahide'}.get(inf['venue'], inf['venue'] + ', ' + inf['city'] if inf['venue'] == 'Sportpark Duivesteijn' else inf['venue'])
    city = 'Malahide' if inf['venue'] == 'The Village' else inf['city']
    date = datetime.datetime.fromtimestamp(int(mrow['date'] and datetime.datetime.strptime(mrow['date'], '%Y-%m-%d %H:%M').replace(tzinfo=datetime.timezone.utc).timestamp()), datetime.timezone.utc).strftime('%Y-%m-%d')
    js = {'meta': {'data_version': 'cricbuzz-scrape', 'source': f'cricbuzz.com match {mid}'},
          'info': {'match_type': 'T20', 'gender': 'male', 'dates': [date], 'season': '2026', 'venue': venue, 'city': city,
                   'teams': team_names, 'toss': {'winner': canon(inf['toss_winner']), 'decision': toss_dec}, 'outcome': outcome,
                   'player_of_match': [ident[n]['key'] for n in inf['pom'].split(';') if n in ident] if inf['pom'] else [],
                   'event': {'name': 'European T20 Premier League', 'match_number': mrow['desc']},
                   'registry': {'people': people}},
          'innings': innings_out}
    json.dump(js, open(OUT / f'cb{mid}.json', 'w'))

with open(SRC / 'etpl_build_check.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
bad = [s for s in summary if not s['ok']]
print('innings', len(summary), 'mismatch', len(bad))
for b in bad: print(b)
print('issues', len(issues))
for i in issues[:60]: print(i)
print(collections.Counter(v['status'] for v in ident.values()))
