"""Fuzzy 'is this match already in the DB' check shared by scraper and converter.
Same date +-1 day and each team shares a distinctive word (Saint->St, club words ignored),
so renamed franchises (St Lucia Zouks / Saint Lucia Kings) still match."""
import collections, datetime, re, sqlite3

STOP = {'st', 'the', 'cc', 'club', 'cricket', 'xi', 'team', 'of', 'and', 'sc', 'fc', 'cricketers', 'in'}


# franchise nicknames shared across many leagues; matching on them gives false positives
GENERIC = set("""kings super royals riders knight knights warriors strikers tigers lions giants titans capitals stars
united challengers gladiators sixers thunder heat hurricanes scorchers renegades panthers bulls eagles falcons hawks
sharks dragons rhinos wolves rockers dockers flames guardians cosmic patriots tallawahs zouks tridents amazon smashers
blasters hitters legends masters eleven sports academy rangers bisons bears lightning storm fire red blue green black
white gold golden indians sunrisers daredevils chargers gillies spartans mavericks superstarz superstars stallions
leopards jaguars cheetahs lynx sparks blitz vikings pirates raiders invincibles originals rockets brave spirit
phoenix superchargers chiefs aces stags knights volts firebirds dolphins cobras lions titans warriors jets nawabs
sultans qalandars zalmi gladiators victorians dynamites platoon""".split())


def tokens(name):
    n = re.sub(r'[^a-z ]', ' ', (name or '').lower().replace('saint', 'st'))
    return set(n.split()) - STOP


def place_tokens(name):
    """Distinctive words only (places, sponsors); falls back to all words if nothing distinctive is left."""
    t = tokens(name)
    return (t - GENERIC) or t


class Existing:
    def __init__(self, db_path):
        db = sqlite3.connect(db_path, check_same_thread=False)
        self.by_date = collections.defaultdict(list)
        for d, a, b in db.execute("""select m.match_date, a.name, b.name from matches m
                                     join teams a on a.id=m.team1_id join teams b on b.id=m.team2_id"""):
            self.by_date[datetime.date.fromisoformat(d)].append((place_tokens(a), place_tokens(b)))

    def has(self, date, t1, t2):
        if isinstance(date, str):
            date = datetime.date.fromisoformat(date)
        a, b = place_tokens(t1), place_tokens(t2)
        for off in (-1, 0, 1):
            for x, y in self.by_date.get(date + datetime.timedelta(days=off), ()):
                if (a & x and b & y) or (a & y and b & x):
                    return True
        return False
