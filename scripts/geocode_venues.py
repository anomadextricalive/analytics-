"""Look up city coordinates and altitude for every venue with Open-Meteo's free geocoder (GeoNames-derived; no key needed).

Writes data/venue_geo_raw.csv (as returned, with a `confidence` note per row) and data/venue_geo.csv (country aliases and place-name-in-country-field
cases reclassified). Altitude is city level, accurate to roughly 100 m. Be polite: about one request per 0.6 s, results cached by (city, country).
  python scripts/geocode_venues.py"""
import collections
import csv
import json
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT))
from config import DB_PATH

ALIAS = {"uk": "united kingdom", "england": "united kingdom", "scotland": "united kingdom", "wales": "united kingdom", "uae": "united arab emirates",
         "usa": "united states", "us": "united states", "u.s.a.": "united states"}


def query(name):
    url = "https://geocoding-api.open-meteo.com/v1/search?count=8&format=json&name=" + urllib.parse.quote(name)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cricket-analytics-venue-enrichment/1.0"}), timeout=15) as r:
                return json.load(r).get("results") or []
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def suffix(name):
    m = re.search(r",\s*([^,]+)$", name or ""); return m.group(1).strip() if m else ""


def main():
    con = sqlite3.connect(DB_PATH); rows, cache = [], {}
    for vid, name, city, country in con.execute("SELECT id, name, city, country FROM venues"):
        c = (city or "").strip() or suffix(name); ctry = (country or "").strip() or suffix(name)
        if not c: rows.append((vid, name, "", ctry, None, None, None, "", "", "none: no city")); continue
        key = (c.lower(), ctry.lower())
        if key not in cache: cache[key] = query(c); time.sleep(0.6)
        res = cache[key]
        if res is None: rows.append((vid, name, c, ctry, None, None, None, "", "", "error")); continue
        if not res: rows.append((vid, name, c, ctry, None, None, None, "", "", "none: not found")); continue
        pick, conf = None, "unverified: country unknown, took top result"
        for r in res:
            got = (r.get("country") or "").lower()
            if ctry and (ctry.lower() in got or got in ctry.lower()): pick, conf = r, "country match"; break
        if pick is None and ctry: pick, conf = res[0], f"weak: country '{ctry}' not matched, took {res[0].get('country')}"
        pick = pick or res[0]
        if pick.get("elevation") is None: conf += "; no elevation"
        rows.append((vid, name, c, ctry, pick.get("latitude"), pick.get("longitude"), pick.get("elevation"), pick.get("name"), pick.get("country"), conf))
    head = ["venue_id", "venue", "city_used", "country_used", "lat", "lon", "elevation_m", "matched_place", "matched_country", "confidence"]
    with open(ROOT / "data" / "venue_geo_raw.csv", "w", newline="") as f: w = csv.writer(f); w.writerow(head); w.writerows(rows)
    recs = [dict(zip(head, r)) for r in rows]; known = {r["matched_country"].lower() for r in recs if r["confidence"].startswith("country match") and r["matched_country"]}
    for r in recs:
        if not r["confidence"].startswith("weak"): continue
        used, got = r["country_used"].lower().strip(), r["matched_country"].lower().strip()
        if ALIAS.get(used) == got: r["confidence"] = "country match (alias)"
        elif used not in known and used not in ALIAS: r["confidence"] = f"unverified: '{r['country_used']}' is a place name, took top result ({r['matched_country']})"
    with open(ROOT / "data" / "venue_geo.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=head); w.writeheader(); w.writerows(recs)
    print(len(recs), "venues;", len(cache), "lookups;", dict(collections.Counter(r["confidence"].split(":")[0].split(" (")[0] for r in recs)))


if __name__ == "__main__":
    main()
