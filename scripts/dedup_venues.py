"""
Venue deduplication script.

Groups duplicate venue entries (same physical ground with different name
suffixes or historical names) into a single canonical row, then updates
all foreign-key references and deletes the orphaned rows.

Run after this: python scripts/pipeline.py venue metrics ratings
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from sqlalchemy import text
from rich.console import Console
from rich.table import Table

from src.db.schema import get_engine

console = Console()

# ---------------------------------------------------------------------------
# MERGE GROUPS
# Format: canonical_id → [ids to absorb into canonical]
# Canonical = physical ground entry to keep.
# ---------------------------------------------------------------------------

MERGE_GROUPS: dict[int, list[int]] = {
    # ── IPL / BCCI venues ───────────────────────────────────────────────────
    # Arun Jaitley Stadium / Feroz Shah Kotla, Delhi
    321: [36, 147],
    # Eden Gardens, Kolkata
    58:  [144],
    # M Chinnaswamy Stadium, Bengaluru
    15:  [65, 151, 219],
    # MA Chidambaram Stadium, Chepauk, Chennai
    60:  [241, 328],
    # Narendra Modi Stadium / Sardar Patel Stadium, Ahmedabad
    198: [108, 311],
    # Wankhede Stadium, Mumbai
    44:  [196],
    # Rajiv Gandhi International Stadium, Hyderabad
    81:  [187, 327],
    # Punjab Cricket Association IS Bindra Stadium, Mohali
    301: [80, 185, 326],
    # Sawai Mansingh Stadium, Jaipur
    329: [142],
    # Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow
    145: [59],
    # Dr DY Patil Sports Academy, Mumbai
    331: [333],
    # Brabourne Stadium, Mumbai
    330: [292],
    # Maharashtra Cricket Association / Subrata Roy Sahara, Pune
    93:  [197, 310],
    # Rajiv Gandhi International Stadium (already merged above as 81)
    # Saurashtra Cricket Association Stadium, Rajkot
    37:  [150],
    # Barabati Stadium, Cuttack
    42:  [148],
    # Vidarbha Cricket Association Stadium, Jamtha, Nagpur
    14:  [186],
    # Dr YS Rajasekhara Reddy ACA-VDCA Stadium, Visakhapatnam
    64:  [149],
    # Barsapara Cricket Stadium, Guwahati
    189: [33],
    # Holkar Cricket Stadium, Indore
    43:  [190],
    # Himachal Pradesh Cricket Association Stadium, Dharamsala
    320: [146],
    # JSCA International Stadium Complex, Ranchi
    32:  [143],
    # Shaheed Veer Narayan Singh International Stadium, Raipur
    335: [218],
    # Greenfield International Stadium, Thiruvananthapuram
    38:  [188],
    # Shrimant Madhavrao Scindia Cricket Stadium, Gwalior — solo, skip
    # Zahur Ahmed Chowdhury Stadium, Chattogram
    87:  [200],

    # ── Pakistan venues ─────────────────────────────────────────────────────
    # National Stadium, Karachi
    158: [48],
    # Gaddafi Stadium, Lahore
    159: [31],

    # ── Sri Lanka venues ─────────────────────────────────────────────────────
    # R Premadasa Stadium, Colombo (also old spelling R.Premadasa/Khettarama)
    120: [27, 29],
    # Mahinda Rajapaksa International Cricket Stadium, Hambantota
    359: [308],

    # ── Bangladesh venues ───────────────────────────────────────────────────
    # Shere Bangla National Stadium, Mirpur
    307: [46],
    # Sylhet International Cricket Stadium
    47:  [316],

    # ── UAE venues ───────────────────────────────────────────────────────────
    # Sheikh Zayed Stadium, Abu Dhabi
    18:  [337],
    # ICC Academy, Dubai
    153: [67],

    # ── Caribbean venues ─────────────────────────────────────────────────────
    # Kensington Oval, Bridgetown, Barbados
    25:  [116],
    # Queen's Park Oval, Port of Spain
    26:  [357],
    # Warner Park, Basseterre, St Kitts
    170: [57, 62],
    # Daren Sammy / Beausejour / Darren Sammy Stadium, St Lucia
    122: [355, 300, 61],
    # Sir Vivian Richards Stadium, North Sound, Antigua
    157: [302],
    # Sabina Park, Kingston, Jamaica
    28:  [171],
    # Brian Lara Stadium, Tarouba, Trinidad
    356: [169],
    # Providence Stadium, Guyana
    83:  [299],
    # National Cricket Stadium, St George's, Grenada
    121: [94, 358],
    # Arnos Vale Ground, Kingstown, St Vincent
    229: [315],
    # Windsor Park, Roseau, Dominica
    168: [317],

    # ── South Africa venues ──────────────────────────────────────────────────
    # SuperSport Park, Centurion
    40:  [112],
    # Newlands, Cape Town
    41:  [360],
    # The Wanderers / New Wanderers Stadium, Johannesburg
    289: [39, 111, 361],
    # St George's Park (Port Elizabeth / Gqeberha)
    78:  [215, 363],
    # Boland Park, Paarl
    362: [104],

    # ── Australia venues ─────────────────────────────────────────────────────
    # Melbourne Cricket Ground — no duplicates
    # Sydney Cricket Ground — no duplicates
    # Adelaide Oval — no duplicates
    # Brisbane Cricket Ground, Woolloongabba
    56:  [163, 347],
    # Bellerive Oval, Hobart
    20:  [162],
    # Perth Stadium — no duplicates
    # Manuka Oval, Canberra
    71:  [123],
    # GMHBA / Simonds Stadium, South Geelong
    161: [2, 342],
    # Docklands Stadium, Melbourne
    339: [348],
    # Aurora Stadium, Launceston
    341: [349],
    # Western Australia Cricket Association Ground / W.A.C.A.
    291: [340],

    # ── New Zealand venues ───────────────────────────────────────────────────
    # Eden Park, Auckland
    8:   [181],
    # Bay Oval, Mount Maunganui
    7:   [178],
    # McLean Park, Napier
    6:   [179],
    # Seddon Park, Hamilton
    22:  [216],
    # Westpac / Sky Stadium, Wellington
    21:  [102, 217],
    # Hagley Oval, Christchurch
    180: [82],
    # University Oval, Dunedin
    182: [101],
    # Saxton Oval, Nelson
    245: [30],

    # ── England venues ───────────────────────────────────────────────────────
    # Edgbaston, Birmingham
    34:  [140],
    # Old Trafford, Manchester
    35:  [107],
    # The Rose Bowl, Southampton
    9:   [110],
    # Trent Bridge, Nottingham
    296: [105],
    # Sophia Gardens, Cardiff
    11:  [109],
    # Riverside Ground, Chester-le-Street
    12:  [195],
    # County Ground, Bristol
    10:  [141],
    # Kennington Oval, London (also "Kenmington" typo)
    290: [214],

    # ── Ireland venues ───────────────────────────────────────────────────────
    # The Village, Malahide, Dublin
    49:  [113, 213],
    # Bready Cricket Club, Magheramason
    129: [4, 66],
    # Civil Service Cricket Club, Stormont, Belfast
    295: [114],

    # ── Scotland venues ──────────────────────────────────────────────────────
    # Grange Cricket Club Ground, Raeburn Place, Edinburgh
    134: [319, 45],

    # ── Netherlands venues ───────────────────────────────────────────────────
    # Sportpark Het Schootsveld, Deventer
    209: [52],
    # Sportpark Maarschalkerweerd, Utrecht
    248: [70],
    # Sportpark Westvliet, The Hague
    166: [85, 279],

    # ── Spain venues ─────────────────────────────────────────────────────────
    # Desert Springs Cricket Ground, Almeria
    138: [95],

    # ── Nepal venues ─────────────────────────────────────────────────────────
    # Tribhuvan University International Cricket Ground, Kirtipur
    119: [96],

    # ── Romania venues ───────────────────────────────────────────────────────
    # Moara Vlasiei Cricket Ground, Ilfov County
    139: [103],

    # ── PNG venues ───────────────────────────────────────────────────────────
    # Amini Park, Port Moresby
    211: [69],

    # ── Zimbabwe venues ──────────────────────────────────────────────────────
    # Queens Sports Club, Bulawayo
    167: [314],

    # ── Namibia venues ───────────────────────────────────────────────────────
    # United Cricket Club Ground, Windhoek
    164: [90],
    # Wanderers Cricket Ground, Windhoek
    118: [117, 313],

    # ── Guernsey ─────────────────────────────────────────────────────────────
    # College Field, St Peter Port
    76:  [165],

    # ── Canada ───────────────────────────────────────────────────────────────
    # Maple Leaf North-West Ground, King City
    254: [297],

    # ── Hong Kong ────────────────────────────────────────────────────────────
    # Mission Road Ground, Mong Kok, Hong Kong
    204: [322],

    # ── Rwanda ───────────────────────────────────────────────────────────────
    # Gahanga International Cricket Stadium, Rwanda (period typo in old entry)
    199: [132],

    # ── India (various) ──────────────────────────────────────────────────────
    # Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur
    332: [268],
}

# Canonical names to use (rename canonical row to this)
CANONICAL_NAMES: dict[int, str] = {
    321: "Arun Jaitley Stadium, Delhi",          # Feroz Shah Kotla → current name
    198: "Narendra Modi Stadium, Ahmedabad",     # Sardar Patel → current name
    301: "Punjab Cricket Association IS Bindra Stadium, Mohali",
    153: "ICC Academy, Dubai",
    21:  "Sky Stadium, Wellington",              # Westpac → current name
    339: "Marvel Stadium, Melbourne",            # Docklands → current name
    291: "Western Australia Cricket Association Ground, Perth",
    161: "GMHBA Stadium, South Geelong",
    93:  "Maharashtra Cricket Association Stadium, Pune",
    307: "Shere Bangla National Stadium, Mirpur",
    120: "R Premadasa Stadium, Colombo",
    159: "Gaddafi Stadium, Lahore",
    158: "National Stadium, Karachi",
    122: "Daren Sammy National Cricket Stadium, Gros Islet, St Lucia",
    289: "New Wanderers Stadium, Johannesburg",
    78:  "St George's Park, Gqeberha",
    129: "Bready Cricket Club, Magheramason",
    134: "Grange Cricket Club Ground, Raeburn Place, Edinburgh",
    295: "Civil Service Cricket Club, Stormont, Belfast",
    290: "Kennington Oval, London",
    199: "Gahanga International Cricket Stadium, Rwanda",
}

# matches is the only table we UPDATE in-place (no unique constraint on venue_id).
# Aggregate tables (venue_difficulty, player_venue_bat, player_venue_bowl) have
# unique constraints on venue_id, so we DELETE stale rows and let the pipeline
# regenerate everything from the cleaned matches table.
TABLES_UPDATE_VENUE_FK = [
    ("matches", "venue_id"),
]
TABLES_DELETE_STALE_VENUE = [
    "venue_difficulty",
    "player_venue_bat",
    "player_venue_bowl",
]


def run_dedup(dry_run: bool = False) -> None:
    engine  = get_engine()
    session = Session(engine)

    # Build flat map: old_id → canonical_id
    redirect: dict[int, int] = {}
    for canon, dupes in MERGE_GROUPS.items():
        for d in dupes:
            redirect[d] = canon

    # Validate canonical IDs exist
    all_ids_needed = set(MERGE_GROUPS.keys()) | set(redirect.keys())
    existing = {r[0] for r in session.execute(text("SELECT id FROM venues")).fetchall()}
    missing  = all_ids_needed - existing
    if missing:
        console.print(f"[yellow]Warning: venue IDs not in DB: {sorted(missing)}[/yellow]")

    # Remove unknowns from redirect
    redirect = {k: v for k, v in redirect.items() if k in existing and v in existing}

    total_reassigned = 0
    total_deleted    = 0

    console.print(f"\n[bold]Venue deduplication — {'DRY RUN' if dry_run else 'LIVE'}[/bold]")
    console.print(f"  Canonical groups : {len(MERGE_GROUPS)}")
    console.print(f"  Venues to absorb : {len(redirect)}\n")

    # ── Step 1: Reassign matches.venue_id ───────────────────────────────────
    for table, col in TABLES_UPDATE_VENUE_FK:
        for old_id, new_id in redirect.items():
            rows = session.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE {col} = :old"),
                {"old": old_id},
            ).scalar()
            if rows:
                if not dry_run:
                    session.execute(
                        text(f"UPDATE {table} SET {col} = :new WHERE {col} = :old"),
                        {"new": new_id, "old": old_id},
                    )
                total_reassigned += rows
                console.print(f"  {table}.{col}: {old_id} → {new_id} ({rows} rows)")

    # ── Step 1b: Delete stale rows from aggregate tables ────────────────────
    # These tables have unique constraints on venue_id, so we cannot UPDATE.
    # The pipeline will recompute them from the cleaned matches table.
    stale_ids = [i for i in redirect.keys() if i in existing]
    if stale_ids:
        placeholders = ",".join(str(i) for i in stale_ids)
        for table in TABLES_DELETE_STALE_VENUE:
            if not dry_run:
                deleted = session.execute(
                    text(f"DELETE FROM {table} WHERE venue_id IN ({placeholders})")
                ).rowcount
                if deleted:
                    console.print(f"  Deleted {deleted} stale rows from {table}")
        # Also delete canonical rows so the full recompute is clean
        canon_ids = [i for i in MERGE_GROUPS.keys() if i in existing]
        canon_ph  = ",".join(str(i) for i in canon_ids)
        for table in TABLES_DELETE_STALE_VENUE:
            if not dry_run:
                deleted = session.execute(
                    text(f"DELETE FROM {table} WHERE venue_id IN ({canon_ph})")
                ).rowcount
                if deleted:
                    console.print(f"  Deleted {deleted} canonical rows from {table} (will be rebuilt)")

    # ── Step 2: Delete absorbed venues (must happen before renaming canonicals)
    ids_to_delete = [i for i in redirect.keys() if i in existing]
    if ids_to_delete and not dry_run:
        placeholders = ",".join(str(i) for i in ids_to_delete)
        session.execute(text(f"DELETE FROM venues WHERE id IN ({placeholders})"))
        total_deleted = len(ids_to_delete)

    # ── Step 3: Rename canonical rows (safe now that duplicates are gone) ───
    for canon_id, new_name in CANONICAL_NAMES.items():
        if canon_id in existing:
            old_name = session.execute(
                text("SELECT name FROM venues WHERE id = :id"), {"id": canon_id}
            ).scalar()
            if old_name != new_name:
                if not dry_run:
                    session.execute(
                        text("UPDATE venues SET name = :name WHERE id = :id"),
                        {"name": new_name, "id": canon_id},
                    )
                console.print(f"  Rename [{canon_id}]: {old_name!r} → {new_name!r}")

    if not dry_run:
        session.commit()

    console.print(f"\n[green]Done.[/green]")
    console.print(f"  Rows reassigned : {total_reassigned}")
    console.print(f"  Venues deleted  : {len(ids_to_delete) if not dry_run else 0} (would delete: {len(ids_to_delete)})")

    # ── Step 4: Verify remaining venue count ────────────────────────────────
    remaining = session.execute(text("SELECT COUNT(*) FROM venues")).scalar()
    console.print(f"  Venues remaining: {remaining}")

    session.close()


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--dry-run", is_flag=True, default=False,
                  help="Print what would happen without making changes")
    def main(dry_run: bool):
        run_dedup(dry_run)

    main()
