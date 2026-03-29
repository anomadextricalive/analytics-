"""Central configuration for the cricket analytics system."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DB_PATH = ROOT / "data" / "cricket.db"

# Cricsheet bulk download URLs (JSON format)
CRICSHEET_BASE = "https://cricsheet.org/downloads"
# T20 ONLY — all men's T20 international and domestic T20 leagues
# Women's formats excluded per project scope.
DOWNLOADS = {
    # International
    "t20i_male":        f"{CRICSHEET_BASE}/t20s_male_json.zip",
    # Big leagues
    "ipl":              f"{CRICSHEET_BASE}/ipl_json.zip",
    "psl":              f"{CRICSHEET_BASE}/psl_json.zip",
    "bbl":              f"{CRICSHEET_BASE}/bbl_json.zip",
    "cpl":              f"{CRICSHEET_BASE}/cpl_json.zip",
    "t20_blast":        f"{CRICSHEET_BASE}/t20blast_json.zip",
    "sa20":             f"{CRICSHEET_BASE}/sa20_json.zip",
    "lpl":              f"{CRICSHEET_BASE}/lpl_json.zip",
    "ilt20":            f"{CRICSHEET_BASE}/ilt20_json.zip",
    "hundred_male":     f"{CRICSHEET_BASE}/thehundred_male_json.zip",
    # Smaller / emerging leagues
    "msl":              f"{CRICSHEET_BASE}/msl_json.zip",
    "ctsl":             f"{CRICSHEET_BASE}/ctsl_json.zip",
    "ncl":              f"{CRICSHEET_BASE}/ncl_json.zip",
    "t20_wc_male":      f"{CRICSHEET_BASE}/icc_mens_t20_world_cup_json.zip",
}

# Over phase definitions
PHASES = {
    "powerplay":  (1, 6),
    "middle":     (7, 15),
    "death":      (16, 20),
}

# Year range filter
YEAR_FROM = 2000
YEAR_TO   = 2025

# Minimum innings thresholds for rating calculations
MIN_BAT_INNINGS   = 10
MIN_BOWL_INNINGS  = 10
MIN_VENUE_INNINGS = 5   # for venue factor estimation

# Bayesian prior for venue shrinkage
VENUE_PRIOR_WEIGHT = 20  # equivalent innings of prior data
