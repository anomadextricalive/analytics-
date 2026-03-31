"""
Cricket Analytics — Main Interactive Dashboard
Run: streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sqlalchemy import text
from sqlalchemy.orm import Session

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from config import DB_PATH
from src.db.schema import get_engine

st.set_page_config(
    page_title="Cricket Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# NEOBRUTALIST CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"], .main {
    background: #FFFCF2 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: #0D0D0D !important;
}

/* Force ALL text inside the app to be dark unless overridden */
.main p, .main span, .main div, .main label,
.main li, .main a, .main strong, .main em {
    color: #0D0D0D !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stAppViewBlockContainer"] {
    padding: 0 2rem 3rem 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D0D0D !important;
    border-right: 4px solid #FFE500 !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label { color: #FFFCF2 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Space Mono', monospace !important;
    font-size: .78rem !important;
    letter-spacing: .05em !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMarkdown p {
    font-family: 'Space Mono', monospace !important;
    font-size: .72rem !important;
    opacity: .7;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #1A1A1A !important;
    border: 2px solid #FFE500 !important;
    border-radius: 0 !important;
    color: #FFFCF2 !important;
}

/* ── Page header ── */
.nb-page-header {
    border-left: 6px solid #FFE500;
    padding: .3rem 0 .3rem 1rem;
    margin: 1.6rem 0 1.4rem;
}
.nb-page-header h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -.02em;
    color: #0D0D0D;
    margin: 0;
}
.nb-page-header p {
    font-family: 'Space Mono', monospace;
    font-size: .68rem;
    color: #555 !important;
    margin: .2rem 0 0;
}

/* ── Section labels ── */
.nb-label {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    background: #FFE500;
    color: #0D0D0D !important;
    border: 2.5px solid #0D0D0D;
    padding: 3px 10px;
    margin-bottom: 1rem;
}
.nb-divider { height: 3px; background: #0D0D0D; margin: 2rem 0; }

/* ── Stat cards ── */
.nb-card {
    background: #FFFCF2;
    border: 3px solid #0D0D0D;
    box-shadow: 5px 5px 0 #0D0D0D;
    padding: 1rem 1.2rem;
    transition: transform .1s, box-shadow .1s;
}
.nb-card:hover { transform: translate(-2px,-2px); box-shadow: 7px 7px 0 #0D0D0D; }
.nb-card.yellow { background: #FFE500; }
.nb-card.blue   { background: #3A86FF; }
.nb-card.green  { background: #06D6A0; }
.nb-card.pink   { background: #FF6B9D; }
.nb-card.orange { background: #FF8C42; }
.nb-card.purple { background: #9B5DE5; }
.nb-card.black  { background: #0D0D0D; }
.nb-card.blue  *,
.nb-card.purple *,
.nb-card.black  * { color: #FFFCF2 !important; }
.nb-card.black .c-val { color: #FFE500 !important; }
.nb-card .c-label {
    font-family: 'Space Mono', monospace;
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #0D0D0D !important;
    opacity: .55;
    margin-bottom: .4rem;
}
.nb-card.blue .c-label,
.nb-card.purple .c-label,
.nb-card.black .c-label { opacity: .75; }
.nb-card .c-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -.03em;
    color: #0D0D0D !important;
}
.nb-card .c-sub {
    font-family: 'Space Mono', monospace;
    font-size: .6rem;
    color: #555 !important;
    margin-top: .25rem;
}

/* ── Tables ── */
.nb-table {
    width: 100%;
    border-collapse: collapse;
    border: 3px solid #0D0D0D;
    box-shadow: 5px 5px 0 #0D0D0D;
    font-family: 'Space Mono', monospace;
    font-size: .75rem;
    background: #FFFCF2;
}
.nb-table thead tr { background: #0D0D0D; }
.nb-table th {
    color: #FFE500 !important;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-size: .6rem;
    padding: .6rem .85rem;
    text-align: left;
    border-right: 1px solid #333;
    white-space: nowrap;
}
.nb-table td {
    padding: .5rem .85rem;
    color: #0D0D0D !important;
    border-bottom: 2px solid #0D0D0D;
    border-right: 1px solid #E8E4D8;
    vertical-align: middle;
}
.nb-table tr:nth-child(even) td { background: #F5F2E8; }
.nb-table tr:hover td { background: #FFF5B0; cursor: pointer; }
.nb-table td:first-child { font-weight: 700; color: #0D0D0D !important; }

/* ── Rating bar ── */
.rat-bar-bg {
    background: #E8E4D8;
    border: 2px solid #0D0D0D;
    height: 12px;
    width: 100%;
}
.rat-bar-fill { height: 100%; border-right: 2px solid #0D0D0D; }

/* ── Plotly charts ── */
.stPlotlyChart > div {
    border: 3px solid #0D0D0D !important;
    box-shadow: 5px 5px 0 #0D0D0D !important;
}

/* ── All native Streamlit text elements ── */
.stMarkdown p, .stMarkdown li, .stMarkdown h1,
.stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #0D0D0D !important;
}

/* ── Inputs ── */
.stTextInput > label,
.stSelectbox > label,
.stMultiSelect > label,
.stSlider > label,
.stNumberInput > label,
.stDateInput > label { color: #0D0D0D !important; font-family: 'Space Mono', monospace !important; font-size: .68rem !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: .06em !important; }

.stTextInput input {
    border: 2.5px solid #0D0D0D !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    background: #FFFCF2 !important;
    color: #0D0D0D !important;
    box-shadow: 3px 3px 0 #0D0D0D !important;
}
.stTextInput input:focus { outline: none !important; background: #FFF5B0 !important; }
.stTextInput input::placeholder { color: #888 !important; }

/* ── Selectbox ── */
.stSelectbox [data-baseweb="select"] > div,
[data-testid="stSelectbox"] > div > div {
    border: 2.5px solid #0D0D0D !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    background: #FFFCF2 !important;
    color: #0D0D0D !important;
    box-shadow: 3px 3px 0 #0D0D0D !important;
}
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div { color: #0D0D0D !important; }

/* Dropdown menu popup */
[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[role="listbox"] li,
[role="option"] {
    background: #FFFCF2 !important;
    color: #0D0D0D !important;
    font-family: 'Space Mono', monospace !important;
    font-size: .75rem !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background: #FFE500 !important;
    color: #0D0D0D !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] > div > div {
    border: 2.5px solid #0D0D0D !important;
    border-radius: 0 !important;
    background: #FFFCF2 !important;
    box-shadow: 3px 3px 0 #0D0D0D !important;
}
[data-testid="stMultiSelect"] span { color: #0D0D0D !important; }
[data-baseweb="tag"] {
    background: #FFE500 !important;
    border: 2px solid #0D0D0D !important;
    border-radius: 0 !important;
}
[data-baseweb="tag"] span { color: #0D0D0D !important; }

/* ── Buttons ── */
.stButton button {
    background: #FFE500 !important;
    color: #0D0D0D !important;
    border: 3px solid #0D0D0D !important;
    box-shadow: 4px 4px 0 #0D0D0D !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: .72rem !important;
    font-weight: 700 !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    padding: .45rem 1.1rem !important;
    transition: all .1s !important;
}
.stButton button:hover  { transform: translate(-2px,-2px) !important; box-shadow: 6px 6px 0 #0D0D0D !important; }
.stButton button:active { transform: translate(2px,2px)   !important; box-shadow: 2px 2px 0 #0D0D0D !important; }
.stButton button p, .stButton button span { color: #0D0D0D !important; }

/* ── Slider ── */
.stSlider [data-testid="stThumbValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: .7rem !important;
    background: #FFE500 !important;
    color: #0D0D0D !important;
    border: 2px solid #0D0D0D !important;
    border-radius: 0 !important;
}
.stSlider [data-testid="stSliderTrack"] { background: #0D0D0D !important; }

/* ── Metric overrides ── */
[data-testid="stMetric"] {
    background: #FFFCF2 !important;
    border: 3px solid #0D0D0D !important;
    box-shadow: 4px 4px 0 #0D0D0D !important;
    padding: .8rem 1rem !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span {
    font-family: 'Space Mono', monospace !important;
    font-size: .62rem !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    color: #555 !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 800 !important;
    color: #0D0D0D !important;
}
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] p { color: #0D0D0D !important; }

/* ── Dataframes / st.dataframe ── */
[data-testid="stDataFrame"] {
    border: 3px solid #0D0D0D !important;
    box-shadow: 4px 4px 0 #0D0D0D !important;
}
[data-testid="stDataFrame"] th {
    background: #0D0D0D !important;
    color: #FFE500 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: .65rem !important;
    text-transform: uppercase !important;
}
[data-testid="stDataFrame"] td {
    color: #0D0D0D !important;
    font-family: 'Space Mono', monospace !important;
    font-size: .72rem !important;
    background: #FFFCF2 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 2.5px solid #0D0D0D !important;
    border-radius: 0 !important;
    box-shadow: 3px 3px 0 #0D0D0D !important;
    background: #FFFCF2 !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    color: #0D0D0D !important;
    background: #FFE500 !important;
    padding: .4rem .8rem !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: .7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: .06em !important;
    border: 2px solid transparent !important;
    border-radius: 0 !important;
    color: #555 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #FFE500 !important;
    border: 2px solid #0D0D0D !important;
    color: #0D0D0D !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 3px solid #0D0D0D !important;
    gap: 4px !important;
}
[data-testid="stTabs"] [role="tabpanel"] {
    background: #FFFCF2 !important;
    padding-top: 1rem !important;
}

/* ── Info / warning / success / error boxes ── */
[data-testid="stAlert"] {
    border: 2.5px solid #0D0D0D !important;
    border-radius: 0 !important;
    box-shadow: 3px 3px 0 #0D0D0D !important;
}
[data-testid="stAlert"] p { color: #0D0D0D !important; }

/* ── H2H comparison panel ── */
.h2h-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    padding: .5rem 1rem;
    border: 3px solid #0D0D0D;
    box-shadow: 4px 4px 0 #0D0D0D;
    margin-bottom: 1rem;
    text-align: center;
    color: #fff !important;
}
.h2h-name.a { background: #3A86FF; }
.h2h-name.b { background: #FF6B9D; }

/* ── Pill ── */
.pill {
    display: inline-block;
    padding: 2px 8px;
    border: 2px solid #0D0D0D;
    font-family: 'Space Mono', monospace;
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: #0D0D0D !important;
}
.pill.blue   { background: #3A86FF; color: #fff !important; }
.pill.yellow { background: #FFE500; color: #0D0D0D !important; }
.pill.green  { background: #06D6A0; color: #0D0D0D !important; }
.pill.pink   { background: #FF6B9D; color: #fff !important; }

/* ── Number input ── */
[data-testid="stNumberInput"] input {
    border: 2.5px solid #0D0D0D !important;
    border-radius: 0 !important;
    background: #FFFCF2 !important;
    color: #0D0D0D !important;
    font-family: 'Space Mono', monospace !important;
}

/* ── Code blocks ── */
code, pre {
    background: #0D0D0D !important;
    color: #FFE500 !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SESSION / HELPERS
# ─────────────────────────────────────────────────────────

@st.cache_resource
def get_session():
    return Session(get_engine(DB_PATH))

session = get_session()


def sql(q: str, **kw) -> pd.DataFrame:
    try:    return pd.read_sql(text(q), session.bind, params=kw or None)
    except: return pd.DataFrame()


def _plotly_defaults(fig, height=360):
    fig.update_layout(
        plot_bgcolor="#FFFCF2", paper_bgcolor="#FFFCF2",
        font=dict(family="Space Grotesk", color="#0D0D0D", size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        legend=dict(
            font=dict(family="Space Mono", size=10),
            bgcolor="#fff", bordercolor="#0D0D0D", borderwidth=2,
        ),
    )
    fig.update_xaxes(linecolor="#0D0D0D", linewidth=2, gridcolor="#E8E4D8",
                     tickfont=dict(family="Space Mono", size=10))
    fig.update_yaxes(linecolor="#0D0D0D", linewidth=2, gridcolor="#E8E4D8",
                     tickfont=dict(family="Space Mono", size=10))
    return fig


COLORS = {"A": "#3A86FF", "B": "#FF6B9D"}


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:.8rem 0 1.2rem;border-bottom:2px solid #333'>
      <div style='font-family:Space Grotesk;font-size:1.1rem;font-weight:800;
                  color:#FFE500;letter-spacing:-.02em'>🏏 CRICKET ANALYTICS</div>
      <div style='font-family:Space Mono;font-size:.6rem;color:#888;
                  margin-top:.3rem;letter-spacing:.06em'>T20 · 2001–2025</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "01  Player Explorer",
        "02  Head-to-Head",
        "03  Pitch Intelligence",
        "04  Prediction Engine",
        "05  Matchup Lab",
        "06  Match Predictor",
    ], label_visibility="collapsed")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    tournament = st.selectbox("Tournament", [
        "ALL", "t20i_male", "t20_wc_male", "ipl", "psl",
        "bbl", "cpl", "t20_blast", "sa20", "lpl", "ilt20", "hundred_male",
    ])

    st.markdown("""
    <div style='margin-top:auto;padding-top:2rem;
                font-family:Space Mono;font-size:.58rem;color:#555;
                border-top:1px solid #333;margin-top:1.5rem;padding:.8rem 0 0'>
      data/cricket.db · SQLite<br>
      cricsheet.org source
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# CACHED DATA
# ─────────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def all_players() -> pd.DataFrame:
    return sql("""
        SELECT p.id, p.cricsheet_key AS name, p.country,
               pcb.innings, pcb.runs, pcb.average, pcb.strike_rate,
               pcb.adj_average, pcb.adj_strike_rate,
               pcb.fifties, pcb.hundreds, pcb.hs,
               pcb.pp_sr, pcb.mid_sr, pcb.death_sr,
               pr.bat_rating, pr.bowl_rating, pr.overall_rating,
               pr.opener_score, pr.finisher_score, pr.anchor_score,
               pr.chase_score, pr.pp_bat_score, pr.death_bat_score,
               pr.pp_bowl_score, pr.death_bowl_score
        FROM players p
        JOIN player_career_bat pcb ON pcb.player_id = p.id AND pcb.tournament = 'ALL'
        LEFT JOIN player_ratings pr ON pr.player_id = p.id AND pr.tournament = 'ALL'
        WHERE pcb.innings >= 5
        ORDER BY pcb.innings DESC
    """)


@st.cache_data(ttl=120)
def player_seasons(pid: int) -> pd.DataFrame:
    return sql("""
        SELECT season, bat_innings AS innings, bat_runs AS runs,
               bat_average AS average, bat_sr AS sr
        FROM player_perf_by_season
        WHERE player_id = :pid AND tournament = 'ALL'
        ORDER BY season
    """, pid=pid)


@st.cache_data(ttl=120)
def player_by_phase(pid: int) -> dict:
    df = sql("""
        SELECT pp_sr, mid_sr, death_sr, adj_strike_rate AS overall_sr
        FROM player_career_bat
        WHERE player_id = :pid AND tournament = 'ALL'
    """, pid=pid)
    if df.empty: return {}
    r = df.iloc[0]
    return {
        "Powerplay": float(r.get("pp_sr") or 0),
        "Middle":    float(r.get("mid_sr") or 0),
        "Death":     float(r.get("death_sr") or 0),
        "Overall":   float(r.get("overall_sr") or 0),
    }


@st.cache_data(ttl=120)
def player_chase(pid: int) -> pd.DataFrame:
    return sql("SELECT * FROM player_chase_bat WHERE player_id = :pid", pid=pid)


@st.cache_data(ttl=120)
def player_positions(pid: int) -> pd.DataFrame:
    return sql("""
        SELECT position, innings, average, strike_rate, pp_sr, mid_sr, death_sr
        FROM player_position_bat WHERE player_id = :pid ORDER BY position
    """, pid=pid)


@st.cache_data(ttl=120)
def player_by_opponent(pid: int) -> pd.DataFrame:
    return sql("""
        SELECT t.name AS opponent, po.bat_innings AS inn,
               po.bat_runs AS runs, po.bat_average AS avg,
               po.bat_sr AS sr, po.bat_fifties AS fifties,
               po.bat_hundreds AS hundreds, po.bat_ducks AS ducks,
               po.bowl_wickets AS wkts, po.bowl_economy AS econ
        FROM player_perf_by_opponent po
        JOIN teams t ON t.id = po.opponent_id
        WHERE po.player_id = :pid ORDER BY po.bat_innings DESC
    """, pid=pid)


@st.cache_data(ttl=120)
def player_venues(pid: int) -> pd.DataFrame:
    return sql("""
        SELECT v.name AS venue, pvb.innings, pvb.runs,
               pvb.average, pvb.strike_rate,
               COALESCE(vd.bat_factor, 1.0) AS bat_factor
        FROM player_venue_bat pvb
        JOIN venues v ON v.id = pvb.venue_id
        LEFT JOIN venue_difficulty vd ON vd.venue_id = pvb.venue_id
        WHERE pvb.player_id = :pid AND pvb.innings >= 2
        ORDER BY pvb.innings DESC
    """, pid=pid)


@st.cache_data(ttl=120)
def player_milestones_df(pid: int) -> pd.DataFrame:
    return sql("""
        SELECT pm.milestone_type, pm.value, pm.match_date,
               pm.tournament, v.name AS venue, t.name AS opposition
        FROM player_milestones pm
        LEFT JOIN venues v ON v.id = pm.venue_id
        LEFT JOIN teams  t ON t.id = pm.opposition_id
        WHERE pm.player_id = :pid ORDER BY pm.match_date DESC
    """, pid=pid)


@st.cache_data(ttl=120)
def all_venues() -> pd.DataFrame:
    return sql("""
        SELECT v.id, v.name, v.country,
               vd.bat_factor, vd.boundary_rate, vd.pace_index, vd.spin_index,
               vd.avg_first_inn_runs, vd.total_matches
        FROM venues v
        LEFT JOIN venue_difficulty vd ON vd.venue_id = v.id
        ORDER BY vd.total_matches DESC NULLS LAST
    """)


@st.cache_data(ttl=120)
def venue_top_batters(vid: int) -> pd.DataFrame:
    return sql("""
        SELECT p.cricsheet_key AS player, pvb.innings, pvb.runs,
               pvb.average, pvb.strike_rate
        FROM player_venue_bat pvb
        JOIN players p ON p.id = pvb.player_id
        WHERE pvb.venue_id = :vid AND pvb.innings >= 3
        ORDER BY pvb.average DESC LIMIT 15
    """, vid=vid)


@st.cache_data(ttl=120)
def venue_top_bowlers(vid: int) -> pd.DataFrame:
    return sql("""
        SELECT p.cricsheet_key AS player, pvb.innings,
               pvb.wickets, pvb.economy, pvb.average
        FROM player_venue_bowl pvb
        JOIN players p ON p.id = pvb.player_id
        WHERE pvb.venue_id = :vid AND pvb.innings >= 3
        ORDER BY pvb.economy ASC LIMIT 15
    """, vid=vid)


@st.cache_data(ttl=120)
def matchup_stats(batter_id: int, bowler_id: int) -> pd.DataFrame:
    """Ball-by-ball stats for a specific batter vs bowler matchup."""
    return sql("""
        SELECT
            d.bat_runs, d.is_wicket, d.wicket_kind,
            d.is_boundary_4, d.is_boundary_6, d.is_dot,
            d.wide, d.no_ball, d.phase,
            d.over_number,
            m.tournament, m.season, m.match_date,
            v.name AS venue
        FROM deliveries d
        JOIN innings  i ON i.id  = d.innings_id
        JOIN matches  m ON m.id  = i.match_id
        LEFT JOIN venues v ON v.id = m.venue_id
        WHERE d.batter_id  = :bid
          AND d.bowler_id  = :oid
          AND d.wide = 0
    """, bid=batter_id, oid=bowler_id)


@st.cache_data(ttl=120)
def bowler_vs_all(bowler_id: int, min_balls: int = 6) -> pd.DataFrame:
    """All batters a bowler has faced — aggregated."""
    return sql("""
        SELECT
            p.cricsheet_key AS batter,
            COUNT(*) AS balls,
            SUM(d.bat_runs) AS runs,
            SUM(d.is_wicket) AS dismissals,
            SUM(d.is_dot) AS dots,
            SUM(d.is_boundary_4 + d.is_boundary_6) AS boundaries
        FROM deliveries d
        JOIN players p ON p.id = d.batter_id
        WHERE d.bowler_id = :oid AND d.wide = 0
        GROUP BY d.batter_id
        HAVING balls >= :mb
        ORDER BY balls DESC
    """, oid=bowler_id, mb=min_balls)


@st.cache_data(ttl=120)
def batter_vs_all(batter_id: int, min_balls: int = 6) -> pd.DataFrame:
    """All bowlers a batter has faced — aggregated."""
    return sql("""
        SELECT
            p.cricsheet_key AS bowler,
            COUNT(*) AS balls,
            SUM(d.bat_runs) AS runs,
            SUM(d.is_wicket) AS dismissals,
            SUM(d.is_dot) AS dots,
            SUM(d.is_boundary_4 + d.is_boundary_6) AS boundaries
        FROM deliveries d
        JOIN players p ON p.id = d.bowler_id
        WHERE d.batter_id = :bid AND d.wide = 0
        GROUP BY d.bowler_id
        HAVING balls >= :mb
        ORDER BY balls DESC
    """, bid=batter_id, mb=min_balls)


def _get_player(name: str) -> dict:
    df = sql("""
        SELECT p.id, p.cricsheet_key AS name, p.country,
               pcb.adj_average, pcb.adj_strike_rate, pcb.average,
               pcb.strike_rate, pcb.innings, pcb.runs, pcb.hs,
               pcb.fifties, pcb.hundreds, pcb.ducks, pcb.thirties,
               pcb.pp_sr, pcb.mid_sr, pcb.death_sr,
               pr.bat_rating, pr.bowl_rating, pr.overall_rating,
               pr.opener_score, pr.finisher_score, pr.anchor_score,
               pr.chase_score, pr.pp_bat_score, pr.death_bat_score,
               pr.pp_bowl_score, pr.death_bowl_score
        FROM players p
        JOIN player_career_bat pcb ON pcb.player_id = p.id AND pcb.tournament = 'ALL'
        LEFT JOIN player_ratings pr ON pr.player_id = p.id AND pr.tournament = 'ALL'
        WHERE p.cricsheet_key = :n
    """, n=name)
    return df.iloc[0].to_dict() if not df.empty else {}


def _get_bowl(pid: int) -> dict:
    df = sql("""
        SELECT adj_economy, economy, wickets, dot_pct, average AS bowl_avg,
               strike_rate AS bowl_sr, pp_economy, mid_economy, death_economy,
               innings AS bowl_inn
        FROM player_career_bowl WHERE player_id = :pid AND tournament = 'ALL'
    """, pid=pid)
    return df.iloc[0].to_dict() if not df.empty else {}


def _rating_bar(val, colour="#FFE500"):
    if val is None: val = 0
    pct = int(float(val))
    return (f'<div class="rat-bar-bg"><div class="rat-bar-fill" '
            f'style="width:{pct}%;background:{colour}"></div></div>')


def _radar(cats, vals_a, vals_b, name_a, name_b):
    cats_c = cats + [cats[0]]
    va = vals_a + [vals_a[0]]
    vb = vals_b + [vals_b[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=va, theta=cats_c, fill="toself",
                                   name=name_a, line=dict(color=COLORS["A"], width=2.5)))
    fig.add_trace(go.Scatterpolar(r=vb, theta=cats_c, fill="toself", opacity=0.7,
                                   name=name_b, line=dict(color=COLORS["B"], width=2.5)))
    fig.update_layout(
        polar=dict(
            bgcolor="#FFFCF2",
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(family="Space Mono", size=8)),
            angularaxis=dict(tickfont=dict(family="Space Mono", size=9)),
        ),
        plot_bgcolor="#FFFCF2", paper_bgcolor="#FFFCF2",
        font=dict(family="Space Grotesk", color="#0D0D0D"),
        margin=dict(l=40, r=40, t=40, b=40),
        height=380,
        legend=dict(font=dict(family="Space Mono", size=10),
                    bgcolor="#fff", bordercolor="#0D0D0D", borderwidth=2),
        showlegend=True,
    )
    return fig


# ─────────────────────────────────────────────────────────
# GROUND INFO — manually curated for major T20 venues
# Keys match venue names in the DB (cricsheet naming)
# ─────────────────────────────────────────────────────────
GROUND_INFO = {
    "Wankhede Stadium":                    {"dims": "65–75m", "surface": "Red soil, flat", "notes": "High bounce, true carry. Small straight boundaries. Dew after sunset."},
    "Eden Gardens":                        {"dims": "70–80m", "surface": "Red soil, slow", "notes": "Spin-assisting surface later in tournament. Outfield quick. Dew a major factor."},
    "M Chinnaswamy Stadium":               {"dims": "58–68m", "surface": "Black soil, flat", "notes": "Smallest boundaries in IPL. Very high-scoring. Pacers get hit; spinners safer."},
    "Narendra Modi Stadium":               {"dims": "75–85m", "surface": "Red soil, flat", "notes": "Largest cricket stadium. Large boundaries. Pacers get movement early. Good for bowlers."},
    "MA Chidambaram Stadium":              {"dims": "65–70m", "surface": "Black soil, spin", "notes": "Classic spin deck. Slow and low. Off-spinners and leg-spinners dominate. Very low-scoring."},
    "Sawai Mansingh Stadium":              {"dims": "68–75m", "surface": "Red soil, balanced", "notes": "Balanced pitch. Slight help for pacers early. Dew factor significant at night."},
    "Sawai Mansingh Stadium, Jaipur":      {"dims": "68–75m", "surface": "Red soil, balanced", "notes": "Balanced pitch. Slight help for pacers early. Dew factor significant at night."},
    "Punjab Cricket Association IS Bindra Stadium, Mohali": {"dims": "68–78m", "surface": "Red soil, true", "notes": "True bounce. Pacers get carry and swing early. Good batting track overall."},
    "Rajiv Gandhi International Stadium, Uppal": {"dims": "65–75m", "surface": "Black soil, flat", "notes": "Good batting surface. Dew heavy in evening games. Spin effective after powerplay."},
    "Arun Jaitley Stadium":                {"dims": "65–75m", "surface": "Red soil, balanced", "notes": "Balanced. Dew is significant. Chasing is an advantage due to dew in second innings."},
    "DY Patil Sports Academy":             {"dims": "65–75m", "surface": "Red soil, flat", "notes": "Hard, true surface. High scoring. Small square boundaries. Spinners can be expensive."},
    "Brabourne Stadium":                   {"dims": "68–75m", "surface": "Red soil, flat", "notes": "Similar to Wankhede. True bounce. Good for strokeplay. High scores common."},
    "Melbourne Cricket Ground":            {"dims": "80–90m", "surface": "Drop-in, bounce", "notes": "Largest boundaries in T20 cricket. Drop-in pitches — good carry, steep bounce. Favours pacers."},
    "Sydney Cricket Ground":               {"dims": "68–75m", "surface": "Drop-in, true", "notes": "Historically spin-friendly. Drop-in pitches now make it more pace-friendly."},
    "Perth Stadium":                       {"dims": "70–82m", "surface": "Drop-in, fast", "notes": "Fastest outfield in Australia. Good carry and bounce for pacers. High-scoring."},
    "Brisbane Cricket Ground, Woolloongabba": {"dims": "68–75m", "surface": "Drop-in, bounce", "notes": "Good carry for pace. Spinner-unfriendly. Evening dew minimal."},
    "Adelaide Oval":                       {"dims": "60–75m", "surface": "Drop-in, true", "notes": "Uneven boundary — short square. Historically spin-friendly drop-in pitch."},
    "Gaddafi Stadium":                     {"dims": "65–75m", "surface": "Spin, crumbling", "notes": "Classic spin deck. Surfaces deteriorate. Low-scoring. Slow outfield."},
    "National Stadium, Karachi":           {"dims": "68–75m", "surface": "Flat, hard", "notes": "Hard, flat surface. High-scoring ground. Relatively quick outfield."},
    "Pindi Cricket Stadium":               {"dims": "65–72m", "surface": "Flat, batting", "notes": "Very flat surface. High-scoring. First innings advantage. Boundary easier to hit."},
    "Lord's":                              {"dims": "70–80m", "surface": "Seam, bounce", "notes": "Uneven slope. Swing and seam movement. Not a typical T20 venue. Bowler-friendly."},
    "The Oval":                            {"dims": "68–78m", "surface": "Flat, fast", "notes": "Flat batting surface. Fast outfield. High-scoring T20 venue."},
    "Edgbaston":                           {"dims": "65–72m", "surface": "Seam, swing", "notes": "Seam and swing conditions. Small boundaries square. Pacers get early movement."},
    "Headingley":                          {"dims": "65–70m", "surface": "Seam, variable", "notes": "Variable bounce. Seamers and swing bowlers thrive. Low-scoring for T20."},
    "Old Trafford":                        {"dims": "65–75m", "surface": "Flat, balanced", "notes": "Flat but variable bounce. Overcast conditions aid swing. Mid-range scores."},
    "SuperSport Park":                     {"dims": "68–78m", "surface": "Pace, bounce", "notes": "Excellent carry and bounce. Pacers dominate. High altitude boosts bat."},
    "Newlands":                            {"dims": "65–72m", "surface": "Seam, swing", "notes": "Cape Town wind aids swing and seam. Pacers get extra movement. Scenic ground."},
    "The Wanderers Stadium":               {"dims": "62–70m", "surface": "Pace, flat", "notes": "Very high altitude (1750m). Ball travels far. Small square boundaries. High-scoring."},
    "Kingsmead":                           {"dims": "65–72m", "surface": "Seam, humid", "notes": "Coastal humidity aids swing. Good carry for pace. Historically spinner-unfriendly."},
    "Providence Stadium":                  {"dims": "65–75m", "surface": "Flat, balanced", "notes": "Good batting surface. Boundary not too short. Pacers get early movement."},
    "Queen's Park Oval":                   {"dims": "65–72m", "surface": "Balanced", "notes": "Classic Caribbean ground. Balanced surface. Good batting conditions."},
    "Sabina Park":                         {"dims": "62–70m", "surface": "Pace, bounce", "notes": "Traditionally fast and bouncy. Pacers with extra height extract steep bounce."},
    "National Cricket Stadium, St George's": {"dims": "62–70m", "surface": "Flat, fast", "notes": "Fast outfield. Good batting conditions. Boundary on the shorter side."},
    "R Premadasa Stadium":                 {"dims": "65–72m", "surface": "Spin, slow", "notes": "Classic Sri Lankan spin deck. Slow and low. Spinners effective throughout."},
    "Dubai International Cricket Stadium": {"dims": "68–78m", "surface": "Slow, spin", "notes": "Desert surface. Very slow and low. Spin dominant. High scores despite sluggish outfield."},
    "Sharjah Cricket Stadium":             {"dims": "58–68m", "surface": "Slow, spin", "notes": "Very short ground. Highest boundary rate in UAE. Spinners get turn but expensive."},
    "Sheikh Zayed Stadium":                {"dims": "65–75m", "surface": "Flat, slow", "notes": "Flat, slow surface. Pacers with good yorkers effective. Spin gets purchase."},
    "VRA Ground":                          {"dims": "60–68m", "surface": "Outfield, flat", "notes": "Club-level ground in Amstelveen. Flat surface. Small boundaries. Limited international data."},
    "Hagley Oval":                         {"dims": "65–72m", "surface": "Seam, swing", "notes": "Swing and seam in Christchurch conditions. Bowler-friendly in early overs."},
    "Eden Park":                           {"dims": "55–65m", "surface": "Flat, small", "notes": "Smallest international ground in NZ. Very short square boundaries. Huge scoring ground."},
    "Basin Reserve":                       {"dims": "65–75m", "surface": "Seam, windy", "notes": "Famous Wellington wind. Swing and seam. Unpredictable conditions."},
}

def _pitch_label(bat_factor, pace_index, boundary_rate):
    """Generate human-readable pitch description from stats."""
    if pd.isna(bat_factor):
        scoring = "Unknown scoring"
    elif bat_factor > 1.10:
        scoring = "High-scoring"
    elif bat_factor < 0.90:
        scoring = "Low-scoring"
    else:
        scoring = "Balanced scoring"

    if pd.isna(pace_index):
        surface = "unknown surface"
    elif pace_index > 0.65:
        surface = "pace-friendly surface"
    elif pace_index < 0.35:
        surface = "spin-friendly surface"
    else:
        surface = "balanced surface"

    if pd.isna(boundary_rate):
        size = ""
    elif boundary_rate > 0.16:
        size = "Short boundaries."
    elif boundary_rate < 0.10:
        size = "Large ground."
    else:
        size = "Standard dimensions."

    return f"{scoring} · {surface}. {size}".strip()


def _pitch_tags(bat_factor, pace_index, boundary_rate):
    """Return list of (label, colour) tags."""
    tags = []
    if not pd.isna(bat_factor):
        if bat_factor > 1.10:   tags.append(("🏏 Batter Paradise", "#FFE500"))
        elif bat_factor > 1.03: tags.append(("🏏 Batter Friendly", "#FFE500"))
        elif bat_factor < 0.90: tags.append(("⚡ Bowler Friendly", "#FF6B9D"))
        elif bat_factor < 0.97: tags.append(("⚡ Slight Bowler Edge", "#FF6B9D"))
        else:                   tags.append(("⚖ Balanced", "#A8DADC"))
    if not pd.isna(pace_index):
        if pace_index > 0.65:   tags.append(("🎳 Pace Dominant", "#FF8C42"))
        elif pace_index < 0.35: tags.append(("🌀 Spin Dominant", "#9B5DE5"))
        else:                   tags.append(("🔄 Mixed Attack", "#A8DADC"))
    if not pd.isna(boundary_rate):
        if boundary_rate > 0.16: tags.append(("📐 Tiny Ground", "#3A86FF"))
        elif boundary_rate < 0.10: tags.append(("📐 Big Ground", "#3A86FF"))
    return tags


# ═══════════════════════════════════════════════════════════
# PAGE 1 — PLAYER EXPLORER
# ═══════════════════════════════════════════════════════════

if "01" in page:
    st.markdown("""
    <div class="nb-page-header">
      <h2>Player Explorer</h2>
      <p>Browse every player in the database · filter · sort · click to expand</p>
    </div>""", unsafe_allow_html=True)

    df = all_players()
    if df.empty:
        st.warning("No players found. Run the ingestion pipeline first.")
        st.stop()

    # ── filters ──
    fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
    with fc1:
        search = st.text_input("Search player", placeholder="e.g. Kohli, Warner…")
    with fc2:
        countries = ["All"] + sorted(df["country"].dropna().unique().tolist())
        country   = st.selectbox("Country", countries)
    with fc3:
        min_inn = st.slider("Min innings", 1, 100, 10)
    with fc4:
        sort_by = st.selectbox("Sort by", [
            "innings", "bat_rating", "overall_rating", "adj_average",
            "adj_strike_rate", "death_sr", "chase_score",
        ])

    filt = df.copy()
    if search:
        filt = filt[filt["name"].str.contains(search, case=False, na=False)]
    if country != "All":
        filt = filt[filt["country"] == country]
    filt = filt[filt["innings"] >= min_inn]
    filt = filt.sort_values(sort_by, ascending=False).reset_index(drop=True)

    st.markdown(f'<div class="nb-label">Showing {len(filt):,} players</div>',
                unsafe_allow_html=True)

    # ── table ──
    def _r(v): return f"{v:.1f}" if pd.notna(v) else "—"

    rows_html = ""
    for rank, (_, r) in enumerate(filt.head(200).iterrows(), 1):
        bat_bar  = _rating_bar(r.get("bat_rating"),  "#3A86FF")
        bowl_bar = _rating_bar(r.get("bowl_rating"), "#06D6A0")
        rows_html += f"""
        <tr>
          <td style='text-align:right;opacity:.4;font-size:.65rem'>{rank}</td>
          <td><strong>{r['name']}</strong></td>
          <td style='opacity:.6'>{r.get('country') or '—'}</td>
          <td style='text-align:right'>{int(r.get('innings',0))}</td>
          <td style='text-align:right'>{int(r.get('runs',0)):,}</td>
          <td style='text-align:right'>{_r(r.get('average'))}</td>
          <td style='text-align:right'>{_r(r.get('adj_average'))}</td>
          <td style='text-align:right'>{_r(r.get('strike_rate'))}</td>
          <td style='text-align:right'>{_r(r.get('pp_sr'))}</td>
          <td style='text-align:right'>{_r(r.get('death_sr'))}</td>
          <td style='min-width:80px'>{bat_bar}<div style='font-size:.6rem;text-align:right'>{_r(r.get("bat_rating"))}</div></td>
          <td style='min-width:80px'>{bowl_bar}<div style='font-size:.6rem;text-align:right'>{_r(r.get("bowl_rating"))}</div></td>
          <td style='text-align:right;font-weight:700'>{_r(r.get('overall_rating'))}</td>
          <td style='text-align:right'>{_r(r.get('opener_score'))}</td>
          <td style='text-align:right'>{_r(r.get('finisher_score'))}</td>
          <td style='text-align:right'>{_r(r.get('chase_score'))}</td>
        </tr>"""

    st.markdown(f"""
    <table class="nb-table">
      <thead><tr>
        <th>#</th><th>Player</th><th>Country</th>
        <th>Inn</th><th>Runs</th><th>Avg</th><th>Adj Avg</th><th>SR</th>
        <th>PP SR</th><th>Death SR</th>
        <th>Bat Rating</th><th>Bowl Rating</th><th>Overall</th>
        <th>Opener</th><th>Finisher</th><th>Chase</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    # ── player drill-down ──
    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nb-label">Player Deep Dive</div>', unsafe_allow_html=True)

    sel = st.selectbox("Select player to expand", filt["name"].head(200).tolist())
    p   = _get_player(sel)
    if p:
        pid = int(p["id"])
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric("Innings",    int(p.get("innings", 0)))
        mc2.metric("Runs",       f"{int(p.get('runs', 0)):,}")
        mc3.metric("Average",    _r(p.get("average")))
        mc4.metric("Adj Avg",    _r(p.get("adj_average")))
        mc5.metric("SR",         _r(p.get("strike_rate")))
        mc6.metric("Bat Rating", _r(p.get("bat_rating")))

        t1, t2, t3, t4, t5 = st.tabs(
            ["Season Trend", "By Position", "By Opponent", "Milestones", "Venues"])

        with t1:
            seas = player_seasons(pid)
            if not seas.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=seas["season"], y=seas["average"],
                                          mode="lines+markers", name="Average",
                                          line=dict(color="#3A86FF", width=2.5),
                                          marker=dict(size=7, color="#3A86FF",
                                                      line=dict(width=2, color="#0D0D0D"))))
                fig.add_trace(go.Bar(x=seas["season"], y=seas["innings"],
                                      name="Innings", yaxis="y2",
                                      marker=dict(color="#FFE500",
                                                  line=dict(color="#0D0D0D", width=2)),
                                      opacity=0.6))
                fig.update_layout(
                    yaxis2=dict(overlaying="y", side="right",
                                tickfont=dict(family="Space Mono", size=9)),
                    title=f"{sel} — Season by Season"
                )
                st.plotly_chart(_plotly_defaults(fig), width="stretch",
                                config={"displayModeBar": False})

        with t2:
            pos = player_positions(pid)
            if not pos.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[f"#{p}" for p in pos["position"]],
                                      y=pos["strike_rate"],
                                      name="SR",
                                      marker=dict(color="#FFE500",
                                                  line=dict(color="#0D0D0D", width=2))))
                fig.add_trace(go.Scatter(x=[f"#{p}" for p in pos["position"]],
                                          y=pos["average"],
                                          name="Average",
                                          mode="lines+markers",
                                          line=dict(color="#FF6B9D", width=2.5),
                                          yaxis="y2"))
                fig.update_layout(
                    yaxis2=dict(overlaying="y", side="right",
                                tickfont=dict(family="Space Mono", size=9)),
                    title="Performance by Batting Position"
                )
                st.plotly_chart(_plotly_defaults(fig), width="stretch",
                                config={"displayModeBar": False})
                st.dataframe(pos, hide_index=True)

        with t3:
            opp = player_by_opponent(pid)
            if not opp.empty:
                top = opp.head(20)
                fig = px.bar(top, x="opponent", y="avg", color="inn",
                              title="Average vs Opponent (top 20 by innings)",
                              color_continuous_scale=["#E8E4D8", "#3A86FF"])
                fig.update_traces(marker_line_color="#0D0D0D", marker_line_width=2)
                st.plotly_chart(_plotly_defaults(fig), width="stretch",
                                config={"displayModeBar": False})
                st.dataframe(opp, hide_index=True)

        with t4:
            mils = player_milestones_df(pid)
            if not mils.empty:
                counts = mils["milestone_type"].value_counts()
                fig = go.Figure(go.Bar(
                    x=counts.index, y=counts.values,
                    marker=dict(color="#FFE500", line=dict(color="#0D0D0D", width=2.5)),
                    text=counts.values, textposition="outside",
                ))
                st.plotly_chart(_plotly_defaults(fig, 260), width="stretch",
                                config={"displayModeBar": False})
                st.dataframe(mils, hide_index=True)

        with t5:
            vdf = player_venues(pid)
            if not vdf.empty:
                fig = px.scatter(vdf, x="bat_factor", y="average",
                                  size="innings", hover_name="venue",
                                  title="Venue Difficulty vs Player Average",
                                  color="average",
                                  color_continuous_scale=["#FF6B9D","#FFE500","#06D6A0"])
                fig.add_vline(x=1.0, line_dash="dash", line_color="#0D0D0D",
                               annotation_text="neutral")
                fig.update_traces(marker_line_color="#0D0D0D", marker_line_width=2)
                st.plotly_chart(_plotly_defaults(fig), width="stretch",
                                config={"displayModeBar": False})
                st.dataframe(vdf, hide_index=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2 — HEAD-TO-HEAD
# ═══════════════════════════════════════════════════════════

elif "02" in page:
    st.markdown("""
    <div class="nb-page-header">
      <h2>Head-to-Head Comparison</h2>
      <p>Select any two players — venue-adjusted · phase-decomposed · chase-split</p>
    </div>""", unsafe_allow_html=True)

    df = all_players()
    names = df["name"].tolist()

    ca, cb = st.columns(2)
    with ca:
        name_a = st.selectbox("Player A", names, index=0)
    with cb:
        name_b = st.selectbox("Player B", names,
                               index=min(1, len(names) - 1))

    pa = _get_player(name_a)
    pb = _get_player(name_b)

    if not pa or not pb:
        st.warning("Player data not found. Run the analytics pipeline first.")
        st.stop()

    st.markdown(f"""
    <div style='display:flex;gap:1rem;margin-bottom:1.5rem'>
      <div class="h2h-name a" style='flex:1'>{name_a}</div>
      <div style='display:flex;align-items:center;font-family:Space Grotesk;
                  font-size:1.1rem;font-weight:800;padding:0 .5rem'>VS</div>
      <div class="h2h-name b" style='flex:1'>{name_b}</div>
    </div>""", unsafe_allow_html=True)

    # ── Radar ──
    radar_keys = ["opener_score","finisher_score","anchor_score",
                  "pp_bat_score","death_bat_score","chase_score",
                  "pp_bowl_score","death_bowl_score"]
    radar_labs = ["Opener","Finisher","Anchor","PP Bat",
                  "Death Bat","Chase","PP Bowl","Death Bowl"]

    def _v(d, k): return float(d.get(k) or 0)

    va = [_v(pa, k) for k in radar_keys]
    vb = [_v(pb, k) for k in radar_keys]

    rc1, rc2 = st.columns([3, 2])
    with rc1:
        st.markdown('<div class="nb-label">Specialisation Radar</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_radar(radar_labs, va, vb, name_a, name_b),
                        width="stretch", config={"displayModeBar": False})

    with rc2:
        st.markdown('<div class="nb-label">Key Stats</div>', unsafe_allow_html=True)
        metrics = [
            ("Innings",    "innings",        None),
            ("Average",    "average",        ".1f"),
            ("Adj Avg",    "adj_average",    ".1f"),
            ("SR",         "strike_rate",    ".1f"),
            ("Bat Rating", "bat_rating",     ".1f"),
            ("Bowl Rating","bowl_rating",    ".1f"),
            ("Overall",    "overall_rating", ".1f"),
        ]
        rows = ""
        for label, key, fmt in metrics:
            va_  = pa.get(key)
            vb_  = pb.get(key)
            def _fmt(v, f):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "—"
                return format(float(v), f) if f else str(int(v))
            winner = ""
            if va_ and vb_:
                if float(va_) > float(vb_): winner = "a"
                elif float(vb_) > float(va_): winner = "b"
            bg_a = "background:#DCEEFF" if winner == "a" else ""
            bg_b = "background:#FFE0ED" if winner == "b" else ""
            rows += f"""
            <tr>
              <td style='opacity:.6;font-size:.68rem'>{label}</td>
              <td style='text-align:right;font-weight:700;{bg_a}'>{_fmt(va_,fmt)}</td>
              <td style='text-align:right;font-weight:700;{bg_b}'>{_fmt(vb_,fmt)}</td>
            </tr>"""
        st.markdown(f"""
        <table class="nb-table">
          <thead><tr>
            <th>Metric</th>
            <th style='background:#3A86FF;text-align:right'>{name_a[:14]}</th>
            <th style='background:#FF6B9D;text-align:right'>{name_b[:14]}</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

    # ── Phase batting ──
    st.markdown('<div class="nb-label">Phase Strike Rate</div>', unsafe_allow_html=True)
    phases = ["Powerplay (1–6)", "Middle (7–15)", "Death (16–20)"]
    pha = [_v(pa,"pp_sr"), _v(pa,"mid_sr"), _v(pa,"death_sr")]
    phb = [_v(pb,"pp_sr"), _v(pb,"mid_sr"), _v(pb,"death_sr")]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=name_a, x=phases, y=pha,
                          marker=dict(color=COLORS["A"], line=dict(color="#0D0D0D",width=2)),
                          text=[f"{v:.0f}" for v in pha], textposition="outside"))
    fig.add_trace(go.Bar(name=name_b, x=phases, y=phb,
                          marker=dict(color=COLORS["B"], line=dict(color="#0D0D0D",width=2)),
                          text=[f"{v:.0f}" for v in phb], textposition="outside"))
    fig.update_layout(barmode="group")
    st.plotly_chart(_plotly_defaults(fig, 300), width="stretch",
                    config={"displayModeBar": False})

    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

    # ── Chase split ──
    st.markdown('<div class="nb-label">Chase vs First Innings</div>',
                unsafe_allow_html=True)
    ch_a = player_chase(int(pa["id"]))
    ch_b = player_chase(int(pb["id"]))

    def _chase_val(df, col, itype):
        row = df[df["innings_type"] == itype]
        return float(row[col].iloc[0]) if not row.empty and not pd.isna(row[col].iloc[0]) else 0

    cc1, cc2 = st.columns(2)
    for col, nm, ch, clr in [(cc1, name_a, ch_a, COLORS["A"]),
                               (cc2, name_b, ch_b, COLORS["B"])]:
        with col:
            labels = ["Chase Avg", "First Avg", "Chase SR", "First SR"]
            vals   = [_chase_val(ch, "average",      "chase"),
                      _chase_val(ch, "average",      "first"),
                      _chase_val(ch, "strike_rate",  "chase"),
                      _chase_val(ch, "strike_rate",  "first")]
            fig = go.Figure(go.Bar(
                x=labels, y=vals,
                marker=dict(color=clr, line=dict(color="#0D0D0D", width=2)),
                text=[f"{v:.1f}" for v in vals], textposition="outside",
            ))
            fig.update_layout(title=nm)
            st.plotly_chart(_plotly_defaults(fig, 280), width="stretch",
                            config={"displayModeBar": False})

    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

    # ── Common venues ──
    st.markdown('<div class="nb-label">Head-to-Head at Shared Venues</div>',
                unsafe_allow_html=True)
    va_df = player_venues(int(pa["id"]))
    vb_df = player_venues(int(pb["id"]))
    if not va_df.empty and not vb_df.empty:
        shared = set(va_df["venue"]) & set(vb_df["venue"])
        if shared:
            va_s = va_df[va_df["venue"].isin(shared)].set_index("venue")
            vb_s = vb_df[vb_df["venue"].isin(shared)].set_index("venue")
            cmp  = pd.DataFrame({
                f"{name_a} Avg": va_s["average"],
                f"{name_b} Avg": vb_s["average"],
                f"{name_a} SR":  va_s["strike_rate"],
                f"{name_b} SR":  vb_s["strike_rate"],
                "Bat Factor":    va_s["bat_factor"],
            }).dropna(how="all").sort_values(f"{name_a} Avg", ascending=False)
            st.dataframe(cmp, height=320)


# ═══════════════════════════════════════════════════════════
# PAGE 3 — PITCH INTELLIGENCE
# ═══════════════════════════════════════════════════════════

elif "03" in page:
    st.markdown("""
    <div class="nb-page-header">
      <h2>Pitch Intelligence</h2>
      <p>Venue difficulty · bat factor · pace vs spin · top performers</p>
    </div>""", unsafe_allow_html=True)

    venues_df = all_venues()
    if venues_df.empty:
        st.warning("No venue data. Run venue and metrics pipeline steps.")
        st.stop()

    # ── Global map ──
    st.markdown('<div class="nb-label">Venue Landscape</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Scatter(
        x=venues_df["bat_factor"],
        y=venues_df["boundary_rate"] * 100,
        mode="markers",
        marker=dict(
            size=np.sqrt(venues_df["total_matches"].fillna(1)) * 3,
            color=venues_df["pace_index"],
            colorscale=[[0,"#9B5DE5"],[0.5,"#FFE500"],[1,"#FF8C42"]],
            colorbar=dict(title="Pace Index", tickfont=dict(family="Space Mono",size=9)),
            line=dict(color="#0D0D0D", width=1.5),
            showscale=True,
        ),
        text=venues_df["name"] + "<br>" + venues_df["country"].fillna(""),
        hovertemplate="<b>%{text}</b><br>Bat Factor: %{x:.3f}<br>Boundary%: %{y:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=1.0, line_dash="dash", line_color="#0D0D0D",
                   annotation_text="Neutral", annotation_font_size=10)
    fig.update_layout(
        xaxis_title="Bat Factor (>1 = batter friendly)",
        yaxis_title="Boundary Rate (%)",
    )
    st.plotly_chart(_plotly_defaults(fig, 420), width="stretch",
                    config={"displayModeBar": False})

    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

    # ── Venue selector ──
    st.markdown('<div class="nb-label">Venue Deep Dive</div>', unsafe_allow_html=True)
    venue_names = venues_df["name"].tolist()
    sel_venue   = st.selectbox("Select venue", venue_names)
    vrow        = venues_df[venues_df["name"] == sel_venue].iloc[0]
    vid         = int(vrow["id"])

    # ── Pitch type tags ──
    bf  = vrow.get("bat_factor")
    pi  = vrow.get("pace_index")
    br  = vrow.get("boundary_rate")
    tags = _pitch_tags(bf, pi, br)
    tag_html = " ".join([
        f'<span style="background:{c};color:#0D0D0D;padding:.2rem .6rem;'
        f'border-radius:4px;font-family:Space Mono;font-size:.72rem;'
        f'font-weight:700;border:2px solid #0D0D0D">{t}</span>'
        for t, c in tags
    ])
    st.markdown(f'<div style="margin:.5rem 0 1rem">{tag_html}</div>',
                unsafe_allow_html=True)

    # ── Human readable summary ──
    summary = _pitch_label(bf, pi, br)
    ground  = GROUND_INFO.get(sel_venue, {})
    if ground:
        summary_extra = f" {ground.get('notes','')}"
        dims_text     = f"Boundary: {ground.get('dims','—')} | Surface: {ground.get('surface','—')}"
    else:
        summary_extra = ""
        dims_text     = ""

    st.markdown(f"""
    <div class="nb-card" style="padding:1rem 1.2rem;margin-bottom:1rem">
      <div style="font-size:1rem;font-weight:700">{summary}{summary_extra}</div>
      {"<div style='font-size:.75rem;opacity:.6;margin-top:.3rem;font-family:Space Mono'>" + dims_text + "</div>" if dims_text else ""}
    </div>""", unsafe_allow_html=True)

    vc1, vc2, vc3, vc4, vc5 = st.columns(5)
    vc1.metric("Bat Factor",       f"{bf:.3f}" if pd.notna(bf) else "—")
    vc2.metric("Avg 1st Inn",      f"{vrow.get('avg_first_inn_runs', 0):.0f}" if pd.notna(vrow.get('avg_first_inn_runs')) else "—")
    vc3.metric("Boundary Rate",    f"{(br*100):.1f}%" if pd.notna(br) else "—")
    vc4.metric("Pace Index",       f"{pi:.2f}" if pd.notna(pi) else "—")
    vc5.metric("Total Matches",    int(vrow.get("total_matches", 0)) if pd.notna(vrow.get("total_matches")) else "—")

    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown('<div class="nb-label">Top Batters Here</div>', unsafe_allow_html=True)
        tb = venue_top_batters(vid)
        if not tb.empty:
            fig = go.Figure(go.Bar(
                y=tb["player"], x=tb["average"], orientation="h",
                marker=dict(color="#3A86FF", line=dict(color="#0D0D0D", width=2)),
                text=[f"{v:.1f}" for v in tb["average"]], textposition="outside",
            ))
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(_plotly_defaults(fig, 380), width="stretch",
                            config={"displayModeBar": False})

    with tc2:
        st.markdown('<div class="nb-label">Top Bowlers Here</div>', unsafe_allow_html=True)
        tb2 = venue_top_bowlers(vid)
        if not tb2.empty:
            fig = go.Figure(go.Bar(
                y=tb2["player"], x=tb2["economy"], orientation="h",
                marker=dict(color="#FF6B9D", line=dict(color="#0D0D0D", width=2)),
                text=[f"{v:.2f}" for v in tb2["economy"]], textposition="outside",
            ))
            fig.update_layout(yaxis=dict(autorange="reversed"),
                               xaxis=dict(autorange="reversed"))
            st.plotly_chart(_plotly_defaults(fig, 380), width="stretch",
                            config={"displayModeBar": False})

    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

    # ── Extreme venues ──
    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown('<div class="nb-label">Most Batter-Friendly</div>',
                    unsafe_allow_html=True)
        top_bat = venues_df.nlargest(8, "bat_factor")[
            ["name","country","bat_factor","avg_first_inn_runs","total_matches"]]
        st.dataframe(top_bat.reset_index(drop=True), hide_index=True)
    with ec2:
        st.markdown('<div class="nb-label">Most Bowler-Friendly</div>',
                    unsafe_allow_html=True)
        top_bowl = venues_df.nsmallest(8, "bat_factor")[
            ["name","country","bat_factor","avg_first_inn_runs","total_matches"]]
        st.dataframe(top_bowl.reset_index(drop=True), hide_index=True)


# ═══════════════════════════════════════════════════════════
# PAGE 4 — PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════

elif "04" in page:
    st.markdown("""
    <div class="nb-page-header">
      <h2>Prediction Engine</h2>
      <p>Train GradientBoosting models · predict player performance at any venue</p>
    </div>""", unsafe_allow_html=True)

    from src.analytics.model import (
        train, predict_bat, predict_bowl,
        models_exist, model_metrics, feature_importance_df,
    )

    # ── Train ──
    st.markdown('<div class="nb-label">Model Training</div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns([1, 3])
    with tc1:
        if st.button("Train / Retrain Models"):
            with st.spinner("Training…"):
                meta = train(session, verbose=False)
            st.success("Models trained and saved.")
            st.cache_data.clear()

    with tc2:
        if models_exist():
            m = model_metrics()
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("Batting R²",   f"{m.get('bat_r2',0):.3f}")
            mc2.metric("Batting MAE",  f"{m.get('bat_mae',0):.1f} runs")
            mc3.metric("CV MAE",       f"{m.get('bat_cv_mae',0):.1f} runs")
            mc4.metric("Bowling R²",   f"{m.get('bowl_r2',0):.3f}")
        else:
            st.info("No trained model yet — click Train to build one.")

    if models_exist():
        st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

        # ── Feature Importance ──
        st.markdown('<div class="nb-label">What Drives Performance?</div>',
                    unsafe_allow_html=True)
        fi1, fi2 = st.columns(2)
        with fi1:
            fi_bat = feature_importance_df("bat")
            fig = go.Figure(go.Bar(
                y=fi_bat["feature"], x=fi_bat["importance"], orientation="h",
                marker=dict(color="#3A86FF", line=dict(color="#0D0D0D", width=2)),
            ))
            fig.update_layout(title="Batting Model — Feature Importance",
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(_plotly_defaults(fig, 400), width="stretch",
                            config={"displayModeBar": False})
        with fi2:
            fi_bowl = feature_importance_df("bowl")
            fig = go.Figure(go.Bar(
                y=fi_bowl["feature"], x=fi_bowl["importance"], orientation="h",
                marker=dict(color="#FF6B9D", line=dict(color="#0D0D0D", width=2)),
            ))
            fig.update_layout(title="Bowling Model — Feature Importance",
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(_plotly_defaults(fig, 400), width="stretch",
                            config={"displayModeBar": False})

        st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

        # ── Live Prediction ──
        st.markdown('<div class="nb-label">Live Prediction — Player × Venue</div>',
                    unsafe_allow_html=True)

        all_df    = all_players()
        venues_df = all_venues()

        pc1, pc2 = st.columns(2)
        with pc1:
            pred_player = st.selectbox("Player", all_df["name"].tolist(), key="pred_p")
        with pc2:
            pred_venue  = st.selectbox("Venue",  venues_df["name"].tolist(), key="pred_v")

        pp = _get_player(pred_player)
        bp = _get_bowl(int(pp["id"])) if pp else {}
        vv = venues_df[venues_df["name"] == pred_venue]
        vrow = vv.iloc[0].to_dict() if not vv.empty else {}

        if pp and vrow and st.button("Run Prediction"):
            player_feat = {
                "career_adj_avg":    pp.get("adj_average", 20),
                "career_adj_sr":     pp.get("adj_strike_rate", 120),
                "career_innings":    pp.get("innings", 20),
                "chase_avg":         20,
                "first_avg":         20,
                "chase_sr":          120,
                "batting_position":  4,
                "pp_sr":             pp.get("pp_sr", 130),
                "mid_sr":            pp.get("mid_sr", 125),
                "death_sr":          pp.get("death_sr", 135),
                "career_adj_econ":   bp.get("adj_economy", 8.5),
                "career_dot_pct":    bp.get("dot_pct", 33),
                "career_bowl_inn":   bp.get("bowl_inn", 20),
                "pp_economy":        bp.get("pp_economy", 8.0),
                "mid_economy":       bp.get("mid_economy", 8.0),
                "death_economy":     bp.get("death_economy", 9.5),
            }
            venue_feat = {
                "bat_factor":    vrow.get("bat_factor", 1.0),
                "boundary_rate": vrow.get("boundary_rate", 0.12),
                "pace_index":    vrow.get("pace_index", 0.5),
            }

            bat_pred  = predict_bat(player_feat, venue_feat)
            bowl_pred = predict_bowl(player_feat, venue_feat)

            st.markdown(f"""
            <div class="nb-card yellow" style="margin-bottom:1rem">
              <div class="c-label">Prediction — {pred_player} at {pred_venue}</div>
              <div class="c-val" style="font-size:1rem;margin-top:.5rem">
                Bat Factor at this venue: <strong>{venue_feat['bat_factor']:.3f}</strong>
                &nbsp;·&nbsp; {'Batter-friendly' if venue_feat['bat_factor'] > 1.05
                               else 'Bowler-friendly' if venue_feat['bat_factor'] < 0.95
                               else 'Neutral'}
              </div>
            </div>""", unsafe_allow_html=True)

            pr1, pr2, pr3, pr4, pr5 = st.columns(5)
            pr1.metric("Pred Runs (1st inn)",  bat_pred["first_innings"])
            pr2.metric("Pred Runs (chase)",    bat_pred["chasing"])
            pr3.metric("80% CI",
                       f"{bat_pred['ci_lo']}–{bat_pred['ci_hi']}")
            pr4.metric("Pred Economy",         bat_pred_e := bowl_pred["predicted_economy"])
            pr5.metric("Economy CI",
                       f"{bowl_pred['ci_lo']}–{bowl_pred['ci_hi']}")

            # historical vs predicted
            hist = player_venues(int(pp["id"]))
            hist_at = hist[hist["venue"] == pred_venue]

            if not hist_at.empty:
                h = hist_at.iloc[0]
                st.markdown(f"""
                <div class="nb-card green" style="margin-top:1rem">
                  <div class="c-label">Historical at this venue ({int(h['innings'])} innings)</div>
                  <div style="margin-top:.5rem;font-family:Space Mono;font-size:.8rem">
                    Actual avg: <strong>{h['average']:.1f}</strong> &nbsp;·&nbsp;
                    Predicted: <strong>{bat_pred['first_innings']}</strong> &nbsp;·&nbsp;
                    Residual: <strong>{(bat_pred['first_innings'] - float(h['average'])):.1f}</strong>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="nb-card orange" style="margin-top:1rem">
                  <div class="c-label">No historical data at this venue</div>
                  <div style="margin-top:.3rem;font-family:Space Mono;font-size:.78rem">
                    Prediction is purely model-based using career stats + venue difficulty.
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

        # ── Multi-player venue prediction table ──
        st.markdown('<div class="nb-label">Compare Multiple Players at One Venue</div>',
                    unsafe_allow_html=True)

        mp_players = st.multiselect(
            "Select players (max 8)",
            all_df["name"].tolist(),
            default=all_df["name"].head(4).tolist(),
            max_selections=8,
        )
        mp_venue = st.selectbox("Venue for comparison", venues_df["name"].tolist(),
                                 key="mp_venue")

        if mp_players and mp_venue and st.button("Compare at Venue"):
            vv2  = venues_df[venues_df["name"] == mp_venue].iloc[0].to_dict()
            vfeat = {"bat_factor":    vv2.get("bat_factor",1.0),
                     "boundary_rate": vv2.get("boundary_rate",0.12),
                     "pace_index":    vv2.get("pace_index",0.5)}

            rows = []
            for nm in mp_players:
                pp2 = _get_player(nm)
                if not pp2: continue
                bp2 = _get_bowl(int(pp2["id"]))
                pf  = {
                    "career_adj_avg":  pp2.get("adj_average",20),
                    "career_adj_sr":   pp2.get("adj_strike_rate",120),
                    "career_innings":  pp2.get("innings",20),
                    "chase_avg": 20, "first_avg": 20, "chase_sr": 120,
                    "batting_position": 4,
                    "pp_sr":    pp2.get("pp_sr",130),
                    "mid_sr":   pp2.get("mid_sr",125),
                    "death_sr": pp2.get("death_sr",135),
                    "career_adj_econ":  bp2.get("adj_economy",8.5),
                    "career_dot_pct":   bp2.get("dot_pct",33),
                    "career_bowl_inn":  bp2.get("bowl_inn",20),
                    "pp_economy":   bp2.get("pp_economy",8.0),
                    "mid_economy":  bp2.get("mid_economy",8.0),
                    "death_economy":bp2.get("death_economy",9.5),
                }
                bp_pred = predict_bat(pf, vfeat)
                bwl_pred = predict_bowl(pf, vfeat)
                rows.append({
                    "Player":         nm,
                    "Country":        pp2.get("country","—"),
                    "Career Adj Avg": round(pp2.get("adj_average") or 0, 1),
                    "Pred Runs (1st)":bp_pred["first_innings"],
                    "Pred Runs (Chase)":bp_pred["chasing"],
                    "80% CI":         f"{bp_pred['ci_lo']}–{bp_pred['ci_hi']}",
                    "Pred Economy":   bwl_pred["predicted_economy"],
                    "Bat Rating":     round(pp2.get("bat_rating") or 0, 1),
                })

            if rows:
                result_df = pd.DataFrame(rows).sort_values("Pred Runs (1st)", ascending=False)
                st.dataframe(result_df.reset_index(drop=True), hide_index=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Predicted (1st inn)",
                    x=result_df["Player"],
                    y=result_df["Pred Runs (1st)"],
                    marker=dict(color="#3A86FF", line=dict(color="#0D0D0D",width=2)),
                ))
                fig.add_trace(go.Bar(
                    name="Predicted (Chase)",
                    x=result_df["Player"],
                    y=result_df["Pred Runs (Chase)"],
                    marker=dict(color="#FF6B9D", line=dict(color="#0D0D0D",width=2)),
                ))
                fig.update_layout(
                    barmode="group",
                    title=f"Predicted Performance at {mp_venue}",
                )
                st.plotly_chart(_plotly_defaults(fig, 340), width="stretch",
                                config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────
# PAGE 5 — MATCHUP LAB
# ─────────────────────────────────────────────────────────
if page == "05  Matchup Lab":
    st.markdown('<h1 class="nb-title">Matchup Lab</h1>', unsafe_allow_html=True)
    st.markdown('<p class="nb-subtitle">Ball-by-ball batter vs bowler analysis — real delivery data, no estimates</p>',
                unsafe_allow_html=True)

    all_df   = all_players()
    names    = all_df["name"].tolist()

    tab_matchup, tab_bvall, tab_allvb = st.tabs([
        "Batter vs Bowler", "Batter vs All Bowlers", "Bowler vs All Batters"
    ])

    # ── TAB 1: specific matchup ──────────────────────────────────────────
    with tab_matchup:
        col1, col2 = st.columns(2)
        with col1:
            batter_name = st.selectbox("Batter", names, key="mu_bat")
        with col2:
            bowler_name = st.selectbox("Bowler", names, key="mu_bowl",
                                       index=min(1, len(names)-1))

        if batter_name and bowler_name and batter_name != bowler_name:
            b_row = all_df[all_df["name"] == batter_name].iloc[0]
            o_row = all_df[all_df["name"] == bowler_name].iloc[0]
            df = matchup_stats(int(b_row["id"]), int(o_row["id"]))

            if df.empty:
                st.info(f"No ball-by-ball data found for {batter_name} vs {bowler_name}.")
            else:
                balls      = len(df)
                runs       = int(df["bat_runs"].sum())
                dismissals = int(df["is_wicket"].sum())
                dots       = int(df["is_dot"].sum())
                fours      = int(df["is_boundary_4"].sum())
                sixes      = int(df["is_boundary_6"].sum())
                sr         = round(runs / balls * 100, 1) if balls else 0
                dot_pct    = round(dots / balls * 100, 1) if balls else 0
                boundary_pct = round((fours + sixes) / balls * 100, 1) if balls else 0

                # Stat cards
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                for col, label, val in [
                    (c1, "Balls",       balls),
                    (c2, "Runs",        runs),
                    (c3, "Dismissals",  dismissals),
                    (c4, "SR",          sr),
                    (c5, "Dot %",       f"{dot_pct}%"),
                    (c6, "Boundary %",  f"{boundary_pct}%"),
                ]:
                    col.markdown(f"""
                    <div class="nb-card" style="text-align:center;padding:.8rem">
                      <div style="font-size:1.6rem;font-weight:800">{val}</div>
                      <div style="font-size:.7rem;opacity:.6;font-family:'Space Mono'">{label}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

                # Dismissal breakdown
                if dismissals > 0:
                    dism_counts = df[df["is_wicket"]==1]["wicket_kind"].value_counts()
                    st.markdown(f"**Dismissals:** " +
                                ", ".join([f"{k} ×{v}" for k, v in dism_counts.items()]))

                # Phase breakdown
                phase_labels = {0: "Powerplay", 1: "Middle", 2: "Death"}
                phase_df = (df.groupby("phase")
                              .agg(balls=("bat_runs","count"),
                                   runs=("bat_runs","sum"),
                                   wkts=("is_wicket","sum"))
                              .reset_index())
                phase_df["SR"]   = (phase_df["runs"] / phase_df["balls"] * 100).round(1)
                phase_df["phase"] = phase_df["phase"].map(phase_labels)
                phase_df = phase_df.rename(columns={"balls":"Balls","runs":"Runs",
                                                     "wkts":"Wkts","phase":"Phase"})

                st.markdown("**Phase Breakdown**")
                st.dataframe(phase_df[["Phase","Balls","Runs","Wkts","SR"]],
                             hide_index=True, use_container_width=True)

                # Runs per ball over time (scatter)
                df_sorted = df.reset_index(drop=True)
                df_sorted["ball_num"] = range(1, len(df_sorted)+1)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_sorted["ball_num"], y=df_sorted["bat_runs"],
                    marker_color="#FFE500", name="Runs per ball"
                ))
                fig.add_hline(y=sr/100, line_dash="dash", line_color="#FF3B3B",
                              annotation_text=f"avg {sr/100:.2f} r/ball")
                fig.update_layout(
                    title=f"{batter_name} vs {bowler_name} — ball by ball",
                    xaxis_title="Delivery #", yaxis_title="Runs",
                    showlegend=False,
                )
                st.plotly_chart(_plotly_defaults(fig, 300), width="stretch",
                                config={"displayModeBar": False})

                # Matches where this matchup occurred
                match_summary = (df.groupby(["tournament","season","venue"])
                                   .agg(balls=("bat_runs","count"),
                                        runs=("bat_runs","sum"),
                                        wkts=("is_wicket","sum"))
                                   .reset_index()
                                   .sort_values("balls", ascending=False))
                st.markdown("**By Match Context**")
                st.dataframe(match_summary, hide_index=True, use_container_width=True)

    # ── TAB 2: batter vs all bowlers ─────────────────────────────────────
    with tab_bvall:
        batter_name2 = st.selectbox("Batter", names, key="bvall_bat")
        if batter_name2:
            b_row2 = all_df[all_df["name"] == batter_name2].iloc[0]
            df2 = batter_vs_all(int(b_row2["id"]))

            if df2.empty:
                st.info("No matchup data found.")
            else:
                df2["SR"]      = (df2["runs"] / df2["balls"] * 100).round(1)
                df2["dot_%"]   = (df2["dots"] / df2["balls"] * 100).round(1)
                df2["bdry_%"]  = (df2["boundaries"] / df2["balls"] * 100).round(1)

                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("**Struggles against (lowest SR, min 12 balls)**")
                    hard = df2[df2["balls"] >= 12].nsmallest(10, "SR")[
                        ["bowler","balls","runs","dismissals","SR","dot_%"]]
                    st.dataframe(hard, hide_index=True, use_container_width=True)
                with col_r:
                    st.markdown("**Dominates (highest SR, min 12 balls)**")
                    easy = df2[df2["balls"] >= 12].nlargest(10, "SR")[
                        ["bowler","balls","runs","dismissals","SR","dot_%"]]
                    st.dataframe(easy, hide_index=True, use_container_width=True)

                # Most dismissed by
                if df2["dismissals"].sum() > 0:
                    st.markdown("**Most dismissed by**")
                    dism = df2[df2["dismissals"] > 0].nlargest(10, "dismissals")[
                        ["bowler","balls","runs","dismissals","SR"]]
                    st.dataframe(dism, hide_index=True, use_container_width=True)

    # ── TAB 3: bowler vs all batters ─────────────────────────────────────
    with tab_allvb:
        bowler_name3 = st.selectbox("Bowler", names, key="allvb_bowl")
        if bowler_name3:
            o_row3 = all_df[all_df["name"] == bowler_name3].iloc[0]
            df3 = bowler_vs_all(int(o_row3["id"]))

            if df3.empty:
                st.info("No matchup data found.")
            else:
                df3["SR"]     = (df3["runs"] / df3["balls"] * 100).round(1)
                df3["dot_%"]  = (df3["dots"] / df3["balls"] * 100).round(1)
                df3["bdry_%"] = (df3["boundaries"] / df3["balls"] * 100).round(1)

                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("**Gets out easily (most dismissals, min 12 balls)**")
                    easy = df3[df3["balls"] >= 12].nlargest(10, "dismissals")[
                        ["batter","balls","runs","dismissals","SR","dot_%"]]
                    st.dataframe(easy, hide_index=True, use_container_width=True)
                with col_r:
                    st.markdown("**Struggles against (highest SR conceded, min 12 balls)**")
                    hard = df3[df3["balls"] >= 12].nlargest(10, "SR")[
                        ["batter","balls","runs","dismissals","SR","dot_%"]]
                    st.dataframe(hard, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────
# PAGE 6 — MATCH PREDICTOR (XI vs XI)
# ─────────────────────────────────────────────────────────
if page == "06  Match Predictor":
    st.markdown('<h1 class="nb-title">Match Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="nb-subtitle">Your XI vs Their XI at a venue — matchup matrix, key threats, predicted score</p>',
                unsafe_allow_html=True)

    all_df     = all_players()
    names      = all_df["name"].tolist()
    venues_df  = all_venues()
    venue_names = venues_df["name"].tolist() if not venues_df.empty else []

    # ── Team selection ─────────────────────────────────────────────────
    st.markdown("### Select Teams")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Your XI**")
        your_xi = []
        for i in range(11):
            label = f"{'Opener' if i<2 else 'Batter' if i<6 else 'All-rounder' if i<9 else 'Bowler'} #{i+1}"
            p = st.selectbox(label, ["— select —"] + names,
                             key=f"your_{i}", label_visibility="collapsed")
            if p != "— select —":
                your_xi.append(p)

    with col_b:
        st.markdown("**Opposition XI**")
        opp_xi = []
        for i in range(11):
            label = f"Opp #{i+1}"
            p = st.selectbox(label, ["— select —"] + names,
                             key=f"opp_{i}", label_visibility="collapsed")
            if p != "— select —":
                opp_xi.append(p)

    col_v, col_bat = st.columns([3, 1])
    with col_v:
        sel_venue = st.selectbox("Venue", venue_names, key="mp_venue") if venue_names else None
    with col_bat:
        bats_first = st.radio("Bats first", ["Your XI", "Opp XI"], key="mp_bat")

    if len(your_xi) < 5 or len(opp_xi) < 5:
        st.info("Select at least 5 players per side to run prediction.")
        st.stop()

    # ── Resolve IDs ────────────────────────────────────────────────────
    id_map = dict(zip(all_df["name"], all_df["id"]))

    your_batters  = your_xi if bats_first == "Your XI" else opp_xi
    opp_batters   = opp_xi  if bats_first == "Your XI" else your_xi
    your_bowlers  = [p for p in (opp_xi if bats_first == "Your XI" else your_xi)]
    opp_bowlers   = [p for p in (your_xi if bats_first == "Your XI" else opp_xi)]

    # Venue row
    vrow = venues_df[venues_df["name"] == sel_venue].iloc[0] if sel_venue and not venues_df.empty else None

    # ── Pitch summary ──────────────────────────────────────────────────
    if vrow is not None:
        bf  = vrow.get("bat_factor")
        pi  = vrow.get("pace_index")
        br  = vrow.get("boundary_rate")
        tags = _pitch_tags(bf, pi, br)
        tag_html = " ".join([
            f'<span style="background:{c};color:#0D0D0D;padding:.15rem .5rem;'
            f'border-radius:4px;font-family:Space Mono;font-size:.68rem;font-weight:700;'
            f'border:1.5px solid #0D0D0D">{t}</span>' for t, c in tags
        ])
        ground = GROUND_INFO.get(sel_venue, {})
        desc   = _pitch_label(bf, pi, br)
        notes  = ground.get("notes", "")
        st.markdown(f"""
        <div class="nb-card" style="padding:.8rem 1rem;margin:1rem 0">
          <b>{sel_venue}</b> — {tag_html}<br>
          <span style="font-size:.85rem">{desc} {notes}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Matchup matrix ─────────────────────────────────────────────────
    st.markdown("### Matchup Matrix — Batters vs Bowlers")
    st.caption("SR = strike rate in real ball-by-ball history. ⚠ = fewer than 6 balls faced. 🔴 = dismissed.")

    def _matchup_cell(bat_name, bowl_name):
        bid = id_map.get(bat_name)
        oid = id_map.get(bowl_name)
        if not bid or not oid:
            return "—"
        df = matchup_stats(int(bid), int(oid))
        if df.empty:
            return "No data"
        balls = len(df)
        runs  = int(df["bat_runs"].sum())
        wkts  = int(df["is_wicket"].sum())
        sr    = round(runs / balls * 100) if balls else 0
        flag  = "⚠ " if balls < 6 else ""
        out   = " 🔴" if wkts > 0 else ""
        return f"{flag}{runs}/{balls}b SR{sr}{out}"

    batting_team  = your_batters[:6]   # top 6 batters
    bowling_team  = your_bowlers[:5]   # top 5 bowlers from opp

    matrix_data = {}
    for bowler in bowling_team:
        matrix_data[bowler] = [_matchup_cell(b, bowler) for b in batting_team]

    matrix_df = pd.DataFrame(matrix_data, index=batting_team)
    matrix_df.index.name = "Batter ↓ / Bowler →"
    st.dataframe(matrix_df, use_container_width=True)

    st.markdown("---")

    # ── Key threats ────────────────────────────────────────────────────
    st.markdown("### Key Threats")
    col_t1, col_t2 = st.columns(2)

    def _threats(batting_team, bowling_team, label):
        rows = []
        for bowl in bowling_team:
            oid = id_map.get(bowl)
            if not oid: continue
            for bat in batting_team:
                bid = id_map.get(bat)
                if not bid: continue
                df = matchup_stats(int(bid), int(oid))
                if df.empty or len(df) < 6: continue
                balls = len(df)
                runs  = int(df["bat_runs"].sum())
                wkts  = int(df["is_wicket"].sum())
                sr    = round(runs / balls * 100)
                rows.append({"Bowler": bowl, "vs Batter": bat,
                             "Balls": balls, "SR": sr, "Wkts": wkts})
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("SR", ascending=False)

    with col_t1:
        st.markdown(f"**Their bowlers dangerous vs your batters**")
        threat1 = _threats(your_batters, opp_xi, "opp")
        if not threat1.empty:
            # Highlight dismissals
            danger = threat1[threat1["Wkts"] > 0].head(5)
            if not danger.empty:
                st.dataframe(danger, hide_index=True, use_container_width=True)
            else:
                st.dataframe(threat1.head(5), hide_index=True, use_container_width=True)
        else:
            st.info("Not enough head-to-head history.")

    with col_t2:
        st.markdown(f"**Your bowlers dangerous vs their batters**")
        threat2 = _threats(opp_batters, your_xi, "your")
        if not threat2.empty:
            danger2 = threat2[threat2["Wkts"] > 0].head(5)
            if not danger2.empty:
                st.dataframe(danger2, hide_index=True, use_container_width=True)
            else:
                st.dataframe(threat2.head(5), hide_index=True, use_container_width=True)
        else:
            st.info("Not enough head-to-head history.")

    st.markdown("---")

    # ── Venue advantage ────────────────────────────────────────────────
    if vrow is not None and not pd.isna(vrow.get("pace_index")):
        st.markdown("### Venue Advantage")
        pi = float(vrow["pace_index"])

        # Count pace vs spin bowlers per team (rough: we don't have bowling style in DB yet)
        # Use career bowling stats as proxy — high economy spinners vs pacers
        your_bowl_str  = ", ".join(your_xi[-4:]) if len(your_xi) >= 4 else "—"
        opp_bowl_str   = ", ".join(opp_xi[-4:])  if len(opp_xi)  >= 4 else "—"

        if pi > 0.65:
            advantage = "Pace-friendly pitch. Bowlers with pace and carry will be effective."
        elif pi < 0.35:
            advantage = "Spin-friendly pitch. Spin bowlers will get turn and slow it up."
        else:
            advantage = "Balanced pitch. Both pace and spin will be effective."

        bf_val = float(vrow.get("bat_factor", 1.0)) if pd.notna(vrow.get("bat_factor")) else 1.0
        if bf_val > 1.08:
            bat_adv = "Batting pitch — chasing team has a significant advantage (dew factor likely)."
        elif bf_val < 0.93:
            bat_adv = "Bowling pitch — setting a target and defending is the better strategy."
        else:
            bat_adv = "Neutral pitch — toss and conditions on the day will decide."

        st.markdown(f"""
        <div class="nb-card" style="padding:1rem">
          <div>🎯 <b>Pitch type:</b> {advantage}</div>
          <div style="margin-top:.5rem">🏏 <b>Toss strategy:</b> {bat_adv}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── GBM Score Prediction ────────────────────────────────────────────
    st.markdown("### Predicted Scores")
    st.caption("GBM model — venue-adjusted per-player run prediction, summed to team total.")

    vrow_dict = vrow.to_dict() if vrow is not None else {}
    venue_feat = {
        "bat_factor":    float(vrow_dict.get("bat_factor")    or 1.0),
        "boundary_rate": float(vrow_dict.get("boundary_rate") or 0.12),
        "pace_index":    float(vrow_dict.get("pace_index")    or 0.5),
    }

    def _predict_xi(player_names, is_chase):
        rows = []
        total = 0.0
        for pos, name in enumerate(player_names, 1):
            pp = _get_player(name)
            if not pp:
                rows.append({"#": pos, "Player": name, "Predicted Runs": "—", "CI": "—"})
                continue
            feat = {
                "career_adj_avg":   float(pp.get("adj_average")     or 20),
                "career_adj_sr":    float(pp.get("adj_strike_rate")  or 120),
                "career_innings":   int(pp.get("innings")            or 20),
                "chase_avg":        20.0,
                "first_avg":        float(pp.get("adj_average")      or 20),
                "chase_sr":         120.0,
                "batting_position": pos,
                "pp_sr":            float(pp.get("pp_sr")   or 130),
                "mid_sr":           float(pp.get("mid_sr")  or 125),
                "death_sr":         float(pp.get("death_sr")or 135),
            }
            try:
                p = predict_bat(feat, venue_feat, n_boot=100)
                pred = p["chasing"] if is_chase else p["first_innings"]
                ci   = f"{p['ci_lo']}–{p['ci_hi']}"
                total += pred
                rows.append({"#": pos, "Player": name,
                             "Predicted Runs": round(pred, 1), "CI (80%)": ci})
            except Exception:
                rows.append({"#": pos, "Player": name, "Predicted Runs": "—", "CI": "—"})
        return pd.DataFrame(rows), round(total, 1)

    batting_first_xi  = your_xi if bats_first == "Your XI" else opp_xi
    batting_second_xi = opp_xi  if bats_first == "Your XI" else your_xi
    first_label       = "Your XI" if bats_first == "Your XI" else "Opp XI"
    second_label      = "Opp XI"  if bats_first == "Your XI" else "Your XI"

    with st.spinner("Running predictions…"):
        df_first,  total_first  = _predict_xi(batting_first_xi,  is_chase=False)
        df_second, total_second = _predict_xi(batting_second_xi, is_chase=True)

    # Score cards
    sc1, sc2 = st.columns(2)
    winner = first_label if total_first > total_second else second_label
    margin = abs(round(total_first - total_second, 1))

    with sc1:
        colour = "#FFE500" if first_label == "Your XI" else "#FF6B9D"
        st.markdown(f"""
        <div class="nb-card" style="background:{colour};padding:1rem;text-align:center">
          <div style="font-family:Space Mono;font-size:.75rem;font-weight:700">{first_label} — BATTING FIRST</div>
          <div style="font-family:Space Grotesk;font-size:2.8rem;font-weight:900;line-height:1">{total_first}</div>
          <div style="font-size:.75rem">predicted runs</div>
        </div>""", unsafe_allow_html=True)
        st.dataframe(df_first, hide_index=True, use_container_width=True)

    with sc2:
        colour = "#FFE500" if second_label == "Your XI" else "#FF6B9D"
        st.markdown(f"""
        <div class="nb-card" style="background:{colour};padding:1rem;text-align:center">
          <div style="font-family:Space Mono;font-size:.75rem;font-weight:700">{second_label} — CHASING</div>
          <div style="font-family:Space Grotesk;font-size:2.8rem;font-weight:900;line-height:1">{total_second}</div>
          <div style="font-size:.75rem">predicted runs</div>
        </div>""", unsafe_allow_html=True)
        st.dataframe(df_second, hide_index=True, use_container_width=True)

    # Verdict
    if total_first > 0 and total_second > 0:
        diff = total_first - total_second
        win_prob = round(50 + min(diff * 0.6, 45), 1)
        if diff < 0:
            win_prob = round(100 - win_prob, 1)
        st.markdown(f"""
        <div class="nb-card" style="background:#0D0D0D;color:#FFE500;padding:1rem;
             text-align:center;margin-top:1rem">
          <span style="font-family:Space Mono;font-size:.8rem">MODEL VERDICT</span><br>
          <span style="font-family:Space Grotesk;font-size:1.6rem;font-weight:900">
            {winner} wins by ~{margin} runs
          </span><br>
          <span style="font-size:.8rem;color:#ccc">
            {first_label} win probability: {win_prob}%
            &nbsp;|&nbsp; {second_label}: {100-win_prob}%
          </span>
        </div>""", unsafe_allow_html=True)
        st.caption("⚠ Predictions are based on career stats + venue factors. T20 cricket is highly variable — treat as a guide, not a guarantee.")
