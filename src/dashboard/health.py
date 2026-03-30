"""
Neobrutalist backend health dashboard.
Run: streamlit run src/dashboard/health.py
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import DB_PATH
from src.db.schema import get_engine, Base

st.set_page_config(
    page_title="CRICKET DB // HEALTH",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# NEOBRUTALIST CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Space+Grotesk:wght@400;500;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.main, section.main {
    background-color: #FFFCF2 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: #0D0D0D !important;
}

/* kill all streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"]  { display: none !important; }

[data-testid="block-container"] {
    padding: 0 !important;
    max-width: 100% !important;
}
[data-testid="stAppViewBlockContainer"] {
    padding: 0 2.5rem 3rem 2.5rem !important;
    max-width: 100% !important;
}

/* ── TOP HEADER BAR ── */
.nb-topbar {
    background: #0D0D0D;
    padding: 0;
    margin: 0 -2.5rem 2.5rem -2.5rem;
    border-bottom: 4px solid #0D0D0D;
    display: flex;
    align-items: stretch;
    height: 72px;
    overflow: hidden;
}
.nb-topbar-title {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 2rem;
    flex: 1;
}
.nb-topbar-title h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #FFFCF2;
    letter-spacing: -0.02em;
    text-transform: uppercase;
}
.nb-topbar-title .glyph {
    font-size: 1.6rem;
    line-height: 1;
}
.nb-topbar-meta {
    display: flex;
    align-items: center;
    gap: 0;
    margin-left: auto;
}
.nb-topbar-tag {
    height: 72px;
    display: flex;
    align-items: center;
    padding: 0 1.4rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #0D0D0D;
    background: #FFE500;
    border-left: 4px solid #0D0D0D;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    white-space: nowrap;
}
.nb-topbar-tag.blue  { background: #3A86FF; color: #fff; }
.nb-topbar-tag.green { background: #06D6A0; color: #0D0D0D; }

/* ── SECTION LABEL ── */
.nb-label {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #0D0D0D;
    background: #FFE500;
    border: 2.5px solid #0D0D0D;
    padding: 3px 10px;
    margin-bottom: 1.1rem;
}
.nb-divider {
    height: 3px;
    background: #0D0D0D;
    margin: 2.2rem 0;
}

/* ── Force all text dark ── */
.main p, .main span, .main div, .main label,
.main li, .main a, .main strong {
    color: #0D0D0D !important;
}

/* ── STAT CARDS ── */
.nb-card {
    background: #FFFCF2;
    border: 3px solid #0D0D0D;
    box-shadow: 5px 5px 0 #0D0D0D;
    padding: 1.1rem 1.3rem 1rem;
    height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform .12s, box-shadow .12s;
}
.nb-card:hover {
    transform: translate(-2px,-2px);
    box-shadow: 7px 7px 0 #0D0D0D;
}
.nb-card.yellow { background: #FFE500; }
.nb-card.blue   { background: #3A86FF; }
.nb-card.green  { background: #06D6A0; }
.nb-card.pink   { background: #FF6B9D; }
.nb-card.orange { background: #FF8C42; }
.nb-card.purple { background: #9B5DE5; }
.nb-card.black  { background: #0D0D0D; }
.nb-card.blue  *, .nb-card.purple *, .nb-card.pink * { color: #FFFCF2 !important; }
.nb-card.black  * { color: #FFE500 !important; }
.nb-card.black .c-label { color: #aaa !important; }

.nb-card .c-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #0D0D0D !important;
    opacity: 0.55;
}

.nb-card .c-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.03em;
    color: #0D0D0D !important;
}
.nb-card .c-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #555 !important;
    letter-spacing: 0.06em;
}

/* ── TABLE ── */
.nb-table {
    width: 100%;
    border-collapse: collapse;
    border: 3px solid #0D0D0D;
    box-shadow: 5px 5px 0 #0D0D0D;
    font-family: 'Space Mono', monospace;
    font-size: 0.76rem;
    background: #FFFCF2;
}
.nb-table thead tr { background: #0D0D0D; }
.nb-table th {
    color: #FFE500 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.6rem;
    font-weight: 700;
    padding: 0.65rem 0.9rem;
    text-align: left;
    border-right: 1px solid #333;
    white-space: nowrap;
}
.nb-table td {
    padding: 0.55rem 0.9rem;
    color: #0D0D0D !important;
    border-bottom: 2px solid #0D0D0D;
    border-right: 1px solid #e0ddd4;
    vertical-align: middle;
}
.nb-table tr:nth-child(even) td { background: #F5F2E8; }
.nb-table tr:hover td { background: #FFF5B0; }
.nb-table td:first-child { font-weight: 700; color: #0D0D0D !important; }

/* ── PILLS ── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border: 2px solid #0D0D0D;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    white-space: nowrap;
    color: #0D0D0D !important;
}
.pill.ok     { background: #06D6A0; color: #0D0D0D !important; }
.pill.warn   { background: #FFE500; color: #0D0D0D !important; }
.pill.empty  { background: #FF3B5C; color: #fff !important; }
.pill.low    { background: #FF8C42; color: #0D0D0D !important; }

/* ── PROGRESS BAR ── */
.nb-prog-wrap {
    border: 2px solid #0D0D0D;
    background: #E8E4D8;
    height: 14px;
    width: 100%;
}
.nb-prog-fill {
    height: 100%;
    background: #FFE500;
    border-right: 2px solid #0D0D0D;
    min-width: 2px;
}

/* ── STAGE CHECKLIST ── */
.stage-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0.75rem 1rem;
    border: 2.5px solid #0D0D0D;
    margin-bottom: -2.5px;
    background: #FFFCF2;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: #0D0D0D !important;
}
.stage-row.done  { background: #E8FFF6; }
.stage-row.pend  { background: #FFF4F4; }
.stage-row:hover { background: #FFF5B0; z-index: 1; position: relative; }
.stage-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    color: #666 !important;
    min-width: 22px;
}
.stage-icon { font-size: 1.1rem; min-width: 22px; text-align: center; }
.stage-name { flex: 1; font-weight: 700; color: #0D0D0D !important; }
.stage-desc {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #666 !important;
    flex: 2;
}

/* ── CMD BOX ── */
.nb-cmd {
    background: #0D0D0D;
    color: #06D6A0;
    border: 3px solid #0D0D0D;
    box-shadow: 5px 5px 0 #0D0D0D;
    padding: 1.2rem 1.4rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.76rem;
    line-height: 2;
}
.nb-cmd .cmd-section {
    color: #FFE500;
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 1rem;
    margin-bottom: 0.1rem;
    display: block;
}
.nb-cmd .cmd-section:first-child { margin-top: 0; }
.nb-cmd code { color: #fff; background: transparent; }

/* ── YEAR CHART ── */
.stPlotlyChart > div { border: 3px solid #0D0D0D !important; box-shadow: 5px 5px 0 #0D0D0D; }

/* ── BUTTON ── */
.stButton button {
    background: #FFE500 !important;
    color: #0D0D0D !important;
    border: 3px solid #0D0D0D !important;
    box-shadow: 4px 4px 0 #0D0D0D !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.2rem !important;
    transition: all .1s !important;
}
.stButton button:hover {
    transform: translate(-2px,-2px) !important;
    box-shadow: 6px 6px 0 #0D0D0D !important;
    background: #fff !important;
}
.stButton button:active {
    transform: translate(2px,2px) !important;
    box-shadow: 2px 2px 0 #0D0D0D !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DB connection
# ─────────────────────────────────────────────
@st.cache_resource
def get_engine_cached():
    return get_engine(DB_PATH)

engine = get_engine_cached()

def sql(q, params=None):
    try:    return pd.read_sql(q, engine, params=params)
    except: return pd.DataFrame()

def scalar(q, default=0):
    df = sql(q)
    return df.iloc[0, 0] if not df.empty else default

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
db_exists   = DB_PATH.exists()
db_mb       = round(DB_PATH.stat().st_size / 1_048_576, 1) if db_exists else 0
now_str     = datetime.now().strftime("%d %b %Y  //  %H:%M:%S")

n_matches   = scalar("SELECT COUNT(*) FROM matches")
n_players   = scalar("SELECT COUNT(*) FROM players")
n_deliveries= scalar("SELECT COUNT(*) FROM deliveries")
n_innings   = scalar("SELECT COUNT(*) FROM innings")
n_venues    = scalar("SELECT COUNT(*) FROM venues")
n_ratings   = scalar("SELECT COUNT(*) FROM player_ratings")
n_milestones= scalar("SELECT COUNT(*) FROM player_milestones")
n_pom       = scalar("SELECT COUNT(*) FROM player_of_match_awards")

# ─────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="nb-topbar">
  <div class="nb-topbar-title">
    <span class="glyph">🏏</span>
    <h1>Cricket DB &nbsp;//&nbsp; Backend Health Monitor</h1>
  </div>
  <div class="nb-topbar-meta">
    <div class="nb-topbar-tag green">T20 Only</div>
    <div class="nb-topbar-tag blue">SQLite · {db_mb} MB</div>
    <div class="nb-topbar-tag">{now_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# OVERVIEW CARDS
# ─────────────────────────────────────────────
st.markdown('<div class="nb-label">01 — Overview</div>', unsafe_allow_html=True)

cards = [
    ("yellow",  "Matches",     f"{n_matches:,}",    "T20 only · 2000–2025"),
    ("blue",    "Players",     f"{n_players:,}",    "unique cricsheet keys"),
    ("green",   "Deliveries",  f"{n_deliveries:,}", "ball-by-ball rows"),
    ("orange",  "Innings",     f"{n_innings:,}",    "batting innings"),
    ("pink",    "Venues",      f"{n_venues:,}",     "grounds indexed"),
    ("purple",  "Rated",       f"{n_ratings:,}",    "player-tournament scores"),
    ("black",   "Milestones",  f"{n_milestones:,}", "50s · 100s · 4wkts · ducks"),
    ("",        "PoM Awards",  f"{n_pom:,}",        "player of match records"),
]

cols = st.columns(8)
for col, (colour, label, value, sub) in zip(cols, cards):
    col.markdown(f"""
    <div class="nb-card {colour}">
      <div class="c-label">{label}</div>
      <div class="c-value">{value}</div>
      <div class="c-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABLE AUDIT
# ─────────────────────────────────────────────
st.markdown('<div class="nb-label">02 — Table Audit</div>', unsafe_allow_html=True)

TABLES = [
    ("players",                       "Core",      "Player registry"),
    ("teams",                         "Core",      "Team registry"),
    ("venues",                        "Core",      "Ground/venue registry"),
    ("matches",                       "Core",      "Match metadata + outcome"),
    ("innings",                       "Core",      "Innings totals"),
    ("deliveries",                    "Raw",       "Ball-by-ball (main fact table)"),
    ("partnerships",                  "Raw",       "Wicket partnerships"),
    ("player_innings",                "Agg",       "Per-batter per-innings summary"),
    ("player_bowling_innings",        "Agg",       "Per-bowler per-innings summary"),
    ("player_career_bat",             "Derived",   "Career batting aggregate"),
    ("player_career_bowl",            "Derived",   "Career bowling aggregate"),
    ("player_position_bat",           "Derived",   "Stats by batting position"),
    ("player_chase_bat",              "Derived",   "Chase vs first-innings split"),
    ("player_venue_bat",              "Derived",   "Per-ground batting stats"),
    ("player_venue_bowl",             "Derived",   "Per-ground bowling stats"),
    ("player_perf_by_opponent",       "Derived",   "Stats vs each opponent"),
    ("player_perf_by_season",         "Derived",   "Season-by-season breakdown"),
    ("player_perf_by_team",           "Derived",   "Stats per franchise/team"),
    ("player_perf_by_result",         "Derived",   "Won / lost / no-result split"),
    ("player_dismissal_analysis",     "Derived",   "How dismissed (batting)"),
    ("player_bowling_dismissal_analysis","Derived","How wickets taken (bowling)"),
    ("player_milestones",             "Derived",   "50s, 100s, 4wkt, ducks — logged"),
    ("player_of_match_awards",        "Derived",   "PoM award registry"),
    ("player_fielding_stats",         "Derived",   "Catches / run-outs / stumpings"),
    ("venue_difficulty",              "Model",     "Bayesian pitch/bat-factor"),
    ("player_ratings",                "Model",     "0–100 composite ratings"),
]

counts = {t: scalar(f"SELECT COUNT(*) FROM {t}") for t, *_ in TABLES}
max_c   = max(counts.values()) if counts else 1

col_left, col_right = st.columns(2)

def _pill(n):
    if n == 0:    return '<span class="pill empty">● Empty</span>'
    if n < 50:    return '<span class="pill warn">◑ Low</span>'
    return             '<span class="pill ok">● OK</span>'

def _bar(n):
    pct = max(2, int(n / max_c * 100))
    return f'<div class="nb-prog-wrap"><div class="nb-prog-fill" style="width:{pct}%"></div></div>'

for idx, (col, half) in enumerate([(col_left, TABLES[:13]), (col_right, TABLES[13:])]):
    rows = ""
    for tbl, kind, desc in half:
        n = counts[tbl]
        rows += f"""
        <tr>
          <td>{tbl}</td>
          <td style="text-align:center;opacity:.5;font-size:.65rem">{kind}</td>
          <td style="text-align:right;font-weight:700">{n:,}</td>
          <td style="min-width:100px">{_bar(n)}</td>
          <td>{_pill(n)}</td>
        </tr>"""
    with col:
        st.markdown(f"""
        <table class="nb-table">
          <thead><tr>
            <th>Table</th><th>Type</th>
            <th style="text-align:right">Rows</th>
            <th>Fill</th><th>Status</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOURNAMENT COVERAGE
# ─────────────────────────────────────────────
st.markdown('<div class="nb-label">03 — Tournament Coverage</div>', unsafe_allow_html=True)

tourn_df = sql("""
    SELECT tournament,
           COUNT(*)                AS matches,
           MIN(match_date)         AS first,
           MAX(match_date)         AS last,
           COUNT(DISTINCT venue_id) AS venues,
           COUNT(DISTINCT team1_id) AS teams
    FROM matches GROUP BY tournament ORDER BY matches DESC
""")

if tourn_df.empty:
    st.markdown('<div class="nb-card orange"><div class="c-label">No data yet</div><div class="c-value" style="font-size:1rem">Run the pipeline first</div></div>', unsafe_allow_html=True)
else:
    total = tourn_df["matches"].sum()
    rows = ""
    for _, r in tourn_df.iterrows():
        pct = int(r["matches"] / total * 100)
        bar = f'<div class="nb-prog-wrap"><div class="nb-prog-fill" style="width:{max(2,pct)}%;background:#3A86FF"></div></div>'
        rows += f"""<tr>
          <td>{r['tournament']}</td>
          <td style="text-align:right;font-weight:800">{int(r['matches']):,}</td>
          <td>{r.get('first','—')}</td><td>{r.get('last','—')}</td>
          <td style="text-align:right">{int(r.get('venues',0))}</td>
          <td style="text-align:right">{int(r.get('teams',0))}</td>
          <td style="min-width:120px">{bar} <span style="font-size:.62rem;opacity:.6">{pct}%</span></td>
        </tr>"""
    st.markdown(f"""
    <table class="nb-table">
      <thead><tr>
        <th>Tournament</th><th style="text-align:right">Matches</th>
        <th>First Match</th><th>Last Match</th>
        <th style="text-align:right">Venues</th>
        <th style="text-align:right">Teams</th>
        <th>Share</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)

st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# YEAR CHART
# ─────────────────────────────────────────────
st.markdown('<div class="nb-label">04 — Matches per Year</div>', unsafe_allow_html=True)

year_df = sql("""
    SELECT strftime('%Y', match_date) AS year, COUNT(*) AS matches
    FROM matches WHERE match_date IS NOT NULL
    GROUP BY year ORDER BY year
""")

if not year_df.empty:
    fig = go.Figure(go.Bar(
        x=year_df["year"], y=year_df["matches"],
        marker=dict(color="#FFE500", line=dict(color="#0D0D0D", width=2.5)),
        text=year_df["matches"], textposition="outside",
        textfont=dict(family="Space Mono", size=10, color="#0D0D0D"),
    ))
    fig.update_layout(
        plot_bgcolor="#FFFCF2", paper_bgcolor="#FFFCF2",
        font=dict(family="Space Grotesk", color="#0D0D0D", size=11),
        margin=dict(l=0, r=0, t=16, b=0), height=240,
        xaxis=dict(showgrid=False, linecolor="#0D0D0D", linewidth=2.5,
                   tickfont=dict(family="Space Mono", size=10)),
        yaxis=dict(showgrid=True, gridcolor="#E8E4D8",
                   linecolor="#0D0D0D", linewidth=2.5,
                   tickfont=dict(family="Space Mono", size=10)),
        bargap=0.25,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
else:
    st.markdown('<p style="font-family:Space Mono;font-size:.8rem;opacity:.5">No year data yet — ingest data first.</p>', unsafe_allow_html=True)

st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PIPELINE STAGES
# ─────────────────────────────────────────────
st.markdown('<div class="nb-label">05 — Pipeline Stage Checklist</div>', unsafe_allow_html=True)

raw_dirs_exist = any(Path("data/raw").glob("*/")) if Path("data/raw").exists() else False
stages = [
    ("01", "Download",        "data/raw populated with .json files",    raw_dirs_exist),
    ("02", "Ingest matches",  "matches table > 0",                      n_matches > 0),
    ("03", "Ingest deliveries","deliveries table > 0",                  n_deliveries > 0),
    ("04", "Venue factors",   "venue_difficulty computed",
                               scalar("SELECT COUNT(*) FROM venue_difficulty") > 0),
    ("05", "Career metrics",  "player_career_bat populated",
                               scalar("SELECT COUNT(*) FROM player_career_bat") > 0),
    ("06", "Chase splits",    "player_chase_bat populated",
                               scalar("SELECT COUNT(*) FROM player_chase_bat") > 0),
    ("07", "Perf breakdowns", "by-opponent / season / team / result",
                               scalar("SELECT COUNT(*) FROM player_perf_by_opponent") > 0),
    ("08", "Dismissal analysis","bat + bowl dismissal tables",
                               scalar("SELECT COUNT(*) FROM player_dismissal_analysis") > 0),
    ("09", "Milestones",      "50s, 100s, 4-wkt hauls, ducks logged",
                               scalar("SELECT COUNT(*) FROM player_milestones") > 0),
    ("10", "Ratings",         "player_ratings built (0–100 normalised)", n_ratings > 0),
]

done_count = sum(1 for *_, d in stages if d)

col_stages, col_summary = st.columns([3, 1])
with col_stages:
    html = ""
    for num, name, desc, done in stages:
        icon  = "✓" if done else "○"
        cls   = "done" if done else "pend"
        pill  = '<span class="pill ok">Done</span>' if done else '<span class="pill empty">Pending</span>'
        html += f"""
        <div class="stage-row {cls}">
          <span class="stage-num">{num}</span>
          <span class="stage-icon">{icon}</span>
          <span class="stage-name">{name}</span>
          <span class="stage-desc">{desc}</span>
          {pill}
        </div>"""
    st.markdown(html, unsafe_allow_html=True)

with col_summary:
    pct = int(done_count / len(stages) * 100)
    st.markdown(f"""
    <div class="nb-card yellow" style="height:auto;padding:1.4rem;gap:1rem">
      <div class="c-label">Pipeline Progress</div>
      <div class="c-value">{done_count}/{len(stages)}</div>
      <div class="nb-prog-wrap" style="height:20px">
        <div class="nb-prog-fill" style="width:{pct}%;background:#0D0D0D"></div>
      </div>
      <div class="c-sub">{pct}% complete</div>
    </div>

    <div style="height:12px"></div>

    <div class="nb-card {'green' if n_matches>0 else 'pink'}" style="height:auto;padding:1.4rem">
      <div class="c-label">DB File</div>
      <div class="c-value" style="font-size:1.4rem">{db_mb} MB</div>
      <div class="c-sub" style="word-break:break-all;margin-top:.4rem">data/cricket.db</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COMMAND REFERENCE
# ─────────────────────────────────────────────
st.markdown('<div class="nb-label">06 — Command Reference</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    <div class="nb-cmd">
      <span class="cmd-section">Full pipeline (first run)</span>
      python scripts/pipeline.py all

      <span class="cmd-section">Individual stages</span>
      python scripts/pipeline.py download
      python scripts/pipeline.py ingest
      python scripts/pipeline.py venue
      python scripts/pipeline.py metrics
      python scripts/pipeline.py ratings

      <span class="cmd-section">Selective download</span>
      python scripts/pipeline.py download \\
        --tournaments ipl psl bbl cpl
    </div>""", unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="nb-cmd">
      <span class="cmd-section">Audit DB directly</span>
      python scripts/inspect_db.py
      sqlite3 data/cricket.db ".tables"
      sqlite3 data/cricket.db \\
        "SELECT COUNT(*) FROM deliveries;"

      <span class="cmd-section">Player queries (CLI)</span>
      python scripts/query.py search "Kohli"
      python scripts/query.py profile "V Kohli"
      python scripts/query.py compare \\
        "V Kohli" "RG Sharma"
      python scripts/query.py leaderboard \\
        --type bat --top 20

      <span class="cmd-section">Main analytics dashboard</span>
      streamlit run src/dashboard/app.py
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# REFRESH
# ─────────────────────────────────────────────
col_btn, col_hint = st.columns([1, 4])
with col_btn:
    if st.button("↺  Refresh Data"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
with col_hint:
    st.markdown("""
    <p style="font-family:'Space Mono',monospace;font-size:.68rem;
              opacity:.5;padding-top:.6rem;line-height:1.8">
      Dashboard reads live from data/cricket.db — hit Refresh after each pipeline stage.<br>
      For DB Browser GUI audit: download <strong>DB Browser for SQLite</strong> → File → Open → data/cricket.db
    </p>""", unsafe_allow_html=True)
