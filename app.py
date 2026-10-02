"""
Streamlit Cloud / Railway entry point.
Decompresses the bundled cricket.db.gz on first run, then loads the dashboard.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

DB_GZ    = ROOT / "data" / "cricket.db.gz"
WORK_DB  = Path("/tmp/cricket.db")   # the dashboard reads this copy

# ── Decompress bundled DB once per gz version (atomic, locked) ──
from src.db.unpack import ensure_db, is_current
if DB_GZ.exists() and not is_current(DB_GZ, WORK_DB):
    if True:
        import streamlit as st
        st.set_page_config(page_title="Cricket Analytics", page_icon="🏏", layout="centered")
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&display=swap');
        html,body,[data-testid="stAppViewContainer"],.main{
            background:#0D0D0D!important;color:#FFE500!important;
            font-family:'Space Mono',monospace!important;}
        p,span,div{color:#FFE500!important;}
        </style>""", unsafe_allow_html=True)
        with st.spinner("Unpacking database — one moment…"):
            ensure_db(DB_GZ, WORK_DB)
        st.rerun()

# ── Run the full dashboard ──
_dash = ROOT / "src" / "dashboard" / "app.py"
__file__ = str(_dash)   # the dashboard resolves ROOT and neo_theme.css from its own __file__, not from this wrapper's
exec(compile(_dash.read_text(), str(_dash), "exec"))
