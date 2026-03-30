"""
Streamlit Cloud / Railway entry point.
Decompresses the bundled cricket.db.gz on first run, then loads the dashboard.
"""
import sys
import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

DB_PATH  = ROOT / "data" / "cricket.db"
DB_GZ    = ROOT / "data" / "cricket.db.gz"

# ── Decompress bundled DB if not yet extracted ──
if not DB_PATH.exists() or DB_PATH.stat().st_size < 5_000_000:
    if DB_GZ.exists():
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
            with gzip.open(DB_GZ, "rb") as f_in, open(DB_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        st.rerun()

# ── Run the full dashboard ──
_src = (ROOT / "src" / "dashboard" / "app.py").read_text()
exec(compile(_src, str(ROOT / "src" / "dashboard" / "app.py"), "exec"))
