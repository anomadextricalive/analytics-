"""
Streamlit Cloud entry point — delegates to src/dashboard/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Execute the main dashboard (exec preserves the Streamlit context)
_app = (ROOT / "src" / "dashboard" / "app.py").read_text()
exec(compile(_app, str(ROOT / "src" / "dashboard" / "app.py"), "exec"))
