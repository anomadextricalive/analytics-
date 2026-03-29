"""
Quick database inspection utility.
Shows every table, row count, and sample rows.
Run: python scripts/inspect_db.py
"""
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
from src.db.schema import get_engine, Base

console = Console()


def main():
    engine = get_engine(DB_PATH)
    Base.metadata.create_all(engine)   # ensure tables exist

    for tbl in Base.metadata.sorted_tables:
        try:
            df = pd.read_sql(f"SELECT * FROM {tbl.name} LIMIT 5", engine)
            n  = pd.read_sql(f"SELECT COUNT(*) AS n FROM {tbl.name}", engine).iloc[0]["n"]
        except Exception as e:
            console.print(f"[red]{tbl.name}:[/red] {e}")
            continue

        console.print(f"\n[bold cyan]{tbl.name}[/bold cyan]  ({n:,} rows)")
        if df.empty:
            console.print("  [dim](empty)[/dim]")
            continue

        t = Table(show_header=True, header_style="bold")
        for col in df.columns:
            t.add_column(col, overflow="fold", max_width=20)
        for _, row in df.iterrows():
            t.add_row(*[str(v)[:20] for v in row.values])
        console.print(t)


if __name__ == "__main__":
    main()
