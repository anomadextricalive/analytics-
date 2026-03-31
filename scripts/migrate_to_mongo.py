"""
Migrate all SQLite tables → MongoDB collections via PyMongo API connection.

Usage:
  python scripts/migrate_to_mongo.py --uri "mongodb+srv://user:pass@cluster.mongodb.net/" --db cricket_analytics
  python scripts/migrate_to_mongo.py --uri "mongodb://localhost:27017" --db cricket_analytics
  python scripts/migrate_to_mongo.py --uri "..." --db cricket_analytics --drop   # drop & recreate collections
  python scripts/migrate_to_mongo.py --uri "..." --db cricket_analytics --tables players matches  # specific tables only

Each SQLite table becomes a MongoDB collection with the same name.
SQLite integer primary keys are stored as "id" and MongoDB adds its own "_id".
Date/datetime columns are converted to Python datetime objects for proper BSON encoding.
"""

import sys
import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
from src.db.schema import get_engine

console = Console()

ALL_TABLES = [
    "players", "venues", "teams", "matches", "innings", "deliveries",
    "player_innings", "player_bowling_innings", "partnerships",
    "player_career_bat", "player_career_bowl",
    "player_phase_bat", "player_phase_bowl",
    "player_position_bat",
    "player_venue_bat", "player_venue_bowl",
    "player_chase_bat",
    "player_perf_by_opponent", "player_perf_by_season",
    "player_perf_by_team", "player_perf_by_result",
    "player_dismissal_analysis", "player_bowling_dismissal_analysis",
    "player_milestones", "player_of_match_awards",
    "player_fielding_stats", "venue_difficulty", "player_ratings",
]


def _coerce_row(row: dict) -> dict:
    """Convert non-BSON-safe types (date → datetime) in a row dict."""
    out = {}
    for k, v in row.items():
        if isinstance(v, datetime.date) and not isinstance(v, datetime.datetime):
            # BSON requires datetime, not date
            v = datetime.datetime(v.year, v.month, v.day)
        out[k] = v
    return out


def migrate_table(engine, mongo_db, table_name: str, drop: bool, batch_size: int = 500) -> int:
    """Read one SQLite table and upsert into MongoDB. Returns row count."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table_name}"))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result]
    except Exception as e:
        console.print(f"  [yellow]⚠ Skipping {table_name}: {e}[/yellow]")
        return 0

    if not rows:
        console.print(f"  [dim]{table_name}: empty, skipped[/dim]")
        return 0

    collection = mongo_db[table_name]

    if drop:
        collection.drop()

    # Insert in batches
    total = len(rows)
    for i in range(0, total, batch_size):
        batch = [_coerce_row(r) for r in rows[i : i + batch_size]]
        collection.insert_many(batch, ordered=False)

    return total


@click.command()
@click.option("--uri",    required=True, help="MongoDB connection URI (mongodb:// or mongodb+srv://)")
@click.option("--db",     required=True, help="MongoDB database name to write into")
@click.option("--drop",   is_flag=True,  default=False, help="Drop existing collections before inserting")
@click.option("--tables", multiple=True, help="Specific tables to migrate (default: all)")
@click.option("--batch-size", default=500, show_default=True, help="Insert batch size")
def main(uri: str, db: str, drop: bool, tables: tuple, batch_size: int):
    """Migrate cricket.db (SQLite) → MongoDB."""
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure
    except ImportError:
        console.print("[red]pymongo not installed. Run: pip install pymongo[/red]")
        sys.exit(1)

    # --- Connect SQLite ---
    if not DB_PATH.exists():
        console.print(f"[red]SQLite DB not found at {DB_PATH}[/red]")
        sys.exit(1)

    sqlite_engine = get_engine()
    inspector = inspect(sqlite_engine)
    existing_tables = set(inspector.get_table_names())
    console.print(f"[green]✓[/green] SQLite connected — {len(existing_tables)} tables found")

    # --- Connect MongoDB ---
    console.print(f"Connecting to MongoDB...")
    try:
        import certifi
        client = MongoClient(uri, serverSelectionTimeoutMS=10_000, tlsCAFile=certifi.where())
        client.admin.command("ping")
    except ConnectionFailure as e:
        console.print(f"[red]MongoDB connection failed: {e}[/red]")
        sys.exit(1)

    mongo_db = client[db]
    console.print(f"[green]✓[/green] MongoDB connected — database: [bold]{db}[/bold]")

    # --- Determine tables ---
    target = list(tables) if tables else ALL_TABLES
    target = [t for t in target if t in existing_tables]
    missing = [t for t in (list(tables) if tables else ALL_TABLES) if t not in existing_tables]
    if missing:
        console.print(f"[yellow]Tables not in SQLite (skipped): {missing}[/yellow]")

    if drop:
        console.print(f"[yellow]--drop flag set: existing collections will be dropped[/yellow]")

    # --- Migrate ---
    total_rows = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Migrating tables...", total=len(target))
        for table in target:
            progress.update(task, description=f"[cyan]{table}[/cyan]")
            n = migrate_table(sqlite_engine, mongo_db, table, drop=drop, batch_size=batch_size)
            total_rows += n
            console.print(f"  [green]✓[/green] {table}: {n:,} rows")
            progress.advance(task)

    console.print(f"\n[bold green]Done![/bold green] {len(target)} collections, {total_rows:,} total documents → MongoDB [bold]{db}[/bold]")


if __name__ == "__main__":
    main()
