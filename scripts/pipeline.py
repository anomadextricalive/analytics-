"""
Master pipeline script — run all stages end-to-end.

Usage:
  python scripts/pipeline.py --help
  python scripts/pipeline.py download            # download all tournaments
  python scripts/pipeline.py ingest              # parse JSON → DB
  python scripts/pipeline.py venue               # compute pitch factors
  python scripts/pipeline.py metrics             # rebuild aggregate tables
  python scripts/pipeline.py ratings             # compute player ratings
  python scripts/pipeline.py all                 # run everything in order

  # Selective tournament download:
  python scripts/pipeline.py download --tournaments t20i_male ipl psl

  # Only process specific tournaments in ratings:
  python scripts/pipeline.py ratings --tournament ipl
"""

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, DOWNLOADS, DB_PATH
from src.db.schema import init_db, get_engine
from src.ingest.downloader import download_all
from src.ingest.parser import ingest_directory
from src.analytics.pitch import compute_venue_factors
from src.analytics.metrics import rebuild_all_metrics
from src.analytics.rating import rebuild_ratings
from src.analytics.similarity import build_similarity, build_form

console = Console()


@click.group()
def cli():
    """Cricket analytics pipeline."""
    pass


@cli.command()
@click.option("--tournaments", "-t", multiple=True,
              help="Specific tournaments to download (default: all)")
@click.option("--force", is_flag=True, help="Re-download even if file exists")
def download(tournaments, force):
    """Download and extract cricsheet data."""
    t = list(tournaments) if tournaments else None
    console.print("[bold]Downloading data from cricsheet.org…[/bold]")
    download_all(t, force=force)
    console.print("[green]Download complete.[/green]")


@cli.command()
@click.option("--tournaments", "-t", multiple=True)
@click.option("--verbose", "-v", is_flag=True)
def ingest(tournaments, verbose):
    """Parse JSON files into the database."""
    console.print("[bold]Initialising database…[/bold]")
    engine = init_db(DB_PATH)
    session = Session(engine)

    targets = list(tournaments) if tournaments else list(DOWNLOADS.keys())
    total_inserted = 0

    for name in targets:
        d = DATA_RAW / name
        if not d.exists():
            console.print(f"[yellow]Skip[/yellow] {name} — not downloaded yet")
            continue
        console.print(f"\n[cyan]Ingesting[/cyan] {name}…")
        t0 = time.time()
        stats = ingest_directory(session, d, tournament=name, verbose=verbose)
        elapsed = time.time() - t0
        console.print(
            f"  inserted={stats['inserted']} skipped={stats['skipped']} "
            f"errors={stats['errors']}  ({elapsed:.1f}s)"
        )
        total_inserted += stats["inserted"]

    session.close()
    console.print(f"\n[green]Ingest complete. Total new matches: {total_inserted}[/green]")


@cli.command()
def venue():
    """Compute venue difficulty / pitch factors."""
    engine = get_engine(DB_PATH)
    session = Session(engine)
    console.print("[bold]Computing venue factors…[/bold]")
    df = compute_venue_factors(session)
    console.print(f"[green]Done. Computed factors for {len(df)} venues.[/green]")
    session.close()


@cli.command()
def metrics():
    """Rebuild all aggregated metric tables."""
    engine = get_engine(DB_PATH)
    session = Session(engine)
    console.print("[bold]Rebuilding metric tables…[/bold]")
    rebuild_all_metrics(session)
    session.close()


@cli.command()
@click.option("--tournament", "-t", default="ALL")
def ratings(tournament):
    """Compute player ratings."""
    engine = get_engine(DB_PATH)
    session = Session(engine)
    console.print(f"[bold]Computing ratings for tournament={tournament}…[/bold]")

    # Rebuild for ALL plus the specific tournament
    for t in (["ALL"] + ([tournament] if tournament != "ALL" else [])):
        rebuild_ratings(session, tournament=t)

    session.close()


@cli.command()
def enrich():
    """Compute player similarity and rolling form metrics."""
    engine = get_engine(DB_PATH)
    from src.db.schema import init_db
    init_db(DB_PATH)   # ensure new tables exist
    session = Session(engine)
    console.print("[bold]Computing player similarity…[/bold]")
    build_similarity(session, tournament="ALL")
    console.print("[bold]Computing rolling form metrics…[/bold]")
    build_form(session)
    session.close()
    console.print("[green]Enrich complete.[/green]")
    # Keep cricket.db.gz in sync so the dashboard fallback path stays current
    import gzip, shutil, tempfile
    gz_path = Path(DB_PATH).parent / "cricket.db.gz"
    console.print("[bold]Updating cricket.db.gz…[/bold]")
    with open(DB_PATH, "rb") as _fi, gzip.open(gz_path, "wb") as _fo:
        shutil.copyfileobj(_fi, _fo)
    _tmp = Path(tempfile.gettempdir()) / "cricket.db"
    if _tmp.exists():
        _tmp.unlink()


@cli.command()
@click.option("--tournaments", "-t", multiple=True)
def all(tournaments):
    """Run the complete pipeline: download → ingest → venue → metrics → ratings → enrich."""
    ctx = click.get_current_context()
    console.print("[bold magenta]Running full pipeline[/bold magenta]")
    ctx.invoke(download, tournaments=tournaments, force=False)
    ctx.invoke(ingest,   tournaments=tournaments, verbose=False)
    ctx.invoke(venue)
    ctx.invoke(metrics)
    ctx.invoke(ratings, tournament="ALL")
    ctx.invoke(enrich)
    console.print("\n[bold green]Pipeline complete.[/bold green]")
    console.print("Launch dashboard:  streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    cli()
