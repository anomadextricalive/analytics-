"""
Quick CLI for ad-hoc player queries — no dashboard needed.

Usage:
  python scripts/query.py search "Kohli"
  python scripts/query.py profile "V Kohli"
  python scripts/query.py compare "V Kohli" "RG Sharma"
  python scripts/query.py venue "Wankhede Stadium"
  python scripts/query.py leaderboard --type bat --top 20
"""

import sys
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
from src.db.schema import get_engine, Player, PlayerRating
from src.analytics.rating import compare_players

console = Console()


def _session():
    return Session(get_engine(DB_PATH))


@click.group()
def cli():
    """Cricket analytics query tool."""
    pass


@cli.command()
@click.argument("name")
def search(name):
    """Search for a player by partial name."""
    s = _session()
    rows = s.query(Player).filter(
        Player.cricsheet_key.ilike(f"%{name}%")
    ).limit(20).all()
    if not rows:
        console.print("[red]No players found.[/red]")
        return
    t = Table("ID", "Name", "Country")
    for r in rows:
        t.add_row(str(r.id), r.cricsheet_key, r.country or "")
    console.print(t)
    s.close()


@cli.command()
@click.argument("name")
@click.option("--tournament", "-t", default="ALL")
def profile(name, tournament):
    """Show full profile for a player."""
    s = _session()
    p = s.query(Player).filter_by(cricsheet_key=name).first()
    if not p:
        console.print(f"[red]Player '{name}' not found. Use 'search' first.[/red]")
        return

    console.print(f"\n[bold]{p.cricsheet_key}[/bold]  [{p.country}]")

    rating = s.query(PlayerRating).filter_by(
        player_id=p.id, tournament=tournament
    ).first()

    if rating:
        console.print(f"\n[bold]Ratings ({tournament})[/bold]")
        items = [
            ("Bat Rating",      rating.bat_rating),
            ("Bowl Rating",     rating.bowl_rating),
            ("Overall",         rating.overall_rating),
            ("Opener Score",    rating.opener_score),
            ("Finisher Score",  rating.finisher_score),
            ("Chase Score",     rating.chase_score),
            ("PP Bat Score",    rating.pp_bat_score),
            ("Death Bat Score", rating.death_bat_score),
            ("Death Bowl Score",rating.death_bowl_score),
        ]
        t = Table("Metric", "Score")
        for label, val in items:
            t.add_row(label, f"{val:.1f}" if val is not None else "—")
        console.print(t)

    # Career bat from DB
    bat = pd.read_sql(
        f"SELECT * FROM player_career_bat WHERE player_id={p.id} "
        f"AND tournament='{tournament}'",
        s.bind,
    )
    if not bat.empty:
        r = bat.iloc[0]
        console.print(f"\n[bold]Batting ({tournament})[/bold]")
        console.print(
            f"  Inn={int(r.get('innings',0))}  Runs={int(r.get('runs',0))}  "
            f"Avg={r.get('average',0):.2f}  SR={r.get('strike_rate',0):.1f}  "
            f"HS={int(r.get('hs',0))}  50s={int(r.get('fifties',0))}  "
            f"100s={int(r.get('hundreds',0))}"
        )
        console.print(
            f"  PP SR={r.get('pp_sr',0):.1f}  "
            f"Mid SR={r.get('mid_sr',0):.1f}  "
            f"Death SR={r.get('death_sr',0):.1f}"
        )
        console.print(
            f"  Adj Avg={r.get('adj_average',0):.2f}  "
            f"Adj SR={r.get('adj_strike_rate',0):.1f}"
        )

    bowl = pd.read_sql(
        f"SELECT * FROM player_career_bowl WHERE player_id={p.id} "
        f"AND tournament='{tournament}'",
        s.bind,
    )
    if not bowl.empty:
        r = bowl.iloc[0]
        console.print(f"\n[bold]Bowling ({tournament})[/bold]")
        console.print(
            f"  Inn={int(r.get('innings',0))}  Wkts={int(r.get('wickets',0))}  "
            f"Econ={r.get('economy',0):.2f}  Avg={r.get('average',0):.2f}  "
            f"SR={r.get('strike_rate',0):.1f}  Dot%={r.get('dot_pct',0):.1f}"
        )
        console.print(
            f"  PP Econ={r.get('pp_economy',0):.2f}  "
            f"Mid Econ={r.get('mid_economy',0):.2f}  "
            f"Death Econ={r.get('death_economy',0):.2f}"
        )

    s.close()


@cli.command()
@click.argument("player_a")
@click.argument("player_b")
@click.option("--tournament", "-t", default="ALL")
def compare(player_a, player_b, tournament):
    """Side-by-side comparison of two players."""
    s = _session()
    pa = s.query(Player).filter_by(cricsheet_key=player_a).first()
    pb = s.query(Player).filter_by(cricsheet_key=player_b).first()

    if not pa or not pb:
        console.print("[red]One or both players not found.[/red]")
        return

    data = compare_players(s, pa.id, pb.id, tournament)
    ra = data["player_a"]["rating"]
    rb = data["player_b"]["rating"]

    t = Table("Metric", player_a, player_b)
    metrics = [
        ("Bat Rating",      "bat_rating"),
        ("Bowl Rating",     "bowl_rating"),
        ("Overall",         "overall_rating"),
        ("Opener Score",    "opener_score"),
        ("Finisher Score",  "finisher_score"),
        ("Chase Score",     "chase_score"),
        ("PP Bat",          "pp_bat_score"),
        ("Death Bat",       "death_bat_score"),
        ("Death Bowl",      "death_bowl_score"),
    ]
    for label, attr in metrics:
        va = getattr(ra, attr, None) if ra else None
        vb = getattr(rb, attr, None) if rb else None
        t.add_row(label,
                  f"{va:.1f}" if va is not None else "—",
                  f"{vb:.1f}" if vb is not None else "—")
    console.print(t)
    s.close()


@cli.command()
@click.option("--type", "rtype", default="bat",
              type=click.Choice(["bat", "bowl", "overall"]))
@click.option("--tournament", "-t", default="ALL")
@click.option("--top", default=20)
def leaderboard(rtype, tournament, top):
    """Show top N players by rating type."""
    col = {"bat": "bat_rating", "bowl": "bowl_rating",
           "overall": "overall_rating"}[rtype]
    s = _session()
    sql = f"""
        SELECT p.cricsheet_key, p.country, r.{col}
        FROM player_ratings r
        JOIN players p ON p.id = r.player_id
        WHERE r.tournament = '{tournament}'
        ORDER BY r.{col} DESC NULLS LAST
        LIMIT {top}
    """
    df = pd.read_sql(sql, s.bind)
    t = Table("Rank", "Player", "Country", rtype.title() + " Rating")
    for i, row in df.iterrows():
        t.add_row(str(i + 1), row["cricsheet_key"],
                  row.get("country") or "", f"{row[col]:.1f}")
    console.print(t)
    s.close()


@cli.command()
@click.argument("name")
def venue(name):
    """Show venue difficulty stats."""
    s = _session()
    sql = """
        SELECT vd.*, v.name, v.country
        FROM venue_difficulty vd
        JOIN venues v ON v.id = vd.venue_id
        WHERE v.name LIKE :n
    """
    df = pd.read_sql(sql, s.bind, params={"n": f"%{name}%"})
    if df.empty:
        console.print("[red]Venue not found.[/red]")
        return
    for _, row in df.iterrows():
        console.print(f"\n[bold]{row['name']}[/bold]  [{row.get('country','')}]")
        console.print(f"  Matches={int(row['total_matches'])}")
        console.print(f"  Avg 1st inn runs={row['avg_first_inn_runs']:.1f}")
        console.print(f"  Bat factor={row['bat_factor']:.3f}  "
                      f"(>1=batter friendly, 95% CI: "
                      f"[{row.get('bat_factor_lo',0):.2f}, {row.get('bat_factor_hi',0):.2f}])")
        console.print(f"  Boundary rate={row.get('boundary_rate',0):.3f}/ball")
        console.print(f"  Pace index={row.get('pace_index',0):.2f}  "
                      f"Spin index={row.get('spin_index',0):.2f}")
    s.close()


if __name__ == "__main__":
    cli()
