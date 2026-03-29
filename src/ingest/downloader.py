"""
Downloads and extracts cricsheet bulk JSON zip files.
Respects rate limits and skips already-downloaded files.
"""

import zipfile
import time
import sys
import os

import requests
from pathlib import Path
from tqdm import tqdm
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import DATA_RAW, DOWNLOADS

console = Console()


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download *url* to *dest*; skip if file exists unless force=True."""
    if dest.exists() and not force:
        console.print(f"  [dim]skip[/dim] {dest.name} (already downloaded)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"  [cyan]↓[/cyan] {url}")

    r = requests.get(url, stream=True, timeout=60,
                     headers={"User-Agent": "cricket-analytics-research/1.0"})
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as bar:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))

    return dest


def extract_zip(zip_path: Path, out_dir: Path) -> int:
    """Extract zip to out_dir, return count of files extracted."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".json")]
        existing = {p.name for p in out_dir.glob("*.json")}
        to_extract = [m for m in members if Path(m).name not in existing]

        if not to_extract:
            console.print(f"  [dim]all {len(members)} files already extracted[/dim]")
            return 0

        for m in tqdm(to_extract, desc=f"extract {zip_path.stem}", leave=False):
            data = zf.read(m)
            (out_dir / Path(m).name).write_bytes(data)

    console.print(f"  [green]✓[/green] extracted {len(to_extract)} new files → {out_dir}")
    return len(to_extract)


def download_all(tournaments: list[str] | None = None, force: bool = False) -> dict[str, Path]:
    """
    Download and extract all configured tournaments (or a subset).
    Returns mapping of tournament name → extracted directory.
    """
    targets = {k: v for k, v in DOWNLOADS.items()
               if tournaments is None or k in tournaments}

    result = {}
    for name, url in targets.items():
        console.print(f"\n[bold]{name}[/bold]")
        zip_path = DATA_RAW / "zips" / f"{name}.zip"
        try:
            download_file(url, zip_path, force=force)
            out_dir = DATA_RAW / name
            extract_zip(zip_path, out_dir)
            result[name] = out_dir
            time.sleep(0.5)   # polite delay between downloads
        except requests.HTTPError as e:
            console.print(f"  [red]HTTP error {e.response.status_code}[/red] for {url}")
        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}")

    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tournaments", nargs="*", help="subset to download")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    download_all(args.tournaments, args.force)
