"""Unpack the bundled cricket.db.gz to a read-only working copy, once per gz version.

Streamlit re-executes the whole script on every interaction and every new session. Rewriting the 600 MB
database in place on each run truncated the file under sessions that were still reading it, and a stale
`-wal`/`-shm` left in /tmp by an earlier deploy made a freshly unpacked file unreadable. Both surface as
"database disk image is malformed".

ensure_db() therefore:
  - skips the work when the working copy already matches the gz (version = CRC32 stored in the gzip trailer);
  - takes a file lock so concurrent first sessions unpack once;
  - writes to a temp file and swaps it in with os.replace (open connections keep their old, intact file);
  - removes stale -wal/-shm and switches the copy to journal_mode=DELETE so the served file never needs them.
"""
import fcntl
import gzip
import os
import shutil
import sqlite3
from pathlib import Path


def _gz_version(gz: Path) -> str:
    """CRC32 + uncompressed size of the data, from the last 8 bytes of the gzip file. Cheap and content-based."""
    with open(gz, "rb") as f:
        f.seek(-8, os.SEEK_END)
        return f.read(8).hex()


def is_current(gz, dest="/tmp/cricket.db") -> bool:
    """True when the working copy already matches this gz (no unpack needed)."""
    dest = Path(dest); marker = dest.with_name(dest.name + ".version")
    return dest.exists() and marker.exists() and marker.read_text() == _gz_version(Path(gz))


def ensure_db(gz, dest="/tmp/cricket.db") -> Path:
    gz, dest = Path(gz), Path(dest)
    marker = dest.with_name(dest.name + ".version")
    version = _gz_version(gz)

    def current() -> bool:
        return dest.exists() and marker.exists() and marker.read_text() == version

    if current():
        return dest
    with open(dest.with_name(dest.name + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if current():           # another session finished while we waited
                return dest
            part = dest.with_name(f"{dest.name}.{os.getpid()}.part")
            with gzip.open(gz, "rb") as f_in, open(part, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            con = sqlite3.connect(part)
            con.execute("PRAGMA journal_mode=DELETE")   # served copy is read-only: no WAL, no shm
            con.close()
            for ext in ("-wal", "-shm"):                # leftovers from a previous version would corrupt the new file
                Path(str(dest) + ext).unlink(missing_ok=True)
            os.replace(part, dest)
            marker.write_text(version)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return dest
