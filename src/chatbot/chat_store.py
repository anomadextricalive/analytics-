"""
Chat History Store
------------------
Persists Cricket GPT conversations to a local SQLite file (data/chats.db).
Completely separate from the analytics database.

Schema:
  conversations  — one row per chat session (auto-titled from first message)
  messages       — one row per turn (user or assistant), with queries/results as JSON
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

_DEFAULT_DB = Path(__file__).parents[2] / "data" / "chats.db"


def _conn(db_path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db(db_path: Path = _DEFAULT_DB) -> None:
    """Create tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _conn(db_path) as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT    NOT NULL DEFAULT 'New chat',
                created_at TEXT    NOT NULL,
                updated_at TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                queries_json    TEXT,
                results_json    TEXT,
                created_at      TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id, created_at);
        """)


def new_conversation(title: str = "New chat", db_path: Path = _DEFAULT_DB) -> int:
    now = datetime.utcnow().isoformat()
    with _conn(db_path) as c:
        cur = c.execute(
            "INSERT INTO conversations(title, created_at, updated_at) VALUES (?,?,?)",
            (title, now, now),
        )
        return cur.lastrowid


def rename_conversation(conv_id: int, title: str, db_path: Path = _DEFAULT_DB) -> None:
    now = datetime.utcnow().isoformat()
    with _conn(db_path) as c:
        c.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (title, now, conv_id),
        )


def delete_conversation(conv_id: int, db_path: Path = _DEFAULT_DB) -> None:
    with _conn(db_path) as c:
        c.execute("DELETE FROM conversations WHERE id=?", (conv_id,))


def add_message(
    conv_id: int,
    role: str,
    content: str,
    queries: list | None = None,
    results: list | None = None,
    db_path: Path = _DEFAULT_DB,
) -> None:
    now = datetime.utcnow().isoformat()
    with _conn(db_path) as c:
        c.execute(
            """INSERT INTO messages
               (conversation_id, role, content, queries_json, results_json, created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                conv_id, role, content,
                json.dumps(queries) if queries else None,
                json.dumps(results) if results else None,
                now,
            ),
        )
        c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conv_id))


def load_messages(conv_id: int, db_path: Path = _DEFAULT_DB) -> list[dict]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at",
            (conv_id,),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "role": r["role"],
            "content": r["content"],
            "created_at": r["created_at"],
            "queries_run": json.loads(r["queries_json"]) if r["queries_json"] else [],
            "raw_results": json.loads(r["results_json"]) if r["results_json"] else [],
        }
        for r in rows
    ]


def list_conversations(db_path: Path = _DEFAULT_DB) -> list[dict]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def admin_stats(db_path: Path = _DEFAULT_DB) -> dict:
    with _conn(db_path) as c:
        total_conv = c.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        total_msg  = c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        daily = c.execute("""
            SELECT DATE(created_at) as day, COUNT(*) as convs
            FROM conversations
            GROUP BY day ORDER BY day DESC LIMIT 14
        """).fetchall()
    return {
        "total_conversations": total_conv,
        "total_messages": total_msg,
        "daily": [dict(r) for r in daily],
    }


def auto_title(question: str, max_len: int = 48) -> str:
    q = question.strip().rstrip("?").strip()
    return q[:max_len] + ("…" if len(q) > max_len else "")
