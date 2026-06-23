import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from fastapi import FastAPI, HTTPException, Query
from src.api.db import get_db, close


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close()


app = FastAPI(title="Cricket Analytics API", version="1.0.0", lifespan=lifespan)


def _clean(doc: dict) -> dict:
    doc.pop("_id", None)
    return doc


# ── Players ──────────────────────────────────────────────────────────────────

@app.get("/players/{player_id}")
async def get_player(player_id: int):
    """Full player profile by numeric player ID."""
    doc = await get_db()["player_profiles"].find_one({"id": player_id})
    if not doc:
        raise HTTPException(404, f"Player {player_id} not found")
    return _clean(doc)


@app.get("/players/uuid/{uuid}")
async def get_player_by_uuid(uuid: str):
    """Full player profile by cricsheet UUID (hex string)."""
    doc = await get_db()["player_profiles"].find_one({"cricsheet_uuid": uuid})
    if not doc:
        raise HTTPException(404, f"UUID {uuid!r} not found")
    return _clean(doc)


@app.get("/players/top")
async def top_players(
    limit: int = Query(100, ge=1, le=500),
    sort_by: str = Query("overall_rating", description="ratings field to sort by"),
):
    """Top N players sorted by a rating field. No query required — for browsing."""
    sort_key = f"ratings.{sort_by}"
    cursor = get_db()["player_profiles"].find(
        {"ratings": {"$exists": True}},
        {"id": 1, "name": 1, "country": 1, "ratings": 1, "_id": 0},
    ).sort(sort_key, -1).limit(limit)
    return [doc async for doc in cursor]


@app.get("/players")
async def search_players(
    q: str = Query(None, min_length=2, description="Partial name search (optional)"),
    limit: int = Query(50, ge=1, le=500),
):
    """Search players by partial name, or return top players if no query."""
    projection = {"id": 1, "name": 1, "country": 1, "ratings": 1, "_id": 0}
    if q:
        pattern = re.compile(re.escape(q), re.IGNORECASE)
        cursor = get_db()["player_profiles"].find({"name": pattern}, projection).limit(limit)
    else:
        cursor = get_db()["player_profiles"].find(
            {"ratings": {"$exists": True}}, projection
        ).sort("ratings.overall_rating", -1).limit(limit)
    return [doc async for doc in cursor]


# ── Venues ───────────────────────────────────────────────────────────────────

@app.get("/venues/{venue_id}")
async def get_venue(venue_id: int):
    doc = await get_db()["venue_profiles"].find_one({"id": venue_id})
    if not doc:
        raise HTTPException(404, f"Venue {venue_id} not found")
    return _clean(doc)


@app.get("/venues")
async def search_venues(
    q: str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=50),
):
    pattern = re.compile(re.escape(q), re.IGNORECASE)
    cursor = get_db()["venue_profiles"].find(
        {"name": pattern},
        {"id": 1, "name": 1, "city": 1, "difficulty": 1, "_id": 0},
    ).limit(limit)
    return [doc async for doc in cursor]


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    db = get_db()
    return {
        "status": "ok",
        "players": await db["player_profiles"].estimated_document_count(),
        "venues":  await db["venue_profiles"].estimated_document_count(),
        "matches": await db["match_profiles"].estimated_document_count(),
    }
