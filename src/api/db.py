from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config import MONGO_URI, MONGO_DB

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        if not MONGO_URI:
            raise RuntimeError("MONGODB_URI env var not set")
        _client = AsyncIOMotorClient(MONGO_URI)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    return get_client()[MONGO_DB]


async def close():
    global _client
    if _client:
        _client.close()
        _client = None
