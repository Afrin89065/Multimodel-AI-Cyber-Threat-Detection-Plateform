import redis.asyncio as aioredis
from core.config import settings
from typing import Optional

_redis: Optional[aioredis.Redis] = None

async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
    return _redis

async def close_redis():
    global _redis
    if _redis:
        await _redis.close()
        _redis = None