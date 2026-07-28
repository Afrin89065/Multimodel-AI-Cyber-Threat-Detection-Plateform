"""
Async batch queue for processing multiple detections in parallel.
PATH: backend/services/batch_queue_service.py
"""
import uuid
import json
import asyncio
from loguru import logger


class BatchQueueService:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.queue_key = "aidtect:queue"
        self.results_prefix = "aidtect:result:"
        self.ttl = 3600  # 1 hour

    async def enqueue(self, job_data: dict) -> str:
        job_id = str(uuid.uuid4())
        job = {"id": job_id, "data": job_data, "status": "QUEUED"}
        await self.redis.lpush(self.queue_key, json.dumps(job))
        await self.redis.setex(f"{self.results_prefix}{job_id}", self.ttl, json.dumps({"status": "QUEUED"}))
        return job_id

    async def get_result(self, job_id: str) -> dict:
        raw = await self.redis.get(f"{self.results_prefix}{job_id}")
        if not raw:
            return {"status": "NOT_FOUND", "job_id": job_id}
        return json.loads(raw)

    async def worker_loop(self, models: dict):
        logger.info("Batch queue worker started")
        while True:
            try:
                item = await self.redis.brpop(self.queue_key, timeout=5)
                if not item:
                    await asyncio.sleep(0.1)
                    continue
                _, raw = item
                job = json.loads(raw)
                job_id = job["id"]
                data = job["data"]

                await self.redis.setex(
                    f"{self.results_prefix}{job_id}",
                    self.ttl,
                    json.dumps({"status": "PROCESSING"})
                )

                result = models["fusion"].fuse(
                    nlp=data.get("nlp_result"),
                    vision=data.get("vision_result"),
                    network=data.get("network_result"),
                    malware=data.get("malware_result"),
                )

                await self.redis.setex(
                    f"{self.results_prefix}{job_id}",
                    self.ttl,
                    json.dumps({"status": "DONE", "result": result})
                )

            except asyncio.CancelledError:
                logger.info("Queue worker cancelled")
                break
            except Exception as e:
                logger.error(f"Queue worker error: {e}")
                await asyncio.sleep(1)