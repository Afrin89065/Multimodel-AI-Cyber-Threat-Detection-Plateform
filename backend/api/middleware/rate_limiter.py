import time
from collections import defaultdict
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls: int = 60, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self._store: dict = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in ("/api/v1/health", "/metrics", "/docs", "/redoc", "/openapi.json"):
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        self._store[client_ip] = [t for t in self._store[client_ip] if now - t < self.period]
        if len(self._store[client_ip]) >= self.calls:
            return JSONResponse(status_code=429, content={
                "detail": f"Rate limit: {self.calls} requests per {self.period}s"})
        self._store[client_ip].append(now)
        return await call_next(request)