import datetime
from fastapi import APIRouter, Request, Depends, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import text
from core.database import AsyncSessionLocal
from core.security import get_current_user, TokenData
from core.websocket_manager import ws_manager

router = APIRouter(tags=["dashboard"])

@router.get("/dashboard/stats")
async def get_stats(hours: int = 24, current_user: TokenData = Depends(get_current_user)):
    async with AsyncSessionLocal() as db:
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
        result = await db.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE severity='CRITICAL') AS critical,
                COUNT(*) FILTER (WHERE severity='HIGH') AS high,
                COUNT(*) FILTER (WHERE severity='MEDIUM') AS medium,
                COUNT(*) FILTER (WHERE severity='LOW') AS low,
                COUNT(*) FILTER (WHERE is_false_positive=TRUE) AS false_positives,
                COUNT(*) FILTER (WHERE needs_human_review=TRUE) AS needs_review,
                AVG(risk_score) AS avg_risk_score,
                COUNT(*) AS total
            FROM threat_events WHERE created_at >= :cutoff
        """), {"cutoff": cutoff})
        row = result.fetchone()
        return {
            "critical": row.critical or 0, "high": row.high or 0,
            "medium": row.medium or 0, "low": row.low or 0,
            "false_positives": row.false_positives or 0,
            "needs_review": row.needs_review or 0,
            "avg_risk_score": round(float(row.avg_risk_score or 0), 4),
            "total": row.total or 0, "period_hours": hours,
        }

@router.get("/dashboard/events")
async def get_events(limit: int = 50, severity: Optional[str] = None, current_user: TokenData = Depends(get_current_user)):
    async with AsyncSessionLocal() as db:
        query = "SELECT * FROM threat_events"
        params = {}
        if severity:
            query += " WHERE severity = :severity"
            params["severity"] = severity
        query += " ORDER BY created_at DESC LIMIT :limit"
        params["limit"] = limit
        result = await db.execute(text(query), params)
        rows = result.fetchall()
        return {"events": [dict(r._mapping) for r in rows], "count": len(rows)}

class VerdictUpdate(BaseModel):
    verdict: str
    notes: Optional[str] = ""

@router.patch("/dashboard/events/{event_id}/verdict")
async def update_verdict(event_id: str, body: VerdictUpdate, current_user: TokenData = Depends(get_current_user)):
    if body.verdict not in ("CONFIRMED", "FALSE_POSITIVE", "INVESTIGATING"):
        raise HTTPException(status_code=400, detail="Invalid verdict")
    async with AsyncSessionLocal() as db:
        await db.execute(text("""
            UPDATE threat_events
            SET analyst_verdict=:verdict, analyst_notes=:notes,
                analyst_id=:analyst, is_false_positive=:is_fp
            WHERE id=:id
        """), {"verdict": body.verdict, "notes": body.notes,
               "analyst": current_user.username,
               "is_fp": body.verdict == "FALSE_POSITIVE", "id": event_id})
        await db.commit()
    return {"updated": True, "event_id": event_id, "verdict": body.verdict}

@router.websocket("/ws/threats")
async def websocket_feed(websocket: WebSocket):
    await ws_manager.connect(websocket, room="soc")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, room="soc")