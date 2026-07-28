"""AIDTECT v3.0 — Neo4j Knowledge Graph"""
import uuid
from datetime import datetime, timedelta
from typing import Optional, List
from loguru import logger

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j not installed")


class ThreatKnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687",
                 user="neo4j", password="aidtect123"):
        if not NEO4J_AVAILABLE:
            self.driver = None
            return
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self._create_schema()
            logger.info("Knowledge Graph connected")
        except Exception as e:
            self.driver = None
            logger.warning(f"Neo4j connection failed: {e}")

    def _create_schema(self):
        with self.driver.session() as s:
            s.run("CREATE INDEX IF NOT EXISTS FOR (n:IPAddress) ON (n.address)")
            s.run("CREATE INDEX IF NOT EXISTS FOR (n:ThreatEvent) ON (n.id)")

    def ingest_event(self, fusion_result: dict, nlp_result: dict = None,
                     network_result: dict = None, malware_result: dict = None,
                     source_ip: str = None, request_id: str = None) -> Optional[str]:
        if not self.driver:
            return None
        event_id = request_id or str(uuid.uuid4())
        try:
            with self.driver.session() as s:
                s.run("""
                    CREATE (e:ThreatEvent {
                        id: $id, threat_class: $tc,
                        severity: $sev, risk_score: $rs, timestamp: $ts
                    })
                """, id=event_id,
                     tc=fusion_result.get("threat_class", "UNKNOWN"),
                     sev=fusion_result.get("severity", "LOW"),
                     rs=float(fusion_result.get("risk_score", 0)),
                     ts=datetime.utcnow().isoformat())

                if source_ip:
                    s.run("""
                        MERGE (ip:IPAddress {address: $addr})
                        WITH ip MATCH (e:ThreatEvent {id: $eid})
                        CREATE (e)-[:TRIGGERED_FROM]->(ip)
                    """, addr=source_ip, eid=event_id)

                if nlp_result and nlp_result.get("url"):
                    s.run("""
                        MERGE (u:URL {value: $url})
                        WITH u MATCH (e:ThreatEvent {id: $eid})
                        CREATE (e)-[:INVOLVES_URL]->(u)
                    """, url=nlp_result["url"], eid=event_id)

            return event_id
        except Exception as e:
            logger.error(f"KG ingestion failed: {e}")
            return None

    def detect_campaigns(self, min_events: int = 3) -> List[dict]:
        if not self.driver:
            return []
        with self.driver.session() as s:
            result = s.run("""
                MATCH (ip:IPAddress)<-[:TRIGGERED_FROM]-(e:ThreatEvent)
                WITH ip, COUNT(e) AS event_count,
                     COLLECT(DISTINCT e.threat_class) AS classes,
                     AVG(e.risk_score) AS avg_risk
                WHERE event_count >= $min_events
                RETURN ip.address AS ip, event_count, classes, avg_risk
                ORDER BY event_count DESC LIMIT 20
            """, min_events=min_events)
            return [dict(r) for r in result]

    def get_related_events(self, ip: str, hours: int = 24) -> List[dict]:
        if not self.driver:
            return []
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self.driver.session() as s:
            result = s.run("""
                MATCH (ip:IPAddress {address: $addr})
                      <-[:TRIGGERED_FROM]-(e:ThreatEvent)
                WHERE e.timestamp > $cutoff
                RETURN e.id AS id, e.threat_class AS threat_class,
                       e.severity AS severity, e.risk_score AS risk_score,
                       e.timestamp AS timestamp
                ORDER BY e.timestamp DESC
            """, addr=ip, cutoff=cutoff)
            return [dict(r) for r in result]

    def find_common_malware(self) -> List[dict]:
        if not self.driver:
            return []
        with self.driver.session() as s:
            result = s.run("""
                MATCH (m:MalwareFamily)<-[:USES_MALWARE]-(e:ThreatEvent)
                WITH m.name AS family, COUNT(e) AS detections
                RETURN family, detections
                ORDER BY detections DESC LIMIT 10
            """)
            return [dict(r) for r in result]

    def close(self):
        if self.driver:
            self.driver.close()