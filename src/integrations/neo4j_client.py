"""
Neo4j client — unified graph store and analytics engine.
Provides Cypher query helpers, native graph analytics (degree centrality,
community grouping, shortest path), and episodic memory storage.
"""
import logging
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver, Session

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j driver wrapper with Cypher helpers and GDS analytics.

    Responsibilities:
    - Full graph CRUD (nodes, edges)
    - LlamaIndex PropertyGraphIndex backend (via neo4j driver)
    - APOC procedures for complex graph operations
    - Full-text indexes on node properties
    - Native Cypher analytics: degree centrality, community grouping, shortest path
    - Episodic memory storage (session summaries)
    """

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self._driver: Optional[Driver] = None
        self._password = password

    def connect(self) -> None:
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self._password))
        self._driver.verify_connectivity()
        logger.info(f"Neo4j connected: {self.uri}")

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def driver(self) -> Driver:
        if not self._driver:
            self.connect()
        return self._driver

    def run_query(
        self, cypher: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return records as dicts."""
        with self.driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]

    def upsert_node(
        self, label: str, node_id: str, properties: Dict[str, Any]
    ) -> None:
        """Create or update a node by id."""
        cypher = (
            f"MERGE (n:{label} {{id: $id}}) "
            "SET n += $props"
        )
        self.run_query(cypher, {"id": node_id, "props": properties})

    def upsert_edge(
        self,
        from_id: str,
        to_id: str,
        relation: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create or update a directed edge between two nodes."""
        cypher = (
            "MATCH (a {id: $from_id}), (b {id: $to_id}) "
            f"MERGE (a)-[r:{relation}]->(b) "
            "SET r += $props"
        )
        self.run_query(cypher, {"from_id": from_id, "to_id": to_id, "props": properties or {}})

    def get_neighbors(
        self, node_id: str, relation: Optional[str] = None, depth: int = 1
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes up to `depth` hops."""
        rel_filter = f"[r:{relation}*1..{depth}]" if relation else f"[*1..{depth}]"
        cypher = (
            f"MATCH (n {{id: $id}})-{rel_filter}-(m) "
            "RETURN DISTINCT m, labels(m) as labels"
        )
        return self.run_query(cypher, {"id": node_id})

    def fault_chain_query(self, symptom_keywords: List[str]) -> List[Dict[str, Any]]:
        """Find faults matching symptoms and their resolution procedures."""
        cypher = """
        MATCH (f:Fault)-[:RESOLVED_BY]->(p:Procedure)
        WHERE any(kw IN $keywords WHERE toLower(f.name) CONTAINS toLower(kw)
                  OR toLower(f.description) CONTAINS toLower(kw))
        RETURN f.name AS fault, f.description AS fault_desc,
               p.name AS procedure, p.steps AS steps,
               f.frequency AS frequency
        ORDER BY f.frequency DESC
        LIMIT 10
        """
        return self.run_query(cypher, {"keywords": symptom_keywords})

    def expert_query(self, system: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Find top experts for a given system based on document authorship."""
        cypher = """
        MATCH (p:Person)-[:WORKS_ON]->(s:System {name: $system})
        OPTIONAL MATCH (p)<-[:AUTHORED_BY]-(d:LogEntry)
        RETURN p.name AS person, count(d) AS doc_count
        ORDER BY doc_count DESC
        LIMIT $top_n
        """
        return self.run_query(cypher, {"system": system, "top_n": top_n})

    def get_documents_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all documents connected to a given entity node."""
        cypher = """
        MATCH (e {id: $entity_id})<-[*1..2]-(d)
        WHERE d:LogEntry OR d:ICMSDocument OR d:WorkRequest OR d:TeamsMessage
        RETURN DISTINCT d.id AS doc_id, d.title AS title,
               d.source_collection AS collection,
               d.creation_datetime AS date
        LIMIT 20
        """
        return self.run_query(cypher, {"entity_id": entity_id})

    def create_fulltext_index(self) -> None:
        """Create full-text indexes on node name/description fields."""
        indexes = [
            "CREATE FULLTEXT INDEX nodeNameIndex IF NOT EXISTS "
            "FOR (n:Fault|Component|System|Procedure|Concept|Person) "
            "ON EACH [n.name, n.description]",
            "CREATE FULLTEXT INDEX docContentIndex IF NOT EXISTS "
            "FOR (n:LogEntry|ICMSDocument|WorkRequest) "
            "ON EACH [n.content, n.title]",
        ]
        for idx in indexes:
            try:
                self.run_query(idx)
            except Exception as e:
                logger.warning(f"Index creation skipped (may already exist): {e}")

    def fulltext_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Full-text search across node names and content."""
        cypher = """
        CALL db.index.fulltext.queryNodes('nodeNameIndex', $query)
        YIELD node, score
        RETURN node.id AS id, node.name AS name, labels(node)[0] AS type, score
        ORDER BY score DESC LIMIT $limit
        """
        return self.run_query(cypher, {"query": query, "limit": limit})

    def health_check(self) -> bool:
        """Check Neo4j connectivity."""
        try:
            result = self.run_query("RETURN 1 AS ok")
            return bool(result and result[0].get("ok") == 1)
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Graph Analytics: degree centrality, community grouping, shortest path
    # Uses native Cypher (no GDS dependency — works on community edition)
    # ------------------------------------------------------------------

    def pagerank(self, top_n: int = 20) -> List[Dict]:
        """Degree-centrality ranking for Person/System nodes (PageRank proxy).

        Counts total relationships as a centrality measure.
        Works on Neo4j community edition without GDS.
        """
        try:
            return self.run_query(
                """
                MATCH (n)-[r]-()
                WHERE n:Person OR n:System
                WITH n, count(r) AS degree
                RETURN n.name AS name, labels(n)[0] AS type,
                       toFloat(degree) AS rank
                ORDER BY degree DESC LIMIT $top_n
                """,
                {"top_n": top_n},
            )
        except Exception as e:
            logger.warning("PageRank (degree centrality) failed: %s", e)
            return []

    def louvain_communities(self) -> List[Dict]:
        """Entity-based community grouping (Louvain proxy).

        Groups documents by their most connected shared entity (System),
        using entity co-occurrence as a community signal.
        Works on Neo4j community edition without GDS.
        """
        try:
            return self.run_query(
                """
                MATCH (d)-[:ABOUT_SYSTEM]->(s:System)
                WHERE d:LogEntry OR d:ICMSDocument
                WITH s.id AS community_id, collect({
                    doc_id: d.id, title: d.title,
                    collection: d.source_collection
                }) AS docs
                WHERE size(docs) > 1
                UNWIND docs AS doc
                RETURN doc.doc_id AS doc_id, doc.title AS title,
                       doc.collection AS collection,
                       community_id
                ORDER BY community_id
                """
            )
        except Exception as e:
            logger.warning("Community detection failed: %s", e)
            return []

    def get_community_documents(self, community_id: Any) -> List[Dict]:
        """Retrieve documents sharing the same System entity (community)."""
        try:
            return self.run_query(
                """
                MATCH (d)-[:ABOUT_SYSTEM]->(s:System {id: $cid})
                WHERE d:LogEntry OR d:ICMSDocument OR d:WorkRequest
                RETURN d.id AS doc_id, d.title AS title,
                       d.source_collection AS collection,
                       d.creation_datetime AS date
                """,
                {"cid": community_id},
            )
        except Exception as e:
            logger.warning("Community documents query failed: %s", e)
            return []

    def shortest_path(self, from_id: str, to_id: str) -> List[Dict]:
        """Find shortest path between two nodes using native Cypher shortestPath."""
        try:
            return self.run_query(
                """
                MATCH path = shortestPath((a {id: $from_id})-[*]-(b {id: $to_id}))
                RETURN [node in nodes(path) | {id: node.id, name: node.name,
                       type: labels(node)[0]}] AS path_nodes,
                       length(path) AS path_length
                """,
                {"from_id": from_id, "to_id": to_id},
            )
        except Exception as e:
            logger.warning("Shortest path failed: %s", e)
            return []

    def bfs_neighbors(self, entity: str, max_hops: int = 3) -> List[Dict]:
        """BFS neighbourhood expansion from a starting entity."""
        try:
            results = self.fulltext_search(entity, limit=1)
            if not results:
                return []
            node_id = results[0]["id"]
            return self.run_query(
                f"""
                MATCH (start {{id: $id}})-[*1..{max_hops}]-(neighbor)
                RETURN DISTINCT neighbor.id AS id, neighbor.name AS name,
                       labels(neighbor)[0] AS type
                LIMIT 50
                """,
                {"id": node_id},
            )
        except Exception as e:
            logger.warning("BFS neighbors failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Episodic Memory (session summaries stored as :Session nodes)
    # ------------------------------------------------------------------

    def store_episode(self, session_id: str, summary: str, topics: List[str]) -> None:
        """Store an episodic memory entry (session summary) in Neo4j."""
        try:
            self.run_query(
                """
                MERGE (s:Session {id: $session_id})
                SET s.last_summary = $summary,
                    s.topics = $topics,
                    s.updated_at = datetime()
                """,
                {"session_id": session_id, "summary": summary, "topics": topics},
            )
        except Exception as e:
            logger.warning("Episode storage failed: %s", e)

    def get_episode(self, session_id: str) -> Optional[Dict]:
        """Retrieve episodic memory for a session."""
        try:
            results = self.run_query(
                """
                MATCH (s:Session {id: $session_id})
                RETURN s.last_summary AS summary, s.topics AS topics,
                       s.updated_at AS updated_at
                """,
                {"session_id": session_id},
            )
            return results[0] if results else None
        except Exception as e:
            logger.warning("Episode retrieval failed: %s", e)
            return None
