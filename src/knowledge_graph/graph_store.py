"""
Graph store — persists KG nodes and edges to Neo4j.
Handles upsert logic, batch writes, and KG-to-vectorstore metadata linkage.
"""
import logging
from typing import List, Dict, Any

from .schema import GraphNode, GraphEdge, KGTriplet

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Persists knowledge graph to Neo4j.
    Adds kg_node_ids and graph_enriched metadata to _ag_rag collections.
    """

    def __init__(self, neo4j_client):
        self.neo4j = neo4j_client

    def store_nodes(self, nodes: List[GraphNode], batch_size: int = 100) -> int:
        """Batch upsert nodes into Neo4j."""
        stored = 0
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i: i + batch_size]
            for node in batch:
                props = {
                    "name": node.name,
                    "source_collection": node.source_collection,
                    "source_doc_id": node.source_doc_id,
                    **node.properties,
                }
                try:
                    self.neo4j.upsert_node(node.node_type.value, node.id, props)
                    stored += 1
                except Exception as e:
                    logger.warning(f"Failed to store node {node.id}: {e}")
        logger.info(f"Stored {stored}/{len(nodes)} nodes to Neo4j")
        return stored

    def store_edges(self, edges: List[GraphEdge], batch_size: int = 200) -> int:
        """Batch upsert edges into Neo4j."""
        stored = 0
        for i in range(0, len(edges), batch_size):
            batch = edges[i: i + batch_size]
            for edge in batch:
                try:
                    self.neo4j.upsert_edge(
                        edge.source_id,
                        edge.target_id,
                        edge.edge_type.value,
                        {"weight": edge.weight, **edge.properties},
                    )
                    stored += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to store edge {edge.source_id}->{edge.target_id}: {e}"
                    )
        logger.info(f"Stored {stored}/{len(edges)} edges to Neo4j")
        return stored

    def store_triplets(self, triplets: List[KGTriplet]) -> Dict[str, int]:
        """Convert triplets to nodes+edges and store to Neo4j."""
        from .schema import GraphNode, GraphEdge

        nodes_seen: Dict[str, GraphNode] = {}
        edges: List[GraphEdge] = []

        for t in triplets:
            # Subject node
            subj_id = f"{t.subject_type.value.lower()}_{t.subject.lower().replace(' ', '_')}"
            if subj_id not in nodes_seen:
                nodes_seen[subj_id] = GraphNode(
                    id=subj_id,
                    node_type=t.subject_type,
                    name=t.subject,
                    source_collection=t.source_collection,
                    source_doc_id=t.source_doc_id,
                )

            # Object node
            obj_id = f"{t.object_type.value.lower()}_{t.object.lower().replace(' ', '_')}"
            if obj_id not in nodes_seen:
                nodes_seen[obj_id] = GraphNode(
                    id=obj_id,
                    node_type=t.object_type,
                    name=t.object,
                    source_collection=t.source_collection,
                    source_doc_id=t.source_doc_id,
                )

            # Edge
            edges.append(
                GraphEdge(
                    source_id=subj_id,
                    target_id=obj_id,
                    edge_type=t.relation,
                    weight=t.confidence,
                    source_doc_id=t.source_doc_id,
                )
            )

        n_nodes = self.store_nodes(list(nodes_seen.values()))
        n_edges = self.store_edges(edges)
        return {"nodes": n_nodes, "edges": n_edges}

    def get_node_ids_for_doc(self, doc_id: str) -> List[str]:
        """Get all KG node IDs extracted from a given document."""
        results = self.neo4j.run_query(
            "MATCH (n {source_doc_id: $doc_id}) RETURN n.id AS id",
            {"doc_id": doc_id},
        )
        return [r["id"] for r in results]

    def graph_stats(self) -> Dict[str, int]:
        """Return node/edge counts per type."""
        node_counts = self.neo4j.run_query(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC"
        )
        edge_counts = self.neo4j.run_query(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC"
        )
        return {
            "nodes": {r["label"]: r["cnt"] for r in node_counts},
            "edges": {r["rel"]: r["cnt"] for r in edge_counts},
        }
