"""
Graph retriever — 3 subtypes of KG-based retrieval:
1. Entity-centric: query entities → KG lookup → connected documents
2. Path-based: shortest path between query entities (Neo4j Cypher)
3. Community-based: Louvain clusters → relevant community docs (Neo4j GDS)
"""
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GraphRetriever:
    """
    Knowledge graph retriever using Neo4j (GDS for analytics).
    """

    def __init__(self, neo4j_client, top_k: int = 30):
        self.neo4j = neo4j_client
        self.top_k = top_k

    def retrieve_entity_centric(
        self, entities: List[str], top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Entity-centric retrieval: extract entities → KG → connected documents.
        Uses Neo4j full-text index to find entity nodes, then retrieves linked docs.
        """
        k = top_k or self.top_k
        all_docs: List[Document] = []
        seen_ids = set()

        for entity in entities:
            # Full-text search for entity in KG
            entity_results = self.neo4j.fulltext_search(entity, limit=5)
            for entity_result in entity_results:
                entity_id = entity_result.get("id", "")
                if not entity_id:
                    continue

                # Get documents connected to this entity
                doc_results = self.neo4j.get_documents_for_entity(entity_id)
                for doc_r in doc_results:
                    doc_id = doc_r.get("doc_id", "")
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)

                    all_docs.append(
                        Document(
                            page_content=doc_r.get("content") or doc_r.get("title") or "",
                            metadata={
                                "id": doc_id,
                                "retriever_type": "graph_entity",
                                "collection": doc_r.get("collection", ""),
                                "kg_entity": entity_result.get("name", ""),
                                "kg_entity_type": entity_result.get("type", ""),
                                "entity_score": entity_result.get("score", 1.0),
                            },
                        )
                    )

        logger.debug(f"Graph entity-centric: {len(all_docs)} docs for {len(entities)} entities")
        return all_docs[:k]

    def retrieve_path_based(
        self, from_entity: str, to_entity: str
    ) -> List[Document]:
        """
        Path-based retrieval: shortest path between two entities via Neo4j.
        Returns documents found along the path.
        """
        if not self.neo4j:
            logger.warning("Neo4j not available — skipping path-based retrieval")
            return []

        # Find entity IDs in Neo4j
        from_results = self.neo4j.fulltext_search(from_entity, limit=1)
        to_results = self.neo4j.fulltext_search(to_entity, limit=1)

        if not from_results or not to_results:
            return []

        from_id = from_results[0]["id"]
        to_id = to_results[0]["id"]

        path_results = self.neo4j.shortest_path(from_id, to_id)
        if not path_results:
            return []

        docs: List[Document] = []
        for path_r in path_results:
            path_nodes = path_r.get("path_nodes", [])
            for node in path_nodes:
                # Get documents connected to path nodes
                connected_docs = self.neo4j.get_documents_for_entity(node.get("id", ""))
                for doc_r in connected_docs:
                    docs.append(
                        Document(
                            page_content=doc_r.get("content") or doc_r.get("title") or "",
                            metadata={
                                "id": doc_r.get("doc_id", ""),
                                "retriever_type": "graph_path",
                                "collection": doc_r.get("collection", ""),
                                "path_node": node.get("name", ""),
                                "path_length": path_r.get("path_length", 0),
                            },
                        )
                    )

        logger.debug(f"Graph path-based: {len(docs)} docs")
        return docs[: self.top_k]

    def retrieve_community_based(
        self, query_entities: List[str], top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Community-based retrieval: Louvain clusters → find community → retrieve docs.
        Uses Neo4j GDS Louvain community detection.
        """
        if not self.neo4j:
            logger.warning("Neo4j not available — skipping community-based retrieval")
            return []

        k = top_k or self.top_k

        # Get community assignments
        communities = self.neo4j.louvain_communities()
        if not communities:
            return []

        # Find communities for query entities
        entity_communities: Dict[int, int] = {}  # community_id → doc count
        for entity in query_entities:
            entity_results = self.neo4j.fulltext_search(entity, limit=3)
            for er in entity_results:
                entity_id = er["id"]
                # Find this entity's community
                for comm in communities:
                    if comm.get("doc_id") == entity_id:
                        cid = comm.get("community_id", -1)
                        entity_communities[cid] = entity_communities.get(cid, 0) + 1

        if not entity_communities:
            return []

        # Get top community by entity match count
        top_community = max(entity_communities, key=entity_communities.get)
        community_docs = self.neo4j.get_community_documents(top_community)

        docs = [
            Document(
                page_content=d.get("content") or d.get("title") or "",
                metadata={
                    "id": d.get("doc_id", ""),
                    "retriever_type": "graph_community",
                    "collection": d.get("collection", ""),
                    "community_id": top_community,
                },
            )
            for d in community_docs
        ]

        logger.debug(f"Graph community-based: {len(docs)} docs from community {top_community}")
        return docs[:k]

    def retrieve(
        self,
        query: str,
        entities: List[str],
        subtype: str = "entity_centric",
    ) -> List[Document]:
        """
        Unified graph retrieval interface.
        subtype: 'entity_centric' | 'path_based' | 'community_based'
        """
        if subtype == "entity_centric":
            return self.retrieve_entity_centric(entities)
        elif subtype == "path_based" and len(entities) >= 2:
            return self.retrieve_path_based(entities[0], entities[1])
        elif subtype == "community_based":
            return self.retrieve_community_based(entities)
        else:
            return self.retrieve_entity_centric(entities)
