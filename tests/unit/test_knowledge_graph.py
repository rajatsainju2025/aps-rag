"""
Unit tests for the knowledge graph module.
Covers: schema validation, entity/relation output parsing,
        incremental updater hash logic.
No live Neo4j connection required.
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestNodeTypes:
    def test_all_expected_node_types_exist(self):
        from src.knowledge_graph.schema import NodeType
        expected = {
            "LOG_ENTRY", "PERSON", "SYSTEM", "FAULT", "COMPONENT",
            "PROCEDURE", "TIME_PERIOD", "KEYWORD", "ICMS_DOCUMENT",
            "WORK_REQUEST", "TEAMS_MESSAGE", "STUDY", "SDDS_TOOL", "OAG_APP",
        }
        actual = {m.name for m in NodeType}
        assert expected == actual, f"Missing: {expected - actual}"

    def test_node_type_values_are_pascal_case(self):
        from src.knowledge_graph.schema import NodeType
        for member in NodeType:
            assert member.value[0].isupper(), f"{member.name} value not PascalCase: {member.value}"

    def test_node_type_is_string_enum(self):
        from src.knowledge_graph.schema import NodeType
        assert isinstance(NodeType.PERSON, str)
        assert NodeType.PERSON == "Person"


class TestEdgeTypes:
    def test_all_expected_edge_types_exist(self):
        from src.knowledge_graph.schema import EdgeType
        expected = {
            "AUTHORED_BY", "WORKS_ON", "ABOUT_SYSTEM", "HAS_FAULT",
            "HAS_COMPONENT", "RESOLVED_BY", "CAUSED_BY", "OCCURRED_AT",
            "REFERENCES_DOC", "RELATED_TO", "USES_TOOL", "PART_OF_STUDY",
            "FOLLOWS_PROCEDURE",
        }
        actual = {m.name for m in EdgeType}
        assert expected == actual, f"Missing: {expected - actual}"

    def test_edge_type_values_are_screaming_snake(self):
        from src.knowledge_graph.schema import EdgeType
        for member in EdgeType:
            assert member.value == member.value.upper(), (
                f"{member.name} value not UPPER_SNAKE: {member.value}"
            )


class TestGraphNodeDataclass:
    def test_create_minimal_node(self):
        from src.knowledge_graph.schema import GraphNode, NodeType
        node = GraphNode(id="person_001", node_type=NodeType.PERSON, name="Doe")
        assert node.id == "person_001"
        assert node.node_type == NodeType.PERSON
        assert node.properties == {}
        assert node.source_collection == ""

    def test_create_node_with_properties(self):
        from src.knowledge_graph.schema import GraphNode, NodeType
        node = GraphNode(
            id="sys_001", node_type=NodeType.SYSTEM, name="SR RF",
            properties={"location": "sector-35"}, source_collection="entry_logbook",
        )
        assert node.properties["location"] == "sector-35"
        assert node.source_collection == "entry_logbook"


class TestGraphEdgeDataclass:
    def test_create_edge(self):
        from src.knowledge_graph.schema import GraphEdge, EdgeType
        edge = GraphEdge(
            source_id="person_001", target_id="sys_001",
            edge_type=EdgeType.WORKS_ON, weight=0.9,
        )
        assert edge.edge_type == EdgeType.WORKS_ON
        assert edge.weight == 0.9
        assert edge.properties == {}


class TestKGTriplet:
    def test_create_triplet(self):
        from src.knowledge_graph.schema import KGTriplet, NodeType, EdgeType
        t = KGTriplet(
            subject="Doe", subject_type=NodeType.PERSON,
            relation=EdgeType.WORKS_ON,
            object="SR RF", object_type=NodeType.SYSTEM,
            confidence=0.95, source_doc_id="doc_123",
        )
        assert t.confidence == 0.95
        assert t.relation == EdgeType.WORKS_ON


class TestTypeMaps:
    def test_node_type_map_lowercase_lookup(self):
        from src.knowledge_graph.schema import NODE_TYPE_MAP, NodeType
        assert NODE_TYPE_MAP["person"] == NodeType.PERSON
        assert NODE_TYPE_MAP["fault"] == NodeType.FAULT
        assert NODE_TYPE_MAP["sddstool"] == NodeType.SDDS_TOOL
        assert NODE_TYPE_MAP["oagapp"] == NodeType.OAG_APP

    def test_edge_type_map_lowercase_lookup(self):
        from src.knowledge_graph.schema import EDGE_TYPE_MAP, EdgeType
        assert EDGE_TYPE_MAP["authored_by"] == EdgeType.AUTHORED_BY
        assert EDGE_TYPE_MAP["resolved_by"] == EdgeType.RESOLVED_BY
        assert EDGE_TYPE_MAP["follows_procedure"] == EdgeType.FOLLOWS_PROCEDURE

    def test_concept_maps_to_keyword(self):
        from src.knowledge_graph.schema import NODE_TYPE_MAP, NodeType
        # "concept" is an LLM-friendly alias for KEYWORD
        assert NODE_TYPE_MAP["concept"] == NodeType.KEYWORD


# ---------------------------------------------------------------------------
# IncrementalKGUpdater tests
# ---------------------------------------------------------------------------

class TestIncrementalKGUpdater:
    @pytest.fixture
    def updater(self, tmp_path):
        from src.knowledge_graph.incremental_updater import IncrementalKGUpdater
        state_file = tmp_path / "kg_sync_state.json"
        return IncrementalKGUpdater(state_file=state_file)

    def test_all_docs_are_new_on_first_run(self, updater):
        docs = [
            {"id": "doc_1", "content": "Beam current anomaly in sector 7"},
            {"id": "doc_2", "content": "RF cavity trip at 10:30"},
        ]
        to_process, unchanged = updater.get_delta(docs)
        assert len(to_process) == 2
        assert len(unchanged) == 0

    def test_unchanged_docs_skipped_on_second_run(self, updater):
        docs = [{"id": "doc_1", "content": "Beam current anomaly in sector 7"}]
        updater.get_delta(docs)  # First run — marks as processed
        to_process, unchanged = updater.get_delta(docs)  # Second run — same content
        assert len(to_process) == 0
        assert len(unchanged) == 1

    def test_modified_doc_reprocessed(self, updater):
        docs = [{"id": "doc_1", "content": "Original content"}]
        updater.get_delta(docs)

        modified = [{"id": "doc_1", "content": "Modified content — new fault added"}]
        to_process, unchanged = updater.get_delta(modified)
        assert len(to_process) == 1
        assert to_process[0]["content"].startswith("Modified")

    def test_hash_deterministic(self, updater):
        doc = {"id": "doc_1", "content": "Deterministic content"}
        h1 = updater._hash_document(doc)
        h2 = updater._hash_document(doc)
        assert h1 == h2

    def test_hash_uses_sha256(self, updater):
        doc = {"id": "doc_1", "content": "test"}
        expected = hashlib.sha256("test".encode()).hexdigest()
        assert updater._hash_document(doc) == expected

    def test_hash_uses_page_content_fallback(self, updater):
        doc = {"id": "doc_1", "page_content": "fallback content"}
        h = updater._hash_document(doc)
        expected = hashlib.sha256("fallback content".encode()).hexdigest()
        assert h == expected

    def test_state_persisted_to_disk(self, tmp_path):
        from src.knowledge_graph.incremental_updater import IncrementalKGUpdater
        state_file = tmp_path / "state.json"
        updater1 = IncrementalKGUpdater(state_file=state_file)
        updater1.get_delta([{"id": "doc_1", "content": "persisted"}])

        updater2 = IncrementalKGUpdater(state_file=state_file)
        to_process, unchanged = updater2.get_delta([{"id": "doc_1", "content": "persisted"}])
        assert len(unchanged) == 1, "Second instance should see doc as unchanged"

    def test_reset_clears_state(self, updater):
        docs = [{"id": "doc_1", "content": "some content"}]
        updater.get_delta(docs)
        updater.reset()
        to_process, unchanged = updater.get_delta(docs)
        assert len(to_process) == 1, "After reset, doc should be new again"

    def test_mark_processed(self, updater):
        updater.mark_processed(["doc_42", "doc_43"])
        # These IDs now have placeholder hashes — real content will differ → to_process
        to_process, unchanged = updater.get_delta([{"id": "doc_42", "content": "real content"}])
        assert len(to_process) == 1

    def test_mixed_new_and_unchanged(self, updater):
        docs = [
            {"id": "a", "content": "alpha"},
            {"id": "b", "content": "beta"},
        ]
        updater.get_delta(docs)

        more_docs = [
            {"id": "a", "content": "alpha"},        # unchanged
            {"id": "b", "content": "beta updated"}, # changed
            {"id": "c", "content": "gamma"},         # new
        ]
        to_process, unchanged = updater.get_delta(more_docs)
        assert len(unchanged) == 1
        assert len(to_process) == 2
        processed_ids = {d["id"] for d in to_process}
        assert "b" in processed_ids
        assert "c" in processed_ids


class TestKnowledgeGraphBuilderFinalize:
    def test_finalize_build_runs_property_graph_index(self):
        from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder

        builder = object.__new__(KnowledgeGraphBuilder)
        builder.pg_builder = SimpleNamespace(
            build=lambda docs: {"index": "ok", "count": len(docs)}
        )
        builder.graph_store = SimpleNamespace(neo4j="neo4j-client")

        result = builder.finalize_build([
            {"id": "doc-1", "content": "Beam trip", "collection": "entry_logbook"}
        ])

        assert result["property_graph_index_built"] is True
        assert result["property_graph_index_documents"] == 1
