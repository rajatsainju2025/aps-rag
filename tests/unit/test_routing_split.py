"""Phase 4 routing verification — counting vs semantic queries.

The defensive auditor only fires when state["counting_intent"] is True. This
test pins the regex-based router so a future refactor doesn't accidentally
flip the intent on us:

  - Aggregation/count queries → counting_intent=True (auditor eligible)
  - Semantic discovery queries → counting_intent=False (auditor must NOT fire)

This is verification only — no production-code change in Phase 4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.retrieval.query_enhancement import is_counting_query


# ── Counting queries: must fire data_analysis path ──────────────────────


@pytest.mark.parametrize("query", [
    # The Doe regression
    "How many unique the system entries is Jane Doe part of?",
    # Existing eval-bank patterns
    "How many work requests for kicker repairs in the last 6 months?",
    "How many unique the system entries mention MOBO?",
    "Number of completed work requests this year",
    "Total count of beam studies in 2025",
    "Count of incidents per machine",
    "Distribution of failures by component",
    "Average duration of work requests",
    "Median time to repair across machines",
    "Frequency of injection issues",
])
def test_counting_query_routes_to_data_analysis(query):
    assert is_counting_query(query), (
        f"Counting regex missed '{query}' — auditor will not fire on this"
    )


# ── Semantic queries: must NOT fire data_analysis path ─────────────────


@pytest.mark.parametrize("query", [
    # Free-text discovery — explicitly de-prioritized in plan Phase 4
    "Find logs where Doe mentions the server crash",
    # Diagnostic / status queries
    "What is the current status of the SR RF system?",
    "Why did beam lifetime drop yesterday?",
    "What happened with the SR RF system on April 15?",
    # Procedural / factual lookups
    "What is the energy of the facility storage ring?",
    "Who should I contact for IETS help?",
    "Show me the procedure for kicker recalibration",
    # Note: "Compare X to Y" deliberately omitted — humans read 'compare' as
    # semantic, but it shares vocabulary with quantitative queries and the
    # current router treats it as ambiguous. Pinning it either way locks us in
    # to a debatable choice; the auditor remains safe in either branch.
])
def test_semantic_query_does_not_route_to_data_analysis(query):
    assert not is_counting_query(query), (
        f"Counting regex spuriously matched semantic query '{query}' — "
        f"auditor would fire unnecessarily and burn an LLM call"
    )
