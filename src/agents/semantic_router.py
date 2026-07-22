"""
Semantic Router Agent — context-aware database routing using dynamic schema ingestion.

Task 2: SOTA Semantic Router Prompt with CoT reasoning and strict JSON output.
Routes user queries to the correct database/collection based on provided metadata schemas.

Designed for integration into the Agentic-RAG pipeline as an alternative or
complement to the rule-based QueryRouter. This router uses schema descriptions
to make routing decisions, making it adaptable to schema changes without code edits.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import get_prompt

logger = logging.getLogger(__name__)




# ──────────────────────────────────────────────────────────────────────────────
# Default the facility Schema Descriptions (injected into the prompt)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DATABASE_SCHEMAS = """
## Database: entry_logbook
- **Display Name:** the logbook
- **Description:** Scientific experiments, beam operations, shift notes, machine studies, and accelerator operational logbook entries.
- **Data Format:** Structured logbook entries with fields: entry_id, author, creation_datetime, content (free-text), tags, subfolder (categorization).
- **Temporal Coverage:** Continuous daily entries from the facility operations.
- **Typical Queries:** "What happened during yesterday's shift?", "Who was on shift when the beam went down?", "What experiments ran last week?", beam status, injection events, fault reports from operators.
- **Key Entities:** Operators (usernames), systems (SR, Booster, Linac), beam parameters, shift reports, experiment notes.

## Database: teams_messages
- **Display Name:** Teams Messages
- **Description:** Microsoft Teams communications, discussions, and meeting notes from the facility operations and engineering teams.
- **Data Format:** Messages with fields: author, content, timestamp, channel/thread context.
- **Temporal Coverage:** Recent team communications.
- **Typical Queries:** "What did the team discuss about the kicker issue?", "Were there any decisions made about the upgrade?", "Who mentioned the BPM calibration?"
- **Key Entities:** Team members, discussion topics, decisions, action items.

## Database: icms_collection
- **Display Name:** ICMS Technical Documents
- **Description:** Integrated Configuration Management System — 87,000+ technical documents including design notes, procedures, specifications, reports, engineering change notices, and safety documentation.
- **Data Format:** Documents with fields: file_id, filename, author, content, creation_datetime, document_type, revision.
- **Temporal Coverage:** Historical archive spanning years of the facility documentation.
- **Typical Queries:** "What is the procedure for undulator alignment?", "Show me the design specification for the SR RF system", "Find the technical report on chromaticity correction", procedure lookups, specification sheets, engineering documentation.
- **Key Entities:** Technical procedures, design specifications, system documentation, engineering reports.

## Database: work_requests
- **Display Name:** Work Request System
- **Description:** Maintenance work orders, equipment repair records, component replacements, and preventive maintenance tracking.
- **Data Format:** Structured records with fields: WRQ_NO, DESCRIPTION, SUBMISSION_DATE, SUBMITTED_BY, STATUS (open/closed/in-progress), PRIORITY, ASSIGNED_TO, COMPLETION_DATE.
- **Temporal Coverage:** Ongoing maintenance records.
- **Typical Queries:** "How many work requests for kicker repairs?", "What maintenance was done on the vacuum system?", "List open work requests for sector 7", counting queries, maintenance history, equipment status.
- **Key Entities:** Work request numbers (WR-xxx), equipment names, maintenance categories, technician assignments.

## Database: sdds_collection
- **Display Name:** SDDS Toolkit Documentation
- **Description:** Self-Describing Data Sets (SDDS) toolkit documentation — command-line tools, API reference, data format specifications, and usage examples for the SDDS scientific data framework.
- **Data Format:** Documentation files with fields: filename, file_type, content, creation_datetime.
- **Temporal Coverage:** Toolkit documentation (version-based, not time-series).
- **Typical Queries:** "How do I use sddsplot?", "What is the SDDS binary format?", "Show me the sddsprocess command options", code/API questions, data format questions, tool usage.
- **Key Entities:** SDDS commands (sddsplot, sddsprocess, sddsconvert, etc.), file formats, API functions.

## Database: oag_collection
- **Display Name:** OAG Applications
- **Description:** OAG Tcl/Tk GUI application documentation — application manuals, script references, and GUI tool documentation.
- **Data Format:** Application documentation with fields: app_metadata, rag_summary, main_actions, parameters.
- **Temporal Coverage:** Application documentation (version-based).
- **Typical Queries:** "How do I use the OAG orbit correction GUI?", "What OAG scripts are available for tune measurement?", "Show me the OAG application for BPM analysis", Tcl/Tk tool questions, GUI documentation.
- **Key Entities:** OAG applications, Tcl/Tk scripts, GUI tools, analysis programs.

## Meta-Database: all_databases
- **Description:** Federated search across ALL of the above collections. Use this when the query spans multiple domains or is too broad/ambiguous to route to a specific collection.
- **Typical Queries:** Broad questions, cross-domain inquiries, queries where the data source is unclear.
"""


class SemanticRouter:
    """
    Context-aware Semantic Router Agent.

    Uses dynamic schema ingestion and CoT reasoning to route user queries
    to the correct database/collection. Designed as a drop-in replacement
    or complement to the rule-based QueryRouter.

    Args:
        llm: LLM instance for routing decisions (fast model recommended).
        database_schemas: Optional custom schema descriptions. If None,
            uses DEFAULT_DATABASE_SCHEMAS for the facility collections.
    """

    def __init__(self, llm, database_schemas: Optional[str] = None):
        self.llm = llm
        self.schemas = database_schemas or DEFAULT_DATABASE_SCHEMAS

    def route(self, query: str) -> Dict[str, Any]:
        """
        Route a user query to the appropriate database/collection.

        Args:
            query: The user's natural language query.

        Returns:
            Dict with routing decision including target_database,
            target_collection, confidence_score, reasoning, and
            extracted_entities.
        """
        prompt = get_prompt("SEMANTIC_ROUTER_PROMPT").format(database_schemas=self.schemas)

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Route this query:\n\n{query}"),
        ]

        try:
            response = self.llm.invoke(messages)
            raw = response.content
            routing = self._parse_routing(raw)
        except Exception as e:
            logger.warning(f"Semantic router failed: {e} — using fallback")
            routing = {
                "routing_type": "fallback",
                "target_database": "all_databases",
                "target_collection": None,
                "confidence_score": 0.1,
                "reasoning": f"Router error: {e}",
                "extracted_entities": [],
                "suggested_retrieval_strategy": "HYBRID_3WAY",
            }

        logger.info(
            f"SemanticRouter: type={routing.get('routing_type')} | "
            f"target={routing.get('target_database', routing.get('targets', []))} | "
            f"confidence={routing.get('confidence_score', routing.get('overall_confidence', 0))} | "
            f"strategy={routing.get('suggested_retrieval_strategy', 'HYBRID_3WAY')}"
        )

        return routing

    def route_to_state(self, query: str) -> Dict[str, Any]:
        """
        Route a query and return state-compatible output matching AgentState
        fields used by the orchestrator.

        Returns dict with: selected_databases, retrieval_strategy, extracted_entities.
        """
        routing = self.route(query)
        routing_type = routing.get("routing_type", "fallback")

        if routing_type == "single":
            collections = [routing["target_database"]]
        elif routing_type == "multi":
            collections = [
                t["target_database"] for t in routing.get("targets", [])
            ]
        else:
            collections = ["all_databases"]

        return {
            "selected_databases": collections,
            "retrieval_strategy": routing.get(
                "suggested_retrieval_strategy", "HYBRID_3WAY"
            ),
            "extracted_entities": routing.get("extracted_entities", []),
            "routing_confidence": routing.get(
                "confidence_score", routing.get("overall_confidence", 0.0)
            ),
            "routing_reasoning": routing.get("reasoning", ""),
        }

    def _parse_routing(self, raw: str) -> Dict[str, Any]:
        """Extract JSON routing decision from LLM response."""
        raw = raw.strip()
        # Strip <routing_analysis> CoT block
        raw = re.sub(
            r"<routing_analysis>.*?</routing_analysis>", "", raw, flags=re.DOTALL
        ).strip()
        # Strip markdown code fences
        if raw.startswith("```"):
            raw = re.sub(r"```\w*", "", raw).strip()

        # Find JSON object
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error in semantic router: {e}")

        return {
            "routing_type": "fallback",
            "target_database": "all_databases",
            "target_collection": None,
            "confidence_score": 0.1,
            "reasoning": "Failed to parse routing response",
            "extracted_entities": [],
            "suggested_retrieval_strategy": "HYBRID_3WAY",
        }
