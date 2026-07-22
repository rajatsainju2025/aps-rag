"""
Studies Copilot service — ReAct loop with code execution and math reasoning.

the system URL support: if the user's query contains one or more the logbook URLs
(https://your-host.example.com/entry/views/itemDomainLogbook/view?id=<digits>),
the service fetches those pages and injects their content as attachments into
the agent state before invoking the orchestrator. Only the exact the system URL
pattern is allowed; all other URLs are silently ignored.
"""
import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.services.base_service import BaseService
from src.utils.entry_fetcher import extract_entry_urls, fetch_entry_page, format_entry_attachment
from src.utils.citation_formatter import format_answer_with_citations

logger = logging.getLogger(__name__)


class StudiesCopilotService(BaseService):
    """Assists with beam physics studies — code execution, math, graph traversal."""

    service_mode = "studies_copilot"

    def __init__(self, settings: Any, components: Dict[str, Any]) -> None:
        self.settings = settings
        self.components = components
        self._orchestrator = None

    def _get_orchestrator(self):
        if self._orchestrator is None:
            from src.agents.orchestrator import create_orchestrator
            self._orchestrator = create_orchestrator(self.settings, self.components)
        return self._orchestrator

    def get_service_mode(self) -> str:
        return self.service_mode

    def _fetch_entry_attachments(self, query: str) -> tuple:
        """Extract the system URLs from the query, fetch each page, return formatted attachments.

        Returns:
            (attachments, failed_urls):
            - attachments: list of formatted content blocks for successfully fetched pages
            - failed_urls: list of (url, entry_id) tuples for pages that could not be read
        """
        found = extract_entry_urls(query)
        if not found:
            return [], []

        attachments: List[str] = []
        failed_urls: List[tuple] = []
        for url, entry_id in found:
            logger.info("Studies Copilot: fetching logbook entry id=%s", entry_id)
            try:
                content = fetch_entry_page(url)
                if content:
                    attachments.append(format_entry_attachment(url, entry_id, content))
                    logger.info("Studies Copilot: attached logbook entry id=%s (%d chars)", entry_id, len(content))
                else:
                    logger.warning("Studies Copilot: empty content for logbook entry id=%s", entry_id)
                    failed_urls.append((url, entry_id))
            except ValueError as e:
                logger.warning("Studies Copilot: the system URL rejected: %s", e)
                failed_urls.append((url, entry_id))
            except Exception as e:
                logger.warning("Studies Copilot: failed to fetch logbook entry id=%s: %s", entry_id, e)
                failed_urls.append((url, entry_id))

        return attachments, failed_urls

    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        start = time.time()
        orchestrator = self._get_orchestrator()

        # Fetch any the logbook pages linked in the query
        entry_attachments, failed_urls = self._fetch_entry_attachments(query)

        # If the user provided the system URLs but ALL of them failed to open,
        # return an early error message asking them to paste the page content.
        if failed_urls and not entry_attachments:
            failed_ids = ", ".join(
                f"[logbook entry {bid}]({url})" for url, bid in failed_urls
            )
            latency_ms = (time.time() - start) * 1000
            return {
                "answer": (
                    f"I was unable to read the the logbook page(s) you linked: {failed_ids}.\n\n"
                    "This can happen when the page requires authentication, the server is "
                    "temporarily unavailable, or the entry ID does not exist.\n\n"
                    "**Could you please copy and paste the page content directly into the chat?** "
                    "I'll be happy to analyse it once I can see the text."
                ),
                "code_output": "",
                "citations": [],
                "reasoning_trace": ["the system page fetch failed — asked user to paste content"],
                "service_mode": "studies_copilot",
                "latency_ms": round(latency_ms, 2),
            }

        if entry_attachments:
            logger.info(
                "Studies Copilot: %d the system attachment(s) fetched for this query",
                len(entry_attachments),
            )

        state = {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "enhanced_query": query,
            "session_id": session_id or "default",
            "service_mode": "studies_copilot",
            "selected_databases": kwargs.get("selected_databases", ["all_databases"]),
            "query_intent": "",
            "plan": "",
            "reasoning_trace": [],
            "retrieved_docs": [],
            "graph_context": [],
            "web_results": [],
            "retrieval_sources": [],
            "draft_answer": "",
            "final_answer": "",
            "citations": [],
            "confidence_score": 0.0,
            "needs_correction": False,
            "correction_round": 0,
            "tool_calls": [],
            "tool_results": [],
            "memory_context": {},
            "next_node": "",
            "error": None,
            "iteration_count": 0,
            "temporal_context": None,
            # the logbook pages fetched from URLs found in the user's query
            "entry_attachments": entry_attachments,
        }

        try:
            result = orchestrator.invoke(state)
        except Exception as exc:
            logger.error("StudiesCopilot query failed: %s", exc)
            result = {
                "final_answer": f"Error: {exc}",
                "citations": [],
                "retrieved_docs": [],
                "tool_results": [],
                "reasoning_trace": [],
            }

        latency_ms = (time.time() - start) * 1000
        code_output = ""
        for tr in result.get("tool_results", []):
            if isinstance(tr, dict) and tr.get("tool") == "code_executor":
                code_output = tr.get("output", "")
                break

        # Apply inline citation replacement + clickable links + footer
        raw_answer = result.get("final_answer", "")
        retrieved_docs = result.get("retrieved_docs", [])
        owui_citations = []
        if retrieved_docs:
            formatted_answer, citations, owui_citations = format_answer_with_citations(
                raw_answer, retrieved_docs
            )
        else:
            formatted_answer = raw_answer
            citations = result.get("citations", [])

        return {
            "answer": formatted_answer,
            "code_output": code_output,
            "citations": citations,
            "openwebui_citations": owui_citations,
            "sources": [d.metadata if hasattr(d, "metadata") else d for d in retrieved_docs],
            "reasoning_trace": result.get("reasoning_trace", []),
            "service_mode": "studies_copilot",
            "latency_ms": round(latency_ms, 2),
        }

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "service_mode": "studies_copilot"}
