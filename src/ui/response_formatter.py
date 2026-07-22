"""
Response formatting utilities for OpenWebUI and API responses.
"""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLDS = [
    (0.8, "🟢 High"),
    (0.6, "🟡 Medium"),
    (0.4, "🟠 Low"),
    (0.0, "🔴 Very Low"),
]


class ResponseFormatter:
    """Formats answers, citations, KG subgraphs, and confidence indicators."""

    def format_citations(self, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return ""
        lines = []
        for i, doc in enumerate(docs, start=1):
            author = doc.get("author") or doc.get("Author") or "Unknown"
            title = doc.get("title") or doc.get("Title") or doc.get("source", "Untitled")
            date = doc.get("date") or doc.get("Date") or doc.get("timestamp", "")
            lines.append(f"[{i}] {author}. *{title}*. {date}".strip().rstrip(".") + ".")
        return "\n".join(lines)

    def format_answer_with_citations(
        self,
        answer: str,
        docs: List[Dict[str, Any]],
    ) -> str:
        citation_block = self.format_citations(docs)
        if not citation_block:
            return answer
        return f"{answer}\n\n**References:**\n{citation_block}"

    def format_mermaid_kg_subgraph(
        self,
        nodes: List[Any],
        edges: List[Any],
    ) -> str:
        lines = ["```mermaid", "graph TD"]
        for node in nodes:
            if isinstance(node, dict):
                node_id = str(node.get("id", "")).replace(" ", "_")
                label = node.get("label") or node.get("name") or node_id
                lines.append(f'    {node_id}["{label}"]')
            else:
                safe = str(node).replace(" ", "_")
                lines.append(f'    {safe}["{node}"]')

        for edge in edges:
            if isinstance(edge, dict):
                src = str(edge.get("source", "")).replace(" ", "_")
                tgt = str(edge.get("target", "")).replace(" ", "_")
                rel = edge.get("relation") or edge.get("type", "")
                lines.append(f'    {src} -->|"{rel}"| {tgt}')
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                src = str(edge[0]).replace(" ", "_")
                tgt = str(edge[1]).replace(" ", "_")
                rel = str(edge[2]) if len(edge) > 2 else ""
                lines.append(f'    {src} -->|"{rel}"| {tgt}')

        lines.append("```")
        return "\n".join(lines)

    def format_confidence(self, score: float) -> str:
        for threshold, label in _CONFIDENCE_THRESHOLDS:
            if score >= threshold:
                return f"{label} ({score:.0%})"
        return f"🔴 Very Low ({score:.0%})"
