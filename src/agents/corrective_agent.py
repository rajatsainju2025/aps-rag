"""
Corrective agent — key paper contribution.
Validates answer quality and triggers re-retrieval when needed.

Checks:
1. Relevance grade each retrieved doc (LLM: relevant/partial/irrelevant)
2. Hallucination: cross-verify answer claims against evidence
3. KG entity validation: verify named entities exist in Neo4j
4. Trigger re-planning if confidence < 0.4 or >50% docs irrelevant
"""
import logging
import re
from typing import Dict, Any, List

from src.utils.json_parser import parse_json_with_repair

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentState
from config.prompts import get_prompt

logger = logging.getLogger(__name__)


class CorrectiveAgent:
    """
    Corrective RAG agent — evaluates answer quality and triggers correction.
    Core novel contribution: KG-validated corrective loops.
    """

    def __init__(self, llm, neo4j_client=None, quality_threshold: float = 0.4,
                 irrelevant_threshold: float = 0.5, max_rounds: int = 3):
        self.llm = llm
        self.neo4j = neo4j_client
        self.quality_threshold = quality_threshold
        self.irrelevant_threshold = irrelevant_threshold
        self.max_rounds = max_rounds

    def evaluate(self, state: AgentState) -> Dict[str, Any]:
        """
        Evaluate draft answer quality and decide next step.
        Returns updated state with confidence_score and needs_correction.
        """
        query = state.get("enhanced_query") or state.get("original_query", "")
        draft = state.get("draft_answer_raw") or state.get("draft_answer", "")
        docs: List[Document] = state.get("retrieved_docs", [])
        correction_round = state.get("correction_round", 0)

        # Force-exit if max rounds reached — fail-closed with degraded status
        if correction_round >= self.max_rounds:
            logger.warning(
                f"Max correction rounds ({self.max_rounds}) reached — "
                f"accepting draft with DEGRADED quality status"
            )
            return {
                "final_answer": draft,
                "confidence_score": max(0.3, state.get("confidence_score", 0.3)),
                "needs_correction": False,
                "quality_status": "max_rounds_exhausted",
            }

        # Step 1: LLM quality evaluation — evaluate the same doc slice the
        # reasoning agent showed the synthesis LLM (top 20). Mismatched slices
        # cause phantom "source not found" critiques on valid citations.
        context_str = self._format_context(docs[:20])
        memory_context = state.get("memory_context", {})
        mem_str = self._format_memory(memory_context)

        corrective_prompt = get_prompt("CORRECTIVE_PROMPT")
        messages = [
            SystemMessage(content=(
                "You are a rigorous quality evaluator for a RAG system serving "
                "the facility accelerator physicists and operators. Be honest and precise — "
                "do not inflate scores. Your evaluation directly determines answer quality."
            )),
            HumanMessage(
                content=corrective_prompt.format(
                    question=query, draft=draft, context=context_str,
                    memory_context=mem_str,
                )
            ),
        ]

        try:
            response = self.llm.invoke(messages)
            evaluation = self._parse_json(response.content)
        except Exception as e:
            logger.error(f"Corrective evaluation failed: {e}")
            # Fail-closed: do NOT silently accept an unverified draft
            if correction_round + 1 < self.max_rounds:
                return {
                    "confidence_score": 0.0,
                    "needs_correction": True,
                    "correction_round": correction_round + 1,
                    "quality_status": "evaluator_error",
                    "correction_issues": [f"Evaluator error: {e}"],
                }
            # Last resort: accept but flag as unverified so UI can warn
            return {
                "final_answer": draft,
                "confidence_score": 0.0,
                "needs_correction": False,
                "quality_status": "unverified",
            }

        overall_score = float(evaluation.get("overall_score", 0.6))
        relevance_fraction = float(evaluation.get("relevance_fraction", 1.0))
        refined_query = evaluation.get("refined_query")

        # Step 2: KG entity validation using LLM-extracted entities
        kg_issues = self._validate_kg_entities_llm(draft, state) if self.neo4j else []

        # Apply KG penalty: each unverified entity deducts from score
        kg_penalty = len(kg_issues) * 0.05
        overall_score = max(0.0, overall_score - kg_penalty)
        if kg_issues:
            logger.info(
                f"KG validation: {len(kg_issues)} issue(s), "
                f"penalty={kg_penalty:.2f}, adjusted score={overall_score:.2f}"
            )

        # Decision logic
        needs_correction = (
            overall_score < self.quality_threshold
            or (1.0 - relevance_fraction) > self.irrelevant_threshold
            or len(kg_issues) >= 3  # 3+ unverified entities forces re-retrieval
        )

        logger.info(
            f"Corrective eval: score={overall_score:.2f}, "
            f"relevance={relevance_fraction:.2f}, "
            f"needs_correction={needs_correction}, "
            f"round={correction_round + 1}/{self.max_rounds}"
        )

        # Update reasoning trace for UI thinking display
        reasoning_trace = state.get("reasoning_trace", [])
        issues = evaluation.get("issues", [])
        if needs_correction and refined_query:
            # Escalate retrieval strategy per correction round
            escalated_strategy = self._escalate_strategy(
                state.get("retrieval_strategy", "HYBRID_3WAY"), correction_round
            )
            all_issues = issues + [f"KG: {ki}" for ki in kg_issues]
            reasoning_trace.append(
                f"Corrective round {correction_round + 1}: score={overall_score:.2f}, "
                f"strategy→{escalated_strategy}, "
                f"re-retrieving — {'; '.join(all_issues[:3]) if all_issues else 'low confidence'}"
            )
            return {
                "confidence_score": overall_score,
                "needs_correction": True,
                "correction_round": correction_round + 1,
                "enhanced_query": refined_query,
                "retrieval_strategy": escalated_strategy,
                "reasoning_trace": reasoning_trace,
                "correction_issues": all_issues,
                "quality_status": "needs_correction",
                # Clear stale query_variants so the retrieval agent
                # regenerates fresh variants from the refined_query
                # instead of reusing the previous round's variants.
                "query_variants": [],
            }
        else:
            quality_status = "verified" if not kg_issues else "verified_with_kg_warnings"
            reasoning_trace.append(
                f"Quality check passed: score={overall_score:.2f}, "
                f"relevance={relevance_fraction:.2f}"
                + (f", kg_issues={len(kg_issues)}" if kg_issues else "")
            )
            return {
                "final_answer": draft,
                "confidence_score": overall_score,
                "needs_correction": False,
                "reasoning_trace": reasoning_trace,
                "quality_status": quality_status,
            }

    def _validate_kg_entities_llm(self, text: str, state: AgentState) -> List[str]:
        """Check named entities in the answer against Neo4j KG.

        Uses LLM-extracted entities from state (more accurate than regex),
        falling back to KG fulltext search for any unrecognized entities.
        """
        issues = []
        if not self.neo4j:
            return issues

        # Use entities already extracted by the router (more reliable than regex)
        candidates = state.get("extracted_entities", [])

        # Also check any entities mentioned in the answer that look like the facility systems/components
        # Use known the facility patterns: abbreviations, known system names
        aps_pattern = re.compile(
            r'\b(SR|BPM|PSU|RF|IK\d+|AMC\d+|kicker|undulator|[A-Z]{2,5}[-\s]\d+|'
            r'Linac|Booster|[A-Z][a-z]+-\d+[A-Z]?)\b'
        )
        text_entities = aps_pattern.findall(text)
        candidates = list(set(candidates + text_entities))

        for candidate in candidates[:8]:  # Check up to 8 entities
            try:
                results = self.neo4j.fulltext_search(str(candidate), limit=1)
                if not results:
                    issues.append(f"Entity not found in KG: {candidate}")
            except Exception:
                pass
        return issues

    def _escalate_strategy(self, current_strategy: str, correction_round: int) -> str:
        """Escalate retrieval strategy on successive correction rounds.

        Round 0→1: Keep strategy, broaden query — EXCEPT KG_ONLY, which
                   re-retrieves the identical doc set on the same query
                   (graph traversal is deterministic), making the round a
                   no-op that doubles latency without changing the answer.
                   For KG_ONLY, escalate immediately to HYBRID_3WAY.
        Round 1→2: Switch to complementary strategy
                   (but never to KG_ONLY — it returns 0 docs when there
                    are no extracted entities, which tanks the score)
        Round 2→3: Use HYBRID_3WAY (broadest coverage)
        """
        if correction_round == 0:
            if current_strategy == "KG_ONLY":
                return "HYBRID_3WAY"
            return current_strategy  # First correction: just broaden query
        elif correction_round == 1:
            # Switch to complementary strategy — but avoid KG_ONLY
            # because it returns nothing without extracted entities.
            escalation_map = {
                "HYBRID_3WAY": "VECTOR_ONLY",
                "KG_ONLY": "HYBRID_3WAY",
                "VECTOR_ONLY": "HYBRID_3WAY",
                "CODE_EXEC": "HYBRID_3WAY",
            }
            return escalation_map.get(current_strategy, "HYBRID_3WAY")
        else:
            return "HYBRID_3WAY"  # Final round: maximum coverage

    def _format_context(self, docs: List[Document]) -> str:
        # Must match the reasoning agent's context format, otherwise the
        # evaluator sees `[1]..[10]` while the draft cites `[source_N]` with
        # N up to 20 — and reports phantom "source not found" for every
        # citation past index 10.
        from src.utils.citation_formatter import format_context_for_llm
        if not docs:
            return "No context"
        return format_context_for_llm(docs)

    def _format_memory(self, memory_context: Dict[str, Any]) -> str:
        """Build memory context string for corrective prompt."""
        if memory_context and memory_context.get("has_context"):
            parts = []
            if memory_context.get("short_term"):
                parts.append(f"[Recent conversation]\n{memory_context['short_term']}")
            if memory_context.get("long_term"):
                parts.append(f"[Related entities]\n{memory_context['long_term']}")
            if memory_context.get("episodic"):
                parts.append(f"[Session summary]\n{memory_context['episodic']}")
            return "\n\n".join(parts)
        return "No prior conversation context."

    def _parse_json(self, raw: str) -> Dict:
        """Parse JSON evaluation result with repair fallback."""
        result = parse_json_with_repair(raw, llm=self.llm, context="corrective")
        return result if isinstance(result, dict) else {}
