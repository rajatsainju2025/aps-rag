"""Reasoning agent — Thinking-mode COT synthesis with self-reflective critique.

Generates draft answer from retrieved context using chain-of-thought prompting.
Includes an internal self-critique gate (gpt-4o-mini) that evaluates the draft
before it reaches the external CorrectionAgent — reducing unnecessary corrective
loop iterations.

Models:
  - synthesis_llm (gpt-4o): primary answer synthesis
  - critique_llm  (gpt-4o-mini): fast internal quality gate
"""
import logging
from typing import Dict, Any, List, Optional, Mapping

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from .answer_auditor import AnswerAuditor
from .state import AgentState
from ._tool_schema_render import render_tool_schemas
from config.prompts import get_prompt
from src.utils.citation_formatter import (
    build_source_map,
    format_context_for_llm,
    format_answer_with_citations,
)

logger = logging.getLogger(__name__)


class ReasoningAgent:
    """
    Thinking-mode COT synthesis agent with self-reflective critique.

    Two-model architecture:
      - llm (gpt-4o): primary synthesis — deep reasoning, CoT
      - critique_llm (gpt-4o-mini): internal quality gate — binary pass/fail

    The self-critique step evaluates the draft *before* the external CorrectionAgent,
    catching obvious issues (unanswered questions, missing citations, verbosity)
    at ~1/10th the cost of a full corrective loop iteration.
    """

    def __init__(self, llm, critique_llm=None, auditor_llm=None,
                 tools: Optional[Dict[str, Any]] = None):
        self.critique_llm = critique_llm or llm
        # AnswerAuditor: evidence-based critique of `data_analysis` count
        # results. Fires only on counting queries when the executed pandas
        # code emitted a {count, matched_sample, near_miss_sample} dict;
        # otherwise no-op. Reuses the fast critique LLM when caller doesn't
        # pass an explicit auditor_llm.
        _aud_llm = auditor_llm or self.critique_llm
        self._auditor = AnswerAuditor(_aud_llm) if _aud_llm else None
        # Render once at construction — tool registry is fixed at startup.
        # Kept as an empty-string fallback for the SYNTHESIS_PROMPT template
        # placeholder; native tool calling no longer uses this block.
        self._tool_schemas_block = ""

        # Stage B: bind tools natively when available. The LLM emits structured
        # `response.tool_calls` instead of in-text `TOOL_CALL:` directives, which
        # eliminates the regex-extraction failure modes (the self-critique gate
        # was rewriting drafts to remove TOOL_CALL: lines, killing tool loops).
        # If the LLM doesn't support `bind_tools` (e.g. legacy
        # `ArgoLLMClientAG` without native tool calling), fall through to the
        # legacy prompt-based path: keep the rendered tool schemas block and
        # rely on `_extract_tool_calls(draft)` for backward compatibility.
        self.llm = llm
        self._tools_bound_natively = False
        # Gated variant: same tool set minus `data_analysis`. Used for any
        # non-counting query — the LLM cannot reach for data_analysis on a
        # lookup/temporal query because the schema isn't in its toolbox.
        self._llm_without_data_analysis = None
        if tools and getattr(llm.__class__, "supports_tool_calls", False):
            try:
                self.llm = llm.bind_tools(list(tools.values()))
                self._tools_bound_natively = True
                if "data_analysis" in tools:
                    gated = [t for k, t in tools.items() if k != "data_analysis"]
                    self._llm_without_data_analysis = llm.bind_tools(gated)
                logger.info(
                    "ReasoningAgent: bound %d tool(s) natively via %s "
                    "(data_analysis gated on counting_intent: %s)",
                    len(tools), type(llm).__name__,
                    self._llm_without_data_analysis is not None,
                )
            except Exception as exc:
                logger.warning(
                    "ReasoningAgent: bind_tools failed (%s) — "
                    "falling back to prompt-based tool dispatch",
                    exc,
                )
                self._tool_schemas_block = render_tool_schemas(tools or {})
        elif tools:
            # LLM doesn't support native tool calling — fall back to the
            # prompt-based path that renders schemas in the synthesis prompt.
            self._tool_schemas_block = render_tool_schemas(tools or {})

    def synthesize(self, state: AgentState) -> Dict[str, Any]:
        """Generate draft answer with internal self-critique gate.

        Flow:
          1. LLM synthesis (gpt-4o) → draft with <think> CoT block
                    2. Self-critique (gpt-4o-mini) → pass/fail + rewrite hint
          3. If fail: re-synthesize with hint injected (one retry only)
          4. Extract tool calls and citations
        """
        query = state.get("enhanced_query") or state.get("original_query", "")
        docs: List[Document] = state.get("retrieved_docs", [])
        memory_context = state.get("memory_context", {})
        tool_results = state.get("tool_results", [])

        # ── Auditor gate ────────────────────────────────────────────────
        # When (a) we have an auditor, (b) this is a counting query, (c) the
        # latest tool_result was a successful data_analysis call, and (d) we
        # haven't already burned our one retry — ask the auditor to inspect
        # the {count, matched_sample, near_miss_sample} payload. If the
        # auditor wants a retry, short-circuit synthesis and re-emit a single
        # data_analysis tool_call with the prior code + auditor guidance
        # prepended as a comment.
        audit_retry = self._maybe_audit(state, query, tool_results)
        if audit_retry is not None:
            return audit_retry

        # Format context
        context = self._format_context(docs)
        mem_str = self._format_memory(memory_context)
        tool_section = self._format_tool_results(tool_results)
        plan_tool_section = self._format_plan_tool_instructions(state, tool_results)
        counting_section = self._format_counting_directive(state, tool_results)
        from datetime import date as _date
        today_iso = _date.today().isoformat()

        # Temporal disclosure: when the temporal filter widened beyond the
        # strict window, surface that to the LLM so it does not present
        # off-window docs as if they were in-range. retrieval_agent.py populates
        # state["temporal_stage"]; default to "strict" when absent.
        from src.utils.temporal_filter import build_temporal_context_block
        _tc = state.get("temporal_context") or {}
        temporal_info = build_temporal_context_block(
            _tc.get("start"),
            _tc.get("end"),
            stage=state.get("temporal_stage") or "strict",
        )

        synthesis_prompt = get_prompt("SYNTHESIS_PROMPT")

        # Counting/aggregation directive lives in the SystemMessage, not the
        # user message: thinking models (Opus, GPT-5) follow SystemMessage
        # rules strictly, but treat user-message addenda as advisory and will
        # ask for permission instead of acting ("Do you want me to run..."
        # rather than just running it). Empty string when not a counting query.
        system_text = (
            "You are a senior accelerator physicist and controls engineer at the facility, "
            "your organization. You reason step-by-step inside <think> blocks, "
            "cite every claim, and never fabricate information beyond the provided context."
        )
        if counting_section:
            system_text += counting_section

        messages = [
            SystemMessage(content=system_text),
            HumanMessage(
                content=synthesis_prompt.format(
                    context=context,
                    memory_context=mem_str,
                    question=query,
                    today=today_iso,
                    temporal_info=temporal_info,
                ) + tool_section + plan_tool_section
            ),
        ]

        # Pick LLM variant: hide data_analysis from the toolbox unless
        # this is a counting/aggregation query.
        if self._llm_without_data_analysis is not None and not state.get("counting_intent"):
            effective_llm = self._llm_without_data_analysis
        else:
            effective_llm = self.llm

        try:
            response = effective_llm.invoke(messages)
            draft = response.content.strip() if response.content else ""
        except Exception as e:
            logger.error(f"Reasoning synthesis failed: {e}")
            response = None
            draft = f"I was unable to generate an answer due to: {e}"

        # --- Tool-call extraction (Stage B: native first, regex fallback) ---
        # Prefer the structured `response.tool_calls` populated by
        # NativeToolsArgoLLM; fall back to the legacy regex extractor for
        # text-completion LLMs that emit `TOOL_CALL: name(args)` directives in
        # the draft body. Either path produces the orchestrator's expected
        # `[{"tool": <name>, "args": <dict|str>}]` shape.
        native_tool_calls = self._tool_calls_from_response(response)
        if native_tool_calls:
            tool_calls = native_tool_calls
            tool_call_origin = "native"
        else:
            tool_calls = self._extract_tool_calls(draft)
            tool_call_origin = "regex" if tool_calls else "none"

        # Initialise reasoning trace early so self-critique can append to it
        reasoning_trace = state.get("reasoning_trace", [])
        round_num = state.get("correction_round", 0) + 1
        n_docs = len(docs)
        n_tools = len(tool_results)
        trace_msg = f"Synthesizing answer (round {round_num}) from {n_docs} documents"
        if n_tools:
            trace_msg += f" + {n_tools} tool result(s)"
        if tool_calls:
            tool_names = ", ".join(tc.get("tool", "?") for tc in tool_calls)
            trace_msg += f" → emitting {len(tool_calls)} tool call(s) [{tool_call_origin}]: {tool_names}"
        reasoning_trace.append(trace_msg)

        # --- Self-critique gate (gpt-4o-mini) ---
        # Skip when:
        #   1. Tool results from a prior round are present (tool flow handles
        #      its own re-synthesis after the result lands).
        #   2. The current draft is mid-flight (the LLM emitted tool_calls but
        #      hasn't seen results yet). Without this guard the critic sees
        #      "only a plan, no numeric answer", marks the draft FAIL, and
        #      rewrites it WITHOUT the tool calls — killing the tool loop.
        if not tool_results and not tool_calls and not state.get("skip_critique"):
            critique = self._self_critique(query, draft, len(docs))
            if critique:
                commentary = critique.get("commentary", "")
                passed = critique.get("pass", True)
                scores_str = (f"answerable={critique.get('answerable')}, "
                              f"cited={critique.get('cited')}, "
                              f"concise={critique.get('concise')}")
                reasoning_trace.append(
                    f"Self-critique [{'PASS' if passed else 'FAIL'}]: {scores_str}"
                    + (f" — {commentary}" if commentary else "")
                )
            if critique and not critique.get("pass", True):
                hint = critique.get("rewrite_hint", "")
                logger.info(f"Self-critique FAILED — rewriting. Hint: {hint}")
                # Re-synthesize with the critique hint injected
                rewrite_msg = (
                    f"\n\n[SELF-CRITIQUE FEEDBACK — address this in your rewrite]\n"
                    f"{hint}\n"
                    f"Rewrite your answer to fix the issue above. Keep <think> block."
                )
                messages[-1] = HumanMessage(
                    content=synthesis_prompt.format(
                        context=context,
                        memory_context=mem_str,
                        question=query,
                        tool_schemas=self._tool_schemas_block,
                        today=today_iso,
                        temporal_info=temporal_info,
                    ) + tool_section + rewrite_msg
                )
                try:
                    response = effective_llm.invoke(messages)
                    draft = response.content.strip() if response.content else ""
                    # Re-extract tool calls — the rewrite may have produced a
                    # response that now emits tool calls (or strips ones that
                    # were there). Without this, we'd return stale tool_calls
                    # from the pre-rewrite draft.
                    rewrite_native_calls = self._tool_calls_from_response(response)
                    if rewrite_native_calls:
                        tool_calls = rewrite_native_calls
                    else:
                        tool_calls = self._extract_tool_calls(draft)
                    logger.info("Self-critique rewrite complete")
                except Exception as e:
                    logger.warning(f"Rewrite after self-critique failed: {e} — using original draft")

        # Termination: trust the LLM to stop emitting tool calls when it has
        # enough information. The orchestrator caps the outer loop at
        # settings.agent.tool_chain_safety_cap as a safety net against runaway
        # chains. tool_calls was already extracted before the self-critique
        # gate (native first, regex fallback) — see _tool_calls_from_response.
        citations = self._extract_citations(docs)

        return {
            "draft_answer": draft,
            "citations": citations,
            "tool_calls": tool_calls,
            "reasoning_trace": reasoning_trace,
        }

    # ── Internal helpers ────────────────────────────────────────────────

    def _maybe_audit(
        self,
        state: AgentState,
        query: str,
        tool_results: List[Dict],
    ) -> Optional[Dict[str, Any]]:
        """Run AnswerAuditor on the latest data_analysis result.

        Returns a synthesize-shaped dict to short-circuit re-synthesis when
        the auditor requests a retry; returns None to let synthesis proceed
        normally (auditor unavailable, wrong intent, no data_analysis result,
        retry budget exhausted, or auditor approves).
        """
        if not self._auditor:
            return None
        if not state.get("counting_intent"):
            return None
        if not tool_results:
            return None
        if state.get("audit_rounds", 0) >= 1:
            return None
        last = tool_results[-1]
        if last.get("tool") != "data_analysis" or last.get("error"):
            return None
        executed_code = ""
        prior_args = last.get("args") or {}
        data_source = ""
        if isinstance(prior_args, dict):
            executed_code = str(prior_args.get("analysis_code") or "")
            data_source = str(prior_args.get("data_source") or "")
        verdict = self._auditor.audit(
            query=query,
            code=executed_code,
            tool_output=last.get("result", ""),
        )
        if not verdict.get("retry"):
            return None
        guidance = verdict.get("guidance") or "Widen the filter to catch entity name variants."
        # Hand the LLM the previous code + guidance comment block; the
        # data_analysis subprocess strips comments before safety checks so
        # this is purely informational. Include the dict-emit reminder so the
        # auditor can still inspect the rerun.
        revised_code = (
            f"# AUDITOR FEEDBACK: {guidance}\n"
            f"# Previous code (rewrite to address feedback above; emit the same\n"
            f"# {{count, matched_sample, near_miss_sample}} dict shape):\n"
            f"{executed_code}\n"
        )
        reasoning_trace = list(state.get("reasoning_trace") or [])
        reasoning_trace.append(f"Auditor → retry: {guidance}")
        # Returning tool_calls (not draft_answer) routes back through
        # tool_node → reasoning_node via the orchestrator's `check_tools`.
        return {
            "draft_answer": "",
            "citations": self._extract_citations(state.get("retrieved_docs", [])),
            "tool_calls": [{
                "tool": "data_analysis",
                "args": {"analysis_code": revised_code, "data_source": data_source},
            }],
            "reasoning_trace": reasoning_trace,
            "audit_rounds": state.get("audit_rounds", 0) + 1,
        }

    def _format_memory(self, memory_context: Dict[str, Any]) -> str:
        """Build rich memory context string from all 3 tiers."""
        if memory_context and memory_context.get("has_context"):
            mem_parts = []
            if memory_context.get("short_term"):
                mem_parts.append(f"[Recent conversation]\n{memory_context['short_term']}")
            if memory_context.get("long_term"):
                mem_parts.append(f"[Related entities discussed previously]\n{memory_context['long_term']}")
            if memory_context.get("episodic"):
                mem_parts.append(f"[Session summary from previous session]\n{memory_context['episodic']}")
            return "\n\n".join(mem_parts)
        return "No prior conversation context."

    def _format_tool_results(self, tool_results: List[Dict]) -> str:
        """Format tool execution results for injection into prompt.

        Sanitizes tool output to prevent prompt-injection attacks:
        removes known injection patterns, fake system messages, and
        instruction-override attempts from untrusted tool output.
        """
        if not tool_results:
            return ""
        parts = []
        for tr in tool_results:
            tool_name = tr.get("tool", "unknown")
            result = self._sanitize_tool_output(str(tr.get("result", "")))
            parts.append(f"TOOL: {tool_name}\nRESULT:\n{result}")
        return (
            "\n\nTool execution results (already executed — use these, "
            "do NOT re-emit tool calls for the same request).\n"
            "IMPORTANT: Present these results as natural language ONLY. "
            "Do NOT include any code, pandas snippets, or TOOL_CALL "
            "directives in your FINAL ANSWER.\n\n"
            + "\n---\n".join(parts)
        )

    @staticmethod
    def _format_plan_tool_instructions(state: Mapping[str, Any], tool_results: List[Dict]) -> str:
        """Inject explicit tool call instructions from the planner when PV tools are planned."""
        if tool_results:
            return ""
        plan_steps = state.get("plan_steps", [])
        pv_tools = {"channel_finder", "pv_read", "archiver_read", "pv_search"}
        planned_pv_steps = [
            s for s in plan_steps
            if isinstance(s, dict) and s.get("tool") in pv_tools
        ]
        if not planned_pv_steps:
            return ""
        lines = []
        for s in planned_pv_steps:
            tool = s["tool"]
            query = s.get("query", "")
            if tool == "pv_search":
                lines.append(f"TOOL_CALL: pv_search({query})")
            elif tool == "channel_finder":
                lines.append(f"TOOL_CALL: channel_finder({query})")
            elif tool == "pv_read":
                lines.append(f"TOOL_CALL: pv_read(<use PV addresses from channel_finder result>)")
            elif tool == "archiver_read":
                lines.append(f"TOOL_CALL: archiver_read(<PV addresses>||1h)")
        return (
            "\n\n[PLANNED PV TOOL CALLS — you MUST emit these TOOL_CALL directives "
            "before answering. Do NOT skip them even if retrieved documents contain "
            "related information.]\n" + "\n".join(lines)
        )

    @staticmethod
    def _format_counting_directive(state: Mapping[str, Any], tool_results: List[Dict]) -> str:
        """Force `data_analysis` tool use for counting/aggregation queries.

        Counting from retrieved snippets is unreliable: the LLM is a poor
        counter, the top-K cap (~20 docs) undercounts, and chunked retrieval
        can double-count entries. This directive runs once before any tool
        execution; once `data_analysis` results land in `tool_results`, it
        falls silent so the LLM uses the deterministic count.
        """
        if not state.get("counting_intent"):
            return ""
        # If data_analysis has already returned a result, don't re-instruct.
        for tr in tool_results or []:
            if tr.get("tool") == "data_analysis":
                return ""
        # Map selected database(s) to a data_source the tool understands.
        # Falls back to keyword inspection of the query text — important
        # because OpenWebUI sends selected_databases=["all_databases"] which
        # otherwise short-circuits to the generic fallback hint.
        selected = state.get("selected_databases") or []
        query_text = state.get("enhanced_query") or state.get("original_query") or ""
        ds = None
        for db in selected:
            db_l = str(db).lower()
            if "work_request" in db_l:
                ds = "work_requests"
                break
            if "entry" in db_l or "logbook" in db_l:
                ds = "entry_logbook"
                break
            if "teams" in db_l:
                ds = "teams_messages"
                break
        if ds is None:
            ql = query_text.lower()
            if any(k in ql for k in ("work request", "wrq", "work-request", "wr ")):
                ds = "work_requests"
            elif any(k in ql for k in ("logbook", "entry", "study", "beam study", "studies")):
                ds = "entry_logbook"
            elif "teams" in ql or "chat message" in ql:
                ds = "teams_messages"
        ds_hint = (
            f"Use `data_source=\"{ds}\"`."
            if ds
            else "Choose `data_source` from {work_requests, entry_logbook, teams_messages}."
        )
        return (
            "\n\n## COUNTING / AGGREGATION QUERY — MANDATORY TOOL USE\n"
            "The user is asking for a count or aggregate. You MUST call the "
            "`data_analysis` tool (it's bound to your tool list) before stating "
            "any numeric answer. Do not provide a number before the tool runs.\n\n"
            "FORBIDDEN behaviors (do NOT do these):\n"
            "  - Estimating from the retrieved [source_N] snippets (the top-K is "
            "capped and chunked retrieval undercounts).\n"
            "  - Asking the user 'Do you want me to run a database count?' — they "
            "already asked, just run it.\n"
            "  - Saying 'I cannot confirm from the retrieved records' — the tool is "
            "the source of truth, not the snippets.\n"
            "  - Listing the records you can see and inferring a partial count.\n\n"
            "REQUIRED behavior:\n"
            "  1. Inside <think>: identify columns + filters from the data_analysis "
            "tool's schema (visible in your bound-tools list).\n"
            f"  2. Call `data_analysis` with `analysis_code` (pandas that prints the count) "
            f"and `data_source`. {ds_hint}\n"
            "  3. The tool result will land in your tool_results — then write the "
            "final answer using that deterministic count."
        )

    @staticmethod
    def _sanitize_tool_output(text: str, max_length: int = 8000) -> str:
        """Strip prompt-injection patterns from untrusted tool output.

        Removes:
        - Instruction-override phrases ("ignore previous instructions", etc.)
        - Fake system/assistant role markers
        - Fake XML tags that mimic internal control flow
        - Excessive length (truncate to max_length)
        """
        import re

        # Truncate oversized output first
        if len(text) > max_length:
            text = text[:max_length] + "\n[...output truncated]"

        # Remove common prompt-injection patterns (case-insensitive)
        injection_patterns = [
            r"(?i)ignore\s+(all\s+)?previous\s+instructions?",
            r"(?i)disregard\s+(all\s+)?(prior|previous|above)\s+",
            r"(?i)you\s+are\s+now\s+",
            r"(?i)new\s+instructions?:\s*",
            r"(?i)system\s*:\s*",
            r"(?i)assistant\s*:\s*",
            r"(?i)human\s*:\s*",
            r"(?i)user\s*:\s*",
            r"(?i)<\s*/?(?:system|instruction|prompt|role|command)\s*>",
            r"(?i)\[INST\].*?\[/INST\]",
            r"(?i)<<\s*SYS\s*>>.*?<<\s*/SYS\s*>>",
        ]
        for pattern in injection_patterns:
            text = re.sub(pattern, "[FILTERED]", text)

        return text

    def _self_critique(self, query: str, draft: str, source_count: int) -> Optional[Dict]:
        """Internal quality gate using critique_llm (gpt-4o-mini).

        Returns dict with {answerable, cited, concise, pass, rewrite_hint, commentary}
        or a fail-closed default if critique errors (never silently passes).
        """
        try:
            critique_prompt = get_prompt("SELF_CRITIQUE_PROMPT")

            # Warn if draft is being truncated significantly
            draft_len = len(draft)
            max_critique_len = 4000
            if draft_len > max_critique_len:
                logger.info(
                    f"Self-critique: truncating draft from {draft_len} to "
                    f"{max_critique_len} chars for critique evaluation"
                )

            messages = [
                SystemMessage(content="You are a concise quality evaluator. Respond with JSON only."),
                HumanMessage(content=critique_prompt.format(
                    question=query,
                    draft=draft[:max_critique_len],
                    source_count=source_count,
                )),
            ]
            response = self.critique_llm.invoke(messages)
            raw = response.content.strip()

            # Parse JSON from response — try repair for robustness
            from src.utils.json_parser import parse_json_with_repair
            result = parse_json_with_repair(raw, llm=self.critique_llm, context="self_critique")
            if not isinstance(result, dict) or "pass" not in result:
                logger.warning(
                    "Self-critique returned non-dict or missing 'pass' key — "
                    "fail-closed: treating as FAIL"
                )
                return {"pass": False, "rewrite_hint": "Self-critique output was malformed; re-check answer quality."}
            commentary = result.get("commentary", "")
            scores_summary = (f"answerable={result.get('answerable')}, "
                             f"cited={result.get('cited')}, "
                             f"concise={result.get('concise')}, "
                             f"pass={result.get('pass')}")
            logger.info(f"Self-critique scores: {scores_summary}")
            if commentary:
                logger.info(f"Self-critique commentary: {commentary}")
            return result
        except Exception as e:
            # Fail-closed: if critique cannot run, assume the draft needs review
            logger.warning(f"Self-critique failed (fail-closed): {e}")
            return {"pass": False, "rewrite_hint": f"Self-critique unavailable ({e}); re-verify answer."}

    def _format_context(self, docs: List[Document], max_docs: int = 20) -> str:
        """Format retrieved documents as numbered context using centralized formatter."""
        if not docs:
            return "No context retrieved."
        return format_context_for_llm(docs[:max_docs])

    @staticmethod
    def _tool_calls_from_response(response: Any) -> List[Dict[str, Any]]:
        """Extract native tool_calls from an AIMessage and convert to the
        orchestrator's `[{"tool": <name>, "args": <dict>}]` format.

        Returns an empty list when:
          - response is None (synthesis raised),
          - response has no `tool_calls` attribute (text-only LLM), or
          - tool_calls is empty / None.

        The native shape from `NativeToolsArgoLLM` is
        `[{"id", "name", "args": dict, "type": "tool_call"}]` (uniform across
        OpenAI / Anthropic / Gemini after vendor-specific normalization).
        Tool names are mapped to the legacy `tool` key consumed by ToolExecutor.
        """
        if response is None:
            return []
        raw = getattr(response, "tool_calls", None)
        if not raw:
            return []
        out: List[Dict[str, Any]] = []
        for tc in raw:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name") or tc.get("tool")
            if not name:
                continue
            args = tc.get("args")
            if args is None:
                args = tc.get("arguments")  # defensive — pre-normalized OpenAI
            out.append({"tool": name, "args": args if args is not None else {}})
        return out

    def _extract_tool_calls(self, draft: str) -> List[Dict[str, Any]]:
        """Extract TOOL_CALL directives from draft answer.

        Supports two formats:
        - TOOL_CALL: calculator(2+2)
        - TOOL_CALL: data_analysis(pandas_code||data_source)

        Uses balanced-parenthesis matching to handle nested parens in code.
        """
        import re
        tool_calls = []

        # Find each TOOL_CALL: position and use balanced paren matching
        for m in re.finditer(r"TOOL_CALL:\s*(\w+)\(", draft):
            tool_name = m.group(1)
            start = m.end()  # char right after opening '('
            depth = 1
            i = start
            while i < len(draft) and depth > 0:
                if draft[i] == "(":
                    depth += 1
                elif draft[i] == ")":
                    depth -= 1
                i += 1
            if depth != 0:
                continue  # unbalanced — skip
            raw_args = draft[start:i - 1].strip()

            if tool_name == "data_analysis" and "||" in raw_args:
                parts = raw_args.rsplit("||", 1)
                tool_calls.append({
                    "tool": "data_analysis",
                    "args": {"analysis_code": parts[0].strip(), "data_source": parts[1].strip()},
                })
            elif tool_name == "data_analysis":
                # Pass an empty data_source so pydantic validation succeeds; the
                # tool's _run then returns the clear "Unknown data_source ''.
                # Available: [...]" error, which the agent can self-correct on
                # the next round. Without this branch, the call falls through
                # to the generic single-arg `else` and crashes with an opaque
                # ValidationError that is poor LLM feedback.
                tool_calls.append({
                    "tool": "data_analysis",
                    "args": {"analysis_code": raw_args, "data_source": ""},
                })
            elif tool_name == "archiver_read" and "||" in raw_args:
                parts = raw_args.rsplit("||", 1)
                tool_calls.append({
                    "tool": "archiver_read",
                    "args": {"pv_names": parts[0].strip(), "start": parts[1].strip()},
                })
            else:
                tool_calls.append({
                    "tool": tool_name,
                    "args": raw_args,
                })
        return tool_calls

    def _extract_citations(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Build citation list from retrieved documents using centralized formatter."""
        citations, _ = build_source_map(docs)
        return citations
