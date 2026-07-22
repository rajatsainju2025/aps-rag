"""
Master LangGraph orchestrator.
Routes queries to appropriate service sub-graphs.
Implements the corrective ReAct loop.

Graph topology:
START → query_analyzer → service_router → {rag|agentic_rag|daily_summary|troubleshooting|studies}
      → response_formatter → END
"""
import logging
from typing import Any, Dict, List, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from src.utils.pv_utils import resolve_pvs_from_channel_finder
from .state import AgentState
from .router import QueryRouter
from .planner import PlannerAgent
from .retrieval_agent import RetrievalAgent
from .reasoning_agent import ReasoningAgent
from .corrective_agent import CorrectiveAgent
from .tool_executor import ToolExecutor
from .daily_summary_agent import DailySummaryAgent
from .troubleshooting_agent import TroubleshootingAgent
from .studies_copilot_agent import StudiesCopilotAgent
from .clarification_agent import ClarificationAgent

logger = logging.getLogger(__name__)
import os  # noqa: E402 — used by clarify_node kill-switch


def create_orchestrator(settings, components: Dict[str, Any]):
    """
    Build the master LangGraph StateGraph orchestrator.

    components: dict containing all initialized clients and agents:
      - fast_llm, synthesis_llm
      - vector_retriever, bm25_retriever, graph_retriever
      - fusion, reranker, query_enhancer
      - neo4j_client
      - memory_manager
      - tools dict
    """
    fast_llm = components["fast_llm"]
    synthesis_llm = components["synthesis_llm"]
    neo4j = components.get("neo4j_client")
    memory = components.get("memory_manager")

    # Initialize agents
    router = QueryRouter(llm=fast_llm)
    # Planner uses the native-tools /chat/ client when configured; falls back
    # to synthesis_llm (/streamchat/) so the agent works even without it.
    planner_llm = components.get("planner_llm") or synthesis_llm
    planner = PlannerAgent(llm=planner_llm)
    retrieval_agent = RetrievalAgent(
        vector_retriever=components["vector_retriever"],
        bm25_retriever=components["bm25_retriever"],
        graph_retriever=components["graph_retriever"],
        fusion=components["fusion"],
        reranker=components["reranker"],
        query_enhancer=components["query_enhancer"],
    )
    reasoning_agent = ReasoningAgent(
        llm=synthesis_llm,
        critique_llm=components.get("critique_llm", fast_llm),
        # Auditor reuses the fast critique LLM (gpt-4o-mini) — same cost class as
        # self-critique. Falls back to fast_llm if no critique_llm configured.
        auditor_llm=components.get("critique_llm", fast_llm),
        tools=components.get("tools", {}),
    )

    # Optional R2 path: prebuilt LangGraph create_react_agent.
    # Prefer the native-tools `tools_llm` (Argo /chat/) when available; fall
    # back to whatever `synthesis_llm` provides if its class advertises
    # `supports_tool_calls = True` (e.g. R2's structured-output adapter).
    react_executor = None
    react_llm = components.get("tools_llm")
    if react_llm is None and getattr(synthesis_llm.__class__, "supports_tool_calls", False):
        react_llm = synthesis_llm
    if settings.agent.use_create_react_agent and react_llm is not None:
        from .react_executor import ReactExecutor
        tools_dict = components.get("tools", {}) or {}
        react_executor = ReactExecutor(
            llm=react_llm,
            tools=list(tools_dict.values()),
            recursion_limit=settings.agent.tool_chain_safety_cap * 2,
        )
        logger.info(
            "create_react_agent path enabled — %d tools bound to %s",
            len(tools_dict),
            react_llm.__class__.__name__,
        )
    corrective_agent = CorrectiveAgent(
        llm=synthesis_llm,
        neo4j_client=neo4j,
        quality_threshold=settings.agent.quality_threshold,
        max_rounds=settings.agent.max_correction_rounds,
    )
    tool_executor = ToolExecutor(tools=components.get("tools", {}))
    # Pre-flight scope-ambiguity gate (counting queries only). Reuses the
    # critique LLM (gpt-4o-mini) — same cost class as auditor / self-critique.
    clarification_agent = ClarificationAgent(
        llm=components.get("critique_llm", fast_llm),
    )
    daily_summary = DailySummaryAgent(
        llm=synthesis_llm,
        bm25_retriever=components["bm25_retriever"],
        vector_retriever=components["vector_retriever"],
    )
    troubleshooting = TroubleshootingAgent(
        llm=synthesis_llm,
        neo4j_client=neo4j,
        vector_retriever=components["vector_retriever"],
        bm25_retriever=components["bm25_retriever"],
        graph_retriever=components["graph_retriever"],
    )
    studies_copilot = StudiesCopilotAgent(
        llm=synthesis_llm,
        vector_retriever=components["vector_retriever"],
        bm25_retriever=components["bm25_retriever"],
        neo4j_client=neo4j,
    )

    # --- Node functions ---

    def analyze_query(state: AgentState) -> Dict:
        """Enhance query; for non-naive_rag also routes in the same LLM call."""
        query = state["original_query"]
        enhancer = components.get("query_enhancer")
        if enhancer is None:
            from src.retrieval.query_enhancement import QueryEnhancer
            enhancer = QueryEnhancer()

        service_mode = state.get("service_mode", "")
        is_naive = service_mode == "naive_rag"
        # A/B switch for latency benchmarking: ENTRY_DISABLE_COMBINED_ROUTING=1
        # forces the legacy 3-call sequential path (router + temporal + variants).
        # ENTRY_DISABLE_COMBINED_PLAN=1 forces the dedicated PlannerAgent call
        # (keeps combined routing on but separates planning).
        disable_combined = os.environ.get("ENTRY_DISABLE_COMBINED_ROUTING") == "1"
        disable_plan = os.environ.get("ENTRY_DISABLE_COMBINED_PLAN") == "1"
        # Only agentic_rag uses combined route+plan (its node owns the planner call).
        # Other non-naive modes (troubleshooting, daily_summary, studies_copilot) keep
        # the routing-only combined call.
        use_combined_plan = (
            service_mode == "agentic_rag"
            and not disable_combined and not disable_plan
        )
        use_combined_routing = (not is_naive and not disable_combined and not use_combined_plan)
        enhancement = enhancer.enhance(
            query,
            combined=is_naive,
            combined_routing=use_combined_routing,
            combined_routing_with_plan=use_combined_plan,
        )

        mem_ctx = {}
        if memory:
            mem_ctx = memory.get_full_context(state.get("session_id", "default"))

        result: Dict[str, Any] = {
            "enhanced_query": enhancement["enhanced_query"],
            "temporal_context": enhancement.get("temporal_context"),
            "query_variants": enhancement.get("query_variants") or [enhancement["enhanced_query"]],
            "counting_intent": enhancement.get("counting_intent", False),
            "memory_context": mem_ctx,
            "iteration_count": 0,
            "correction_round": 0,
            "needs_correction": False,
        }

        # If combined routing ran, populate routing fields directly from enhancement
        # so dispatch_after_analysis can skip route_to_service.
        if enhancement.get("_routing_done"):
            intent = enhancement["_query_intent"]
            strategy = enhancement["_retrieval_strategy"]
            collections = enhancement["_selected_databases"]
            reasoning_trace = list(state.get("reasoning_trace") or [])
            reasoning_trace.append(
                f"Routing: intent={intent}, strategy={strategy}, "
                f"collections={collections}"
            )
            # Respect user-provided service_mode (same rule as route_to_service).
            llm_mode = enhancement["_service_mode"]
            effective_mode = service_mode if (service_mode and service_mode != "naive_rag") else llm_mode
            result.update({
                "query_intent": intent,
                "selected_databases": collections,
                "retrieval_strategy": strategy,
                "service_mode": effective_mode,
                "extracted_entities": enhancement["_extracted_entities"],
                "reasoning_trace": reasoning_trace,
            })

            # If combined route+plan also produced a plan, pre-populate it so
            # agentic_rag_node can skip the dedicated PlannerAgent call.
            if enhancement.get("_plan_done"):
                plan_steps = enhancement["_plan_steps"]
                n_steps = len(plan_steps)
                step_summary = ", ".join(
                    f"{s.get('tool', '?')}({(s.get('query') or '')[:30]})"
                    for s in plan_steps[:4]
                    if isinstance(s, dict)
                )
                reasoning_trace.append(
                    f"Planning (combined): {n_steps}-step plan for intent={intent} → {step_summary}"
                )
                result.update({
                    "plan": enhancement["_plan_raw"],
                    "plan_steps": plan_steps,
                    "reasoning_trace": reasoning_trace,
                })

        return result

    def dispatch_after_analysis(state: AgentState) -> str:
        """Route after analyze_query.

        naive_rag: skip router entirely (no graph, VECTOR_BM25 always).
        Other modes: if combined routing already populated query_intent, skip
        route_to_service and dispatch directly; otherwise fall through to LLM router.
        """
        if state.get("service_mode") == "naive_rag":
            return "naive_rag_node"
        if state.get("query_intent"):
            # Combined routing ran in analyze_query — dispatch without route_to_service
            return service_dispatch(state)
        return "route_service"

    def route_to_service(state: AgentState) -> Dict:
        """Route query to appropriate service + determine retrieval strategy.

        If the user already set a specific service_mode (via API), keep it.
        Otherwise use the router's auto-detected service_mode.
        """
        routing = router.route(state)

        # Respect user-provided service_mode if it was explicitly set
        user_mode = state.get("service_mode", "")
        if user_mode and user_mode != "naive_rag":
            # User explicitly chose a pipeline — don't override
            routing["service_mode"] = user_mode

        return routing

    def service_dispatch(state: AgentState) -> Literal[
        "naive_rag_node", "clarify_node", "agentic_rag_node", "daily_summary_node",
        "troubleshooting_node", "studies_node"
    ]:
        """Conditional edge: dispatch to correct service node.

        agentic_rag is gated through ``clarify_node`` first so scope-ambiguous
        counting queries can ask the user before retrieval/planning. The gate
        no-ops on resume turns and on non-counting queries.
        """
        mode = state.get("service_mode", "naive_rag")
        dispatch_map = {
            "naive_rag": "naive_rag_node",
            "agentic_rag": "clarify_node",
            "daily_summary": "daily_summary_node",
            "troubleshooting": "troubleshooting_node",
            "studies_copilot": "studies_node",
        }
        return dispatch_map.get(mode, "naive_rag_node")

    def clarify_node(state: AgentState) -> Dict:
        """Pre-flight scope-ambiguity gate for agentic_rag counting queries.

        Three exit shapes via ``post_clarify_router``:
          - resume turn (clarification_choice set) → pass through to agentic_rag_node
          - non-counting OR gate disabled OR LLM says unambiguous → pass through
          - LLM emits options → set clarification_pending; route to END so the
            service layer can render the question and persist it
        """
        if os.environ.get("ENTRY_DISABLE_CLARIFY") == "1":
            return {"clarification_pending": None}
        if state.get("clarification_choice"):
            return {"clarification_pending": None}
        if not state.get("counting_intent"):
            return {"clarification_pending": None}
        pending = clarification_agent.maybe_clarify(state.get("original_query", ""))
        return {"clarification_pending": pending}

    def post_clarify_router(state: AgentState) -> Literal["agentic_rag_node", "__end__"]:
        """Route to END when the gate emitted options; else proceed to agentic_rag."""
        return END if state.get("clarification_pending") else "agentic_rag_node"

    def naive_rag_node(state: AgentState) -> Dict:
        """Naive RAG: retrieve + synthesize (no corrective loop).

        Mirrors entry_rag_app's single-pass pipeline:
        preprocess → ensemble retrieve → rerank → prompt → LLM → format.
        If the LLM emits TOOL_CALL directives, execute them and
        re-synthesize so the user never sees raw code.

        Tool-chain termination is LLM-driven: loop continues while the LLM
        emits new TOOL_CALL directives, capped by tool_chain_safety_cap to
        bound runaway chains.
        """
        # naive_rag never uses graph; skip router + self-critique overhead
        state = {**state, "retrieval_strategy": "VECTOR_BM25", "skip_critique": True}
        retrieval_result = retrieval_agent.retrieve(state)
        state_with_docs = {**state, **retrieval_result}
        synthesis_result = reasoning_agent.synthesize(state_with_docs)

        for _ in range(settings.agent.tool_chain_safety_cap):
            if not synthesis_result.get("tool_calls"):
                break
            tool_state = {**state_with_docs, **synthesis_result}
            tool_result = tool_executor.execute(tool_state)
            state_with_docs = {**state_with_docs, **tool_result}
            synthesis_result = reasoning_agent.synthesize(state_with_docs)

        return {
            **retrieval_result,
            **synthesis_result,
            "draft_answer_raw": synthesis_result.get("draft_answer", ""),
            "final_answer": synthesis_result.get("draft_answer", ""),
        }

    def agentic_rag_node(state: AgentState) -> Dict:
        """Agentic RAG entry: plan + plan-guided retrieval.

        Consumes plan_steps from the planner to:
        1. Extract targeted collections and sub-queries from the plan
        2. Override retrieval parameters so retrieval follows the plan
        3. Fall back to default retrieval if plan parsing yields nothing
        """
        # Emit early so the streaming path opens <think> immediately,
        # giving the user a real-time "Planning..." indicator before any
        # tool results arrive. Each phase emit adds one line to the
        # collapsible reasoning trace AND keeps the SSE stream from going
        # idle long enough to trip upstream proxy timeouts (~30 s on the
        # your organization off-box proxy fronting your-rag-host.example.com).
        from .event_emitter import get_emitter as _get_emitter
        _get_emitter().emit("thinking_start", {"status": "Planning..."})

        # Skip the dedicated PlannerAgent call when analyze_query's combined
        # route+plan path already produced a plan (saves ~12s sonnet46 call).
        # Re-plan on correction rounds (combined plan is one-shot, can't react
        # to retrieval failure context).
        if state.get("plan_steps") and state.get("correction_round", 0) == 0:
            plan_result = {
                "plan": state.get("plan", ""),
                "plan_steps": state["plan_steps"],
                "reasoning_trace": state.get("reasoning_trace", []),
            }
        else:
            plan_result = planner.plan(state)
        state_with_plan = {**state, **plan_result}

        # Extract plan-guided retrieval overrides from structured plan_steps
        plan_steps = plan_result.get("plan_steps", [])
        plan_collections = set()
        plan_queries = []
        for step in plan_steps:
            if not isinstance(step, dict):
                continue  # guard: skip malformed items
            # Each step may have: {"step", "tool", "query", "collections", "rationale"}
            step_colls = step.get("collections", [])
            if isinstance(step_colls, list):
                plan_collections.update(step_colls)
            step_query = step.get("query")
            if step_query and isinstance(step_query, str):
                plan_queries.append(step_query)

        # Override state with plan-derived parameters if the plan produced actionable info
        if plan_collections:
            # Merge plan collections with router-selected ones (plan adds specificity)
            existing = set(state.get("selected_databases", []))
            merged = list(plan_collections | existing)
            # Don't override if plan just says 'all_databases'
            if "all_databases" not in plan_collections:
                state_with_plan["selected_databases"] = merged

        if plan_queries:
            # Inject plan sub-queries as additional query variants for multi-query retrieval
            existing_variants = state_with_plan.get("query_variants", [])
            # Deduplicate while preserving order
            seen = set(existing_variants)
            for pq in plan_queries:
                if pq not in seen:
                    existing_variants.append(pq)
                    seen.add(pq)
            state_with_plan["query_variants"] = existing_variants

        # Plan-step tool dispatch: deterministically execute any plan steps
        # whose tool is in the registered tools dict (channel_finder, pv_read,
        # archiver_read, etc.) before retrieval, so the synthesis prompt sees
        # real tool_results via reasoning_agent._format_tool_results.
        # Auto-chains channel_finder → pv_read / archiver_read by feeding the
        # resolved PV addresses into downstream calls (the planner emits
        # placeholder args like "<use PV addresses from channel_finder result>").
        tool_registry = components.get("tools", {}) or {}
        tool_status = components.get("tool_status", {}) or {}
        # `data_analysis` requires LLM-generated pandas code, not the planner
        # step's natural-language `query` field. Dispatch it at synthesis time
        # via `TOOL_CALL: data_analysis(<code>||<source>)` instead. Including
        # it here would send `args=<NL query>` → BaseTool wraps under the
        # first schema field (analysis_code) → data_source missing → pydantic
        # ValidationError. The reasoning_agent's _format_counting_directive
        # already nudges the LLM to emit a proper TOOL_CALL.
        _UPFRONT_DISPATCH_SKIP = {"data_analysis"}
        tool_steps = [
            s for s in plan_steps
            if isinstance(s, dict)
            and s.get("tool") in tool_registry
            and s.get("tool") not in _UPFRONT_DISPATCH_SKIP
        ]

        # Surface silently-skipped plan steps (tool referenced but not loaded).
        # Without this, the planner says "use pv_read" but the orchestrator
        # silently falls back to retrieval-only and the operator never knows
        # the live PV read didn't happen.
        # Exclude planner sentinels: `synthesize` is a no-op meaning "just
        # synthesize from retrieval"; `vector_search`/`bm25_search` describe
        # the retrieval modality and are handled by retrieval_node, not as
        # registered tools.
        _NON_TOOL_SENTINELS = {"synthesize", "vector_search", "bm25_search", "text_search"}
        skipped: List[Dict[str, str]] = []
        for s in plan_steps:
            if not isinstance(s, dict):
                continue
            tname = s.get("tool")
            if tname and tname not in tool_registry and tname not in _NON_TOOL_SENTINELS:
                reason = tool_status.get(tname, {}).get(
                    "reason", "tool not registered at startup"
                )
                skipped.append({"tool": tname, "reason": reason})
        if skipped:
            state_with_plan["tool_skipped"] = skipped
            trace = list(state_with_plan.get("reasoning_trace", []))
            for sk in skipped:
                trace.append(f"Tool skipped: {sk['tool']} ({sk['reason']})")
            state_with_plan["reasoning_trace"] = trace

        plan_tool_results: List[Dict] = []
        if tool_steps:
            cf_step = next((s for s in tool_steps if s.get("tool") == "channel_finder"), None)
            needs_chain = cf_step is not None and any(
                s.get("tool") in ("pv_read", "archiver_read") for s in tool_steps
            )
            resolved_pvs = ""

            if needs_chain:
                cf_dispatch = tool_executor.execute({
                    **state_with_plan,
                    "tool_calls": [{"tool": "channel_finder", "args": cf_step.get("query") or ""}],
                    "tool_results": [],
                })
                cf_results = cf_dispatch.get("tool_results", [])
                plan_tool_results.extend(cf_results)
                if cf_results and not cf_results[0].get("error"):
                    resolved_pvs = resolve_pvs_from_channel_finder(cf_results[0].get("result"))

            remaining_calls: List[Dict] = []
            for s in tool_steps:
                tname = s.get("tool")
                if tname == "channel_finder" and needs_chain:
                    continue  # already executed in the chaining pre-pass
                if tname == "pv_read" and resolved_pvs:
                    remaining_calls.append({"tool": "pv_read", "args": resolved_pvs})
                elif tname == "archiver_read" and resolved_pvs:
                    remaining_calls.append({
                        "tool": "archiver_read",
                        "args": {"pv_names": resolved_pvs, "start": "1h"},
                    })
                else:
                    remaining_calls.append({"tool": tname, "args": s.get("query") or ""})

            if remaining_calls:
                rest_dispatch = tool_executor.execute({
                    **state_with_plan,
                    "tool_calls": remaining_calls,
                    "tool_results": [],
                })
                plan_tool_results.extend(rest_dispatch.get("tool_results", []))

            n_ok = sum(1 for r in plan_tool_results if not r.get("error"))
            n_err = len(plan_tool_results) - n_ok
            tool_names = ",".join(sorted({r.get("tool", "?") for r in plan_tool_results}))
            trace = list(state_with_plan.get("reasoning_trace", []))
            msg = f"PV tool dispatch: {tool_names}, {n_ok} result(s)"
            if n_err:
                msg += f", {n_err} error(s)"
            trace.append(msg)
            state_with_plan["reasoning_trace"] = trace
            state_with_plan["tool_results"] = plan_tool_results

        retrieval_result = retrieval_agent.retrieve(state_with_plan)
        out: Dict[str, Any] = {**plan_result, **retrieval_result}
        if plan_tool_results:
            out["tool_results"] = plan_tool_results
            out["reasoning_trace"] = state_with_plan["reasoning_trace"]
        if skipped:
            out["tool_skipped"] = skipped
            out["reasoning_trace"] = state_with_plan["reasoning_trace"]
        return out

    def reasoning_node(state: AgentState) -> Dict:
        """Synthesize answer from retrieved context.

        When the prebuilt ReAct path is enabled (R2), drive the full
        LLM↔tool loop via `langgraph.prebuilt.create_react_agent` and
        return the final answer + tool_results in one shot. Otherwise
        fall back to the bespoke text-DSL synthesis loop.
        """
        if react_executor is not None:
            query = state.get("enhanced_query") or state.get("original_query", "")
            react_result = react_executor.run(question=query)
            # Return shape compatible with downstream check_tools / corrective_node:
            # tool_calls is empty so check_tools routes straight to corrective_node.
            return {
                "draft_answer": react_result["final_answer"],
                "draft_answer_raw": react_result["final_answer"],
                "final_answer": react_result["final_answer"],
                "tool_calls": [],
                "tool_results": react_result["tool_results"],
                "citations": [],
                "reasoning_trace": state.get("reasoning_trace", []) + [
                    f"create_react_agent loop: {len(react_result['tool_results'])} tool result(s)"
                ],
            }
        from .event_emitter import get_emitter as _get_emitter
        _get_emitter().emit("thinking_start", {"status": "Synthesizing answer..."})
        synth = reasoning_agent.synthesize(state)
        synth.setdefault("draft_answer_raw", synth.get("draft_answer", ""))
        return synth

    def check_tools(state: AgentState) -> Literal["tool_node", "corrective_node"]:
        """If tool calls present, run them (supports multi-round chaining).

        Termination is LLM-driven (empty tool_calls). The safety cap on
        accumulated tool_results bounds runaway chains regardless.
        """
        if not state.get("tool_calls"):
            return "corrective_node"
        if len(state.get("tool_results", [])) >= settings.agent.tool_chain_safety_cap:
            logger.warning(
                "Tool-chain safety cap (%d) reached — forcing termination",
                settings.agent.tool_chain_safety_cap,
            )
            return "corrective_node"
        return "tool_node"

    def tool_node(state: AgentState) -> Dict:
        """Execute tool calls and return results."""
        return tool_executor.execute(state)

    def corrective_node(state: AgentState) -> Dict:
        """Evaluate quality and decide if correction needed."""
        return corrective_agent.evaluate(state)

    def correction_router(state: AgentState) -> Literal["agentic_rag_node", "format_node"]:
        """Route: re-retrieve if needed, else format final answer."""
        if state.get("needs_correction") and state.get("correction_round", 0) < settings.agent.max_correction_rounds:
            return "agentic_rag_node"
        return "format_node"

    def daily_summary_node(state: AgentState) -> Dict:
        return daily_summary.summarize(state)

    def troubleshooting_node(state: AgentState) -> Dict:
        return troubleshooting.diagnose(state)

    def studies_node(state: AgentState) -> Dict:
        return studies_copilot.assist(state)

    def _format_with_reasoning_dropdown(raw_answer: str) -> str:
        """Move CoT reasoning into a collapsible block for OpenWebUI.

        Handles multiple LLM output formats:
          1. <think>…</think> — kept as-is; OpenWebUI renders natively
          2. <scratchpad>…</scratchpad> — normalized to <think>
          3. **REASONING:** … ---REASONING_END--- — wrapped in <think>
          4. **FINAL ANSWER:** — stripped to clean output

        OpenWebUI recognises <think>, <thinking>, <reason> etc. and renders
        them as collapsible "Thought" dropdowns automatically.

        Security: redacts sensitive internal metadata (memory summaries,
        tool traces, internal heuristics) from exposed reasoning blocks.
        """
        import re

        # Step 1: Normalize <scratchpad>/<thinking>/<routing_analysis> → <think>
        for tag in ("scratchpad", "thinking", "routing_analysis"):
            raw_answer = re.sub(
                rf"<{tag}>(.*?)</{tag}>",
                r"<think>\1</think>",
                raw_answer,
                flags=re.DOTALL,
            )

        # Step 1.5: Redact sensitive content inside <think> blocks
        def _redact_sensitive(match: re.Match) -> str:
            """Strip internal metadata from reasoning blocks before UI exposure."""
            content = match.group(1)
            # Remove memory context dumps
            content = re.sub(
                r"\[(?:Recent conversation|Related entities|Session summary)[^\]]*\]"
                r".*?(?=\n\n|\n[A-Z]|\Z)",
                "[memory context redacted]",
                content,
                flags=re.DOTALL,
            )
            # Remove tool execution traces
            content = re.sub(
                r"(?:TOOL_CALL|TOOL:|Tool execution).*?(?=\n\n|\Z)",
                "[tool trace redacted]",
                content,
                flags=re.DOTALL,
            )
            # Remove raw score/confidence internals
            content = re.sub(
                r"(?:confidence_score|quality_threshold|retrieval_strategy)\s*[:=]\s*[^\n]+",
                "",
                content,
            )
            return f"<think>{content.strip()}</think>"

        raw_answer = re.sub(
            r"<think>(.*?)</think>",
            _redact_sensitive,
            raw_answer,
            flags=re.DOTALL,
        )

        # Step 2: If <think> blocks already exist, just strip the FINAL ANSWER marker
        if "<think>" in raw_answer:
            cleaned = re.sub(
                r"\*{0,2}FINAL\s+ANSWER:\*{0,2}\s*",
                "",
                raw_answer,
                flags=re.IGNORECASE,
            ).strip()
            return cleaned

        # Step 3: Handle old **REASONING:** … ---REASONING_END--- format
        final_split = re.split(
            r"\*{0,2}FINAL\s+ANSWER:\*{0,2}\s*",
            raw_answer,
            maxsplit=1,
            flags=re.IGNORECASE,
        )

        if len(final_split) < 2:
            return raw_answer

        before_final = final_split[0]
        final_content = final_split[1].strip()

        # Strip old-format REASONING header and wrap in <think>
        reasoning_body = re.sub(
            r"^\s*\*{0,2}REASONING:\*{0,2}\s*",
            "",
            before_final,
            flags=re.IGNORECASE,
        ).strip()
        # Also strip ---REASONING_END--- marker if present
        reasoning_body = re.sub(
            r"---REASONING_END---\s*",
            "",
            reasoning_body,
        ).strip()

        if reasoning_body:
            return f"<think>\n{reasoning_body}\n</think>\n\n{final_content}"

        return final_content

    def _strip_tool_call_artifacts(text: str) -> str:
        """Remove TOOL_CALL directives and stray code fences from final answer.

        The reasoning agent may emit TOOL_CALL: lines or echo executed code.
        These are internal directives and must never reach the user.
        """
        import re

        # 1. Remove full TOOL_CALL: ... lines (balanced-paren aware)
        #    Match from 'TOOL_CALL:' to the closing ')' on the same or next lines
        text = re.sub(
            r"^\s*TOOL_CALL:\s*\w+\(.*?\)\s*$",
            "",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        # Greedy multi-line variant for data_analysis calls with || separator
        text = re.sub(
            r"TOOL_CALL:\s*data_analysis\(.*?\|\|.*?\)",
            "",
            text,
            flags=re.DOTALL,
        )
        # Catch any remaining single-line TOOL_CALL
        text = re.sub(r"TOOL_CALL:[^\n]*", "", text)

        # 2. Remove ```python ... ``` code fences that contain pandas/df operations
        #    (these are tool-code echoes, not user-requested code)
        def _is_tool_code(block: str) -> bool:
            tool_indicators = ["df[", "df.", ".str.contains", "pd.DataFrame",
                               "import pandas", ".to_string()", ".groupby(",
                               "filtered =", 'print(f"', "print(filtered"]
            return any(ind in block for ind in tool_indicators)

        def _replace_code_fence(m: re.Match) -> str:
            code = m.group(1)
            return "" if _is_tool_code(code) else m.group(0)

        text = re.sub(
            r"```(?:python)?\s*\n(.*?)```",
            _replace_code_fence,
            text,
            flags=re.DOTALL,
        )

        # 3. Collapse excessive blank lines left by removals
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def format_response(state: AgentState) -> Dict:
        """Format final response and update memory."""
        raw = (
            state.get("final_answer")
            or state.get("draft_answer_raw")
            or state.get("draft_answer", "")
        )
        answer = _format_with_reasoning_dropdown(raw)
        answer = _strip_tool_call_artifacts(answer)

        # Update short-term memory
        if memory and state.get("session_id"):
            session_id = state["session_id"]
            query = state.get("original_query", "")
            memory.add_turn(session_id, "user", query)
            memory.add_turn(session_id, "assistant", answer[:1000])

            # Write-back long-term + episodic memory
            try:
                entities = state.get("extracted_entities", [])
                if entities:
                    memory.update_long_term(
                        session_id=session_id,
                        entities=entities,
                        query=query,
                        answer_summary=answer[:500],
                    )
            except Exception as e:
                logger.error(f"Long-term memory write-back failed: {e}", exc_info=True)

            try:
                topics = state.get("extracted_entities", [])
                memory.store_episode(
                    session_id=session_id,
                    summary=f"Q: {query[:200]}\nA: {answer[:300]}",
                    topics=topics[:10],
                )
            except Exception as e:
                logger.error(f"Episodic memory write-back failed: {e}", exc_info=True)

        return {
            "final_answer": answer,
            "quality_status": state.get("quality_status", "verified"),
            "messages": [AIMessage(content=answer)],
        }

    # --- Build graph ---
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("route_service", route_to_service)
    graph.add_node("naive_rag_node", naive_rag_node)
    graph.add_node("clarify_node", clarify_node)
    graph.add_node("agentic_rag_node", agentic_rag_node)
    graph.add_node("reasoning_node", reasoning_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("corrective_node", corrective_node)
    graph.add_node("daily_summary_node", daily_summary_node)
    graph.add_node("troubleshooting_node", troubleshooting_node)
    graph.add_node("studies_node", studies_node)
    graph.add_node("format_node", format_response)

    # Edges
    graph.set_entry_point("analyze_query")
    # Combined routing may dispatch directly to any service node, so list all
    # possible destinations (same set as service_dispatch + "route_service" fallback).
    graph.add_conditional_edges("analyze_query", dispatch_after_analysis, {
        "naive_rag_node": "naive_rag_node",
        "clarify_node": "clarify_node",
        "agentic_rag_node": "agentic_rag_node",
        "daily_summary_node": "daily_summary_node",
        "troubleshooting_node": "troubleshooting_node",
        "studies_node": "studies_node",
        "route_service": "route_service",
    })
    graph.add_conditional_edges("route_service", service_dispatch)

    # Naive RAG path (simple single-pass)
    graph.add_edge("naive_rag_node", "format_node")

    # Pre-flight clarification gate — END when options emitted, else proceed.
    graph.add_conditional_edges("clarify_node", post_clarify_router)

    # Agentic RAG path (with corrective loop)
    graph.add_edge("agentic_rag_node", "reasoning_node")
    graph.add_conditional_edges("reasoning_node", check_tools)
    graph.add_edge("tool_node", "reasoning_node")  # Tool results → re-synthesize with results
    graph.add_conditional_edges("corrective_node", correction_router)

    # Domain agent paths
    graph.add_edge("daily_summary_node", "format_node")
    graph.add_edge("troubleshooting_node", "format_node")
    graph.add_edge("studies_node", "format_node")

    # End
    graph.add_edge("format_node", END)

    return graph.compile()
