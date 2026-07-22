# Agent Harness, ReAct Loop & Tool-Calling Review

**Date:** 2026-04-27
**Reviewer:** internal review
**Scope:** `agentic-graph-rag/` orchestration, ReAct mechanics, tool dispatch and MCP integration
**Comparison set:** Anthropic Claude tool-use API, OpenAI function-calling / Responses API, LangGraph `create_react_agent` + `ToolNode`, MCP-native agents, smolagents/AutoGen patterns

---

## 1. Executive Summary

This codebase ships a working, opinionated agent harness with several genuinely strong design choices: a five-mode service selector, three-tier memory (short / long / episodic), MCP-backed control-system tools (live EPICS PVs + archiver), a corrective ReAct loop, and a self-critique gate. Compared to current SOTA, however, the **tool-calling layer is the weakest link** — it parses LLM output via regex instead of using a structured tool API, executes tools sequentially with a hard 2-round cap, and never shows the LLM tool schemas. The orchestration layer (LangGraph `StateGraph`) is solid but carries a 36-field `AgentState` that is hard to evolve. None of these are blockers; all are addressable incrementally without rewriting the system.

**Top five actions for this quarter, in priority order:**

1. **Add a tools-capable LLM adapter** alongside Argo (Claude Sonnet 4.6 / Opus 4.7 native tool-use) and route the `agentic_rag` and `troubleshooting` modes through it. Argo stays for `naive_rag` and synthesis.
2. **Replace the bespoke `TOOL_CALL:` text DSL with native tool-call binding** for the tools-capable path; keep the text DSL only as the Argo fallback.
3. **Enable parallel tool execution within a single turn** (e.g., `pv_read` + `archiver_read` simultaneously) — currently impossible.
4. **Lift the hard 2-round tool-chain cap into a confidence-driven termination** so legitimate multi-hop chains (`channel_finder → pv_read → archiver_read`) are not truncated.
5. **Stream tool events** (start / result / error) over SSE on the FastAPI side; operators triaging an active fault should not wait 30 s in silence.

The remaining sections back each of these claims with code references.

---

## 2. Current State

### 2.1 Orchestrator topology

LangGraph `StateGraph` constructed in [src/agents/orchestrator.py:449-486](../src/agents/orchestrator.py#L449-L486). The DAG has ten nodes: `analyze_query`, `route_service`, `naive_rag_node`, `agentic_rag_node`, `reasoning_node`, `tool_node`, `corrective_node`, `daily_summary_node`, `troubleshooting_node`, `studies_node`, `format_node`. Conditional edges in [orchestrator.py:128-141, 224-242](../src/agents/orchestrator.py#L128-L141) dispatch to one of five service modes and route between `tool_node` ↔ `reasoning_node` ↔ `corrective_node` based on the `tool_calls` and `needs_correction` flags. Iteration budget is enforced by three independent counters: `tool_round` (max 2, [reasoning_agent.py:137-142](../src/agents/reasoning_agent.py#L137-L142)), `correction_round` (max 3, [orchestrator.py:240](../src/agents/orchestrator.py#L240)), and `iteration_count` (max 10, [state.py:68](../src/agents/state.py#L68)).

### 2.2 Service modes

Five modes wired in [api/routes/query.py:49-68](../api/routes/query.py#L49-L68) — `naive_rag`, `agentic_rag`, `daily_summary`, `troubleshooting`, `studies_copilot`. Each maps to a service class in `src/services/` and ultimately to an orchestrator node. The router (`src/agents/router.py` and `src/agents/semantic_router.py`) is itself a small LLM agent (Claude Haiku 4.5) that emits a JSON intent + `service_mode` recommendation.

### 2.3 ReAct loop

The loop lives in [`reasoning_agent.py`](../src/agents/reasoning_agent.py) and is **text-DSL-based**. The synthesis LLM (`gpt-4o` via Argo) is prompted ([config/prompts/v1.py:133-152](../config/prompts/v1.py#L133-L152)) to emit lines of the form `TOOL_CALL: name(args)`. After `llm.invoke` returns, [`_extract_tool_calls`](../src/agents/reasoning_agent.py#L312-L357) walks the draft with a balanced-parenthesis matcher and pulls out tool calls. Two tools (`data_analysis`, `archiver_read`) use a `||`-separated arg convention because nested commas in pandas code break naive `,` splitting.

Observation feedback is handled by [`_format_tool_results`](../src/agents/reasoning_agent.py#L167-L188), which appends a `TOOL: <name>\nRESULT:\n<text>` block to the next prompt and admonishes the LLM not to re-emit the same call. Tool output is sanitized for prompt-injection patterns and truncated to 8 000 chars in [`_sanitize_tool_output`](../src/agents/reasoning_agent.py#L219-L252).

There is also a **dual-LLM self-critique gate** at [reasoning_agent.py:254-304](../src/agents/reasoning_agent.py#L254-L304) — `gpt-4o-mini` scores the draft on `answerable / cited / concise` and may trigger one rewrite. This is genuinely good design; very few RAG harnesses bother.

### 2.4 Tool registry and MCP

Tools are registered statically at startup in [api/main.py:156-239](../api/main.py#L156-L239). Eight tools land in the registry: `calculator`, `code_executor`, `data_analysis`, `graph_search`, `math_reasoning`, `channel_finder`, `pv_read`, `archiver_read`. Each is a LangChain `BaseTool` with a Pydantic `args_schema`. The PV-related three optionally delegate to an MCP server via [`MCPClientManager`](../src/tools/mcp_client.py), which spawns the MCP server as a subprocess over **stdio transport only** ([mcp_client.py:57-97](../src/tools/mcp_client.py#L57-L97)). [`.mcp.json`](../.mcp.json) wires two servers: `controls` (EPICS) and `archiver`. Notably, `MCPClientManager.list_tools()` exists ([mcp_client.py:114-130](../src/tools/mcp_client.py#L114-L130)) but is **never called at startup** — the LangChain wrappers are hand-coded with their schemas duplicated against the MCP server contract.

### 2.5 LLM backend

`ArgoLLMClientAG` in [src/integrations/argo/llm_client.py:39-128](../src/integrations/argo/llm_client.py#L39-L128) is a custom LangChain `LLM` over your organization's Argo proxy. `LLMFactory` (lines 130-203) routes by role: synthesis `gpt-4o`, routing `claude-3-5-haiku`, critique `gpt-4o-mini`, judge `claude-3-opus`. Argo accepts `messages`, `temperature`, `max_tokens` (or `max_completion_tokens` for `gpt5x`), `top_p`, and `stop`. It does **not** accept a `tools` or `functions` parameter, and does **not** emit `tool_use` blocks even when proxying Claude. The wrapper accumulates the full streamed response before returning ([llm_client.py:101-117](../src/integrations/argo/llm_client.py#L101-L117)) — wire-level streaming exists, but no token is exposed to the agent layer.

### 2.6 State schema

`AgentState` ([state.py:9-69](../src/agents/state.py#L9-L69)) is a flat `TypedDict` with **36 fields** spanning conversation, query, planning, retrieval, generation, quality control, routing, tools, memory, PV context, and control flow. Each node returns a partial dict that LangGraph merges into state. There is no namespacing.

---

## 3. Gap Analysis (8 Axes)

| # | Axis | Current | SOTA reference | Gap |
|---|---|---|---|---|
| 1 | Tool-call format | Regex-parsed `TOOL_CALL: name(args)` text directives | Anthropic `tool_use` blocks; OpenAI `tool_calls` JSON; LangGraph `bind_tools` | **Severe** |
| 2 | Parallel tool execution | Sequential, capped at 2 rounds | Anthropic returns ≥1 `tool_use` blocks per assistant turn, all run in parallel | **Material** |
| 3 | Schema validation surfaces | Pydantic on input, but LLM never sees the schema; errors as `{"error": True, "result": str(e)}` | `tool_result` with `is_error`; SDK validates against schema before dispatch | **Material** |
| 4 | Streaming | None at agent layer (`llm.invoke` + buffered Argo stream) | SSE/WebSocket token + tool-event streams (Anthropic, OpenAI) | **Material** |
| 5 | MCP integration depth | stdio-only; static hand-wrapped tools; `list_tools()` defined but unused | Dynamic discovery, SSE/HTTP transports, multi-server composition (Claude Code, Cline) | **Material** |
| 6 | Iteration termination | Three independent hard caps; no `end_turn` signal from LLM | `stop_reason="end_turn"` is the canonical terminator; budget is a safety net, not the main loop | **Material** |
| 7 | Routing as separate stage | Two routers (`router.py`, `semantic_router.py`) + planner — all *outside* the ReAct loop | LangGraph prebuilt + OpenAI Assistants treat routing as tool-choice inside one loop | **Minor** |
| 8 | State schema | Flat 36-field `AgentState` | Append-only message history + a small handful of first-class fields (`messages`, `remaining_steps`, `structured_response`) | **Material** |

### 3.1 Why the tool-call format gap is severe

The current loop hangs on the LLM to emit a syntactically clean `TOOL_CALL:` directive in the middle of natural-language prose, then strips it out before showing the answer to the user. Two failure modes are baked in:

- **Placeholder leakage.** The planner injection at [reasoning_agent.py:210](../src/agents/reasoning_agent.py#L210) emits `TOOL_CALL: pv_read(<use PV addresses from channel_finder result>)` into the prompt as a literal placeholder. If the LLM copies it verbatim, the regex extracts a tool call with the placeholder text as arguments and the dispatch fails.
- **Free-form arg parsing.** `data_analysis(pandas_code||data_source)` works only because pandas code rarely contains the literal string `||`. A model that hallucinates a different separator, or a tool that needs three arguments, has no escape hatch.

Both go away if the LLM emits structured tool calls and the harness binds the tools through a typed API. Anthropic's `tools` parameter and OpenAI's `tools=[{...}]` both accept JSON Schema and return validated tool calls — no string parsing on our side.

### 3.2 Why parallel execution matters here

A common the facility query — *"What is the beam current right now and how has it trended over the last hour?"* — needs `pv_read("S:SRcurrentAI")` and `archiver_read("S:SRcurrentAI||1h")` simultaneously. Today the LLM emits one call, waits, sees the result, emits the second, waits again. Two MCP server spawns over stdio, two Argo round-trips, two synthesis passes. With Anthropic's parallel `tool_use`, both calls land in one assistant turn, both run concurrently, and the second synthesis pass has both results in hand. Latency drops by roughly half on tool-heavy queries.

### 3.3 Why the 2-round cap is too rigid

The cap at [reasoning_agent.py:137-142](../src/agents/reasoning_agent.py#L137-L142) — `if tool_round >= 2: tool_calls = []` — was added to prevent runaway loops. But three-hop chains are legitimate: `channel_finder` (resolve "beam current" to PV name) → `pv_read` (current value) → `archiver_read` (1-hour trend) is a single user intent. A confidence-driven terminator (LLM returns `stop_reason="end_turn"` or emits zero tool calls after observation injection) is both safer and more flexible; the hard cap can stay as a final safety net at, say, 10 rounds.

### 3.4 Why schema-on-prompt helps even on Argo

Even if the LLM cannot return native `tool_use` blocks, showing it the full Pydantic JSON Schema for each tool — instead of the current free-text `TOOL_CALL: pv_read(<comma-separated PV names>)` — measurably improves call validity in published evals (smolagents, ToolBench). This is a one-file change in [config/prompts/v1.py:133-152](../config/prompts/v1.py#L133-L152) plus a small render util that calls `tool.args_schema.model_json_schema()`.

### 3.5 Why state-trimming is last

Every node and every test reads from `AgentState`. Refactoring it before the tool-calling layer is rebuilt risks two large rewrites in succession. Sequence: do the LLM-adapter and ReAct-loop work first, then trim state with the new tool-call message format already in place.

---

## 4. Prioritized Recommendations

### R1. Tools-capable LLM adapter (Claude / OpenAI) alongside Argo
**Files:** new `src/integrations/anthropic/llm_client.py`, switch in [src/integrations/argo/llm_client.py:130-203](../src/integrations/argo/llm_client.py#L130-L203)
**Effort:** ~2 days. **Risk:** Low — additive, behind a settings flag.
**What:** Implement `AnthropicLLMClient` (or OpenAI equivalent) as a LangChain `BaseChatModel` that supports `.bind_tools()`. Add `settings.llm.backend` ∈ `{"argo", "anthropic", "openai"}`. Default remains `argo`.
**Why:** Unlocks every other recommendation in this list. Until the LLM can emit structured tool calls, R2/R3/R6 are workarounds.
**Do this only if:** licensing / data residency permits sending the facility query text to Anthropic or OpenAI. If not, R3 (schema-on-prompt) becomes the highest-ROI item instead.

### R2. Replace bespoke ReAct loop with `langgraph.prebuilt.create_react_agent` for the tools-capable path
**Files:** [src/agents/reasoning_agent.py:312-357](../src/agents/reasoning_agent.py#L312-L357), [src/agents/orchestrator.py:449-486](../src/agents/orchestrator.py#L449-L486)
**Effort:** ~3 days. **Risk:** Medium — touches the corrective loop's interface.
**What:** When the active backend supports `bind_tools`, replace `reasoning_node` + `tool_node` + `check_tools` + the manual round-counter with a single `create_react_agent(llm, tools)` subgraph. The corrective loop wraps it; the text DSL stays as the Argo fallback.
**Why:** Eliminates the regex extractor, the balanced-paren matcher, the `||` arg convention, the placeholder-leakage failure mode, and the manual 2-round cap in one move.

### R3. Surface tool schemas in the Argo prompt
**Files:** [config/prompts/v1.py:133-152](../config/prompts/v1.py#L133-L152), new `src/agents/_tool_schema_render.py`
**Effort:** ~half a day. **Risk:** None.
**What:** Replace the free-text tool list with a rendered JSON Schema block: for each registered tool, call `tool.args_schema.model_json_schema()` and emit it under a `# Tool schemas` heading. Add a worked example per tool.
**Why:** Cheapest pure-Argo improvement. Reduces malformed tool calls and removes the `<use PV addresses from channel_finder result>` placeholder pattern.

### R4. Confidence-driven termination
**Files:** [src/agents/reasoning_agent.py:137-142](../src/agents/reasoning_agent.py#L137-L142)
**Effort:** ~half a day. **Risk:** Low — keep a 10-round safety net.
**What:** Replace `if tool_round >= 2: tool_calls = []` with: terminate iff (a) the LLM emits no tool calls after observation injection, OR (b) `tool_round >= settings.agent.tool_chain_safety_cap` (default 10).
**Why:** Unblocks legitimate three-hop chains; the safety cap still exists.

### R5. Stream tool events over SSE
**Files:** [api/main.py](../api/main.py), [api/routes/openai_compat.py](../api/routes/openai_compat.py), [src/agents/tool_executor.py](../src/agents/tool_executor.py)
**Effort:** ~2 days. **Risk:** Medium — UI integration depends on it.
**What:** Emit Server-Sent Events for `tool_call_start`, `tool_call_result`, `tool_call_error`, `synthesis_token` (where supported). Wire the FastAPI `/query` endpoint to stream them via `text/event-stream`.
**Why:** Operators triaging a 2 AM beam-loss don't want a 30 s blank screen. Even partial visibility into "channel_finder is running…" is a usability win.

### R6. Dynamic MCP tool discovery
**Files:** [src/tools/mcp_client.py:114-130](../src/tools/mcp_client.py#L114-L130), [api/main.py:193-234](../api/main.py#L193-L234)
**Effort:** ~1 day. **Risk:** Low.
**What:** At startup, call `MCPClientManager.list_tools(server)` for each server in `.mcp.json`. Dynamically generate a thin `BaseTool` subclass per discovered tool using the MCP server's reported schema. Keep the hand-coded fallback path for direct (non-MCP) execution.
**Why:** The MCP server contract is the source of truth. Today the LangChain wrappers duplicate it manually and silently drift if the server adds a parameter.

### R7. Trim `AgentState`
**Files:** [src/agents/state.py:9-69](../src/agents/state.py#L9-L69), every `*_node` in [src/agents/orchestrator.py](../src/agents/orchestrator.py)
**Effort:** ~3 days. **Risk:** High — touches every node and test.
**What:** Group ephemeral fields under namespaces: `state.routing`, `state.retrieval`, `state.tools`, `state.pv`. Promote `messages` to the canonical conversation log (`Annotated[List[BaseMessage], add_messages]`) and stop carrying `draft_answer`/`final_answer` as siblings.
**Why:** 36 first-class fields is unmanageable; a future contributor cannot tell which fields a node is allowed to write. Sequence: do this last so the new tool-call message format is already in place.

---

## 5. Argo vs Tools-Capable Backend Trade-off

Argo is the default LLM proxy at your organization and the system's only path to internal-policy-compliant inference. Replacing it for the agent path is a real trade-off, not a free upgrade:

| | Argo (today) | Anthropic / OpenAI native |
|---|---|---|
| Data residency | On-prem, your organization policy | External provider |
| Tool-use API | None — text only | Native `tool_use` / `tool_calls` |
| Parallel tool calls | No | Yes |
| Streaming | Wire-level, not exposed | Full SSE/event stream |
| Cost (per query) | Effectively zero (internal) | $0.003 – $0.05 depending on model |
| Latency | Comparable | Comparable, often faster |

The recommended posture is **dual backend**: Argo remains default for `naive_rag`, retrieval, planner, router, critique, judge, and synthesis. Anthropic / OpenAI gets switched on only for `agentic_rag` and `troubleshooting` *if* policy permits, behind `settings.llm.tools_backend`. This isolates the dependency and keeps the costly external traffic to the queries that actually need parallel structured tool-use.

---

## 6. Open Questions for the Team

1. **Argo external-traffic posture:** is sending the facility query text to Anthropic / OpenAI APIs permitted under the project's data-handling policy? Recommendations R1, R2, R5 hinge on this answer.
2. **Tool budget by mode:** is the 2-round cap a deliberate guardrail against runaway operations cost, or a leftover from early development? If the former, R4 needs an explicit per-mode override.
3. **OpenWebUI streaming contract:** the front-end wired through `api/routes/openai_compat.py` — does it currently consume SSE token streams, or only the final response? R5 effort estimate doubles if the front-end needs work too.
4. **MCP server roadmap:** are additional MCP servers planned beyond `controls` and `archiver` (e.g., a beamline-status server, a procedure-runner)? R6 is much higher value if the answer is yes.
5. **Self-critique gate cost:** the dual-LLM gate at [reasoning_agent.py:254-304](../src/agents/reasoning_agent.py#L254-L304) is great design — has its impact been measured in eval (rewrite-rate, correction-loop-skip rate)? If so, the metric should appear in `researchpaper.md`.

---

## Appendix A: Verification trail

Every claim about current behavior in this document was validated against the codebase during the review. The cited file ranges were read directly:

- `src/agents/orchestrator.py:90-260, 449-486`
- `src/agents/state.py:1-69` (full file)
- `src/agents/reasoning_agent.py:130-220, 254-357`
- `src/agents/tool_executor.py:18-50`
- `src/integrations/argo/llm_client.py:39-203`
- `src/tools/mcp_client.py:1-131` (full file)
- `src/utils/citation_formatter.py:60-170, 540-734`
- `config/prompts/v1.py:120-207`
- `api/main.py:156-239`
- `api/routes/query.py:49-68`
- `.mcp.json` (full file)
- `data/raw/icms/DOCU_1412275.json` (sample)

SOTA references rely on the public docs of Anthropic Messages API (tool-use), OpenAI Responses / function-calling, LangGraph `langgraph.prebuilt.create_react_agent` and `ToolNode`, and the Model Context Protocol specification (transports: stdio, SSE, streamable HTTP).
