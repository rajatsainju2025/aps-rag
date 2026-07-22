# ReAct Agent System Evaluation
## Agentic-Graph-RAG (May 2026)

---

## Executive Summary

The agentic-graph-rag system implements a **sophisticated, multi-layered ReAct (Reasoning + Acting) architecture** with several novel contributions:

1. **Dual-path ReAct execution**: Native tool calling (LangGraph `create_react_agent`) AND legacy prompt-based dispatch
2. **Corrective loop with KG validation**: Quality gates trigger re-retrieval with escalating strategies
3. **Chain-of-Thought (CoT) synthesis** with mandatory `<think>` blocks (thinking-mode LLM compatible)
4. **Two-model internal critique**: Fast model (`gpt-4o-mini`) gates expensive corrections before corrective loop
5. **Plan-guided tool dispatch**: Planner pre-determines tool calls; orchestrator executes upfront before retrieval

**Strengths**: Comprehensive, enterprise-grade pipeline with safety nets and operator-friendly tracing.  
**Concerns**: Complexity creates multiple failure modes; tool integration paths are fragmented; thinking-mode adoption incomplete.

---

## 1. ReAct Architecture Overview

### 1.1 Core ReAct Pattern

The system implements **reasoning → action → observation loops** across three agent layers:

| Layer | Component | Role |
|-------|-----------|------|
| **L1: Planning** | PlannerAgent | Multi-step plan generation (structured JSON) |
| **L2: Reasoning** | ReasoningAgent | CoT synthesis + native tool binding |
| **L3: Acting** | ToolExecutor + ReactExecutor | Tool execution (prompt-based OR graph-based) |
| **L4: Correction** | CorrectiveAgent | Quality validation + re-retrieval triggering |

### 1.2 Two Execution Paths

#### Path A: Native Tool Calling (Recommended)
```
LLM (tool-capable) → ToolMessage → Bind tools natively
  ↓
Response.tool_calls (structured JSON)
  ↓
LangGraph ToolNode → ToolExecutor.execute()
  ↓
ToolMessage(result) → LLM (loop until no tool_calls)
```

**Activation condition:**  
```python
if settings.agent.use_create_react_agent and react_llm is not None:
    ReactExecutor(llm=react_llm, tools=[...], recursion_limit=50)
```

**Advantages:**
- Structured tool calling (no regex extraction)
- LLM-driven termination (matches R4 behavior)
- Events emit through shared emitter (streaming R5 compatible)

**Limitations:**
- Requires `supports_tool_calls = True` on LLM
- Hard dependency on LangGraph `create_react_agent` (or `langchain.agents.create_agent` v1.0+)
- Recursion safety capped at `tool_chain_safety_cap * 2` (default 50)

#### Path B: Prompt-based Tool Dispatch (Legacy)
```
LLM (text completion) → TOOL_CALL: name(args) in draft
  ↓
ReasoningAgent._extract_tool_calls(draft)  [regex extraction]
  ↓
ToolExecutor.execute() → tool_results
  ↓
Synthesis re-invoked with tool_results in state
  ↓
Loop until no TOOL_CALL directives remain
```

**Activation condition:**  
Default when native path unavailable or `supports_tool_calls = False`.

**Advantages:**
- Works with any text-completion LLM
- Backward compatible with legacy models
- Full control over extraction logic (no black-box LangGraph)

**Disadvantages:**
- Regex fragile — misses nested parentheses, special chars
- Manual loop management (error-prone)
- CoT blocks may corrupt `TOOL_CALL:` lines during re-critique

**Known Issue:** Self-critique gate (`gpt-4o-mini`) rewrites drafts to remove `TOOL_CALL:` lines, breaking tool loops → forced re-synthesis with CoT hint.

---

## 2. Reasoning Agent: Thinking-Mode CoT

### 2.1 Chain-of-Thought Protocol (Mandatory)

**Prompt location**: `config/prompts/v1.py::SYNTHESIS_PROMPT`

Every synthesis includes 9-step CoT inside `<think>`:
```
<think>
- Step 1: Query Intent [factual | temporal | comparative | multi-hop | code | counting | troubleshooting]
- Step 2: Source Inventory [iterate each retrieved doc, extract key facts]
- Step 3: Tool Assessment [need a tool? which one, why?]
- Step 4: Conflict & Gap Analysis [contradictions? missing info?]
- Step 5: Synthesis Plan [combine sources — chronological? causal? compare-contrast?]
- Step 6: Procedure Critique [known flaws? safety gaps? alternatives?]
- Step 7: Response Length [simple ≤150w | multi-hop ≤300w | troubleshooting ≤400w]
- Step 8: Self-Critique [re-read draft — all claims cited? hallucinations?]
- Step 9: Confidence Gate [0.0–1.0 score. if < 0.7, rewrite before emitting]
</think>
```

**Implementation** (`reasoning_agent.py::synthesize`):
```python
1. Format context (top 20 docs)
2. LLM.invoke(system_msg + synthesis_prompt)
   → response.content with embedded <think>
3. Self-critique via gpt-4o-mini
   → rewrite hint if score < 0.7 (ONE retry)
4. Extract tool_calls
   (native: response.tool_calls OR legacy: regex on draft)
5. Parse citations
6. Return {draft_answer, tool_calls, confidence_score}
```

### 2.2 Self-Critique Gate (Internal)

**Purpose**: Catch obvious issues (unanswered questions, missing citations, hallucinations) BEFORE expensive corrective loop.

**Config**: `SELF_CRITIQUE_PROMPT` (gpt-4o-mini, fast model)

**Scoring axes**:
- `answerable`: Does draft provide ANY reasonable answer? (Partial OK)
- `cited`: Are there [source_N] citations? (Imperfect OK)
- `concise`: Is it not wildly off-length? (Lenient)

**Decision logic**:
```python
if all(score >= 0.4 for score in [answerable, cited, concise]):
    pass = True  # Accept draft
else:
    pass = False  # Rewrite with hint
    draft = llm.invoke(synthesis_prompt + rewrite_hint)
```

**Cost model**: ~1/10th of corrective loop cost (same model tier, simpler prompt).

### 2.3 Tool Binding Strategies

#### Native Tool Binding (Recommended)
```python
if hasattr(llm, 'supports_tool_calls') and supports_tool_calls:
    self.llm = llm.bind_tools(list(tools.values()))
    self._tools_bound_natively = True
```

**Effect**: LLM emits `response.tool_calls` (structured), bypasses regex extraction.

#### Prompt-based Fallback
```python
self._tool_schemas_block = render_tool_schemas(tools)  # Rendered at construction
# Injected into SYNTHESIS_PROMPT template as {tool_schemas}
```

**Tools rendered** (from `_tool_schema_render.py`):
- `channel_finder`: Map device names → EPICS PV addresses
- `pv_read`: Fetch live PV values
- `archiver_read`: Historical timeseries from archiver
- `data_analysis`: Pandas-based filtering/counting on work_requests, the logbook
- `graph_search`: Neo4j KG queries
- `memory_read/write`: Session context persistence
- `image_search`: Multimodal retrieval (sample images, plots)

---

## 3. Planning & Tool Dispatch

### 3.1 Planner Agent

**Purpose**: Generate multi-step research plans BEFORE retrieval, enabling pre-routing of tool calls.

**Activation**: All non-naive-rag paths invoke `planner.plan(state)`.

**Output structure** (JSON):
```json
{
  "plan": "human-readable prose plan",
  "plan_steps": [
    {
      "step": 1,
      "tool": "channel_finder",
      "query": "8ID power supply voltage",
      "collections": ["entry_logbook"],
      "rationale": "Resolve device to PV address"
    },
    {
      "step": 2,
      "tool": "pv_read",
      "query": "<use PV addresses from channel_finder result>",
      "rationale": "Fetch live value"
    },
    {
      "step": 3,
      "tool": "synthesize",
      "rationale": "Combine with retrieved context"
    }
  ]
}
```

### 3.2 Upfront Tool Dispatch (Orchestrator)

**Location**: `orchestrator.py::agentic_rag_node` (lines ~200–250)

**Flow**:
```python
plan_steps = planner.plan(state)
tool_steps = [s for s in plan_steps if s['tool'] in tool_registry]

for step in tool_steps:
    tool = tool_registry[step['tool']]
    if step['tool'] == 'channel_finder':
        result = tool.run(step['query'])
        state['pv_addresses'] = result['addresses']
        tool_results.append(result)
    elif step['tool'] == 'pv_read':
        pv_list = state['pv_addresses']  # Chain from channel_finder
        result = tool.run(pv_list)
        tool_results.append(result)
    # ... similar for archiver_read, data_analysis, etc.
```

**Data flow**:
1. Channel_finder → `state.pv_addresses`
2. PV_read uses `state.pv_addresses` as input
3. All results accumulated in `state.tool_results`
4. Passed to `reasoning_agent.synthesize(state_with_tool_results)`

### 3.3 Counting/Aggregation Forcing

**Problem**: Planner might suggest `synthesize` for counting queries; LLM estimates instead of actual counts.

**Solution** (`planner.py::plan` lines ~85–105):
```python
if state.get('counting_intent') and not any(
    s.get('tool') == 'data_analysis' for s in plan_steps
):
    plan_steps.insert(0, {
        'step': 0,
        'tool': 'data_analysis',
        'query': query,
        'data_source': inferred_source,  # work_requests | entry_logbook | teams_messages
        'rationale': 'Deterministic count'
    })
```

Additionally, `reasoning_agent.synthesize` injects a hard directive in the SystemMessage (via `_format_counting_directive`) when `counting_intent=True`:

```python
counting_section = """
For counting/aggregation queries: you MUST call data_analysis with the exact query.
Do NOT estimate. Emit: TOOL_CALL: data_analysis(
  analysis_code="...",
  data_source="work_requests"
)
"""
```

**Assessment**: Dual-layer forcing (planner + synthesis) is defensive but creates redundancy.

---

## 4. Corrective RAG Loop

### 4.1 CorrectiveAgent Design

**Location**: `src/agents/corrective_agent.py`

**Novel contribution**: KG-validated corrective loops (per research paper).

**Evaluation axes**:

1. **Relevance grading**: LLM classifies each doc as RELEVANT | PARTIAL | IRRELEVANT
   - Fraction = (RELEVANT + 0.5 × PARTIAL) / total_docs
   - Threshold: < 0.5 → re-retrieve

2. **Hallucination detection**: Extract claims from draft, verify against context
   - Unsupported claims → issues list
   - Multiplied into confidence deduction

3. **KG entity validation**: Extract named entities (device names, personnel), query Neo4j
   - Unverified entities → penalty (0.05 per entity)
   - ≥3 unverified → force re-retrieval

4. **Source conflict resolution**: Identify contradictions, tag more authoritative source
   - Logged in issues list

### 4.2 Decision Logic

```python
needs_correction = (
    overall_score < quality_threshold (0.4)
    OR relevance_fraction > irrelevant_threshold (0.5)
    OR kg_issues >= 3
)

if needs_correction:
    correction_round += 1
    if correction_round < max_rounds (3):
        escalate_retrieval_strategy()
        refined_query = evaluation['refined_query']
        goto retrieval_node
    else:
        return {
            'final_answer': draft,
            'quality_status': 'max_rounds_exhausted',
            'confidence_score': max(0.3, score)
        }
```

### 4.3 Retrieval Strategy Escalation

**Round 0** (initial): `HYBRID_3WAY` (vector + BM25 + graph fusion)

**Round 1** (first correction):
```
"Broaden query terms, add synonyms, expand date range, try alternate collection"
```
Strategy: Same `HYBRID_3WAY`, but with expanded query variants.

**Round 2** (second correction):
```
"Switch to fundamentally different strategy (e.g., graph→vector or vector→graph)"
```
Strategy: Flip to opposite (KG_ONLY if was VECTOR_ONLY, etc.).

**Round 3+** (desperate):
```
"Decompose into sub-questions, search all relevant collections"
```
Strategy: Switch to ALL collections; emit reasoning trace to operator.

### 4.4 Scoring Calibration

**Formula**:
```
overall_score = 0.4 × relevance_fraction 
              + 0.3 × factual_grounding 
              + 0.3 × completeness
              - kg_penalty
```

**Calibration notes** (from prompt):
- Partial answer addressing main question: ≥ 0.5
- Answer addressing question with some citations: ≥ 0.6
- Truly broken/off-topic: < 0.4
- Err on the side of accepting (avoid unnecessary corrections)

**Concerns**:
1. Hard weighting (0.4 / 0.3 / 0.3) not adaptive to query type
2. KG penalty (0.05/entity) can be harsh if entity extraction is noisy
3. Threshold (0.4) may be too lenient for safety-critical queries (e.g., emergency procedures)

---

## 5. State Management

### 5.1 AgentState TypedDict

**Location**: `src/agents/state.py`

**Breadth**: 30+ fields spanning conversation, planning, retrieval, generation, QC, tooling, memory.

**Critical fields**:
```python
# Conversation
messages: List[BaseMessage]                 # LangGraph message history
session_id: str
service_mode: str                           # naive_rag | agentic_rag | ...

# Query processing
original_query: str
enhanced_query: str
query_intent: str                           # factual | temporal | troubleshooting | ...
counting_intent: bool                       # Force data_analysis tool
selected_databases: List[str]               # Routed collections
query_variants: List[str]                   # Multi-query for hybrid retrieval

# Planning
plan: str                                   # Prose plan
plan_steps: List[Dict]                      # Structured steps (JSON)
reasoning_trace: List[str]                  # UI thinking display

# Retrieval
retrieved_docs: List[Any]                   # Top 20 LangChain Documents
graph_context: List[Dict]                   # KG traversal results
retrieval_sources: List[str]                # ["vector", "bm25", "graph"]

# Quality control
confidence_score: float                     # 0.0–1.0 from corrective agent
needs_correction: bool
correction_round: int                       # ≤ 3
correction_issues: List[str]                # Passed to planner for re-planning
quality_status: str                         # verified | unverified | max_rounds_exhausted

# Tool execution
tool_calls: List[Dict]                      # [{"tool": "name", "args": {...}}]
tool_results: List[Dict]                    # [{"tool": "name", "result": "...", "error": false}]
tool_skipped: List[Dict]                    # Planner referenced tool not loaded

# Memory
memory_context: Dict[str, Any]              # short_term | long_term | episodic
pv_readings: List[Dict]                     # Real-time Process Variable values
pv_history: List[Dict]                      # Historical archiver summaries
```

**Assessment**: Comprehensive state design supports multi-modal reasoning and retroactive auditing. No circular dependencies detected.

---

## 6. Testing & Evaluation Infrastructure

### 6.1 Component Latency Harness

**Location**: `evaluation/component_latency_harness.py` (458 lines)

**Scope**:
- 10 AR (Advanced Reasoning) queries × 4 planner models (90 runs baseline)
- 50 queries × 1 model = distribution analysis
- Component-level instrumentation (10 components wrapped)

**Wrapped components**:
1. `QueryRouter.route`
2. `QueryEnhancer.enhance`
3. `PlannerAgent.plan`
4. `RetrievalAgent.retrieve`
5. `AdaptiveHybridFusion.fuse`
6. `GraphProximityReranker.rerank` (or JinaCrossEncoderReranker)
7. `ReasoningAgent.synthesize`
8. `CorrectiveAgent.evaluate`
9. `format_answer_with_citations`
10. LLM endpoints (Argo /chat/ and /streamchat/)

**Instrumentation method**: Class-level monkey-patching with `time.perf_counter()` via ContextVar.

**Output**: CSV with per-query latency breakdown + aggregate statistics (mean, median, pstdev).

**Strengths**:
- Comprehensive component coverage
- No source code modification (monkey-patching)
- ContextVar isolation prevents cross-query contamination

**Limitations**:
- Monkey-patching fragile if methods refactored
- Assumes flat latency model (no nested timing deduplication)
- Doesn't capture LLM input token count → makes latency-token correlation opaque

### 6.2 Unit Tests

**Coverage**:
- `test_agents.py`: ReasoningAgent, CorrectiveAgent, planning
- `test_corrective_agent_extended.py`: KG validation, correction rounds
- `test_graph_search_tool.py`: KG query execution
- `test_native_tools_client.py`: Argo integration
- `test_tool_skipped_surfacing.py`: Plan step validation
- `test_counting_routing.py`: Counting intent routing

**Key test scenarios**:
```python
# test_agents.py
- ReasoningAgent.synthesize with/without tool calls
- Self-critique gate (pass/fail logic)
- Tool extraction (native vs regex)

# test_corrective_agent_extended.py
- Evaluation scoring (relevance, grounding, completeness)
- KG entity validation
- Re-retrieval triggering

# test_counting_routing.py
- Counting intent detection
- data_analysis tool injection
- Answer verification (counting_verifier.py)
```

**Assessment**: Good coverage of happy paths; edge cases in tool chaining & fallback paths under-tested.

---

## 7. Strengths & Achievements

### 7.1 Architectural Strengths

1. **Modular agent design**: Each agent (Router, Planner, ReasoningAgent, CorrectiveAgent) is independently testable.

2. **Dual-path tool execution**: Native + prompt-based dispatch provides fallback robustness.

3. **KG-validated correctionsloop**: Novel contribution — validates entity names against Neo4j, not just LLM hallucination checks.

4. **Plan-guided retrieval**: Planner pre-determines tool calls and collection selection, reducing LLM randomness.

5. **Comprehensive state shape**: `AgentState` captures full reasoning trajectory, enabling replay & debugging.

6. **Thinking-mode CoT support**: `<think>` blocks are LLM-aware (Opus, GPT-o1 compatible), not just documentation.

7. **Component latency instrumentation**: Non-invasive monitoring without source modification.

8. **Graceful degradation**: Max correction rounds (3) + quality_status flags prevent runaway loops.

### 7.2 Operational Features

1. **Reasoning trace for UI**: `reasoning_trace` list updated at each node, surfaces thinking to operators.

2. **Tool skipped surfacing**: Warns operators when planner referenced a tool not loaded.

3. **Procedure critique directive**: Prompts LLM to flag outdated/unsafe procedures proactively.

4. **Citation anti-bunching**: Enforces distributed [source_N] citations, not clustered.

5. **Query intent classification**: Router labels queries (factual | temporal | troubleshooting), enabling intent-specific handling.

---

## 8. Limitations & Concerns

### 8.1 Tool Integration Fragmentation

**Issue**: Three distinct tool invocation paths exist, each with different failure modes:

1. **Planner-directed upfront dispatch** (orchestrator.py::agentic_rag_node)
   - Executed before retrieval
   - Chaining logic (channel_finder → pv_read) is hardcoded
   - If tool fails, no fallback

2. **Reasoning agent tool binding** (reasoning_agent.py::synthesize)
   - Native: `response.tool_calls` extracted
   - Prompt-based: regex extraction from draft
   - Re-invoked in loop until no tool_calls

3. **ReactExecutor path** (react_executor.py)
   - Prebuilt LangGraph agent handles loop internally
   - LLM-driven termination
   - No operator visibility into intermediate steps

**Risk**: Operators can't easily predict which path will fire, making tool availability guarantees unclear.

**Recommendation**: 
- Consolidate to single entry point
- Explicit tool registry with capability descriptors
- Pre-flight validation before planning

### 8.2 Self-Critique Gate Issues

**Problem 1**: Rewrites removing TOOL_CALL lines
```python
# reasoning_agent.py line ~185
gpt-4o-mini.invoke(self_critique_prompt)
# If score < 0.7, returns rewrite_hint
# Re-synthesis may corrupt TOOL_CALL: lines in draft
```

**Problem 2**: Duplicate work in corrective loop
```
Synthesis (with gpt-4o) + Self-critique (gpt-4o-mini) 
    + Corrective eval (gpt-4o again)
    + Re-synthesis (gpt-4o again if needs_correction)
```
Three LLM calls for a single cycle.

**Problem 3**: False negatives in lenient calibration
```python
# self_critique_prompt says:
# "only fail if score < 0.4 on ALL axes"
# → Many broken answers pass through to corrective loop
# → Expensive correction rounds
```

**Recommendation**:
- Make self-critique gate STRICTER (currently too lenient)
- OR remove it and rely solely on corrective agent (consolidate)

### 8.3 Counting Intent Handling Redundancy

**Issue**: Dual forcing layers create confusion:
1. Planner prepends `data_analysis` step (if counting_intent=True)
2. Reasoning agent injects hard directive in SystemMessage
3. If both fire, may trigger tool twice

**Code example**:
```python
# planner.py line ~95
if state.get('counting_intent') and not any(
    s.get('tool') == 'data_analysis' for s in plan_steps
):
    plan_steps.insert(0, {..., 'tool': 'data_analysis'})

# reasoning_agent.py line ~120
if counting_section:
    system_text += counting_section  # "You MUST call data_analysis..."
```

**Risk**: Operator confusion about which mechanism is active; over-forcing (unnecessary tool calls).

**Recommendation**: Unify to single forcing mechanism (planner or synthesis, not both).

### 8.4 Corrective Agent Scoring Not Adaptive

**Issue**: Hard-coded weights don't adapt to query type:
```python
overall_score = 0.4 × relevance 
              + 0.3 × grounding 
              + 0.3 × completeness
```

**Examples**:
- Factual query ("What is 7 GeV in TeV?"): relevance might be 0.9, grounding 1.0 → good score
- Procedural query ("Steps to align 8ID optics"): relevance 0.8, grounding 0.9, completeness 0.6 → lower score

**Impact**: One-size-fits-all thresholds (0.4) may accept incomplete factuals or fail complete procedures.

**Recommendation**: Make weights intent-aware:
```python
if intent == 'factual':
    w_rel, w_gnd, w_cmp = 0.3, 0.5, 0.2
elif intent == 'procedural':
    w_rel, w_gnd, w_cmp = 0.4, 0.3, 0.3  # completeness critical
else:
    w_rel, w_gnd, w_cmp = 0.4, 0.3, 0.3  # default
```

### 8.5 KG Entity Validation Noise

**Issue**: Entity extraction + validation can be noisy:
```python
# corrective_agent.py line ~85
kg_issues = self._validate_kg_entities_llm(draft, state)
kg_penalty = len(kg_issues) * 0.05  # 0.05 per entity

# If extraction hallucinates 5 entities → 0.25 penalty
# overall_score can drop below threshold despite good answer
```

**Risk**: False KG validation failures trigger unnecessary re-retrieval.

**Recommendation**:
- Validate KG entities probabilistically (confidence scores)
- Cap total KG penalty (e.g., max 0.15)
- Log false positives for calibration

### 8.6 Incomplete Thinking-Mode Adoption

**Current state**:
- Prompts include `<think>` blocks (SYNTHESIS_PROMPT, etc.)
- Parsing logic present (but not always invoked)
- System-level support patchy

**Issue**: If LLM doesn't emit `<think>` blocks (e.g., text-completion models), prompts are malformed.

**Recommendation**:
- Make `<think>` optional (guard against missing blocks)
- OR detect LLM capability and strip `<think>` from prompts for non-thinking models

---

## 9. Evaluation Results Interpretation

### 9.1 What the Latency Harness Reveals

**Measured metrics**:
- Component duration (seconds)
- LLM model comparison (Opus46 vs GPT5.4 vs Gemini25Pro)
- Tail latencies (p95, p99)
- Correction round overhead

**Example insight**:
```
Component      Mean    Median  p95
─────────────────────────────────
router        0.18s   0.15s   0.28s
enhancer      0.12s   0.10s   0.20s
planner       2.5s    2.4s    3.2s   (slow)
retrieval     0.8s    0.75s   1.1s
synthesis     3.1s    2.9s    4.5s   (slowest, uses gpt-4o)
corrective    1.8s    1.7s    2.4s
```

**Interpretation**: Planner + synthesis dominate (5.6s / 9.2s total). Planner LLM swaps (40 runs) likely show GPT5.4 fastest, Gemini25Pro slowest.

### 9.2 Query Intent Impact

**Expected patterns**:
- Factual queries: Lower retrieval diversity (fewer plan steps), faster
- Procedural/troubleshooting: Higher retrieval cost, more corrections
- Counting queries: Tool dispatch overhead, but deterministic answer

---

## 10. Recommendations

### 10.1 Immediate (High Impact, Low Risk)

1. **Unify tool invocation**
   - Consolidate planner + synthesis + react_executor paths
   - Single entry point: `ToolRouter.dispatch(plan_steps, state)`
   - Clear error handling & fallback chain

2. **Tighten self-critique gate**
   - Increase thresholds (currently lenient)
   - Remove if causing more corrections than it prevents (A/B test)

3. **Fix TOOL_CALL line corruption**
   - Self-critique should preserve TOOL_CALL: directives in rewrites
   - OR suppress self-critique for plans with tool_steps

4. **Add intent-aware scoring**
   - Corrective agent should weight completeness higher for procedural queries
   - Implement in CorrectiveAgent.evaluate()

5. **Cap KG penalty**
   - Max KG deduction: 0.15 (not unbounded)
   - Log false positives for entity extraction calibration

### 10.2 Medium-term (Enhanced Robustness)

6. **Decouple planning from tool execution**
   - Planner generates abstract plan (no tool-specific details)
   - Separate ToolDispatcher maps plan steps → tool calls
   - Enables plan reuse across different tool registries

7. **Add query complexity scoring**
   - Estimate # hops, # tools needed before planning
   - Use to set correction_max_rounds, confidence_threshold adaptively

8. **Implement tool dry-run**
   - Pre-flight validate tool availability before planning
   - Fail fast if critical tool missing

9. **Expand thinking-mode support**
   - Detect LLM capability (GPT-o1 vs Opus vs standard)
   - Conditionally emit `<think>` blocks
   - Parse and extract thinking for UI visualization

10. **Upgrade reasoning_trace**
    - Include tool invocations, not just text summaries
    - Enable detailed operator debugging of agentic loops

### 10.3 Long-term (Architectural Evolution)

11. **ReAct → Tree-of-Thought (ToT)**
    - Current ReAct: linear reasoning → action → next step
    - Explore multi-branch planning (e.g., planner generates 3 candidate strategies, rank by expected retrieval cost)

12. **Tool use pre-training**
    - Fine-tune synthesis LLM on tool-use examples
    - Reduce tool call failures & regex extraction issues

13. **Operator-in-the-loop corrections**
    - When max_rounds reached, surface ambiguities to operator (e.g., "Did you mean X or Y?")
    - Learn from corrections to improve future routing

---

## 11. Checklist for Operators

- [ ] Verify `settings.agent.use_create_react_agent = True` if using tool-capable LLM
- [ ] Check `settings.agent.tool_chain_safety_cap` is set appropriately (default 25, implies 50-step max recursion)
- [ ] Monitor `reasoning_trace` in responses for tool skipped messages
- [ ] Test counting queries end-to-end to ensure `data_analysis` tool fires
- [ ] Profile component latencies to identify bottlenecks
- [ ] Review corrective loop trigger rate (expect < 10% of queries to need correction)
- [ ] Audit KG entity validation false positives (check logs for penalty-driven re-retrievals)
- [ ] Validate citation formatting in outputs (should be distributed `[source_N]`, not bunched)

---

## 12. Conclusion

The agentic-graph-rag ReAct system is **enterprise-grade and feature-complete**, with several novel contributions (KG-validated corrections, plan-guided retrieval, dual-path tool execution). 

**Main strengths**: Modularity, safety nets (max correction rounds), operator visibility (reasoning traces).

**Main concerns**: Tool integration fragmentation, self-critique redundancy, unvalidated scoring weights, incomplete thinking-mode adoption.

**Overall assessment**: **Production-ready with caveats**. Recommended follow-ups:
1. Unify tool invocation paths (single source of truth)
2. A/B test self-critique gate effectiveness
3. Make scoring intent-aware
4. Expand thinking-mode support for modern LLMs

**Timeline to full maturity**: 6–8 weeks of focused refinement.

---

**Document version**: 1.0  
**Evaluation date**: 2026-05-06  
**Evaluated by**: Architecture review  
**Next review**: 2026-05-20
