"""
AgentState TypedDict — shared state across all LangGraph nodes and agents.
"""
from typing import TypedDict, Annotated, List, Optional, Any, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # --- Conversation ---
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    service_mode: str       # "naive_rag"|"agentic_rag"|"daily_summary"|"troubleshooting"|"studies_copilot"

    # --- Query processing ---
    original_query: str
    enhanced_query: str
    query_intent: str       # factual|temporal|troubleshooting|comparative|multi_hop|general|...
    counting_intent: bool   # True when query asks for counts/aggregates → force data_analysis tool
    selected_databases: List[str]   # e.g. ['entry_logbook'] or ['all_databases']
    temporal_context: Optional[Dict[str, Any]]
    query_variants: List[str]       # multi-query variants (enhanced_query + LLM variants)

    # --- Planning ---
    plan: str               # Multi-step plan from planner node
    plan_steps: List[Dict]  # Parsed structured plan steps from planner
    reasoning_trace: List[str]

    # --- Retrieval ---
    retrieved_docs: List[Any]       # LangChain Documents
    graph_context: List[Dict]       # KG traversal results
    web_results: List[Dict]         # Tavily/web results
    retrieval_sources: List[str]    # which retrievers fired

    # --- Generation ---
    draft_answer: str
    final_answer: str
    citations: List[Dict]

    # --- Quality control (Corrective RAG) ---
    confidence_score: float         # 0.0–1.0 from corrective agent
    needs_correction: bool
    correction_round: int           # max = 3
    correction_issues: List[str]    # issues from last corrective eval (passed to planner)
    quality_status: str             # "verified"|"unverified"|"max_rounds_exhausted"|"evaluator_error"

    # --- Routing ---
    retrieval_strategy: str         # VECTOR_ONLY|HYBRID_3WAY|KG_ONLY|CODE_EXEC|VECTOR_BM25|BM25_ONLY
    retrieval_strategy_override: Optional[str]  # eval: pin strategy, bypassing the router
    extracted_entities: List[str]   # named entities extracted by router

    # --- Tool execution ---
    tool_calls: List[Dict]
    tool_results: List[Dict]
    tool_skipped: List[Dict]        # planner referenced a tool not loaded at startup
    audit_rounds: int               # AnswerAuditor retries — capped at 1 per query

    # --- Memory context ---
    memory_context: Dict[str, Any]  # short_term + long_term + episodic

    # --- PV (Process Variable) context ---
    pv_addresses: List[str]         # resolved PV addresses from channel_finder
    pv_readings: List[Dict]         # real-time PV values from pv_read
    pv_history: List[Dict]          # historical PV data summaries from archiver_read

    # --- Studies Copilot attachments ---
    entry_attachments: List[str]     # fetched the logbook page content (formatted blocks)

    # --- Clarification gate (scope-ambiguity handler for counting queries) ---
    clarification_pending: Optional[Dict[str, Any]]   # {"question", "options": [{"id","label","interpretation"}]}
    clarification_choice: Optional[str]               # User's chosen option id on resume

    # --- Control flow ---
    next_node: str
    error: Optional[str]
    iteration_count: int            # max = 10
