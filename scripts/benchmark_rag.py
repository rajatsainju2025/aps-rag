"""
RAG Benchmarking Script
=======================
Tests 10 queries against both Naive-RAG and Agentic-RAG and reports:
  - Query latency (seconds)
  - Number of citations returned
  - First 120 chars of the answer
  - A summary table

Usage:
    conda activate agentic_aps_retrieval
    python scripts/benchmark_rag.py [--api-url http://localhost:8003]
"""
import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from requests.exceptions import RequestException

# ──────────────────────────────────────────────────────────────────────────────
# 10 benchmark queries — mix of temporal, synonym-heavy, factual, multi-hop
# ──────────────────────────────────────────────────────────────────────────────
QUERIES = [
    # 1 — temporal (last week) + synonym (BPM → orbit monitor)
    "What BPM or orbit feedback issues occurred last week?",
    # 2 — temporal (recent) + synonym (RF)
    "Summarise recent RF cavity faults or klystron trips.",
    # 3 — factual look-up
    "What is the maximum number of Linac pulses per second?",
    # 4 — troubleshooting / causal
    "Why does the beam abort when the machine protection system trips?",
    # 5 — temporal (this month) + synonym (insertion device / undulator)
    "List any insertion device or undulator problems reported this month.",
    # 6 — multi-hop: equipment → procedure
    "How is the chromaticity measured and corrected at the facility?",
    # 7 — factual + acronym resolution
    "What does SDDS stand for and what tools are available for data analysis?",
    # 8 — operational / comparative
    "Compare the kicker thyratron failure modes from the work requests.",
    # 9 — temporal (this year) + synonym (radiation monitor / radmon)
    "Have there been any radiation monitor or radmon alarms this year?",
    # 10 — system-level multi-hop
    "What are the main causes of orbit drift at the facility storage ring?",
]

MODELS = ["naive-rag", "agentic-rag"]
TIMEOUT_SEC = 600  # Agentic-RAG can take up to ~10 min (ReAct loop + multi-LLM)

# ANSI colours
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Result:
    query_idx: int
    model: str
    latency_s: Optional[float] = None
    citations: int = 0
    answer_snippet: str = ""
    error: Optional[str] = None
    server_latency_ms: Optional[float] = None


def query_model(api_url: str, model: str, query: str) -> Result:
    """POST to /v1/chat/completions (non-streaming) and return a Result."""
    result = Result(query_idx=0, model=model)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{api_url}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT_SEC,
        )
        result.latency_s = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()

        # Extract answer text
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {}).get("content", "")
        result.answer_snippet = msg[:120].replace("\n", " ").strip()

        # Count citation markers [XXX_NNN] in the answer
        import re
        result.citations = len(re.findall(r"\[[A-Z]+_[^\]]+\]", msg))

        # Server-side latency if reported
        usage = data.get("usage", {})
        result.server_latency_ms = data.get("latency_ms")

    except RequestException as exc:
        result.latency_s = time.perf_counter() - t0
        result.error = str(exc)[:80]
    except Exception as exc:
        result.latency_s = time.perf_counter() - t0
        result.error = f"Parse error: {exc!s}"[:80]

    return result


def run_benchmark(api_url: str) -> List[Result]:
    results: List[Result] = []

    total_runs = len(QUERIES) * len(MODELS)
    run_no = 0

    for q_idx, query in enumerate(QUERIES, start=1):
        print(f"\n{BOLD}{'─'*72}{RESET}")
        print(f"{CYAN}Query {q_idx:2d}/{len(QUERIES)}{RESET}: {query}")
        print(f"{'─'*72}")

        for model in MODELS:
            run_no += 1
            print(
                f"  {YELLOW}[{run_no}/{total_runs}] {model:<14}{RESET} … ",
                end="",
                flush=True,
            )
            r = query_model(api_url, model, query)
            r.query_idx = q_idx
            results.append(r)

            if r.error:
                print(f"{BOLD}\033[91mERROR{RESET} ({r.latency_s:.1f}s) — {r.error}")
            else:
                print(
                    f"{GREEN}{r.latency_s:.2f}s{RESET}  "
                    f"citations={r.citations}  "
                    f"snippet: {r.answer_snippet[:60]!r}…"
                )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Table rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _trunc(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def print_summary_table(results: List[Result]) -> None:
    """Print a compact side-by-side comparison table."""

    naive  = {r.query_idx: r for r in results if r.model == "naive-rag"}
    agentic = {r.query_idx: r for r in results if r.model == "agentic-rag"}

    col_q   = 36  # query column width
    col_t   = 9   # time column
    col_c   = 5   # citation count column

    # Header
    sep = (
        f"{'─'*4}─┬─{'─'*col_q}─┬─"
        f"{'─'*col_t}─{'─'*col_c}─{'─'*col_t}─┬─"
        f"{'─'*col_t}─{'─'*col_c}─{'─'*col_t}─┐"
    )
    header_line1 = (
        f"{'':4} │ {'Query':<{col_q}} │ "
        f"{'── Naive-RAG ──':^{col_t + col_c + col_t + 2}} │ "
        f"{'── Agentic-RAG ──':^{col_t + col_c + col_t + 2}} │"
    )
    header_line2 = (
        f"{'  #':4} │ {'':^{col_q}} │ "
        f"{'Time(s)':>{col_t}} {'Cit':>{col_c}} {'Status':>{col_t}} │ "
        f"{'Time(s)':>{col_t}} {'Cit':>{col_c}} {'Status':>{col_t}} │"
    )
    divider = (
        f"{'─'*4}─┼─{'─'*col_q}─┼─"
        f"{'─'*col_t}─{'─'*col_c}─{'─'*col_t}─┼─"
        f"{'─'*col_t}─{'─'*col_c}─{'─'*col_t}─┤"
    )
    closer = (
        f"{'─'*4}─┴─{'─'*col_q}─┴─"
        f"{'─'*col_t}─{'─'*col_c}─{'─'*col_t}─┴─"
        f"{'─'*col_t}─{'─'*col_c}─{'─'*col_t}─┘"
    )

    print(f"\n{BOLD}{'═'*92}")
    print(f"  BENCHMARK RESULTS  —  Naive-RAG  vs  Agentic-RAG  ({len(QUERIES)} queries)")
    print(f"{'═'*92}{RESET}")
    print(sep)
    print(header_line1)
    print(header_line2)

    naive_times   = []
    agentic_times = []

    for q_idx in range(1, len(QUERIES) + 1):
        print(divider)
        q_text = _trunc(QUERIES[q_idx - 1], col_q)
        nr = naive.get(q_idx)
        ar = agentic.get(q_idx)

        def fmt(r: Optional[Result]) -> str:
            if r is None:
                return f"{'—':>{col_t}} {'—':>{col_c}} {'—':>{col_t}}"
            if r.error:
                return f"{'ERR':>{col_t}} {'—':>{col_c}} {'timeout':>{col_t}}"
            status = "✓" if r.citations > 0 else "no-cit"
            return f"{r.latency_s:>{col_t}.2f} {r.citations:>{col_c}} {status:>{col_t}}"

        print(
            f"{CYAN}{q_idx:>4}{RESET} │ {q_text:<{col_q}} │ "
            f"{fmt(nr)} │ {fmt(ar)} │"
        )

        if nr and not nr.error and nr.latency_s:
            naive_times.append(nr.latency_s)
        if ar and not ar.error and ar.latency_s:
            agentic_times.append(ar.latency_s)

    print(closer)

    # Statistics footer
    def stats_row(label: str, times: List[float]) -> str:
        if not times:
            return f"  {label:14}: no data"
        return (
            f"  {label:14}: "
            f"min={min(times):.2f}s  "
            f"max={max(times):.2f}s  "
            f"mean={statistics.mean(times):.2f}s  "
            f"median={statistics.median(times):.2f}s  "
            f"total={sum(times):.1f}s"
        )

    print(f"\n{BOLD}Statistics{RESET}")
    print(stats_row("naive-rag",    naive_times))
    print(stats_row("agentic-rag",  agentic_times))
    if naive_times and agentic_times:
        ratio = statistics.mean(agentic_times) / statistics.mean(naive_times)
        print(f"\n  Agentic-RAG is {ratio:.1f}× slower than Naive-RAG on average.")


def save_json(results: List[Result], path: str) -> None:
    data = [
        {
            "query_idx": r.query_idx,
            "query": QUERIES[r.query_idx - 1],
            "model": r.model,
            "latency_s": round(r.latency_s, 3) if r.latency_s else None,
            "citations": r.citations,
            "answer_snippet": r.answer_snippet,
            "error": r.error,
        }
        for r in results
    ]
    with open(path, "w") as fh:
        json.dump({"queries": QUERIES, "results": data}, fh, indent=2)
    print(f"\n  Results saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Naive-RAG vs Agentic-RAG")
    parser.add_argument("--api-url", default="http://localhost:8003",
                        help="Base URL of the FastAPI server (default: http://localhost:8003)")
    parser.add_argument("--output", default="data/eval/benchmark_results.json",
                        help="Path to save JSON results")
    parser.add_argument("--naive-only",  action="store_true", help="Only run Naive-RAG")
    parser.add_argument("--agentic-only", action="store_true", help="Only run Agentic-RAG")
    args = parser.parse_args()

    global MODELS
    if args.naive_only:
        MODELS = ["naive-rag"]
    elif args.agentic_only:
        MODELS = ["agentic-rag"]

    # Connectivity check
    try:
        r = requests.get(f"{args.api_url}/health", timeout=5)
        print(f"✅  API reachable at {args.api_url}  (status={r.status_code})")
    except RequestException:
        print(f"❌  Cannot reach API at {args.api_url}. Is the server running?")
        sys.exit(1)

    print(f"\n{BOLD}Running {len(QUERIES)} queries × {len(MODELS)} model(s) "
          f"= {len(QUERIES)*len(MODELS)} total requests{RESET}")
    print(f"Timeout per request: {TIMEOUT_SEC}s\n")

    t_start = time.perf_counter()
    results = run_benchmark(args.api_url)
    t_total = time.perf_counter() - t_start

    print_summary_table(results)
    print(f"\n  Wall-clock total: {t_total:.1f}s")

    save_json(results, args.output)


if __name__ == "__main__":
    main()
