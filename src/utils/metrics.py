"""
In-memory metrics collection — thread-safe.
"""
import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies: Dict[str, List[float]] = defaultdict(list)
        self._retrieval_counts: List[Dict[str, int]] = []
        self._token_counts: List[Dict[str, Any]] = []

    def record_latency(self, operation: str, duration_ms: float) -> None:
        with self._lock:
            self._latencies[operation].append(duration_ms)

    def record_retrieval(
        self,
        n_vector: int,
        n_bm25: int,
        n_graph: int,
        n_fused: int,
    ) -> None:
        with self._lock:
            self._retrieval_counts.append(
                {
                    "n_vector": n_vector,
                    "n_bm25": n_bm25,
                    "n_graph": n_graph,
                    "n_fused": n_fused,
                }
            )

    def record_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> None:
        with self._lock:
            self._token_counts.append(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "model": model,
                }
            )

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            latency_summary: Dict[str, Any] = {}
            for op, values in self._latencies.items():
                if values:
                    latency_summary[op] = {
                        "count": len(values),
                        "mean_ms": sum(values) / len(values),
                        "min_ms": min(values),
                        "max_ms": max(values),
                    }

            total_input = sum(t["input_tokens"] for t in self._token_counts)
            total_output = sum(t["output_tokens"] for t in self._token_counts)

            return {
                "latencies": latency_summary,
                "retrieval_calls": len(self._retrieval_counts),
                "tokens": {
                    "total_input": total_input,
                    "total_output": total_output,
                    "total": total_input + total_output,
                    "calls": len(self._token_counts),
                },
            }
