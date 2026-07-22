"""
Unit tests for MetricsCollector.
Covers: latency recording, retrieval counts, token counts,
        summary aggregation, thread safety.
No live services required.
"""
import sys
import threading
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class TestMetricsCollectorLatency:
    @pytest.fixture
    def mc(self):
        from src.utils.metrics import MetricsCollector
        return MetricsCollector()

    def test_record_single_latency(self, mc):
        mc.record_latency("embedding", 42.5)
        summary = mc.get_summary()
        assert summary["latencies"]["embedding"]["count"] == 1
        assert summary["latencies"]["embedding"]["mean_ms"] == pytest.approx(42.5)

    def test_record_multiple_latencies(self, mc):
        mc.record_latency("embedding", 10.0)
        mc.record_latency("embedding", 20.0)
        mc.record_latency("embedding", 30.0)
        summary = mc.get_summary()
        assert summary["latencies"]["embedding"]["count"] == 3
        assert summary["latencies"]["embedding"]["mean_ms"] == pytest.approx(20.0)
        assert summary["latencies"]["embedding"]["min_ms"] == pytest.approx(10.0)
        assert summary["latencies"]["embedding"]["max_ms"] == pytest.approx(30.0)

    def test_different_operations_separate(self, mc):
        mc.record_latency("embedding", 10.0)
        mc.record_latency("llm_call", 500.0)
        summary = mc.get_summary()
        assert "embedding" in summary["latencies"]
        assert "llm_call" in summary["latencies"]
        assert summary["latencies"]["embedding"]["mean_ms"] != summary["latencies"]["llm_call"]["mean_ms"]

    def test_empty_latencies(self, mc):
        summary = mc.get_summary()
        assert summary["latencies"] == {}


class TestMetricsCollectorRetrieval:
    @pytest.fixture
    def mc(self):
        from src.utils.metrics import MetricsCollector
        return MetricsCollector()

    def test_record_retrieval(self, mc):
        mc.record_retrieval(n_vector=5, n_bm25=3, n_graph=2, n_fused=8)
        summary = mc.get_summary()
        assert summary["retrieval_calls"] == 1

    def test_multiple_retrieval_calls(self, mc):
        mc.record_retrieval(5, 3, 2, 8)
        mc.record_retrieval(10, 5, 0, 12)
        summary = mc.get_summary()
        assert summary["retrieval_calls"] == 2

    def test_no_retrieval_calls(self, mc):
        summary = mc.get_summary()
        assert summary["retrieval_calls"] == 0


class TestMetricsCollectorTokens:
    @pytest.fixture
    def mc(self):
        from src.utils.metrics import MetricsCollector
        return MetricsCollector()

    def test_record_tokens(self, mc):
        mc.record_tokens(input_tokens=100, output_tokens=50, model="gpt-4o")
        summary = mc.get_summary()
        assert summary["tokens"]["total_input"] == 100
        assert summary["tokens"]["total_output"] == 50
        assert summary["tokens"]["total"] == 150
        assert summary["tokens"]["calls"] == 1

    def test_accumulate_tokens_across_calls(self, mc):
        mc.record_tokens(100, 50, "gpt-4o")
        mc.record_tokens(200, 80, "gemini-1.5-flash")
        summary = mc.get_summary()
        assert summary["tokens"]["total_input"] == 300
        assert summary["tokens"]["total_output"] == 130
        assert summary["tokens"]["total"] == 430
        assert summary["tokens"]["calls"] == 2

    def test_no_token_calls(self, mc):
        summary = mc.get_summary()
        assert summary["tokens"]["total"] == 0
        assert summary["tokens"]["calls"] == 0


class TestMetricsCollectorSummary:
    def test_full_summary_structure(self):
        from src.utils.metrics import MetricsCollector
        mc = MetricsCollector()
        mc.record_latency("embedding", 25.0)
        mc.record_retrieval(5, 3, 2, 8)
        mc.record_tokens(100, 50, "gpt-4o")
        summary = mc.get_summary()
        assert "latencies" in summary
        assert "retrieval_calls" in summary
        assert "tokens" in summary
        assert isinstance(summary["tokens"]["total"], int)


class TestMetricsCollectorThreadSafety:
    def test_concurrent_writes(self):
        from src.utils.metrics import MetricsCollector
        mc = MetricsCollector()
        n_threads = 10
        n_ops = 100

        def writer(thread_id):
            for i in range(n_ops):
                mc.record_latency(f"op_{thread_id}", float(i))
                mc.record_tokens(10, 5, "model")

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = mc.get_summary()
        assert summary["tokens"]["calls"] == n_threads * n_ops
        assert summary["tokens"]["total_input"] == n_threads * n_ops * 10
        # Each thread creates its own operation key
        assert len(summary["latencies"]) == n_threads
