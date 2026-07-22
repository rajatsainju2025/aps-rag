"""
Unit tests for the tools module.
Covers: CodeExecutorTool (sandbox safety, timeout, success),
        CalculatorTool (numeric, symbolic, error handling),
        and memory_tool read/write via MemoryManager mock.
No live LLM, Neo4j, or external services required.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# CodeExecutorTool
# ---------------------------------------------------------------------------

class TestCodeExecutorToolSafety:
    @pytest.fixture
    def tool(self):
        from src.tools.code_executor import CodeExecutorTool
        return CodeExecutorTool()

    @pytest.mark.parametrize("banned", [
        "os.system('rm -rf /')",
        "__import__('os').getcwd()",
        "subprocess.run(['ls'])",
        "subprocess.call(['id'])",
        "subprocess.Popen(['bash'])",
        "eval('1+1')",
        "exec('pass')",
        "open('/etc/passwd')",
        "shutil.rmtree('/tmp')",
    ])
    def test_banned_patterns_rejected(self, tool, banned):
        result = json.loads(tool._run(banned))
        assert result["returncode"] == 1
        assert "Forbidden" in result["stderr"]

    def test_safe_arithmetic_executes(self, tool):
        result = json.loads(tool._run("print(2 + 2)"))
        assert result["returncode"] == 0
        assert "4" in result["stdout"]

    def test_safe_list_ops_execute(self, tool):
        code = "nums = [1,2,3]; print(sum(nums))"
        result = json.loads(tool._run(code))
        assert result["returncode"] == 0
        assert "6" in result["stdout"]

    def test_syntax_error_captured(self, tool):
        result = json.loads(tool._run("def broken(:"))
        assert result["returncode"] != 0

    def test_runtime_error_captured(self, tool):
        result = json.loads(tool._run("raise ValueError('test error')"))
        assert result["returncode"] != 0
        assert "ValueError" in result["stderr"]

    def test_stdout_truncated_at_4000_chars(self, tool):
        # Print something large; stdout should be capped
        code = "print('x' * 10000)"
        result = json.loads(tool._run(code))
        assert len(result["stdout"]) <= 4000

    def test_check_safety_raises_on_banned(self, tool):
        with pytest.raises(ValueError, match="Forbidden"):
            tool._check_safety("os.system('ls')")

    def test_check_safety_passes_on_clean_code(self, tool):
        # Should not raise
        tool._check_safety("result = 2 ** 10\nprint(result)")

    def test_result_always_has_stdout_stderr_returncode(self, tool):
        result = json.loads(tool._run("print('hello')"))
        assert "stdout" in result
        assert "stderr" in result
        assert "returncode" in result


class TestCodeExecutorTimeout:
    def test_timeout_returns_negative_returncode(self):
        from src.tools.code_executor import CodeExecutorTool
        tool = CodeExecutorTool(timeout_seconds=1)
        # Infinite loop — should time out
        result = json.loads(tool._run("while True: pass"))
        assert result["returncode"] == -1
        assert "timed out" in result["stderr"].lower()


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------

class TestCalculatorTool:
    @pytest.fixture
    def tool(self):
        from src.tools.calculator import CalculatorTool
        return CalculatorTool()

    def test_simple_addition(self, tool):
        result = tool._run("2 + 3")
        assert "5" in result

    def test_simple_multiplication(self, tool):
        result = tool._run("7 * 8")
        assert "56" in result

    def test_integer_division(self, tool):
        result = tool._run("10 / 2")
        assert "5" in result

    def test_power_operation(self, tool):
        result = tool._run("2**10")
        assert "1024" in result

    def test_invalid_expression_returns_error_string(self, tool):
        # Malformed expression (syntax error that both sympy and eval reject)
        result = tool._run("2 ++ +++ 5 !!!")
        assert "Error" in result or "error" in result.lower()

    def test_division_by_zero_handled(self, tool):
        result = tool._run("1 / 0")
        # Should return an error string, not raise
        assert isinstance(result, str)

    def test_float_result(self, tool):
        result = tool._run("1.5 + 2.5")
        assert "4" in result

    def test_abs_function(self, tool):
        result = tool._run("abs(-42)")
        assert "42" in result

    def test_round_function(self, tool):
        result = tool._run("round(3.14159, 2)")
        # Should produce something near 3.14
        assert "3.14" in result


# ---------------------------------------------------------------------------
# Memory tool
# ---------------------------------------------------------------------------

class TestMemoryTool:
    @pytest.fixture
    def memory_manager(self):
        mgr = MagicMock()
        mgr.get_full_context.return_value = {
            "short_term": [{"role": "user", "content": "prior question"}],
            "long_term": {"topics": ["beam loss", "RF cavity"]},
            "episodic": None,
        }
        mgr.store.return_value = None
        mgr.summarize.return_value = "Session involved beam loss discussion."
        return mgr

    def test_read_returns_context_string(self, memory_manager):
        from src.tools.memory_tool import MemoryTool
        tool = MemoryTool(memory_manager=memory_manager, session_id="sess_1")
        result = tool._run(action="read")
        assert isinstance(result, str)
        memory_manager.get_full_context.assert_called_once_with("sess_1")

    def test_read_with_key_returns_specific_value(self, memory_manager):
        from src.tools.memory_tool import MemoryTool
        tool = MemoryTool(memory_manager=memory_manager, session_id="sess_1")
        result = tool._run(action="read", key="short_term")
        assert "prior question" in result

    def test_read_missing_key_returns_not_found(self, memory_manager):
        from src.tools.memory_tool import MemoryTool
        tool = MemoryTool(memory_manager=memory_manager, session_id="sess_1")
        result = tool._run(action="read", key="nonexistent_key")
        assert "not found" in result

    def test_write_calls_store(self, memory_manager):
        from src.tools.memory_tool import MemoryTool
        tool = MemoryTool(memory_manager=memory_manager, session_id="sess_1")
        result = tool._run(action="write", key="fact_001", value="beam loss at sector 35")
        memory_manager.store.assert_called_once_with("sess_1", "fact_001", "beam loss at sector 35")
        assert "Stored" in result

    def test_write_without_key_returns_error(self, memory_manager):
        from src.tools.memory_tool import MemoryTool
        tool = MemoryTool(memory_manager=memory_manager, session_id="sess_1")
        result = tool._run(action="write", key="", value="some value")
        assert "Error" in result

    def test_summarize_calls_summarize(self, memory_manager):
        from src.tools.memory_tool import MemoryTool
        tool = MemoryTool(memory_manager=memory_manager, session_id="sess_1")
        result = tool._run(action="summarize")
        memory_manager.summarize.assert_called_once_with("sess_1")
        assert "beam loss" in result

    def test_unknown_action_returns_error(self, memory_manager):
        from src.tools.memory_tool import MemoryTool
        tool = MemoryTool(memory_manager=memory_manager, session_id="sess_1")
        result = tool._run(action="delete_all")
        assert "Unknown action" in result

    def test_no_memory_manager_returns_message(self):
        from src.tools.memory_tool import MemoryTool
        tool = MemoryTool(memory_manager=None, session_id="sess_1")
        result = tool._run(action="read")
        assert "not configured" in result

    def test_memory_manager_exception_handled(self, memory_manager):
        from src.tools.memory_tool import MemoryTool
        memory_manager.get_full_context.side_effect = RuntimeError("DB unavailable")
        tool = MemoryTool(memory_manager=memory_manager, session_id="sess_1")
        result = tool._run(action="read")
        assert "error" in result.lower()
