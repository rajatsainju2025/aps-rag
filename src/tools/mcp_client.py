"""
Shared MCP client manager for connecting to MCP servers from LangChain tools.

Spawns MCP server subprocesses via stdio transport and provides both
async and sync interfaces for calling tools.

Stage C additions:
  - Automatic retry with exponential backoff on transient transport errors
    (broken pipe, subprocess crash, JSON decode failure). Tool-LEVEL errors
    (a server returning ``{"status": "error"}``) are NOT retried — those are
    deliberate signals to the LLM, which retries them via the ReAct loop.
  - ``${VAR:-default}`` env-var expansion in .mcp.json so secrets can stay
    in shell env and out of the config file.

Stage D additions (this module):
  - **Persistent sessions** — one ``ClientSession`` per server name, reused
    across calls. Replaces the per-call subprocess spawn (~500ms-1s) so heavy
    servers (image_search index load, math_reasoning LLM client) keep their
    state across calls. Implemented via a single background asyncio loop in a
    daemon thread; calls are dispatched with ``run_coroutine_threadsafe``.
  - **Crash recovery** — when the next call to a dead session fails with a
    transport error (BrokenPipe, ClosedResource, EOF), the pool tears down
    the dead entry; the outer retry layer reopens on the next attempt.
  - **atexit cleanup** — registered hook drains pending shutdown messages
    so MCP subprocesses exit cleanly when the agent process terminates.
"""
import asyncio
import atexit
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry policy: tunable via env vars so callers can crank in production
# ---------------------------------------------------------------------------
_MCP_RETRY_ATTEMPTS = max(1, int(os.environ.get("MCP_RETRY_ATTEMPTS", "2")))
_MCP_RETRY_BASE_DELAY_S = float(os.environ.get("MCP_RETRY_BASE_DELAY_S", "0.5"))
_MCP_CALL_TIMEOUT_S = float(os.environ.get("MCP_CALL_TIMEOUT_S", "60"))
_MCP_SESSION_INIT_TIMEOUT_S = float(os.environ.get("MCP_SESSION_INIT_TIMEOUT_S", "30"))

# Exception types worth retrying — transient transport / subprocess failures.
# Tool-level errors come back as a normal string response (status=error inside
# the JSON envelope) and are intentionally NOT retried here.
_RETRIABLE_EXC_NAMES = (
    "BrokenPipeError",
    "ConnectionResetError",
    "ConnectionRefusedError",
    "JSONDecodeError",
    "TimeoutError",
    "TimeoutException",
    "EOFError",
    "ClosedResourceError",
    "BrokenResourceError",
    "EndOfStream",
)


def _is_retriable_exception(exc: BaseException) -> bool:
    return type(exc).__name__ in _RETRIABLE_EXC_NAMES


_ENV_VAR_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)(?::-([^}]*))?\}$")


def _expand_env_value(value: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` patterns from os.environ.

    Returns the original string if no pattern matches. Empty default when the
    var is unset and no default is provided in the pattern.
    """
    if not isinstance(value, str):
        return value
    m = _ENV_VAR_PATTERN.match(value)
    if not m:
        return value
    var_name, default = m.group(1), m.group(2) or ""
    return os.environ.get(var_name, default)


def _extract_text_blocks(content_blocks: Any) -> str:
    parts = []
    for block in content_blocks:
        if hasattr(block, "text"):
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "\n".join(parts) if parts else "{}"


# ---------------------------------------------------------------------------
# Persistent session pool (Stage D)
# ---------------------------------------------------------------------------
class _SessionTask:
    """A long-lived asyncio task that owns one MCP ClientSession.

    Why a task (not just a session reference): the MCP SDK's stdio_client
    and ClientSession use anyio task groups internally, which require their
    enter/exit to happen inside one task scope. So the ENTIRE session
    lifecycle (open → loop reading commands → close) lives inside ``_run()``.

    Calls are submitted via the asyncio.Queue; results come back via the
    per-call future. A ``None`` sentinel on the queue triggers a clean
    shutdown.
    """

    def __init__(self, server_name: str, params: Any, loop: asyncio.AbstractEventLoop):
        self.server_name = server_name
        self.params = params
        self.loop = loop
        # Queue + ready event MUST be created on the loop's thread, not here.
        # Set up lazily inside _run().
        self._queue: Optional[asyncio.Queue] = None
        self._ready: Optional[asyncio.Event] = None
        self._init_error: Optional[BaseException] = None
        self.task: Optional[asyncio.Task] = None
        self.connect_count = 0  # incremented each successful initialize()

    def start(self) -> None:
        """Schedule _run() on the loop. Must be called from the loop's thread."""
        self._queue = asyncio.Queue()
        self._ready = asyncio.Event()
        self.task = asyncio.create_task(self._run(), name=f"mcp-session-{self.server_name}")

    async def _run(self) -> None:
        from mcp.client.stdio import stdio_client
        from mcp.client.session import ClientSession
        try:
            async with stdio_client(self.params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self.connect_count += 1
                    assert self._ready is not None
                    self._ready.set()
                    await self._serve(session)
        except BaseException as exc:  # noqa: BLE001 — we want to capture init failures too
            self._init_error = exc
            if self._ready is not None:
                self._ready.set()
            # Re-raise CancelledError so the loop's executor sees cancellation.
            if isinstance(exc, asyncio.CancelledError):
                raise

    async def _serve(self, session: Any) -> None:
        assert self._queue is not None
        while True:
            cmd = await self._queue.get()
            if cmd is None:  # shutdown sentinel
                return
            tool_name, args, fut = cmd
            try:
                result = await session.call_tool(tool_name, args)
                if not fut.done():
                    fut.set_result(_extract_text_blocks(result.content))
            except BaseException as exc:  # noqa: BLE001
                if not fut.done():
                    fut.set_exception(exc)
                # Transport-level failures: stop serving so the pool tears down
                # this dead task and reopens on the next call.
                if _is_retriable_exception(exc):
                    return

    async def call(self, tool_name: str, args: Dict[str, Any], timeout: float) -> str:
        assert self._ready is not None and self._queue is not None
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=_MCP_SESSION_INIT_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"MCP session for '{self.server_name}' did not initialize in "
                f"{_MCP_SESSION_INIT_TIMEOUT_S:.0f}s"
            ) from exc
        if self._init_error is not None:
            raise self._init_error
        fut: asyncio.Future = self.loop.create_future()
        await self._queue.put((tool_name, args, fut))
        return await asyncio.wait_for(fut, timeout=timeout)

    @property
    def alive(self) -> bool:
        return (
            self.task is not None
            and not self.task.done()
            and self._init_error is None
        )

    async def shutdown(self) -> None:
        if self._queue is not None:
            try:
                await self._queue.put(None)
            except Exception:  # noqa: BLE001
                pass
        if self.task is not None:
            try:
                await asyncio.wait_for(self.task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.task.cancel()
                try:
                    await self.task
                except (asyncio.CancelledError, BaseException):  # noqa: BLE001
                    pass


class _SessionPool:
    """Process-singleton pool of long-lived MCP sessions, one per server name.

    Backed by a single background asyncio loop in a daemon thread. Calls from
    sync code dispatch via ``run_coroutine_threadsafe``.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._tasks: Dict[str, _SessionTask] = {}
        self._open_lock = threading.Lock()
        self._async_locks: Dict[str, asyncio.Lock] = {}
        self._shutdown_registered = False

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None and not self._loop.is_closed():
            return self._loop
        with self._open_lock:
            if self._loop is not None and not self._loop.is_closed():
                return self._loop
            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def _run() -> None:
                asyncio.set_event_loop(loop)
                ready.set()
                try:
                    loop.run_forever()
                finally:
                    try:
                        loop.close()
                    except Exception:  # noqa: BLE001
                        pass

            t = threading.Thread(target=_run, daemon=True, name="mcp-session-pool")
            t.start()
            ready.wait(timeout=5)
            self._loop = loop
            self._thread = t
            if not self._shutdown_registered:
                atexit.register(self._shutdown)
                self._shutdown_registered = True
            return loop

    def call(
        self,
        server_name: str,
        params: Any,
        tool_name: str,
        args: Dict[str, Any],
        timeout: float,
    ) -> str:
        loop = self._ensure_loop()
        coro = self._call_async(server_name, params, tool_name, args, timeout)
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        # Add a small grace period for the inner asyncio.wait_for to fire first.
        return fut.result(timeout=timeout + 5)

    async def _call_async(
        self,
        server_name: str,
        params: Any,
        tool_name: str,
        args: Dict[str, Any],
        timeout: float,
    ) -> str:
        try:
            task = await self._get_task(server_name, params)
            return await task.call(tool_name, args, timeout)
        except BaseException as exc:  # noqa: BLE001
            # On transport-level failures, tear down the dead session so the
            # NEXT attempt (driven by the outer retry layer) gets a fresh one.
            if _is_retriable_exception(exc):
                await self._teardown(server_name)
            raise

    async def _get_task(self, server_name: str, params: Any) -> _SessionTask:
        existing = self._tasks.get(server_name)
        if existing is not None and existing.alive:
            return existing
        # Per-server lock so concurrent first-callers don't open two sessions.
        if server_name not in self._async_locks:
            self._async_locks[server_name] = asyncio.Lock()
        async with self._async_locks[server_name]:
            existing = self._tasks.get(server_name)
            if existing is not None and existing.alive:
                return existing
            # Discard a dead entry before opening a fresh one.
            if existing is not None:
                await self._teardown(server_name)
            task = _SessionTask(server_name, params, asyncio.get_running_loop())
            task.start()
            self._tasks[server_name] = task
            # Wait for initialize() to complete (or fail) before returning.
            try:
                await asyncio.wait_for(task._ready.wait(), timeout=_MCP_SESSION_INIT_TIMEOUT_S)  # type: ignore[union-attr]
            except asyncio.TimeoutError as exc:
                await self._teardown(server_name)
                raise TimeoutError(
                    f"MCP session for '{server_name}' did not initialize in "
                    f"{_MCP_SESSION_INIT_TIMEOUT_S:.0f}s"
                ) from exc
            if task._init_error is not None:
                err = task._init_error
                await self._teardown(server_name)
                raise err
            return task

    async def _teardown(self, server_name: str) -> None:
        task = self._tasks.pop(server_name, None)
        if task is None:
            return
        try:
            await task.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.debug("MCP teardown of '%s' raised: %s", server_name, exc)

    def shutdown_now(self) -> None:
        """Synchronous shutdown for tests / explicit use."""
        self._shutdown()

    def _shutdown(self) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        async def _close_all() -> None:
            for name in list(self._tasks.keys()):
                await self._teardown(name)

        try:
            fut = asyncio.run_coroutine_threadsafe(_close_all(), loop)
            fut.result(timeout=10)
        except Exception as exc:  # noqa: BLE001
            logger.debug("MCP pool shutdown drained with error: %s", exc)
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:  # noqa: BLE001
            pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None
        self._tasks.clear()
        self._async_locks.clear()

    # --- Inspection helpers used by tests ---
    def connect_count(self, server_name: str) -> int:
        task = self._tasks.get(server_name)
        return task.connect_count if task is not None else 0

    def session_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.alive)


_SESSION_POOL = _SessionPool()


def get_session_pool() -> _SessionPool:
    """Return the process-singleton session pool. Exposed for test inspection."""
    return _SESSION_POOL


# ---------------------------------------------------------------------------
# MCPClientManager — the public façade
# ---------------------------------------------------------------------------
class MCPClientManager:
    """Manages connections to MCP servers defined in .mcp.json.

    Usage:
        mgr = MCPClientManager("/path/to/.mcp.json")
        result = mgr.call_tool_sync("controls", "channel_find", {"query": "beam current"})
    """

    def __init__(self, mcp_config_path: str):
        self._config_path = mcp_config_path
        self._config: Optional[Dict] = None
        self._available = True
        try:
            self._load_config()
        except Exception as e:
            logger.warning("MCP config load failed: %s — MCP tools unavailable", e)
            self._available = False

    def _load_config(self):
        path = Path(self._config_path)
        if not path.exists():
            raise FileNotFoundError(f"MCP config not found: {self._config_path}")
        with open(path) as f:
            self._config = json.load(f)
        servers = self._config.get("mcpServers", {})
        logger.info("MCP config loaded: %d server(s) — %s", len(servers), list(servers.keys()))

    @property
    def is_available(self) -> bool:
        return self._available and self._config is not None

    def _get_server_config(self, server_name: str) -> Dict:
        if not self._config:
            raise RuntimeError("MCP config not loaded")
        servers = self._config.get("mcpServers", {})
        if server_name not in servers:
            raise KeyError(f"MCP server '{server_name}' not found in config")
        return servers[server_name]

    def _build_stdio_params(self, server_name: str) -> Any:
        from mcp.client.stdio import StdioServerParameters
        srv_config = self._get_server_config(server_name)
        env = os.environ.copy()
        for k, v in srv_config.get("env", {}).items():
            env[k] = _expand_env_value(v)
        command = srv_config["command"]
        # Bare "python" in .mcp.json means "the interpreter this app runs under",
        # not "whatever PATH resolves to". Without this substitution, MCP servers
        # launched from a cron-spawned uvicorn pick up /usr/bin/python (system 3.9,
        # no `mcp` package installed) and crash with ModuleNotFoundError. An
        # explicit path like "/opt/conda/bin/python" is passed through unchanged.
        if command == "python":
            command = sys.executable
        return StdioServerParameters(
            command=command,
            args=srv_config.get("args", []),
            env=env,
            cwd=str(Path(self._config_path).parent),
        )

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on an MCP server via stdio transport.

        One-shot call — opens a fresh session, invokes the tool, closes the
        session. Used for ad-hoc async access (e.g. ``list_tools``); the
        agent's hot path uses ``call_tool_sync`` which routes through the
        persistent pool instead.
        """
        from mcp.client.stdio import stdio_client
        from mcp.client.session import ClientSession

        params = self._build_stdio_params(server_name)
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return _extract_text_blocks(result.content)

    def _invoke_once(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """One synchronous attempt against the persistent session pool.

        Tests monkeypatch this method to simulate transport failures, so the
        retry/error-envelope semantics in ``call_tool_sync`` stay unit-testable
        independently of the pool/loop machinery.
        """
        params = self._build_stdio_params(server_name)
        return _SESSION_POOL.call(
            server_name, params, tool_name, arguments, _MCP_CALL_TIMEOUT_S
        )

    def call_tool_sync(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Synchronous wrapper for call_tool with automatic retry on transient
        transport failures (broken pipe, subprocess crash, JSON decode error).

        Tool-level errors — the server intentionally returning
        ``{"status": "error", ...}`` — are NOT retried here. Those are
        deliberate signals to the LLM and the ReAct loop handles re-attempts
        with corrected arguments.

        Returns either the tool's response string OR a structured error
        envelope ``{"status": "error", "error": "...", ...}`` if retries are
        exhausted. Either way the caller never sees a raised exception.
        """
        last_exc: Optional[BaseException] = None
        for attempt in range(1, _MCP_RETRY_ATTEMPTS + 1):
            try:
                return self._invoke_once(server_name, tool_name, arguments)
            except Exception as exc:  # noqa: BLE001 — we filter by name below
                last_exc = exc
                if not _is_retriable_exception(exc) or attempt >= _MCP_RETRY_ATTEMPTS:
                    break
                delay = _MCP_RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                logger.warning(
                    "MCP %s/%s attempt %d/%d failed (%s: %s) — retrying in %.2fs",
                    server_name, tool_name, attempt, _MCP_RETRY_ATTEMPTS,
                    type(exc).__name__, exc, delay,
                )
                time.sleep(delay)
        # Out of retries (or non-retriable). Surface as a structured error so
        # the LangChain shim returns a JSON the LLM can read, not a stack trace.
        err_msg = f"{type(last_exc).__name__ if last_exc else 'UnknownError'}: {last_exc}"
        logger.error(
            "MCP %s/%s failed after %d attempt(s): %s",
            server_name, tool_name, _MCP_RETRY_ATTEMPTS, err_msg,
        )
        return json.dumps({
            "status": "error",
            "result": None,
            "error": err_msg,
            "tool": tool_name,
            "server": server_name,
            "transport": "mcp_stdio",
            "attempts": _MCP_RETRY_ATTEMPTS,
        })

    async def list_tools(self, server_name: str) -> list:
        """List available tools from an MCP server."""
        from mcp.client.stdio import stdio_client
        from mcp.client.session import ClientSession

        params = self._build_stdio_params(server_name)
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                return [{"name": t.name, "description": t.description} for t in result.tools]


# ---------------------------------------------------------------------------
# Module-level singleton + envelope helpers (Stage C)
# ---------------------------------------------------------------------------
_DEFAULT_CLIENT: Optional[MCPClientManager] = None


def _project_mcp_config_path() -> str:
    """Resolve the agentic-graph-rag/.mcp.json path. Independent of cwd —
    walks up from this module's location."""
    here = Path(__file__).resolve()
    # src/tools/mcp_client.py → agentic-graph-rag/.mcp.json (3 parents up)
    return str(here.parent.parent.parent / ".mcp.json")


def get_default_client() -> MCPClientManager:
    """Return a process-singleton MCPClientManager bound to the project's
    .mcp.json. Used by every LangChain BaseTool shim in src/tools/*.py.

    Lazy-initialized so importing mcp_client doesn't open the config file
    in test/CI environments that don't have it.
    """
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = MCPClientManager(_project_mcp_config_path())
    return _DEFAULT_CLIENT


def unwrap_mcp_envelope(envelope_json: str) -> str:
    """Convert a `_common.ok/err` envelope to the legacy tool return shape.

    On ``status="ok"``, returns the inner ``result`` payload as a string —
    preserving pre-Stage-C behavior where, for example, ``calculator`` returned
    just ``"4"`` rather than a JSON object. This means LangChain prompts and
    self-critique heuristics calibrated against the old shape keep working.

    On ``status="error"``, returns the FULL envelope JSON so the LLM sees the
    structured error (``error``, ``hint``, ``traceback_excerpt``) and can
    self-correct on the next ReAct iteration.

    On malformed input (not JSON, missing fields), passes through unchanged
    rather than swallowing it — the caller can still surface raw text.
    """
    if not isinstance(envelope_json, str):
        return str(envelope_json)
    try:
        env = json.loads(envelope_json)
    except json.JSONDecodeError:
        return envelope_json
    if not isinstance(env, dict) or "status" not in env:
        return envelope_json
    if env.get("status") == "ok":
        result = env.get("result")
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        # Complex payload — re-serialize so the LLM sees structured JSON.
        return json.dumps(result)
    # status=error: surface full envelope so the model can read .error
    return envelope_json
