"""
LangChain Tool: MemoryTool — read/write conversation and long-term memory.
"""
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _MemoryToolInput(BaseModel):
    action: str = Field(description="Action: 'read', 'write', or 'summarize'")
    key: str = Field(default="", description="Memory key to read or write")
    value: str = Field(default="", description="Value to write (for 'write' action)")


class MemoryTool(BaseTool):
    name: str = "memory_tool"
    description: str = "Read and write to conversation and long-term memory"
    args_schema: Type[BaseModel] = _MemoryToolInput
    memory_manager: Any = Field(default=None, exclude=True)
    session_id: str = "default"

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        memory_manager: Any = None,
        session_id: str = "default",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "memory_manager", memory_manager)
        self.session_id = session_id

    def _run(self, action: str, key: str = "", value: str = "") -> str:
        if self.memory_manager is None:
            return "Memory manager not configured"

        try:
            if action == "read":
                ctx = self.memory_manager.get_full_context(self.session_id)
                if key:
                    return str(ctx.get(key, f"Key '{key}' not found"))
                return str(ctx)

            elif action == "write":
                if not key:
                    return "Error: 'key' required for write action"
                self.memory_manager.store(self.session_id, key, value)
                return f"Stored '{key}' in memory"

            elif action == "summarize":
                summary = self.memory_manager.summarize(self.session_id)
                return str(summary)

            else:
                return f"Unknown action '{action}'. Use: read, write, summarize"

        except Exception as exc:
            logger.error("Memory tool error: %s", exc)
            return f"Memory tool error: {exc}"
