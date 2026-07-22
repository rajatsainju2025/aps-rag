"""
LangChain Tool: WebBrowserTool — web search via Tavily API.
"""
import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _WebBrowserInput(BaseModel):
    query: str = Field(description="Search query for the web")


class WebBrowserTool(BaseTool):
    name: str = "web_browser"
    description: str = "Search the web for information"
    args_schema: Type[BaseModel] = _WebBrowserInput
    tavily_api_key: Optional[str] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, tavily_api_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "tavily_api_key", tavily_api_key)

    def _run(self, query: str) -> str:
        if not self.tavily_api_key:
            return "Web search not configured"

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.tavily_api_key)
            response = client.search(query, max_results=5)
            results = response.get("results", [])
            formatted = []
            for r in results:
                formatted.append(
                    f"Title: {r.get('title', '')}\n"
                    f"URL: {r.get('url', '')}\n"
                    f"Content: {(r.get('content') or '')[:300]}"
                )
            return "\n\n".join(formatted) if formatted else "No results found."
        except ImportError:
            logger.warning("tavily-python not installed")
            return "Web search not configured (tavily package missing)"
        except Exception as exc:
            logger.error("Web search failed: %s", exc)
            return f"Web search error: {exc}"
