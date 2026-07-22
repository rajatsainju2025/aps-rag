"""
LangChain Tool: TextRetrievalTool — 3-way hybrid retrieval.
"""
import json
import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _TextRetrievalInput(BaseModel):
    query: str = Field(description="Search query for document retrieval")
    top_k: int = Field(default=10, description="Number of documents to return")


class TextRetrievalTool(BaseTool):
    name: str = "text_retrieval"
    description: str = "Retrieve relevant documents using hybrid search"
    args_schema: Type[BaseModel] = _TextRetrievalInput
    vector_retriever: Any = Field(default=None, exclude=True)
    bm25_retriever: Any = Field(default=None, exclude=True)
    fusion: Any = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vector_retriever: Any = None,
        bm25_retriever: Any = None,
        fusion: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "vector_retriever", vector_retriever)
        object.__setattr__(self, "bm25_retriever", bm25_retriever)
        object.__setattr__(self, "fusion", fusion)

    def _run(self, query: str, top_k: int = 10) -> str:
        try:
            vector_docs = []
            bm25_docs = []

            if self.vector_retriever:
                vector_docs = self.vector_retriever.retrieve(query, top_k=top_k)
            if self.bm25_retriever:
                bm25_docs = self.bm25_retriever.retrieve(query, top_k=top_k)

            if self.fusion and (vector_docs or bm25_docs):
                fused = self.fusion.fuse(vector_docs=vector_docs, bm25_docs=bm25_docs, top_k=top_k)
            else:
                fused = vector_docs or bm25_docs

            serialized = []
            for doc in fused:
                if hasattr(doc, "page_content"):
                    serialized.append({
                        "content": (doc.page_content or '')[:500],
                        "metadata": doc.metadata,
                    })
                elif isinstance(doc, dict):
                    serialized.append(doc)

            return json.dumps({"documents": serialized, "count": len(serialized)}, default=str)
        except Exception as exc:
            logger.error("Text retrieval failed: %s", exc)
            return json.dumps({"error": str(exc), "documents": []})
