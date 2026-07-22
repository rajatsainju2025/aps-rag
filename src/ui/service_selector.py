"""
ServiceSelector — routes service_mode strings to lazily initialized service instances.
"""
import logging
from typing import Any, Dict, Optional

from src.services.base_service import BaseService

logger = logging.getLogger(__name__)

_SUPPORTED_MODES = ("naive_rag", "agentic_rag", "rag", "daily_summary", "troubleshooting", "studies_copilot")


class ServiceSelector:
    """Caches service instances per mode and returns the appropriate one."""

    def __init__(self) -> None:
        self._cache: Dict[str, BaseService] = {}

    def get_service(
        self,
        mode: str,
        settings: Any,
        components: Dict[str, Any],
    ) -> BaseService:
        if mode not in _SUPPORTED_MODES:
            logger.warning("Unknown service mode '%s', falling back to 'naive_rag'", mode)
            mode = "naive_rag"

        if mode not in self._cache:
            self._cache[mode] = self._create_service(mode, settings, components)
            logger.info("Initialized service for mode '%s'", mode)

        return self._cache[mode]

    @staticmethod
    def _create_service(
        mode: str,
        settings: Any,
        components: Dict[str, Any],
    ) -> BaseService:
        if mode == "naive_rag":
            # Route to the orchestrator-backed RAGService rather than the legacy
            # NaiveRAGService. The orchestrator's naive_rag_node already does
            # "retrieve + synthesize, no corrective loop, with tool execution
            # loop" — exactly the naive_rag contract — and crucially has tool
            # calling, which the legacy NaiveRAGService lacks. mode_label is
            # propagated so API responses still report service_mode="naive_rag".
            from src.services.rag_service import RAGService
            return RAGService(settings, components, mode_label="naive_rag")
        elif mode == "agentic_rag":
            from src.services.agentic_rag_service import AgenticRAGService
            return AgenticRAGService(settings, components)
        elif mode == "rag":
            from src.services.rag_service import RAGService
            return RAGService(settings, components, mode_label="rag")
        elif mode == "daily_summary":
            from src.services.daily_summary_service import DailySummaryService
            return DailySummaryService(settings, components)
        elif mode == "troubleshooting":
            from src.services.troubleshooting_service import TroubleshootingService
            return TroubleshootingService(settings, components)
        elif mode == "studies_copilot":
            from src.services.studies_copilot_service import StudiesCopilotService
            return StudiesCopilotService(settings, components)
        else:
            # Fallback for unknown modes — also goes through the orchestrator.
            from src.services.rag_service import RAGService
            return RAGService(settings, components, mode_label="naive_rag")
