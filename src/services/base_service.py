"""
Abstract base class for all service modes.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Abstract base service interface."""

    @abstractmethod
    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a query and return a structured response dict."""

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return health status for this service."""

    @abstractmethod
    def get_service_mode(self) -> str:
        """Return the service mode identifier string."""
