"""
Incremental KG updater — hash-based change detection for delta builds.
Avoids re-processing unchanged documents when syncing from entry_rag_app.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent.parent.parent / "data" / "kg_sync_state.json"


class IncrementalKGUpdater:
    """
    Tracks document hashes to detect new/modified documents.
    Only processes changes — critical for large collections like ICMS (~8 GB).
    """

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._hashes: Dict[str, str] = self._load_state()
        # Hashes staged by get_delta() but not yet written to disk —
        # only persisted by commit_pending() after successful extraction,
        # so a crashed build can never orphan a doc by marking it "processed"
        # before its entities/relations are stored in Neo4j.
        self._pending: Dict[str, str] = {}

    def _load_state(self) -> Dict[str, str]:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {}

    def _save_state(self) -> None:
        with open(self.state_file, "w") as f:
            json.dump(self._hashes, f, indent=2)

    def _hash_document(self, doc: Dict[str, Any]) -> str:
        content = doc.get("content", "") or doc.get("page_content", "")
        return hashlib.sha256(content.encode()).hexdigest()

    def get_delta(
        self, documents: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Partition documents into new/changed and unchanged.

        Returns:
            (to_process, unchanged) — only to_process needs KG extraction.
        """
        to_process: List[Dict[str, Any]] = []
        unchanged: List[Dict[str, Any]] = []

        for doc in documents:
            doc_id = doc.get("id", "")
            current_hash = self._hash_document(doc)
            stored_hash = self._hashes.get(doc_id)

            if stored_hash != current_hash:
                to_process.append(doc)
                self._pending[doc_id] = current_hash
            else:
                unchanged.append(doc)

        logger.info(
            f"Delta check: {len(to_process)} new/changed, {len(unchanged)} unchanged"
        )
        return to_process, unchanged

    def commit_pending(self) -> int:
        """Promote pending hashes to persisted state after successful extraction.

        Call this only after the corresponding entity/relation extraction and
        Neo4j persistence have succeeded — otherwise an interrupted run would
        silently orphan docs that were marked processed but never indexed.
        """
        if not self._pending:
            return 0
        n = len(self._pending)
        self._hashes.update(self._pending)
        self._pending = {}
        self._save_state()
        return n

    def mark_processed(self, doc_ids: List[str]) -> None:
        """Explicitly mark document IDs as processed (no re-processing needed)."""
        for doc_id in doc_ids:
            if doc_id not in self._hashes:
                self._hashes[doc_id] = "processed"
        self._save_state()

    def reset(self) -> None:
        """Clear all state — forces full rebuild on next run."""
        self._hashes = {}
        self._save_state()
        logger.info("KG sync state reset — full rebuild on next run")
