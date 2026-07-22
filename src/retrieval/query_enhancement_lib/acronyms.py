"""Acronym resolution feature module."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AcronymMatch:
    """Represents a matched acronym with its context."""

    acronym: str
    full_form: str
    position: int
    confidence: float
    context: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)


class AcronymResolver:
    """Context-aware acronym resolution system."""

    def __init__(self, acronyms_path: Optional[str] = None):
        self.acronyms: Dict[str, List[str]] = {}
        self.reverse_map: Dict[str, List[str]] = {}
        self.context_specific: Dict[str, Dict[str, List[str]]] = {}
        self.fuzzy_threshold = 0.8
        self._load_success = False

        try:
            # Default dictionary ships EMPTY (pass-through). Users populate the
            # co-located placeholder template `data/acronyms.yaml`, or pass an
            # explicit path. See that file's header for the schema.
            path = acronyms_path or str(Path(__file__).parent / "data" / "acronyms.yaml")
            if Path(path).exists():
                self._load_acronym_dict(path)
            else:
                logger.warning("Acronym dictionary not found at %s", path)
                logger.warning("Acronym resolution will be disabled (pass-through)")

            if self.acronyms:
                self._build_context_maps()
                self._load_success = True
                logger.info("✅ Loaded %s acronyms", len(self.acronyms))
            else:
                logger.warning("⚠️ No acronyms loaded, resolver operating in pass-through mode")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("❌ Failed to initialize acronym resolver: %s", exc, exc_info=True)
            logger.warning("Acronym resolution will be disabled")

    def _load_acronym_dict(self, filepath: str) -> None:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Acronym file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        # Support both 'acronyms' and 'abbreviations' keys
        acronym_data = None
        if data:
            if "acronyms" in data and isinstance(data["acronyms"], dict):
                acronym_data = data["acronyms"]
            elif "abbreviations" in data and isinstance(data["abbreviations"], dict):
                acronym_data = data["abbreviations"]
        
        if not acronym_data:
            logger.warning("Empty or invalid YAML file (missing 'acronyms' or 'abbreviations' key): %s", filepath)
            return

        for acronym, definition in acronym_data.items():
            if not isinstance(acronym, str) or len(acronym) < 2 or len(acronym) > 10:
                logger.debug("Skipping invalid acronym: %s", acronym)
                continue

            canonical = acronym.upper()

            if isinstance(definition, str) and "|" in definition:
                forms = [item.strip() for item in definition.split("|") if item.strip()]
            else:
                forms = [str(definition).strip()] if str(definition).strip() else []

            if not forms:
                continue

            self.acronyms[canonical] = forms
            for form in forms:
                self.reverse_map.setdefault(form.lower(), []).append(canonical)

        logger.info("Loaded acronyms from %s", filepath)

    def _build_context_maps(self) -> None:
        # Optional per-collection acronym overrides (acronym → expansions that
        # apply only within a named collection). Ships empty and
        # institution-neutral; populate to bias resolution per data source.
        self.context_specific = {}

    def detect_acronyms(self, query: str) -> List[AcronymMatch]:
        pattern = re.compile(r"\b[A-Z][A-Z0-9-]{1,}[A-Z0-9]\b")
        matches: List[AcronymMatch] = []

        for result in pattern.finditer(query):
            acronym = result.group(0)
            position = result.start()

            if acronym in self.acronyms:
                forms = self.acronyms[acronym]
                matches.append(
                    AcronymMatch(
                        acronym=acronym,
                        full_form=forms[0],
                        position=position,
                        confidence=1.0,
                        alternatives=forms[1:] if len(forms) > 1 else [],
                    )
                )
            else:
                fuzzy = self._fuzzy_match_acronym(acronym)
                if fuzzy:
                    matches.append(fuzzy)

        return matches

    def _fuzzy_match_acronym(self, acronym: str) -> Optional[AcronymMatch]:
        candidates = get_close_matches(acronym, self.acronyms.keys(), n=1, cutoff=self.fuzzy_threshold)
        if not candidates:
            return None

        canonical = candidates[0]
        forms = self.acronyms[canonical]
        return AcronymMatch(
            acronym=acronym,
            full_form=forms[0],
            position=-1,
            confidence=0.7,
            alternatives=forms[1:] if len(forms) > 1 else [],
        )

    def expand(self, query: str, collection: Optional[str] = None) -> str:
        if not query or not isinstance(query, str):
            logger.warning("Invalid query for acronym expansion: %s", query)
            return str(query) if query else ""

        if not self._load_success or not self.acronyms:
            return query

        try:
            matches = self.detect_acronyms(query)
            if not matches:
                return query

            augmented = query
            for match in sorted(matches, key=lambda item: item.position, reverse=True):
                expansion = self._select_expansion(match, collection)
                insert_pos = match.position + len(match.acronym)
                augmented = augmented[:insert_pos] + f" {expansion}" + augmented[insert_pos:]

            logger.info("Expanded query: %s", augmented[:100])
            return augmented
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error in acronym expansion: %s", exc, exc_info=True)
            return query

    def _select_expansion(self, match: AcronymMatch, collection: Optional[str]) -> str:
        if collection and collection in self.context_specific:
            context_forms = self.context_specific[collection].get(match.acronym, [])
            if context_forms:
                return context_forms[0]
        return match.full_form

    def expand_with_variants(self, query: str, collection: Optional[str] = None) -> List[str]:
        variants = [query]
        matches = self.detect_acronyms(query)
        if not matches:
            return variants

        fully_expanded = query
        for match in sorted(matches, key=lambda item: item.position, reverse=True):
            expansion = self._select_expansion(match, collection)
            fully_expanded = (
                fully_expanded[: match.position]
                + expansion
                + fully_expanded[match.position + len(match.acronym) :]
            )
        variants.append(fully_expanded)

        both_forms = self.expand(query, collection)
        if both_forms != query:
            variants.append(both_forms)

        return list(dict.fromkeys(variants))

    def get_expansion_for_acronym(self, acronym: str, collection: Optional[str] = None) -> Optional[str]:
        canonical = acronym.upper()

        if collection and collection in self.context_specific:
            context_forms = self.context_specific[collection].get(canonical)
            if context_forms:
                return context_forms[0]

        if canonical in self.acronyms:
            return self.acronyms[canonical][0]

        fuzzy = self._fuzzy_match_acronym(canonical)
        if fuzzy:
            return fuzzy.full_form
        return None

    def add_custom_acronym(self, acronym: str, full_form: str, collection: Optional[str] = None) -> None:
        canonical = acronym.upper()
        if collection:
            context_map = self.context_specific.setdefault(collection, {})
            context_map.setdefault(canonical, []).append(full_form)
        else:
            self.acronyms.setdefault(canonical, []).append(full_form)
            self.reverse_map.setdefault(full_form.lower(), []).append(canonical)
        logger.info("Added custom acronym: %s -> %s", acronym, full_form)

    def disambiguate(self, acronym: str, context: str, collection: Optional[str] = None) -> str:
        canonical = acronym.upper()
        candidates: List[str] = []

        if collection and collection in self.context_specific:
            candidates.extend(self.context_specific[collection].get(canonical, []))

        if canonical in self.acronyms:
            candidates.extend(self.acronyms[canonical])

        if not candidates:
            return acronym

        if len(candidates) == 1:
            return candidates[0]

        context_lower = context.lower()
        scores = []
        for candidate in candidates:
            candidate_words = set(candidate.lower().split())
            context_words = set(context_lower.split())
            overlap = candidate_words & context_words
            confidence = len(overlap) * 2
            for word in candidate_words:
                if word in context_lower:
                    confidence += 1
            scores.append((confidence, candidate))

        scores.sort(reverse=True)
        best_confidence, best_candidate = scores[0]
        logger.debug("Disambiguation scores for %s: %s", acronym, scores)
        return best_candidate if best_confidence > 0 else candidates[0]


__all__ = ["AcronymMatch", "AcronymResolver"]
