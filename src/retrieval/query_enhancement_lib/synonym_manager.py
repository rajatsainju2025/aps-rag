"""
Synonym Management System for the system RAG
Handles operator vs. physicist terminology mapping and query expansion
"""
import logging
import yaml
import re
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class SynonymMapping:
    """Represents a synonym mapping with metadata"""
    canonical_term: str
    synonyms: List[str]
    domain: str = "general"  # e.g., "physics", "operations", "hardware"
    confidence: float = 1.0
    last_updated: str = ""
    reviewed_by: str = ""

class SynonymManager:
    """Manages operator/physicist terminology synonyms and query expansion"""
    
    def __init__(self, synonym_file_path: Optional[str] = None):
        # Default dictionary ships EMPTY (pass-through). Users populate the
        # co-located placeholder template `data/synonyms.yaml`, or pass an
        # explicit path. See that file's header for the schema.
        self.synonym_file_path = synonym_file_path or str(
            Path(__file__).parent / "data" / "synonyms.yaml"
        )
        self.synonym_maps: Dict[str, SynonymMapping] = {}
        self.reverse_index: Dict[str, str] = {}  # synonym -> canonical
        self.max_expansions = 5  # Limit to prevent query drift

        # Load built-in defaults (none by default — institution-neutral), then
        # overlay any user-provided synonyms from the YAML file.
        self._load_default_synonyms()
        self._load_synonyms_from_file()

    def _load_default_synonyms(self):
        """Built-in synonym mappings.

        Ships empty so the public release carries no domain-specific vocabulary.
        Add entries to the co-located `data/synonyms.yaml`, or hard-code a small
        default set here if you prefer to bundle a domain dictionary.
        """
        default_synonyms: Dict[str, "SynonymMapping"] = {}

        for canonical, mapping in default_synonyms.items():
            self.synonym_maps[canonical] = mapping
            self._update_reverse_index(canonical, mapping)
    
    def _update_reverse_index(self, canonical: str, mapping: SynonymMapping):
        """Update reverse lookup index"""
        # Add canonical term
        self.reverse_index[canonical.lower()] = canonical
        
        # Add all synonyms
        for synonym in mapping.synonyms:
            self.reverse_index[synonym.lower()] = canonical
    
    def _load_synonyms_from_file(self):
        """Load synonyms from YAML file if it exists"""
        try:
            synonym_path = Path(self.synonym_file_path)
            if synonym_path.exists():
                with open(synonym_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    
                for canonical, config in data.get('synonyms', {}).items():
                    mapping = SynonymMapping(
                        canonical_term=canonical,
                        synonyms=config.get('synonyms', []),
                        domain=config.get('domain', 'user_defined'),
                        confidence=config.get('confidence', 0.8),
                        last_updated=config.get('last_updated', ''),
                        reviewed_by=config.get('reviewed_by', '')
                    )
                    self.synonym_maps[canonical] = mapping
                    self._update_reverse_index(canonical, mapping)
                    
                logger.info(f"Loaded {len(data.get('synonyms', {}))} synonym mappings from {synonym_path}")
        except Exception as e:
            logger.warning(f"Could not load synonyms from {self.synonym_file_path}: {e}")
    
    def expand_query(self, query: str, max_expansions: Optional[int] = None) -> Tuple[str, Dict[str, List[str]]]:
        """
        Expand query with synonyms using MMR to avoid drift
        
        Returns:
            Tuple of (expanded_query, expansion_metadata)
        """
        max_exp = max_expansions or self.max_expansions
        expanded_terms = {}
        expanded_query = query
        
        # Find terms that have synonyms
        query_lower = query.lower()
        found_synonyms = {}
        
        for term, canonical in self.reverse_index.items():
            # Word-boundary match: a plain substring test makes 2-letter
            # synonyms (DA, ID, PV) fire inside ordinary words — "da" matches
            # to·da·y / yester·da·y / da·ta / Mon·da·y — injecting unrelated
            # beam-physics terms into nearly every operations query.
            if re.search(rf"\b{re.escape(term)}\b", query_lower):
                if canonical not in found_synonyms:
                    found_synonyms[canonical] = []
                found_synonyms[canonical].append(term)
        
        # Expand each canonical term with its synonyms (limited by max_expansions)
        for canonical, found_terms in found_synonyms.items():
            mapping = self.synonym_maps.get(canonical)
            if not mapping:
                continue
                
            # Get synonyms, limit to max_expansions
            synonyms = mapping.synonyms[:max_exp]
            
            # Build expansion for the first found term
            if found_terms:
                original_term = found_terms[0]
                # Create expansion: "original_term (also: synonym1, synonym2)"
                if synonyms:
                    synonym_list = ", ".join(synonyms[:3])  # Show top 3 in expansion
                    expansion = f"{original_term} (also: {synonym_list})"
                    
                    # Replace in query using case-insensitive, word-boundary match
                    pattern = re.compile(rf"\b{re.escape(original_term)}\b", re.IGNORECASE)
                    expanded_query = pattern.sub(expansion, expanded_query, count=1)
                    
                    expanded_terms[canonical] = {
                        'original': original_term,
                        'synonyms': synonyms,
                        'domain': mapping.domain
                    }
        
        expansion_metadata = {
            'expanded_terms': expanded_terms,
            'total_expansions': len(expanded_terms),
            'query_drift_risk': 'high' if len(expanded_terms) > 3 else 'low'
        }
        
        return expanded_query, expansion_metadata
    
    def get_canonical_term(self, term: str) -> Optional[str]:
        """Get canonical term for a given synonym"""
        return self.reverse_index.get(term.lower())
    
    def get_synonyms(self, term: str) -> List[str]:
        """Get all synonyms for a term (including canonical)"""
        canonical = self.get_canonical_term(term)
        if canonical and canonical in self.synonym_maps:
            mapping = self.synonym_maps[canonical]
            return [canonical] + mapping.synonyms
        return [term]
    
    def add_synonym_mapping(self, canonical: str, synonyms: List[str], domain: str = "user_defined", 
                          confidence: float = 0.8, reviewed_by: str = "") -> bool:
        """Add a new synonym mapping"""
        try:
            mapping = SynonymMapping(
                canonical_term=canonical,
                synonyms=synonyms,
                domain=domain,
                confidence=confidence,
                reviewed_by=reviewed_by
            )
            
            self.synonym_maps[canonical] = mapping
            self._update_reverse_index(canonical, mapping)
            logger.info(f"Added synonym mapping for '{canonical}' with {len(synonyms)} synonyms")
            return True
        except Exception as e:
            logger.error(f"Failed to add synonym mapping for '{canonical}': {e}")
            return False
    
    def save_synonyms_to_file(self) -> bool:
        """Save current synonym mappings to YAML file"""
        try:
            # Prepare data for YAML export
            data = {
                'synonyms': {},
                'metadata': {
                    'version': '1.0',
                    'description': 'the system RAG synonym mappings for operator/physicist terminology',
                    'last_updated': '2025-09-02'
                }
            }
            
            for canonical, mapping in self.synonym_maps.items():
                data['synonyms'][canonical] = {
                    'synonyms': mapping.synonyms,
                    'domain': mapping.domain,
                    'confidence': mapping.confidence,
                    'reviewed_by': mapping.reviewed_by,
                    'last_updated': mapping.last_updated or '2025-09-02'
                }
            
            # Ensure directory exists
            Path(self.synonym_file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write YAML file
            with open(self.synonym_file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=True)
            
            logger.info(f"Saved {len(self.synonym_maps)} synonym mappings to {self.synonym_file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save synonyms to {self.synonym_file_path}: {e}")
            return False
    
    def get_domain_terms(self, domain: str) -> Dict[str, SynonymMapping]:
        """Get all terms for a specific domain"""
        return {k: v for k, v in self.synonym_maps.items() if v.domain == domain}
    
    def suggest_synonym_candidates(self, query_terms: List[str]) -> List[Tuple[str, str, float]]:
        """Suggest potential synonym relationships based on co-occurrence"""
        # This is a placeholder for the "learned alias map" approach
        # In a full implementation, this would analyze embedding clusters
        # and co-occurrence patterns to suggest new synonyms
        suggestions = []
        
        # Simple heuristic: look for terms that might be acronyms
        for term in query_terms:
            if len(term) <= 5 and term.isupper():
                # Look for potential expansions in existing synonyms
                for canonical, mapping in self.synonym_maps.items():
                    if term.lower() not in [s.lower() for s in mapping.synonyms + [canonical]]:
                        # Check if term might be related (simple character overlap)
                        if any(term.lower() in s.lower() for s in mapping.synonyms):
                            suggestions.append((term, canonical, 0.6))
        
        return suggestions[:5]  # Return top 5 suggestions

# Global synonym manager instance
synonym_manager = SynonymManager()

def expand_query_with_synonyms(query: str, max_expansions: int = 5) -> Tuple[str, Dict]:
    """Public interface for query expansion with synonyms"""
    return synonym_manager.expand_query(query, max_expansions)

def get_synonyms_for_term(term: str) -> List[str]:
    """Get all synonyms for a given term"""
    return synonym_manager.get_synonyms(term)
