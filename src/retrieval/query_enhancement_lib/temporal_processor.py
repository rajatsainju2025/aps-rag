"""
Consolidated Temporal Processing Module
Merges all temporal logic from:
- advanced_temporal_parser.py
- temporal_enhancement.py  
- temporal_metadata_filter.py
- temporal_normalizer.py
- temporal_filtered_retriever.py (retriever moved to rag/)

This is the SINGLE SOURCE OF TRUTH for all temporal operations in the system RAG.
"""
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class TemporalContext:
    """Represents temporal context for RAG queries"""
    current_date: str
    current_year: int
    current_month: int
    temporal_expansions: Dict[str, str]
    dataset_time_windows: Dict[str, Dict[str, Any]]


# ============================================================================
# MAIN TEMPORAL PROCESSOR (replaces TemporalProcessor + TemporalEnhancement)
# ============================================================================

class TemporalProcessor:
    """
    Consolidated temporal processor handling:
    - Query temporal expansion ("latest" -> "past 7 days")
    - Dataset time window tracking
    - Temporal context generation for prompts
    """
    
    def __init__(self):
        self.current_date = datetime.now()
        self.temporal_patterns = {
            # Relative time patterns
            r'\bthis year\b': self._expand_this_year,
            r'\blast year\b': self._expand_last_year,
            r'\bthis month\b': self._expand_this_month,
            r'\blast month\b': self._expand_last_month,
            r'\bthis week\b': self._expand_this_week,
            r'\blast week\b': self._expand_last_week,
            r'\btoday\b': self._expand_today,
            r'\byesterday\b': self._expand_yesterday,
            r'\brecently\b': self._expand_recently,
            r'\bcurrent\b': self._expand_current,
            r'\blatest\b': self._expand_latest,
            r'\bmost recent\b': self._expand_latest,
            r'\bnew\b': self._expand_recent_new,
            r'\bpast few days\b': self._expand_past_few_days,
            
            # Seasonal patterns
            r'\bthis quarter\b': self._expand_this_quarter,
            r'\blast quarter\b': self._expand_last_quarter,
            
            # Academic/fiscal year patterns
            r'\bFY\s?(\d{4})\b': self._expand_fiscal_year,
            r'\bfiscal year\s?(\d{4})?\b': self._expand_fiscal_year_generic,
        }
        
        # Dataset time windows
        self.dataset_windows = {
            'entry_logbook': {
                'start_date': '2023-01-01',
                'end_date': self.current_date.strftime('%Y-%m-%d'),
                'description': 'the logbook entries',
                'total_entries': 'Unknown',
                'peak_activity_periods': ['2023-01-01 to 2023-12-31', '2024-01-01 to 2024-12-31']
            },
            'aps_works': {
                'start_date': '2018-01-01', 
                'end_date': self.current_date.strftime('%Y-%m-%d'),
                'description': 'the facility Scientific Publications',
                'total_entries': 'Unknown',
                'peak_activity_periods': ['2022-01-01 to 2024-12-31']
            },
            'work_requests': {
                'start_date': '2019-01-01',
                'end_date': self.current_date.strftime('%Y-%m-%d'), 
                'description': 'Work Request System Data',
                'total_entries': 'Unknown',
                'peak_activity_periods': ['2023-01-01 to present']
            }
        }
    
    def get_temporal_context(self) -> TemporalContext:
        """Get current temporal context for RAG system"""
        expansions = {}
        for pattern, func in self.temporal_patterns.items():
            try:
                if callable(func):
                    expansions[pattern] = func()
                else:
                    expansions[pattern] = func
            except Exception as e:
                logger.warning(f"Failed to expand temporal pattern {pattern}: {e}")
                expansions[pattern] = ""
        
        return TemporalContext(
            current_date=self.current_date.strftime('%Y-%m-%d'),
            current_year=self.current_date.year,
            current_month=self.current_date.month,
            temporal_expansions=expansions,
            dataset_time_windows=self.dataset_windows
        )
    
    def enhance_query(self, query: str, collection_name: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance a query with temporal expansions and context
        
        Returns:
            Tuple of (enhanced_query, temporal_metadata)
        """
        enhanced_query = query
        temporal_metadata = {
            'original_query': query,
            'temporal_expansions_applied': [],
            'current_date_context': self.current_date.strftime('%Y-%m-%d'),
            'dataset_time_window': None
        }
        
        # Apply temporal expansions
        for pattern, expansion_func in self.temporal_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    if callable(expansion_func):
                        expansion = expansion_func()
                    else:
                        expansion = expansion_func
                    
                    # Replace the pattern with expanded form
                    enhanced_query = re.sub(
                        pattern, 
                        f"({match.group()} - {expansion})", 
                        enhanced_query, 
                        flags=re.IGNORECASE
                    )
                    
                    temporal_metadata['temporal_expansions_applied'].append({
                        'pattern': pattern,
                        'expansion': expansion,
                        'original_match': match.group()
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to apply temporal expansion for pattern {pattern}: {e}")
        
        # Add dataset time window info if collection is specified
        if collection_name and collection_name in self.dataset_windows:
            temporal_metadata['dataset_time_window'] = self.dataset_windows[collection_name]
        
        return enhanced_query, temporal_metadata
    
    # =================================================================
    # TEMPORAL EXPANSION METHODS
    # =================================================================
    
    def _expand_this_year(self) -> str:
        return f"{self.current_date.year} (current year)"
    
    def _expand_last_year(self) -> str:
        return f"{self.current_date.year - 1}"
    
    def _expand_this_month(self) -> str:
        return f"{self.current_date.strftime('%B %Y')}"
    
    def _expand_last_month(self) -> str:
        last_month = self.current_date.replace(day=1) - timedelta(days=1)
        return f"{last_month.strftime('%B %Y')}"
    
    def _expand_this_week(self) -> str:
        week_start = self.current_date - timedelta(days=self.current_date.weekday())
        week_end = week_start + timedelta(days=6)
        return f"{week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}"
    
    def _expand_last_week(self) -> str:
        last_week_end = self.current_date - timedelta(days=self.current_date.weekday() + 1)
        last_week_start = last_week_end - timedelta(days=6)
        return f"{last_week_start.strftime('%Y-%m-%d')} to {last_week_end.strftime('%Y-%m-%d')}"
    
    def _expand_today(self) -> str:
        return self.current_date.strftime('%Y-%m-%d')
    
    def _expand_yesterday(self) -> str:
        yesterday = self.current_date - timedelta(days=1)
        return yesterday.strftime('%Y-%m-%d')
    
    def _expand_recently(self) -> str:
        recent_start = self.current_date - timedelta(days=30)
        return f"past 30 days ({recent_start.strftime('%Y-%m-%d')} to {self.current_date.strftime('%Y-%m-%d')})"
    
    def _expand_current(self) -> str:
        return f"as of {self.current_date.strftime('%Y-%m-%d')}"
    
    def _expand_latest(self) -> str:
        """Expand 'latest' to mean the most recent 7 days"""
        recent_start = self.current_date - timedelta(days=7)
        return f"most recent entries from the past 7 days ({recent_start.strftime('%Y-%m-%d')} to {self.current_date.strftime('%Y-%m-%d')})"
    
    def _expand_recent_new(self) -> str:
        """Expand 'new' to mean the past 3 days"""
        recent_start = self.current_date - timedelta(days=3)
        return f"new entries from the past 3 days ({recent_start.strftime('%Y-%m-%d')} to {self.current_date.strftime('%Y-%m-%d')})"
    
    def _expand_past_few_days(self) -> str:
        """Expand 'past few days' to mean the past 5 days"""
        recent_start = self.current_date - timedelta(days=5)
        return f"past 5 days ({recent_start.strftime('%Y-%m-%d')} to {self.current_date.strftime('%Y-%m-%d')})"
    
    def _expand_this_quarter(self) -> str:
        quarter = (self.current_date.month - 1) // 3 + 1
        return f"Q{quarter} {self.current_date.year}"
    
    def _expand_last_quarter(self) -> str:
        if self.current_date.month <= 3:
            return f"Q4 {self.current_date.year - 1}"
        else:
            quarter = (self.current_date.month - 1) // 3
            return f"Q{quarter} {self.current_date.year}"
    
    def _expand_fiscal_year(self, match=None) -> str:
        if match and match.group(1):
            fy_year = int(match.group(1))
            return f"FY{fy_year} (October {fy_year-1} - September {fy_year})"
        return "fiscal year (October - September cycle)"
    
    def _expand_fiscal_year_generic(self, match=None) -> str:
        current_fy = self.current_date.year if self.current_date.month >= 10 else self.current_date.year - 1
        return f"FY{current_fy + 1} (October {current_fy} - September {current_fy + 1})"


# ============================================================================
# GLOBAL INSTANCE & PUBLIC API
# ============================================================================

# Global temporal processor instance (singleton pattern)
temporal_processor = TemporalProcessor()

def get_enhanced_prompt_context() -> str:
    """Get enhanced prompt context with current date and temporal understanding"""
    context = temporal_processor.get_temporal_context()
    
    return f"""The current date is {context.current_date} ({datetime.now().strftime('%B %d, %Y')}). 
When users reference temporal terms like "this year", "recently", "current", etc., 
understand them in relation to this date. Current year is {context.current_year}."""

def enhance_query_with_temporal_context(query: str, collection_name: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Public interface for enhancing queries with temporal context"""
    return temporal_processor.enhance_query(query, collection_name)

def get_dataset_time_window_info(collection_name: str) -> Optional[Dict[str, Any]]:
    """Get time window information for a specific dataset/collection"""
    return temporal_processor.dataset_windows.get(collection_name)

def format_dataset_context(collection_name: str) -> str:
    """Format dataset time window information for inclusion in prompts"""
    window_info = get_dataset_time_window_info(collection_name)
    if not window_info:
        return "Dataset time window: Information not available."
    
    return f"""Dataset Context: {window_info['description']} spanning from {window_info['start_date']} to {window_info['end_date']}. 
Peak activity periods: {', '.join(window_info['peak_activity_periods'])}."""


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Core classes
    'TemporalContext',
    'TemporalProcessor',
    
    # Global instance
    'temporal_processor',
    
    # Public API functions
    'get_enhanced_prompt_context',
    'enhance_query_with_temporal_context',
    'get_dataset_time_window_info',
    'format_dataset_context',
]
