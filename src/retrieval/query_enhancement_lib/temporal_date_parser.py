"""
Temporal Date Range Parser using Argo GPT-4o

This module provides intelligent temporal parsing that:
1. Analyzes user queries for temporal intent
2. Uses Argo GPT-4o to extract/infer date ranges
3. Returns structured date ranges in MM/DD/YYYY format
4. Integrates with RAG pipeline for query enhancement

Key Features:
- Natural language date understanding ("last week", "recent", "this month")
- Contextual date inference based on query intent
- Structured output with start/end dates
- Integration with MultiQuery Retriever (MQR)
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class TemporalDateRange:
    """Structured temporal date range with metadata"""
    start_date: str  # MM/DD/YYYY format
    end_date: str    # MM/DD/YYYY format
    confidence: float  # 0.0 to 1.0
    temporal_phrase: str  # Original temporal expression
    inferred: bool  # True if dates were inferred vs explicitly stated
    context: str  # Explanation of why these dates were chosen


class TemporalDateParser:
    """
    Intelligent temporal date range parser using Argo GPT-4o
    
    This parser uses LLM reasoning to:
    - Understand natural language temporal expressions
    - Infer appropriate date ranges from context
    - Handle relative dates ("recent", "last month", etc.)
    - Provide confidence scores and explanations
    
    Example:
        parser = TemporalDateParser()
        result = parser.parse_query("What were the recent BPM issues?")
        # Returns: start_date="11/04/2025", end_date="11/11/2025" (last 7 days)
    """
    
    def __init__(self, argo_model: str = "gpt-4o", argo_username: str = "your-username"):
        """
        Initialize temporal date parser with Argo GPT-4o
        
        Args:
            argo_model: Argo model name (default: gpt-4o)
            argo_username: Argo API username
        """
        self.argo_model = argo_model
        self.argo_username = argo_username
        self.llm = None
        
        # Current date for context
        self.today = datetime.now()
        self.today_str = self.today.strftime("%m/%d/%Y")
        
        logger.info(f"🕒 Temporal Date Parser initialized")
        logger.info(f"   ├─ Model: {argo_model}")
        logger.info(f"   ├─ Username: {argo_username}")
        logger.info(f"   └─ Reference date: {self.today_str}")
    
    def _get_argo_llm(self):
        """Lazy load Argo GPT-4o LLM"""
        if self.llm is None:
            try:
                from src.integrations.argo import ArgoLLMClient
                
                self.llm = ArgoLLMClient(
                    model=self.argo_model,
                    username=self.argo_username,
                    temperature=0.1,  # Low temperature for consistent date parsing
                    max_tokens=1000   # Short responses for date ranges
                )
                logger.info(f"✅ Argo LLM loaded: {self.argo_model}")
            except Exception as e:
                logger.error(f"❌ Failed to load Argo LLM: {e}")
                raise
        
        return self.llm
    
    def _create_temporal_parsing_prompt(self, query: str) -> str:
        """
        Create prompt for Argo GPT-4o to parse temporal intent
        
        Args:
            query: User query to analyze
            
        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""You are a temporal date range parser for a scientific accelerator facility query system.

Current date: {self.today_str} ({self.today.strftime("%A, %B %d, %Y")})

Task: Analyze the user query and extract or infer an appropriate date range.

User Query: "{query}"

Instructions:
1. Identify any temporal expressions (e.g., "recent", "last week", "this month", "yesterday")
2. If explicit dates are mentioned, extract them
3. If temporal expressions are vague, infer reasonable date ranges:
   - "recent" / "latest" / "new" → Last 7 days
   - "this week" → Start of current week to today
   - "last week" → Previous Monday to Sunday
   - "this month" → First day of current month to today
   - "last month" → First to last day of previous month
   - "today" → Today only
   - "yesterday" → Yesterday only
   - No temporal expression → No date filter (return null)

4. Return ONLY a valid JSON object (no markdown, no extra text):

{{
  "has_temporal_intent": true/false,
  "start_date": "MM/DD/YYYY" or null,
  "end_date": "MM/DD/YYYY" or null,
  "confidence": 0.0-1.0,
  "temporal_phrase": "extracted phrase" or null,
  "inferred": true/false,
  "context": "brief explanation"
}}

Examples:

Query: "What were the recent BPM issues?"
Output: {{"has_temporal_intent": true, "start_date": "11/04/2025", "end_date": "11/11/2025", "confidence": 0.9, "temporal_phrase": "recent", "inferred": true, "context": "Recent interpreted as last 7 days"}}

Query: "Show me work requests from last month"
Output: {{"has_temporal_intent": true, "start_date": "10/01/2025", "end_date": "10/31/2025", "confidence": 1.0, "temporal_phrase": "last month", "inferred": false, "context": "Explicit temporal reference to previous month"}}

Query: "What is the status of Q07B01?"
Output: {{"has_temporal_intent": false, "start_date": null, "end_date": null, "confidence": 1.0, "temporal_phrase": null, "inferred": false, "context": "No temporal context in query"}}

Query: "Find entries between 01/15/2025 and 02/28/2025"
Output: {{"has_temporal_intent": true, "start_date": "01/15/2025", "end_date": "02/28/2025", "confidence": 1.0, "temporal_phrase": "between 01/15/2025 and 02/28/2025", "inferred": false, "context": "Explicit date range provided"}}

Now analyze the query and return ONLY the JSON object:"""
        
        return prompt
    
    def parse_query(self, query: str) -> Optional[TemporalDateRange]:
        """
        Parse query for temporal intent and extract date range
        
        Args:
            query: User query to analyze
            
        Returns:
            TemporalDateRange object if temporal intent detected, None otherwise
        """
        try:
            logger.info(f"🕒 Parsing temporal intent from query: '{query}'")
            
            # Get Argo LLM
            llm = self._get_argo_llm()
            
            # Create prompt
            prompt = self._create_temporal_parsing_prompt(query)
            
            # Call Argo GPT-4o
            logger.info(f"📡 Calling Argo GPT-4o for temporal parsing...")
            response = llm.invoke(prompt)
            
            # Extract content
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            logger.debug(f"Raw LLM response: {response_text}")
            
            # Parse JSON response
            # Handle markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif response_text.startswith('```'):
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                logger.warning(f"Response text: {response_text}")
                return None
            
            # Check if temporal intent detected
            if not parsed.get('has_temporal_intent', False):
                logger.info("✓ No temporal intent detected in query")
                return None
            
            # Extract date range
            start_date = parsed.get('start_date')
            end_date = parsed.get('end_date')
            
            if not start_date or not end_date:
                logger.warning("Temporal intent detected but dates missing")
                return None
            
            # Validate date format
            if not self._validate_date_format(start_date) or not self._validate_date_format(end_date):
                logger.warning(f"Invalid date format: start={start_date}, end={end_date}")
                return None
            
            # Create result
            result = TemporalDateRange(
                start_date=start_date,
                end_date=end_date,
                confidence=parsed.get('confidence', 0.5),
                temporal_phrase=parsed.get('temporal_phrase', ''),
                inferred=parsed.get('inferred', True),
                context=parsed.get('context', '')
            )
            
            logger.info(f"✅ Temporal date range extracted:")
            logger.info(f"   ├─ Period: {result.start_date} to {result.end_date}")
            logger.info(f"   ├─ Phrase: '{result.temporal_phrase}'")
            logger.info(f"   ├─ Confidence: {result.confidence:.2%}")
            logger.info(f"   ├─ Inferred: {result.inferred}")
            logger.info(f"   └─ Context: {result.context}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error parsing temporal intent: {e}")
            return None
    
    def _validate_date_format(self, date_str: str) -> bool:
        """
        Validate date is in MM/DD/YYYY format
        
        Args:
            date_str: Date string to validate
            
        Returns:
            True if valid format, False otherwise
        """
        try:
            # Check format with regex
            if not re.match(r'^\d{2}/\d{2}/\d{4}$', date_str):
                return False
            
            # Try to parse to ensure valid date
            datetime.strptime(date_str, "%m/%d/%Y")
            return True
        except ValueError:
            return False
    
    def enhance_query_with_dates(self, query: str, date_range: TemporalDateRange) -> str:
        """
        Enhance query by appending date range
        
        Args:
            query: Original query
            date_range: Parsed date range
            
        Returns:
            Enhanced query with explicit date range
        """
        if not date_range:
            return query
        
        # Create date range suffix
        date_suffix = f" (date range: {date_range.start_date} to {date_range.end_date})"
        
        # Append to query
        enhanced_query = query + date_suffix
        
        logger.info(f"📝 Query enhanced with date range:")
        logger.info(f"   ├─ Original: {query}")
        logger.info(f"   └─ Enhanced: {enhanced_query}")
        
        return enhanced_query
    
    def get_date_range_filter(self, date_range: TemporalDateRange) -> Dict[str, Any]:
        """
        Convert date range to database filter format
        
        Args:
            date_range: Parsed date range
            
        Returns:
            Dictionary with filter parameters for different backends
        """
        if not date_range:
            return {}
        
        try:
            # Parse dates
            start_dt = datetime.strptime(date_range.start_date, "%m/%d/%Y")
            end_dt = datetime.strptime(date_range.end_date, "%m/%d/%Y")
            
            # Set end time to end of day
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
            
            # Create filters for different backends
            filters = {
                "qdrant": {
                    "must": [
                        {
                            "key": "creation_datetime",
                            "range": {
                                "gte": start_dt.isoformat() + "Z",
                                "lte": end_dt.isoformat() + "Z"
                            }
                        }
                    ]
                },
                "elasticsearch": {
                    "range": {
                        "creation_datetime": {
                            "gte": start_dt.strftime("%Y-%m-%d"),
                            "lte": end_dt.strftime("%Y-%m-%d")
                        }
                    }
                },
                "metadata": {
                    "start_date": date_range.start_date,
                    "end_date": date_range.end_date,
                    "start_datetime": start_dt.isoformat(),
                    "end_datetime": end_dt.isoformat()
                }
            }
            
            return filters
            
        except Exception as e:
            logger.error(f"Error creating date range filter: {e}")
            return {}


# Convenience function for direct use
def parse_temporal_query(query: str, argo_model: str = "gpt-4o") -> Optional[TemporalDateRange]:
    """
    Convenience function to parse temporal intent from query
    
    Args:
        query: User query to analyze
        argo_model: Argo model to use (default: gpt-4o)
        
    Returns:
        TemporalDateRange if temporal intent found, None otherwise
    """
    parser = TemporalDateParser(argo_model=argo_model)
    return parser.parse_query(query)
