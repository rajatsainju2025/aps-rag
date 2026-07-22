"""
Query Understanding and Intent Recognition for the system RAG
Provides advanced query parsing, intent classification, and context extraction
"""
import logging
import re
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Types of user query intents"""
    FACTUAL_LOOKUP = "factual"           # Looking for specific facts/data
    DIAGNOSTIC_HELP = "diagnostic"       # Troubleshooting/problem solving
    PROCEDURAL_GUIDANCE = "procedural"   # How-to/procedure questions
    COMPARATIVE_ANALYSIS = "comparative" # Comparing systems/approaches
    HISTORICAL_ANALYSIS = "historical"   # Understanding past events
    TREND_ANALYSIS = "trend"            # Understanding patterns/trends
    PREDICTIVE = "predictive"           # Future planning/prediction
    CAUSAL_ANALYSIS = "causal"          # Understanding cause-effect
    STATUS_CHECK = "status"             # Current system status
    CONFIGURATION = "configuration"     # System configuration questions

class QueryComplexity(Enum):
    """Complexity levels of queries"""
    SIMPLE = "simple"         # Single concept, direct answer
    MODERATE = "moderate"     # Multiple concepts, some reasoning
    COMPLEX = "complex"       # Multiple systems, complex reasoning
    RESEARCH = "research"     # Open-ended, requires synthesis

class QueryDomain(Enum):
    """Domain areas for accelerator queries"""
    BEAM_PHYSICS = "beam_physics"
    MACHINE_OPERATIONS = "operations"
    CONTROLS_SYSTEMS = "controls"
    VACUUM_SYSTEMS = "vacuum"
    RF_SYSTEMS = "rf"
    MAGNET_SYSTEMS = "magnets"
    DIAGNOSTICS = "diagnostics"
    SAFETY_SYSTEMS = "safety"
    MAINTENANCE = "maintenance"
    GENERAL = "general"

@dataclass
class EntityExtraction:
    """Extracted entities from query"""
    devices: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    measurements: List[str] = field(default_factory=list)

@dataclass
class QueryContext:
    """Context information for query understanding"""
    user_role: Optional[str] = None
    session_history: List[str] = field(default_factory=list)
    current_shift: Optional[str] = None
    active_systems: List[str] = field(default_factory=list)
    recent_incidents: List[str] = field(default_factory=list)
    maintenance_status: Dict[str, str] = field(default_factory=dict)

@dataclass
class ParsedQuery:
    """Comprehensive query analysis result"""
    original_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    domain: QueryDomain
    entities: EntityExtraction
    context: QueryContext
    confidence: float
    key_concepts: List[str]
    implied_constraints: Dict[str, Any]
    suggested_expansions: List[str]
    ambiguities: List[str]
    processing_hints: Dict[str, Any]

class QueryUnderstanding:
    """Advanced query understanding and intent recognition"""
    
    def __init__(self):
        self.intent_patterns = self._create_intent_patterns()
        self.complexity_indicators = self._create_complexity_indicators()
        self.domain_keywords = self._create_domain_keywords()
        self.entity_patterns = self._create_entity_patterns()
        self.parameter_mappings = self._create_parameter_mappings()
        
    def _create_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Create patterns for intent recognition"""
        return {
            QueryIntent.FACTUAL_LOOKUP: [
                r"what is", r"what was", r"define", r"explain", r"show me",
                r"value of", r"reading", r"measurement", r"data for"
            ],
            QueryIntent.DIAGNOSTIC_HELP: [
                r"problem", r"issue", r"error", r"fault", r"troubleshoot",
                r"not working", r"failed", r"broken", r"abnormal", r"wrong"
            ],
            QueryIntent.PROCEDURAL_GUIDANCE: [
                r"how to", r"procedure", r"steps", r"process", r"method",
                r"guide", r"instruction", r"protocol", r"standard"
            ],
            QueryIntent.COMPARATIVE_ANALYSIS: [
                r"compare", r"versus", r"difference", r"similar", r"better",
                r"worse", r"contrast", r"relative", r"compared to"
            ],
            QueryIntent.HISTORICAL_ANALYSIS: [
                r"history", r"past", r"previous", r"before", r"earlier",
                r"last time", r"when did", r"timeline", r"chronicle"
            ],
            QueryIntent.TREND_ANALYSIS: [
                r"trend", r"pattern", r"over time", r"progression", r"evolution",
                r"increasing", r"decreasing", r"changing", r"development"
            ],
            QueryIntent.PREDICTIVE: [
                r"predict", r"forecast", r"future", r"next", r"will",
                r"expect", r"anticipate", r"projection", r"outlook"
            ],
            QueryIntent.CAUSAL_ANALYSIS: [
                r"why", r"cause", r"reason", r"because", r"due to",
                r"resulted in", r"led to", r"triggered", r"explanation"
            ],
            QueryIntent.STATUS_CHECK: [
                r"status", r"current", r"now", r"present", r"today",
                r"operational", r"running", r"active", r"online"
            ],
            QueryIntent.CONFIGURATION: [
                r"setting", r"configuration", r"parameter", r"value",
                r"setup", r"tune", r"adjust", r"optimize", r"calibrate"
            ]
        }
    
    def _create_complexity_indicators(self) -> Dict[QueryComplexity, List[str]]:
        """Create indicators for query complexity"""
        return {
            QueryComplexity.SIMPLE: [
                # Single concept indicators
                r"^what is \w+", r"^status of \w+", r"^value of \w+"
            ],
            QueryComplexity.MODERATE: [
                # Multiple concept indicators
                r"\w+ and \w+", r"\w+ or \w+", r"during \w+", r"after \w+"
            ],
            QueryComplexity.COMPLEX: [
                # Complex reasoning indicators
                r"relationship between", r"impact of \w+ on \w+", 
                r"correlation", r"optimization", r"multiple"
            ],
            QueryComplexity.RESEARCH: [
                # Research-level indicators
                r"comprehensive", r"detailed analysis", r"investigation",
                r"all instances", r"systematic", r"thorough"
            ]
        }
    
    def _create_domain_keywords(self) -> Dict[QueryDomain, List[str]]:
        """Create domain classification keywords"""
        return {
            QueryDomain.BEAM_PHYSICS: [
                "beam", "emittance", "tune", "chromaticity", "orbit", "optics",
                "dynamics", "lifetime", "aperture", "coupling", "dispersion"
            ],
            QueryDomain.MACHINE_OPERATIONS: [
                "operations", "shift", "run", "study", "commissioning",
                "delivery", "schedule", "mode", "configuration"
            ],
            QueryDomain.CONTROLS_SYSTEMS: [
                "epics", "pv", "ioc", "control", "feedback", "alarm",
                "interface", "hmi", "scada", "automation"
            ],
            QueryDomain.VACUUM_SYSTEMS: [
                "vacuum", "pressure", "pump", "gauge", "leak", "chamber",
                "outgassing", "bakeout", "venting", "roughing"
            ],
            QueryDomain.RF_SYSTEMS: [
                "rf", "cavity", "klystron", "frequency", "phase", "power",
                "amplifier", "circulator", "waveguide", "llrf"
            ],
            QueryDomain.MAGNET_SYSTEMS: [
                "magnet", "quad", "dipole", "sext", "corrector", "field",
                "current", "power supply", "ps", "trim"
            ],
            QueryDomain.DIAGNOSTICS: [
                "bpm", "screen", "camera", "detector", "monitor", "measurement",
                "diagnostic", "calibration", "tune measurement"
            ],
            QueryDomain.SAFETY_SYSTEMS: [
                "safety", "interlock", "pps", "radiation", "permit",
                "protection", "emergency", "personnel"
            ],
            QueryDomain.MAINTENANCE: [
                "maintenance", "repair", "replace", "service", "pm",
                "preventive", "corrective", "scheduled"
            ]
        }
    
    def _create_entity_patterns(self) -> Dict[str, List[str]]:
        """Create patterns for entity extraction"""
        return {
            "devices": [
                r"[A-Z]+\d+[A-Z]?",  # Standard device codes
                r"BPM\d+", r"Q\d+[A-Z]?", r"S\d+[A-Z]?", r"ID\d+",
                r"RF\d+", r"PS\d+", r"VAC\d+"
            ],
            "parameters": [
                r"(?:beam )?current", r"(?:beam )?lifetime", r"(?:beam )?size",
                r"emittance", r"tune", r"chromaticity", r"orbit",
                r"pressure", r"temperature", r"voltage", r"frequency"
            ],
            "timeframes": [
                r"\d{4}-\d{2}-\d{2}", r"yesterday", r"today", r"last week",
                r"this month", r"Q[1-4] \d{4}", r"shift", r"during"
            ],
            "locations": [
                r"sector \d+", r"S\d+", r"straight section", r"cell \d+",
                r"beamline", r"tunnel", r"gallery", r"control room"
            ],
            "measurements": [
                r"\d+\.?\d*\s*(?:μm|mm|cm|m|km)",  # Length units
                r"\d+\.?\d*\s*(?:nA|μA|mA|A)",     # Current units
                r"\d+\.?\d*\s*(?:Hz|kHz|MHz|GHz)", # Frequency units
                r"\d+\.?\d*\s*(?:V|kV|MV)",        # Voltage units
                r"\d+\.?\d*\s*(?:Torr|mbar|Pa)",   # Pressure units
            ]
        }
    
    def _create_parameter_mappings(self) -> Dict[str, List[str]]:
        """Create mappings for parameter synonyms"""
        return {
            "beam_current": ["current", "beam current", "stored current", "I"],
            "beam_lifetime": ["lifetime", "beam lifetime", "tau", "τ"],
            "emittance": ["emittance", "beam emittance", "ε", "εx", "εy"],
            "tune": ["tune", "betatron tune", "Qx", "Qy", "ν"],
            "chromaticity": ["chromaticity", "ξ", "xi", "dQ/dp"],
            "orbit": ["orbit", "closed orbit", "beam position", "trajectory"],
            "coupling": ["coupling", "xy coupling", "betatron coupling"],
            "dispersion": ["dispersion", "η", "eta", "momentum compaction"]
        }
    
    def parse_query(self, query: str, context: Optional[QueryContext] = None) -> ParsedQuery:
        """Parse and analyze a user query comprehensively"""
        if context is None:
            context = QueryContext()
        
        # Clean and normalize query
        normalized_query = self._normalize_query(query)
        
        # Extract entities
        entities = self._extract_entities(normalized_query)
        
        # Classify intent
        intent = self._classify_intent(normalized_query)
        
        # Determine complexity
        complexity = self._determine_complexity(normalized_query, entities)
        
        # Classify domain
        domain = self._classify_domain(normalized_query, entities)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(normalized_query, entities)
        
        # Infer constraints
        constraints = self._infer_constraints(normalized_query, entities, context)
        
        # Generate expansions
        expansions = self._generate_expansions(normalized_query, entities, intent)
        
        # Identify ambiguities
        ambiguities = self._identify_ambiguities(normalized_query, entities)
        
        # Calculate confidence
        confidence = self._calculate_parsing_confidence(
            normalized_query, intent, complexity, domain, entities
        )
        
        # Generate processing hints
        processing_hints = self._generate_processing_hints(
            intent, complexity, domain, entities
        )
        
        return ParsedQuery(
            original_query=query,
            intent=intent,
            complexity=complexity,
            domain=domain,
            entities=entities,
            context=context,
            confidence=confidence,
            key_concepts=key_concepts,
            implied_constraints=constraints,
            suggested_expansions=expansions,
            ambiguities=ambiguities,
            processing_hints=processing_hints
        )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text"""
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Expand common abbreviations
        abbreviations = {
            r'\bsr\b': 'storage ring',
            r'\bbpm\b': 'beam position monitor',
            r'\brf\b': 'radio frequency',
            r'\bpv\b': 'process variable',
            r'\bps\b': 'power supply',
            r'\bhv\b': 'high voltage',
            r'\blv\b': 'low voltage',
            r'\bioc\b': 'input output controller'
        }
        
        for abbrev, expansion in abbreviations.items():
            normalized = re.sub(abbrev, expansion, normalized)
        
        return normalized
    
    def _extract_entities(self, query: str) -> EntityExtraction:
        """Extract entities from normalized query"""
        entities = EntityExtraction()
        
        # Extract devices
        for pattern in self.entity_patterns["devices"]:
            entities.devices.extend(re.findall(pattern, query, re.IGNORECASE))
        
        # Extract parameters
        for pattern in self.entity_patterns["parameters"]:
            entities.parameters.extend(re.findall(pattern, query, re.IGNORECASE))
        
        # Extract timeframes
        for pattern in self.entity_patterns["timeframes"]:
            entities.timeframes.extend(re.findall(pattern, query, re.IGNORECASE))
        
        # Extract locations
        for pattern in self.entity_patterns["locations"]:
            entities.locations.extend(re.findall(pattern, query, re.IGNORECASE))
        
        # Extract measurements
        for pattern in self.entity_patterns["measurements"]:
            entities.measurements.extend(re.findall(pattern, query, re.IGNORECASE))
        
        # Remove duplicates
        entities.devices = list(set(entities.devices))
        entities.parameters = list(set(entities.parameters))
        entities.timeframes = list(set(entities.timeframes))
        entities.locations = list(set(entities.locations))
        entities.measurements = list(set(entities.measurements))
        
        return entities
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of the query"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            intent_scores[intent] = score
        
        # Return highest scoring intent, or factual as default
        if intent_scores and max(intent_scores.values()) > 0:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return QueryIntent.FACTUAL_LOOKUP
    
    def _determine_complexity(self, query: str, entities: EntityExtraction) -> QueryComplexity:
        """Determine query complexity"""
        # Count various complexity indicators
        complexity_score = 0
        
        # Multiple entities increase complexity
        total_entities = (len(entities.devices) + len(entities.parameters) + 
                         len(entities.timeframes) + len(entities.locations))
        
        if total_entities > 3:
            complexity_score += 2
        elif total_entities > 1:
            complexity_score += 1
        
        # Check for complexity patterns
        for complexity, patterns in self.complexity_indicators.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    if complexity == QueryComplexity.COMPLEX:
                        complexity_score += 3
                    elif complexity == QueryComplexity.RESEARCH:
                        complexity_score += 4
                    elif complexity == QueryComplexity.MODERATE:
                        complexity_score += 1
        
        # Determine complexity level
        if complexity_score >= 4:
            return QueryComplexity.RESEARCH
        elif complexity_score >= 3:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 1:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _classify_domain(self, query: str, entities: EntityExtraction) -> QueryDomain:
        """Classify the domain of the query"""
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    score += 1
            domain_scores[domain] = score
        
        # Boost scores based on entities
        if entities.devices:
            # Analyze device types to infer domain
            for device in entities.devices:
                if re.match(r'BPM\d+', device, re.IGNORECASE):
                    domain_scores[QueryDomain.DIAGNOSTICS] = domain_scores.get(QueryDomain.DIAGNOSTICS, 0) + 2
                elif re.match(r'Q\d+', device, re.IGNORECASE):
                    domain_scores[QueryDomain.MAGNET_SYSTEMS] = domain_scores.get(QueryDomain.MAGNET_SYSTEMS, 0) + 2
                elif re.match(r'RF\d+', device, re.IGNORECASE):
                    domain_scores[QueryDomain.RF_SYSTEMS] = domain_scores.get(QueryDomain.RF_SYSTEMS, 0) + 2
        
        # Return highest scoring domain, or general as default
        if domain_scores and max(domain_scores.values()) > 0:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return QueryDomain.GENERAL
    
    def _extract_key_concepts(self, query: str, entities: EntityExtraction) -> List[str]:
        """Extract key concepts from the query"""
        concepts = []
        
        # Add significant entities
        concepts.extend(entities.devices[:3])  # Limit to top 3
        concepts.extend(entities.parameters[:3])
        
        # Extract important nouns and technical terms
        technical_terms = [
            "emittance", "tune", "chromaticity", "orbit", "feedback",
            "vacuum", "pressure", "current", "lifetime", "beam",
            "magnet", "rf", "cavity", "power", "supply", "measurement"
        ]
        
        for term in technical_terms:
            if term in query and term not in concepts:
                concepts.append(term)
        
        return concepts[:10]  # Limit to 10 key concepts
    
    def _infer_constraints(self, query: str, entities: EntityExtraction, 
                          context: QueryContext) -> Dict[str, Any]:
        """Infer implicit constraints from query and context"""
        constraints = {}
        
        # Time constraints
        if entities.timeframes:
            constraints["time_filter"] = entities.timeframes
        
        # Device constraints
        if entities.devices:
            constraints["device_filter"] = entities.devices
        
        # Location constraints
        if entities.locations:
            constraints["location_filter"] = entities.locations
        
        # Context-based constraints
        if context.current_shift:
            constraints["shift_context"] = context.current_shift
        
        if context.active_systems:
            constraints["system_context"] = context.active_systems
        
        # Implicit scope constraints
        if "storage ring" in query:
            constraints["machine_scope"] = "storage_ring"
        elif "injector" in query or "linac" in query:
            constraints["machine_scope"] = "injector"
        
        return constraints
    
    def _generate_expansions(self, query: str, entities: EntityExtraction, 
                           intent: QueryIntent) -> List[str]:
        """Generate query expansions based on analysis"""
        expansions = []
        
        # Intent-based expansions
        if intent == QueryIntent.DIAGNOSTIC_HELP:
            expansions.extend([
                "troubleshooting procedures",
                "error symptoms",
                "diagnostic steps",
                "resolution methods"
            ])
        elif intent == QueryIntent.HISTORICAL_ANALYSIS:
            expansions.extend([
                "timeline of events",
                "historical data",
                "past incidents",
                "trend analysis"
            ])
        
        # Entity-based expansions
        if entities.devices:
            for device in entities.devices[:2]:
                expansions.append(f"related {device} activities")
                expansions.append(f"{device} configuration")
        
        # Parameter-based expansions
        if entities.parameters:
            for param in entities.parameters[:2]:
                expansions.append(f"{param} measurements")
                expansions.append(f"{param} optimization")
        
        return expansions[:8]  # Limit expansions
    
    def _identify_ambiguities(self, query: str, entities: EntityExtraction) -> List[str]:
        """Identify potential ambiguities in the query"""
        ambiguities = []
        
        # Check for ambiguous device references
        if "Q1" in query and not any("Q1" in device for device in entities.devices):
            ambiguities.append("Q1 could refer to multiple quadrupole magnets")
        
        # Check for ambiguous time references
        if "last" in query and not entities.timeframes:
            ambiguities.append("'last' timeframe is ambiguous - last shift/day/week?")
        
        # Check for parameter ambiguities
        if "current" in query:
            ambiguities.append("'current' could mean beam current or present status")
        
        # Check for pronoun ambiguities
        pronouns = ["it", "this", "that", "they"]
        for pronoun in pronouns:
            if pronoun in query:
                ambiguities.append(f"Pronoun '{pronoun}' reference unclear")
        
        return ambiguities
    
    def _calculate_parsing_confidence(self, query: str, intent: QueryIntent,
                                    complexity: QueryComplexity, domain: QueryDomain,
                                    entities: EntityExtraction) -> float:
        """Calculate confidence in parsing results"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for clear patterns
        if len(entities.devices) > 0:
            confidence += 0.2
        
        if len(entities.parameters) > 0:
            confidence += 0.15
        
        if len(entities.timeframes) > 0:
            confidence += 0.1
        
        # Reduce confidence for complexity
        if complexity == QueryComplexity.COMPLEX:
            confidence -= 0.1
        elif complexity == QueryComplexity.RESEARCH:
            confidence -= 0.2
        
        # Boost confidence for domain specificity
        if domain != QueryDomain.GENERAL:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_processing_hints(self, intent: QueryIntent, complexity: QueryComplexity,
                                 domain: QueryDomain, entities: EntityExtraction) -> Dict[str, Any]:
        """Generate hints for query processing"""
        hints = {
            "search_strategy": "semantic",
            "result_limit": 10,
            "aggregation_needed": False,
            "temporal_filtering": False,
            "cross_reference": False
        }
        
        # Adjust based on intent
        if intent == QueryIntent.HISTORICAL_ANALYSIS:
            hints["temporal_filtering"] = True
            hints["aggregation_needed"] = True
            hints["search_strategy"] = "temporal"
        
        elif intent == QueryIntent.DIAGNOSTIC_HELP:
            hints["search_strategy"] = "problem_focused"
            hints["cross_reference"] = True
            hints["result_limit"] = 15
        
        elif intent == QueryIntent.COMPARATIVE_ANALYSIS:
            hints["aggregation_needed"] = True
            hints["cross_reference"] = True
        
        # Adjust based on complexity
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.RESEARCH]:
            hints["result_limit"] = 20
            hints["aggregation_needed"] = True
            hints["cross_reference"] = True
        
        # Adjust based on entities
        if entities.devices:
            hints["device_filtering"] = True
        
        if entities.timeframes:
            hints["temporal_filtering"] = True
        
        return hints

# Global query understanding instance
query_understanding = QueryUnderstanding()

def parse_user_query(query: str, context: Optional[QueryContext] = None) -> ParsedQuery:
    """Parse and analyze a user query comprehensively"""
    return query_understanding.parse_query(query, context)

def extract_query_entities(query: str) -> EntityExtraction:
    """Extract entities from a query"""
    normalized_query = query_understanding._normalize_query(query)
    return query_understanding._extract_entities(normalized_query)

def classify_query_intent(query: str) -> QueryIntent:
    """Classify the intent of a query"""
    normalized_query = query_understanding._normalize_query(query)
    return query_understanding._classify_intent(normalized_query)
