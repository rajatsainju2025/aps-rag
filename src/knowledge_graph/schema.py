"""
Knowledge Graph schema — node/edge type definitions.
Validated on real the facility SR logbook data: 1,812 nodes, 5,055 edges (from demo notebook).
Extended here to cover all 6 collections.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional


class NodeType(str, Enum):
    LOG_ENTRY = "LogEntry"            # the logbook entry
    PERSON = "Person"                 # Operator, engineer (username or last name)
    SYSTEM = "System"                 # SR, SR RF, SR Booster, 7-BM SR, etc.
    FAULT = "Fault"                   # Equipment failure, beam loss, vacuum event
    COMPONENT = "Component"           # BPM, kicker, PSU, AMC543, undulator, etc.
    PROCEDURE = "Procedure"           # Alignment, tune correction, injection optimization
    TIME_PERIOD = "TimePeriod"        # Shift, date range, event window
    KEYWORD = "Keyword"               # Technical term or concept
    ICMS_DOCUMENT = "ICMSDocument"    # ICMS technical document
    WORK_REQUEST = "WorkRequest"      # Maintenance work order
    TEAMS_MESSAGE = "TeamsMessage"    # Teams communication
    STUDY = "Study"                   # Accelerator physics study
    SDDS_TOOL = "SDDSTool"            # SDDS toolkit command/function
    OAG_APP = "OAGApp"               # OAG Tcl/Tk GUI application


class EdgeType(str, Enum):
    AUTHORED_BY = "AUTHORED_BY"          # Document → Person (creator)
    WORKS_ON = "WORKS_ON"               # Person → System (expertise domain)
    ABOUT_SYSTEM = "ABOUT_SYSTEM"        # Document → System
    HAS_FAULT = "HAS_FAULT"             # Document → Fault
    HAS_COMPONENT = "HAS_COMPONENT"      # Document/System → Component
    RESOLVED_BY = "RESOLVED_BY"         # Fault → Procedure
    CAUSED_BY = "CAUSED_BY"             # Fault → Component or Fault → Fault
    OCCURRED_AT = "OCCURRED_AT"         # Document → TimePeriod
    REFERENCES_DOC = "REFERENCES_DOC"    # Document → Document (cross-reference)
    RELATED_TO = "RELATED_TO"           # Generic semantic similarity
    USES_TOOL = "USES_TOOL"             # Study/LogEntry → SDDSTool
    PART_OF_STUDY = "PART_OF_STUDY"     # LogEntry → Study
    FOLLOWS_PROCEDURE = "FOLLOWS_PROCEDURE"  # Document → Procedure


@dataclass
class GraphNode:
    id: str                          # Unique identifier (e.g., "person_borland")
    node_type: NodeType
    name: str                        # Display name
    properties: Dict[str, Any] = field(default_factory=dict)
    source_collection: str = ""      # Which of the 6 collections this came from
    source_doc_id: str = ""          # Original document ID for back-reference


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    source_doc_id: str = ""          # Document this edge was extracted from


@dataclass
class KGTriplet:
    """A single extracted knowledge graph triplet."""
    subject: str
    subject_type: NodeType
    relation: EdgeType
    object: str
    object_type: NodeType
    confidence: float = 1.0
    source_doc_id: str = ""
    source_collection: str = ""


# Mapping from string labels to NodeType (for LLM output parsing)
NODE_TYPE_MAP: Dict[str, NodeType] = {
    "person": NodeType.PERSON,
    "system": NodeType.SYSTEM,
    "fault": NodeType.FAULT,
    "component": NodeType.COMPONENT,
    "procedure": NodeType.PROCEDURE,
    "concept": NodeType.KEYWORD,
    "keyword": NodeType.KEYWORD,
    "logentry": NodeType.LOG_ENTRY,
    "document": NodeType.LOG_ENTRY,
    "icmsdocument": NodeType.ICMS_DOCUMENT,
    "workrequest": NodeType.WORK_REQUEST,
    "teamsmessage": NodeType.TEAMS_MESSAGE,
    "study": NodeType.STUDY,
    "sddstool": NodeType.SDDS_TOOL,
    "oagapp": NodeType.OAG_APP,
    "timeperiod": NodeType.TIME_PERIOD,
}

EDGE_TYPE_MAP: Dict[str, EdgeType] = {
    "authored_by": EdgeType.AUTHORED_BY,
    "works_on": EdgeType.WORKS_ON,
    "about_system": EdgeType.ABOUT_SYSTEM,
    "has_fault": EdgeType.HAS_FAULT,
    "has_component": EdgeType.HAS_COMPONENT,
    "resolved_by": EdgeType.RESOLVED_BY,
    "caused_by": EdgeType.CAUSED_BY,
    "occurred_at": EdgeType.OCCURRED_AT,
    "references_doc": EdgeType.REFERENCES_DOC,
    "related_to": EdgeType.RELATED_TO,
    "uses_tool": EdgeType.USES_TOOL,
    "part_of_study": EdgeType.PART_OF_STUDY,
    "follows_procedure": EdgeType.FOLLOWS_PROCEDURE,
}
