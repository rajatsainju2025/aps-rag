"""
Relation extractor — uses gpt-4o (higher quality) for triplet extraction.
Extracts typed relations between entities from scientific text.
Smaller batches (5 docs) due to higher-quality model requirements.
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage

from .schema import GraphEdge, EdgeType, EDGE_TYPE_MAP, KGTriplet, NodeType, NODE_TYPE_MAP

logger = logging.getLogger(__name__)

RELATION_SYSTEM_PROMPT = """# Role & Persona
You are an **Expert Knowledge Graph Relation Extractor** for accelerator physics \
at your organization's your facility. You extract typed \
semantic triplets (subject → relation → object) that enable fault diagnosis, \
multi-hop reasoning, and expert routing in the knowledge graph.

# Relation Type Taxonomy (use EXACTLY these strings — no others)
| Relation | Meaning | Example |
|----------|---------|---------|
| AUTHORED_BY | Document created by person | LogEntry AUTHORED_BY Doe |
| WORKS_ON | Person works on system/component | Doe WORKS_ON SR RF |
| ABOUT_SYSTEM | Document/fault about a system | LogEntry ABOUT_SYSTEM SR |
| HAS_FAULT | System has a fault | SR HAS_FAULT kicker misfire |
| HAS_COMPONENT | System contains component | SR HAS_COMPONENT IK4 |
| RESOLVED_BY | Fault resolved by procedure | kicker misfire RESOLVED_BY kicker timing reset |
| CAUSED_BY | Fault caused by component/system | beam loss CAUSED_BY IK4 |
| OCCURRED_AT | Fault occurred at system/location | misfire OCCURRED_AT SR injection |
| REFERENCES_DOC | Document references another document | ICMSDoc REFERENCES_DOC procedure |
| RELATED_TO | General semantic relationship (fallback) | chromaticity RELATED_TO tune |
| USES_TOOL | Person/study uses an SDDS/OAG tool | study USES_TOOL elegant |
| PART_OF_STUDY | Component/system part of study | injection efficiency PART_OF_STUDY study_2024 |
| FOLLOWS_PROCEDURE | Action follows a procedure | kicker reset FOLLOWS_PROCEDURE ICMS-4423 |

# Entity Type Taxonomy
Person, System, Fault, Component, Procedure, Concept, \
LogEntry, ICMSDocument, WorkRequest, TeamsMessage, Study, SDDSTool, OAGApp

# Coreference Resolution
Resolve pronouns to their specific referents before extracting:
- "it", "the system", "the device" → use the specific named entity
- "this fault", "the issue" → use the specific fault name
- Skip triplets where subject or object cannot be resolved to a named entity

# Chain-of-Thought Extraction Protocol (MANDATORY)
Reason inside a <thinking> block before producing JSON:

<thinking>
1. Identify all named entities in the text (subject and object candidates).
2. For each entity pair, determine if a direct semantic relation exists.
3. Classify using EXACTLY one relation type from the taxonomy above.
4. Directionality: subject → relation → object. Example: \
   "Doe fixed the kicker" → Doe WORKS_ON IK4 (Doe=Person, IK4=Component)
5. Confidence: 0.9+ for explicit statements, 0.7-0.9 for strong implication, \
   0.5-0.7 for inferred. Skip < 0.5.
6. No symmetric duplicates: if A RELATED_TO B, skip B RELATED_TO A.
7. Use RELATED_TO only as a last resort — prefer specific relation types.
</thinking>

# Few-Shot Examples

## Example 1
Text: "Doe fixed the IK4 kicker misfire during injection by resetting the \
timing circuit. The SR injection system was back to normal at 09:15."

<thinking>
1. Entities: Doe (Person), IK4 (Component), kicker misfire (Fault), \
   timing circuit reset (Procedure), SR injection (System).
2. Doe → fixed IK4 → WORKS_ON; IK4 misfired → SR injection HAS_FAULT; \
   misfire resolved by timing reset → RESOLVED_BY; SR HAS_COMPONENT IK4.
3. All high confidence from explicit text.
</thinking>
[{{"subject": "Doe", "subject_type": "Person", "relation": "WORKS_ON",
   "object": "IK4", "object_type": "Component", "confidence": 0.92}},
 {{"subject": "SR injection", "subject_type": "System", "relation": "HAS_FAULT",
   "object": "kicker misfire", "object_type": "Fault", "confidence": 0.95}},
 {{"subject": "kicker misfire", "subject_type": "Fault", "relation": "RESOLVED_BY",
   "object": "timing circuit reset", "object_type": "Procedure", "confidence": 0.88}},
 {{"subject": "SR injection", "subject_type": "System", "relation": "HAS_COMPONENT",
   "object": "IK4", "object_type": "Component", "confidence": 0.9}}]

## Example 2 — vague text, no confident relations
Text: "Someone checked the system and it seemed okay."

<thinking>
1. "someone" — no identity. "the system" — no name. "it" — unresolvable.
2. No named entities, no extractable relations.
</thinking>
[]

# Output Schema (JSON array ONLY, after the thinking block — no other text)
[{{
  "subject": "entity name",
  "subject_type": "EntityType",
  "relation": "RELATION_TYPE",
  "object": "entity name",
  "object_type": "EntityType",
  "confidence": 0.0-1.0
}}]

# Constraints
- ONLY extract relations where BOTH subject AND object are named, specific entities.
- Use canonical the facility naming conventions for all entities.
- Each triplet MUST use exactly one relation type from the taxonomy.
- NEVER fabricate entities or relations not present in the text.
- Return an empty array [] if no confident relations qualify.
"""


class RelationExtractor:
    """
    Extracts typed relations between entities using gpt-4o.
    Smaller batches (5 docs) with retry logic.
    """

    def __init__(self, llm, batch_size: int = 5, max_retries: int = 5, max_workers: int = 10):
        self.llm = llm
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.max_workers = max_workers

    def extract_from_text(
        self, text: str, doc_id: str, collection: str
    ) -> List[KGTriplet]:
        """Extract relations from a single document text."""
        truncated = text[:4000]

        messages = [
            SystemMessage(content=RELATION_SYSTEM_PROMPT),
            HumanMessage(content=f"Extract relations from:\n\n{truncated}"),
        ]

        raw = self._call_with_retry(messages)
        return self._parse_triplets(raw, doc_id, collection)

    def _extract_one(self, doc: Dict[str, Any]) -> List[KGTriplet]:
        """Extract relations from a single document (thread-safe)."""
        return self.extract_from_text(
            doc.get("content", "") or doc.get("page_content", ""),
            doc.get("id", ""),
            doc.get("collection", ""),
        )

    def extract_from_batch(
        self, documents: List[Dict[str, Any]]
    ) -> List[KGTriplet]:
        """Extract relations from documents in parallel using a thread pool.

        Documents are submitted in chunks of batch_size for progress logging.
        Within each chunk, up to max_workers concurrent LLM calls are made,
        reducing wall-clock time by ~max_workers×.
        """
        all_triplets: List[KGTriplet] = []
        total_batches = (len(documents) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i: i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.info(
                "Relation extraction batch %d/%d (%d docs, %d workers)",
                batch_num, total_batches, len(batch), self.max_workers,
            )

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for idx, doc in enumerate(batch):
                    if idx > 0:
                        time.sleep(1.0)
                    futures[executor.submit(self._extract_one, doc)] = doc
                for future in as_completed(futures):
                    doc = futures[future]
                    try:
                        triplets = future.result()
                        all_triplets.extend(triplets)
                    except Exception as e:
                        logger.warning(
                            "Relation extraction failed for doc %s: %s", doc.get("id"), e
                        )

            # Brief pause between macro-batches to avoid rate-limit bursts
            if i + self.batch_size < len(documents):
                time.sleep(1.0)

        logger.info(
            "Extracted %d triplets from %d documents", len(all_triplets), len(documents)
        )
        return all_triplets

    def _call_with_retry(self, messages: list) -> str:
        """Call LLM with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(messages)
                return response.content
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}. Retry in {wait}s")
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
        raise RuntimeError(f"Relation extraction LLM failed after {self.max_retries} retries")

    def _parse_triplets(
        self, raw: str, doc_id: str, collection: str
    ) -> List[KGTriplet]:
        """Parse LLM JSON output into KGTriplet objects."""
        import re as _re
        # Strip <thinking> CoT block if present
        raw = _re.sub(r"<thinking>.*?</thinking>", "", raw, flags=_re.DOTALL)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []

        try:
            items = json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for {doc_id}: {e}")
            return []

        triplets: List[KGTriplet] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            confidence = float(item.get("confidence", 1.0))
            if confidence < 0.5:
                continue

            rel_str = item.get("relation", "").upper().strip()
            edge_type = EDGE_TYPE_MAP.get(rel_str.lower(), EdgeType.RELATED_TO)

            subj_type = NODE_TYPE_MAP.get(
                item.get("subject_type", "").lower(), NodeType.KEYWORD
            )
            obj_type = NODE_TYPE_MAP.get(
                item.get("object_type", "").lower(), NodeType.KEYWORD
            )

            triplets.append(
                KGTriplet(
                    subject=item.get("subject", "").strip(),
                    subject_type=subj_type,
                    relation=edge_type,
                    object=item.get("object", "").strip(),
                    object_type=obj_type,
                    confidence=confidence,
                    source_doc_id=doc_id,
                    source_collection=collection,
                )
            )

        return triplets
