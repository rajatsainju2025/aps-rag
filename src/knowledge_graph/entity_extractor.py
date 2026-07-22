"""
Entity extractor — uses gemini-1.5-flash (fast) for batched entity extraction.
Extracts node-type entities from scientific text.
Rate-limited batches with exponential backoff (mirrors generate_700_qa_pairs.ipynb pattern).
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage

from .schema import GraphNode, NodeType, NODE_TYPE_MAP

logger = logging.getLogger(__name__)

ENTITY_SYSTEM_PROMPT = """# Role & Persona
You are an **Expert Knowledge Graph Entity Extractor** for accelerator physics \
at your organization's your facility. You build a \
knowledge graph that enables fault diagnosis, expert routing, and multi-hop \
retrieval — so entity quality directly determines system usefulness.

# Task
Extract named entities from the given scientific text and classify them into \
the predefined type taxonomy. Precision over recall: it is better to miss \
a borderline entity than to add a noisy one.

# Entity Type Taxonomy
- Person: operators, engineers, scientists. Use usernames (Doe, user002) or \
  last names. NOT generic roles like "the operator" or "user".
- System: accelerator systems and subsystems (SR, SR RF, SR Booster, SR BPM, \
  Linac, PAR, the facility, 7-BM, etc.). NOT "the system" or "the machine".
- Fault: equipment failures, beam losses, vacuum events, trips, interlocks, \
  misfires. Include the fault description, not just "fault" or "problem".
- Component: hardware devices with specific identifiers (BPM, IK4, PSU-23, \
  AMC543, undulator U23, kicker, septum, RF cavity, etc.).
- Procedure: named operational processes (injection optimization, orbit \
  correction, tune correction, beam alignment, shutdown procedure).
- Concept: technical concepts with specific meaning (injection efficiency, \
  emittance, chromaticity, beam lifetime, Touschek lifetime, tune).

# Coreference Resolution
Resolve pronouns and generic references to their specific referents when clear:
- "it", "the system", "the device" → resolve to named entity if clear from context
- "the issue", "the problem", "this fault" → resolve to specific fault description
- If ambiguous, do NOT extract the pronoun/generic term

# Chain-of-Thought Extraction Protocol (MANDATORY)
Reason inside a <thinking> block before producing JSON:

<thinking>
1. Scan the text for proper nouns, technical terms, and domain-specific jargon.
2. For each candidate entity: (a) What type is it? (b) Is it specific enough for \
   a KG node? (c) Resolve any coreferences to the canonical name.
3. FILTER OUT: generic terms ("the system", "the beam", "the issue"), \
   standalone pronouns, vague descriptions without identifiable names.
4. Assign confidence: 0.9+ for explicitly named entities, 0.7-0.9 for strongly \
   implied by context, 0.5-0.7 for inferred. Do not include < 0.5.
</thinking>

# Few-Shot Examples

## Example 1
Text: "Doe noticed that IK4 misfired during injection at 08:45. The kicker \
charging circuit PSU-23 showed a voltage drop. This caused beam loss in the SR."

<thinking>
1. Entities: Doe (Person), IK4 (Component), injection (Procedure), PSU-23 \
   (Component), beam loss (Fault), SR (System).
2. All are specific and named. "The kicker" resolves to IK4. "voltage drop" is a \
   symptom, not a standalone entity.
3. No generic terms to filter.
4. All high-confidence.
</thinking>
[{{"name": "Doe", "type": "Person", "confidence": 0.95}},
 {{"name": "IK4", "type": "Component", "confidence": 0.99}},
 {{"name": "injection", "type": "Procedure", "confidence": 0.85}},
 {{"name": "PSU-23", "type": "Component", "confidence": 0.99}},
 {{"name": "beam loss during injection", "type": "Fault", "confidence": 0.95}},
 {{"name": "SR", "type": "System", "confidence": 0.99}}]

## Example 2 — vague text
Text: "There was a problem with the system today. Someone checked it and fixed it."

<thinking>
1. "problem" — too vague, no name. "the system" — generic, no name. "someone" — \
   no identity. "it" — unresolvable pronoun.
2. None are specific enough for a KG node.
3. Filter everything.
4. Return empty array.
</thinking>
[]

# Output Schema (JSON array ONLY, after the thinking block — no other text)
[{{"name": "entity name", "type": "Person|System|Fault|Component|Procedure|Concept", "confidence": 0.0-1.0}}]

# Constraints
- ONLY extract named, specific entities. Skip pronouns, generic terms, and \
  uncertain references.
- Use canonical the facility naming conventions (SR not Storage Ring; BPM not beam position monitor).
- Confidence < 0.5 entities will be discarded — only include entities you are sure about.
- Return an empty array [] if no entities qualify.
"""


class EntityExtractor:
    """
    Extracts entities from scientific text using gemini-1.5-flash.
    Batch processing with rate limiting and retry logic.
    """

    def __init__(self, llm, batch_size: int = 20, max_retries: int = 5, max_workers: int = 10):
        self.llm = llm
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.max_workers = max_workers

    def extract_from_text(self, text: str, doc_id: str, collection: str) -> List[GraphNode]:
        """Extract entities from a single document text."""
        truncated = text[:3000]  # Limit context to avoid token overflow

        messages = [
            SystemMessage(content=ENTITY_SYSTEM_PROMPT),
            HumanMessage(content=f"Extract entities from:\n\n{truncated}"),
        ]

        raw = self._call_with_retry(messages)
        return self._parse_entities(raw, doc_id, collection)

    def _extract_one(self, doc: Dict[str, Any]) -> List[GraphNode]:
        """Extract entities from a single document (thread-safe)."""
        return self.extract_from_text(
            doc.get("content", "") or doc.get("page_content", ""),
            doc.get("id", ""),
            doc.get("collection", ""),
        )

    def extract_from_batch(
        self, documents: List[Dict[str, Any]]
    ) -> List[GraphNode]:
        """Extract entities from documents in parallel using a thread pool.

        Documents are submitted in chunks of batch_size to allow progress
        logging. Within each chunk, up to max_workers concurrent LLM calls
        are made, reducing wall-clock time by ~max_workers×.
        """
        all_nodes: List[GraphNode] = []
        total_batches = (len(documents) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i: i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.info(
                "Entity extraction batch %d/%d (%d docs, %d workers)",
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
                        nodes = future.result()
                        all_nodes.extend(nodes)
                    except Exception as e:
                        logger.warning(
                            "Entity extraction failed for doc %s: %s", doc.get("id"), e
                        )

            # Brief pause between macro-batches to avoid rate-limit bursts
            if i + self.batch_size < len(documents):
                time.sleep(0.5)

        logger.info(
            "Extracted %d entity nodes from %d documents", len(all_nodes), len(documents)
        )
        return all_nodes

    def _call_with_retry(self, messages: list) -> str:
        """Call LLM with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(messages)
                return response.content
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}. Retrying in {wait}s")
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries")

    def _parse_entities(
        self, raw: str, doc_id: str, collection: str
    ) -> List[GraphNode]:
        """Parse LLM JSON output into GraphNode objects."""
        import re
        # Strip <thinking> CoT block if present
        raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        # Find JSON array bounds
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            logger.warning(f"No JSON array in entity extraction response for {doc_id}")
            return []

        try:
            items = json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for {doc_id}: {e}")
            return []

        nodes: List[GraphNode] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "").strip()
            type_str = item.get("type", "").lower().strip()
            confidence = float(item.get("confidence", 1.0))

            if not name or confidence < 0.5:
                continue

            node_type = NODE_TYPE_MAP.get(type_str, NodeType.KEYWORD)
            node_id = f"{node_type.value.lower()}_{name.lower().replace(' ', '_')}"

            nodes.append(
                GraphNode(
                    id=node_id,
                    node_type=node_type,
                    name=name,
                    properties={"confidence": confidence},
                    source_collection=collection,
                    source_doc_id=doc_id,
                )
            )

        return nodes
