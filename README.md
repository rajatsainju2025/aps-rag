# APS-RAG — Corrective Agentic Graph-RAG

APS-RAG is a retrieval-augmented generation (RAG) system for question-answering
over your own document collections. It combines:

- **Hybrid retrieval** — dense vectors (Qdrant) + BM25 keyword search
  (Elasticsearch) + an optional **knowledge graph** (Neo4j), fused with adaptive
  Reciprocal Rank Fusion.
- **Reranking** — a cross-encoder reranker with a **graph-proximity** signal that
  boosts documents connected to the query's entities in the knowledge graph.
- **Query enhancement** — temporal parsing, acronym/synonym expansion,
  multi-query generation.
- **Agentic generation** — a corrective loop (self-critique → re-retrieve) with
  cited answers, plus a simple single-pass "naive RAG" mode.
- **Folder-per-database ingestion** — drop documents into `data/raw/<name>/` and
  ingest; no code changes to add a corpus.
- **InPARS QA construction** — generate evaluation questions **with answers and
  nuggets** directly from your documents (`scripts/generate_inpars.py`).

The code is provider-agnostic: point it at **any OpenAI-compatible LLM gateway**
(OpenAI, Azure, vLLM, Ollama, LM Studio, …) and any Hugging Face embedding /
reranker model.

---

## Requirements

- **Python 3.11** recommended (3.10–3.13 supported).
- **Services** (run locally via Docker or point at your own):
  - Qdrant (vector store) — default `localhost:6333`
  - Elasticsearch 8.x (BM25) — default `localhost:9200`
  - Neo4j (knowledge graph) — default `bolt://localhost:7687` — *only needed for
    KG build / graph retrieval*
- **An LLM** reachable via an OpenAI-compatible API (for generation, KG
  extraction, and InPARS).
- **Hardware:** embedding + reranking run locally.
  - GPU (≥ ~6 GB VRAM) strongly recommended for `intfloat/e5-large-v2` +
    `jinaai/jina-reranker-v3`. Set `vector.embedding_device: cpu` to run on CPU
    (slower). Ingestion is the GPU-heavy step; querying is light.

An example `infrastructure/docker-compose.yml` is provided to bring up
Qdrant + Elasticsearch locally.

---

## Quickstart

```bash
# 1. Clone + install
git clone <your-fork-url> aps-rag && cd aps-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download the default models into models/ (idempotent)
python scripts/download_models.py

# 3. Bring up the stores (or point config at your own)
#    (example compose file brings up Qdrant + Elasticsearch)
docker compose -f infrastructure/docker-compose.yml up -d

# 4. Configure
cp config/config.example.yaml config/config.yaml     # edit hosts/models if needed
cp .env.example .env                                  # fill in LLM_API_KEY etc.

# 5. Add data — one folder per database
mkdir -p data/raw/my_docs
cp /path/to/your/*.json data/raw/my_docs/             # or .txt / .md
#    (or try the bundled example:)
cp -r examples/data/raw/example_database data/raw/

# 6. Ingest (chunk → embed → Qdrant + Elasticsearch)
python scripts/ingest.py                              # all databases under data/raw/
#    add --build-kg to also populate the Neo4j knowledge graph

# 7. Query
python scripts/query.py --retrieve-only "your question here"   # local, no LLM
#    or, for a full generated answer, start the API and query it:
bash run_api.sh                                       # serves on :8000
python scripts/query.py "your question here"
```

### First-run smoke test (bundled example)

```bash
cp -r examples/data/raw/example_database data/raw/
python scripts/ingest.py --databases example_database
python scripts/query.py --retrieve-only "how is torque drift on the calibration station fixed?"
```

You should see chunks from `doc_002.json` (the torque-drift record) come back.

<details>
<summary>Verified first-run transcript (bundled example, isolated Qdrant + Elasticsearch)</summary>

```text
$ python scripts/ingest.py --databases example_database
Loading embedding model 'intfloat/e5-large-v2' (device=cuda:0)...
DATABASE: example_database   (strategy=granular-chunking)
Loaded schema for 'example_database' (4 fields)
Loaded 4 docs from database 'example_database'
GranularChunker: example_database → 4 docs → 4 chunks
Created Qdrant collection: agrag_gc_example_database (dim=1024)
ES indexed 4 docs to agrag_gc_example_database
INGEST SUMMARY
database                     docs    vectors   collection
example_database                4          4   agrag_gc_example_database

# Qdrant: points=4, dim=1024, payload carries source_database + schema fields
# Elasticsearch: 4 docs indexed
# Fallback (schema removed): all 4 chunks still carry source_database and
#   embed auto-captured metadata into their content.
```

Both the schema path and the no-schema fallback path were validated end-to-end;
all three stores are populated on ingest (Qdrant + Elasticsearch always; Neo4j
with `--build-kg`).
</details>

---

## The `data/raw/<database>/` convention

Everything you want indexed goes under `data/raw/`, **one folder per source
database**:

```
data/raw/
├── arxiv/          ├── internal_reports/
│   ├── a.json      │   ├── r1.pdf-converted.json
│   └── b.txt       │   └── r2.md
```

- The **folder name is the database id** and is stamped on every chunk as
  `source_database`.
- Ingestion **auto-discovers** every subfolder — adding a database is just a new
  folder. `python scripts/ingest.py --list` shows what will be picked up.
- Supported file types: `.json` (a single record, a list of records, or
  `{"entries": [...]}`), `.txt`, `.md`. Other types are skipped with a log
  message. Your documents are **gitignored**.

See [`data/raw/README.md`](data/raw/README.md).

## Optional per-database schemas (with generic fallback)

You can declare which metadata fields to extract per database by adding
`schemas/<database>.yaml` (or `data/raw/<database>/schema.yaml`). See
[`schemas/README.md`](schemas/README.md) and the worked example
[`schemas/example_database.yaml`](schemas/example_database.yaml).

**No schema? That's fine** — the generic fallback uses the same chunker and
auto-captures metadata (`source_database`, filename, path, obvious fields),
embedding it into each chunk so retrieval stays metadata-aware.

## Pointing at your own LLM gateway and models

Everything is config-driven — no code edits:

- **LLM** (`config/config.yaml` → `llm:` / `.env`): set `base_url`, `LLM_API_KEY`,
  and `model` for any OpenAI-compatible gateway. For local gateways (Ollama, LM
  Studio) leave the key blank. Optional per-role model overrides are supported.
- **Embedding / reranker** (`vector.embedding_model`, `agent.reranker_model_name`):
  a Hugging Face id or a local path. If you change the embedding model, set
  `vector.embedding_dim` to its output dimensionality. `e5-large-v2` is used
  **without** query/passage prefixes by default (matching the original behavior).

> ⚠️ **Reranker license:** the default `jinaai/jina-reranker-v3` is **CC BY-NC 4.0
> (non-commercial)**. For commercial use, set a permissive reranker such as
> `BAAI/bge-reranker-v2-m3` (Apache-2.0).

## InPARS: generate evaluation QA (question + nuggets)

Build a retrieval/RAG evaluation set grounded in *your* corpus. For each sampled
document an LLM produces a question, the expected answer, and the **nuggets** —
the atomic supporting facts the answer depends on (the unit a retrieval evaluator
scores for recall):

```bash
python scripts/generate_inpars.py --database my_docs --num 20 --output qa.jsonl
```

Each output row: `{question, answer, nuggets[], source_doc_id, source_database, …}`.

---

## Project structure

```
aps-rag/
├── scripts/
│   ├── ingest.py            # auto-discover data/raw → chunk → embed → Qdrant + ES (+ --build-kg)
│   ├── query.py             # CLI query (retrieval-only, or full answer via the API)
│   ├── build_kg.py          # build the Neo4j knowledge graph
│   ├── generate_inpars.py   # InPARS QA (question + nuggets) from source docs
│   └── download_models.py   # fetch e5 + reranker into models/
├── src/
│   ├── vectorstore/         # builder, granular chunker, collection registry (folder-per-db)
│   ├── retrieval/           # bm25, vector, graph retrievers; RRF fusion; reranker; query enhancement
│   ├── integrations/        # Qdrant, Elasticsearch, Neo4j clients; LLM factory (OpenAI-compat + Argo)
│   ├── knowledge_graph/     # entity/relation extraction, graph store, KG build
│   ├── services/            # naive RAG + agentic RAG services
│   ├── agents/, tools/      # orchestrator, corrective loop, planner; optional site-specific tools
│   └── utils/               # document loader, schema loader, citation formatter, …
├── api/                     # FastAPI app: /query, /v1/chat/completions, /health, /graph
├── config/                  # settings.py + config.example.yaml + prompts/
├── schemas/                 # optional per-database schemas (+ README + example)
├── data/raw/                # your documents go here (gitignored)
├── models/                  # downloaded weights (gitignored)
└── examples/                # tiny synthetic dataset for first-run validation
```

## Agent skills (Claude Code & Codex)

Skeleton `/aps-rag` skills let coding agents query a running APS-RAG deployment
as a command: [`.claude/skills/aps-rag/SKILL.md`](.claude/skills/aps-rag/SKILL.md)
(Claude Code, auto-discovered) and [`.codex/prompts/aps-rag.md`](.codex/prompts/aps-rag.md)
(Codex, copy to `~/.codex/prompts/`). Customize per [`docs/skills.md`](docs/skills.md).

## Chunking (default, preserved)

`granular-chunking` (default): a 3-stage recursive splitter — pre-split at
`pre_chunk_size=1000` / overlap `200`, then atomic split at `max_chunk_size=4000`
/ overlap `200` (`config.chunking`). A `whole-doc` strategy (1 vector per record)
is also available via `--strategy whole-doc`.

## Optional / site-specific features

The agentic layer includes tools that are **disabled by default** because they
require site-specific infrastructure: EPICS control-system (PV) reads, an
archiver client, a channel finder, web search, and code execution. Enable them in
config (`agent.enable_*`, `pv.enable_pv_tools`) and set the relevant hosts. The
core RAG (retrieval + generation) and KG + InPARS work without them.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot reach Qdrant / Elasticsearch` | Start the stores (`docker compose -f infrastructure/docker-compose.yml up -d`) or fix `vector.*` in your config. |
| `LLM_API_KEY is not set` | Copy `.env.example` → `.env` and fill it in (local gateways may leave it blank). |
| Reranker loads but scores are all 0 | The Jina reranker needs `transformers` + `trust_remote_code`; ensure `pip install -r requirements.txt` succeeded. |
| CUDA out of memory / no GPU | Set `vector.embedding_device: cpu` (slower). |
| `scripts/query.py` (full mode) can't connect | Start the API first (`bash run_api.sh`), or use `--retrieve-only`. |
| Embedding dim mismatch on ingest | If you changed `embedding_model`, set `vector.embedding_dim` to match; the collection is recreated automatically on dim change. |

---

## Citing

See [`CITATION.cff`](CITATION.cff) (fill in authors/DOI before publishing). The
InPARS methodology follows Bonifacio et al., *InPars* (SIGIR 2022).

## License

Code: **MIT** (see [`LICENSE`](LICENSE)). Note the **non-commercial** default
reranker model license described in `LICENSE` and above.
