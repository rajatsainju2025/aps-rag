# Agent skills for APS-RAG (skeletons)

Skeleton "skills" that let coding agents query a running APS-RAG deployment as a
first-class command. Both are **starting points** — edit the `CONFIGURE` values
(API URL, checkout path) for your setup.

Both call the same backend as [`scripts/query.py`](../scripts/query.py) / the
FastAPI `/query` endpoint, so APS-RAG must be set up and data ingested first
(see the main [README](../README.md)).

## Claude Code

- File: [`.claude/skills/aps-rag/SKILL.md`](../.claude/skills/aps-rag/SKILL.md)
- Auto-discovered when the repo is opened in Claude Code (skills live in
  `.claude/skills/<name>/SKILL.md`). Invoke with:

  ```
  /aps-rag <your question>
  ```

## Codex

- File: [`.codex/prompts/aps-rag.md`](../.codex/prompts/aps-rag.md)
- Codex reads custom prompts from `~/.codex/prompts/`. Install with:

  ```bash
  mkdir -p ~/.codex/prompts
  cp .codex/prompts/aps-rag.md ~/.codex/prompts/
  ```

  Then invoke with:

  ```
  /aps-rag <your question>
  ```

## Customize

- `APS_RAG_API_URL` — base URL of your running API (default `http://localhost:8000`).
- Default `collection` (`all_databases`), `service_mode` (`agentic_rag`), and
  `retrieval_strategy` (`HYBRID_3WAY`) — adjust to taste.
- For a no-LLM connectivity check, both skeletons document
  `python scripts/query.py --retrieve-only "<question>"`.
