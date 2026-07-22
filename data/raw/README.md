# `data/raw/` — drop your documents here (one folder per database)

This is where you put the documents you want the RAG system to index. The
convention is **one subfolder per source database**:

```
data/raw/
├── <database_name_A>/        # the folder name IS the database id
│   ├── doc1.json
│   ├── doc2.txt
│   └── notes.md
├── <database_name_B>/
│   └── ...
```

## Rules

- **The folder name is the database identifier.** It is stamped onto every
  chunk as `source_database` metadata, so retrieval and citations can attribute
  a chunk back to its origin.
- **Auto-discovery.** `python scripts/ingest.py` finds every subfolder here and
  ingests each as its own database. **Adding a database = create a new folder and
  drop files in — no code changes.** (`python scripts/ingest.py --list` shows what
  will be picked up.)
- **Supported file types** (mixed freely within a folder):
  - `.json` — a single record `{...}`, a list of records `[{...}, ...]`, or a
    logbook-style `{"entries": [{...}, ...]}`.
  - `.txt`, `.md` / `.markdown`, `.rst` — one record per file (the file text is
    the content, the filename becomes the title).
  - Any other extension is **skipped with a log message** (never a crash).
- **Optional schema.** To control which metadata fields are extracted, add
  `schemas/<database_name>.yaml` (preferred) or `data/raw/<database_name>/schema.yaml`.
  See [`../../schemas/README.md`](../../schemas/README.md). With no schema, a
  generic fallback captures `source_database`, filename, path, and any obvious
  fields, and embeds them into each chunk so retrieval stays metadata-aware.

## Your data is never committed

Everything under `data/raw/` (except this README and `.gitkeep`) is **gitignored**
so your documents are not accidentally published. See `.gitignore`.

## Try it with the bundled example

A tiny synthetic dataset ships under [`../../examples/`](../../examples/). Copy it
in and ingest:

```bash
cp -r examples/data/raw/example_database data/raw/
python scripts/ingest.py --databases example_database
```
