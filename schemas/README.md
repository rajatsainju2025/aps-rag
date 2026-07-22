# Per-database schemas (optional)

A **schema** lets you declare which metadata fields to extract from a database's
records and attach to every chunk. Schemas are **optional** — with none, the
generic fallback still captures useful metadata automatically (see
[Fallback behavior](#fallback-behavior-no-schema)).

## Where to put a schema (resolution order)

For a database named `<db>` (i.e. the folder `data/raw/<db>/`), the loader looks
for a schema in this order and uses the **first** it finds:

1. `schemas/<db>.yaml`   ← **primary convention (recommended)**
2. `schemas/<db>.yml`
3. `schemas/<db>.json`
4. `data/raw/<db>/schema.yaml`
5. `data/raw/<db>/schema.yml`
6. `data/raw/<db>/schema.json`

Keeping schemas in `schemas/` (options 1–3) keeps them under version control and
separate from your (gitignored) documents.

## Annotated example

```yaml
database: my_database          # optional — documentation only
description: "One-line summary of this corpus."   # optional

fields:
  # <field_name>:  the metadata key that will be stored on each chunk
  title:
    source: headline           # OPTIONAL: the raw key to read (default: the field name)
    type: string               # OPTIONAL: see supported types below
    description: "Document title"   # OPTIONAL: human note
    default: ""                # OPTIONAL: value used when the raw key is absent

  author:
    type: string

  published:
    source: creation_datetime
    type: datetime

  tags:
    type: list
```

A copy-pasteable, working example for the bundled dataset is in
[`example_database.yaml`](example_database.yaml).

## Supported field types

| `type`               | Meaning                                  |
|----------------------|------------------------------------------|
| `string` / `str`     | Free text                                |
| `integer` / `int`    | Whole number                             |
| `float` / `number`   | Decimal number                           |
| `date` / `datetime`  | ISO-8601 date or timestamp               |
| `list` / `array`     | List of values                           |
| `bool` / `boolean`   | True / false                             |

Types are advisory metadata for now (values are stored as-is); declaring them
documents intent and makes the schema self-describing. Unknown types are
rejected at load time.

## What happens with a schema

- Each declared field is read from every record (using `source:` if given, else
  the field name), with `default:` filling in when absent.
- The extracted fields are stored as **structured metadata on every chunk**
  (queryable/filterable downstream).
- The **existing chunking strategy is used unchanged** — a schema never changes
  how documents are split.

## Fallback behavior (no schema)

If no schema is found for a database, ingestion uses the **generic fallback**:

- The **existing chunker** (default: 3-stage granular chunking) is used verbatim.
- Metadata is auto-captured: `source_database` (the folder name), the source
  filename and path, and any obvious fields present in the record (`title`,
  `author`, date-like fields, …).
- That metadata is **embedded into each chunk's text** *and* stored as chunk
  metadata, so retrieval stays metadata-aware even without a schema.

## Validation

A schema that is present but malformed causes ingestion to **fail fast** with a
message naming the file and the problem (e.g. a field spec that isn't a mapping,
or an unknown `type`). Fix the file and re-run.
