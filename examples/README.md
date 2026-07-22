# Examples

A tiny, fully synthetic dataset for validating a first-run end to end. It
contains no private data — just a fictional "Widget Assembly Line".

```
examples/
└── data/raw/example_database/
    ├── doc_001.json        # structured record (title/author/date/system/content)
    ├── doc_002.json        # a "fault + resolution" record
    ├── intake_notes.txt    # plain-text note
    └── packaging_guide.md  # markdown guide
```

A matching optional schema lives at
[`../schemas/example_database.yaml`](../schemas/example_database.yaml).

## Use it

Copy the example database into your `data/raw/` and ingest:

```bash
cp -r examples/data/raw/example_database data/raw/
python scripts/ingest.py --databases example_database
python scripts/query.py "How is torque drift on the calibration station fixed?"
```

### Exercise both metadata paths

- **With a schema** (default here): `schemas/example_database.yaml` is present, so
  its declared fields are extracted as chunk metadata.
- **Fallback (no schema):** rename the schema aside and re-ingest to see the
  generic fallback embed auto-captured metadata into each chunk:

  ```bash
  mv schemas/example_database.yaml schemas/example_database.yaml.off
  python scripts/ingest.py --databases example_database
  mv schemas/example_database.yaml.off schemas/example_database.yaml
  ```
