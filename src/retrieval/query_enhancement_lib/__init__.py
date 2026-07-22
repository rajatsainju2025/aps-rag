"""
Query-enhancement library (vendored, self-contained).

These modules were originally shared with a sibling application; they are
vendored here so the public release has no external-repo dependency. Each is
importable on its own:

  - temporal_processor.TemporalProcessor   — inline temporal expansion
  - temporal_date_parser.TemporalDateParser — LLM-based structured date ranges
  - acronyms.AcronymResolver               — acronym/abbreviation expansion
  - synonym_manager.SynonymManager         — synonym expansion
  - query_understanding.QueryIntent        — query-intent enum/classifier

Domain vocabulary (acronyms/synonyms) ships as EMPTY placeholder templates in
`data/` — see those files for the schema. With empty dictionaries the resolvers
run in harmless pass-through mode.
"""
