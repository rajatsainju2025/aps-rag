"""
Unit tests for src.utils.citation_formatter.

Regression context: when KG retrieval is active (HYBRID_3WAY), Neo4j returns
Document nodes whose `id` is a human-readable slug like "logentry_linac/ltp_tune_up"
— not a real the system/WRQ id. graph_retriever puts that slug into metadata["id"]
and citation_formatter would prepend "ENTRY_" to make a broken citation link.
The fix: skip docs with retriever_type startswith "graph_" in the citation panel.
The same docs MUST still appear in format_context_for_llm (the LLM benefits
from KG-derived facts in context).
"""
import logging
import sys
from pathlib import Path

from langchain_core.documents import Document

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.citation_formatter import (
    build_source_map,
    format_context_for_llm,
    replace_source_markers,
)


def _vec_doc(doc_id: str = "10749") -> Document:
    return Document(
        page_content="vector-source content",
        metadata={
            "id": f"{doc_id}__chunk_0",
            "doc_id": doc_id,
            "collection": "entry_logbook",
            "retriever_type": "vector",
            "title": f"the system {doc_id}",
            "creation_datetime": "2026-05-25T11:17:04",
            "author": "operator",
        },
    )


def _bm25_doc(doc_id: str = "10750") -> Document:
    return Document(
        page_content="bm25-source content",
        metadata={
            "id": f"{doc_id}__chunk_0",
            "doc_id": doc_id,
            "collection": "entry_logbook",
            "retriever_type": "bm25",
            "title": f"the system {doc_id}",
            "creation_datetime": "2026-05-25T23:17:03",
            "author": "operator",
        },
    )


def _graph_entity_doc(slug: str = "logentry_linac/ltp_tune_up") -> Document:
    return Document(
        page_content="graph-derived facts",
        metadata={
            "id": slug,                      # Neo4j-slug — not a real id
            "retriever_type": "graph_entity",
            "collection": "entry_logbook",
            "kg_entity": "linac",
        },
    )


def _graph_path_doc(slug: str = "workrequest_downtime_#108306") -> Document:
    return Document(
        page_content="path-derived",
        metadata={
            "id": slug,
            "retriever_type": "graph_path",
            "collection": "work_requests",
            "path_node": "S36",
        },
    )


class TestCitationFiltersGraphDocs:
    def test_build_source_map_excludes_graph_entity_docs(self):
        docs = [_vec_doc(), _bm25_doc(), _graph_entity_doc()]
        citations, source_map = build_source_map(docs)
        # Only vector + BM25 should appear; graph_entity dropped.
        display_ids = {c["display_id"] for c in citations}
        assert "ENTRY_10749" in display_ids
        assert "ENTRY_10750" in display_ids
        # The slug must NOT appear in any display_id.
        for did in display_ids:
            assert "logentry_linac" not in did, (
                f"Graph slug leaked into citation panel: {did}"
            )
        assert len(citations) == 2

    def test_build_source_map_excludes_all_graph_subtypes(self):
        """graph_entity, graph_path, graph_community must all be filtered."""
        docs = [
            _vec_doc("10749"),
            _graph_entity_doc("logentry_a"),
            _graph_path_doc("workrequest_b"),
            Document(
                page_content="comm",
                metadata={
                    "id": "icmsdocument_c",
                    "retriever_type": "graph_community",
                    "collection": "icms_collection",
                },
            ),
        ]
        citations, _ = build_source_map(docs)
        assert len(citations) == 1
        assert citations[0]["display_id"] == "ENTRY_10749"

    def test_format_context_for_llm_KEEPS_graph_docs(self):
        """Filter applies only to the citation panel — the LLM context still
        receives graph docs so it can reason over KG-derived facts."""
        docs = [_vec_doc(), _graph_entity_doc()]
        ctx = format_context_for_llm(docs)
        assert "vector-source content" in ctx
        assert "graph-derived facts" in ctx, (
            "Graph docs were stripped from LLM context — they should still be "
            "passed to the LLM, just not shown in the user-facing source panel."
        )

    def test_no_citation_drop_when_all_docs_are_real_sources(self):
        """Sanity: pre-existing behaviour preserved when no graph docs present."""
        docs = [_vec_doc("10749"), _bm25_doc("10750")]
        citations, source_map = build_source_map(docs)
        assert len(citations) == 2
        assert len(source_map) == 2

    def test_all_graph_docs_yield_empty_citation_list(self):
        """If retrieval returns only graph docs, the source panel is empty
        (rather than full of broken links)."""
        docs = [_graph_entity_doc("a"), _graph_path_doc("b")]
        citations, source_map = build_source_map(docs)
        assert citations == []
        assert source_map == {}


class TestCitationMarkerIdempotency:
    """Regression tests for the double-URL bug: when the LLM emits
    `[WRQ_62347](url)` directly (already a markdown link), Pass 2b must NOT
    append a second URL — the output `[WRQ_62347](citation_url)(llm_url)`
    rendered as a link followed by literal text "(llm_url)", producing the
    broken `WRQ_62347(url)` user-visible output reported on 2026-05-26.
    """

    def _wrq_citation(self, num: str = "62347"):
        url = f"https://your-rag-host.example.com/api/entries/work_requests/{num}.html"
        return {
            "id": num,
            "display_id": f"WRQ_{num}",
            "url": url,
            "collection": "work_requests",
        }

    def _entry_citation(self, num: str = "10739"):
        url = f"https://your-host.example.com/entry/views/itemDomainLogbook/view?id={num}"
        return {
            "id": num,
            "display_id": f"ENTRY_{num}",
            "url": url,
            "collection": "entry_logbook",
        }

    def test_already_linked_wrq_is_left_alone(self):
        """LLM-emitted `[WRQ_62347](url)` must NOT get a second URL appended."""
        citations = [self._wrq_citation("62347")]
        url = "https://your-rag-host.example.com/api/entries/work_requests/62347.html"
        answer = f"Maintenance happened. [WRQ_62347]({url})."
        out = replace_source_markers(answer, source_map={}, citations=citations)
        # The link must appear exactly once — no `(url)(url)` double-append.
        assert out.count(url) == 1, (
            f"Pass 2b double-appended URL — got: {out!r}"
        )
        assert "[WRQ_62347](" in out
        assert ")(https" not in out, (
            f"Found `)(https` double-paren — Pass 2b idempotency broken: {out!r}"
        )

    def test_already_linked_entry_short_form_is_left_alone(self):
        """LLM-emitted `[the system-10739](url)` (hyphen variant) must be idempotent."""
        citations = [self._entry_citation("10739")]
        url = "https://your-host.example.com/entry/views/itemDomainLogbook/view?id=10739"
        answer = f"Yesterday's run. [the system-10739]({url})."
        out = replace_source_markers(answer, source_map={}, citations=citations)
        assert out.count(url) == 1, f"Pass 2 double-appended: {out!r}"
        assert ")(https" not in out

    def test_bare_wrq_still_gets_substituted(self):
        """Sanity: the LLM emitting bare `[WRQ_62347]` (no URL) must STILL get
        a URL appended — the fix only skips already-linked forms."""
        citations = [self._wrq_citation("62347")]
        answer = "Maintenance happened. [WRQ_62347]."
        out = replace_source_markers(answer, source_map={}, citations=citations)
        assert "[WRQ_62347](https://your-rag-host.example.com/api/entries/work_requests/62347.html)" in out

    def test_bare_entry_short_form_still_gets_substituted(self):
        """Sanity: `[the system-10739]` (no URL) must still resolve."""
        citations = [self._entry_citation("10739")]
        answer = "Yesterday's run. [the system-10739]."
        out = replace_source_markers(answer, source_map={}, citations=citations)
        assert "[ENTRY_10739](" in out  # display_id uses underscore form
        assert "view?id=10739)" in out

    def test_mixed_linked_and_bare_in_same_answer(self):
        """The real-world case: same answer has both `[WRQ_62347](url)` and
        a separate `[WRQ_62354]`; both should end up properly linked exactly
        once."""
        cits = [self._wrq_citation("62347"), self._wrq_citation("62354")]
        url_47 = "https://your-rag-host.example.com/api/entries/work_requests/62347.html"
        answer = f"One: [WRQ_62347]({url_47}). Two: [WRQ_62354]."
        out = replace_source_markers(answer, source_map={}, citations=cits)
        assert out.count(url_47) == 1
        assert "[WRQ_62354](https://your-rag-host.example.com/api/entries/work_requests/62354.html)" in out
        assert ")(https" not in out

    # ── Pass 2.5: freelance [ID](url) repair ──────────────────────────
    # The freelance class observed in naive_rag EIC BAR runs: the synthesis
    # LLM emits `[ENTRY_8975](url)` directly — already a markdown link, so
    # Passes 2/2b/3 all skip it via (?!\(). Pass 2.5 catches these.

    def _icms_citation(self, file_id: str = "DOC_1192906"):
        url = (
            "https://your-docs-host.example.com/docs/idcplg"
            f"?IdcService=DISPLAY_URL&dDocName={file_id}"
        )
        return {
            "id": file_id,
            "display_id": file_id,
            "url": url,
            "collection": "icms_collection",
        }

    def test_freelance_linked_with_known_id_gets_canonical_url(self):
        """LLM emits `[ENTRY_8975](wrong-url)` for a known ID → Pass 2.5
        swaps the wrong URL for the canonical citation URL."""
        citations = [self._entry_citation("8975")]
        wrong = "https://your-host.example.com/some/wrong/path?id=8975"
        canonical = "https://your-host.example.com/entry/views/itemDomainLogbook/view?id=8975"
        answer = f"Spin tracking [ENTRY_8975]({wrong})."
        out = replace_source_markers(answer, source_map={}, citations=citations)
        assert canonical in out
        assert wrong not in out, f"wrong URL leaked through: {out!r}"

    def test_freelance_linked_with_unknown_id_gets_warn_glyph(self, caplog):
        """LLM emits `[ENTRY_99999](url)` for an ID not in the citation panel
        → Pass 2.5 prepends ⚠ glyph and logs an unknown-id warning."""
        citations = [self._entry_citation("8975")]  # 99999 deliberately absent
        leaked = "https://your-host.example.com/some/url?id=99999"
        answer = f"Mystery doc [ENTRY_99999]({leaked})."
        with caplog.at_level(logging.WARNING, logger="src.utils.citation_formatter"):
            out = replace_source_markers(answer, source_map={}, citations=citations)
        assert f"[⚠ ENTRY_99999]({leaked})" in out
        assert any(
            "Pass 2.5 unknown-id" in rec.getMessage() and "ENTRY_99999" in rec.getMessage()
            for rec in caplog.records
        ), f"unknown-id warning not logged; records: {[r.getMessage() for r in caplog.records]}"

    def test_freelance_linked_idempotent_when_url_already_canonical(self, caplog):
        """Input URL == canonical → output equals input, no wrong-url warning."""
        citations = [self._entry_citation("8975")]
        canonical = "https://your-host.example.com/entry/views/itemDomainLogbook/view?id=8975"
        answer = f"Spin tracking [ENTRY_8975]({canonical})."
        with caplog.at_level(logging.WARNING, logger="src.utils.citation_formatter"):
            out = replace_source_markers(answer, source_map={}, citations=citations)
        assert out == answer
        assert not any(
            "Pass 2.5 wrong-url" in rec.getMessage() for rec in caplog.records
        ), "wrong-url warning fired on a canonical URL"

    def test_freelance_aps_with_icms_url_gets_repaired(self):
        """The form actually observed in EIC BAR runs:
        `[DOC_1192906](https://your-docs-host.example.com/.../dDocName=DOC_1192906)`.
        With known ID and already-canonical URL → idempotent no-op."""
        citations = [self._icms_citation("DOC_1192906")]
        canonical = (
            "https://your-docs-host.example.com/docs/idcplg"
            "?IdcService=DISPLAY_URL&dDocName=DOC_1192906"
        )
        answer = f"See [DOC_1192906]({canonical})."
        out = replace_source_markers(answer, source_map={}, citations=citations)
        assert out == answer

    def test_freelance_linked_form_does_not_double_url(self):
        """Regression: re-running replace_source_markers on its own output must
        not produce `)(https` double-URLs (defends 2026-05-26 bug)."""
        citations = [self._entry_citation("8975")]
        canonical = "https://your-host.example.com/entry/views/itemDomainLogbook/view?id=8975"
        answer = f"Spin tracking [ENTRY_8975]({canonical})."
        out1 = replace_source_markers(answer, source_map={}, citations=citations)
        out2 = replace_source_markers(out1, source_map={}, citations=citations)
        assert ")(https" not in out2, f"double-URL after second pass: {out2!r}"
        assert out2 == out1


class TestChunkBodyUrlSanitization:
    """format_context_for_llm must strip ICMS / the system URLs from chunk BODY text
    before the LLM sees it. ICMS PDF bodies contain bibliographies with full
    `https://your-docs-host.example.com/...` URLs that the LLM was copying verbatim
    into freelance `[ID](url)` citations — root cause of the naive_rag
    citation freelance bug confirmed in EIC BAR runs.
    """

    def _icms_doc(self, body: str) -> Document:
        return Document(
            page_content=body,
            metadata={
                "id": "DOC_1192906__chunk_0",
                "doc_id": "DOC_1192906",
                "collection": "icms_collection",
                "retriever_type": "vector",
                "title": "EIC BAR document",
                "creation_datetime": "2026-05-25T11:17:04",
                "author": "operator",
            },
        )

    def test_format_context_for_llm_strips_icms_urls_from_body(self):
        """ICMS bibliographic URL in chunk body → replaced with [ref]."""
        body = (
            "Discussion of spin tracking. See "
            "https://your-docs-host.example.com/docs/idcplg?IdcService=DISPLAY_URL&dDocName=DOC_2208258 "
            "for prior work."
        )
        ctx = format_context_for_llm([self._icms_doc(body)])
        assert "your-docs-host.example.com" not in ctx, (
            f"ICMS URL leaked into LLM context: {ctx!r}"
        )
        assert "[ref]" in ctx

    def test_format_context_for_llm_strips_entry_urls_from_body(self):
        """the system URL in chunk body → replaced with [ref]."""
        body = (
            "Earlier entry at "
            "https://your-host.example.com/entry/views/itemDomainLogbook/view?id=8975 "
            "covered the same scope."
        )
        doc = Document(
            page_content=body,
            metadata={
                "id": "10749__chunk_0",
                "doc_id": "10749",
                "collection": "entry_logbook",
                "retriever_type": "vector",
                "creation_datetime": "2026-05-25T11:17:04",
                "author": "operator",
            },
        )
        ctx = format_context_for_llm([doc])
        assert "your-host.example.com" not in ctx
        assert "[ref]" in ctx

    def test_format_context_for_llm_keeps_bare_ids_in_body(self):
        """Bare `DOC_xxx` IDs in body are legitimate content references (e.g.
        'see related DOC_2208258') and must be preserved — only URLs go."""
        body = "See related DOC_2208258 and ENTRY_9254 for prior work."
        ctx = format_context_for_llm([self._icms_doc(body)])
        assert "DOC_2208258" in ctx
        assert "ENTRY_9254" in ctx

    def test_format_context_for_llm_keeps_other_internal_urls(self):
        """Sanitizer is scoped to icmsdocs + entry only — other internal URLs
        (svn.example, osprey.example, etc.) must pass through untouched."""
        body = (
            "OAG source at https://svn.example.com/repo/apps/foo.c "
            "and dashboard https://osprey.example.com/dashboards/123 ."
        )
        ctx = format_context_for_llm([self._icms_doc(body)])
        assert "svn.example.com" in ctx
        assert "osprey.example.com" in ctx


class TestSourceNumberingAlignment:
    """Regression: format_context_for_llm source_N numbers must match
    build_source_map's compact source_N keys for every doc in the list."""

    def test_graph_doc_does_not_offset_subsequent_source_numbers(self):
        """A graph doc between two vector docs must not shift source numbers.

        Bug: format_context_for_llm gave graph docs [Source N] labels even
        though build_source_map skips them.  With docs=[vec_A, graph, vec_B],
        the LLM saw [Source 1], [Source 2], [Source 3] but source_map only had
        source_1→A, source_2→B — so [source_3] for vec_B was unreplaced.
        """
        vec_a = _vec_doc("10749")
        graph = _graph_entity_doc()
        vec_b = _vec_doc("10750")
        docs = [vec_a, graph, vec_b]

        ctx = format_context_for_llm(docs)
        _, source_map = build_source_map(docs)

        # Context must have Source 1 and Source 2 (compact), not Source 3.
        assert "[Source 1 " in ctx
        assert "[Source 2 " in ctx
        assert "[Source 3 " not in ctx

        # Graph content still present, but as an unnumbered label.
        assert "graph-derived facts" in ctx
        assert "[Context (KG fact)]" in ctx

        # source_map must have source_1 and source_2.
        assert "source_1" in source_map
        assert "source_2" in source_map
        assert "source_3" not in source_map

        # The LLM-emitted [source_2] should resolve to vec_b (ENTRY_10750).
        answer = "Spin tracking [source_1]. Applications [source_2]."
        out = replace_source_markers(answer, source_map)
        assert "ENTRY_10749" in out
        assert "ENTRY_10750" in out
        # Ensure the unreplaced marker bug is gone.
        assert "[source_2]" not in out

    def test_duplicate_does_not_gap_source_numbers(self):
        """Duplicate docs must not create gaps in source numbering.

        Bug: format_context_for_llm skipped the duplicate but used the raw
        enumerate index, so doc C after a duplicate of B got [Source 4] while
        build_source_map assigned it source_3.
        """
        doc_a = _vec_doc("10749")
        doc_b = _vec_doc("10750")
        doc_b_dup = _vec_doc("10750")  # same doc_id → dedup key matches
        doc_c = _vec_doc("10751")
        docs = [doc_a, doc_b, doc_b_dup, doc_c]

        ctx = format_context_for_llm(docs)
        _, source_map = build_source_map(docs)

        # Compact: 3 unique docs → Source 1, 2, 3 with no gaps.
        assert "[Source 1 " in ctx
        assert "[Source 2 " in ctx
        assert "[Source 3 " in ctx
        assert "[Source 4 " not in ctx

        assert set(source_map.keys()) == {"source_1", "source_2", "source_3"}

        # doc_c must be source_3 in both context and map.
        answer = "Result C [source_3]."
        out = replace_source_markers(answer, source_map)
        assert "ENTRY_10751" in out
        assert "[source_3]" not in out
