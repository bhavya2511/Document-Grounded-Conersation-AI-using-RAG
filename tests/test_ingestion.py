"""
Basic ingestion tests — run with: pytest tests/test_ingestion.py -v
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_pdf_extractor_import():
    from core.ingestion.pdf_extractor import extract_pdf
    assert callable(extract_pdf)


def test_chunker_import():
    from core.ingestion.chunker import chunk_pages
    assert callable(chunk_pages)


def test_indexer_import():
    from core.ingestion.indexer import index_chunks, list_indexed_documents
    assert callable(index_chunks)
    assert callable(list_indexed_documents)


def test_chunk_metadata_fields():
    from core.ingestion.chunker import _make_chunk
    chunk = _make_chunk(
        text="Total Income H1-26 was 44,281 crore.",
        page_num=2,
        chunk_index=3,
        source_file="test.pdf",
        content_type="text",
        bbox={"x0": 10, "y0": 20, "x1": 200, "y1": 40},
        page_width=960,
        page_height=540,
        extraction_methods=["text"]
    )

    assert chunk["chunk_id"] == "p2:c3"
    assert chunk["page_number"] == 2
    assert chunk["chunk_index"] == 3
    assert "bbox" in chunk
    assert "numbers_present" in chunk
    assert "44,281" in chunk["numbers_present"]
    assert chunk["content_type"] == "text"


def test_paragraph_splitting():
    from core.ingestion.chunker import _split_into_paragraphs

    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    paras = _split_into_paragraphs(text)
    assert len(paras) == 3


def test_table_chunking():
    from core.ingestion.chunker import _chunk_tables

    table_md = """| Metric | H1-25 | H1-26 |
| --- | --- | --- |
| Total Income | 49,263 | 44,281 |
| EBITDA | 8,654 | 7,688 |"""

    chunks = _chunk_tables(
        tables_markdown=table_md,
        page_num=2,
        chunk_start=0,
        source_file="test.pdf",
        page_width=960,
        page_height=540
    )

    assert len(chunks) >= 1
    assert chunks[0]["content_type"] == "table"
    assert "44,281" in chunks[0]["text"]