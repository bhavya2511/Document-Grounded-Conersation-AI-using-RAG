"""
Retrieval tests — run with: pytest tests/test_retrieval.py -v
Requires: a document already indexed (run ingestion first)
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_reranker_import():
    from core.retrieval.reranker import rerank, get_reranker
    assert callable(rerank)


def test_retriever_import():
    from core.retrieval.retriever import hybrid_search, check_relevance
    assert callable(hybrid_search)
    assert callable(check_relevance)


def test_rrf_fusion():
    from core.retrieval.retriever import _reciprocal_rank_fusion

    dense = [
        {"chunk_id": "p1:c0", "text": "a", "page_number": 1,
         "dense_score": 0.9, "bm25_score": 0.0, "hybrid_score": 0.9,
         "source_file": "test.pdf"},
        {"chunk_id": "p2:c1", "text": "b", "page_number": 2,
         "dense_score": 0.7, "bm25_score": 0.0, "hybrid_score": 0.7,
         "source_file": "test.pdf"},
    ]

    bm25 = [
        {"chunk_id": "p2:c1", "text": "b", "page_number": 2,
         "dense_score": 0.0, "bm25_score": 0.8, "hybrid_score": 0.8,
         "source_file": "test.pdf"},
        {"chunk_id": "p3:c2", "text": "c", "page_number": 3,
         "dense_score": 0.0, "bm25_score": 0.6, "hybrid_score": 0.6,
         "source_file": "test.pdf"},
    ]

    fused = _reciprocal_rank_fusion(dense, bm25, top_k=3)

    # p2:c1 appears in both → should score highest or second
    chunk_ids = [c["chunk_id"] for c in fused]
    assert "p2:c1" in chunk_ids
    # All results deduplicated
    assert len(chunk_ids) == len(set(chunk_ids))


def test_reranker_output():
    from core.retrieval.reranker import rerank

    chunks = [
        {"chunk_id": "p1:c0", "text": "EBITDA decreased to 7,688 crore in H1-26",
         "page_number": 1, "hybrid_score": 0.8},
        {"chunk_id": "p2:c1", "text": "The weather was nice today",
         "page_number": 2, "hybrid_score": 0.6},
    ]

    reranked = rerank("What is the EBITDA?", chunks, top_k=2)

    assert len(reranked) >= 1
    # Financial chunk should rank higher than weather chunk
    assert reranked[0]["chunk_id"] == "p1:c0"
    assert "rerank_score" in reranked[0]


def test_relevance_check():
    from core.retrieval.retriever import check_relevance

    high_score = [{"hybrid_score": 0.8}]
    low_score  = [{"hybrid_score": 0.1}]
    empty      = []

    assert check_relevance(high_score, threshold=0.35) is True
    assert check_relevance(low_score,  threshold=0.35) is False
    assert check_relevance(empty) is False