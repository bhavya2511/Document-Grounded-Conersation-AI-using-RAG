import os
import logging
from typing import List, Dict, Any, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from core.generation.llm import get_query_embedding
from core.ingestion.indexer import get_qdrant_client, get_collection_name, load_bm25_index
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 10))
SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", -10))

# Fusion weights
DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4


def hybrid_search(
    query: str,
    doc_id: str,
    workspace_id: str,
    top_k: int = TOP_K
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval combining dense vector search + BM25 keyword search.
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    
    Returns list of chunk dicts with scores.
    """
    
    # --- Dense search ---
    dense_results = _dense_search(query, doc_id, workspace_id, top_k * 2)
    
    # --- BM25 search ---
    bm25_results = _bm25_search(query, doc_id, top_k * 2)
    
    # --- Fuse with RRF ---
    fused = _reciprocal_rank_fusion(dense_results, bm25_results, top_k)
    
    logger.info(
        f"Hybrid search: {len(dense_results)} dense + "
        f"{len(bm25_results)} BM25 → {len(fused)} fused"
    )
    
    return fused


def _dense_search(query: str, doc_id: str, workspace_id: str, top_k: int) -> List[Dict[str, Any]]:
    """Dense vector search using Gemini embeddings."""
    try:
        client = get_qdrant_client()
        collection_name = get_collection_name(workspace_id)
        
        query_vector = get_query_embedding(query)
        
        results = client.search(
            collection_name=collection_name,
            query_vector=("dense", query_vector),
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
            score_threshold=0.1  # low threshold, reranker will filter
        )
        
        chunks = []
        for r in results:
            chunk = dict(r.payload)
            chunk["dense_score"] = round(r.score, 4)
            chunk["bm25_score"] = 0.0
            chunk["hybrid_score"] = round(r.score, 4)
            chunk["retrieval_method"] = "dense"
            chunks.append(chunk)
        
        return chunks
    
    except Exception as e:
        logger.error(f"Dense search failed: {e}")
        return []


def _bm25_search(query: str, doc_id: str, top_k: int) -> List[Dict[str, Any]]:
    """BM25 keyword search."""
    try:
        index_data = load_bm25_index(doc_id)
        if not index_data:
            logger.warning(f"No BM25 index found for doc_id: {doc_id}")
            return []
        
        bm25 = index_data["bm25"]
        chunk_refs = index_data["chunk_refs"]
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = bm25.get_scores(query_tokens)
        
        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        chunks = []
        max_score = max(scores) if max(scores) > 0 else 1.0
        
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            
            ref = chunk_refs[idx]
            normalized_score = scores[idx] / max_score
            
            chunk = {
                "text": ref["text"],
                "chunk_id": ref["chunk_id"],
                "page_number": ref["page_number"],
                "source_file": ref["source_file"],
                "bbox": ref.get("bbox"),
                "page_width": ref.get("page_width"),
                "page_height": ref.get("page_height"),
                "content_type": ref.get("content_type", "text"),
                "dense_score": 0.0,
                "bm25_score": round(float(normalized_score), 4),
                "hybrid_score": round(float(normalized_score), 4),
                "retrieval_method": "bm25"
            }
            chunks.append(chunk)
        
        return chunks
    
    except Exception as e:
        logger.error(f"BM25 search failed: {e}")
        return []


def _reciprocal_rank_fusion(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    top_k: int,
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion to merge dense and BM25 results.
    RRF score = sum(1 / (k + rank)) for each result list.
    Deduplicates by chunk_id.
    """
    rrf_scores = {}
    chunk_data = {}
    
    # Process dense results
    for rank, chunk in enumerate(dense_results):
        cid = chunk["chunk_id"]
        rrf_score = DENSE_WEIGHT * (1.0 / (k + rank + 1))
        rrf_scores[cid] = rrf_scores.get(cid, 0) + rrf_score
        if cid not in chunk_data:
            chunk_data[cid] = chunk
        else:
            chunk_data[cid]["dense_score"] = chunk.get("dense_score", 0)
    
    # Process BM25 results
    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        rrf_score = BM25_WEIGHT * (1.0 / (k + rank + 1))
        rrf_scores[cid] = rrf_scores.get(cid, 0) + rrf_score
        if cid not in chunk_data:
            chunk_data[cid] = chunk
        else:
            chunk_data[cid]["bm25_score"] = chunk.get("bm25_score", 0)
    
    # Sort by RRF score
    sorted_cids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for cid, rrf_score in sorted_cids[:top_k]:
        chunk = chunk_data[cid].copy()
        chunk["hybrid_score"] = round(rrf_score, 4)
        
        # Determine retrieval method
        has_dense = chunk.get("dense_score", 0) > 0
        has_bm25 = chunk.get("bm25_score", 0) > 0
        if has_dense and has_bm25:
            chunk["retrieval_method"] = "hybrid"
        elif has_dense:
            chunk["retrieval_method"] = "dense"
        else:
            chunk["retrieval_method"] = "bm25"
        
        results.append(chunk)
    
    return results


def check_relevance(results: List[Dict], threshold: float = SCORE_THRESHOLD) -> bool:
    """Check if any retrieved results are above confidence threshold."""
    if not results:
        return False
    return results[0].get("hybrid_score", 0) > threshold