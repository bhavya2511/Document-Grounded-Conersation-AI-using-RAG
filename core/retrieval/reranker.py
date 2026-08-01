import logging
import os
from typing import List, Dict, Any
import voyageai

logger = logging.getLogger(__name__)

# Loaded once at module level
_voyage_client = None

def get_reranker() -> voyageai.Client:
    """Lazy-load the voyageai client."""
    global _voyage_client
    if _voyage_client is None:
        logger.info("Initializing Voyage AI client...")
        # Will automatically use VOYAGE_API_KEY from environment
        _voyage_client = voyageai.Client()
        logger.info("Voyage AI client initialized.")
    return _voyage_client


def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Rerank chunks using Voyage AI reranker.
    
    Returns top_k reranked chunks with updated scores.
    """
    if not chunks:
        return []
    
    if len(chunks) == 1:
        chunks[0]["rerank_score"] = chunks[0].get("hybrid_score", 0.5)
        return chunks
    
    try:
        client = get_reranker()
        
        # Build documents list
        # Extract text from chunks, Voyage API supports strings
        documents = [chunk["text"] for chunk in chunks]
        
        # Get reranking results
        reranking = client.rerank(
            query=query, 
            documents=documents, 
            model="rerank-2.5", 
            top_k=top_k
        )
        
        # The result returns a list of RerankingResult objects, which have `index` and `relevance_score`
        # We need to map the scores back to the original chunks
        reranked_chunks = []
        for r in reranking.results:
            chunk = chunks[r.index].copy() # Keep original chunk
            chunk["rerank_score"] = round(float(r.relevance_score), 4)
            reranked_chunks.append(chunk)
            
        logger.info(
            f"Reranking: {len(chunks)} → {len(reranked_chunks)} chunks. "
            f"Top score: {reranked_chunks[0]['rerank_score']:.3f}"
        )
        
        return reranked_chunks
    
    except Exception as e:
        logger.error(f"Reranking failed, returning original order: {e}")
        # Fallback: return by hybrid score
        for chunk in chunks:
            chunk["rerank_score"] = chunk.get("hybrid_score", 0.0)
        
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]