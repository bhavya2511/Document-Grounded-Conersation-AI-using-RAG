import os
import logging
from typing import List
from core.graph.state import RAGState
from core.generation.llm import get_llm_response
from core.generation.prompts import (
    QUERY_REWRITE_PROMPT,
    GROUNDED_ANSWER_PROMPT,
    HALLUCINATION_CHECK_PROMPT
)
from core.retrieval.retriever import hybrid_search, check_relevance
from core.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

TOP_K_AFTER_RERANK = int(os.getenv("TOP_K_AFTER_RERANK", 5))
SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", -10))


def rewrite_query_node(state: RAGState) -> RAGState:
    """
    Rewrite the query using conversation history to make it standalone.
    First turn: return as-is.
    Follow-up turns: expand pronouns and add context.
    """
    query = state["original_query"]
    messages = state.get("messages", [])
    
    # First turn - no rewriting needed
    if len(messages) <= 1:
        state["rewritten_query"] = query
        return state
    
    # Build history string (last 6 messages max)
    history_msgs = messages[-7:-1]  # exclude current
    history_str = ""
    for msg in history_msgs:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
    
    if not history_str.strip():
        state["rewritten_query"] = query
        return state
    
    prompt = QUERY_REWRITE_PROMPT.format(
        history=history_str,
        question=query
    )
    
    try:
        rewritten = get_llm_response(prompt, max_tokens=200)
        # Sanity check: if rewritten is too long or weird, use original
        if len(rewritten) > 500 or "\n" in rewritten:
            rewritten = query
        state["rewritten_query"] = rewritten
        print("FINAL REWRITTEN QUERY:", rewritten)
        logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}")
        state["rewritten_query"] = query
    
    return state


def retrieve_node(state: RAGState) -> RAGState:
    """
    Hybrid retrieval: dense + BM25 with RRF fusion.
    """
    query = state["rewritten_query"]
    doc_id = state.get("doc_id", "")
    workspace_id = state.get("workspace_id", "")
    
    try:
        results = hybrid_search(query, doc_id, workspace_id=workspace_id)
        state["retrieved_chunks"] = results
        state["retrieval_scores"] = [r.get("hybrid_score", 0) for r in results]
        
        logger.info(f"Retrieved {len(results)} chunks")
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        state["retrieved_chunks"] = []
        state["retrieval_scores"] = []
    
    return state


def rerank_node(state: RAGState) -> RAGState:
    """
    Rerank retrieved chunks with cross-encoder for better precision.
    """
    chunks = state.get("retrieved_chunks", [])
    query = state["rewritten_query"]
    
    if not chunks:
        state["reranked_chunks"] = []
        return state
    
    reranked = rerank(query, chunks, top_k=TOP_K_AFTER_RERANK)
    state["reranked_chunks"] = reranked
    
    return state


def grade_relevance_node(state: RAGState) -> RAGState:
    """
    Check if retrieved chunks are relevant enough to answer.
    If not relevant → route to not_found.
    """
    chunks = state.get("reranked_chunks", [])
    
    if not chunks:
        state["not_found"] = True
        state["confidence"] = 0.0
        return state
    
    # top_score = chunks[0].get("rerank_score", chunks[0].get("hybrid_score", 0))
    top_score = chunks[0].get("rerank_score")

    if top_score is None:
        top_score = chunks[0].get("hybrid_score", 0)
    
    # Grade: is top chunk relevant?
    is_relevant = top_score > SCORE_THRESHOLD
    
    state["not_found"] = not is_relevant
    state["confidence"] = round(float(top_score), 3)
    
    logger.info(f"Relevance check: score={top_score:.3f}, relevant={is_relevant}")
    
    return state

def _normalize_chunk_text(text: str, content_type: str = "text") -> str:
    """
    Convert tables into structured, LLM-understandable format.
    """

    if content_type == "table" or "|" in text:
        lines = [l.strip() for l in text.split("\n") if "|" in l]

        parsed = []
        for line in lines:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) >= 2:
                parsed.append(cells)

        # Not enough structure
        if len(parsed) < 2:
            return text[:800]

        header = parsed[0]
        rows = parsed[1:]

        structured = ["TABLE:"]
        structured.append("Columns: " + ", ".join(header))

        for row in rows:
            if len(row) == len(header):
                row_map = ", ".join(f"{header[i]}={row[i]}" for i in range(len(row)))
                structured.append(f"Row: {row_map}")
            else:
                structured.append("Row: " + " | ".join(row))

        return "\n".join(structured)

    return text[:800]

import re

def clean_citations(answer: str) -> str:
    """
    Remove duplicate citations like [p2:c3] appearing multiple times.
    Keeps only first occurrence.
    """
    seen = set()

    def repl(match):
        m = match.group(0)
        if m in seen:
            return ""
        seen.add(m)
        return m

    return re.sub(r'\[p\d+:c\d+\]', repl, answer)

def generate_node(state: RAGState) -> RAGState:
    """
    Generate grounded answer using retrieved context.
    Forces citations in [pN:cN] format.
    """
    query = state["original_query"]
    chunks = state.get("reranked_chunks", [])
    
    if not chunks:
        state["answer"] = "Not found in the document."
        state["citations"] = []
        state["is_grounded"] = False
        return state
    
    # Build context string with citation IDs embedded
    context_parts = []
    for chunk in chunks:
        cid = chunk["chunk_id"]
        #text = chunk["text"][:800] cap length per chunk
        text = _normalize_chunk_text(chunk["text"])[:800]
        context_parts.append(f"[{cid}]: {text}")
    
    context_str = "\n\n".join(context_parts)
    
    prompt = GROUNDED_ANSWER_PROMPT.format(
        question=query,
        context=context_str
    )
    
    try:
        answer = get_llm_response(prompt)
        answer = clean_citations(answer)
        
        # Extract citations from answer
        import re
        citations = list(dict.fromkeys(re.findall(r'\[p\d+:c\d+\]', answer)))
        
        state["answer"] = answer
        state["citations"] = citations
        state["is_grounded"] = True
        
        logger.info(f"Generated answer with {len(citations)} citations")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        state["answer"] = "Not found in the document."
        state["citations"] = []
        state["is_grounded"] = False
    
    return state


def hallucination_check_node(state: RAGState) -> RAGState:
    """
    Verify the answer is grounded in retrieved context.
    Uses the Auditor persona to verify factual adherence.
    """
    answer = state.get("answer", "")
    chunks = state.get("reranked_chunks", [])
    
    # Skip check for not_found responses
    if "not contain information" in answer.lower() or "not found" in answer.lower():
        state["is_grounded"] = True
        return state
    
    # Build robust context for check (more chunks, more text)
    context_str = "\n\n".join([f"[{c['chunk_id']}]: {c['text'][:1000]}" for c in chunks[:5]])
    
    prompt = HALLUCINATION_CHECK_PROMPT.format(
        answer=answer,
        context=context_str
    )
    
    try:
        result = get_llm_response(prompt, max_tokens=100)
        is_grounded = result.strip().upper().startswith("YES")
        state["is_grounded"] = is_grounded
        
        if not is_grounded:
            logger.warning(f"Hallucination check returned NO: {result}")
            # We no longer append the warning to the answer string directly.
            # The 'is_grounded' flag will be handled by the UI / debug info.
        
    except Exception as e:
        logger.warning(f"Hallucination check failed: {e}")
        state["is_grounded"] = True  # assume ok if check fails
    
    return state


def not_found_node(state: RAGState) -> RAGState:
    """Return not found response without calling LLM."""
    state["answer"] = "Not found in the document."
    state["citations"] = []
    state["is_grounded"] = True
    state["confidence"] = 0.0
    return state


def update_memory_node(state: RAGState) -> RAGState:
    """
    Append current Q&A to conversation history.
    Summarize old history if it gets too long.
    """
    messages = state.get("messages", [])
    answer = state.get("answer", "")
    
    # Add assistant response to history
    messages.append({
        "role": "assistant",
        "content": answer
    })
    
    # Trim history if too long (keep last 20 messages)
    if len(messages) > 20:
        messages = messages[-20:]
    
    state["messages"] = messages
    
    # Build debug info
    state["debug"] = {
        "original_query": state.get("original_query", ""),
        "rewritten_query": state.get("rewritten_query", ""),
        "retrieved_count": len(state.get("retrieved_chunks", [])),
        "reranked_count": len(state.get("reranked_chunks", [])),
        "confidence": round(min(max(state.get("confidence", 0), 0), 1) * 100, 2),
        "is_grounded": state.get("is_grounded", False),
        "not_found": state.get("not_found", False),
        "retrieved_chunks": [
            {
                "chunk_id": c.get("chunk_id"),
                "page_number": c.get("page_number"),
                "text_preview": c.get("text", "")[:150] + "...",
                "hybrid_score": c.get("hybrid_score", 0),
                "dense_score": c.get("dense_score", 0),
                "bm25_score": c.get("bm25_score", 0),
                "rerank_score": c.get("rerank_score", 0),
                "content_type": c.get("content_type", "text"),
                "retrieval_method": c.get("retrieval_method", ""),
                "bbox": c.get("bbox"),
                "page_width": c.get("page_width"),
                "page_height": c.get("page_height"),
            }
            for c in state.get("reranked_chunks", [])
        ]
    }
    
    return state


def route_after_grade(state: RAGState) -> str:
    """Conditional edge: route to generate or not_found."""
    if state.get("not_found", False):
        return "not_found"
    return "generate"