import logging
from langgraph.graph import StateGraph, END
from core.graph.state import RAGState
from core.graph.nodes import (
    rewrite_query_node,
    retrieve_node,
    rerank_node,
    grade_relevance_node,
    generate_node,
    hallucination_check_node,
    not_found_node,
    update_memory_node,
    route_after_grade
)

logger = logging.getLogger(__name__)

# Build the graph once at module level
def build_rag_graph():
    """Build and compile the LangGraph RAG pipeline."""
    
    graph = StateGraph(RAGState)
    
    # Add all nodes
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("grade_relevance", grade_relevance_node)
    graph.add_node("generate", generate_node)
    graph.add_node("hallucination_check", hallucination_check_node)
    graph.add_node("not_found_node", not_found_node)
    graph.add_node("update_memory", update_memory_node)
    
    # Set entry point
    graph.set_entry_point("rewrite_query")
    
    # Linear edges
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade_relevance")
    
    # Conditional edge: grade → generate OR not_found
    graph.add_conditional_edges(
        "grade_relevance",
        route_after_grade,
        {
            "generate": "generate",
            "not_found": "not_found_node"
        }
    )
    
    # Both paths converge to update_memory
    graph.add_edge("generate", "hallucination_check")
    graph.add_edge("hallucination_check", "update_memory")
    graph.add_edge("not_found_node", "update_memory")
    
    graph.add_edge("update_memory", END)
    
    compiled = graph.compile()
    logger.info("RAG graph compiled successfully")
    return compiled


# Singleton graph instance
_rag_graph = None

def get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph


def run_rag_pipeline(
    query: str,
    doc_id: str,
    conversation_history: list,
    workspace_id: str
) -> dict:
    """
    Run the full RAG pipeline for a single query.
    
    Args:
        query: User's question
        doc_id: Which document to search
        conversation_history: List of previous {role, content} messages
        workspace_id: Which workspace to search in
    
    Returns:
        Dict with answer, citations, debug info
    """
    graph = get_rag_graph()
    
    # Add current query to history
    messages = conversation_history.copy()
    messages.append({"role": "user", "content": query})
    
    initial_state = RAGState(
        messages=messages,
        original_query=query,
        rewritten_query=query,
        retrieved_chunks=[],
        reranked_chunks=[],
        retrieval_scores=[],
        answer="",
        citations=[],
        confidence=0.0,
        is_grounded=False,
        not_found=False,
        debug={},
        doc_id=doc_id,
        workspace_id=workspace_id
    )
    
    try:
        final_state = graph.invoke(initial_state)
        
        return {
            "answer": final_state["answer"],
            "citations": final_state["citations"],
            "confidence": final_state["confidence"],
            "is_grounded": final_state["is_grounded"],
            "not_found": final_state.get("not_found", False),
            "messages": final_state["messages"],
            "debug": final_state.get("debug", {}),
            "rewritten_query": final_state.get("rewritten_query", query)
        }
    
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}", exc_info=True)
        return {
            "answer": "An error occurred while processing your question.",
            "citations": [],
            "confidence": 0.0,
            "is_grounded": False,
            "not_found": True,
            "messages": messages,
            "debug": {"error": str(e)},
            "rewritten_query": query
        }