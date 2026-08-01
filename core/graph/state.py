from typing import TypedDict, List, Optional, Any


class RAGState(TypedDict):
    # Conversation
    messages: List[dict]
    original_query: str
    rewritten_query: str

    # Retrieval
    retrieved_chunks: List[dict]
    reranked_chunks: List[dict]
    retrieval_scores: List[float]

    # Generation
    answer: str
    citations: List[str]
    confidence: float
    is_grounded: bool
    not_found: bool

    # Metadata 
    doc_id: str   
    workspace_id: str

    # Debug info passed to frontend
    debug: dict