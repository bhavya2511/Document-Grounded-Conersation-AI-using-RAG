import os
import uuid
import pickle
import logging
from typing import List, Dict, Any
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, FilterSelector
)
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from core.generation.llm import get_embedding
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

QDRANT_MODE = os.getenv("QDRANT_MODE", "local")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./storage/qdrant_data")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./storage/bm25_index")

COLLECTION_PREFIX = "workspace"
DENSE_DIM = 768  # Gemini text-embedding-004 dimension


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client based on config."""
    if QDRANT_MODE == "local":
        Path(QDRANT_LOCAL_PATH).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    else:
        return QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=os.getenv("QDRANT_API_KEY") or None
        )


def get_collection_name(workspace_id: str) -> str:
    """Generate a clean collection name from workspace_id."""
    clean = "".join(c if c.isalnum() else "_" for c in workspace_id)
    return f"{COLLECTION_PREFIX}_{clean}"


def index_chunks(chunks: List[Dict[str, Any]], doc_id: str, workspace_id: str) -> Dict[str, Any]:
    """
    Embed all chunks and store in Qdrant.
    Also builds BM25 sparse index for hybrid retrieval.
    
    Returns indexing stats.
    """
    if not chunks:
        raise ValueError("No chunks to index")
    
    client = get_qdrant_client()
    collection_name = get_collection_name(workspace_id)
    
    # Create or recreate collection
    _ensure_collection(client, collection_name)
    
    # Embed and store chunks in batches
    logger.info(f"Indexing {len(chunks)} chunks into collection '{collection_name}' for doc '{doc_id}'")
    
    points = []
    failed = 0
    
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks")):
        try:
            text = chunk["text"]
            if not text.strip():
                continue
            
            # Get dense embedding from Gemini
            dense_vector = get_embedding(text)
            
            # Deterministic point ID based on workspace, doc, and chunk id
            point_id_str = f"{workspace_id}_{doc_id}_{chunk['chunk_id']}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_id_str))
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector={"dense": dense_vector},
                payload={
                    # Core fields
                    "text": text,
                    "chunk_id": chunk["chunk_id"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "source_file": chunk["source_file"],
                    "doc_id": doc_id,
                    "workspace_id": workspace_id,
                    
                    # Highlighting
                    "bbox": chunk.get("bbox"),
                    "page_width": chunk.get("page_width"),
                    "page_height": chunk.get("page_height"),
                    
                    # Content metadata
                    "content_type": chunk.get("content_type", "text"),
                    "extraction_methods": chunk.get("extraction_methods", []),
                    "numbers_present": chunk.get("numbers_present", []),
                    "periods_present": chunk.get("periods_present", []),
                    
                    # Stats
                    "char_count": chunk.get("char_count", len(text)),
                    "word_count": chunk.get("word_count", len(text.split())),
                    
                    # Retrieval tracking
                    "retrieval_count": 0,
                }
            )
            points.append(point)
            
        except Exception as e:
            logger.error(f"Failed to embed chunk {i}: {e}")
            failed += 1
            continue
    
    # Batch upload to Qdrant
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
    
    logger.info(f"Indexed {len(points)} chunks. Failed: {failed}")
    
    # Build and save BM25 index
    _build_bm25_index(chunks, doc_id)
    
    return {
        "collection_name": collection_name,
        "total_chunks": len(chunks),
        "indexed": len(points),
        "failed": failed
    }


def _ensure_collection(client: QdrantClient, collection_name: str):
    """Ensure Qdrant collection exists."""
    collections = client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE
            )
        }
    )
    logger.info(f"Created collection: {collection_name}")


def _build_bm25_index(chunks: List[Dict[str, Any]], doc_id: str):
    """Build and save BM25 sparse index for keyword retrieval."""
    Path(BM25_INDEX_PATH).mkdir(parents=True, exist_ok=True)
    
    # Tokenize all chunk texts
    tokenized_corpus = []
    chunk_refs = []
    
    for chunk in chunks:
        text = chunk["text"].lower()
        tokens = text.split()
        tokenized_corpus.append(tokens)
        chunk_refs.append({
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "text": chunk["text"],
            "source_file": chunk["source_file"],
            "bbox": chunk.get("bbox"),
            "page_width": chunk.get("page_width"),
            "page_height": chunk.get("page_height"),
            "content_type": chunk.get("content_type", "text"),
        })
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    index_data = {
        "bm25": bm25,
        "chunk_refs": chunk_refs,
        "doc_id": doc_id
    }
    
    index_path = os.path.join(BM25_INDEX_PATH, f"{doc_id}.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)
    
    logger.info(f"BM25 index saved: {index_path}")


def load_bm25_index(doc_id: str) -> Dict | None:
    """Load BM25 index for a document."""
    index_path = os.path.join(BM25_INDEX_PATH, f"{doc_id}.pkl")
    
    if not os.path.exists(index_path):
        return None
    
    with open(index_path, "rb") as f:
        return pickle.load(f)


def delete_document_index(doc_id: str, workspace_id: str):
    """Delete a document's points from Qdrant and BM25 index file."""
    client = get_qdrant_client()
    collection_name = get_collection_name(workspace_id)
    
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted points for doc {doc_id} in collection: {collection_name}")
    except Exception as e:
        logger.warning(f"Could not delete points: {e}")
    
    index_path = os.path.join(BM25_INDEX_PATH, f"{doc_id}.pkl")
    if os.path.exists(index_path):
        os.remove(index_path)
        logger.info(f"Deleted BM25 index: {index_path}")