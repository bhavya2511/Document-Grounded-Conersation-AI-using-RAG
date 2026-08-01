import logging
import fitz
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from .graph import ExtractionState
from .nodes import (
    load_page_node,
    layout_detection_node,
    router_node,
    process_text_node,
    process_title_node,
    process_table_node,
    process_figure_node,
    merge_node
)
from .routing import route_block
from core.ingestion.pdf_extractor import extract_pdf

logger = logging.getLogger(__name__)

def build_extraction_graph():
    """Build the LangGraph pipeline for multimodal extraction."""
    workflow = StateGraph(ExtractionState)
    
    # Add nodes
    workflow.add_node("load_page_node", load_page_node)
    workflow.add_node("layout_detection_node", layout_detection_node)
    workflow.add_node("router_node", router_node)
    workflow.add_node("process_text_node", process_text_node)
    workflow.add_node("process_title_node", process_title_node)
    workflow.add_node("process_table_node", process_table_node)
    workflow.add_node("process_figure_node", process_figure_node)
    workflow.add_node("merge_node", merge_node)
    
    # Add edges
    workflow.set_entry_point("load_page_node")
    workflow.add_edge("load_page_node", "layout_detection_node")
    workflow.add_edge("layout_detection_node", "router_node")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "router_node",
        route_block,
        {
            "process_text_node": "process_text_node",
            "process_title_node": "process_title_node",
            "process_table_node": "process_table_node",
            "process_figure_node": "process_figure_node",
            "merge_node": "merge_node"
        }
    )
    
    # All processing nodes loop back to router
    workflow.add_edge("process_text_node", "router_node")
    workflow.add_edge("process_title_node", "router_node")
    workflow.add_edge("process_table_node", "router_node")
    workflow.add_edge("process_figure_node", "router_node")
    
    workflow.add_edge("merge_node", END)
    
    return workflow.compile()

def extract_pdf_multimodal(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Main entry point for multimodal PDF extraction.
    Uses LangGraph pipeline for per-page extraction.
    Falls back to PyMuPDF extraction on failure.
    """
    logger.info(f"Starting multimodal extraction for: {pdf_path}")
    
    try:
        # Pre-check if document can be opened
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}")
            raise e
            
        graph = build_extraction_graph()
        all_results = []
        
        for page_num in range(1, num_pages + 1):
            logger.info(f"Processing multimodal page {page_num}/{num_pages}")
            
            initial_state = {
                "pdf_path": pdf_path,
                "page_number": page_num,
                "blocks": [],
                "processed_blocks": [],
                "results": []
            }
            
            final_state = graph.invoke(initial_state)
            if final_state["results"]:
                all_results.append(final_state["results"][0])
                
        logger.info(f"Multimodal extraction complete. {len(all_results)} pages processed.")
        return all_results
        
    except Exception as e:
        logger.error(f"Multimodal extraction failed: {e}. Falling back to standard PyMuPDF extraction.")
        return extract_pdf(pdf_path)
