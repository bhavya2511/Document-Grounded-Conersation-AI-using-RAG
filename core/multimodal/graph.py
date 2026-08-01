from typing import TypedDict, List, Dict, Any
import numpy as np

class ExtractionState(TypedDict):
    pdf_path: str
    page_image: np.ndarray
    blocks: List[Dict[str, Any]]
    current_block: Dict[str, Any]
    processed_blocks: List[Dict[str, Any]]
    page_number: int
    results: List[Dict[str, Any]]
    page_width: float
    page_height: float
    scale: float
