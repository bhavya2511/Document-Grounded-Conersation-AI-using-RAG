import logging
import fitz
from .graph import ExtractionState
from .utils import page_to_numpy, crop_image, normalize_bbox
from .ocr import detect_layout
from .vision import describe_image
from .table import parse_html_table

logger = logging.getLogger(__name__)

def load_page_node(state: ExtractionState) -> ExtractionState:
    """Load PDF page, convert to image, set dimensions."""
    pdf_path = state["pdf_path"]
    page_number = state["page_number"]
    
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1] # 0-indexed in fitz
    
    state["page_width"] = page.rect.width
    state["page_height"] = page.rect.height
    
    # Convert page to image for PaddleOCR at 200 DPI
    dpi = 200
    state["page_image"] = page_to_numpy(page, dpi=dpi)
    state["scale"] = dpi / 72.0
    
    doc.close()
    return state

def layout_detection_node(state: ExtractionState) -> ExtractionState:
    """Run Paddle PP-Structure to detect blocks."""
    image = state["page_image"]
    blocks = detect_layout(image)
    
    # Filter out empty blocks or invalid ones if needed
    valid_blocks = [b for b in blocks if "bbox" in b and "type" in b]
    
    state["blocks"] = valid_blocks
    state["processed_blocks"] = []
    
    return state

def router_node(state: ExtractionState) -> ExtractionState:
    """Pop the next block to process."""
    blocks = state["blocks"]
    if blocks:
        state["current_block"] = blocks.pop(0)
    else:
        state["current_block"] = None
    return state

def process_text_node(state: ExtractionState) -> ExtractionState:
    """Extract text and spans from a text block."""
    block = state["current_block"]
    scale = state["scale"]
    
    # PaddleOCR returns 'res' which is a list of lines, each line is a dict
    text_content = ""
    spans = []
    
    if "res" in block and isinstance(block["res"], list):
        for line in block["res"]:
            if "text" in line:
                line_text = line["text"]
                # line['text_region'] is [x0, y0, x1, y1, x2, y2, x3, y3] or [x0, y0, x1, y1]
                region = line.get("text_region", block["bbox"])
                if len(region) == 8: # Polygon to bbox
                    x_coords = region[0::2]
                    y_coords = region[1::2]
                    line_bbox_list = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                else:
                    line_bbox_list = region
                    
                line_bbox = normalize_bbox(line_bbox_list, scale)
                
                text_content += line_text + " "
                spans.append({
                    "text": line_text,
                    "bbox": line_bbox
                })
    
    processed = {
        "text": text_content.strip(),
        "type": "text",
        "bbox": normalize_bbox(block["bbox"], scale),
        "spans": spans
    }
    
    state["processed_blocks"].append(processed)
    return state

def process_title_node(state: ExtractionState) -> ExtractionState:
    """Extract text and spans, mark as title."""
    block = state["current_block"]
    scale = state["scale"]
    
    text_content = ""
    spans = []
    
    if "res" in block and isinstance(block["res"], list):
        for line in block["res"]:
            if "text" in line:
                line_text = line["text"]
                region = line.get("text_region", block["bbox"])
                if len(region) == 8:
                    x_coords = region[0::2]
                    y_coords = region[1::2]
                    line_bbox_list = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                else:
                    line_bbox_list = region
                    
                line_bbox = normalize_bbox(line_bbox_list, scale)
                
                text_content += line_text + " "
                spans.append({
                    "text": line_text,
                    "bbox": line_bbox
                })
                
    processed = {
        "text": text_content.strip(),
        "type": "title",
        "bbox": normalize_bbox(block["bbox"], scale),
        "spans": spans,
        "is_section_header": True
    }
    
    state["processed_blocks"].append(processed)
    return state

def process_table_node(state: ExtractionState) -> ExtractionState:
    """Process table block, extracting HTML and converting to Markdown."""
    block = state["current_block"]
    scale = state["scale"]
    
    html_content = ""
    if "res" in block and isinstance(block["res"], dict) and "html" in block["res"]:
        html_content = block["res"]["html"]
        
    markdown_table = parse_html_table(html_content)
    
    processed = {
        "text": markdown_table,
        "type": "table",
        "bbox": normalize_bbox(block["bbox"], scale)
    }
    
    state["processed_blocks"].append(processed)
    return state

def process_figure_node(state: ExtractionState) -> ExtractionState:
    """Crop figure and send to Gemini Vision."""
    block = state["current_block"]
    scale = state["scale"]
    image = state["page_image"]
    bbox = block["bbox"]
    
    cropped_img = crop_image(image, bbox)
    description = describe_image(cropped_img)
    
    processed = {
        "text": f"[Image Description]: {description}",
        "type": "figure",
        "bbox": normalize_bbox(bbox, scale)
    }
    
    state["processed_blocks"].append(processed)
    return state

def merge_node(state: ExtractionState) -> ExtractionState:
    """Sort processed blocks by bbox (y0, x0) and format output."""
    processed_blocks = state["processed_blocks"]
    
    # Sort blocks by vertical position, then horizontal
    processed_blocks.sort(key=lambda x: (x["bbox"]["y0"], x["bbox"]["x0"]))
    
    result = {
        "page_number": state["page_number"],
        "content": processed_blocks,
        "page_width": state["page_width"],
        "page_height": state["page_height"],
        "source_file": state["pdf_path"]
    }
    
    state["results"].append(result)
    return state
