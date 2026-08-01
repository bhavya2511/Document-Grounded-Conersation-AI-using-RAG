import fitz  # PyMuPDF
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def extract_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Fully offline PDF extraction using ONLY PyMuPDF.
    No LLM, no vision, no pdfplumber.
    """
    logger.info(f"Extracting PDF: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        logger.info(f"Processing page {page_num}/{len(doc)}")

        pages.append(_extract_page(page, page_num, pdf_path))

    doc.close()
    logger.info(f"Extraction complete. {len(pages)} pages processed.")
    return pages


def _extract_page(page, page_num: int, pdf_path: str) -> Dict[str, Any]:
    """
    Layout-aware extraction:
    - text blocks
    - table regions
    - filtered duplication removal
    - reading order reconstruction
    """

    page_width, page_height = page.rect.width, page.rect.height

    # -----------------------------
    # 1. Detect tables (geometry)
    # -----------------------------
    tables = page.find_tables()
    table_bboxes = [t.bbox for t in tables]

    # -----------------------------
    # 2. Extract text blocks
    # -----------------------------
    blocks = page.get_text("dict")["blocks"]

    text_elements = []

    for b in blocks:
        if b["type"] != 0:
            continue

        for line in b["lines"]:
            line_text = "".join(span["text"] for span in line["spans"]).strip()
            if not line_text:
                continue

            x0, y0, x1, y1 = line["bbox"]

            text_elements.append({
                "text": line_text,
                "bbox": (x0, y0, x1, y1),
                "type": "text"
            })

    # -----------------------------
    # 3. Table extraction (clean)
    # -----------------------------
    table_elements = []

    for t in tables:
        try:
            data = t.extract()
            md = _table_to_markdown(data)

            table_elements.append({
                "text": md,
                "bbox": t.bbox,
                "type": "table"
            })
        except Exception:
            continue

    # -----------------------------
    # 4. Remove duplicate text inside tables
    # -----------------------------
    filtered_text = []

    for el in text_elements:
        if not _inside_any(el["bbox"], table_bboxes):
            filtered_text.append(el)

    # -----------------------------
    # 5. Merge everything (layout order)
    # -----------------------------
    all_elements = filtered_text + table_elements

    all_elements.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

    content = []
    for el in all_elements:
        content.append({
            "text": el["text"],
            "type": el["type"],
            "bbox": {
                "x0": el["bbox"][0],
                "y0": el["bbox"][1],
                "x1": el["bbox"][2],
                "y1": el["bbox"][3],
            }
        })

    return {
        "page_number": page_num,
        "content": content,
        "page_width": page_width,
        "page_height": page_height,
        "source_file": pdf_path
    }


# -----------------------------
# helpers
# -----------------------------

def _inside_any(bbox, regions):
    x0, y0, x1, y1 = bbox

    for r in regions:
        rx0, ry0, rx1, ry1 = r

        if (
            x0 >= rx0 - 2 and y0 >= ry0 - 2 and
            x1 <= rx1 + 2 and y1 <= ry1 + 2
        ):
            return True

    return False


def _table_to_markdown(table):
    if not table:
        return ""

    table = [[str(c or "").replace("|", "\\|") for c in row] for row in table]
    table = [r for r in table if any(x.strip() for x in r)]

    if not table:
        return ""

    cols = max(len(r) for r in table)

    for r in table:
        while len(r) < cols:
            r.append("")

    header = "| " + " | ".join(table[0]) + " |"
    sep = "| " + " | ".join(["---"] * cols) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in table[1:])

    return header + "\n" + sep + "\n" + body