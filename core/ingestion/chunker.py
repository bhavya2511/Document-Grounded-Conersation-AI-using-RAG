import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_pages(pages: List[Dict[str, Any]], source_file: str) -> List[Dict[str, Any]]:
    """
    Offline deterministic chunker for layout-based PDF content.
    """

    chunks = []
    chunk_id = 0

    for page in pages:
        page_num = page["page_number"]
        page_width = page.get("page_width")
        page_height = page.get("page_height")
        source = page.get("source_file", source_file)

        elements = page.get("content", [])

        buffer = []
        buffer_type = None
        buffer_bboxes = []

        for el in elements:
            text = (el.get("text") or "").strip()
            if not text:
                continue

            el_type = el.get("type", "text")
            el_bbox = el.get("bbox")

            # -------------------------
            # TABLE HANDLING (atomic)
            # -------------------------
            if el_type == "table":
                chunks.append(_make_chunk(
                    text=text,
                    page_num=page_num,
                    chunk_index=chunk_id,
                    source_file=source,
                    content_type="table",
                    bbox=el_bbox,
                    page_width=page_width,
                    page_height=page_height
                ))
                chunk_id += 1
                continue

            # -------------------------
            # TEXT MERGING LOGIC
            # -------------------------
            if len(buffer) == 0:
                buffer.append(text)
                if el_bbox: buffer_bboxes.append(el_bbox)
                buffer_type = el_type
                continue

            # merge same type
            if el_type == buffer_type and len(" ".join(buffer)) < CHUNK_SIZE:
                buffer.append(text)
                if el_bbox: buffer_bboxes.append(el_bbox)
            else:
                new_chunks, num_created = _flush_buffer(buffer, buffer_bboxes, buffer_type, page_num, source, chunk_id, page_width, page_height)
                chunks.extend(new_chunks)
                chunk_id += num_created
                
                buffer = [text]
                buffer_bboxes = [el_bbox] if el_bbox else []
                buffer_type = el_type

        # flush last buffer
        if buffer:
            new_chunks, num_created = _flush_buffer(buffer, buffer_bboxes, buffer_type, page_num, source, chunk_id, page_width, page_height)
            chunks.extend(new_chunks)
            chunk_id += num_created

    logger.info(f"Created {len(chunks)} deterministic chunks")
    return chunks


def _flush_buffer(buffer, bboxes, ctype, page_num, source, start_chunk_id, page_width, page_height):
    text = " ".join(buffer)
    
    # Calculate union of all bounding boxes in the buffer
    combined_bbox = None
    if bboxes:
        try:
            # Handle dictionary bbox {x0, y0, x1, y1}
            if isinstance(bboxes[0], dict) and 'x0' in bboxes[0]:
                combined_bbox = {
                    "x0": min(b["x0"] for b in bboxes),
                    "y0": min(b["y0"] for b in bboxes),
                    "x1": max(b["x1"] for b in bboxes),
                    "y1": max(b["y1"] for b in bboxes),
                }
            # Handle array bbox [x0, y0, x1, y1]
            elif isinstance(bboxes[0], (list, tuple)) and len(bboxes[0]) >= 4:
                combined_bbox = [
                    min(b[0] for b in bboxes),
                    min(b[1] for b in bboxes),
                    max(b[2] for b in bboxes),
                    max(b[3] for b in bboxes),
                ]
        except Exception as e:
            logger.warning(f"Failed to combine bboxes: {e}")

    parts = []
    current_chunk_id = start_chunk_id
    
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        part = text[i:i + CHUNK_SIZE]

        parts.append(_make_chunk(
            text=part,
            page_num=page_num,
            chunk_index=current_chunk_id,
            source_file=source,
            content_type=ctype or "text",
            bbox=combined_bbox,
            page_width=page_width,
            page_height=page_height
        ))
        current_chunk_id += 1

    return parts, (current_chunk_id - start_chunk_id)


def _make_chunk(text, page_num, chunk_index, source_file, content_type, bbox=None, page_width=None, page_height=None):
    return {
        "text": text,
        "chunk_id": f"p{page_num}:c{chunk_index}",
        "page_number": page_num,
        "chunk_index": chunk_index,
        "source_file": source_file,
        "content_type": content_type,
        "bbox": bbox,
        "page_width": page_width,
        "page_height": page_height,
        "char_count": len(text),
        "word_count": len(text.split())
    }