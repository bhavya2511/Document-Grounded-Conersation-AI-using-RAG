import fitz
import numpy as np
from PIL import Image

def page_to_numpy(page: fitz.Page, dpi: int = 200) -> np.ndarray:
    """
    Convert a PyMuPDF page to a numpy array (RGB image).
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Convert PyMuPDF pixmap to PIL Image, then to numpy array
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(img)

def crop_image(image: np.ndarray, bbox: list) -> np.ndarray:
    """
    Crop an image using a bounding box [x0, y0, x1, y1].
    """
    x0, y0, x1, y1 = [int(v) for v in bbox]
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    
    return image[y0:y1, x0:x1]

def normalize_bbox(bbox: list, scale: float = 1.0) -> dict:
    """
    Convert list [x0, y0, x1, y1] to dictionary format, 
    scaling from pixels back to PDF points if necessary.
    """
    return {
        "x0": float(bbox[0] / scale),
        "y0": float(bbox[1] / scale),
        "x1": float(bbox[2] / scale),
        "y1": float(bbox[3] / scale)
    }
