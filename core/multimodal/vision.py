import os
import logging
import numpy as np
from PIL import Image
from io import BytesIO
import google.generativeai as genai

logger = logging.getLogger(__name__)

_vision_model = None

def get_vision_model():
    global _vision_model
    if _vision_model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY is not set.")
            raise ValueError("GEMINI_API_KEY is required for vision processing.")
        
        genai.configure(api_key=api_key)
        # Assuming the model name from env or default
        model_name = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
        _vision_model = genai.GenerativeModel(model_name)
    return _vision_model

def describe_image(image: np.ndarray) -> str:
    """
    Send an image array to Gemini Vision and return a description.
    """
    try:
        model = get_vision_model()
        
        # Convert numpy array to PIL Image, then to bytes
        img = Image.fromarray(image)
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        prompt = "Describe this image or figure in detail. Extract any relevant text or data."
        
        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": img_bytes}
        ])
        
        return response.text.strip()
    except Exception as e:
        logger.error(f"Vision processing failed: {e}")
        return "Image description unavailable."
