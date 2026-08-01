import logging
import numpy as np
import os

# Resolve Windows DLL conflicts (Paddle vs Torch OpenMP)
# This MUST happen before any paddle/torch imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Disable MKLDNN/PIR at env level to bypass 'ConvertPirAttribute2RuntimeAttribute' error
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_new_executor"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_new_executor_serial_run"] = "1"

logger = logging.getLogger(__name__)

# Hack: Pre-import torch to lock its DLLs before Paddle tries to load its own
try:
    import torch
    logger.info("Pre-imported torch to resolve DLL conflicts.")
except Exception:
    pass

# Global engine instance
_paddle_engine = None

def _setup_paddle_flags():
    """Set global flags to avoid common Windows/oneDNN errors."""
    try:
        import paddle
        # Disable PIR and New Executor which are unstable in Paddle 3.x on Windows
        flags = {
            "FLAGS_enable_pir_api": 0,
            "FLAGS_use_mkldnn": 0,
            "FLAGS_enable_new_executor": 0,
            "FLAGS_enable_pir_in_executor": 0,
        }
        paddle.set_flags(flags)
        # Force dynamic mode if available
        if hasattr(paddle, "disable_static"):
            paddle.disable_static()
        logger.info("PaddlePaddle flags optimized for Windows compatibility.")
    except Exception as e:
        logger.warning(f"Could not set PaddlePaddle flags: {e}")

def get_paddle_engine():
    """
    Initialize and return the global PaddleOCR PP-Structure engine via paddleocr.
    """
    global _paddle_engine
    if _paddle_engine is None:
        _setup_paddle_flags()
        try:
            from paddleocr import PPStructureV3
            logger.info("Initializing PaddleOCR PP-StructureV3 engine...")
            # Use the high-level PPStructureV3 class as per documentation
            _paddle_engine = PPStructureV3(device="cpu", use_formula_recognition=False)
            logger.info("PaddleOCR engine initialized.")
        except ImportError:
            logger.error("paddleocr is not installed or PPStructureV3 is missing.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PPStructureV3 engine: {e}")
            raise
    return _paddle_engine

def detect_layout(image: np.ndarray) -> list:
    """
    Run PaddleOCR PP-Structure to detect layout blocks in an image.
    Returns a list of blocks compatible with the multimodal pipeline.
    """
    engine = get_paddle_engine()
    
    # PPStructureV3 predict returns a list of result objects (one per image)
    try:
        # If it's a single image, wrap it in a list or pass directly
        outputs = engine.predict(image)
        if not outputs:
            return []
            
        res_obj = outputs[0]
        
        # PPStructureV3 result object usually has a 'doc_res' attribute or similar
        # Based on the documentation, we can extract the layout blocks.
        if hasattr(res_obj, "doc_res") and isinstance(res_obj.doc_res, dict):
            if "layout" in res_obj.doc_res:
                return res_obj.doc_res["layout"]
        
        # Try to see if it's a dict or has a to_dict method
        res_dict = {}
        if isinstance(res_obj, dict):
            res_dict = res_obj
        elif hasattr(res_obj, "to_dict"):
            res_dict = res_obj.to_dict()
            
        if "doc_res" in res_dict and "layout" in res_dict["doc_res"]:
            return res_dict["doc_res"]["layout"]
        if "layout" in res_dict:
            return res_dict["layout"]
            
    except Exception as e:
        logger.warning(f"Error running PPStructureV3 prediction: {e}")

    logger.warning("Could not parse blocks from PP-StructureV3 result.")
    return []
