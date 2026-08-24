"""
Text extraction inference — takes a processed image (segmented and preprocessed).
Returns detected receipt text boxes and the text.
"""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).parents[2]


def process_text_extraction_results(img: np.ndarray, result: dict):
    """
    Processes the raw results from the text extraction model to map coordinates back to the original image size.
    
    Args:
        img: the original input image
        result: raw result from the text extraction model
    Returns:
        A list of dictionaries containing the text and the corresponding bounding box coordinates in the original image size.
    """
    words, boxes = [], []
    w, h = img.shape[:2]
    texts = result["rec_texts"]
    polys = result["rec_polys"]
    for text, poly in zip(texts, polys):
        xs = poly[:, 0]
        ys = poly[:, 1]
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        # normalize because LayoutLMv3 train and inf on 1000 max
        nx0 = int(1000*x0/w)
        ny0 = int(1000*y0/h)
        nx1 = int(1000*x1/w)
        ny1 = int(1000*y1/h)

        nx0 = max(0, min(1000, nx0))
        ny0 = max(0, min(1000, ny0))
        nx1 = max(0, min(1000, nx1))
        ny1 = max(0, min(1000, ny1))

        boxes.append([nx0, ny0, nx1, ny1])
        words.append(text)
    return words, boxes

def run_text_extraction(
        model_wrapper, 
        image: np.ndarray) -> dict:
    """
    Runs the text extraction model on a single image. best in RGB format (trained on)

    Args:
        model_wrapper: text-extraction model wrapper returned by model_loader.load_model()
        image: numpy array, already preprocessed (segmented and deskewed)

    Returns:
        Raw result of the text extraction model, which includes detected text boxes and recognized text.
    """
    # Run the OCR text extraction model
    result = model_wrapper.predict(image)[0]
    return result


if __name__ == "__main__":
    from pathlib import Path
    from .model_loader import load_model
    from app.config import paddle_ocr_params

    ROOT_DIR = Path(__file__).parents[2]
    params = paddle_ocr_params.model_dump()
    params_tuple = tuple(sorted(params.items()))
    print("load model wrapper")
    model_wrapper = load_model(
        det_model_path=Path(ROOT_DIR / "models/detection/PP-OCRv5_server_det"),
        rec_model_path=Path(ROOT_DIR / "models/recognition/PP-OCRv5_server_rec"),
        params=params_tuple
    )
    print("run text extraction")
    # Dev on my macbook, image >1024x1024 will crash the paddleocr model (known issue)
    img = Image.open(ROOT_DIR / "receipt_0.jpg").convert("RGB")
    w, h = img.size
    # resize keep ratio, max side 1024, to avoid paddleocr crash on large images
    # Calculate new dimensions based on max side
    max_side = 1024*2.3 # best ratio to maintain the performance
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        # Resize
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Convert to NumPy array
        img_np = np.ascontiguousarray(np.array(img_resized), dtype=np.uint8)
    else:
        img_np = np.ascontiguousarray(np.array(img), dtype=np.uint8)

    result = run_text_extraction(
        model_wrapper=model_wrapper, 
        image=img_np)
    result.save_to_img("outtest")
    result.save_to_json("outtest")