if __name__ == "__main__":
    from paddleocr import TextDetection
    from pathlib import Path
    from app.preprocessing import get_rotate_crop_image
    ROOT_DIR = Path(__file__).parents[2]
    model = TextDetection(
        model_name="PP-OCRv5_server_det",
        model_dir=Path(ROOT_DIR / "models/detection/PP-OCRv5_server_det")
    )
    result = model.predict(str(ROOT_DIR / "receipt_0.jpg"))
    pass