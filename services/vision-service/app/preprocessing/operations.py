"""
Preprocessing operations — takes a raw image, segmentation results to preprocess.
Operations: crop, deskew, perspective correct, ...
"""

import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass

@dataclass
class PreprocessingResult:
    image: np.ndarray

# def resize_big_image(img: np.ndarray, max_side: float = 1024*2.3) -> np.ndarray:
#     """
#     Reduce the size of high resolution (Macbook problem) for paddleOCR
#     Keep the original dim ratio
#     Args:
#         img: the original image (numpy array)
#     Returns:
#         The resized image
#     """
#     h, w = img.shape[:2]  # numpy convention: (H, W, C)

#     if max(w, h) > max_side:
#         scale = max_side / max(w, h)
#         new_w, new_h = int(w * scale), int(h * scale)
#         # cv2.resize takes size as (width, height) — the reverse of .shape
#         img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
#     else:
#         img_resized = img

#     return np.ascontiguousarray(img_resized, dtype=np.uint8)

def resize_big_image(img: np.ndarray, max_side: float = 1024 * 2.3) -> np.ndarray:
    """
    Reduce the size of high resolution (Macbook problem) for paddleOCR
    Keep the original dim ratio
    Args:
        img: the original image (numpy array)
        max_side: the max length if longest side
    Returns:
        The resized image
    """
    h, w = img.shape[:2]
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = Image.fromarray(img)  # assumes img is already RGB
        img_resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img = np.array(img_resized)
    return np.ascontiguousarray(img, dtype=np.uint8)

def order_points_clockwise(pts):
    # pts: (4,2) — returns [tl, tr, br, bl]
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def apply_black_background(orig_img: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """
    Applies a black background to the original image, keeping only the area inside the polygon.
    Args:
        orig_img: the original image (numpy array)
        poly: the polygon (numpy array of shape (N, 2)) representing the receipt
    Returns:
        The image with a black background, keeping only the area inside the polygon.
    """
    poly_int = poly.astype(np.int32)
    mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly_int], 255)
    return cv2.bitwise_and(orig_img, orig_img, mask=mask)

def rotate_and_crop(img: np.ndarray, poly: np.ndarray) -> PreprocessingResult:
    poly_int = poly.astype(np.int32)
    rect = cv2.minAreaRect(poly_int)
    (w_rect, h_rect) = rect[1]
    angle = rect[2]

    if w_rect < h_rect:
        angle -= 90

    # normalize into (-45, 45] so this only ever deskews — never swaps orientation
    while angle <= -45:
        angle += 90
    while angle > 45:
        angle -= 90

    h_img, w_img = img.shape[:2]
    center = (w_img // 2, h_img // 2)

    M = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
    rotated = cv2.warpAffine(
        src=img, M=M, dsize=(w_img, h_img),
        flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))

    ones = np.ones((poly_int.shape[0], 1))
    poly_hom = np.hstack([poly_int, ones])
    rotated_poly = (M @ poly_hom.T).T.astype(np.int32)

    x, y, w, h = cv2.boundingRect(rotated_poly)
    x, y = max(0, x), max(0, y)
    if w <= 0 or h <= 0:
        return None
    final_cropped = rotated[y:y+h, x:x+w]
    return PreprocessingResult(image=final_cropped)

def get_rotate_crop_perspective(img: np.ndarray, poly: np.ndarray) -> PreprocessingResult:
    points = np.asarray(poly, dtype=np.float32).reshape(-1, 2)

    if points.shape[0] != 4:
        # Not a clean quadrilateral (e.g. detector returned a >4-point polygon) -
        # fall back to the minimum-area rectangle around it.
        rect = cv2.minAreaRect(points)
        points = cv2.boxPoints(rect).astype(np.float32)

    points = order_points_clockwise(points)
    points = points.reshape(4, 2).astype(np.float32)

    w = int(max(np.linalg.norm(points[0] - points[1]),
                 np.linalg.norm(points[2] - points[3])))
    h = int(max(np.linalg.norm(points[0] - points[3]),
                 np.linalg.norm(points[1] - points[2])))
    if w <= 0 or h <= 0:
        return None

    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(points, dst)
    crop = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # if crop.shape[0] / max(crop.shape[1], 1) >= 1.5:
    #     crop = np.rot90(crop, k=3)

    return PreprocessingResult(image=crop)

if __name__ == "__main__":
    from pathlib import Path
    from app.segmentation.model_loader import load_model
    from app.segmentation.infer import run_segmentation
    from PIL import Image
    ROOT_DIR = Path(__file__).parents[2]
    model = load_model(
        model_path=ROOT_DIR / "models/segmentation/last.onnx"
    )
    test_img = ROOT_DIR / "tests/test_data/seg.jpg"
    img = Image.open(test_img)

    data = run_segmentation(model, img, debug=False)

    orig_img = data.original_image
    for i, poly in enumerate(data.mask_polygons):
        img_with_black_bg = apply_black_background(orig_img, poly)
        cropped_result = rotate_and_crop(img_with_black_bg, poly)
        cv2.imwrite(f"receipt_{i}.jpg", cropped_result.image)


def add_white_padding(image, padding_px=100):
    # Add a white border around the original image
    padded_image = cv2.copyMakeBorder(
        image, 
        top=padding_px, 
        bottom=padding_px, 
        left=padding_px, 
        right=padding_px, 
        borderType=cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )
    return padded_image