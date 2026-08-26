import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass

@dataclass
class PreprocessingResult:
    image: np.ndarray


class ProcessingService:

    @staticmethod
    def merge_same_line_boxes(words, boxes, y_tolerance=6, x_gap_max=40):
        """Merges adjacent detection boxes that sit on the same visual line

        and are close enough horizontally to plausibly be one phrase.
        """
        if len(words) == 0 or len(boxes) == 0:
            return [], []

        items = list(zip(words, boxes))

        merged = []
        for word, box in items:
            if merged:
                pw, pb = merged[-1]
                same_line = (
                    abs(pb[0, 1] - box[0, 1]) <= y_tolerance
                    and abs(pb[3, 1] - box[3, 1]) <= y_tolerance
                )
                gap = box[0, 0] - pb[1, 0]

                if same_line and abs(gap) <= x_gap_max:
                    merged_box = np.array([pb[0], box[1], box[2], pb[3]])
                    merged[-1] = (pw + " " + word, merged_box)
                    continue

            merged.append((word, np.asarray(box)))

        return [m[0] for m in merged], [m[1] for m in merged]

    @staticmethod
    def process_text_extraction_results(
        img: np.ndarray,
        result: dict,
        y_tolerance: int,
        x_gap_max: int,
        data_prep: bool = False,
    ):
        """Processes the raw results from the text extraction model."""
        words, boxes = [], []
        h, w = img.shape[:2]
        texts = result["rec_texts"]
        polys = result["rec_polys"]

        # Called via class name since 'self' does not exist in @staticmethod
        texts, polys = ProcessingService.merge_same_line_boxes(
            texts, polys, y_tolerance=y_tolerance, x_gap_max=x_gap_max
        )

        for text, poly in zip(texts, polys):
            xs = poly[:, 0]
            ys = poly[:, 1]
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()

            nx0 = int(1000.0 * x0 / w)
            ny0 = int(1000.0 * y0 / h)
            nx1 = int(1000.0 * x1 / w)
            ny1 = int(1000.0 * y1 / h)

            nx0 = max(0, min(1000, nx0))
            ny0 = max(0, min(1000, ny0))
            nx1 = max(0, min(1000, nx1))
            ny1 = max(0, min(1000, ny1))

            boxes.append(
                [nx0, ny0, nx1, ny1] if not data_prep else poly.flatten()
            )
            words.append(text)
        return words, boxes

    @staticmethod
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

    @staticmethod
    def order_points_clockwise(pts):
        # pts: (4,2) — returns [tl, tr, br, bl]
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def get_rotate_crop_perspective(img: np.ndarray, poly: np.ndarray) -> PreprocessingResult:
        points = np.asarray(poly, dtype=np.float32).reshape(-1, 2)

        if points.shape[0] != 4:
            # Not a clean quadrilateral (e.g. detector returned a >4-point polygon) -
            # fall back to the minimum-area rectangle around it.
            rect = cv2.minAreaRect(points)
            points = cv2.boxPoints(rect).astype(np.float32)

        points = ProcessingService.order_points_clockwise(points)
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

    @staticmethod
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