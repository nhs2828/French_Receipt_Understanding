"""
The module to post-process the result of text extraction module
"""
import numpy as np


def merge_same_line_boxes(words, boxes, y_tolerance=6, x_gap_max=40):
    """Merges adjacent detection boxes that sit on the same visual line

    and are close enough horizontally to plausibly be one phrase.

    words: list of strings boxes: list or 3D numpy array of shape (N, 4, 2)
    where each box is a numpy array of shape (4, 2):
           [[x0, y0],   # 0: Top-Left
            [x1, y1],   # 1: Top-Right
            [x2, y2],   # 2: Bottom-Right
            [x3, y3]]   # 3: Bottom-Left
    """
    if len(words) == 0 or len(boxes) == 0:
        return [], []

    items = list(zip(words, boxes))
    # Sort by Top-Left y (box[0, 1]), then Top-Left x (box[0, 0])
    #items.sort(key=lambda t: (t[1][0, 1], t[1][0, 0]))

    merged = []
    for word, box in items:
        if merged:
            pw, pb = merged[-1]
            # Compare Top-Left y [0, 1] and Bottom-Left y [3, 1]
            same_line = (
                abs(pb[0, 1] - box[0, 1]) <= y_tolerance
                and abs(pb[3, 1] - box[3, 1]) <= y_tolerance
            )
            # Distance between previous Top-Right x [1, 0] and current Top-Left x [0, 0]
            gap = box[0, 0] - pb[1, 0]

            if same_line and abs(gap) <= x_gap_max:
                # Combine left points of previous box with right points of current box
                merged_box = np.array([pb[0], box[1], box[2], pb[3]])
                merged[-1] = (pw + " " + word, merged_box)
                continue

        # Ensure box is retained as np.ndarray
        merged.append((word, np.asarray(box)))

    return [m[0] for m in merged], [m[1] for m in merged]

def process_text_extraction_results(
        img: np.ndarray, 
        result: dict,
        y_tolerance: int,
        x_gap_max: int,
        data_prep: bool=False):
    """
    Processes the raw results from the text extraction model to map coordinates back to the original image size.
    
    Args:
        img: the original input image
        result: raw result from the text extraction model
        y_tolerance: max shift in height to be considered as same line
        x_gap_max: gap between boxes to considered to be merged
        data_prep: disable coor normalization for x0, y0, .. x3, y3
    Returns:
        A list of dictionaries containing the text and the corresponding bounding box coordinates in the original image size.
    """
    words, boxes = [], []
    h, w = img.shape[:2]
    texts = result["rec_texts"]
    polys = result["rec_polys"]


    texts, polys = merge_same_line_boxes(texts, polys, y_tolerance=y_tolerance, x_gap_max=x_gap_max)


    for text, poly in zip(texts, polys):
        xs = poly[:, 0]
        ys = poly[:, 1]
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        # normalize because LayoutLMv3 train and inf on 1000 max
        nx0 = int(1000.0*x0/w)
        ny0 = int(1000.0*y0/h)
        nx1 = int(1000.0*x1/w)
        ny1 = int(1000.0*y1/h)

        nx0 = max(0, min(1000, nx0))
        ny0 = max(0, min(1000, ny0))
        nx1 = max(0, min(1000, nx1))
        ny1 = max(0, min(1000, ny1))

        boxes.append([nx0, ny0, nx1, ny1] if not data_prep else poly.flatten())
        words.append(text)
    return words, boxes
