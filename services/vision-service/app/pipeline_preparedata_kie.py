import os
import shutil
import glob
from pathlib import Path
import argparse
from loguru import logger
from tqdm import tqdm
import cv2

from app.segmentation import (
    load_model as load_seg_model,
    run_segmentation
)
from app.preprocessing import (
    apply_black_background,
    rotate_and_crop
)
from app.text_extraction import (
    load_model as load_text_ext_model,
    run_text_extraction,
    process_text_extraction_results
)
from app.config import (
    segmentation_params, 
    paddle_ocr_params
)
from app.utils import ndarray_to_jpg, draw_polys


def merge_same_line_boxes(words, boxes, y_tolerance=6, x_gap_max=40):
    """
    Merges adjacent detection boxes that sit on the same visual line
    and are close enough horizontally to plausibly be one phrase.
    words/boxes: parallel lists, boxes as [x0, y0, x1, y1] (pixel coords, pre-normalization)
    """
    items = list(zip(words, boxes))
    items.sort(key=lambda t: (t[1][1], t[1][0]))  # sort by y, then x

    merged = []
    for word, box in items:
        if merged:
            pw, pb = merged[-1]
            same_line = abs(pb[1] - box[1]) <= y_tolerance and abs(pb[3] - box[3]) <= y_tolerance
            gap = box[0] - pb[2]
            if same_line and 0 <= gap <= x_gap_max:
                merged[-1] = (
                    pw + " " + word,
                    [min(pb[0], box[0]), min(pb[1], box[1]),
                     max(pb[2], box[2]), max(pb[3], box[3])]
                )
                continue
        merged.append((word, box))

    return [m[0] for m in merged], [m[1] for m in merged]

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--data_prep", type=bool, required=True)
    parser.add_argument("--x_gap_max", type=int, default=35)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    ROOT_DIR = Path(__file__).parents[1]
    DEBUG = args.debug

    input_path = Path(args.input_data)
    input_list = []
    if input_path.is_file():
        logger.info(f"Run for an image {input_path}")
        input_list.append(input_path)
    elif input_path.is_dir():
        logger.info(f"Run for images in dir {str(input_path)}")
        input_list.extend(glob.glob(f"{input_path}/*.jpg"))
    input_list.sort()

    data_prep = args.data_prep
    if data_prep:
        OUTPUT_DIR_PREP = "output_data_prep_"
        logger.info("Create prep data folder")
        # Delete if directory exists
        if os.path.exists(OUTPUT_DIR_PREP):
            shutil.rmtree(OUTPUT_DIR_PREP)
        # Create fresh directory
        os.makedirs(OUTPUT_DIR_PREP, exist_ok=True)
        list_sub_dir = ['image', 'box']
        DICT_SUB_DIR_PATH = {}
        for sub_dir in list_sub_dir:
            s_path = f"{OUTPUT_DIR_PREP}/{sub_dir}"
            os.makedirs(f"{OUTPUT_DIR_PREP}/{sub_dir}", exist_ok=False)
            DICT_SUB_DIR_PATH[sub_dir] = s_path

    logger.info("Get models")
    logger.info("Get segmentation model")
    seg_model = load_seg_model(
        model_path=ROOT_DIR / "models/segmentation/last.onnx"
    )
    logger.info("Get text-extraction model")
    params_text_ext = paddle_ocr_params.model_dump()
    params_text_ext_tuple = tuple(sorted(params_text_ext.items()))
    text_ext_model = load_text_ext_model(
        det_model_path=Path(ROOT_DIR / f"models/detection/{params_text_ext['text_detection_model_name']}"),
        rec_model_path=Path(ROOT_DIR / f"models/recognition/{params_text_ext['text_recognition_model_name']}"),
        params=params_text_ext_tuple
    )

    for img in tqdm(input_list):
        file_path = Path(img)
        data = run_segmentation(seg_model, img, params=segmentation_params, debug=False)
        orig_img = data.original_image
        for i, poly in enumerate(data.mask_polygons):
            img_with_black_bg = apply_black_background(orig_img, poly)
            #cropped_result = get_rotate_crop_image(img_with_black_bg, poly)
            cropped_result = rotate_and_crop(img_with_black_bg, poly)
            #ndarray_to_jpg(img_np=cropped_result.image, img_path=f"{DICT_SUB_DIR_PATH['image']}/{file_path.stem}_{i}{file_path.suffix}")
            cropped_result.image = add_white_padding(cropped_result.image, padding_px=70)
            result = run_text_extraction(
                model_wrapper=text_ext_model, 
                image=cropped_result.image)
            
            words, boxes = process_text_extraction_results(
                img=cropped_result.image, 
                result=result, 
                data_prep=True,
                y_tolerance=6,
                x_gap_max=args.x_gap_max)
            ndarray_to_jpg(img_np=result['doc_preprocessor_res']['output_img'], img_path=f"{DICT_SUB_DIR_PATH['image']}/{file_path.stem}_{i}{file_path.suffix}")
            if DEBUG:
                result.save_to_img(save_path=f"{DICT_SUB_DIR_PATH['image']}/")
                ndarray_to_jpg(img_np=result['doc_preprocessor_res']['output_img'], img_path=f"{DICT_SUB_DIR_PATH['image']}/{file_path.stem}_{i}_processed{file_path.suffix}")
                #boxes result["rec_polys"]
                im = draw_polys(result['doc_preprocessor_res']['output_img'], polys=boxes)
                ndarray_to_jpg(im, img_path=f"{DICT_SUB_DIR_PATH['image']}/{file_path.stem}_{i}_draw{file_path.suffix}")

            output_path = f"{DICT_SUB_DIR_PATH['box']}/{file_path.stem}_{i}.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                for word, box in zip(words, boxes):
                    x0, y0, x1, y1, x2, y2, x3, y3 = box
                    f.write(f"{x0},{y0},{x1},{y1},{x2},{y2},{x3},{y3},{word}\n")


