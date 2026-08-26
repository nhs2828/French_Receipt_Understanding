import onnxruntime as ort
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from paddleocr import PaddleOCR

# det_session = ort.InferenceSession(
#     "/Users/son/Documents/my_ORC/Receipt_Understanding/services/vision-service/models/detection/PP-OCRv6_medium_det/model.onnx", providers=['CPUExecutionProvider']
# )

def preprocess(image_path, limit_side_len=960):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # resize keeping aspect ratio, round to multiple of 32
    ratio = limit_side_len / max(h, w)
    resize_h, resize_w = int(h * ratio), int(w * ratio)
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)

    resized = cv2.resize(image, (resize_w, resize_h))

    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    img = resized.astype('float32') / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]

    return img, (h, w), (resize_h, resize_w)

def postprocess(pred, orig_size, resize_size, thresh=0.3, box_thresh=0.6, unclip_ratio=1.5):
    pred = pred[0, 0]  # probability map
    bitmap = (pred > thresh).astype(np.uint8)

    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h_orig, w_orig = orig_size
    h_resize, w_resize = resize_size

    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)

        # unclip (expand box slightly, DB-style)
        poly = Polygon(box)
        if poly.area == 0:
            continue
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance)[0])

        # scale back to original image size
        expanded[:, 0] = expanded[:, 0] * w_orig / w_resize
        expanded[:, 1] = expanded[:, 1] * h_orig / h_resize
        boxes.append(expanded.astype(int))

    return boxes

model = PaddleOCR(
    text_detection_model_dir="/Users/son/Documents/my_ORC/Receipt_Understanding/services/vision-service/models/detection/PP-OCRv6_medium_det",
    text_recognition_model_dir="/Users/son/Documents/my_ORC/Receipt_Understanding/services/vision-service/models/recognition/PP-OCRv6_medium_rec",
    text_detection_model_name="PP-OCRv6_medium_det",
    text_recognition_model_name="PP-OCRv6_medium_rec",
    engine="onnxruntime"   # or similar flag depending on version
)

image_path = '/Users/son/Documents/my_ORC/Receipt_Understanding/training/KIE/LayoutLM/data/FR_SROIE/train/img/image_1_0.jpg'
# input_blob, orig_size, resize_size = preprocess(image_path)

# input_name = det_session.get_inputs()[0].name  # confirm this is actually "x"
# outputs = det_session.run(None, {input_name: input_blob})

# boxes = postprocess(outputs[0], orig_size, resize_size)
# print(f"Found {len(boxes)} text boxes")
o = model.predict(image_path)
print(o)