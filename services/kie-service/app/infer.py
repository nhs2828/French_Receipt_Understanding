import base64
import io
from PIL import Image
from loguru import logger

# from app.clients.vision_client import ReceiptResult
from vision_client import ReceiptResult
from app.model_loader import KIEPredictor
from app.schemas import KIEInstanceResult


def run_kie_on_instance(predictor: KIEPredictor, instance: ReceiptResult) -> KIEInstanceResult:
    image_bytes = base64.b64decode(instance.processed_image_b64)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    entities = predictor.predict(pil_image, instance.words, instance.boxes)

    return KIEInstanceResult(instance_id=instance.instance_id, entities=entities)


def run_kie_on_all_instances(predictor: KIEPredictor, instances: list[ReceiptResult]) -> list[KIEInstanceResult]:
    results = []
    for instance in instances:
        try:
            results.append(run_kie_on_instance(predictor, instance))
        except Exception as e:
            logger.error(f"KIE failed on instance {instance.instance_id}: {e}")
            results.append(KIEInstanceResult(instance_id=instance.instance_id, entities={}))
    return results