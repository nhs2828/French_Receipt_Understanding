from functools import lru_cache
from paddleocr import DocPreprocessor

@lru_cache(maxsize=1)
def load_preprocessing_model(
    ori_model_path: str,
    rectify_model_path: str,
    params: tuple):
    dict_params = dict(params)
    model = DocPreprocessor(
        doc_orientation_classify_model_dir=ori_model_path,
        doc_unwarping_model_dir=rectify_model_path,
        **dict_params
    )
    return model