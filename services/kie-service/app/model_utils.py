"""
Shared utilities: load config, load model/processor, metrics evaluation, class-weighting,
and custom callback to save the processor alongside each checkpoint.
"""

import sys
import json
import yaml
import numpy as np
import torch
from pathlib import Path
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    LayoutXLMProcessor,
    LayoutLMv2ForTokenClassification,
)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_cache_dir(config, config_path):
    """Directory to store preprocessed data (OCR-parse + tokenize + align label).

    Prioritizes dataset.cache_dir declared in config; if not present, infers
    based on config file name (e.g., configs/sroie.yaml -> ./cache/sroie).
    Used jointly by prepare_data.py (writing) and train.py (reading).
    """
    dcfg = config.get("dataset", {})
    if dcfg.get("cache_dir"):
        return dcfg["cache_dir"]
    return f"./cache/{Path(config_path).stem}"


class _Tee:
    """Write simultaneously to multiple streams (e.g., print to terminal while logging to a file).

    Simulates a complete file object interface to maintain compatibility with libraries
    that inspect stream attributes (tqdm calls isatty(), some libraries call fileno()/encoding...).
    Terminal-specific attributes (isatty, fileno, encoding) are always inherited from the
    FIRST stream (typically the original terminal), rather than the log file stream.
    """

    def __init__(self, *streams):
        # streams[0] is ALWAYS the original terminal -- receives everything as-is (including progress bars).
        # streams[1:] are log files -- filters out progress-bar updates (\r without a corresponding \n)
        # to prevent log files from bloating with thousands of tqdm/datasets progress "frames".
        self.streams = streams

    def write(self, data):
        self.streams[0].write(data)
        self.streams[0].flush()

        for s in self.streams[1:]:
            if "\r" in data and "\n" not in data:
                # 1 frame update of progress bar (tqdm, datasets Generating split...) -- skip
                continue
            # If the chunk contains both \r and \n (a progress bar that just completed), keep only the portion after the final \r
            clean = data.split("\r")[-1] if "\r" in data else data
            if clean:
                s.write(clean)
                s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return self.streams[0].isatty()

    def fileno(self):
        return self.streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

    @property
    def closed(self):
        return self.streams[0].closed

    def writable(self):
        return True

    def readable(self):
        return False


def setup_logging(output_dir, log_filename="run.log"):
    """Duplicate all stdout/stderr (print statements, transformers/datasets logs, tqdm, etc.)
    to log_filename in output_dir while maintaining normal terminal output.

    Call this function AS EARLY AS POSSIBLE at the top of your script (before any other print calls)
    to avoid missing any log output.
    Returns the path to the log file for reference.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / log_filename
    log_file = open(log_path, "a", encoding="utf-8")

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"\n{'=' * 70}\n[LOG] Writing log to file: {log_path}\n{'=' * 70}")
    return log_path


# ---------------------------------------------------------------------------
# Architecture Registry: adding or changing models only requires specifying "architecture"
# in the .yaml config (e.g., "layoutlmv3" or "layoutxlm") -- NO NEED to modify this file
# or any other .py files. To support a new architecture, simply add an entry to the dict below.
# ---------------------------------------------------------------------------
ARCHITECTURES = {
    "layoutlmv3": {
        "processor_cls": LayoutLMv3Processor,
        "model_cls": LayoutLMv3ForTokenClassification,
    },
    "layoutxlm": {
        # LayoutXLM uses the same architecture as LayoutLMv2 (backbone ResNet-FPN,
        # unlike LayoutLMv3 which uses patch embedding like ViT) -- XLM-R tokenizer
        # handles Vietnamese better than LayoutLMv3-base.
        "processor_cls": LayoutXLMProcessor,
        "model_cls": LayoutLMv2ForTokenClassification,
    },
}


def load_model_and_processor(config, label_list, checkpoint=None):
    """checkpoint=None  -> Load original pretrained model (used when starting training).
    checkpoint="..." -> Load fine-tuned model from a specific checkpoint (used during eval/predict).

    Processor is always loaded from base_checkpoint in config because tokenizer/image-processor
    DOES NOT change during fine-tuning -- only the classifier head is trained.

    Architecture (LayoutLMv3 / LayoutXLM / ...) is automatically selected according to
    config["model"]["architecture"] -- changing models only requires updating the config.
    """
    model_cfg = config["model"]
    base_ckpt = model_cfg["base_checkpoint"]
    arch_name = model_cfg.get("architecture", "layoutlmv3")

    if arch_name not in ARCHITECTURES:
        raise ValueError(
            f"Architecture '{arch_name}' is not registered in ARCHITECTURES. "
            f"Supported architectures: {list(ARCHITECTURES.keys())}"
        )
    arch = ARCHITECTURES[arch_name]

    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    processor = arch["processor_cls"].from_pretrained(base_ckpt, apply_ocr=False)

    model_source = checkpoint if checkpoint else base_ckpt
    model = arch["model_cls"].from_pretrained(
        model_source,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model, processor