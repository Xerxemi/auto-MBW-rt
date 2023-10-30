import os
import torch
import safetensors
from transformers import CLIPModel, CLIPProcessor
from scripts.classifiers.cafe_aesthetic.aesthetic import judge

from scripts.util.auto_mbw_rt_logger import logger_autombwrt as logger

def score(image, prompt="", reverse=False):
    aesthetic, _, _ = judge(image)
    if reverse:
        logger.warning("Reverse scoring not supported with this classifier. Are you sure you want to continue?")
    return aesthetic["aesthetic"]

