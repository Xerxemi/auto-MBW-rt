import os
import torch
import safetensors
from transformers import CLIPModel, CLIPProcessor
from scripts.classifiers.cafe_aesthetic.aesthetic import judge

def score(image, prompt="", reverse=False):
    _, _, waifu = judge(image)
    if reverse:
        print("Reverse scoring not supported with this classifier. Are you sure you want to continue?")
    return waifu["waifu"]

