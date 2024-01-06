import os
import torch
import math
import ImageReward as reward

from scripts.util.auto_mbw_rt_logger import logger_autombwrt as logger

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

model = None

def score(image, prompt="", reverse=False):
    #I think it is the culprit or "locked score".
    global model
    if model == None:
        logger.info("Loading ImageReward...")
        model = reward.load("ImageReward-v1.0")
    score_origin = model.score(prompt, image)
    #Why it can be identical?
    logger.debug(f"Image address: {image}")
    logger.debug(f"Raw model score: {score_origin}")
    if reverse:
        score_origin = score_origin*-1
    score = sigmoid(score_origin)
    return score
