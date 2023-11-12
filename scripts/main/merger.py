import os #, sys
from collections import deque
import threading
# import requests
import msgspec

import datetime
import statistics
import numpy as np
# import gradio as gr

from modules import shared #, devices
from modules.scripts import basedir

from scripts.util.webui_api import txt2img, refresh_models, set_model
from scripts.util.util_funcs import change_dir, extend_path
from scripts.util.draw_unet import DataPlot

from scripts.main.wildcards import CardDealer
dealer = CardDealer()

from scripts.main.pluslora import pluslora

from scripts.util.auto_mbw_rt_logger import logger_autombwrt as logger

# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
__location__ = basedir()
config_path = os.path.join(__location__, "settings", "internal.toml")
url = msgspec.toml.decode(open(config_path, "rb").read())["url"]

__extensions__ = os.path.dirname(__location__)
with extend_path(os.path.join(__extensions__, msgspec.toml.decode(open(config_path, "rb").read())["extension_counterpart"])):
    from scripts.runtime_block_merge import on_save_checkpoint

# model handling functions
def handle_model_load(modelA, modelB, force_cpu_checkbox, slALL, lora=False):
    global plot
    plot = DataPlot()
    #not sure if this is necessary
    refresh_models()
    set_model(modelA)
    if not lora:
        shared.UNBMSettingsInjector.weights = slALL
        shared.UNBMSettingsInjector.modelB = modelB
        load_flag = shared.UNetBManager.load_modelB(modelB, force_cpu_checkbox, slALL)
        if not load_flag:
            return None
        shared.UNBMSettingsInjector.enabled = True
    return modelA, modelB

def disable_injection():
    if shared.UNBMSettingsInjector.enabled:
        logger.info("Disable injection.")
        try: 
            shared.UNetBManager.restore_original_unet()
            shared.UNetBManager.unload_all()
            shared.UNBMSettingsInjector.enabled = False
        except Exception as e:
            logger.error(e)
            logger.error("Disable injection failed. Recommended to restart WebUI.")
    else:
        logger.info("Disable injection skipped: Injection is not activated.")

lora_disabled = [0, 3, 6, 9, 10, 11, 13, 14, 15, 16, 26]
# helper func for sl_NAT to lora weight conversion
def lora_conv(slALL):
    slALL_lora = slALL.copy()
    for idx in lora_disabled:
        slALL_lora[idx] = None
    slALL_lora = [weight for weight in slALL_lora if weight is not None]
    slALL_lora = deque(slALL_lora)
    slALL_lora.appendleft(0)
    slALL_lora = [*slALL_lora]
    return slALL_lora

# main merging function
def adjust_weights_score(payload_paths, classifier, tally_type, save_output_files, slALL, modelB, lora=False, seedplus=0):
    _weights_lora = ""
    if lora:
        _weights_lora = ','.join([str(i) for i in lora_conv(slALL)])
    else:
        shared.UNBMSettingsInjector.weights = slALL
    _weights = ','.join([str(i) for i in slALL])

    # threading for unet vis since pygal is slow as f**k
    def add_data(weights, score, style, show_labels, save_output_files=False, save_output_path=None):
        #change cwd path to fix stupid pygal interactions with webui style.css <-- THIS TOOK ME AN ENTIRE 8 HOURS TO DEBUG ARGGHH
        with change_dir("/"):
            shared.UnetVisualizer.unet_vis, xml = plot.add_data(weights, score, style, show_labels)
        if save_output_files:
            with open(save_output_path, "wb") as f: f.write(xml)

    images_prompt = []
    for payload_path in payload_paths:
        for args in dealer.wildcard_payload(payload_path):
            #we don't want lora to leak into the image-reward scoring prompt, though maybe adding modelB would be good?
            prompt = args["prompt"]
            if lora:
                args["prompt"] = args["prompt"] + f'\n<lora:{modelB}:1:{_weights_lora}>'
            #this is fine since webui_api layer discards all extra payload arguments it doesn't use
            # reverse =  args["reverse_scoring"]
            reverse = False
            #I feel dirty just doing this
            args["seed"] = args["seed"] + seedplus
            for image in txt2img(**args):
                images_prompt.append((image, prompt, reverse,))

    imagescores = []
    for (image, prompt, reverse) in images_prompt:
        score = classifier.score(image, prompt=prompt, reverse=reverse)
        imagescores.append(score)

    if tally_type == "Arithmetic Mean":
        testscore = statistics.mean(imagescores)
    elif tally_type == "Geometric Mean":
        testscore = statistics.geometric_mean(imagescores)
    elif tally_type == "Harmonic Mean":
        testscore = statistics.harmonic_mean(imagescores)
    elif tally_type == "Quadratic Mean":
        testscore = np.sqrt(np.mean(np.array(imagescores)**2))
    elif tally_type == "Cubic Mean":
        testscore = np.cbrt(np.mean(np.array(imagescores)**3))
    elif tally_type == "A/G Mean":
        testscore = (statistics.mean(imagescores)/statistics.geometric_mean(imagescores))*statistics.mean(imagescores)
    elif tally_type == "G/H Mean":
        testscore = (statistics.geometric_mean(imagescores)/statistics.harmonic_mean(imagescores))*statistics.mean(imagescores)
    elif tally_type == "A/H Mean":
        testscore = (statistics.mean(imagescores)/statistics.harmonic_mean(imagescores))*statistics.mean(imagescores)
    elif tally_type == "Median":
        testscore = statistics.median(imagescores)
    elif tally_type == "Min":
        testscore = min(imagescores)
    elif tally_type == "Max":
        testscore = max(imagescores)
    elif tally_type == "Mid-Range":
        testscore = (min(imagescores)+max(imagescores))/2

    logger.info("test score: " + str(testscore))

    if save_output_files:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%I-%M%p-%S")
        folder_name = f"{timestamp}"
        folder_path = os.path.join(shared.cmd_opts.data_dir, "logs", 'auto_mbw_output', folder_name)
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            pass
        for idx, (image, _, _) in enumerate(images_prompt):
            image.save(os.path.join(folder_path, f"{idx}.png"))

        with open(os.path.join(folder_path, "_000-weights.txt"), 'w') as f:
            f.write("Weights: ")
            f.write(_weights)
            f.write('\n')
            f.write("LoRA Weights: ")
            f.write(_weights_lora)
            f.write('\n')
            f.write("Test Score: ")
            f.write(str(testscore))
            f.write('\n')
            f.write("Prompt: ")
            f.write(str(prompt))
            f.write('\n')
            f.write("Reverse Scoring: ")
            f.write(str(reverse))

        t = threading.Thread(name="unet_vis", target=add_data, args=(slALL, testscore, shared.UnetVisualizer.pygal_style, shared.UnetVisualizer.show_labels), kwargs={"save_output_files": True, "save_output_path": os.path.join(folder_path, "unet_vis.svg")})
    else:
        t = threading.Thread(name="unet_vis", target=add_data, args=(slALL, testscore, shared.UnetVisualizer.pygal_style, shared.UnetVisualizer.show_labels))

    t.start()

    images = [image for (image, _, _) in images_prompt]
    return testscore, images

lora_blocks = [
    "BASE",
    "IN1", "IN2", "IN4", "IN5", "IN7", "IN8",
    "M00",
    "OUT3", "OUT4", "OUT5", "OUT6", "OUT7", "OUT8", "OUT9", "OUT10", "OUT11"
]

# final save function
def save_checkpoint(output_mode_radio, position_id_fix_radio, output_format_radio, save_checkpoint_name, output_recipe_checkbox, weights, modelA, modelB, lora=False):
    logger.info(f"Saving checkpoint to ${save_checkpoint_name}")
    if lora:
        _weights_lora = ','.join([str(i) for i in lora_conv(weights)])
        savesets = ["overwrite"]
        if "safetensors" in output_format_radio:
            savesets.append("safetensors")
        pluslora(f"{modelB}:1:CUSTOM", f"CUSTOM:{_weights_lora}", savesets, save_checkpoint_name, modelA, "float")
    else:
        # convert to standard from nat
        weights_s = weights.copy()
        out = weights_s.pop(13)
        time_embed = weights_s.pop(-1)
        weights_s.append(out)
        weights_s.append(time_embed)
        on_save_checkpoint(output_mode_radio, position_id_fix_radio, output_format_radio, save_checkpoint_name, output_recipe_checkbox, *weights, *weights_s)

