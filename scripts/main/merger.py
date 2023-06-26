import os, sys
import threading
import requests
import msgspec

import datetime
import statistics
import numpy as np
import gradio as gr

from modules import shared, devices
from modules.scripts import basedir

from scripts.util.webui_api import txt2img
from scripts.util.util_funcs import change_dir, extend_path
from scripts.util.draw_unet import DataPlot

from scripts.main.wildcards import CardDealer
dealer = CardDealer()

# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
__location__ = basedir()
config_path = os.path.join(__location__, "settings", "internal.toml")
url = msgspec.toml.decode(open(config_path, "rb").read())["url"]

__extensions__ = os.path.dirname(__location__)
with extend_path(os.path.join(__extensions__, msgspec.toml.decode(open(config_path, "rb").read())["extension_counterpart"])):
    from scripts.runtime_block_merge import on_save_checkpoint

# model handling functions
def handle_model_load(modelA, modelB, force_cpu_checkbox, slALL):
    global plot
    plot = DataPlot()
    payload = {"sd_model_checkpoint": modelA}
    print(requests.post(url=f'{url}/sdapi/v1/options', json=payload))
    shared.UNBMSettingsInjector.weights = slALL
    shared.UNBMSettingsInjector.modelB = modelB
    load_flag = shared.UNetBManager.load_modelB(modelB, force_cpu_checkbox, slALL)
    if load_flag:
        shared.UNBMSettingsInjector.enabled = True
        return modelA, modelB
    else:
        return None

def disable_injection():
    shared.UNetBManager.restore_original_unet()
    shared.UNetBManager.unload_all()
    shared.UNBMSettingsInjector.enabled = False

# main merging function
def adjust_weights_score(payload_paths, classifier, tally_type, save_output_files, slALL, modelB, lora=False):

    _weights = ','.join([str(i) for i in slALL])

    # threading for unet vis since pygal is slow as f**k
    def add_data(weights, score, style, show_labels, save_output_files=False, save_output_path=None):
        #change cwd path to fix stupid pygal interactions with webui style.css <-- THIS TOOK ME AN ENTIRE 8 HOURS TO DEBUG ARGGHH
        with change_dir("/"):
            shared.UnetVisualizer.unet_vis, xml = plot.add_data(weights, score, style, show_labels)
        if save_output_files:
            with open(save_output_path, "wb") as f: f.write(xml)

    if not lora:
        shared.UNBMSettingsInjector.weights = slALL
    images_prompt = []
    for payload_path in payload_paths:
        for args in dealer.wildcard_payload(payload_path):
            if lora:
                args["prompt"] = args["prompt"] + f'\n<lora:"{modelB}":1:{_weights}>'
            images_prompt = images_prompt + (txt2img(**args), args["prompt"],)

    imagescores = []
    for image, prompt in images_prompt:
        score = classifier.score(image, prompt=prompt)
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
    elif tally_type == "Median":
        testscore = statistics.median(imagescores)
    elif tally_type == "Min":
        testscore = min(imagescores)
    elif tally_type == "Max":
        testscore = max(imagescores)
    elif tally_type == "Mid-Range":
        testscore = (min(imagescores)+max(imagescores))/2

    print("\n test score: " + str(testscore))

    if save_output_files:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%I-%M%p-%S")
        folder_name = f"{timestamp}"
        folder_path = os.path.join(shared.cmd_opts.data_dir, "logs", 'auto_mbw_output', folder_name)
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            pass
        for idx, image in enumerate(images):
            image.save(os.path.join(folder_path, f"{idx}.png"))

        with open(os.path.join(folder_path, "_000-weights.txt"), 'w') as f:
            f.write("Weights: ")
            f.write(_weights)
            f.write('\n')
            f.write("Test Score: ")
            f.write(str(testscore))

        t = threading.Thread(name="unet_vis", target=add_data, args=(slALL, testscore, shared.UnetVisualizer.pygal_style, shared.UnetVisualizer.show_labels), kwargs={"save_output_files": True, "save_output_path": os.path.join(folder_path, f"unet_vis.svg")})
    else:
        t = threading.Thread(name="unet_vis", target=add_data, args=(slALL, testscore, shared.UnetVisualizer.pygal_style, shared.UnetVisualizer.show_labels))

    t.start()

    return testscore, images

# final save function
def save_checkpoint(output_mode_radio, position_id_fix_radio, output_format_radio, save_checkpoint_name, output_recipe_checkbox, weights):
    # convert to standard from nat
    weights_s = weights.copy()
    out = weights_s.pop(13)
    time_embed = weights_s.pop(-1)
    weights_s.append(out)
    weights_s.append(time_embed)
    on_save_checkpoint(output_mode_radio, position_id_fix_radio, output_format_radio, save_checkpoint_name, output_recipe_checkbox, *weights, *weights_s)

