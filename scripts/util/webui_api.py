import os, io
import requests
import msgspec
import base64
from PIL import Image

from modules.scripts import basedir

# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
__location__ = basedir()
config_path = os.path.join(__location__, "settings", "internal.toml")
url = msgspec.toml.decode(open(config_path, "rb").read())["url"]

def txt2img(**args):
    payload = {
    "enable_hr": args["enable_hr"],
    "denoising_strength": args["denoising_strength"],
    "firstphase_width": args["firstphase_width"],
    "firstphase_height": args["firstphase_height"],
    "hr_scale": args["hr_scale"],
    "hr_upscaler": args["hr_upscaler"],
    "hr_second_pass_steps": args["hr_second_pass_steps"],
    "hr_resize_x": args["hr_resize_x"],
    "hr_resize_y": args["hr_resize_y"],
    "prompt": args["prompt"],
    "seed": args["seed"],
    "sampler_name": args["sampler_name"],
    "batch_size": args["batch_size"],
    "n_iter": args["n_iter"],
    "steps": args["steps"],
    "cfg_scale": args["cfg_scale"],
    "width": args["width"],
    "height": args["height"],
    "restore_faces": args["restore_faces"],
    "tiling": args["tiling"],
    "negative_prompt": args["negative_prompt"],
    "do_not_save_samples": True,
    "do_not_save_grid": True,
    "save_images": False
    }

    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    print(response)
    r = response.json()

    images = []
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        images.append(image)

    return images

def refresh_models():
    requests.post(url=f'{url}/sdapi/v1/refresh-checkpoints')

def set_model(model):
    payload = {"sd_model_checkpoint": model}
    requests.post(url=f'{url}/sdapi/v1/options', json=payload)
