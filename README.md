# WebUI 1.6.0 fix on 231030 with install guide by 6DammK9

- *Only tested in Winodws a.k.a my machine.* I'm not [gradio](https://www.gradio.app/) / [webUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) expert therefore do not expect any auto / e2e solutions.

- **NO SUPPORT FOR aki / "秋葉" build.**

- Also I do not gruntee to have any decent test coverage.

## Install prerequisites

1. Install these extensions via "Extensions" > "Install from URL":

- [sd-webui-runtime-block-merge](https://github.com/Xynonners/sd-webui-runtime-block-merge)

- [sd-webui-lora-block-weight](https://github.com/hako-mikan/sd-webui-lora-block-weight)

2. Install `dynamicprompts` via [wheels from pypi](https://pypi.org/project/dynamicprompts/#files):

- Download the *.whl file (`dynamicprompts-0.29.0-py2.py3-none-any.whl`)

- Run in cmd: `"FULL_PATH_OF_YOUR_A1111_WEBUI\venv\Scripts\python.exe" -m pip install path_of_the_whl_file.whl --prefer-binary`

3. You may face "Premission denied" while moving extension from `tmp` to `extensions`: 

- Either `cd extensions` and then `git clone https_github_com_this_repo` and then restart WebUI 

- Or make a directory `auto-MBW-rt` directly in `tmp` then rerun the installation.

4. From [AutoMBW V1](https://github.com/Xerxemi/sdweb-auto-MBW), make sure your WebUI instance has API enabled as `--api` in `COMMANDLINE_ARGS`.

```bat
REM 2nd SD (7861) for 2nd GPU (1)
set COMMANDLINE_ARGS=--medvram --disable-safe-unpickle --deepdanbooru --xformers --no-half-vae --api --port=7861 --device-id=1
```

5. Install these extensions via "Extensions" > "Install from URL":

- [Obviously this branch.](https://github.com/6DammK9/auto-MBW-rt/tree/webui-160-update)

## Basic procedure

1. "Make payload". Treat it like "trigger words", or anything you like, or [testing dataset in AI/ML.](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets)

- A minimal payload (e.g. single 512x512 image) is suggested if you are using it for the first time, to make sure the code works. ~~programmer's life~~

2. "Set classifier". I like [BayesianOptimizer](https://nbviewer.org/github/SimonBlanke/hyperactive-tutorial/blob/main/notebooks/hyperactive_tutorial.ipynb) with [ImageReward](https://github.com/THUDM/ImageReward). 

3. "Search". For RTX 3090, it *requires around 60x time for each payload.* If the payload takes around 15 seconds to complete, it takes around 15 minutes. 
- Optimization part (on test score) takes only a few seconds to compelete. 26 parameters is easy, comparing to [860M for SD](https://huggingface.co/docs/diffusers/v0.5.1/en/api/pipelines/stable_diffusion). 
- **"Force CPU" is forced on.** I see `RuntimeError: expected device cuda:0 but got device cpu` if it is off ~~and it is a headache to trace and move all tensors.~~

## If you encounter errors

- Trust me. **Always reboot webUI first.** State control in WebUI (even python) is awful.

## Change Log

- Logger is added. Inspired from [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff) and [sd-webui-controlnet
](https://github.com/Mikubill/sd-webui-controlnet).

- Fix for multiple SD instandces. It reads `--port` instead of hardcoded `http://127.0.0.1:7860`.

## This is part of my research.

- Just a hobby. [If you are feared by tuning for numbers, try "averaging" by simply 0.5, 0.33, 0.25... for 20 models. It works.](https://github.com/6DammK9/nai-anime-pure-negative-prompt/tree/main/ch05).

----

# auto-MBW-rt | a.k.a V2-BETA
*NOTE: THIS IS IN BETA. NEWER COMMITS MAY BREAK OLDER ONES. FUNCTIONALITY NOT GUARANTEED.*

An automated (yes, that's right, **AUTOMATIC**) MBW extension for AUTO1111.

Rewritten from scratch (not a deviation) UI and code.

Old (V1) example models here: https://huggingface.co/Xynon/SD-Silicon

Old (V1) article here: https://medium.com/@media_97267/the-automated-stable-diffusion-checkpoint-merger-autombw-44f8dfd38871

----

Made by both Xynon#7407 and Xerxemi#6423.

----

Big thanks to bbc-mc for the original codebase and the start of this merge paradigm. 

You can find it here: https://github.com/bbc-mc/sdweb-merge-block-weighted-gui

**MERGING BACKEND**: Huge thanks to ashen

https://github.com/ashen-sensored/sd-webui-runtime-block-merge

**LORA BACKEND**: Huge thanks to hako-mikan

https://github.com/hako-mikan/sd-webui-lora-block-weight

**LORA BACKEND (SOLID)**: Huge thanks to hako-mikan

https://github.com/hako-mikan/sd-webui-supermerger

**OPTIMIZER LIB**: Massive thanks to SimonBlanke

https://github.com/SimonBlanke/Hyperactive

----

Wiki/Documentation

*coming soon<sup>TM</sup>*
