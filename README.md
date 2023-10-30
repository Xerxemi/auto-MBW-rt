# WebUI 1.6.0 fix on 231030 with install guide by 6DammK9

- *Only tested in Winodws a.k.a my machine.* I'm not gradio / webUI expert therefore do not expect any auto / e2e solutions.

- **NO SUPPORT FOR aki / "秋葉" build.**

- Also I do not gruntee to have any decent test coverage.

## "No thanks", install the required extensions first

1. Install via "Extensions" > "Install from URL":

- [sd-webui-runtime-block-merge](https://github.com/Xynonners/sd-webui-runtime-block-merge)

- [sd-webui-lora-block-weight](https://github.com/hako-mikan/sd-webui-lora-block-weight)

2. Install 'dynamicprompts' via [wheels from pypi](https://pypi.org/project/dynamicprompts/#files):

- Download the *.whl file (`dynamicprompts-0.29.0-py2.py3-none-any.whl`)

- Run in cmd: `"FULL_PATH_OF_YOUR_A1111_WEBUI\venv\Scripts\python.exe" -m pip install path_of_the_whl_file.whl --prefer-binary`

3. You may face "Premission denied" while moving extension from `tmp` to `extensions`: 

- Either `cd extensions` and then `git clone https_github_com_this_repo` and then restart WebUI 

- Or make a directory `auto-MBW-rt` directly in `tmp` then rerun the installation.

4. Now the extension starts with warning messages.

## Fixing warning messages

- Install from branch `webui-160-update`.

- Meanwhile I've added a logger.

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
