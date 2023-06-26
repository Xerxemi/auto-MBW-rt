import launch

#package dependencies
if not launch.is_installed("memory_tempfile"):
    launch.run_pip("install memory-tempfile", "requirements for autoMBW [memory-tempfile]")

if not launch.is_installed("hyperactive"):
    launch.run_pip("install hyperactive", "requirements for autoMBW [hyperactive]")

if not launch.is_installed("search_data_collector"):
    launch.run_pip("install search-data-collector", "requirements for autoMBW [search-data-collector]")

if not launch.is_installed("tomli_w"):
    launch.run_pip("install tomli-w", "requirements for autoMBW [tomli-w]")

if not launch.is_installed("msgspec"):
    launch.run_pip("install msgspec", "requirements for autoMBW [msgspec]")

if not launch.is_installed("pyexcel"):
    launch.run_pip("install pyexcel", "requirements for autoMBW [pyexcel]")

if not launch.is_installed("pygal"):
    launch.run_pip("install pygal", "requirements for autoMBW [pygal]")

if not launch.is_installed("cairosvg"):
    launch.run_pip("install cairosvg", "requirements for autoMBW [cairosvg]")

if not launch.is_installed("image_reward"):
    launch.run_pip("install git+https://github.com/Oyaxira/ImageReward.git@main", "requirements for autoMBW [image-reward]")

if not launch.is_installed("dynamicprompts"):
    launch.run_pip("install 'dynamicprompts'", "requirements for autoMBW [dynamicprompts]")

# else:
#     try:
#         from dynamicprompts.generators.magicprompt import MagicPromptGenerator
#         from dynamicprompts.generator.attentiongenerator import AttentionGenerator
#     except ImportError:
#         launch.run_pip("install 'dynamicprompts[magicprompt,attentiongrabber]'", "requirements for autoMBW [dynamicprompts]")

#extension dependencies
import os
import msgspec

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
config_path = os.path.join(__location__, "settings", "internal.toml")

__extensions__ = os.path.dirname(__location__)
script_path = os.path.join(__extensions__, msgspec.toml.decode(open(config_path, "rb").read())["extension_counterpart"], "scripts", "runtime_block_merge.py")

if os.path.isfile(script_path):
    if "SettingsInjector" in open(script_path, "r").read():
        pass
    else:
        raise Exception("autoMBW [error]: runtime-block-merge extension missing injector. MERGING WILL NOT WORK - https://github.com/Xynonners/sd-webui-runtime-block-merge")
else:
    raise Exception("autoMBW [error]: runtime-block-merge extension not found. MERGING WILL NOT WORK - https://github.com/Xynonners/sd-webui-runtime-block-merge")
