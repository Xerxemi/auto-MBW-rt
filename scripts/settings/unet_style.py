#this script kind of really needs a rewrite but too lazy to fix
import os, time
import msgspec
import gradio as gr

from pygal.style import (
        DefaultStyle,
        DarkStyle,
        NeonStyle,
        DarkSolarizedStyle,
        LightSolarizedStyle,
        LightStyle,
        CleanStyle,
        RedBlueStyle,
        DarkColorizedStyle,
        LightColorizedStyle,
        TurquoiseStyle,
        LightGreenStyle,
        DarkGreenStyle,
        DarkGreenBlueStyle,
        BlueStyle
)

from modules import shared
from modules.scripts import basedir

from scripts.util.auto_mbw_rt_logger import logger_autombwrt as logger

# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
__location__ = basedir()
config_path = os.path.join(__location__, "settings", "unet_style.toml")

shared.PygalStyle = None

pygal_styles = {
    "DefaultStyle": DefaultStyle,
    "DarkStyle": DarkStyle,
    "NeonStyle": NeonStyle,
    "DarkSolarizedStyle": DarkSolarizedStyle,
    "LightSolarizedStyle": LightSolarizedStyle,
    "LightStyle": LightStyle,
    "CleanStyle": CleanStyle,
    "RedBlueStyle": RedBlueStyle,
    "DarkStyle": DarkStyle,
    "LightColorizedStyle": LightColorizedStyle,
    "TurquoiseStyle": TurquoiseStyle,
    "LightGreenStyle": LightGreenStyle,
    "DarkGreenStyle": DarkGreenStyle,
    "DarkGreenBlueStyle": DarkGreenBlueStyle,
    "BlueStyle": BlueStyle
}

class UnetVisualizer():
    def __init__(self):
        self.pygal_style = pygal_styles["DarkStyle"]
        self.show_labels = True
        self.unet_vis = None
        self.unet_vis_xml = None
    def set_pygal_style(self, pygal_style):
        self.pygal_style = pygal_styles[pygal_style]
    def set_show_labels(self, show_labels):
        self.show_labels = show_labels


def on_ui_tabs(main_block):
    if shared.UnetVisualizer is None:
        shared.UnetVisualizer = UnetVisualizer()

    with gr.Row(variant="panel"):
        btn_save_settings = gr.Button(value="Save Settings")
        btn_load_settings = gr.Button(value="Load Settings")
        btn_apply_settings = gr.Button(value="Apply Settings", variant="primary")
    with gr.Column(variant="panel"):
        dropdown_pygal_style = gr.Dropdown(label="Pygal Style", choices=[*pygal_styles.keys()], value="DarkStyle")
        chk_unet_show_labels = gr.Checkbox(label="Show Labels", value=False)
    html_info = gr.HTML()

    elements = {
        "dropdown_pygal_style": dropdown_pygal_style,
        "chk_unet_show_labels": chk_unet_show_labels
    }

    def save_settings(args):
        try:
            settings = {}
            for element_key in elements:
                settings.update({element_key: args[elements[element_key]]})
            config = msgspec.toml.encode(settings)
            with open(config_path, "wb") as f:
                f.write(config)
        except BaseException as e:
            logger.error("" + repr(e))
            return f"error: config failed to save to {config_path}.<br>"
        return f"success: config saved to {config_path}.<br>"

    btn_save_settings.click(fn=save_settings, inputs={*elements.values()}, outputs=[html_info])

    def apply_settings(args):
        try:
            shared.UnetVisualizer.set_pygal_style(args[elements["dropdown_pygal_style"]])
            shared.UnetVisualizer.set_show_labels(args[elements["chk_unet_show_labels"]])
        except BaseException as e:
            logger.error("" + repr(e))
            return f"error: config failed to apply from blocks.<br>"
        return f"success: config applied from blocks.<br>"

    btn_apply_settings.click(fn=apply_settings, inputs={*elements.values()}, outputs=[html_info])

    element_keys = [*elements.keys()]
    def load_settings():
        try:
            settings = msgspec.toml.decode(open(config_path, "rb").read())
            return_list = []
            for element_key in element_keys:
                return_list.append(gr.update(value=settings[element_key]))
        except BaseException as e:
            logger.error("" + repr(e))
            return [f"error: config failed to load from {config_path}.<br>"] + element_keys
        return [f"success: config loaded from {config_path}.<br>"] + return_list

    btn_load_settings.click(fn=load_settings, inputs=[], outputs=[html_info] + [elements[element_key] for element_key in element_keys])

    settings = msgspec.toml.decode(open(config_path, "rb").read())
    shared.UnetVisualizer.set_pygal_style(settings["dropdown_pygal_style"])
    shared.UnetVisualizer.set_show_labels(settings["chk_unet_show_labels"])
