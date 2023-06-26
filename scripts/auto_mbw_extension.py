import os
import gradio as gr

from modules import script_callbacks, shared, extensions

from scripts.main import mbw as auto_mbw
from scripts.payload import create_payload
from scripts.settings import optimizers, unet_style
from scripts.settings.optimizers import HyperOptimizers
from scripts.settings.unet_style import UnetVisualizer

# UI callback
def on_ui_tabs():

    shared.HyperOptimizers = HyperOptimizers()
    shared.UnetVisualizer = UnetVisualizer()

    with gr.Blocks() as main_block:
        with gr.Tab("Auto MBW", elem_id="tab_auto_mbw"):
            auto_mbw.on_ui_tabs(main_block)
        with gr.Tab("Payload Creator", elem_id="tab_auto_mbw_payload"):
            create_payload.on_ui_tabs(main_block)
        with gr.Tab("Settings", elem_id="tab_auto_mbw_settings"):
            with gr.Tab("Optimizers"):
                optimizers.on_ui_tabs(main_block)
            with gr.Tab("UNET Visualizer"):
                unet_style.on_ui_tabs(main_block)

    # return required as (gradio_component, title, elem_id)
    return (main_block, "Auto MBW", "auto_mbw"),

# on_UI
script_callbacks.on_ui_tabs(on_ui_tabs)
