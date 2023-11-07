import os  #, sys, gc, pdb
# import re
# import statistics
# import random
import datetime
import numpy as np
import gradio as gr

from modules import sd_models, shared
from modules.scripts import basedir
# try:
#     from modules import hashes
#     from modules.sd_models import CheckpointInfo
# except:
#     pass

# #dirty lora import
# import importlib
# lora = importlib.import_module("extensions-builtin.Lora.lora")
# #lora.available_lora_aliases

#util funcs
# from scripts.util.util_funcs import grouped
#history & presets
from scripts.util.history import MergeHistory
history = MergeHistory()

#main merger imports
from scripts.main.merger import handle_model_load, disable_injection, adjust_weights_score, save_checkpoint

#classifier plugins
from scripts.util.util_funcs import LazyLoader
#from importlib.machinery import SourceFileLoader

#required for LazyLoader to work
import scripts.classifiers

from scripts.util.auto_mbw_rt_logger import logger_autombwrt as logger

# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
__location__ = basedir()
classifiers_path = os.path.join(__location__, "scripts", "classifiers")
payloads_path = os.path.join(__location__, "payloads")
history_path = os.path.join(__location__, "csv", "history")

def refresh_plugins():
    global discovered_plugins
    discovered_plugins = {}
    for _, dirs, _ in os.walk(classifiers_path):
        for directory in dirs:
            directory_path = os.path.join(classifiers_path, directory)
            # exclude __pycache__ directory
            if directory_path.endswith('__pycache__'):
                continue
            for module in os.listdir(directory_path):
                if module.startswith('score') and module.endswith('.py'):
                    module_name = os.path.splitext(module)[0]
                    discovered_plugins.update({module_name: LazyLoader(module_name, globals(), f"scripts.classifiers.{directory}.{module_name}")})
    logger.info("discovered " + str(len(discovered_plugins)) + " classifier plugins.")

refresh_plugins()

#payloads
def refresh_payloads():
    global discovered_payloads
    discovered_payloads = []
    for _, _, files in os.walk(payloads_path):
        for f in files:
            if os.path.splitext(f)[1] in [".json", ".msgpack", ".toml", ".yaml"]:
                discovered_payloads.append(os.path.splitext(f)[0])
    discovered_payloads = [*set(discovered_payloads)]
    logger.info("discovered " + str(len(discovered_payloads)) + " payloads.")
    return [gr.update(choices=discovered_payloads), gr.update(choices=discovered_payloads), gr.update(choices=discovered_payloads)]

refresh_payloads()

#optimizers
from hyperactive import Hyperactive
from hyperactive.optimizers.strategies import CustomOptimizationStrategy
hyper = Hyperactive(n_processes=1, distribution="joblib")

from search_data_collector import SearchDataCollector

tally_types = ["Harmonic Mean", "Geometric Mean", "Arithmetic Mean", "Quadratic Mean", "Cubic Mean", "A/G Mean", "G/H Mean", "A/H Mean", "Median", "Min", "Max", "Mid-Range"]

def on_ui_tabs(main_block):
    import lora

    search_types = shared.HyperOptimizers.optimizers

    # Injecting my preset. "Trust me I am a AI / ML student".
    BAYESIAN_OPTIMIZER_INDEX = 0
    IMAGE_REWARD_INDEX = 0
    ARITHMETIC_MEAN_INDEX = 0
    try:
        BAYESIAN_OPTIMIZER_INDEX = [*search_types.keys()].index("BayesianOptimizer")
    except:
        logger.debug("Suggested BayesianOptimizer is not found.")
    try:
        IMAGE_REWARD_INDEX = [*discovered_plugins.keys()].index("score_image_reward")
    except:
        logger.debug("Suggested score_image_reward is not found.")        
    try:
        ARITHMETIC_MEAN_INDEX = tally_types.index("Arithmetic Mean")
    except:
        logger.debug("Suggested Arithmetic Mean is not found.")

    if shared.cmd_opts.no_gradio_queue:
        logger.info("--no-gradio-queue found in COMMANDLINE_ARGS | live gallery [disabled].\n")
    else:
        logger.info("--no-gradio-queue not found in COMMANDLINE_ARGS | live gallery [enabled].\n")

    display_images = None
    def get_display_images():
        return display_images

    #this one got moved to shared object for threading
    def get_display_unet():
        return shared.UnetVisualizer.unet_vis

    # (P1,P2,P3) to circumvent ui-config shenanigans when labels are identical
    with gr.Column():
        # The most important info of merge comes first
        with gr.Row(variant="panel"):
            dropdown_model_A = gr.Dropdown(label="Model A", choices=sd_models.checkpoint_tiles(), value=sd_models.checkpoint_tiles()[0])
            dropdown_model_B = gr.Dropdown(label="Model B", choices=sd_models.checkpoint_tiles(), value=sd_models.checkpoint_tiles()[0])
            txt_model_O = gr.Text(label="Output Model Name", elem_id="autombw_model_o")
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row(variant="panel"):
                    btn_do_mbw = gr.Button(value="Run Merge", variant="primary")
                    btn_reload_checkpoint = gr.Button(value="Reload Checkpoint")
                    btn_reload_payloads= gr.Button(value="Reload Payloads")
            with gr.Column():
                html_output_block_weight_info = gr.HTML()
        # P1 layout will be adjusted
        with gr.Row(variant="panel"):
            with gr.Accordion(label = "P1", open = True):
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            enabled_1 = gr.Checkbox(label="Enabled (P1)", value=True)
                            chk_save_output_files_1 = gr.Checkbox(label="Save Output Files", value=False)
                            sl_search_type_balance_1 = gr.Slider(label="Opt (A->B)", minimum=0, maximum=1, step=0.1, value=0)
                        with gr.Column():
                            payloads_1 = gr.Dropdown(label="Payloads", choices=discovered_payloads, multiselect=True, elem_id="autombw_payloads_1")
                        with gr.Column():
                            chk_enable_early_stop_1 = gr.Checkbox(label="Early Stop", value=True)
                            sl_n_iter_no_change_1 = gr.Slider(label="Iterations Tolerance", minimum=0, maximum=1000, step=1, value=27, interactive=True)           
                            sl_tol_abs_1 = gr.Slider(label="Absolute Tolerance", minimum=0.0, maximum=1.0, step=0.0001, value=0, interactive=True)
                            sl_tol_rel_1 = gr.Slider(label="Relative Tolerance", minimum=0.0, maximum=1.0, step=0.0001, value=0, interactive=True)
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                dropdown_search_type_A_1 = gr.Dropdown(label="Search Type A", choices=[*search_types.keys()], value=[*search_types.keys()][BAYESIAN_OPTIMIZER_INDEX], elem_id="autombw_search_type_A_1")
                                dropdown_search_type_B_1 = gr.Dropdown(label="Search Type B", choices=[*search_types.keys()], value=[*search_types.keys()][BAYESIAN_OPTIMIZER_INDEX], elem_id="autombw_search_type_B_1")
                            dropdown_classifiers_1 = gr.Dropdown(label='Image Classifier', choices=[*discovered_plugins.keys()], value=[*discovered_plugins.keys()][IMAGE_REWARD_INDEX], elem_id="autombw_classifiers_1")
                            dropdown_tally_type_1 = gr.Dropdown(label="Tally Type", choices=tally_types, value=tally_types[ARITHMETIC_MEAN_INDEX], elem_id="autombw_tally_type_1")
                        with gr.Column():
                            sl_search_iterations_1 = gr.Slider(label="Search Iterations", minimum=10, maximum=1000, step=1, value=270)
                            sl_search_time_1 = gr.Slider(label="Search Time (min)", minimum=1, maximum=10000, step=1, value=2880)
                            sl_test_grouping_1 = gr.Slider(label="Test Grouping", minimum=1, maximum=4, step=1, value=1)
                            sl_test_interval_1 = gr.Slider(label="Test Intervals", minimum=1, maximum=10000, step=1, value=20)
                        with gr.Column():
                            sl_initialize_grid_1 = gr.Slider(label="Initialize Points [grid]", minimum=0, maximum=50, step=1, value=4)
                            sl_initialize_vertices_1 = gr.Slider(label="Initialize Points [vertices]", minimum=0, maximum=50, step=1, value=4)
                            sl_initialize_random_1 = gr.Slider(label="Initialize Points [random]", minimum=0, maximum=50, step=1, value=2)
                            chk_warm_start_1 = gr.Checkbox(label="Warm Start", value=False)                    
        # P2 and P3 will be hidden becuase it is over complicated
        with gr.Row(variant="panel"):
            with gr.Accordion(label = "P2", open = False):
                with gr.Column():
                    with gr.Row(equal_height=True):
                        with gr.Column(min_width=150):
                            enabled_2 = gr.Checkbox(label="Enabled (P2)", value=False)
                            chk_save_output_files_2 = gr.Checkbox(label="Save Output Files", value=False)
                            sl_search_type_balance_2 = gr.Slider(label="Opt (A->B)", minimum=0, maximum=1, step=0.1, value=0)
                        with gr.Column(scale=32):
                            payloads_2 = gr.Dropdown(label="Payloads", choices=discovered_payloads, multiselect=True, elem_id="autombw_payloads_2")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                dropdown_search_type_A_2 = gr.Dropdown(label="Search Type A", choices=[*search_types.keys()], value=[*search_types.keys()][BAYESIAN_OPTIMIZER_INDEX], elem_id="autombw_search_type_A_2")
                                dropdown_search_type_B_2 = gr.Dropdown(label="Search Type B", choices=[*search_types.keys()], value=[*search_types.keys()][BAYESIAN_OPTIMIZER_INDEX], elem_id="autombw_search_type_B_2")
                            dropdown_classifiers_2 = gr.Dropdown(label='Image Classifier', choices=[*discovered_plugins.keys()], value=[*discovered_plugins.keys()][IMAGE_REWARD_INDEX], elem_id="autombw_classifiers_2")
                            dropdown_tally_type_2 = gr.Dropdown(label="Tally Type", choices=tally_types, value=tally_types[ARITHMETIC_MEAN_INDEX], elem_id="autombw_tally_type_2")
                        with gr.Column():
                            sl_search_iterations_2 = gr.Slider(label="Search Iterations", minimum=10, maximum=1000, step=1, value=270)
                            sl_search_time_2 = gr.Slider(label="Search Time (min)", minimum=1, maximum=10000, step=1, value=2880)
                            sl_test_grouping_2 = gr.Slider(label="Test Grouping", minimum=1, maximum=4, step=1, value=1)
                            sl_test_interval_2 = gr.Slider(label="Test Intervals", minimum=1, maximum=10000, step=1, value=20)
                        with gr.Column():
                            sl_initialize_grid_2 = gr.Slider(label="Initialize Points [grid]", minimum=0, maximum=50, step=1, value=4)
                            sl_initialize_vertices_2 = gr.Slider(label="Initialize Points [vertices]", minimum=0, maximum=50, step=1, value=4)
                            sl_initialize_random_2 = gr.Slider(label="Initialize Points [random]", minimum=0, maximum=50, step=1, value=2)
                            chk_warm_start_2 = gr.Checkbox(label="Warm Start", value=True)
                        with gr.Column():
                            chk_enable_early_stop_2 = gr.Checkbox(label="Early Stop", value=False)
                            sl_n_iter_no_change_2 = gr.Slider(label="Iterations Tolerance", minimum=0, maximum=1000, step=1, value=27, interactive=False)
                            sl_tol_abs_2 = gr.Slider(label="Absolute Tolerance", minimum=0.0, maximum=1.0, step=0.0001, value=0, interactive=False)
                            sl_tol_rel_2 = gr.Slider(label="Relative Tolerance", minimum=0.0, maximum=1.0, step=0.0001, value=0, interactive=False)
        with gr.Row(variant="panel"):
            with gr.Accordion(label = "P3", open = False):
                with gr.Column():
                    with gr.Row(equal_height=True):
                        with gr.Column(min_width=150):
                            enabled_3 = gr.Checkbox(label="Enabled (P3)", value=False)
                            chk_save_output_files_3 = gr.Checkbox(label="Save Output Files", value=False)
                            sl_search_type_balance_3 = gr.Slider(label="Opt (A->B)", minimum=0, maximum=1, step=0.1, value=0)
                        with gr.Column(scale=32):
                            payloads_3 = gr.Dropdown(label="Payloads", choices=discovered_payloads, multiselect=True, elem_id="autombw_payloads_3")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                dropdown_search_type_A_3 = gr.Dropdown(label="Search Type A", choices=[*search_types.keys()], value=[*search_types.keys()][BAYESIAN_OPTIMIZER_INDEX], elem_id="autombw_search_type_A_3")
                                dropdown_search_type_B_3 = gr.Dropdown(label="Search Type B", choices=[*search_types.keys()], value=[*search_types.keys()][BAYESIAN_OPTIMIZER_INDEX], elem_id="autombw_search_type_B_3")
                            dropdown_classifiers_3 = gr.Dropdown(label='Image Classifier', choices=[*discovered_plugins.keys()], value=[*discovered_plugins.keys()][IMAGE_REWARD_INDEX], elem_id="autombw_classifiers_3")
                            dropdown_tally_type_3 = gr.Dropdown(label="Tally Type", choices=tally_types, value=tally_types[ARITHMETIC_MEAN_INDEX], elem_id="autombw_tally_type_3")
                        with gr.Column():
                            sl_search_iterations_3 = gr.Slider(label="Search Iterations", minimum=10, maximum=1000, step=1, value=270)
                            sl_search_time_3 = gr.Slider(label="Search Time (min)", minimum=1, maximum=10000, step=1, value=2880)
                            sl_test_grouping_3 = gr.Slider(label="Test Grouping", minimum=1, maximum=4, step=1, value=1)
                            sl_test_interval_3 = gr.Slider(label="Test Intervals", minimum=1, maximum=10000, step=1, value=20)
                        with gr.Column():
                            sl_initialize_grid_3 = gr.Slider(label="Initialize Points [grid]", minimum=0, maximum=50, step=1, value=4)
                            sl_initialize_vertices_3 = gr.Slider(label="Initialize Points [vertices]", minimum=0, maximum=50, step=1, value=4)
                            sl_initialize_random_3 = gr.Slider(label="Initialize Points [random]", minimum=0, maximum=50, step=1, value=2)
                            chk_warm_start_3 = gr.Checkbox(label="Warm Start", value=True)
                        with gr.Column():
                            chk_enable_early_stop_3 = gr.Checkbox(label="Early Stop", value=False)
                            sl_n_iter_no_change_3 = gr.Slider(label="Iterations Tolerance", minimum=0, maximum=1000, step=1, value=27, interactive=False)
                            sl_tol_abs_3 = gr.Slider(label="Absolute Tolerance", minimum=0.0, maximum=1.0, step=0.0001, value=0, interactive=False)
                            sl_tol_rel_3 = gr.Slider(label="Relative Tolerance", minimum=0.0, maximum=1.0, step=0.0001, value=0, interactive=False)
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion(label = "Gallery [enabled]" if shared.cmd_opts.gradio_queue else "Gallery [disabled]", open = True if shared.cmd_opts.gradio_queue else False):
                            gallery_display_images = gr.Gallery(label="Gallery [enabled]", value=get_display_images, every=0.5, elem_id="autombw_gallery", columns=4, height=1024, container=True) if shared.cmd_opts.gradio_queue else gr.Gallery(label="Gallery [disabled]", elem_id="autombw_gallery", columns=4, height=2048)
                    with gr.Column():
                        txt_multi_merge = gr.Text(label="Multi Merge CMD", lines=6)
                        txt_block_weight = gr.Text(label="Weight Values (_nat)", placeholder="Put weight sets. float number x 27")
                        with gr.Row():
                            btn_apply_block_weight_from_txt = gr.Button(value="Apply Weights to SL")
                            btn_apply_block_weight_from_txt_cl = gr.Button(value="Apply Weights to CL", interactive=False)
                            btn_apply_block_weight_from_txt_cu = gr.Button(value="Apply Weights to CU", interactive=False)
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column():
                                        with gr.Row():
                                            sl_test_base = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="SL Base", value=0)
                                        with gr.Row():
                                            cl_test_base = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="CL Base", value=0, interactive=False)
                                            cu_test_base = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="CU Base", value=1, interactive=False)
                                        chk_enable_shared_memory = gr.Checkbox(label="Pass Shared Memory (requires consistent pass params)", value=False)
                                        experimental_range_checkbox = gr.Checkbox(label='Enable Experimental Range', value=False)
                                        with gr.Row(variant="panel"): 
                                            chk_enable_clamping = gr.Checkbox(label="Search Space Clamping", value=False)
                                            chk_TIME_EMBED = gr.Checkbox(label="TIME_EMBED", value=True, elem_id="autombw_time_embed", interactive=False)
                                            with gr.Column(variant="compact"):
                                                sl_TIME_EMBED = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="TIME_EMBED", value=0, elem_id="autombw_time_embed_sl", interactive=False)
                                                cl_TIME_EMBED = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="CL_TIME_EMBED", value=0, elem_id="autombw_time_embed_cl", interactive=False)
                                                cu_TIME_EMBED = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="CU_TIME_EMBED", value=1, elem_id="autombw_time_embed_cu", interactive=False)
                                        chk_enable_lora_merging = gr.Checkbox(label="LoRA Merging", value=False)
                                        chk_enable_multi_merge_twostep = gr.Checkbox(label="Multi Merge Twostep", value=False)
                            with gr.Column():
                                force_cpu_checkbox = gr.Checkbox(label='Force CPU (Max Precision)', value=True, interactive=False)
                                output_mode_radio = gr.Radio(label="Output Mode",choices=["Max Precision", "Runtime Snapshot"], value="Max Precision", type="value", interactive=False)
                                position_id_fix_radio = gr.Radio(label="Skip/Reset CLIP position_ids", choices=["Keep Original", "Fix"], value="Keep Original", type="value", interactive=True)
                                output_format_radio = gr.Radio(label="Output Format", choices=[".ckpt", ".safetensors"], value=".safetensors", type="value", interactive=True)
                                output_recipe_checkbox = gr.Checkbox(label="Output Recipe", value=True, interactive=True)
                with gr.Row():
                    with gr.Accordion(label = "UNET Visualizer [enabled]" if shared.cmd_opts.gradio_queue else "UNET Visualizer [disabled]", open = True if shared.cmd_opts.gradio_queue else False):
                        image_display_unet = gr.HTML(label="UNET Visualizer [enabled]", value=get_display_unet, every=0.5, elem_id="autombw_unet_vis") if shared.cmd_opts.gradio_queue else gr.HTML(label="UNET Visualizer [disabled]", elem_id="autombw_unet_vis")
        with gr.Accordion(label = "Warm Up Parameters (MBW, shared for P1 / P2 / P3)", open = False):
            with gr.Column():                  
                with gr.Row():
                    with gr.Column():
                        with gr.Row(variant="panel"):
                            chk_M_00 = gr.Checkbox(label="M00", value=True, elem_id="autombw_m00")
                            with gr.Column(variant="compact"):
                                sl_M_00 = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="M00", value=0, elem_id="autombw_m00_sl")
                                cl_M_00 = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="CL_M00", value=0, elem_id="autombw_m00_cl", interactive=False)
                                cu_M_00 = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="CU_M00", value=1, elem_id="autombw_m00_cu", interactive=False)   
                    with gr.Column():
                        with gr.Row(variant="panel"):
                            chk_OUT = gr.Checkbox(label="OUT", value=True, elem_id="autombw_out")
                            with gr.Column(variant="compact"):
                                sl_OUT = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="OUT", value=0, elem_id="autombw_out_sl")
                                cl_OUT = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="CL_OUT", value=0, elem_id="autombw_out_cl", interactive=False)
                                cu_OUT = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="CU_OUT", value=1, elem_id="autombw_out_cu", interactive=False)
                with gr.Row():
                    with gr.Column():
                        chks_in, sliders_in, clamp_lower_in, clamp_upper_in = [], [], [], []
                        for num in range(0, 12):
                            with gr.Row(variant="panel"):
                                chks_in.append(gr.Checkbox(label="IN" + format(num, "0>2"), value=True, elem_id="autombw_in" + format(num, "0>2")))
                                with gr.Column(variant="compact"):
                                    sliders_in.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="IN" + format(num, "0>2"), value=0, elem_id="autombw_in" + format(num, "0>2") + "_sl"))
                                    clamp_lower_in.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="CL_IN" + format(num, "0>2"), value=0, elem_id="autombw_in" + format(num, "0>2") + "_cl", interactive=False))
                                    clamp_upper_in.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="CU_IN" + format(num, "0>2"), value=1, elem_id="autombw_in" + format(num, "0>2") + "_cu", interactive=False))
                    with gr.Column():
                        chks_out, sliders_out, clamp_lower_out, clamp_upper_out = [], [], [], []
                        for num in reversed(range(0, 12)):
                            with gr.Row(variant="panel"):
                                chks_out.append(gr.Checkbox(label="OUT" + format(num, "0>2"), value=True, elem_id="autombw_out" + format(num, "0>2")))
                                with gr.Column(variant="compact"):
                                    sliders_out.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="OUT" + format(num, "0>2"), value=0, elem_id="autombw_out" + format(num, "0>2") + "_sl"))
                                    clamp_lower_out.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="CL_OUT" + format(num, "0>2"), value=0, elem_id="autombw_out" + format(num, "0>2") + "_cl", interactive=False))
                                    clamp_upper_out.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label="CU_OUT" + format(num, "0>2"), value=1, elem_id="autombw_out" + format(num, "0>2") + "_cu", interactive=False))
                        chks_out.reverse()
                        sliders_out.reverse()
                        clamp_lower_out.reverse()
                        clamp_upper_out.reverse()

    chks = chks_in + [chk_M_00] + [chk_OUT] + chks_out + [chk_TIME_EMBED]
    sliders = sliders_in + [sl_M_00] + [sl_OUT] + sliders_out + [sl_TIME_EMBED]
    clamp_lower = clamp_lower_in + [cl_M_00] + [cl_OUT] + clamp_lower_out + [cl_TIME_EMBED]
    clamp_upper = clamp_upper_in + [cu_M_00] + [cu_OUT] + clamp_upper_out + [cu_TIME_EMBED]

    lora_disabled = [0, 3, 6, 9, 10, 11, 13, 14, 15, 16, 26]

    params = {
        "dropdown_model_A": dropdown_model_A,
        "dropdown_model_B": dropdown_model_B,
        "txt_model_O": txt_model_O,
        "txt_multi_merge": txt_multi_merge,
        "chk_enable_shared_memory": chk_enable_shared_memory,
        "chk_enable_clamping": chk_enable_clamping,
        "chk_enable_lora_merging": chk_enable_lora_merging,
        "chk_enable_multi_merge_twostep": chk_enable_multi_merge_twostep,
        "force_cpu_checkbox": force_cpu_checkbox,
        "experimental_range_checkbox": experimental_range_checkbox,
        "output_mode_radio": output_mode_radio,
        "position_id_fix_radio": position_id_fix_radio,
        "output_format_radio": output_format_radio,
        "output_recipe_checkbox": output_recipe_checkbox
    }

    pass_params_1 = {
        "enabled": enabled_1,
        "payloads": payloads_1,
        "sl_search_type_balance": sl_search_type_balance_1,
        "dropdown_search_type_A": dropdown_search_type_A_1,
        "dropdown_search_type_B": dropdown_search_type_B_1,
        "dropdown_classifiers": dropdown_classifiers_1,
        "dropdown_tally_type": dropdown_tally_type_1,
        "sl_search_iterations": sl_search_iterations_1,
        "sl_search_time": sl_search_time_1,
        "sl_test_grouping": sl_test_grouping_1,
        "sl_test_interval": sl_test_interval_1,
        "sl_initialize_grid": sl_initialize_grid_1,
        "sl_initialize_vertices": sl_initialize_vertices_1,
        "sl_initialize_random": sl_initialize_random_1,
        "chk_warm_start": chk_warm_start_1,
        "chk_save_output_files": chk_save_output_files_1,
        "chk_enable_early_stop":  chk_enable_early_stop_1,
        "sl_n_iter_no_change": sl_n_iter_no_change_1,
        "sl_tol_abs": sl_tol_abs_1,
        "sl_tol_rel": sl_tol_rel_1
    }

    pass_params_2 = {
        "enabled": enabled_2,
        "payloads": payloads_2,
        "sl_search_type_balance": sl_search_type_balance_2,
        "dropdown_search_type_A": dropdown_search_type_A_2,
        "dropdown_search_type_B": dropdown_search_type_B_2,
        "dropdown_classifiers": dropdown_classifiers_2,
        "dropdown_tally_type": dropdown_tally_type_2,
        "sl_search_iterations": sl_search_iterations_2,
        "sl_search_time": sl_search_time_2,
        "sl_test_grouping": sl_test_grouping_2,
        "sl_test_interval": sl_test_interval_2,
        "sl_initialize_grid": sl_initialize_grid_2,
        "sl_initialize_vertices": sl_initialize_vertices_2,
        "sl_initialize_random": sl_initialize_random_2,
        "chk_warm_start": chk_warm_start_2,
        "chk_save_output_files": chk_save_output_files_2,
        "chk_enable_early_stop":  chk_enable_early_stop_2,
        "sl_n_iter_no_change": sl_n_iter_no_change_2,
        "sl_tol_abs": sl_tol_abs_2,
        "sl_tol_rel": sl_tol_rel_2
    }

    pass_params_3 = {
        "enabled": enabled_3,
        "payloads": payloads_3,
        "sl_search_type_balance": sl_search_type_balance_3,
        "dropdown_search_type_A": dropdown_search_type_A_3,
        "dropdown_search_type_B": dropdown_search_type_B_3,
        "dropdown_classifiers": dropdown_classifiers_3,
        "dropdown_tally_type": dropdown_tally_type_3,
        "sl_search_iterations": sl_search_iterations_3,
        "sl_search_time": sl_search_time_3,
        "sl_test_grouping": sl_test_grouping_3,
        "sl_test_interval": sl_test_interval_3,
        "sl_initialize_grid": sl_initialize_grid_3,
        "sl_initialize_vertices": sl_initialize_vertices_3,
        "sl_initialize_random": sl_initialize_random_3,
        "chk_warm_start": chk_warm_start_3,
        "chk_save_output_files": chk_save_output_files_3,
        "chk_enable_early_stop":  chk_enable_early_stop_3,
        "sl_n_iter_no_change": sl_n_iter_no_change_3,
        "sl_tol_abs": sl_tol_abs_3,
        "sl_tol_rel": sl_tol_rel_3
    }

    # main function
    def do_mbw(args):
        try:
            #initial parsing
            model_A = args[params["dropdown_model_A"]]
            model_B = args[params["dropdown_model_B"]]
            model_O = args[params["txt_model_O"]]
            multi_merge = args[params["txt_multi_merge"]]
            lora = args[params["chk_enable_lora_merging"]]
            multi_merge_twostep = args[params["chk_enable_multi_merge_twostep"]]

            logger.info("#### AutoMBW - V2 ####")

            #parsing multi merge txt block
            multi_model_A = []
            multi_model_B = []
            multi_model_O = []
            disable_singular_merge = False
            if multi_merge.strip() != "":
                for line in multi_merge.splitlines():
                    multi_model_A.append(line.split("+")[0].strip())
                    multi_model_B.append(line.split("+")[1].split("=")[0].strip())
                    multi_model_O.append(line.split("+")[1].split("=")[1].strip())
                if len(multi_model_A) == len(multi_model_B) == len(multi_model_O):
                    disable_singular_merge = True
                    logger.info("multi merge detected.")
                else:
                    logger.error("multi merge parse error.")
            if not disable_singular_merge:
                multi_model_A = [model_A]
                multi_model_B = [model_B]
                multi_model_O = [model_O.strip()]

            def lora_sanitize(weights, search_space=False):
                if lora:
                    if search_space:
                        disabled = [str(idx) for idx in lora_disabled]
                        zero = [0.0]
                    else:
                        disabled = lora_disabled
                        zero = 0
                    for idx in disabled:
                        weights[idx] = zero

            weights = []
            for interface in sliders:
                weights.append(float(args[interface]))
            cl_weights = []
            for interface in clamp_lower:
                cl_weights.append(float(args[interface]))
            cu_weights = []
            for interface in clamp_upper:
                cu_weights.append(float(args[interface]))
            #sanitize for lora (disabled weights)
            # for weight_set in [weights, cl_weights, cu_weights]:
            #     lora_sanitize(weight_set)

            # start testing stuff
            def hyper_score(localargs):
                nonlocal display_images

                grouping = localargs.pass_through["grouping"]
                tunables = localargs.pass_through["tunables"]
                testweights = localargs.pass_through["weights"].copy()
                for key in tunables:
                    for interval in range(grouping):
                        testweights[int(key)*grouping+interval] = localargs[key]

                _weights = ','.join([str(i) for i in testweights])
                logger.info("testweights: " + _weights)

                payloads = localargs.pass_through["payloads"]
                payload_paths = []
                for payload in payloads:
                    for payload_ext in [".json", ".msgpack", ".toml", ".yaml"]:
                        payload_path = os.path.join(payloads_path, payload + payload_ext)
                        if os.path.isfile(payload_path):
                            payload_paths.append(payload_path)
                classifier = discovered_plugins[localargs.pass_through["classifier"]]
                tally_type = localargs.pass_through["tally_type"]
                save_output_files = localargs.pass_through["save_output_files"]
                model_B = localargs.pass_through["model_B"]
                lora = localargs.pass_through["lora"]
                seedplus = localargs.pass_through["seedplus"]

                score, images = adjust_weights_score(payload_paths, classifier, tally_type, save_output_files, testweights, model_B, lora=lora, seedplus=seedplus)
                display_images = images

                return score

            passes = 0
            for pass_params in [pass_params_1, pass_params_2, pass_params_3]:
                if args[pass_params["enabled"]] and args[pass_params["payloads"]] != None and args[pass_params["payloads"]] != []:
                    passes = passes + 1
                else:
                    break

            for idx, (model_A, model_B, model_O) in enumerate(zip(multi_model_A, multi_model_B, multi_model_O)):

                #twostep seed, add one to seed on every other merge to stop merger from settling into a peak from a certain set of images
                #we can also use this value for more seed modifications in the future
                seedplus = 0 if idx % 2 == 0 or multi_merge_twostep == False else 1

                if model_O == "":
                    model_O = "autoMBW_" + os.path.splitext(model_A)[0] +"_" + os.path.splitext(model_B)[0]

                #required since a webui update caused webui to not load newly created checkpoints without a list (which is just also regeneration of the model list)
                sd_models.list_models()
                model_A = sd_models.get_closet_checkpoint_match(model_A).title

                logger.info("----------AUTOMERGE START----------")
                logger.info("modelA: " + str(model_A) + "")
                logger.info("modelB: " + str(model_B) + "")
                logger.info("modelO: " + str(model_O) + "")

                handle_model_load(model_A, model_B, args[params["force_cpu_checkbox"]], weights, lora=lora)

                memory_warm_start = None
                for current_pass in range(passes):
                    if current_pass == 0:
                        pass_params = pass_params_1
                    if current_pass == 1:
                        pass_params = pass_params_2
                    if current_pass == 2:
                        pass_params = pass_params_3

                    if args[pass_params["payloads"]] == None or args[pass_params["payloads"]] == []:
                        raise ValueError("autoMBW [error]: no payloads selected.")

                    (lower, upper) = (-1.0, 2.0) if args[params["experimental_range_checkbox"]] else (0.0, 1.0)

                    search_space = {}
                    idx_range = [*range(0, 27, args[pass_params["sl_test_grouping"]])]
                    for idx, interface in enumerate(chks):
                        if idx in idx_range:
                            if args[interface]:
                                if args[params["chk_enable_clamping"]]:
                                    search_space.update({str(idx): [*np.round(np.linspace(args[clamp_lower[idx]], args[clamp_upper[idx]], num=args[pass_params["sl_test_interval"]]+1), 8)]})
                                else:
                                    search_space.update({str(idx): [*np.round(np.linspace(lower, upper, num=args[pass_params["sl_test_interval"]]+1), 8)]})
                            else:
                                search_space.update({str(idx): [float(args[sliders[idx]])]})
                    # lora_sanitize(search_space, search_space=True)

                    warm_start = [{str(k): v for k, v in enumerate(weights)}] if args[pass_params["chk_warm_start"]] else []
                    early_stopping = {"n_iter_no_change": args[pass_params["sl_n_iter_no_change"]], "tol_abs": args[pass_params["sl_tol_abs"]], "tol_rel": args[pass_params["sl_tol_rel"]]} if args[pass_params["chk_enable_early_stop"]] else None

                    pass_through = {
                        "grouping": args[pass_params["sl_test_grouping"]],
                        "tunables": [*search_space.keys()],
                        "weights": weights,
                        "payloads": args[pass_params["payloads"]],
                        "classifier": args[pass_params["dropdown_classifiers"]],
                        "tally_type": args[pass_params["dropdown_tally_type"]],
                        "save_output_files": args[pass_params["chk_save_output_files"]],
                        "model_A": model_A,
                        "model_B": model_B,
                        "lora": lora,
                        "seedplus": seedplus
                    }

                    #optimizer strategy to combine 2 opts
                    #we set search_iterations min to 10 and step of search_type_balance to 0.1 since hyperactive/pandas has an odd tolerance to low iterations during use of dual optimizers
                    if args[pass_params["sl_search_type_balance"]] != 0 and args[pass_params["sl_search_type_balance"]] != 1:
                        opt_strat = CustomOptimizationStrategy()
                        opt_strat.add_optimizer(search_types[args[pass_params["dropdown_search_type_A"]]], duration=round(1-args[pass_params["sl_search_type_balance"]], 1))
                        opt_strat.add_optimizer(search_types[args[pass_params["dropdown_search_type_B"]]], duration=args[pass_params["sl_search_type_balance"]])
                    elif args[pass_params["sl_search_type_balance"]] == 0:
                        opt_strat = search_types[args[pass_params["dropdown_search_type_A"]]]
                    elif args[pass_params["sl_search_type_balance"]] == 1:
                        opt_strat = search_types[args[pass_params["dropdown_search_type_B"]]]

                    hyper.opt_pros.pop(0, None)
                    #memory=True here or else memory defaults to "share", spawning a mp.Manager() that goes rogue due to webui os._exit(0) on SIGINT
                    hyper.add_search(
                        hyper_score,
                        search_space,
                        optimizer=opt_strat,
                        n_iter=args[pass_params["sl_search_iterations"]],
                        n_jobs=1,
                        initialize={"grid": args[pass_params["sl_initialize_grid"]], "vertices": args[pass_params["sl_initialize_vertices"]], "random": args[pass_params["sl_initialize_random"]], "warm_start": warm_start} if args[pass_params["chk_warm_start"]] else { "random": 1 },
                        pass_through=pass_through,
                        early_stopping=early_stopping,
                        memory=True,
                        memory_warm_start=memory_warm_start
                    )

                    #run & time
                    hyper.run(args[pass_params["sl_search_time"]]*60)

                    best_para = hyper.best_para(hyper_score)
                    for key in best_para:
                        for interval in range(args[pass_params["sl_test_grouping"]]):
                            weights[int(key)*args[pass_params["sl_test_grouping"]]+interval] = best_para[key]

                    if args[params["chk_enable_shared_memory"]]:
                        memory_warm_start = hyper.search_data(hyper_score)

                    #search data save
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%I-%M%p-%S")
                    folder_name = model_A +" - " + model_B
                    folder_path = os.path.join(history_path, folder_name)
                    try:
                        os.makedirs(folder_path)
                    except FileExistsError:
                        pass
                    collector = SearchDataCollector(os.path.join(folder_path, f"{model_O}-{current_pass}-{timestamp}.csv"))
                    collector.save(hyper.search_data(hyper_score, times=True))

                    _weights = ','.join([str(i) for i in weights])
                    _cl_weights = ','.join([str(i) for i in cl_weights])
                    _cu_weights = ','.join([str(i) for i in cu_weights])

                    #history save
                    history.add_history(
                        current_pass,
                        model_A,
                        model_B,
                        model_O,
                        _weights,
                        _cl_weights,
                        _cu_weights,
                        args[pass_params["dropdown_search_type_A"]],
                        args[pass_params["dropdown_search_type_B"]],
                        args[pass_params["dropdown_classifiers"]],
                        args[pass_params["dropdown_tally_type"]],
                        args[pass_params["sl_initialize_grid"]],
                        args[pass_params["sl_initialize_vertices"]],
                        args[pass_params["sl_initialize_random"]],
                        args[pass_params["chk_warm_start"]]
                    )
                    history.write_history()

                save_checkpoint(args[params["output_mode_radio"]], args[params["position_id_fix_radio"]], args[params["output_format_radio"]], model_O, args[params["output_recipe_checkbox"]], weights, model_A, model_B, lora=lora)

            logger.info("merge completed.")
            return gr.update(value="merge completed.<br>")
        except:
            raise
        finally:
            if lora:
                logger.info("LoRA does not use injection - skipping injection disable.")
            else:
                disable_injection()
                logger.info("injection disabled (hopefully).")


    btn_do_mbw.click(
        fn=do_mbw,
        inputs={*[*params.values()], *[*pass_params_1.values()], *[*pass_params_2.values()], *[*pass_params_3.values()], *chks, *sliders, *clamp_lower, *clamp_upper},
        outputs=[html_output_block_weight_info]
    )

    def reload_checkpoint():
        sd_models.list_models()
        return [gr.update(choices=sd_models.checkpoint_tiles()), gr.update(choices=sd_models.checkpoint_tiles())]
    btn_reload_checkpoint.click(
        fn=reload_checkpoint,
        inputs=[],
        outputs=[dropdown_model_A, dropdown_model_B]
    )

    btn_reload_payloads.click(
        fn=refresh_payloads,
        inputs=[],
        outputs=[payloads_1, payloads_2, payloads_3]
    )

    def on_change_test_base(test_base):
        _list = [test_base] * 27
        return [gr.update(value=x, visible=True) for x in _list]
    sl_test_base.release(
        fn=on_change_test_base,
        inputs=[sl_test_base],
        outputs=sliders
    )
    cl_test_base.release(
        fn=on_change_test_base,
        inputs=[cl_test_base],
        outputs=clamp_lower
    )
    cu_test_base.release(
        fn=on_change_test_base,
        inputs=[cu_test_base],
        outputs=clamp_upper
    )

    def on_btn_apply_block_weight_from_txt(txt_block_weight):
        if not txt_block_weight or txt_block_weight == "":
            return [gr.update() for _ in range(27)]
        _list = [x.strip() for x in txt_block_weight.split(",")]
        if(len(_list) != 27):
            return [gr.update() for _ in range(27)]
        return [gr.update(value=x, visible=True) for x in _list]
    btn_apply_block_weight_from_txt.click(
        fn=on_btn_apply_block_weight_from_txt,
        inputs=[txt_block_weight],
        outputs=sliders
    )
    btn_apply_block_weight_from_txt_cl.click(
        fn=on_btn_apply_block_weight_from_txt,
        inputs=[txt_block_weight],
        outputs=clamp_lower
    )
    btn_apply_block_weight_from_txt_cu.click(
        fn=on_btn_apply_block_weight_from_txt,
        inputs=[txt_block_weight],
        outputs=clamp_upper
    )

    def update_slider_range(experimental_range_flag):
        if experimental_range_flag:
            return [gr.update(minimum=-1, maximum=2) for _ in sliders + clamp_lower + clamp_upper + [sl_test_base, cl_test_base, cu_test_base]]
        else:
            return [gr.update(minimum=0, maximum=1) for _ in sliders + clamp_lower + clamp_upper + [sl_test_base, cl_test_base, cu_test_base]]

    experimental_range_checkbox.change(fn=update_slider_range, inputs=[experimental_range_checkbox], outputs=sliders + clamp_lower + clamp_upper + [sl_test_base, cl_test_base, cu_test_base])

    def on_change_clamping(clamping_flag):
        if clamping_flag:
            return [gr.update(visible=True, interactive=True) for _ in clamp_lower + clamp_upper + [cl_test_base, cu_test_base, btn_apply_block_weight_from_txt_cl, btn_apply_block_weight_from_txt_cu, chk_TIME_EMBED, sl_TIME_EMBED]]
        else:
            return [gr.update(visible=False, interactive=False) for _ in clamp_lower + clamp_upper + [cl_test_base, cu_test_base, btn_apply_block_weight_from_txt_cl, btn_apply_block_weight_from_txt_cu, chk_TIME_EMBED, sl_TIME_EMBED]]

    def on_change_clamping_btn(clamping_flag):
        if clamping_flag:
            return [gr.update(visible=True) for _ in [btn_apply_block_weight_from_txt_cl, btn_apply_block_weight_from_txt_cu]]
        else:
            return [gr.update(visible=False) for _ in [btn_apply_block_weight_from_txt_cl, btn_apply_block_weight_from_txt_cu]]

    chk_enable_clamping.change(fn=on_change_clamping, inputs=[chk_enable_clamping], outputs=clamp_lower + clamp_upper + [cl_test_base, cu_test_base, chk_TIME_EMBED, sl_TIME_EMBED])
    chk_enable_clamping.change(fn=on_change_clamping_btn, inputs=[chk_enable_clamping], outputs=[btn_apply_block_weight_from_txt_cl, btn_apply_block_weight_from_txt_cu])

    def on_change_force_cpu(force_cpu_flag):
        if force_cpu_flag:
            return gr.update(choices=["Max Precision", "Runtime Snapshot"], value="Max Precision")
        else:
            return gr.update(choices=["Runtime Snapshot"], value="Runtime Snapshot")

    force_cpu_checkbox.change(fn=on_change_force_cpu, inputs=[force_cpu_checkbox], outputs=[output_mode_radio])

    def on_change_shared_memory(shared_memory_flag):
        if shared_memory_flag:
            return gr.update(value=1, interactive=False), gr.update(value=1, interactive=False), gr.update(value=1, interactive=False)
        else:
            return gr.update(value=1, interactive=True), gr.update(value=1, interactive=True), gr.update(value=1, interactive=True)

    chk_enable_shared_memory.change(fn=on_change_shared_memory, inputs=[chk_enable_shared_memory], outputs=[sl_test_grouping_1, sl_test_grouping_2, sl_test_grouping_3])

    def on_change_early_stop(early_stop_flag):
        if early_stop_flag:
            return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
        else:
            return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

    chk_enable_early_stop_1.change(fn=on_change_early_stop, inputs=[chk_enable_early_stop_1], outputs=[sl_n_iter_no_change_1, sl_tol_abs_1, sl_tol_rel_1])
    chk_enable_early_stop_2.change(fn=on_change_early_stop, inputs=[chk_enable_early_stop_2], outputs=[sl_n_iter_no_change_2, sl_tol_abs_2, sl_tol_rel_2])
    chk_enable_early_stop_3.change(fn=on_change_early_stop, inputs=[chk_enable_early_stop_3], outputs=[sl_n_iter_no_change_3, sl_tol_abs_3, sl_tol_rel_3])

    def on_change_lora_merging(lora_merging_flag):
        if lora_merging_flag:
            return gr.update(choices=[*lora.available_lora_aliases.keys()])
        else:
            return gr.update(choices=sd_models.checkpoint_tiles())

    lora_disable_opts = [force_cpu_checkbox, output_mode_radio, position_id_fix_radio, output_recipe_checkbox]
    lora_disable_chks = [chks[idx] for idx in lora_disabled]
    lora_disable_sliders = [sliders[idx] for idx in lora_disabled]

    def on_change_lora_merging_opts(lora_merging_flag):
        if lora_merging_flag:
            return [gr.update(interactive=False, visible=False) for _ in range(len(lora_disable_opts))]
        else:
            return [gr.update(interactive=True, visible=True) for _ in range(len(lora_disable_opts))]

    def on_change_lora_merging_chks(lora_merging_flag):
        if lora_merging_flag:
            return [gr.update(interactive=False, value=False) for _ in range(len(lora_disable_chks))]
        else:
            return [gr.update(interactive=True, value=True) for _ in range(len(lora_disable_chks))]

    def on_change_lora_merging_sliders(lora_merging_flag):
        if lora_merging_flag:
            return [gr.update(interactive=False, value=0) for _ in range(len(lora_disable_sliders))]
        else:
            return [gr.update(interactive=True, value=0) for _ in range(len(lora_disable_sliders))]

    chk_enable_lora_merging.change(fn=on_change_lora_merging, inputs=[chk_enable_lora_merging], outputs=[dropdown_model_B])
    chk_enable_lora_merging.change(fn=on_change_lora_merging_opts, inputs=[chk_enable_lora_merging], outputs=lora_disable_opts)
    chk_enable_lora_merging.change(fn=on_change_lora_merging_chks, inputs=[chk_enable_lora_merging], outputs=lora_disable_chks)
    chk_enable_lora_merging.change(fn=on_change_lora_merging_sliders, inputs=[chk_enable_lora_merging], outputs=lora_disable_sliders)

    # def on_change_search_type_balance(balance):
    #     if balance == 0:
    #         return gr.update(interactive=True), gr.update(interactive=False)
    #     elif balance == 1:
    #         return gr.update(interactive=False), gr.update(interactive=True)
    #     else:
    #         return gr.update(interactive=True), gr.update(interactive=True)
    #
    # sl_search_type_balance_1.release(fn=on_change_search_type_balance, inputs=[sl_search_type_balance_1], outputs=[dropdown_search_type_A_1, dropdown_search_type_B_1])
    # sl_search_type_balance_2.release(fn=on_change_search_type_balance, inputs=[sl_search_type_balance_2], outputs=[dropdown_search_type_A_2, dropdown_search_type_B_2])
    # sl_search_type_balance_3.release(fn=on_change_search_type_balance, inputs=[sl_search_type_balance_3], outputs=[dropdown_search_type_A_3, dropdown_search_type_B_3])

    #fix for gradio blank slider shenanigans
    main_block.load(fn=on_change_test_base, inputs=[sl_test_base], outputs=sliders)
