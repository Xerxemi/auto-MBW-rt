#this file needs a rewrite
import os, time
import msgspec
import gradio as gr

from hyperactive.optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
    RandomSearchOptimizer,
    GridSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    PowellsMethod,
    PatternSearch,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    EvolutionStrategyOptimizer,
    LipschitzOptimizer,
    DirectAlgorithm,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer
)

from modules import shared
from modules.scripts import basedir

# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
__location__ = basedir()
config_path = os.path.join(__location__, "settings", "optimizers.toml")

shared.HyperOptimizers = None

settings_list = {
    "HillClimbingOptimizer": {"epsilon": 0.03, "distribution": ["normal", "laplace", "logistic", "gumbel"], "n_neighbours": 3},
    "StochasticHillClimbingOptimizer": {"epsilon": 0.03, "distribution": ["normal", "laplace", "logistic", "gumbel"], "n_neighbours": 3, "p_accept": 0.1},
    "RepulsingHillClimbingOptimizer": {"epsilon": 0.03, "distribution": ["normal", "laplace", "logistic", "gumbel"], "n_neighbours": 3, "repulsion_factor": 5.0},
    "SimulatedAnnealingOptimizer": {"epsilon": 0.03, "distribution": ["normal", "laplace", "logistic", "gumbel"], "n_neighbours": 3, "start_temp": 1.0, "annealing_rate": 0.97},
    "DownhillSimplexOptimizer": {"alpha": 1.0, "gamma": 2.0, "beta": 0.5, "sigma": 0.5},
    "RandomSearchOptimizer": {},
    "GridSearchOptimizer": {"step_size": 1},
    "RandomRestartHillClimbingOptimizer": {"epsilon": 0.03, "distribution": ["normal", "laplace", "logistic", "gumbel"], "n_neighbours": 3, "n_iter_restart": 10},
    "RandomAnnealingOptimizer": {"epsilon": 0.03, "distribution": ["normal", "laplace", "logistic", "gumbel"], "n_neighbours": 3, "start_temp": 10.0, "annealing_rate": 0.98},
    "PowellsMethod": {"iters_p_dim": 10}, "PatternSearch": {"n_positions": 4, "pattern_size": 0.25, "reduction": 0.9},
    "ParallelTemperingOptimizer": {"population": 10, "n_iter_swap": 10, "rand_rest_p": 0.0},
    "ParticleSwarmOptimizer": {"population": 10, "inertia": 0.5, "cognitive_weight": 0.5, "social_weight": 0.5, "rand_rest_p": 0.0},
    "SpiralOptimization": {"population": 10, "decay_rate": 0.99}, "EvolutionStrategyOptimizer": {"population": 10, "mutation_rate": 0.7, "crossover_rate": 0.3, "rand_rest_p": 0.0},
    "LipschitzOptimizer": {"max_sample_size": 10000000, "sampling": {"random": 1000000}},
    "DirectAlgorithm": {},
    "BayesianOptimizer": {"xi": 0.3, "max_sample_size": 10000000, "sampling": {"random": 1000000}, "rand_rest_p": 0.0},
    "TreeStructuredParzenEstimators": {"gamma_tpe": 0.2, "max_sample_size": 10000000, "sampling": {"random": 1000000}, "rand_rest_p": 0.0},
    "ForestOptimizer": {"xi": 0.3, "tree_regressor": ["extra_tree", "random_forest", "gradient_boost"], "max_sample_size": 10000000, "sampling": {"random": 1000000}, "rand_rest_p": 0.0}
}

def parse_settings(optimizer_settings):
    settings_string = ""
    for key in optimizer_settings:
        if type(optimizer_settings[key]) is str:
            settings_string = settings_string + key + '="' + optimizer_settings[key] + '", '
        else:
            settings_string = settings_string + key + '=' + str(optimizer_settings[key]) + ', '
    return settings_string

from scripts.util.util_funcs import LazyDict

class HyperOptimizers():
    def __init__(self):
        self.optimizers = LazyDict()
        for optimizer in settings_list:
            self.optimizers.update({optimizer: (eval(optimizer), "")})
    def reset_optimizers(self):
        for optimizer in settings_list:
            self.optimizers.update({optimizer: (eval(optimizer), "")})
    def regenerate_optimizers(self, settings):
        for optimizer in settings_list:
            self.optimizers.update({optimizer: (eval(optimizer), parse_settings(settings[optimizer]))})

# I know I should have used list instead of exec here ;-; needs a rewrite but lazy
# The author takes no responsibility for eye damage caused by this incident
seperator = "___"

def on_ui_tabs(main_block):
    if shared.HyperOptimizers is None:
        shared.HyperOptimizers = HyperOptimizers()

    with gr.Row(variant="panel"):
        btn_save_settings = gr.Button(value="Save Settings")
        btn_load_settings = gr.Button(value="Load Settings")
        btn_apply_settings = gr.Button(value="Apply Settings", variant="primary")
    elements = {}
    with gr.Column(variant="panel"):
        for optimizer in settings_list:
            with gr.Accordion(optimizer, open=False):
                for setting in settings_list[optimizer]:
                    with gr.Row(variant="panel"):
                        value = settings_list[optimizer][setting]
                        element_name = optimizer + seperator + setting
                        if type(value) is int:
                            exec(f'{element_name} = gr.Number(label=setting, value=value, precision=0)')
                        elif type(value) is float:
                            exec(f'{element_name} = gr.Number(label=setting, value=value, precision=8)')
                        elif type(value) is list:
                            exec(f'{element_name} = gr.Dropdown(label=setting, choices=value, value=value[0])')
                        else:
                            exec(f'{element_name} = gr.Textbox(label=setting, value=value)')
                        elements.update({element_name: eval(element_name)})
    html_info = gr.HTML()

    def save_settings(args):
        try:
            settings = {}
            for optimizer in settings_list:
                optimizer_settings = {}
                for setting in settings_list[optimizer]:
                    optimizer_settings.update({setting: args[elements[optimizer + seperator + setting]]})
                settings.update({optimizer: optimizer_settings})
            config = msgspec.toml.encode(settings)
            with open(config_path, "wb") as f:
                f.write(config)
        except BaseException as e:
            print("autoMBW [error]: " + repr(e))
            return f"error: config failed to save to {config_path}.<br>"
        return f"success: config saved to {config_path}.<br>"

    btn_save_settings.click(fn=save_settings, inputs={*elements.values()}, outputs=[html_info])

    def apply_settings(args):
        try:
            settings = {}
            for optimizer in settings_list:
                optimizer_settings = {}
                for setting in settings_list[optimizer]:
                    optimizer_settings.update({setting: args[elements[optimizer + seperator + setting]]})
                settings.update({optimizer: optimizer_settings})
            shared.HyperOptimizers.regenerate_optimizers(settings)
        except BaseException as e:
            print("autoMBW [error]: " + repr(e))
            return f"error: config failed to apply from blocks.<br>"
        return f"success: config applied from blocks.<br>"

    btn_apply_settings.click(fn=apply_settings, inputs={*elements.values()}, outputs=[html_info])

    setting_keys = [*elements.keys()]
    def load_settings():
        try:
            settings = msgspec.toml.decode(open(config_path, "rb").read())
            return_list = []
            for setting in setting_keys:
                setting_split = setting.split(seperator)
                return_list.append(gr.update(value=settings[setting_split[0]][setting_split[1]]))
        except BaseException as e:
            print("autoMBW [error]: " + repr(e))
            return [f"error: config failed to load from {config_path}.<br>"] + setting_keys
        return [f"success: config loaded from {config_path}.<br>"] + return_list

    btn_load_settings.click(fn=load_settings, inputs=[], outputs=[html_info] + [elements[setting] for setting in setting_keys])

    # # this code is disgusting but too lazy to fix
    # def apply_settings_delay(args):
    #     while True:
    #         if not shared.HyperOptimizers is None:
    #             break
    #     return apply_settings(args)
    #
    # main_block.load(fn=load_settings, inputs=[], outputs=[html_info] + [elements[setting] for setting in setting_keys])
    # main_block.load(fn=apply_settings_delay, inputs={*elements.values()}, outputs=[html_info])
