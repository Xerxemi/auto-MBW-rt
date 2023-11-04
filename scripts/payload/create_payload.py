import os
import msgspec
msgspec_encoders = {".json": msgspec.json.Encoder(), ".msgpack": msgspec.msgpack.Encoder(), ".toml": msgspec.toml, ".yaml": msgspec.yaml}
msgspec_decoders = {".json": msgspec.json.Decoder(), ".msgpack": msgspec.msgpack.Decoder(), ".toml": msgspec.toml, ".yaml": msgspec.yaml}

import gradio as gr

from modules import shared, processing, devices
from modules.scripts import basedir
from modules.sd_samplers import samplers

from scripts.util.auto_mbw_rt_logger import logger_autombwrt as logger

# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
__location__ = basedir()
payloads_path = os.path.join(__location__, "payloads")

#payload function with ext
def refresh_payloads():
    global discovered_payloads
    discovered_payloads = []
    for _, _, files in os.walk(payloads_path):
        for f in files:
            if os.path.splitext(f)[1] in [".json", ".msgpack", ".toml", ".yaml"]:
                discovered_payloads.append(f)
    discovered_payloads = [*set(discovered_payloads)]
    logger.info("discovered " + str(len(discovered_payloads)) + " payloads.")
    return gr.update(choices=discovered_payloads)

refresh_payloads()

def on_ui_tabs(main_block):
    def create_sampler_and_steps_selection(choices, tabname):
        with gr.Row(elem_id=f"sampler_selection_{tabname}"):
            sampler = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value=choices[0].name)
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
        return steps, sampler
    with gr.Row():
        with gr.Column(variant="panel"):
            radio_wildcard_type = gr.Radio(label="Wildcards Type", choices=["none", "random", "combinatorial", "jinja2", "lucky"], value="random")
            positive_prompt = gr.Text(label="Positive Prompt", elem_id="autombw_positive_prompt", lines=4, placeholder="Positive prompt here")
            negative_prompt = gr.Text(label="Negative Prompt", elem_id="autombw_negative_prompt", lines=4, placeholder="Negative prompt here")
            with gr.Row():
                steps, sampler = create_sampler_and_steps_selection(samplers, "autombw")
            with gr.Row():
                with gr.Column(elem_id="autombw_column_size", scale=4):
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512, elem_id="autombw_width")
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512, elem_id="autombw_height")
                with gr.Column(elem_id="autombw_column_batch"):
                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count (Wildcards)', value=1, elem_id="autombw_batch_count")
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="autombw_batch_size")
            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id="autombw_cfg_scale")
            seed = gr.Number(label="Seed", value=1, precision=0)
            with gr.Row(elem_id="autombw_checkboxes"):
                with gr.Column(scale=3):
                    with gr.Row():
                        restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1, elem_id="autombw_restore_faces")
                        tiling = gr.Checkbox(label='Tiling', value=False, elem_id="autombw_tiling")
                        enable_hr = gr.Checkbox(label='Hires. fix', value=False, elem_id="autombw_enable_hr")
                        reverse_scoring = gr.Checkbox(label='reverse_scoring', value=False)
                with gr.Column(scale=1, min_width=150):
                    with gr.Row():
                        hr_final_resolution = gr.HTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution", interactive=False, visible=False)
            with gr.Row(elem_id="autombw_hires_fix_row1"):
                hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="autombw_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode, visible=False)
                hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="autombw_hires_steps", visible=False)
                denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label='Denoising strength', value=0.7, elem_id="autombw_denoising_strength", visible=False)
            with gr.Row(elem_id="autombw_hires_fix_row2"):
                hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="autombw_hr_scale", visible=False)
                hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize width to", value=0, elem_id="autombw_hr_resize_x", visible=False)
                hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize height to", value=0, elem_id="autombw_hr_resize_y", visible=False)
        with gr.Column(variant="panel"):
            btn_create_payload = gr.Button(value="Create Payload", variant="primary")
            filename = gr.Text(label="Filename")
            radio_ext = gr.Radio(label="Format", choices=[".json", ".msgpack", ".toml", ".yaml"], value=".json")
            chk_overwrite = gr.Checkbox(label="Allow Overwrite", value=False)
            with gr.Row():
                btn_load_payload = gr.Button(value="Load Payload", variant="primary")
                btn_reload_payloads= gr.Button(value="Reload Payloads")
            dropdown_payloads = gr.Dropdown(label="Payload", choices=discovered_payloads)
            html_output_info = gr.HTML()

    def create_payload(args):
        payload = {
            "enable_hr": args[enable_hr],
            "denoising_strength": args[denoising_strength],
            "firstphase_width": 0,
            "firstphase_height": 0,
            "hr_scale": args[hr_scale],
            "hr_upscaler": args[hr_upscaler],
            "hr_second_pass_steps": args[hr_second_pass_steps],
            "hr_resize_x": args[hr_resize_x],
            "hr_resize_y": args[hr_resize_y],
            "prompt": args[positive_prompt],
            "seed": args[seed],
            "sampler_name": args[sampler],
            "batch_size": args[batch_size],
            "n_iter": args[batch_count],
            "steps": args[steps],
            "cfg_scale": args[cfg_scale],
            "width": args[width],
            "height": args[height],
            "restore_faces": args[restore_faces],
            "tiling": args[tiling],
            "negative_prompt": args[negative_prompt],
            "wildcard_type": args[radio_wildcard_type],
            "reverse_scoring": args[reverse_scoring]
        }
        payload = msgspec_encoders[args[radio_ext]].encode(payload)
        payload_path = os.path.join(__location__, "payloads", args[filename] + args[radio_ext])
        if not os.path.exists(payload_path) or args[chk_overwrite]:
            with open(payload_path, "wb") as f:
                f.write(payload)
            if args[chk_overwrite]:
                logger.warning("[overwrite]: file already exists")
                return f"success: file overwritten [{payload_path}].<br>"
        else:
            logger.error("Error: File already exists")
            return "error: file already exists.<br>"
        return f"success: file created [{payload_path}].<br>"

    payload_args = {
        "enable_hr": enable_hr,
        "denoising_strength": denoising_strength,
        "hr_scale": hr_scale,
        "hr_upscaler": hr_upscaler,
        "hr_second_pass_steps": hr_second_pass_steps,
        "hr_resize_x": hr_resize_x,
        "hr_resize_y": hr_resize_y,
        "prompt": positive_prompt,
        "seed": seed,
        "sampler_name": sampler,
        "batch_size": batch_size,
        "n_iter": batch_count,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "restore_faces": restore_faces,
        "tiling": tiling,
        "negative_prompt": negative_prompt,
        "wildcard_type": radio_wildcard_type,
        "reverse_scoring": reverse_scoring
    }
    # payload_args = [enable_hr, denoising_strength, hr_scale, hr_upscaler, hr_second_pass_steps, hr_resize_x, hr_resize_y,
    #     positive_prompt, seed, sampler, batch_size, batch_count, steps, cfg_scale, width, height, restore_faces, tiling, negative_prompt]

    btn_create_payload.click(
        fn=create_payload,
        inputs={filename, chk_overwrite, radio_ext, *[*payload_args.values()]},
        outputs=[html_output_info]
    )

    payload_keys = [*payload_args.keys()]
    def load_payload(payload):
        payload_path = os.path.join(payloads_path, payload)
        if os.path.isfile(payload_path):
            payload = msgspec_decoders[os.path.splitext(payload_path)[1]].decode(open(payload_path, "rb").read())
        else:
            return f"error: file not found [{payload_path}].<br>"
        return_list = []
        for key in payload_keys:
            try:
                return_list.append(gr.update(value=payload[key]))
            except KeyError:
                return_list.append(gr.update())
        return [f"success: file loaded [{payload_path}].<br>"] + return_list

    btn_load_payload.click(
        fn=load_payload,
        inputs=[dropdown_payloads],
        outputs=[html_output_info] + [payload_args[key] for key in payload_keys]
    )

    btn_reload_payloads.click(
        fn=refresh_payloads,
        inputs=[],
        outputs=[dropdown_payloads]
    )

    def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
        if not enable:
            return ""
        p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
        with devices.autocast():
            p.init([""], [0], [0])
        return f"resize: from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"

    hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]
    for input in hr_resolution_preview_inputs:
        input.change(
            fn=calc_resolution_hires,
            inputs=hr_resolution_preview_inputs,
            outputs=[hr_final_resolution],
            show_progress=False,
        )
        input.change(
            None,
            _js="onCalcResolutionHires",
            inputs=hr_resolution_preview_inputs,
            outputs=[],
            show_progress=False,
        )

    def gr_show(visible_flag):
        if visible_flag:
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)

    hr_options = [hr_final_resolution, hr_upscaler, hr_second_pass_steps, denoising_strength, hr_scale, hr_resize_x, hr_resize_y]
    enable_hr.change(
    fn=lambda x: [gr_show(x) for _ in range(len(hr_options))],
    inputs=[enable_hr],
    outputs=hr_options
    )
