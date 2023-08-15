#code proudly stolen from https://github.com/hako-mikan/sd-webui-supermerger

import os, gc, re

import torch
from safetensors.torch import load_file, save_file

from modules import sd_models #, shared

#dirty lora import
import importlib
lora = importlib.import_module("extensions-builtin.Lora.lora")

re_digits = re.compile(r"\d+")

re_unet_down_blocks = re.compile(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)")
re_unet_mid_blocks = re.compile(r"lora_unet_mid_block_attentions_(\d+)_(.+)")
re_unet_up_blocks = re.compile(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)")

re_unet_down_blocks_res = re.compile(r"lora_unet_down_blocks_(\d+)_resnets_(\d+)_(.+)")
re_unet_mid_blocks_res = re.compile(r"lora_unet_mid_block_resnets_(\d+)_(.+)")
re_unet_up_blocks_res = re.compile(r"lora_unet_up_blocks_(\d+)_resnets_(\d+)_(.+)")

re_unet_downsample = re.compile(r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv(.+)")
re_unet_upsample = re.compile(r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv(.+)")

re_text_block = re.compile(r"lora_te_text_model_encoder_layers_(\d+)_(.+)")

def filenamecutter(name,model_a = False):
    from modules import sd_models
    if name =="" or name ==[]: return
    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    name= os.path.splitext(checkpoint_info.filename)[0]

    if not model_a:
        name = os.path.basename(name)
    return name

def load_state_dict(file_name, dtype):
  if os.path.splitext(file_name)[1] == '.safetensors':
    sd = load_file(file_name)
  else:
    sd = torch.load(file_name, map_location='cpu')
  for key in list(sd.keys()):
    if type(sd[key]) == torch.Tensor:
      sd[key] = sd[key].to(dtype)
  return sd

def dimalpha(lora_sd, base_dims={}, base_alphas={}):
    alphas = {}                             # alpha for current model
    dims = {}                               # dims for current model
    for key in lora_sd.keys():
      if 'alpha' in key:
        lora_module_name = key[:key.rfind(".alpha")]
        alpha = float(lora_sd[key].detach().numpy())
        alphas[lora_module_name] = alpha
        if lora_module_name not in base_alphas:
          base_alphas[lora_module_name] = alpha
      elif "lora_down" in key:
        lora_module_name = key[:key.rfind(".lora_down")]
        dim = lora_sd[key].size()[0]
        dims[lora_module_name] = dim
        if lora_module_name not in base_dims:
          base_dims[lora_module_name] = dim

    for lora_module_name in dims.keys():
      if lora_module_name not in alphas:
        alpha = dims[lora_module_name]
        alphas[lora_module_name] = alpha
        if lora_module_name not in base_alphas:
          base_alphas[lora_module_name] = alpha
    return base_dims, base_alphas, dims, alphas

def dimgetter(filename):
    lora_sd = load_state_dict(filename, torch.float)
    alpha = None
    dim = None
    type = None

    if "lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight" in lora_sd.keys():
      type = "LoCon"
      _, _, dim, _ = dimalpha(lora_sd)

    for key, value in lora_sd.items():

        if alpha is None and 'alpha' in key:
            alpha = value
        if dim is None and 'lora_down' in key and len(value.size()) == 2:
            dim = value.size()[0]
        if "hada_" in key:
            dim,type = "LyCORIS","LyCORIS"
        if alpha is not None and dim is not None:
            break
    if alpha is None:
        alpha = dim
    if type == None:type = "LoRA"
    if dim :
      return dim,type
    else:
      return "unknown","unknown"

def convert_diffusers_name_to_compvis(key):
    def match(match_list, regex):
        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, re_unet_down_blocks):
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_mid_blocks):
        return f"diffusion_model_middle_block_1_{m[1]}"

    if match(m, re_unet_up_blocks):
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_down_blocks_res):
        block = f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_mid_blocks_res):
        block = f"diffusion_model_middle_block_{m[0]*2}_"
        if m[1].startswith('conv1'):
            return f"{block}in_layers_2{m[1][len('conv1'):]}"
        elif m[1].startswith('conv2'):
            return f"{block}out_layers_3{m[1][len('conv2'):]}"
        elif m[1].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[1][len('time_emb_proj'):]}"
        elif m[1].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[1][len('conv_shortcut'):]}"

    if match(m, re_unet_up_blocks_res):
        block = f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_downsample):
        return f"diffusion_model_input_blocks_{m[0]*3+3}_0_op{m[1]}"

    if match(m, re_unet_upsample):
        return f"diffusion_model_output_blocks_{m[0]*3 + 2}_{1+(m[0]!=0)}_conv{m[1]}"

    if match(m, re_text_block):
        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key

def to_half(sd):
    for key in sd.keys():
        if 'model' in key and sd[key].dtype == torch.float:
            sd[key] = sd[key].half()
    return sd

def savemodel(state_dict,currentmodel,fname,savesets,model_a,metadata={}):
    if "fp16" in savesets:
        state_dict = to_half(state_dict)
        pre = ".fp16"
    else:pre = ""
    ext = ".safetensors" if "safetensors" in savesets else ".ckpt"

    # is it a inpainting or instruct-pix2pix2 model?
    if "model.diffusion_model.input_blocks.0.0.weight" in state_dict.keys():
        shape = state_dict["model.diffusion_model.input_blocks.0.0.weight"].shape
        if shape[1] == 9:
            pre += "-inpainting"
        if shape[1] == 8:
            pre += "-instruct-pix2pix"

    checkpoint_info = sd_models.get_closet_checkpoint_match(model_a)
    model_a_path= checkpoint_info.filename
    modeldir = os.path.split(model_a_path)[0]

    if not fname or fname == "":
        fname = currentmodel.replace(" ","").replace(",","_").replace("(","_").replace(")","_")+pre+ext
        if fname[0]=="_":fname = fname[1:]
    else:
        fname = fname if ext in fname else fname +pre+ext

    fname = os.path.join(modeldir, fname)

    if len(fname) > 255:
       fname.replace(ext,"")
       fname=fname[:240]+ext

    # check if output file already exists
    if os.path.isfile(fname) and not "overwrite" in savesets:
        _err_msg = f"Output file ({fname}) existed and was not saved]"
        print(_err_msg)
        return _err_msg

    print("Saving...")
    if ext == ".safetensors":
        save_file(state_dict, fname, metadata=metadata)
    else:
        torch.save(state_dict, fname)
    print("Done!")
    return "Merged model saved in "+fname

LORABLOCKS=["encoder",
"diffusion_model_input_blocks_1_",
"diffusion_model_input_blocks_2_",
"diffusion_model_input_blocks_4_",
"diffusion_model_input_blocks_5_",
"diffusion_model_input_blocks_7_",
"diffusion_model_input_blocks_8_",
"diffusion_model_middle_block_",
"diffusion_model_output_blocks_3_",
"diffusion_model_output_blocks_4_",
"diffusion_model_output_blocks_5_",
"diffusion_model_output_blocks_6_",
"diffusion_model_output_blocks_7_",
"diffusion_model_output_blocks_8_",
"diffusion_model_output_blocks_9_",
"diffusion_model_output_blocks_10_",
"diffusion_model_output_blocks_11_"]

def pluslora(lnames,loraratios,settings,output,model,precision):
    if model == []:
      return "ERROR: No model Selected"
    if lnames == "":
      return "ERROR: No LoRA Selected"

    print("plus LoRA start")
    lnames = [lnames] if "," not in lnames else lnames.split(",")

    for i, n in enumerate(lnames):
        lnames[i] = n.split(":")

    loraratios=loraratios.splitlines()
    ldict ={}

    for i,l in enumerate(loraratios):
        if ":" not in l or not (l.count(",") == 16 or l.count(",") == 25) : continue
        ldict[l.split(":")[0].strip()]=l.split(":")[1]

    names=[]
    filenames=[]
    lweis=[]

    for n in lnames:
        if len(n) ==3:
            if n[2].strip() in ldict:
                ratio = [float(r)*float(n[1]) for r in ldict[n[2]].split(",")]
            else:ratio = [float(n[1])]*17
        else:ratio = [float(n[1])]*17
        c_lora = lora.available_loras.get(n[0], None)
        names.append(n[0])
        filenames.append(c_lora.filename)
        _,t = dimgetter(c_lora.filename)
        if "LyCORIS" in t: return "LyCORIS merge is not supported"
        lweis.append(ratio)

    modeln=filenamecutter(model,True)
    dname = modeln
    for n in names:
      dname = dname + "+"+n

    checkpoint_info = sd_models.get_closet_checkpoint_match(model)
    print(f"Loading {model}")
    theta_0 = sd_models.read_state_dict(checkpoint_info.filename,"cpu")

    keychanger = {}
    for key in theta_0.keys():
        if "model" in key:
            skey = key.replace(".","_").replace("_weight","")
            keychanger[skey.split("model_",1)[1]] = key

    for name,filename, lwei in zip(names,filenames, lweis):
      print(f"loading: {name}")
      lora_sd = load_state_dict(filename, torch.float)

      print("merging..." ,lwei)
      for key in lora_sd.keys():
        ratio = 1

        fullkey = convert_diffusers_name_to_compvis(key)

        for i,block in enumerate(LORABLOCKS):
            if block in fullkey:
                ratio = lwei[i]

        msd_key, lora_key = fullkey.split(".", 1)

        if "lora_down" in key:
            up_key = key.replace("lora_down", "lora_up")
            alpha_key = key[:key.index("lora_down")] + 'alpha'

            # print(f"apply {key} to {module}")

            down_weight = lora_sd[key].to(device="cpu")
            up_weight = lora_sd[up_key].to(device="cpu")

            dim = down_weight.size()[0]
            alpha = lora_sd.get(alpha_key, dim)
            scale = alpha / dim
            # W <- W + U * D
            weight = theta_0[keychanger[msd_key]].to(device="cpu")

            if len(weight.size()) == 2:
                # linear
                weight = weight + ratio * (up_weight @ down_weight) * scale

            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                    weight
                    + ratio
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                # print(conved.size(), weight.size(), module.stride, module.padding)
                weight = weight + ratio * conved * scale

            theta_0[keychanger[msd_key]] = torch.nn.Parameter(weight)
    #usemodelgen(theta_0,model)
    settings.append(precision)
    result = savemodel(theta_0,dname,output,settings,model)
    del theta_0
    gc.collect()
    return result
