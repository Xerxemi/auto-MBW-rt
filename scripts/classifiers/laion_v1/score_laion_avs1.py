#adapted from grexzen/SD-Chad

import torch
import torch.nn as nn
import numpy as np
import clip
import os

state_name = "sa_0_4_vit_l_14_linear.pth"
dirname = os.path.dirname(__file__)
aesthetic_path = os.path.join(dirname, state_name)

# class AestheticPredictor(nn.Module):
#     def __init__(self, input_size):
#         super().__init__()
#         self.input_size = input_size
#         self.layers = nn.Linear(self.input_size, 1)
#
#     def forward(self, x):
#         return self.layers(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
# load the model you trained previously or the model available in this repo
pt_state = torch.load(aesthetic_path, map_location=torch.device('cpu'))

# CLIP embedding dim is 768 for CLIP ViT L 14
predictor = nn.Linear(768, 1)
predictor.load_state_dict(pt_state)
predictor.to(device)
predictor.eval()

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

def get_image_features(image, device=device, model=clip_model, preprocess=clip_preprocess):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        # l2 normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()
    return image_features

def score(image, prompt="", reverse=False):
    image_features = get_image_features(image)
    score = predictor(torch.from_numpy(image_features).to(device).float())
    if reverse:
        print("Reverse scoring currently not supported with this classifier. Are you sure you want to continue?")
    return score.item()

