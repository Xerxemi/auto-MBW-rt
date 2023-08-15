import torch
import numpy as np

use_cuda = torch.cuda.is_available()

def image_embeddings_direct(image, model, processor):
    inputs = processor(images=image, return_tensors='pt')['pixel_values']
    if use_cuda:
        inputs = inputs.to('cuda')
    result = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
    return (result / np.linalg.norm(result)).squeeze(axis=0)

# binary classifier that consumes CLIP embeddings
class Classifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = torch.nn.Linear(hidden_size//2, output_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

import os
import torch
import safetensors
from transformers import CLIPModel, CLIPProcessor

dirname = os.path.dirname(__file__)
aesthetic_path = os.path.join(dirname, "aes-B32-v0.safetensors")
clip_name = 'openai/clip-vit-base-patch32'
clipprocessor = CLIPProcessor.from_pretrained(clip_name)
clipmodel = CLIPModel.from_pretrained(clip_name).to('cuda').eval()
aes_model = Classifier(512, 256, 1).to('cuda')
aes_model.load_state_dict(safetensors.torch.load_file(aesthetic_path))

def score(image, prompt="", reverse=False):
    image_embeds = image_embeddings_direct(image, clipmodel, clipprocessor)
    prediction = aes_model(torch.from_numpy(image_embeds).float().to('cuda'))
    if reverse:
        print("Reverse scoring not supported with this classifier. Are you sure you want to continue?")
    return prediction.item()

