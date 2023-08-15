import torch
import torchvision
#     import models as models
import scripts.classifiers.hybridIQA.models as models
import numpy as np

# state_name = "hyperIQA_hybrid_latest.pth"
# dirname = os.path.dirname(__file__)
# pth_path = os.path.join(dirname, state_name)

#deterministic crop
torch.manual_seed(0)

model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
model_hyper.train(False)
# load pre-trained model
# model_hyper.load_state_dict(torch.load(pth_path, map_location="cuda:0"))
model_hyper.load_state_dict(torch.hub.load_state_dict_from_url("https://huggingface.co/Xerxemi/hybridIQA-V1/resolve/main/hyperIQA_hybrid_latest.pth", map_location="cuda:0"))

transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])

rng_reset = torch.get_rng_state()
# random crop 25 patches and calculate mean quality score
def score(image, prompt="", reverse=False):
    if reverse:
        print("Reverse scoring not supported with this classifier. Are you sure you want to continue?")
    image = image.convert("RGB")
    pred_scores = []
    for i in range(25):
        img = transforms(image)
        img = img.cuda().clone().detach().unsqueeze(0)
        paras = model_hyper(img)  # 'paras' contains the network weights conveyed to target network

        # Building target network
        model_target = models.TargetNet(paras).cuda()
        for param in model_target.parameters():
            param.requires_grad = False

        # Quality prediction
        pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
        pred_scores.append(float(pred.item()))
    torch.set_rng_state(rng_reset)
    return np.mean(pred_scores)

