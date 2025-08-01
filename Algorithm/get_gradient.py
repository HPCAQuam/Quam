
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib import cm
from datasets import load_dataset
import numpy as np
import re

def plot_2d_tensor(name, tensor, model_name):
    if tensor.dim() != 2:
        raise ValueError("Input must be a 2D tensor, but got a tensor with shape: {}".format(tensor.dim()))
    z_data = tensor.cpu().numpy()
    rows, cols = tensor.shape
    print(name, " max and min:", torch.max(tensor), torch.min(tensor))
    
    if torch.max(tensor) == torch.min(tensor):
        return
    z_data = z_data.astype(np.float32)
    z_data = zoom(z_data, 0.2, order=1) 
    rows, cols = z_data.shape
    X = np.arange(0, cols, 1)
    Y = np.arange(0, rows, 1)
    X = X*5
    Y = Y*5
    X, Y = np.meshgrid(X, Y)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, z_data, 
                          cmap=cm.coolwarm,
                          linewidth=0, 
                          antialiased=True,
                          rstride=1, 
                          cstride=1)
    
    z_min, z_max = np.min(z_data), np.max(z_data)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('Row Index', labelpad=12)
    ax.set_ylabel('Column Index', labelpad=12)
    cbar = fig.colorbar(surf, shrink=0.6, aspect=20)
    cbar.set_label('Value', rotation=270, labelpad=15)
    
    plt.show()
    plt.savefig("./gradient_result/" + model_name + '/' + "gradient_" + name + '.png')

    
def loadmodel(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    return tokenizer, model

def name_available(name):
    flag = False
    if "lm_head" in name:
        flag = True
    elif "proj.weight" in name:
        flag = True
    elif "fc1.weight" in name:
        flag = True
    elif "fc2.weight" in name:
        flag = True
    elif "dense.weight" in name:
        flag = True
    else:
        flag = False
    return flag

def extract_info(input_str):
    match = re.search(r'layers\.(\d+)', input_str)
    if match:
        return int(match.group(1))
    
    if "lm_head" in input_str:
        return -1
    
    return None

def view_grad(loaded_grad, model_name):
    for name, param in loaded_grad.items():
        if name_available(name):
            plot_2d_tensor(name, param, model_name)

def save_gradient(model, tokenizer):
    model.config.return_dict = True
    model.seqlen = 512
    model.eval() 
    n_samples = 512
    data = "pileval"
    if data == "pileval":
        dataset = load_dataset("./datasets/calib_dataset/", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)

    nsamples = cat_samples.numel() // model.seqlen
    torch.autograd.set_grad_enabled(True)
    print("nsamples: ", nsamples)
    for i in tqdm(range(nsamples), desc="evaluating..."):
        batch = cat_samples[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        outputs = model(batch)
        logits = outputs.logits
        first_logit = logits[:, 0, :].mean()
        first_logit.backward()  
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().clone()  
            gradients[name] = torch.abs(gradients[name])
    return gradients
