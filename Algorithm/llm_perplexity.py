import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datasets import load_from_disk
import random, argparse, os
from tqdm import tqdm
from typing import Optional
from get_gradient import save_gradient
from quant_utils.quant_weight import quant_model
from quant_utils.write_results import write_results

def llm_eval(args, model_name, model_str,  wq_datatype, wq_bits, wq_groupsize, in_dim_tilesize, out_dim_tilesize, dataset, loaded_grad, th_percent = None):
    model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
    max_threshold_id, tile_num = quant_model(model, model_name, wq_bits, wq_datatype, wq_groupsize, in_dim_tilesize, out_dim_tilesize, loaded_grad, th_percent)

    random.seed(0)
    model_net = model_str.split('/')[-1]
    model_family = '_'.join(model_net.lower().split('-')[:-1])
    model.seqlen = 2048
    model = model.eval()

    cache_testloader = f'./data_cache//testloader_{model_family}_{dataset}_{model.seqlen}.cache'
    os.makedirs(os.path.dirname(cache_testloader), exist_ok=True)
    
    if os.path.exists(cache_testloader):
        testenc = torch.load(cache_testloader)
        print(f"load calibration from {cache_testloader}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=False, trust_remote_code=True)
        if dataset == 'wikitext2':
            testenc = load_from_disk("./datasets/" + "wikitext")
            testenc = testenc["test"]

            testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
            testenc = testenc.input_ids.to(model.device)
            torch.save(testenc, cache_testloader)
        elif dataset == 'c4':
            valenc = []
            testenc = load_dataset("./datasets/c4/", data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split="validation")
            for _ in range(256): # run 256 samples
                while True:
                    i = random.randint(0, len(testenc) - 1)
                    tmp = tokenizer(testenc[i]['text'], return_tensors='pt')
                    if tmp.input_ids.shape[1] > (model.seqlen + 1):
                        break
                i = random.randint(0, tmp.input_ids.shape[1] - model.seqlen - 1)
                j = i + model.seqlen
                valenc.append(tmp.input_ids[:, i:j].to(model.device))
            testenc = torch.hstack(valenc)
            # testenc = testenc.to(model.device)
            torch.save(testenc, cache_testloader)
        elif dataset == 'pileval':
            testenc = load_dataset("./datasets/calib_dataset/", split="validation")
            testenc = testenc.shuffle(seed=42)
            samples = []
            n_run = 0
            for data in testenc:
                line = data["text"]
                line = line.strip()
                line_encoded = tokenizer.encode(line)
                if len(line_encoded) > model.seqlen:
                    continue
                sample = torch.tensor([line_encoded]).to(model.device)
                if sample.numel() == 0:
                    continue
                samples.append(sample)
                n_run += 1
                if n_run == 512:
                    break
            # now concatenate all samples and split according to block size
            testenc = torch.cat(samples, dim=1)
            torch.save(testenc, cache_testloader)
        else:
            print("NO DATASETS!!!")
            exit()
            
    nsamples = testenc.numel() // model.seqlen
    loss_fct = torch.nn.CrossEntropyLoss()
    nlls = []
    for i in tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f'{dataset} perplexity: {ppl.item()}')
    write_results("./results_quant/each_infer_output.csv", ppl.item(), model_name, dataset, wq_bits, wq_datatype, wq_groupsize, th_percent, out_dim_tilesize)
    model = None
    return ppl.item(), max_threshold_id, tile_num
    

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    '--model-name', default = "opt-1.3b", type=str, help="Name of model e.g. `opt-1.3b`"
)
parser.add_argument(
    "--model-path", "-m", type=str, default="facebook/opt-1.3b", help="Path of model"
)
parser.add_argument(
    "--wq-datatype", type=str, default="int_asym_tile", help="The weight datatype for weight-only quantization",
)
parser.add_argument(
    "--wq-bits", type=int, default=4, help="The weight precision for weight-only quantization",
)
parser.add_argument(
    "--wq-groupsize", type=int, default=128, help="The quantization group size for weight-only quantization",
)
parser.add_argument(
    "--out-dim-tilesize", type=int, default=128, help="The output dimsion of weight tile for weight-only quantization",
)
parser.add_argument(
    "--in-dim-tilesize", type=int, default=128, help="The input dimsion of weight tile for weight-only quantization",
)

args = parser.parse_args()
model_name = args.model_name
model_path = args.model_path
wq_datatype = args.wq_datatype
wq_bits = args.wq_bits
wq_groupsize = args.wq_groupsize
out_dim_tilesize = args.out_dim_tilesize
in_dim_tilesize = args.in_dim_tilesize
evaluation_datasets = ["wikitext2", "c4"]
calibration_dataset = 'pileval'

loaded_grad_loader = f'./gradient_result/{model_name}_gradients.pt'
os.makedirs(os.path.dirname(loaded_grad_loader), exist_ok=True)
if os.path.exists(loaded_grad_loader):
    loaded_grad = torch.load(loaded_grad_loader)
    print(f"load calibration from {loaded_grad_loader}")
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    gradient_path = "./gradient_result/" + model_name + '_gradients.pt'
    loaded_grad = save_gradient(model, tokenizer, gradient_path)
    torch.save(loaded_grad, gradient_path) 

if ("tile" in wq_datatype):
    if wq_bits == 3:
        delta_ppl_th = 0.01
    elif wq_bits == 4:
        delta_ppl_th = 0.002
    min_th_percent = 0
    max_th_percent = 0.5
    
    ppl_current, max_threshold_id, tile_num = llm_eval(args, model_name, model_path,  wq_datatype, wq_bits, wq_groupsize, in_dim_tilesize, out_dim_tilesize, calibration_dataset, loaded_grad)
    current_th_percent = max_threshold_id/tile_num
    left_th_percent = max(min_th_percent, current_th_percent / 2)
    right_th_percent = min(max_th_percent, current_th_percent * 2)
    ppl_left, max_threshold_id, tile_num = llm_eval(args, model_name, model_path,  wq_datatype, wq_bits, wq_groupsize, in_dim_tilesize, out_dim_tilesize, calibration_dataset, loaded_grad, left_th_percent)
    ppl_right, max_threshold_id, tile_num = llm_eval(args, model_name, model_path,  wq_datatype, wq_bits, wq_groupsize, in_dim_tilesize, out_dim_tilesize, calibration_dataset, loaded_grad, right_th_percent)
    
    ite_num = 0
    while True:
        ite_num += 1 
        if ite_num > 50:
            print("ite_num:", ite_num)
            break
        
        if ppl_left < ppl_current and ppl_left < ppl_right:
            print("left <-")
            delta_ppl = abs(ppl_left - min(ppl_current, ppl_right))
            if delta_ppl < delta_ppl_th:
                current_th_percent = left_th_percent
                ppl_current = ppl_left
                break
            
            max_th_percent = right_th_percent
            right_th_percent = current_th_percent
            current_th_percent = left_th_percent
            left_th_percent = max(min_th_percent, left_th_percent/2)
            
            ppl_right = ppl_current
            ppl_current = ppl_left
            ppl_left, max_threshold_id, tile_num = llm_eval(args, model_name, model_path,  wq_datatype, wq_bits, wq_groupsize, in_dim_tilesize, out_dim_tilesize, calibration_dataset, loaded_grad, left_th_percent)
        elif ppl_current < ppl_left and ppl_current < ppl_right:
            print("keep current")
            delta_ppl = abs(ppl_current - min(ppl_left, ppl_right))
            if delta_ppl < delta_ppl_th:
                break
            
            min_th_percent = left_th_percent
            left_th_percent = (left_th_percent + current_th_percent)/2
            max_th_percent = right_th_percent
            right_th_percent = (right_th_percent + current_th_percent)/2
            ppl_left, max_threshold_id, tile_num = llm_eval(args, model_name, model_path,  wq_datatype, wq_bits, wq_groupsize, in_dim_tilesize, out_dim_tilesize, calibration_dataset, loaded_grad, left_th_percent)
            ppl_right, max_threshold_id, tile_num = llm_eval(args, model_name, model_path,  wq_datatype, wq_bits, wq_groupsize, in_dim_tilesize, out_dim_tilesize, calibration_dataset, loaded_grad, right_th_percent)
        else:
            print("-> right")
            delta_ppl = abs(ppl_right - min(ppl_current, ppl_left))
            if delta_ppl < delta_ppl_th:
                current_th_percent = right_th_percent
                ppl_current = ppl_right
                break
            
            min_th_percent = left_th_percent
            left_th_percent = current_th_percent
            current_th_percent = right_th_percent
            right_th_percent = min(max_th_percent, right_th_percent*2)
            ppl_left = ppl_current
            ppl_current = ppl_right
            ppl_right, max_threshold_id, tile_num = llm_eval(args, model_name, model_path,  wq_datatype, wq_bits, wq_groupsize, in_dim_tilesize, out_dim_tilesize, calibration_dataset, loaded_grad, right_th_percent)
    
    # print("ppl: ", ppl_current)
    # print("current_th_percent: ", current_th_percent)
    write_results("./results_quant/each_type_output.csv", ppl_current, model_name, calibration_dataset, wq_bits, wq_datatype, wq_groupsize, current_th_percent, out_dim_tilesize)
    
    for dataset_final in evaluation_datasets:
        ppl, _, _ = llm_eval(args, model_name, model_path,  wq_datatype, wq_bits, wq_groupsize, in_dim_tilesize, out_dim_tilesize, dataset_final, loaded_grad, current_th_percent)
        write_results("./results_quant/each_type_output.csv", ppl, model_name, dataset_final, wq_bits, wq_datatype, wq_groupsize, current_th_percent, out_dim_tilesize)
else:
    ppl, max_threshold_id, tile_num = llm_eval(args, model_name, model_path,  wq_datatype, wq_bits, wq_groupsize, in_dim_tilesize, out_dim_tilesize, calibration_dataset, loaded_grad, -1)
    write_results("./results_quant/each_type_output.csv", ppl, model_name, calibration_dataset, wq_bits, wq_datatype, wq_groupsize, None)
