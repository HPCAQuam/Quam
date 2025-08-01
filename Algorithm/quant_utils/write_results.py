import os 
import csv

def write_results(BASE_PATH: str, ppl: float, model_name: str, dataset: str, wq_bits: int, wq_datatype: str, wq_groupsize: int, th_percent: float, out_dim_tilesize: int):
    if wq_groupsize <= 0:
        wq_groupsize = "none"
    dataset_dir = f"{dataset}"
    wq_bits_groupsize      = f"w_{wq_bits}_gs_{wq_groupsize}"
    dtype  = f"{wq_datatype}"
    if (wq_datatype == 'fp16'):
        wq_bits_groupsize = "naive"
    data = [dataset_dir, model_name, wq_bits_groupsize, dtype, out_dim_tilesize, ppl, th_percent]
        
    with open(BASE_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    
    print('Successfully written results. \n\n')