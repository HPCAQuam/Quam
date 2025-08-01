#!/bin/bash
# Modify the model path to your own path. 
declare -a model_name_list=("opt-1.3b" "phi-2" "Yi-6B" "Llama-2-7b" "Meta-Llama-3-8B")

# Set the quantization precision list.
wq_bit_list=(4 3)

# Set the quantization group size. Use -1 for per-channel quantization
wq_groupsize=128
datatype_list=("int_asym_tile")

for model_name in "${model_name_list[@]}"
do
    for wq_bit in "${wq_bit_list[@]}"
    do
        for  datatype in "${datatype_list[@]}"
        do
            if [[ ${model_name} == "opt-1.3b"]]
            then
                model_path="facebook/opt-1.3b"
            elif [[ ${model_name} == "phi-2"]]
            then
                model_path="microsoft/phi-2"
            elif [[ ${model_name} == "Yi-6B"]]
            then
                model_path="01-ai/Yi-6B"
            elif [[ ${model_name} == "Llama-2-7b"]]
            then
                model_path="meta-llama/Llama-2-7b-hf"
            elif [[ ${model_name} == "Meta-Llama-3-8B"]]
            then
                model_path="meta-llama/Meta-Llama-3-8B"
            fi

            echo "#################### Running Experiment ####################"
            echo "Model             = ${model_name}"
            echo "Quant precision   = ${wq_bit}"
            echo "Quant group size  = ${wq_groupsize}"
            echo "Quant datatype    = ${datatype}"
            echo "############################################################"
    
            python llm_perplexity.py \
                --model-name ${model_name}\
                --model-path ${model_path} \
                --wq-datatype ${datatype} \
                --wq-bits ${wq_bit} \
                --wq-groupsize ${wq_groupsize} \
                --out-dim-tilesize 128 \
                --in-dim-tilesize 128 
        done
    done
done