
# Quam Hardware Simulator

This folder contains the files and scripts necessary to reproduce the original results of our Quam hardware simulator.
To run the experiments:

1. Navigate to this directory and activate the Quam conda environment.
If you haven't set it up yet, please follow the setup instructions in the Algorithm folder.
```
cd Hardware
conda activate Quam
```

2. Modify the default HuggingFace cache directory in run_shape_profile.sh:
```
export HF_HOME="your/HF_HOME/directory"
```

3. Profile the LLM configuration and layer shape. The profiled information will be saved in a new folder **model_shape_config** under this directory:
```
bash run_shape_profile.sh
```

4. Run latency and energy evaluation for different models on various accelerators:
```
python test_baseline.py --is_generation  # Baseline FP16 accelerator
python test_ant.py      --is_generation  # ANT accelerator
python test_olive.py    --is_generation  # OliVe accelerator
python test_bitmod.py   --is_generation  # BitMoD accelerator
python test_quam.py   --is_generation  # Quam accelerator
```
> ⚠️ Note:
> The weight precision of ANT and OliVe is hard-coded in our simulator. 
> The precision is profiled offline by ensuring their quantized perplexity and accuracy are acceptable after applying their quantization data types and algorithms. 
