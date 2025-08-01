# Quam Algorithm

This folder contains the implementation of Quam's algorithm module.

## 🔧 Environment Setup

First, create and activate the `Quam`  environment:

```bash
cd Algorithm
conda create -n Quam python=3.10 -y
conda activate Quam
```


All required packages and their versions are listed in `require.txt`. You can install them.


## 🚀 Run the Experiment

We provide a script `run_exp.sh` to launch the experiments directly:

```bash
bash run_exp.sh
```

> ⚠️ **Note:**  
> Please make sure to adjust the **model path** and **dataset path** in the script according to your local environment.

## 📁 File Descriptions

- `run_exp.sh`: Main script to run the experiments.
- `require.txt`: Python dependency list.
- `get_gradient.py`: Computes gradients of LLM layers for analysis or calibration.
- `llm_perplexity.py`: Evaluates perplexity of large language models under quantization.
- `data_cache/`: Stores intermediate data, such as model shape configurations or preprocessed inputs.
- `datasets/`: Contains the input datasets used in experiments.
- `gradient_result/`: Stores computed gradients and related results.
- `results_quant/`: Contains output results of quantized model experiments.
- `quant_utils/`: Utility functions for quantization operations.

