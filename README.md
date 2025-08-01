# 🔬 Quam: A Cross-Layer Power-Law-Aware Quantization Accelerator for Efficient LLM Inference

This repository contains the source code for **Quam**, a cross-layer quantization framework and accelerator design for efficient large language model (LLM) inference.

> 📝 This repository is released as part of our **HPCA submission**.  
> All author-identifying information has been **removed or anonymized** in accordance with the double-blind review policy.

---

## 📂 Repository Structure

- `Algorithm/`  
  Contains the software-level implementation of Quam's quantization algorithms, calibration routines, and evaluation pipeline.

- `Hardware/`  
  Contains the simulator and scripts used to reproduce latency and energy results for different accelerators, including Quam.

---

## ⚙️ Quick Start

### 1. Setup the Python Environment

```bash
cd Algorithm
conda create -n Quam python=3.10 -y
conda activate Quam
```

### 2. Run Algorithm Module

```bash
bash run_exp.sh
```

> ⚠️ Make sure to adjust the model and dataset paths in `run_exp.sh` as needed.

### 3. Run Hardware Simulation

```bash
cd ../Hardware
conda activate Quam  # Reuse the same environment

# Set HuggingFace cache path
export HF_HOME="your/HF_HOME/directory"

# Profile model shapes
bash run_shape_profile.sh

# Run inference simulation on different accelerator backends
python test_baseline.py --is_generation   # Baseline FP16
python test_ant.py      --is_generation   # ANT
python test_olive.py    --is_generation   # OliVe
python test_bitmod.py   --is_generation   # BitMoD
python test_quam.py   --is_generation   # Quam (proposed)
```



## 📬 Contact

Due to the double-blind review process, we are unable to disclose author information at this time.  

---

## 🛡️ License

This code is provided solely for the purpose of academic peer review.
Redistribution, commercial use, and modification are **not permitted** during the review process.
A full license will be published upon paper acceptance.
