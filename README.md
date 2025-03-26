# Qlora-intro
 Implements QLoRA (4-bit quantized LoRA) for cost-efficient



#  QLoRA Fine-Tuning 

Efficiently fine-tune Llama 3 (8B) using QLoRA (4-bit quantization + LoRA) for insurance domain adaptation. Achieves low-memory adaptation (~20GB GPU) while preserving model performance.

##  Features
- 4-bit quantization via `bitsandbytes`
- LoRA adapters targeting attention layers (`q_proj`, `v_proj`, etc.)
- Hugging Face ecosystem integration (`transformers`, `peft`, `trl`)
- Memory-efficient training (~5x reduction vs full fine-tuning)

##  Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set up Hugging Face token
export HF_TOKEN="your_hf_token_here"



## 📊 Technical Details
| Hyperparameter       | Value       |
|----------------------|-------------|
| Base Model           | `meta-llama/Meta-Llama-3.1-8B` |
| Quantization         | NF4 (4-bit) |
| LoRA Rank (r)        | 32          |
| LoRA Alpha           | 64          |
| Target Modules       | `q_proj`, `v_proj`, `k_proj`, `o_proj` |


## 📜 Requirements
See [requirements.txt](./requirements.txt)

### Key Notes:
1. **Memory Efficiency**:  
   - Quantization reduces base model memory to ~20GB (vs 32GB+ for FP16)
   - LoRA adds only ~0.5% trainable parameters

2. **Reproducibility**:  
   - `set_seed()` and deterministic configs recommended

3. **Customization**:  
   - Modify `TARGET_MODULES` for different architectures
   - Adjust `LORA_R` based on your GPU constraints

