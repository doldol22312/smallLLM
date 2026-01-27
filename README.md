***

# 🧠 TinyGPT

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

**A minimal, single-file implementation of a GPT-style language model trainer in PyTorch.**
<br>
*Designed for learning, experimentation, and training small language models on custom datasets.*

[Features](#-features) • [Quick Start](#-quick-start) • [Configuration](#-configuration) • [Examples](#-examples)

</div>

---

## ✨ Features

TinyGPT is packed with modern LLM features despite its small footprint.

| Category | Features |
| :--- | :--- |
| **🧠 Architecture** | • **Configurable Transformer:** Adjustable layers, heads, and dimensions.<br>• **Norms:** LayerNorm or RMSNorm.<br>• **Activations:** GELU or SwiGLU.<br>• **Positional:** Learned embeddings or RoPE (Rotary).<br>• **Attention:** Sliding window, QK Norm, Parallel Blocks. |
| **🚂 Training** | • **Precision:** AMP (Automatic Mixed Precision) with BF16/FP16.<br>• **Optimizers:** AdamW, **8-bit AdamW** (bitsandbytes), Lion.<br>• **Scheduling:** Cosine, WSD (Warmup-Stable-Decay), Constant.<br>• **Regularization:** Dropout, label smoothing, z-loss, grad clipping. |
| **⚡ Efficiency** | • **torch.compile:** Full JIT compilation support.<br>• **Grad Checkpointing:** Trade compute for VRAM.<br>• **DDP:** Multi-GPU Distributed Data Parallel training.<br>• **Weight Tying:** Option to tie embedding and output weights. |
| **🔡 Tokenization** | • **Character-level:** Zero dependencies, simple.<br>• **BPE:** Byte-level BPE via HuggingFace tokenizers. |
| **🤖 Generation** | • **KV-Cache:** Fast autoregressive generation.<br>• **Sampling:** Temp, Top-k, Top-p (Nucleus), Repetition Penalty. |

---

## 🚀 Quick Start

### 1. Installation
Requires Python 3.10+ and PyTorch 2.0+.
```bash
# Core dependencies
pip install torch

# Optional (for BPE and 8-bit optimizers)
pip install tokenizers bitsandbytes
```

### 2. Prepare Data
Download a text file to train on (e.g., a book from Gutenberg).
```bash
curl -o data/tiny_corpus.txt https://www.gutenberg.org/files/1342/1342-0.txt
```

### 3. Train
Train a simple character-level model.
```bash
python train_tinyllm.py --data_path data/tiny_corpus.txt --out_dir out/my_model
```

### 4. Generate
Run inference using your trained model.
```bash
python train_tinyllm.py --generate_only --out_dir out/my_model --prompt "Once upon a time"
```

---

## 📖 Usage & Arguments

You can run `train_tinyllm.py` with command-line arguments or a config file.

### 💾 Data & I/O
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data_path` | `data/tiny_corpus.txt` | Path to training text file. |
| `--out_dir` | `out/tiny_char_gpt` | Output directory for checkpoints. |
| `--resume` | `False` | Resume training from latest checkpoint. |
| `--memmap_dataset` | `False` | Memory-map tokenized data (saves RAM). |

### 🧮 Model Architecture
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--n_layer` | `4` | Number of transformer layers. |
| `--n_head` | `4` | Number of attention heads. |
| `--n_embd` | `128` | Embedding dimension. |
| `--block_size` | `128` | Max sequence length (context window). |
| `--norm` | `layernorm` | `layernorm` or `rmsnorm`. |
| `--mlp` | `gelu` | `gelu` or `swiglu`. |
| `--pos_encoding` | `learned` | `learned` or `rope`. |

### ⚙️ Training Configuration
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--max_steps` | `2000` | Total training steps. |
| `--batch_size` | `64` | Batch size per device. |
| `--learning_rate` | `3e-4` | Peak learning rate. |
| `--optimizer` | `adamw` | `adamw`, `adamw8bit`, or `lion`. |
| `--amp` | `True` | Enable Mixed Precision. |
| `--compile` | `False` | Enable `torch.compile` (faster). |

*(See `--help` for the full list including dropout, gradient accumulation, and schedulers).*

---

## 🔧 Examples

### 📄 JSON Configuration
Instead of long CLI commands, use a JSON config:
```json
{
  "data_path": "data/shakespeare.txt",
  "out_dir": "out/shakespeare_gpt",
  "tokenizer": "char",
  "n_layer": 6,
  "n_head": 6,
  "n_embd": 384,
  "compile": true,
  "amp": true
}
```
Run it:
```bash
python train_tinyllm.py --config config/my_config.json
```

### 🦙 Modern "Llama-style" Architecture
Use RoPE, RMSNorm, and SwiGLU:
```bash
python train_tinyllm.py \
  --pos_encoding rope \
  --norm rmsnorm \
  --mlp swiglu \
  --qk_norm
```

### 🏎️ High-Performance BPE Training
Use Byte-Pair Encoding, Compilation, and Memory Mapping for larger datasets:
```bash
python train_tinyllm.py \
  --data_path data/large_corpus.txt \
  --tokenizer bpe --bpe_vocab_size 8000 \
  --n_layer 8 --n_head 8 --n_embd 512 \
  --memmap_dataset \
  --compile --amp
```

### 🌐 Multi-GPU (DDP)
```bash
torchrun --standalone --nproc_per_node=2 train_tinyllm.py \
  --data_path data/corpus.txt \
  --ddp \
  --batch_size 32 --grad_accum_steps 2
```

---

## 📊 Logging & Artifacts

The training loop provides real-time feedback:
*   **Logs:** Loss & LR every `--log_interval`.
*   **Eval:** Validation loss every `--eval_interval`.
*   **Samples:** Text generation every `--sample_interval`.

**Output Structure:**
```text
out/my_model/
├── ckpt.pt              # Latest checkpoint
├── ckpt_best.pt         # Best validation loss checkpoint
├── effective_config.json # Full config used for training
└── tokenizer.json       # BPE tokenizer data (if applicable)
```

---

## 📐 Model Size Reference

| Config | Params | VRAM (Train) | Use Case |
| :--- | :--- | :--- | :--- |
| **4L-4H-128E** | ~1M | <1GB | Quick CPU/Laptop experiments |
| **6L-6H-384E** | ~10M | ~2GB | Small but capable char-level models |
| **8L-8H-512E** | ~30M | ~4GB | Good for specialized BPE datasets |
| **12L-12H-768E** | ~85M | ~8GB | GPT-2 Small scale |

---

## 🧩 Advanced Features

<details>
<summary><strong>Exponential Moving Average (EMA)</strong></summary>

Maintains a moving average of model weights for potentially better generalization.
```bash
python train_tinyllm.py --ema --ema_decay 0.9999
```
</details>

<details>
<summary><strong>Curriculum Learning</strong></summary>

Gradually increases the context length (block size) during training to speed up convergence.
```bash
python train_tinyllm.py --curriculum --curriculum_start_block_size 32
```
</details>

<details>
<summary><strong>Custom Generation Params</strong></summary>

```bash
python train_tinyllm.py \
  --generate_only \
  --prompt "The future of AI is" \
  --temperature 0.8 --top_p 0.9 --kv_cache
```
</details>

---

## 🙏 Acknowledgments

This project is intended for educational purposes. Inspiration drawn from:
*   [nanoGPT](https://github.com/karpathy/nanoGPT) & [minGPT](https://github.com/karpathy/minGPT) by Andrej Karpathy.
*   The Llama and GPT-2 architectural papers.

## 📄 License

[MIT License](LICENSE)
