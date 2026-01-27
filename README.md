# Tiny LLM (Character-Level) Training

This repo trains a **tiny GPT-style Transformer** that predicts the next **character** in a text file.

## GUI Interface

You can use the graphical interface to configure training and generation:

```bash
python gui.py
```

The GUI allows you to:
- Pick data files and output directories.
- Configure model architecture and training hyperparameters.
- Monitor training progress in real-time.
- Generate text from trained models with a simple interface.

## Quickstart

Train on the included corpus:

```bash
python train_tinyllm.py
```

The default settings come from `config/tiny_char_gpt.json` and the dataset is `data/tiny_corpus.txt`.
Checkpoints and configs are written to `out/tiny_char_gpt/`.

## Faster Smoke Test

```bash
python train_tinyllm.py --out_dir out/smoke --max_steps 50 --n_layer 2 --n_head 2 --n_embd 64 --block_size 64 --batch_size 16
```

## BPE Tokenizer (Better Text)

Character-level models learn slowly; a **BPE tokenizer** usually gives more coherent output.

Train with BPE (requires the `tokenizers` package):

```bash
pip install tokenizers
python train_tinyllm.py --tokenizer bpe --bpe_vocab_size 8000 --data_path data/wikitext2.txt --out_dir out/wikitext2_bpe
```

This writes `tokenizer.json` into the `out_dir` (and also stores it inside the checkpoint).

## Clean WikiText (Recommended)

WikiText raw contains markup tokens like `@-@` and headings like `== Title ==`. Clean it first:

```bash
python prepare_wikitext.py --input data/wiki.train.TXT --output data/wiki.clean.txt
```

Then train on `data/wiki.clean.txt`.

### Speed Tips (RTX 4060)

- Use `--amp` (mixed precision) for a big speedup on CUDA.
- Try `--compile` (PyTorch 2.x) for extra speed after the first-step compile overhead. If `inductor` fails with a Triton error on Windows, use `--compile_backend aot_eager` (or install a working `triton`).
- If you’re memory-bound, try `--optimizer adamw8bit` (requires `bitsandbytes`, CUDA-only).
- If you run out of VRAM, try `--grad_checkpointing` (slower, but uses less memory).
- For large datasets, use `--memmap_dataset` to tokenize once into `out_dir/tokens.bin` and stream batches without loading all tokens into RAM/VRAM.
- If your GPU is underutilized, try `--dataloader_workers 2` (or 4) to sample batches in parallel (works best with `--memmap_dataset`).
- Reduce eval overhead while iterating: `--eval_interval 0 --sample_interval 0`.
- Bigger `block_size` and bigger models get expensive fast; start smaller and scale up.
- Learning rate uses warmup+cosine by default; try `--lr_schedule wsd --lr_stable_steps 2000` for warmup → stable → decay (tune `--lr_warmup_steps` / `--lr_min` too).
- If you run out of VRAM, keep `--batch_size` small and use `--grad_accum_steps N` to get a larger effective batch.

### Model Upgrades

- `--weight_tying` ties embedding/output weights (fewer params).
- `--mlp swiglu` uses SwiGLU MLPs (often better quality).
- `--pos_encoding rope` uses rotary position embeddings (RoPE) instead of learned absolute embeddings.
- `--norm rmsnorm` uses RMSNorm instead of LayerNorm (often a small win).
- `--qk_norm` applies RMS normalization to Q/K (can help stability).
- `--parallel_block` computes attention+MLP in parallel (parallel residual).
- `--sliding_window 256` limits attention to the last N tokens.
- `--layerdrop 0.1` randomly drops layers during training (regularization).

### Training Extras

- `--label_smoothing 0.1` (regularization)
- `--z_loss 1e-4` (logit stabilization)
- `--optimizer lion` (alternative optimizer)
- `--ema --ema_decay 0.9999` (EMA weights for eval/sampling)
- `--curriculum --curriculum_start_block_size 64 --curriculum_steps 5000` (sequence-length curriculum)

## Distributed (Multi-GPU) Training (DDP)

If you have multiple GPUs, launch with `torchrun` and enable DDP:

```bash
torchrun --standalone --nproc_per_node=2 train_tinyllm.py --ddp --out_dir out/ddp_run --data_path data/wiki.clean.txt --tokenizer bpe --memmap_dataset --amp
```

Notes:

- `--batch_size` is **per GPU**. Effective batch is `batch_size * grad_accum_steps * world_size`.
- Only rank 0 writes checkpoints/logs to `--out_dir`.
- Backend defaults to `--ddp_backend auto` (NCCL on Linux+CUDA, otherwise Gloo).
- If `torchrun` prints an `OMP_NUM_THREADS` warning, set it yourself before launching, e.g. `OMP_NUM_THREADS=1 torchrun ...`.
- If you see noisy `[c10d] hostname ...` warnings in some notebook/container environments, it’s usually harmless (reverse DNS). You can silence C++ warnings with `TORCH_CPP_LOG_LEVEL=ERROR`.

## Generate Text

After training, sample from the latest checkpoint:

```bash
python train_tinyllm.py --out_dir out/tiny_char_gpt --generate_only --prompt "the cat " --sample_chars 300
```

`--out_dir` must point at a folder that contains `ckpt.pt`. When generating/resuming, the script loads the
model config + tokenizer from the checkpoint (so you don’t need to pass `--data_path`, and you can’t change
shape-related flags like `--block_size` / `--n_embd` for that checkpoint).

`ckpt.pt` is always written at the end of training (and on Ctrl+C). If eval is enabled, the best-so-far
checkpoint is also saved as `ckpt_best.pt`.

`--sample_chars` means “new tokens”: characters for `--tokenizer char`, tokens for `--tokenizer bpe`.

Generation knobs:

- `--top_p 0.9` (nucleus sampling)
- `--repetition_penalty 1.1` (reduce loops)
- `--kv_cache` (faster sampling)

## Train On Your Own Text

Put a UTF-8 text file anywhere and point the trainer at it:

```bash
python train_tinyllm.py --data_path path/to/your.txt --out_dir out/my_run
```

## Notes

- This is intentionally small and simple; it’s meant for learning the workflow.
- Because it’s character-level, output quality depends heavily on how much text you give it.
