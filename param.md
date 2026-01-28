# `train_tinyllm.py` Parameters

This file lists the supported command-line parameters for `train_tinyllm.py`, plus the underlying config keys (same names, without the leading `--`).

## How configuration works

- `--config` points to a JSON file. If it exists (default: `config/tiny_char_gpt.json`), it is loaded first.
- Any CLI flag you pass overrides the JSON/config value.
- Any value not provided by CLI or config falls back to the internal defaults from `TrainConfig`.
- Boolean flags use the `argparse.BooleanOptionalAction` pattern where available: `--foo` / `--no-foo`.

## Config & I/O

- `--config` (str, default: `config/tiny_char_gpt.json`): JSON config file to load (if present).
- `--data_path` (str, default: `data/tiny_corpus.txt`): Path to a UTF-8 text file to train on.
- `--out_dir` (str, default: `out/tiny_char_gpt`): Output directory for checkpoints, tokenizer, memmap tokens, etc.
- `--resume` (flag, default: off): Resume training from `out_dir/ckpt.pt` (or `ckpt_best.pt` if needed).
- `--memmap_dataset` / `--no-memmap_dataset` (bool, default: `False`): Build/use `out_dir/tokens.bin` + `tokens_meta.json` to stream tokens from disk.

## Tokenizer

- `--tokenizer` (`char`|`bpe`, default: `char`): Tokenizer type.
- `--bpe_vocab_size` (int, default: `2000`): BPE vocab size (only used when `--tokenizer bpe`).
- `--bpe_min_frequency` (int, default: `2`): BPE merge min frequency (only used when `--tokenizer bpe`).

## Training Loop

- `--max_steps` (int, default: `2000`): Total optimizer steps.
- `--batch_size` (int, default: `64`): Micro-batch size per step (per rank when using DDP).
- `--grad_accum_steps` (int, default: `1`): Gradient accumulation steps (effective batch = `batch_size * grad_accum_steps * world_size`).
- `--block_size` (int, default: `128`): Context length (sequence length).

## DataLoader (multi-worker)

- `--dataloader_workers` (int, default: `0`): Number of DataLoader worker processes (0 = no workers).
- `--dataloader_prefetch_factor` (int, default: `2`): Prefetch factor for workers (only used when `dataloader_workers > 0`).
- `--dataloader_persistent_workers` / `--no-dataloader_persistent_workers` (bool, default: `True`): Keep workers alive between iterations (only used when `dataloader_workers > 0`).

## Optimization

- `--learning_rate` (float, default: `3e-4`): Peak/base learning rate (depends on schedule).
- `--lr_schedule` (`constant`|`cosine`|`wsd`, default: `cosine`): LR schedule.
- `--lr_warmup_steps` (int, default: `200`): Warmup steps.
- `--lr_stable_steps` (int, default: `0`): Stable steps (only used for `--lr_schedule wsd`).
- `--lr_min` (float, default: `3e-5`): Minimum LR after decay.
- `--optimizer` (`adamw`|`adamw8bit`|`lion`, default: `adamw`): Optimizer type.
- `--lion_beta1` (float, default: `0.9`): Lion beta1 (only used when `--optimizer lion`).
- `--lion_beta2` (float, default: `0.99`): Lion beta2 (only used when `--optimizer lion`).
- `--weight_decay` (float, default: `0.1`): Weight decay (decoupled for AdamW-style optimizers).
- `--grad_clip` (float, default: `1.0`): Global gradient clipping max norm (0 disables).
- `--grad_checkpointing` / `--no-grad_checkpointing` (bool, default: `False`): Gradient checkpointing (reduces VRAM, slower).
- `--label_smoothing` (float, default: `0.0`): Cross-entropy label smoothing in `[0, 1)`.

## EMA (Exponential Moving Average)

- `--ema` / `--no-ema` (bool, default: `False`): Track an EMA of weights and use it for eval/sampling.
- `--ema_decay` (float, default: `0.9999`): EMA decay factor.
- `--ema_update_every` (int, default: `1`): Update EMA every N optimizer steps.
- `--ema_start_step` (int, default: `0`): Start updating EMA at this step.
- `--use_ema` / `--no-use_ema` (bool, default: `auto`): For `--generate_only`, use EMA weights if available (auto = use if checkpoint has EMA).

## Curriculum Learning

- `--curriculum` / `--no-curriculum` (bool, default: `False`): Ramp the effective sequence length up over time.
- `--curriculum_start_block_size` (int, default: `32`): Starting sequence length when curriculum is enabled.
- `--curriculum_steps` (int, default: `2000`): Number of steps to ramp from start length to `--block_size`.

## Performance

- `--amp` / `--no-amp` (bool, default: `True`): Mixed precision training on CUDA (recommended).
- `--compile` / `--no-compile` (bool, default: `False`): Enable `torch.compile`.
- `--compile_backend` (`inductor`|`aot_eager`|`eager`, default: `inductor`): `torch.compile` backend.
- `--compile_mode` (`default`|`reduce-overhead`|`max-autotune`, default: `default`): `torch.compile` mode.
- `--compile_fullgraph` / `--no-compile_fullgraph` (bool, default: `False`): Pass `fullgraph=True` to `torch.compile`.
- `--compile_dynamic` / `--no-compile_dynamic` (bool, default: `unset/auto`): Pass `dynamic=True/False` to `torch.compile` (unset means PyTorch default).

## Eval & Logging

- `--eval_interval` (int, default: `200`): Evaluate every N steps (0 disables).
- `--eval_iters` (int, default: `50`): Number of eval batches to average for train/val losses.
- `--log_interval` (int, default: `50`): Log training loss every N steps.
- `--sample_interval` (int, default: `200`): Generate a sample every N steps (0 disables).
- `--sample_chars` (int, default: `400`): Tokens/chars to generate for samples and generation.
- `--prompt` (str, default: empty): Prompt for sampling and `--generate_only`.

## Model Architecture

- `--n_layer` (int, default: `4`): Transformer blocks.
- `--n_head` (int, default: `4`): Attention heads.
- `--n_embd` (int, default: `128`): Embedding dimension.
- `--dropout` (float, default: `0.1`): Dropout probability.
- `--norm` (`layernorm`|`rmsnorm`, default: `layernorm`): Normalization type.
- `--norm_eps` (float, default: `1e-5`): Norm epsilon.
- `--mlp` (`gelu`|`swiglu`, default: `gelu`): Feed-forward activation.
- `--layerdrop` (float, default: `0.0`): LayerDrop probability in `[0, 1)`.
- `--pos_encoding` (`learned`|`rope`, default: `learned`): Positional encoding type.
- `--rope_base` (float, default: `10000.0`): RoPE base (only used when `--pos_encoding rope`).
- `--sliding_window` (int, default: `0`): Sliding-window attention window size (0 disables).
- `--qk_norm` / `--no-qk_norm` (bool, default: `False`): Apply QK-Norm in attention.
- `--qk_norm_eps` (float, default: `1e-5`): QK-Norm epsilon.
- `--scaled_init` / `--no-scaled_init` (bool, default: `True`): Scale residual projection init by `1/sqrt(2*n_layer)` (GPT-2 style).
- `--weight_tying` / `--no-weight_tying` (bool, default: `False`): Tie token embedding and output projection weights.

## Runtime & Distributed (DDP)

- `--device` (`auto`|`cpu`|`cuda`, default: `auto`): Device selection.
- `--ddp` / `--no-ddp` (bool, default: `False`): Enable DistributedDataParallel (launch with `torchrun`).
- `--ddp_backend` (`auto`|`nccl`|`gloo`, default: `auto`): DDP backend.
- `--ddp_find_unused_parameters` / `--no-ddp_find_unused_parameters` (bool, default: `False`): Pass `find_unused_parameters=True` to DDP.
- `--ddp_timeout` (int, default: `1800`): DDP init timeout (seconds).
- `--seed` (int, default: `1337`): Random seed.

## Generation-only Parameters

Enable generation-only mode:

- `--generate_only` (flag, default: off): Load checkpoint from `--out_dir` and generate text (no training).

Sampling controls (also used for periodic samples during training):

- `--temperature` (float, default: `1.0`): Softmax temperature.
- `--top_k` (int, default: `0`): Top-k sampling (0 disables).
- `--top_p` (float, default: `1.0`): Nucleus sampling (1.0 disables).
- `--repetition_penalty` (float, default: `1.0`): Penalty for repeating recent tokens (1.0 disables).

