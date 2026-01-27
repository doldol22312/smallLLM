import argparse
import json
import hashlib
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainConfig:
    # Repro / IO
    seed: int = 1337
    data_path: str = "data/tiny_corpus.txt"
    out_dir: str = "out/tiny_char_gpt"
    resume: bool = False
    memmap_dataset: bool = False

    # Tokenizer
    tokenizer: str = "char"  # char|bpe
    bpe_vocab_size: int = 2000
    bpe_min_frequency: int = 2

    # Training
    max_steps: int = 2000
    batch_size: int = 64
    grad_accum_steps: int = 1
    block_size: int = 128
    learning_rate: float = 3e-4
    lr_schedule: str = "cosine"  # constant|cosine
    lr_warmup_steps: int = 200
    lr_min: float = 3e-5
    optimizer: str = "adamw"  # adamw|lion
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    grad_checkpointing: bool = False
    amp: bool = True
    compile: bool = False
    compile_backend: str = "inductor"  # inductor|aot_eager|eager
    compile_mode: str = "default"  # default|reduce-overhead|max-autotune
    compile_fullgraph: bool = False
    compile_dynamic: bool | None = None

    # Evaluation / sampling
    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 50
    sample_interval: int = 200
    sample_chars: int = 400
    prompt: str = ""

    # Model
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    norm: str = "layernorm"  # layernorm|rmsnorm
    norm_eps: float = 1e-5
    mlp: str = "gelu"  # gelu|swiglu
    parallel_block: bool = False
    layerdrop: float = 0.0
    pos_encoding: str = "learned"  # learned|rope
    rope_base: float = 10000.0
    sliding_window: int = 0
    qk_norm: bool = False
    qk_norm_eps: float = 1e-5
    scaled_init: bool = True
    weight_tying: bool = False

    # Runtime
    device: str = "auto"  # auto|cpu|cuda
    ddp: bool = False
    ddp_backend: str = "auto"  # auto|nccl|gloo
    ddp_find_unused_parameters: bool = False
    ddp_timeout: int = 1800
    kv_cache: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    label_smoothing: float = 0.0
    z_loss: float = 0.0


@dataclass(frozen=True)
class DdpInfo:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "gloo"

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


class Lion(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            wd = float(group.get("weight_decay", 0.0))

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                m = state["exp_avg"]
                if wd:
                    p.mul_(1.0 - lr * wd)

                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                p.add_(m.sign(), alpha=-lr)
                m.mul_(beta2).add_(g, alpha=1.0 - beta2)

        return loss


class CharTokenizer:
    def __init__(self, text: str):
        vocab = sorted(set(text))
        self.itos = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}

    @classmethod
    def from_itos(cls, itos: list[str]) -> "CharTokenizer":
        obj = cls.__new__(cls)
        obj.itos = list(itos)
        obj.stoi = {ch: i for i, ch in enumerate(obj.itos)}
        return obj

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        out: list[int] = []
        for c in s:
            if c not in self.stoi:
                raise ValueError(f"Character {c!r} is not in the tokenizer vocabulary.")
            out.append(self.stoi[c])
        return out

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class BpeTokenizer:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    @classmethod
    def train(
        cls,
        *,
        data_path: str,
        vocab_size: int,
        min_frequency: int,
    ) -> "BpeTokenizer":
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "BPE tokenization requires the `tokenizers` package. Install with: pip install tokenizers"
            ) from e

        bpe = ByteLevelBPETokenizer()
        bpe.train(
            files=[data_path],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
        )
        return cls(bpe._tokenizer)

    @classmethod
    def from_json(cls, json_str: str) -> "BpeTokenizer":
        try:
            from tokenizers import Tokenizer
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "BPE tokenization requires the `tokenizers` package. Install with: pip install tokenizers"
            ) from e

        return cls(Tokenizer.from_str(json_str))

    def to_json(self) -> str:
        return self._tokenizer.to_str()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, s: str) -> list[int]:
        return self._tokenizer.encode(s).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        encs = self._tokenizer.encode_batch(texts)
        return [enc.ids for enc in encs]

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        n_embd = cfg.n_embd
        n_head = cfg.n_head
        dropout = cfg.dropout

        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.sliding_window = int(cfg.sliding_window) if cfg.sliding_window else 0
        self.max_cache_len = (
            min(self.sliding_window, cfg.block_size) if self.sliding_window > 0 else cfg.block_size
        )
        if self.sliding_window > 0:
            i = torch.arange(cfg.block_size)[:, None]
            j = torch.arange(cfg.block_size)[None, :]
            mask = (j <= i) & ((i - j) < self.sliding_window)
            self.register_buffer("sliding_attn_mask", mask, persistent=False)

        self.qk_norm = bool(cfg.qk_norm)
        self.qk_norm_eps = float(cfg.qk_norm_eps)

        self.use_rope = cfg.pos_encoding == "rope"
        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError("RoPE requires head_dim to be even")
            inv_freq = 1.0 / (
                cfg.rope_base
                ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            )
            t = torch.arange(cfg.block_size, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)  # (block_size, head_dim/2)
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
            self.register_buffer("rope_cos", freqs.cos(), persistent=False)
            self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    def _get_rope_cos_sin(
        self, seq_len: int, *, start_pos: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end = start_pos + seq_len
        if 0 <= start_pos and end <= self.rope_cos.size(0):
            cos = self.rope_cos[start_pos:end].to(dtype=dtype)[None, None, :, :]
            sin = self.rope_sin[start_pos:end].to(dtype=dtype)[None, None, :, :]
            return cos, sin

        t = torch.arange(start_pos, end, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.rope_inv_freq.to(device=device))
        cos = freqs.cos().to(dtype=dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

    def _apply_qk_norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.qk_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(embed_dim, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = self._get_rope_cos_sin(
                seq_len, start_pos=0, device=x.device, dtype=q.dtype
            )
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

        if self.qk_norm:
            q = self._apply_qk_norm(q)
            k = self._apply_qk_norm(k)

        attn_mask = None
        if self.sliding_window > 0 and self.sliding_window < seq_len:
            attn_mask = self.sliding_attn_mask[:seq_len, :seq_len]

        # (B, nh, T, hs) -> (B, nh, T, hs) with causal masking handled by SDPA
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        y = self.resid_dropout(self.proj(y))
        return y

    def forward_with_kv_cache(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        start_pos: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(embed_dim, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = self._get_rope_cos_sin(
                seq_len, start_pos=start_pos, device=x.device, dtype=q.dtype
            )
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

        if self.qk_norm:
            q = self._apply_qk_norm(q)
            k = self._apply_qk_norm(k)

        attn_mask = None
        if kv_cache is None and self.sliding_window > 0 and self.sliding_window < seq_len:
            attn_mask = self.sliding_attn_mask[:seq_len, :seq_len]

        if kv_cache is not None:
            if seq_len != 1:
                raise ValueError("KV cache only supports seq_len=1 (one token at a time)")
            k_cache, v_cache = kv_cache
            k = torch.cat((k_cache, k), dim=2)
            v = torch.cat((v_cache, v), dim=2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=kv_cache is None,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        y = self.resid_dropout(self.proj(y))
        if self.max_cache_len and k.size(2) > self.max_cache_len:
            k = k[:, :, -self.max_cache_len :, :]
            v = v[:, :, -self.max_cache_len :, :]
        return y, (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.mlp = cfg.mlp
        hidden = 4 * cfg.n_embd
        if self.mlp == "gelu":
            self.fc = nn.Linear(cfg.n_embd, hidden)
            self.proj = nn.Linear(hidden, cfg.n_embd)
        elif self.mlp == "swiglu":
            self.fc = nn.Linear(cfg.n_embd, 2 * hidden)
            self.proj = nn.Linear(hidden, cfg.n_embd)
        else:
            raise ValueError(f"Unknown mlp type: {self.mlp!r}")
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp == "gelu":
            x = F.gelu(self.fc(x))
        else:
            gate, up = self.fc(x).chunk(2, dim=-1)
            x = F.silu(gate) * up

        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: TrainConfig, *, layer_idx: int):
        super().__init__()
        if cfg.norm == "layernorm":
            self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
            self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
        elif cfg.norm == "rmsnorm":
            self.ln_1 = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
            self.ln_2 = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        else:
            raise ValueError(f"Unknown norm type: {cfg.norm!r}")
        self.parallel = bool(cfg.parallel_block)
        if cfg.layerdrop and cfg.layerdrop > 0:
            if cfg.n_layer > 1:
                self.layerdrop = float(cfg.layerdrop) * float(layer_idx) / float(cfg.n_layer - 1)
            else:
                self.layerdrop = float(cfg.layerdrop)
        else:
            self.layerdrop = 0.0
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layerdrop and self.training:
            if torch.rand((), device=x.device) < self.layerdrop:
                return x

        if self.parallel:
            x = x + self.attn(self.ln_1(x)) + self.mlp(self.ln_2(x))
            return x

        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_with_kv_cache(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        start_pos: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.parallel:
            attn_out, new_kv = self.attn.forward_with_kv_cache(
                self.ln_1(x), kv_cache, start_pos=start_pos
            )
            x = x + attn_out + self.mlp(self.ln_2(x))
            return x, new_kv

        attn_out, new_kv = self.attn.forward_with_kv_cache(
            self.ln_1(x), kv_cache, start_pos=start_pos
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = (
            nn.Embedding(cfg.block_size, cfg.n_embd)
            if cfg.pos_encoding == "learned"
            else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg, layer_idx=i) for i in range(cfg.n_layer)]
        )
        if cfg.norm == "layernorm":
            self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
        elif cfg.norm == "rmsnorm":
            self.ln_f = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        else:
            raise ValueError(f"Unknown norm type: {cfg.norm!r}")
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

        if cfg.scaled_init and cfg.n_layer > 0:
            scale = 1.0 / math.sqrt(2.0 * cfg.n_layer)
            for block in self.blocks:
                block.attn.proj.weight.data.mul_(scale)
                block.mlp.proj.weight.data.mul_(scale)

        if cfg.weight_tying:
            self.lm_head.weight = self.token_emb.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits, loss, _ = self._forward_impl(idx, targets)
        return logits, loss

    def _forward_impl(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None,
        *,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        start_pos: int = 0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        list[tuple[torch.Tensor, torch.Tensor]] | None,
    ]:
        batch_size, seq_len = idx.shape
        if seq_len > self.cfg.block_size:
            raise ValueError("Sequence length exceeds block_size")

        x = self.token_emb(idx)
        if self.pos_emb is not None:
            if start_pos < 0 or start_pos + seq_len > self.cfg.block_size:
                raise ValueError("Sequence positions exceed block_size")
            positions = torch.arange(start_pos, start_pos + seq_len, device=idx.device)
            x = x + self.pos_emb(positions)
        x = self.drop(x)

        new_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if kv_cache is None:
            for block in self.blocks:
                if (
                    self.cfg.grad_checkpointing
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
        else:
            if len(kv_cache) != len(self.blocks):
                raise ValueError("kv_cache must have one entry per Transformer block")
            new_kv_cache = []
            for block, layer_cache in zip(self.blocks, kv_cache):
                x, out_cache = block.forward_with_kv_cache(
                    x, layer_cache, start_pos=start_pos
                )
                new_kv_cache.append(out_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                label_smoothing=float(self.cfg.label_smoothing or 0.0),
            )
            if self.cfg.z_loss and self.cfg.z_loss > 0:
                log_z = torch.logsumexp(logits.float(), dim=-1)
                loss = loss + float(self.cfg.z_loss) * (log_z**2).mean()
        return logits, loss, new_kv_cache

    @torch.no_grad()
    def forward_with_kv_cache(
        self,
        idx: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None],
        *,
        start_pos: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        logits, _, new_kv = self._forward_impl(idx, None, kv_cache=kv_cache, start_pos=start_pos)
        assert new_kv is not None
        return logits, new_kv

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        use_kv_cache: bool | None = None,
    ) -> torch.Tensor:
        if use_kv_cache is None:
            use_kv_cache = bool(self.cfg.kv_cache)

        if not use_kv_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.cfg.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]

                if repetition_penalty and repetition_penalty != 1.0:
                    for b in range(logits.size(0)):
                        token_ids = torch.unique(idx_cond[b])
                        token_logits = logits[b, token_ids]
                        logits[b, token_ids] = torch.where(
                            token_logits > 0,
                            token_logits / repetition_penalty,
                            token_logits * repetition_penalty,
                        )

                logits = logits / max(temperature, 1e-8)

                if top_k and top_k > 0:
                    values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = -float("inf")

                if top_p and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = sorted_probs.cumsum(dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                        :, :-1
                    ].clone()
                    sorted_indices_to_remove[:, 0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, -float("inf"))

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)
            return idx

        if self.pos_emb is not None and idx.size(1) + max_new_tokens > self.cfg.block_size:
            # Learned absolute embeddings can't extrapolate positions. Fall back to non-cached generation.
            return self.generate(
                idx,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_kv_cache=False,
            )

        kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.blocks)
        full_prompt_len = idx.size(1)
        idx_cond = idx[:, -self.cfg.block_size :]
        start_pos = 0 if self.pos_emb is not None else full_prompt_len - idx_cond.size(1)
        logits, kv_cache_filled = self.forward_with_kv_cache(
            idx_cond, kv_cache, start_pos=start_pos
        )
        kv_cache = kv_cache_filled

        pos_cursor = full_prompt_len
        for _ in range(max_new_tokens):
            logits_step = logits[:, -1, :]

            idx_cond = idx[:, -self.cfg.block_size :]
            if repetition_penalty and repetition_penalty != 1.0:
                for b in range(logits_step.size(0)):
                    token_ids = torch.unique(idx_cond[b])
                    token_logits = logits_step[b, token_ids]
                    logits_step[b, token_ids] = torch.where(
                        token_logits > 0,
                        token_logits / repetition_penalty,
                        token_logits * repetition_penalty,
                    )

            logits_step = logits_step / max(temperature, 1e-8)

            if top_k and top_k > 0:
                values, _ = torch.topk(logits_step, k=min(top_k, logits_step.size(-1)))
                logits_step[logits_step < values[:, [-1]]] = -float("inf")

            if top_p and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits_step, descending=True, dim=-1
                )
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = sorted_probs.cumsum(dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits_step = logits_step.masked_fill(indices_to_remove, -float("inf"))

            probs = F.softmax(logits_step, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            step_start_pos = pos_cursor
            logits, kv_cache = self.forward_with_kv_cache(
                next_token, kv_cache, start_pos=step_start_pos
            )
            pos_cursor += 1

        return idx


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_text_stream(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def _build_char_tokenizer_from_file(path: str) -> CharTokenizer:
    vocab: set[str] = set()
    for chunk in _read_text_stream(path):
        vocab.update(chunk)
    return CharTokenizer.from_itos(sorted(vocab))


def _tokenizer_fingerprint(tokenizer: CharTokenizer | BpeTokenizer) -> str:
    if isinstance(tokenizer, CharTokenizer):
        payload = "".join(tokenizer.itos).encode("utf-8", errors="replace")
    else:
        payload = tokenizer.to_json().encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()


def _memmap_paths(out_dir: str) -> tuple[str, str]:
    return (
        os.path.join(out_dir, "tokens.bin"),
        os.path.join(out_dir, "tokens_meta.json"),
    )


def _memmap_dtype_for_vocab(vocab_size: int) -> tuple[str, torch.dtype, int]:
    # Keep file compact, but use dtypes that support fast CPU indexing in PyTorch.
    if vocab_size <= 32768:
        return "int16", torch.int16, 2
    if vocab_size <= 2**31:
        return "int32", torch.int32, 4
    raise ValueError(f"vocab_size={vocab_size} is too large for int32 memmap storage.")


def _build_tokens_memmap(
    *,
    tokenizer: CharTokenizer | BpeTokenizer,
    data_path: str,
    out_bin_path: str,
    dtype_str: str,
    batch_lines: int = 1024,
) -> int:
    import numpy as np

    if dtype_str == "int16":
        dtype_np = np.int16
    elif dtype_str == "int32":
        dtype_np = np.int32
    else:
        raise ValueError(f"Unknown memmap dtype: {dtype_str!r}")

    total = 0
    os.makedirs(os.path.dirname(out_bin_path) or ".", exist_ok=True)
    with open(out_bin_path, "wb") as out_f, open(
        data_path, "r", encoding="utf-8", errors="replace"
    ) as in_f:
        batch: list[str] = []
        for line in in_f:
            batch.append(line)
            if len(batch) >= batch_lines:
                ids_batch = tokenizer.encode_batch(batch)
                flat: list[int] = []
                for ids in ids_batch:
                    flat.extend(ids)
                if flat:
                    np.asarray(flat, dtype=dtype_np).tofile(out_f)
                    total += len(flat)
                batch.clear()

        if batch:
            ids_batch = tokenizer.encode_batch(batch)
            flat = []
            for ids in ids_batch:
                flat.extend(ids)
            if flat:
                np.asarray(flat, dtype=dtype_np).tofile(out_f)
                total += len(flat)

    return total


def _load_or_build_memmap_tokens(
    *,
    cfg: TrainConfig,
    tokenizer: CharTokenizer | BpeTokenizer,
) -> torch.Tensor:
    bin_path, meta_path = _memmap_paths(cfg.out_dir)
    dtype_str, torch_dtype, dtype_bytes = _memmap_dtype_for_vocab(tokenizer.vocab_size)
    abs_data_path = os.path.abspath(cfg.data_path)
    data_exists = os.path.exists(cfg.data_path)
    mtime = os.path.getmtime(cfg.data_path) if data_exists else None
    fsize = os.path.getsize(cfg.data_path) if data_exists else None
    tok_fp = _tokenizer_fingerprint(tokenizer)

    meta_ok = False
    length = 0
    if os.path.exists(bin_path) and os.path.exists(meta_path):
        try:
            meta = _load_json(meta_path)
            length = int(meta.get("length", 0))
            expected_bytes = length * dtype_bytes
            meta_ok = (
                meta.get("version") == 1
                and meta.get("dtype") == dtype_str
                and meta.get("vocab_size") == tokenizer.vocab_size
                and meta.get("data_path") == abs_data_path
                and meta.get("tokenizer_sha1") == tok_fp
                and os.path.getsize(bin_path) >= expected_bytes
            )
            if meta_ok and data_exists:
                meta_ok = (
                    float(meta.get("data_mtime")) == float(mtime)
                    and int(meta.get("data_size")) == int(fsize)
                )
        except Exception:
            meta_ok = False

    if not meta_ok:
        if not data_exists:
            raise FileNotFoundError(
                f"Missing dataset at {cfg.data_path!r} and cached memmap tokens are invalid/missing."
            )
        print(f"building memmap dataset tokens at {bin_path} (one-time)...")
        length = _build_tokens_memmap(
            tokenizer=tokenizer,
            data_path=cfg.data_path,
            out_bin_path=bin_path,
            dtype_str=dtype_str,
        )
        _write_json(
            meta_path,
            {
                "version": 1,
                "dtype": dtype_str,
                "length": length,
                "vocab_size": tokenizer.vocab_size,
                "data_path": abs_data_path,
                "data_mtime": mtime,
                "data_size": fsize,
                "tokenizer_sha1": tok_fp,
            },
        )
        print(f"memmap tokens: {length:,} ({dtype_str})")
    elif not data_exists:
        print(f"warning: dataset {cfg.data_path!r} not found; using cached memmap tokens.")

    return torch.from_file(bin_path, dtype=torch_dtype, size=length)


def _write_json(path: str, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="backslashreplace").decode(encoding)
        print(safe_text)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _device_from_cfg(device: str) -> str:
    device = device.lower()
    if device in {"cpu", "cuda"}:
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _init_ddp(*, cfg: TrainConfig, device_str: str) -> DdpInfo:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = bool(cfg.ddp) or world_size > 1
    if not enabled:
        return DdpInfo()

    if world_size <= 1:
        raise RuntimeError(
            "DDP enabled but WORLD_SIZE=1. Launch with torchrun, e.g.:\n"
            "  torchrun --standalone --nproc_per_node=2 train_tinyllm.py --ddp ..."
        )

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    try:
        import torch.distributed as dist
    except Exception as e:
        raise RuntimeError(
            "DDP requested but torch.distributed is unavailable in this PyTorch build."
        ) from e

    backend_cfg = (cfg.ddp_backend or "auto").lower()
    if backend_cfg == "auto":
        backend = (
            "nccl"
            if device_str == "cuda" and dist.is_nccl_available()
            else "gloo"
        )
    else:
        backend = backend_cfg

    if backend == "nccl" and not dist.is_nccl_available():
        if rank == 0:
            print("warning: NCCL backend not available; falling back to gloo.")
        backend = "gloo"

    # Set per-process device before initializing NCCL.
    if device_str == "cuda":
        torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend, init_method="env://", timeout=timedelta(seconds=int(cfg.ddp_timeout))
    )
    return DdpInfo(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
    )


def _get_batch(
    data: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = int(data.size(0)) - block_size - 1
    if max_start <= 0:
        raise ValueError("Dataset is too small for the requested block_size")

    idx_device = data.device
    starts = torch.randint(0, max_start, (batch_size,), device=idx_device)
    offsets = torch.arange(block_size, device=idx_device)
    idx = starts[:, None] + offsets[None, :]
    x = data[idx]
    y = data[idx + 1]

    if x.device != device:
        if x.device.type == "cpu" and device.type == "cuda":
            x = x.pin_memory()
            y = y.pin_memory()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)

    if x.dtype != torch.long:
        x = x.long()
    if y.dtype != torch.long:
        y = y.long()
    return x, y


@torch.no_grad()
def _estimate_loss(
    model: TinyGPT,
    *,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters, device=device)
        for i in range(cfg.eval_iters):
            x, y = _get_batch(
                split_data,
                batch_size=cfg.batch_size,
                block_size=cfg.block_size,
                device=device,
            )
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                _, loss = model(x, y)
            losses[i] = loss
        out[split_name] = losses.mean().item()
    model.train()
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny Transformer LLM (character-level or BPE)."
    )
    parser.add_argument("--config", default="config/tiny_char_gpt.json")

    parser.add_argument("--data_path", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--memmap_dataset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tokenize dataset once to out_dir/tokens.bin and memory-map it (saves RAM/VRAM).",
    )

    parser.add_argument("--tokenizer", default=None, choices=["char", "bpe"])
    parser.add_argument("--bpe_vocab_size", type=int, default=None)
    parser.add_argument("--bpe_min_frequency", type=int, default=None)

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable mixed-precision training on CUDA (recommended for speed).",
    )

    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lr_schedule", default=None, choices=["constant", "cosine"])
    parser.add_argument("--lr_warmup_steps", type=int, default=None)
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--optimizer", default=None, choices=["adamw", "lion"])
    parser.add_argument("--lion_beta1", type=float, default=None)
    parser.add_argument("--lion_beta2", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument(
        "--grad_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable gradient checkpointing to reduce VRAM (slower).",
    )
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--z_loss", type=float, default=None)

    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use torch.compile (PyTorch 2.x) to speed up training (first step may be slower).",
    )
    parser.add_argument(
        "--compile_backend",
        default=None,
        choices=["inductor", "aot_eager", "eager"],
        help="torch.compile backend (inductor/aot_eager/eager).",
    )
    parser.add_argument(
        "--compile_mode",
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default/reduce-overhead/max-autotune).",
    )
    parser.add_argument(
        "--compile_fullgraph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pass fullgraph=True to torch.compile (may fail if graph breaks).",
    )
    parser.add_argument(
        "--compile_dynamic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pass dynamic=True/False to torch.compile (leave unset for default).",
    )

    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--eval_iters", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--sample_interval", type=int, default=None)

    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--norm", default=None, choices=["layernorm", "rmsnorm"])
    parser.add_argument("--norm_eps", type=float, default=None)
    parser.add_argument("--mlp", default=None, choices=["gelu", "swiglu"])
    parser.add_argument(
        "--parallel_block",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute attention and MLP in parallel (parallel residual).",
    )
    parser.add_argument("--layerdrop", type=float, default=None)
    parser.add_argument("--pos_encoding", default=None, choices=["learned", "rope"])
    parser.add_argument("--rope_base", type=float, default=None)
    parser.add_argument("--sliding_window", type=int, default=None)
    parser.add_argument(
        "--qk_norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply RMS normalization to Q and K (QK-Norm).",
    )
    parser.add_argument("--qk_norm_eps", type=float, default=None)
    parser.add_argument(
        "--scaled_init",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Scale residual projection init by 1/sqrt(2*n_layer) (GPT-2 style).",
    )
    parser.add_argument(
        "--weight_tying",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tie token embedding and output projection weights (saves params).",
    )

    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--ddp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable DistributedDataParallel (launch with torchrun).",
    )
    parser.add_argument(
        "--ddp_backend",
        default=None,
        choices=["auto", "nccl", "gloo"],
        help="DDP backend (auto/nccl/gloo).",
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pass find_unused_parameters=True to DDP (slower).",
    )
    parser.add_argument(
        "--ddp_timeout",
        type=int,
        default=None,
        help="torch.distributed init timeout (seconds).",
    )
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--sample_chars", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument(
        "--kv_cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use a KV-cache during generation for faster sampling.",
    )

    return parser.parse_args()


def _merge_cfg(base: TrainConfig, overrides: dict) -> TrainConfig:
    merged = asdict(base)
    for key, value in overrides.items():
        if value is not None and key in merged:
            merged[key] = value
    return TrainConfig(**merged)


def _cfg_from_dict(cfg_dict: dict) -> TrainConfig:
    allowed = {f.name for f in fields(TrainConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in allowed}
    return TrainConfig(**filtered)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _build_optim_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    param_by_id: dict[int, torch.nn.Parameter] = {}
    name_by_id: dict[int, str] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        param_by_id[pid] = param
        name_by_id[pid] = name

    decay: set[int] = set()
    no_decay: set[int] = set()

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            pid = id(param)
            full_name = f"{module_name}.{param_name}" if module_name else param_name

            if param_name.endswith("bias"):
                no_decay.add(pid)
            elif param_name.endswith("weight") and isinstance(module, nn.Linear):
                decay.add(pid)
            elif param_name.endswith("weight") and isinstance(
                module, (nn.LayerNorm, nn.Embedding, RMSNorm)
            ):
                no_decay.add(pid)
            elif full_name.endswith("weight") and full_name in {"pos_emb.weight"}:
                # pos_emb is only present for learned positional encoding.
                no_decay.add(pid)

    # Anything not explicitly assigned falls back to dimensionality heuristic.
    for pid, param in param_by_id.items():
        if pid in decay or pid in no_decay:
            continue
        if param.dim() >= 2:
            decay.add(pid)
        else:
            no_decay.add(pid)

    # If a param was classified both ways (e.g. tied weights), no_decay wins.
    decay = decay - no_decay

    decay_params = [pid for pid in decay]
    no_decay_params = [pid for pid in no_decay]

    decay_params.sort(key=lambda pid: name_by_id.get(pid, ""))
    no_decay_params.sort(key=lambda pid: name_by_id.get(pid, ""))

    return [
        {"params": [param_by_id[pid] for pid in decay_params], "weight_decay": weight_decay},
        {"params": [param_by_id[pid] for pid in no_decay_params], "weight_decay": 0.0},
    ]


def _get_lr(step: int, cfg: TrainConfig) -> float:
    if cfg.lr_schedule == "constant":
        return cfg.learning_rate
    if cfg.lr_schedule != "cosine":
        raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r}")

    if cfg.lr_warmup_steps > 0 and step < cfg.lr_warmup_steps:
        return cfg.learning_rate * float(step + 1) / float(cfg.lr_warmup_steps)

    if cfg.max_steps <= cfg.lr_warmup_steps:
        return cfg.lr_min

    progress = float(step - cfg.lr_warmup_steps) / float(cfg.max_steps - cfg.lr_warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr_min + (cfg.learning_rate - cfg.lr_min) * cosine


def _tokenizer_from_checkpoint(checkpoint: dict) -> CharTokenizer | BpeTokenizer:
    tok = checkpoint.get("tokenizer")
    if isinstance(tok, dict):
        tok_type = tok.get("type")
        if tok_type == "char" and "itos" in tok:
            return CharTokenizer.from_itos(tok["itos"])
        if tok_type == "bpe" and "json" in tok:
            return BpeTokenizer.from_json(tok["json"])

    if "itos" in checkpoint:
        return CharTokenizer.from_itos(checkpoint["itos"])

    raise ValueError("Checkpoint is missing tokenizer data (expected 'tokenizer' or 'itos').")


def _tokenizer_to_checkpoint(tokenizer: CharTokenizer | BpeTokenizer, cfg: TrainConfig) -> dict:
    if cfg.tokenizer == "char":
        if not isinstance(tokenizer, CharTokenizer):
            raise TypeError("cfg.tokenizer='char' but tokenizer is not CharTokenizer")
        return {"type": "char", "itos": tokenizer.itos}
    if cfg.tokenizer == "bpe":
        if not isinstance(tokenizer, BpeTokenizer):
            raise TypeError("cfg.tokenizer='bpe' but tokenizer is not BpeTokenizer")
        return {"type": "bpe", "json": tokenizer.to_json()}
    raise ValueError(f"Unknown tokenizer type: {cfg.tokenizer!r}")


def _build_checkpoint(
    *,
    model: TinyGPT,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    best_val: float,
    tokenizer: CharTokenizer | BpeTokenizer,
    cfg: TrainConfig,
) -> dict:
    ckpt = {
        "model_state": model.state_dict(),
        "step": step,
        "best_val": best_val,
        "tokenizer": _tokenizer_to_checkpoint(tokenizer, cfg),
        "cfg": asdict(cfg),
    }
    if optimizer is not None:
        ckpt["optim_state"] = optimizer.state_dict()
    if cfg.tokenizer == "char":
        ckpt["itos"] = tokenizer.itos
    return ckpt


def main() -> None:
    args = _parse_args()

    if args.config and os.path.exists(args.config):
        file_cfg = _load_json(args.config)
        cfg = _merge_cfg(TrainConfig(), file_cfg)
    else:
        cfg = TrainConfig()

    cfg = _merge_cfg(
        cfg,
        {
            "data_path": args.data_path,
            "out_dir": args.out_dir,
            "resume": args.resume,
            "memmap_dataset": args.memmap_dataset,
            "tokenizer": args.tokenizer,
            "bpe_vocab_size": args.bpe_vocab_size,
            "bpe_min_frequency": args.bpe_min_frequency,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "block_size": args.block_size,
            "learning_rate": args.learning_rate,
            "lr_schedule": args.lr_schedule,
            "lr_warmup_steps": args.lr_warmup_steps,
            "lr_min": args.lr_min,
            "optimizer": args.optimizer,
            "lion_beta1": args.lion_beta1,
            "lion_beta2": args.lion_beta2,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "grad_checkpointing": args.grad_checkpointing,
            "compile": args.compile,
            "compile_backend": args.compile_backend,
            "compile_mode": args.compile_mode,
            "compile_fullgraph": args.compile_fullgraph,
            "compile_dynamic": args.compile_dynamic,
            "amp": args.amp,
            "label_smoothing": args.label_smoothing,
            "z_loss": args.z_loss,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "log_interval": args.log_interval,
            "sample_interval": args.sample_interval,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "dropout": args.dropout,
            "norm": args.norm,
            "norm_eps": args.norm_eps,
            "mlp": args.mlp,
            "parallel_block": args.parallel_block,
            "layerdrop": args.layerdrop,
            "pos_encoding": args.pos_encoding,
            "rope_base": args.rope_base,
            "sliding_window": args.sliding_window,
            "qk_norm": args.qk_norm,
            "qk_norm_eps": args.qk_norm_eps,
            "scaled_init": args.scaled_init,
            "weight_tying": args.weight_tying,
            "device": args.device,
            "ddp": args.ddp,
            "ddp_backend": args.ddp_backend,
            "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
            "ddp_timeout": args.ddp_timeout,
            "seed": args.seed,
            "prompt": args.prompt,
            "sample_chars": args.sample_chars,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "kv_cache": args.kv_cache,
        },
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    ckpt_best_path = os.path.join(cfg.out_dir, "ckpt_best.pt")

    checkpoint: dict | None = None
    if args.resume or args.generate_only:
        load_path = ckpt_path
        if not os.path.exists(load_path) and os.path.exists(ckpt_best_path):
            load_path = ckpt_best_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"No checkpoint found at {ckpt_path!r} (or {ckpt_best_path!r})."
            )
        checkpoint = torch.load(load_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "cfg" in checkpoint:
            ckpt_cfg = _cfg_from_dict(checkpoint["cfg"])
            ckpt_cfg.out_dir = cfg.out_dir

            # Disallow shape-changing overrides when resuming/generating.
            for key in (
                "block_size",
                "n_layer",
                "n_head",
                "n_embd",
                "tokenizer",
                "norm",
                "norm_eps",
                "mlp",
                "pos_encoding",
                "rope_base",
                "weight_tying",
            ):
                arg_val = getattr(args, key)
                if arg_val is not None and arg_val != getattr(ckpt_cfg, key):
                    raise ValueError(
                        f"Cannot override --{key} when resuming/generating. "
                        f"Checkpoint has {key}={getattr(ckpt_cfg, key)}, you passed {arg_val}."
                    )

            if ckpt_cfg.tokenizer == "bpe":
                for key in ("bpe_vocab_size", "bpe_min_frequency"):
                    arg_val = getattr(args, key)
                    if arg_val is not None and arg_val != getattr(ckpt_cfg, key):
                        raise ValueError(
                            f"Cannot override --{key} when resuming/generating. "
                            f"Checkpoint has {key}={getattr(ckpt_cfg, key)}, you passed {arg_val}."
                        )

            cfg = _merge_cfg(
                ckpt_cfg,
                {
                    "data_path": args.data_path,
                    "max_steps": args.max_steps,
                    "batch_size": args.batch_size,
                    "grad_accum_steps": args.grad_accum_steps,
                    "learning_rate": args.learning_rate,
                    "lr_schedule": args.lr_schedule,
                    "lr_warmup_steps": args.lr_warmup_steps,
                    "lr_min": args.lr_min,
                    "memmap_dataset": args.memmap_dataset,
                    "optimizer": args.optimizer,
                    "lion_beta1": args.lion_beta1,
                    "lion_beta2": args.lion_beta2,
                    "weight_decay": args.weight_decay,
                    "grad_clip": args.grad_clip,
                    "grad_checkpointing": args.grad_checkpointing,
                    "compile": args.compile,
                    "compile_backend": args.compile_backend,
                    "compile_mode": args.compile_mode,
                    "compile_fullgraph": args.compile_fullgraph,
                    "compile_dynamic": args.compile_dynamic,
                    "amp": args.amp,
                    "label_smoothing": args.label_smoothing,
                    "z_loss": args.z_loss,
                    "eval_interval": args.eval_interval,
                    "eval_iters": args.eval_iters,
                    "log_interval": args.log_interval,
                    "sample_interval": args.sample_interval,
                    "dropout": args.dropout,
                    "norm": args.norm,
                    "norm_eps": args.norm_eps,
                    "parallel_block": args.parallel_block,
                    "layerdrop": args.layerdrop,
                    "scaled_init": args.scaled_init,
                    "device": args.device,
                    "ddp": args.ddp,
                    "ddp_backend": args.ddp_backend,
                    "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
                    "ddp_timeout": args.ddp_timeout,
                    "seed": args.seed,
                    "prompt": args.prompt,
                    "sample_chars": args.sample_chars,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "sliding_window": args.sliding_window,
                    "qk_norm": args.qk_norm,
                    "qk_norm_eps": args.qk_norm_eps,
                    "kv_cache": args.kv_cache,
                },
            )

    if cfg.sliding_window < 0:
        raise ValueError("--sliding_window must be >= 0")
    if cfg.qk_norm_eps <= 0:
        raise ValueError("--qk_norm_eps must be > 0")
    if cfg.label_smoothing < 0 or cfg.label_smoothing >= 1:
        raise ValueError("--label_smoothing must be in [0, 1)")
    if cfg.z_loss < 0:
        raise ValueError("--z_loss must be >= 0")
    if cfg.layerdrop < 0 or cfg.layerdrop >= 1:
        raise ValueError("--layerdrop must be in [0, 1)")

    device_str = _device_from_cfg(cfg.device)
    ddp = _init_ddp(cfg=cfg, device_str=device_str)
    is_main_process = (not ddp.enabled) or ddp.is_main

    if ddp.enabled and device_str == "cuda":
        device = torch.device("cuda", ddp.local_rank)
    else:
        device = torch.device(device_str)

    if is_main_process:
        if ddp.enabled:
            print(f"DDP: world_size={ddp.world_size} backend={ddp.backend}")
        print(f"Device: {device}")

    if is_main_process:
        _write_json(os.path.join(cfg.out_dir, "effective_config.json"), asdict(cfg))

    seed = int(cfg.seed) + (ddp.rank if ddp.enabled else 0)
    random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    use_amp = bool(cfg.amp) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else (torch.float16 if use_amp else torch.float32)
    )

    tokenizer: CharTokenizer | BpeTokenizer
    train_data: torch.Tensor | None = None
    val_data: torch.Tensor | None = None
    text: str | None = None

    tokenizer_json_path = os.path.join(cfg.out_dir, "tokenizer.json")
    dist = None
    if ddp.enabled:
        import torch.distributed as dist

    if checkpoint is not None:
        tokenizer = _tokenizer_from_checkpoint(checkpoint)
        if (
            is_main_process
            and isinstance(tokenizer, BpeTokenizer)
            and not os.path.exists(tokenizer_json_path)
        ):
            _write_text(tokenizer_json_path, tokenizer.to_json())
        if ddp.enabled:
            dist.barrier()
    else:
        if cfg.tokenizer == "char":
            if not os.path.exists(cfg.data_path):
                raise FileNotFoundError(
                    f"Missing dataset at {cfg.data_path!r}. Provide --data_path or create the file."
                )
            if cfg.memmap_dataset:
                tokenizer = _build_char_tokenizer_from_file(cfg.data_path)
            else:
                text = _read_text(cfg.data_path)
                tokenizer = CharTokenizer(text)
        elif cfg.tokenizer == "bpe":
            if not os.path.exists(tokenizer_json_path):
                if not os.path.exists(cfg.data_path):
                    raise FileNotFoundError(
                        f"Missing dataset at {cfg.data_path!r}. Provide --data_path or create the file."
                    )
                if is_main_process:
                    tokenizer = BpeTokenizer.train(
                        data_path=cfg.data_path,
                        vocab_size=cfg.bpe_vocab_size,
                        min_frequency=cfg.bpe_min_frequency,
                    )
                    _write_text(tokenizer_json_path, tokenizer.to_json())
                if ddp.enabled:
                    dist.barrier()

            # All ranks load the tokenizer from disk to ensure consistency.
            tokenizer = BpeTokenizer.from_json(_read_text(tokenizer_json_path))
        else:
            raise ValueError(f"Unknown tokenizer: {cfg.tokenizer!r}")

    if not args.generate_only:
        if cfg.memmap_dataset:
            if ddp.enabled:
                if is_main_process:
                    data = _load_or_build_memmap_tokens(cfg=cfg, tokenizer=tokenizer)
                dist.barrier()
                if not is_main_process:
                    data = _load_or_build_memmap_tokens(cfg=cfg, tokenizer=tokenizer)
                dist.barrier()
            else:
                data = _load_or_build_memmap_tokens(cfg=cfg, tokenizer=tokenizer)
            split_idx = int(0.9 * data.size(0))
            train_data = data[:split_idx]
            val_data = data[split_idx:]
        else:
            if text is None:
                if not os.path.exists(cfg.data_path):
                    raise FileNotFoundError(
                        f"Missing dataset at {cfg.data_path!r}. Provide --data_path or create the file."
                    )
                text = _read_text(cfg.data_path)

            data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
            split_idx = int(0.9 * data.size(0))
            train_data = data[:split_idx].to(device)
            val_data = data[split_idx:].to(device)

    model = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg)
    start_step = 0
    best_val = math.inf

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        start_step = int(checkpoint.get("step", 0))
        best_val = float(checkpoint.get("best_val", best_val))

    model = model.to(device)

    model_train: nn.Module = model
    if ddp.enabled:
        from torch.nn.parallel import DistributedDataParallel as DDP

        if device.type == "cuda":
            model_train = DDP(
                model,
                device_ids=[int(device.index)],
                output_device=int(device.index),
                broadcast_buffers=False,
                find_unused_parameters=bool(cfg.ddp_find_unused_parameters),
            )
        else:
            model_train = DDP(
                model,
                broadcast_buffers=False,
                find_unused_parameters=bool(cfg.ddp_find_unused_parameters),
            )

    optimizer = None
    scaler: torch.amp.GradScaler | None = None
    if not args.generate_only:
        optim_groups = _build_optim_groups(model, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=cfg.learning_rate,
                betas=(0.9, 0.95),
            )
        elif cfg.optimizer == "lion":
            optimizer = Lion(
                optim_groups,
                lr=cfg.learning_rate,
                betas=(cfg.lion_beta1, cfg.lion_beta2),
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")
        if checkpoint is not None and "optim_state" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optim_state"])
                _optimizer_to(optimizer, device)
            except Exception as e:
                print(f"warning: optimizer state not loaded ({e}); continuing with fresh optimizer.")
        scaler = torch.amp.GradScaler(
            device=device.type, enabled=use_amp and amp_dtype == torch.float16
        )

    model_fwd: nn.Module = model_train
    if not args.generate_only and cfg.compile:
        if not hasattr(torch, "compile"):
            if is_main_process:
                print("warning: torch.compile not available; continuing without compilation.")
        else:
            mode = cfg.compile_mode or None
            backend = cfg.compile_backend or "inductor"
            try:
                compiled = torch.compile(
                    model_train,
                    backend=backend,
                    mode=mode,
                    fullgraph=bool(cfg.compile_fullgraph),
                    dynamic=cfg.compile_dynamic,
                )
                try:
                    assert train_data is not None
                    x0, y0 = _get_batch(
                        train_data,
                        batch_size=cfg.batch_size,
                        block_size=cfg.block_size,
                        device=device,
                    )
                    with torch.amp.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=use_amp
                    ):
                        compiled(x0, y0)
                except Exception as e:
                    if is_main_process:
                        print(
                            f"warning: torch.compile backend {backend!r} failed ({e}); "
                            "continuing without compilation."
                        )
                    model_fwd = model_train
                else:
                    model_fwd = compiled
                    if is_main_process:
                        print(
                            "torch.compile enabled"
                            f" (backend={backend}, mode={mode or 'default'}, fullgraph={bool(cfg.compile_fullgraph)}, dynamic={cfg.compile_dynamic})"
                        )
            except Exception as e:
                if is_main_process:
                    print(f"warning: torch.compile failed ({e}); continuing without compilation.")
                model_fwd = model_train

    if args.generate_only:
        if is_main_process:
            model.eval()
            prompt = cfg.prompt or "\n"
            context = torch.tensor(
                [tokenizer.encode(prompt)], dtype=torch.long, device=device
            )
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                out = model.generate(
                    context,
                    max_new_tokens=cfg.sample_chars,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                    repetition_penalty=cfg.repetition_penalty,
                )[0].tolist()
            _safe_print(tokenizer.decode(out))

        if ddp.enabled:
            import torch.distributed as dist

            dist.barrier()
            dist.destroy_process_group()
        return

    if cfg.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")

    last_log_time = time.time()
    last_log_step = start_step
    last_step = start_step
    try:
        for step in range(start_step, cfg.max_steps):
            assert train_data is not None
            assert val_data is not None
            assert optimizer is not None
            assert scaler is not None
            lr = _get_lr(step, cfg)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for micro_step in range(cfg.grad_accum_steps):
                sync_ctx = (
                    model_train.no_sync()
                    if ddp.enabled and micro_step < (cfg.grad_accum_steps - 1)
                    else nullcontext()
                )
                with sync_ctx:
                    x, y = _get_batch(
                        train_data,
                        batch_size=cfg.batch_size,
                        block_size=cfg.block_size,
                        device=device,
                    )

                    with torch.amp.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=use_amp
                    ):
                        _, loss = model_fwd(x, y)

                    loss_sum += float(loss.item())
                    loss = loss / cfg.grad_accum_steps

                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            last_step = step + 1
            avg_loss = loss_sum / cfg.grad_accum_steps

            if (
                is_main_process
                and cfg.log_interval > 0
                and (step + 1) % cfg.log_interval == 0
            ):
                now = time.time()
                steps = step + 1 - last_log_step
                elapsed = now - last_log_time
                steps_per_sec = steps / max(elapsed, 1e-8)
                toks_per_sec = (
                    steps_per_sec
                    * cfg.batch_size
                    * cfg.block_size
                    * cfg.grad_accum_steps
                    * (ddp.world_size if ddp.enabled else 1)
                )
                print(
                    f"step {step+1:>6} | loss {avg_loss:.4f} | lr {lr:.2e} | {steps_per_sec:.2f} steps/s | {toks_per_sec/1000:.1f}k tok/s"
                )
                last_log_time = now
                last_log_step = step + 1

            if (
                cfg.eval_interval > 0
                and cfg.eval_iters > 0
                and ((step + 1) % cfg.eval_interval == 0 or step == start_step)
            ):
                if is_main_process:
                    losses = _estimate_loss(
                        model_fwd,
                        train_data=train_data,
                        val_data=val_data,
                        cfg=cfg,
                        device=device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                    )
                    print(
                        f"eval @ step {step+1:>6} | train {losses['train']:.4f} | val {losses['val']:.4f}"
                    )

                    if losses["val"] < best_val:
                        best_val = losses["val"]
                        torch.save(
                            _build_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                step=step + 1,
                                best_val=best_val,
                                tokenizer=tokenizer,
                                cfg=cfg,
                            ),
                            ckpt_best_path,
                        )
                        print(f"saved best checkpoint to {ckpt_best_path}")

                if ddp.enabled:
                    dist.barrier()

            if cfg.sample_interval > 0 and (step + 1) % cfg.sample_interval == 0:
                if is_main_process:
                    model_train.eval()
                    model_fwd.eval()
                    prompt = cfg.prompt or "\n"
                    context = torch.tensor(
                        [tokenizer.encode(prompt)], dtype=torch.long, device=device
                    )
                    with torch.amp.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=use_amp
                    ):
                        out = model.generate(
                            context,
                            max_new_tokens=cfg.sample_chars,
                            temperature=cfg.temperature,
                            top_k=cfg.top_k,
                            top_p=cfg.top_p,
                            repetition_penalty=cfg.repetition_penalty,
                        )[0].tolist()
                    model_train.train()
                    model_fwd.train()
                    print("--- sample ---")
                    _safe_print(tokenizer.decode(out))
                    print("--------------")

                if ddp.enabled:
                    dist.barrier()
    except KeyboardInterrupt:
        print("interrupted; saving latest checkpoint...")
    finally:
        if last_step > start_step:
            assert optimizer is not None
            if is_main_process:
                torch.save(
                    _build_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        step=last_step,
                        best_val=best_val,
                        tokenizer=tokenizer,
                        cfg=cfg,
                    ),
                    ckpt_path,
                )
                print(f"saved checkpoint to {ckpt_path}")

        if ddp.enabled:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
