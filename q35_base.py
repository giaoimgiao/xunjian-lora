"""Qwen3.5-27B INT8 冻结权重 + LoRA 训练基座。

要点：
1. int8 权重常驻显存（27.6GB），forward 时逐层临时反量化成 bf16 计算；
   反量化走 autograd.Function，backward 不保留 bf16 权重副本。
2. AttentionInterface.register('chunked') 覆盖 full_attention 的 O(n^2) 实现。
3. lm_head 分块 + checkpoint，避免 32k 时 logits 一次性 16GB。
"""
import gc
import json
import math
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

D = "/workspace/model/Qwen3.8-27B-Channel-INT8-w8a8/main/Qwen3.8-27B-Channel-INT8-w8a8"
LINEAR_ATTRS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                "down_proj", "in_proj_qkv", "in_proj_z", "out_proj", "in_proj_a", "in_proj_b")


def log(*a):
    print(*a, flush=True)


def gb(x):
    return x / 2**30


# ---------------------------------------------------------------- int8 反量化
class _Dequant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_i8, scale):
        return w_i8.to(torch.bfloat16) * scale.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, g):
        return None, None


class Int8Linear(torch.nn.Module):
    def __init__(self, in_f, out_f, has_bias):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.register_buffer("weight_i8", torch.empty(out_f, in_f, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_f, 1, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.empty(out_f, dtype=torch.bfloat16)) if has_bias else None

    def forward(self, x):
        w = _Dequant.apply(self.weight_i8, self.weight_scale)
        return F.linear(x, w, self.bias)


# ---------------------------------------------------------------- chunked attention
def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None].expand(b, h, n_rep, s, d)
    return hidden_states.reshape(b, h * n_rep, s, d)

def make_chunked_attention(chunk=256):
    def _chunk(q, k, v, scaling, lo, hi, ms):
        att = torch.matmul(q, k.transpose(2, 3)) * scaling
        if ms is not None:
            att = att + ms.to(att.dtype)
        else:
            iq = torch.arange(lo, hi, device=att.device)
            ik = torch.arange(k.shape[2], device=att.device)
            causal = ik[None, :] <= iq[:, None]
            att = att.masked_fill(~causal, torch.finfo(att.dtype).min)
        att = torch.nn.functional.softmax(att, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(att, v)

    def fn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        n_rep = getattr(module, "num_key_value_groups", 1)
        key = repeat_kv(key, n_rep)
        value = repeat_kv(value, n_rep)
        Tq = query.shape[2]
        Tk = key.shape[2]
        off = Tk - Tq  # KV cache 解码时 query 在序列中的起始偏移
        grad = torch.is_grad_enabled() and query.requires_grad
        outs = []
        for s in range(0, Tq, chunk):
            e = min(s + chunk, Tq)
            q = query[:, :, s:e]
            k = key[:, :, : off + e]
            v = value[:, :, : off + e]
            ms = None
            if attention_mask is not None and attention_mask.dim() == 4:
                ms = attention_mask[:, :, off + s: off + e, : off + e]
            if grad:
                outs.append(torch.utils.checkpoint.checkpoint(
                    _chunk, q, k, v, scaling, off + s, off + e, ms, use_reentrant=False))
            else:
                outs.append(_chunk(q, k, v, scaling, off + s, off + e, ms))
        out = torch.cat(outs, dim=2)
        return out.transpose(1, 2).contiguous(), None

    return fn


def register_chunked(chunk=1024):
    M.ALL_ATTENTION_FUNCTIONS.register("chunked", make_chunked_attention(chunk))


# ---------------------------------------------------------------- 模型构建
def build_model(load=True, dev="cuda:0", log_fn=log):
    if "chunked" not in M.ALL_ATTENTION_FUNCTIONS:
        M.ALL_ATTENTION_FUNCTIONS.register("chunked", make_chunked_attention(256))
    cfg = AutoConfig.from_pretrained(D)
    tcfg = cfg.text_config
    torch.set_default_dtype(torch.bfloat16)

    idx = json.load(open(os.path.join(D, "model.safetensors.index.json")))
    wm = idx["weight_map"]

    def map_key(k):
        if k.startswith("model.language_model."):
            return "model." + k[len("model.language_model."):]
        if k.startswith("lm_head.") or k.startswith("model.embed_tokens"):
            return k
        return None

    quant = set()
    for k in wm:
        nk = map_key(k)
        if nk and nk.endswith(".weight_scale"):
            quant.add(nk[: -len(".weight_scale")])

    with torch.device("meta"):
        model = M.Qwen3_5ForCausalLM(tcfg)

    replaced = 0
    for name, mod in list(model.named_modules()):
        for attr in LINEAR_ATTRS:
            sub = getattr(mod, attr, None)
            if isinstance(sub, torch.nn.Linear):
                full = name + "." + attr if name else attr
                if full in quant:
                    setattr(mod, attr, Int8Linear(sub.in_features, sub.out_features, sub.bias is not None))
                    replaced += 1
    log_fn("REPLACED_INT8", replaced, "QUANT_EXPECT", len(quant))

    model = model.to_empty(device=dev)
    tcfg._attn_implementation = "chunked"
    model.config._attn_implementation = "chunked"

    if not load:
        return model

    mods = dict(model.named_modules())
    params = dict(model.named_parameters())
    bufs = dict(model.named_buffers())
    loaded = skipped = 0
    missing = []
    for sh in sorted(set(wm.values())):
        with safe_open(os.path.join(D, sh), framework="pt") as h:
            for k in h.keys():
                nk = map_key(k)
                if nk is None:
                    skipped += 1
                    continue
                t = h.get_tensor(k)
                if nk.endswith(".weight_scale"):
                    m = mods.get(nk[: -len(".weight_scale")])
                    if m is None:
                        missing.append(nk)
                        continue
                    m.weight_scale.copy_(t.to(dev))
                    loaded += 1
                    continue
                if nk.endswith(".weight") and t.dtype == torch.int8:
                    m = mods.get(nk[: -len(".weight")])
                    if m is None:
                        missing.append(nk)
                        continue
                    m.weight_i8.copy_(t.to(dev))
                    loaded += 1
                    continue
                dst = params.get(nk)
                if dst is None:
                    dst = bufs.get(nk)
                if dst is None:
                    missing.append(nk)
                    continue
                dst.data.copy_(t.to(dev))
                loaded += 1
        gc.collect()
    log_fn("LOADED", loaded, "SKIPPED_VISION", skipped, "MISSING", len(missing), missing[:5])
    return model


# ---------------------------------------------------------------- LoRA
def _lora_delta(x, a, b):
    return (x @ a.t()) @ b.t()


class LoRALinear(torch.nn.Module):
    def __init__(self, base, r=16, alpha=32):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        dev = base.weight_i8.device if hasattr(base, "weight_i8") else base.weight.device
        out_f = base.out_f if hasattr(base, "out_f") else base.out_features
        in_f = base.in_f if hasattr(base, "in_f") else base.in_features
        self.lora_a = torch.nn.Parameter(torch.zeros(r, in_f, dtype=torch.bfloat16, device=dev))
        self.lora_b = torch.nn.Parameter(torch.zeros(out_f, r, dtype=torch.bfloat16, device=dev))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b)

    def forward(self, x):
        y = self.base(x)
        if torch.is_grad_enabled() and (self.lora_a.requires_grad or self.lora_b.requires_grad):
            d = torch.utils.checkpoint.checkpoint(_lora_delta, x, self.lora_a, self.lora_b,
                                                  use_reentrant=False)
        else:
            d = _lora_delta(x, self.lora_a, self.lora_b)
        return y + d * self.scale


def apply_lora(model, r=16, alpha=32, targets=("q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj",
                                               "in_proj_qkv", "in_proj_z", "out_proj")):
    n = 0
    for mod in list(model.modules()):
        for attr in targets:
            sub = getattr(mod, attr, None)
            if isinstance(sub, Int8Linear):
                setattr(mod, attr, LoRALinear(sub, r, alpha))
                n += 1
    return n


# ---------------------------------------------------------------- 分块 lm_head loss
def chunked_ce(hidden, labels, lm_head_w, chunk=2048, ignore_index=-100):
    total = 0.0
    ntok = 0
    T = hidden.shape[1]
    V = lm_head_w.shape[0]

    def _ce(h, lab):
        logits = h @ lm_head_w.t()
        return F.cross_entropy(logits.reshape(-1, V).float(), lab.reshape(-1),
                               ignore_index=ignore_index, reduction="sum")

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        h = hidden[:, s:e]
        lab = labels[:, s:e]
        if not (lab != ignore_index).any():
            continue
        if h.requires_grad and torch.is_grad_enabled():
            loss = torch.utils.checkpoint.checkpoint(_ce, h, lab, use_reentrant=False)
        else:
            loss = _ce(h, lab)
        total = total + loss
        ntok += int((lab != ignore_index).sum())
    if ntok == 0:
        return None, 0
    return total / ntok, ntok