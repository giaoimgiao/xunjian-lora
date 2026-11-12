"""把 LoRA 合并进已有 int8 权重，输出单目录成品。
W' = W + (lora_b @ lora_a) * alpha/rank，然后按行重算 scale 重新量化。
源目录只读。
"""
import json
import os
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = os.environ.get("SRC", "/workspace/model/Qwen3.8-27B-ab2")
DST = os.environ.get("DST", "/workspace/model/Qwen3.8-27B-ab2-fused")
LORA = os.environ.get("LORA", "/workspace/lora/out_novel3/lora_final.pt")


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


sd = torch.load(LORA, map_location="cpu", weights_only=False)
rank = sd.get("rank", 32)
alpha = sd.get("alpha", rank * 2)
L = sd["lora"]
scale_lora = float(alpha) / float(rank)
log("LORA_TENSORS", len(L), "rank", rank, "alpha", alpha, "scale", scale_lora)

# 把 lora 的键映射成权重键
pairs = {}
for k, v in L.items():
    for tag in (".lora_a", ".lora_b"):
        if k.endswith(tag):
            base = k[: -len(tag)]
            wb = base.replace("model.layers.", "model.language_model.layers.", 1)
            pairs.setdefault(wb, {})[tag] = v
            break
log("MODULES_WITH_LORA", len(pairs))

idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
wmap = idx["weight_map"]
files = sorted(set(wmap.values()))
if not os.path.isdir(DST):
    os.makedirs(DST, exist_ok=True)

done = 0
t0 = time.time()
for fi, fn in enumerate(files):
    sp = os.path.join(SRC, fn)
    dp = os.path.join(DST, fn)
    keys = [k for k, v in wmap.items() if v == fn]
    tensors = {}
    with safe_open(sp, framework="pt", device="cpu") as f:
        for k in keys:
            tensors[k] = f.get_tensor(k).clone()

    n_here = 0
    for k in list(tensors.keys()):
        if not k.endswith(".weight"):
            continue
        wb = k[: -len(".weight")]
        if wb not in pairs:
            continue
        ab = pairs[wb]
        if ".lora_a" not in ab or ".lora_b" not in ab:
            continue
        skey = wb + ".weight_scale"
        if skey not in tensors:
            continue
        a = ab[".lora_a"].to(torch.float32)
        b = ab[".lora_b"].to(torch.float32)
        delta = (b @ a) * scale_lora
        w = tensors[k].to(torch.float32) * tensors[skey].to(torch.float32)
        if tuple(delta.shape) != tuple(w.shape):
            log("SHAPE_MISMATCH", wb, tuple(w.shape), tuple(delta.shape))
            continue
        wn = w + delta
        sc = wn.abs().amax(dim=1, keepdim=True) / 127.0
        sc = torch.where(sc == 0, torch.ones_like(sc), sc)
        q = torch.round(wn / sc).clamp_(-128, 127)
        tensors[k] = q.to(torch.int8)
        tensors[skey] = sc
        n_here += 1
        done += 1
    save_file(tensors, dp, metadata={"format": "pt"})
    del tensors
    log("WROTE %d/%d %s fused=%d" % (fi + 1, len(files), fn, n_here))

for f in os.listdir(SRC):
    if not f.endswith(".safetensors"):
        shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))
log("TOTAL_FUSED", done, "%.1fmin" % ((time.time() - t0) / 60))
log("DONE", DST)