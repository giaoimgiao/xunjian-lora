"""第二步：按采集到的拒绝方向改权重，另存为新模型目录。

对每层的出口投影矩阵（mlp.down_proj / self_attn.o_proj / linear_attn.out_proj）：
  1) 反量化  w = w_int8 * scale
  2) 减方向  w' = w - λ · r ⊗ (r · w)      r 为 5120 维单位向量
  3) 重新量化 scale' = max|w'|/127, w_int8' = round(w'/scale')
原模型目录只读，不改动。
DRY=1 时只处理第 0 层，打印数值变化。
"""
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/workspace/model/Qwen3.8-27B-Channel-INT8-w8a8/main/Qwen3.8-27B-Channel-INT8-w8a8"
DST = os.environ.get("DST", "/workspace/model/Qwen3.8-27B-abliterated")
DIRS = "/workspace/lora/directions.pt"
LAM = float(os.environ.get("LAM", "1.0"))
DRY = int(os.environ.get("DRY", "0"))
MIN_L = int(os.environ.get("MIN_L", "0"))
TGT = os.environ.get("TGT", "dpo")   # d=down_proj p=attn.o_proj o=linear_attn.out_proj

_ALL = {".mlp.down_proj.weight": "d", ".self_attn.o_proj.weight": "p",
        ".linear_attn.out_proj.weight": "o"}
TARGET_SUFFIX = tuple(s for s, c in _ALL.items() if c in TGT)


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


dirs = torch.load(DIRS, map_location="cpu")
log("DIRECTIONS", len(dirs), "LAM", LAM, "DRY", DRY)

idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
wmap = idx["weight_map"]
files = sorted(set(wmap.values()))
log("FILES", len(files))

if not DRY:
    os.makedirs(DST, exist_ok=True)

stats = []
t_start = time.time()
for fi, fn in enumerate(files):
    sp = os.path.join(SRC, fn)
    dp = os.path.join(DST, fn)
    keys = [k for k, v in wmap.items() if v == fn]

    todo = []
    for k in keys:
        if not k.endswith(TARGET_SUFFIX):
            continue
        if "mtp." in k:
            continue
        parts = k.split(".")
        li = None
        for j, p in enumerate(parts):
            if p == "layers" and j + 1 < len(parts):
                li = int(parts[j + 1])
                break
        if li is None or li not in dirs or li < MIN_L:
            continue
        skey = k.replace(".weight", ".weight_scale")
        if skey not in wmap:
            continue
        todo.append((k, skey, li))

    if DRY:
        todo = [x for x in todo if x[2] == 0]
    if not todo:
        if not DRY:
            shutil.copy2(sp, dp)
        continue

    tensors = {}
    with safe_open(sp, framework="pt", device="cpu") as f:
        for k in keys:
            tensors[k] = f.get_tensor(k)

    for k, skey, li in todo:
        w_i8 = tensors[k].to(torch.float32)
        sc = tensors[skey].to(torch.float32)
        w = w_i8 * sc
        r = dirs[li].to(torch.float32)
        if r.numel() != w.shape[0]:
            log("SKIP_SHAPE", k, tuple(w.shape), r.numel())
            continue
        coef = r @ w                      # [in]
        w_new = w - LAM * r[:, None] * coef[None, :]
        sc_new = w_new.abs().amax(dim=1, keepdim=True) / 127.0
        sc_new = torch.where(sc_new == 0, torch.ones_like(sc_new), sc_new)
        q = torch.round(w_new / sc_new).clamp_(-128, 127)
        err = (q * sc_new - w_new).abs()
        stats.append((li, k.split(".")[-2], float(err.mean()), float(err.max()),
                      float(w_new.abs().mean())))
        tensors[k] = q.to(torch.int8)
        tensors[skey] = sc_new

    if DRY:
        for s in stats:
            log("STAT layer%d %s mean_err %.6f max_err %.6f |w| %.6f" % s)
        log("DRY_DONE tensors", len(todo))
        sys.exit(0)

    save_file(tensors, dp, metadata={"format": "pt"})
    del tensors
    log("WROTE %d/%d %s (%d proj)" % (fi + 1, len(files), fn, len(todo)))

log("PROJ_EDITED", len(stats), "%.1fmin" % ((time.time() - t_start) / 60))
if stats:
    log("ERR_MEAN %.6f  ERR_MAX %.6f" % (
        sum(s[2] for s in stats) / len(stats), max(s[3] for s in stats)))

if not DRY:
    for f in os.listdir(SRC):
        if not f.endswith(".safetensors"):
            shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))
    log("COPIED_CONFIGS")
    log("DONE", DST)