# 去对齐（Abliteration）复现配方

把 Qwen3.8-27B（INT8 w8a8）的**拒绝先验从权重里削掉**，再合并 LoRA，得到一个不拒绝成人内容、
同时保留原作文风的模型。全程不依赖提示词绕过。

## 原理

模型对"有害请求"会产生一个稳定的激活偏移方向（refusal direction）。把它从残差流的出口矩阵里减掉，
拒绝行为就消失了，而语言能力基本不受影响。

本配方采用**逐层方向向量**（64 层各一个，每层 5120 维，单位化），而不是单一全局方向。

## 文件

| 文件 | 说明 |
|---|---|
| `abliteration/directions.pt` | 64 层拒绝方向向量，约 1.3 MB |
| `abliteration/ablate_apply.py` | 削减脚本：把方向分量从出口矩阵里减掉，重新量化回 int8 |
| `abliteration/lora_fuse.py` | 把 LoRA 增量合并进 int8 权重（`W' = W + B@A · alpha/rank`） |
| `q35_base.py` | INT8 权重加载器（transformers 原生读不了这套格式） |

## 采集方向向量

对同一批语义对（有害句 / 无害句）各跑一遍前向，抓每层输出的最后 token 激活，
求「有害均值 − 无害均值」，再单位化，得到每层一个方向。

```python
d[i] = (acc_harm[i] / n_harm) - (acc_benign[i] / n_benign)
d[i] = d[i] / d[i].norm()
```

## 第一步：削减

```bash
# 需要原始基座：Qwen3.8-27B-Channel-INT8-w8a8
#   18 个 model-xxxxx-of-00018.safetensors + model.safetensors.index.json
#   config.model_type = qwen3_5，64 层，hidden 5120

DIRS=./directions.pt \
DST=/output/Qwen3.8-27B-ab2 \
LAM=0.45 \
MIN_L=8 \
TGT=dp \
DRY=0 \
python3 ablate_apply.py
```

参数含义（这几个值是实测定下来的，不要乱改）：

| 参数 | 值 | 说明 |
|---|---|---|
| `LAM` | **0.45** | 削减强度。**1.0 会直接把模型打废**，输出全是结束符 |
| `MIN_L` | **8** | 跳过第 0–7 层，前几层是低层特征，改了就崩 |
| `TGT` | **dp** | 只改 `mlp.down_proj` 与**全注意力** `o_proj`。**不要加 `o`**（线性注意力出口），加了会显著损伤语言能力 |

产出：约 70 个矩阵被修改并重新量化，平均量化误差约 8e-5，目录约 29 GB。

## 第二步：合并 LoRA（可选）

```bash
SRC=/output/Qwen3.8-27B-ab2 \
DST=/output/Qwen3.8-27B-ab2-fused \
LORA=lora_v2_120step.pt \
python3 lora_fuse.py
```

合并公式 `W' = W + (B@A) * alpha/rank`（rank 32 / alpha 64，即乘 2.0）。

**注意**：int8 量化的一个台阶约为 `1/256 · scale`，而 LoRA 增量的平均幅度通常只有台阶的 1/4 左右，
所以**合并进 int8 权重后，LoRA 的大部分增量会被量化抹掉**。实测融合版的文风明显弱于「权重 + 独立挂载 LoRA」。
建议保留分离形式，推理时挂载。

## 第三步：推理

```python
import q35_base as B
B.D = "/output/Qwen3.8-27B-ab2"     # 去对齐基座
model = B.build_model()
B.apply_lora(model, r=32, alpha=64)  # 再载入 LoRA 张量
```

采样：`do_sample=True, temperature=0.85, top_p=0.9, repetition_penalty=1.1`，
**禁用贪心解码**（会在同一句上无限重复）。

## 实测效果

| 项 | 原模型 | 去对齐后 |
|---|---|---|
| 思维链开头 | "这个请求相当棘手……" | "用户要求用第三人称写一个场景……" |
| 对内容的态度 | "我决定明确拒绝这个请求" | "让我仔细构思这个场景" |
| 输出 | 拒绝理由 | 直接写正文 |

显存：单卡 64 GB，batch=1 约 30–34 GB。生成速度：约 3 tok/s。

## 已知损伤

- 改得太狠（LAM ≥ 0.8 或含线性注意力出口）会导致模型退化为只输出结束符
- 长文本生成（2000 字以上）偶发人名串位
- 极少数位置中英混杂
