# yolo26n 模型结构改动记录

**日期：** 2026-06-08（更新）  
**目的：** 消除 QAT 量化图中检测头内部的 Concat 操作，避免跨尺度 scale 冲突导致的量化精度损失

---

## 当前状态（2026-06-08）

### 改动概览

| 文件              | 函数/位置                       | 改动内容                                                                                        | 原因                                   |
| ----------------- | ------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------- |
| `head.py`         | `forward_head(concat_flag=...)` | 新增参数，始终返回 per-scale list；训练传 `False`，one2one 优化模式可传 `True` 合回单 dict      | 按需控制是否 concat                    |
| `head.py`         | `Detect.forward()`              | 训练直返 list；推理内联 `isinstance` + `torch.cat` 汇合后再 decode                              | 去除 `_merge_per_scale_preds` 间接调用 |
| `loss.py`         | `v8DetectionLoss.loss()`        | `isinstance(preds, list)` → `torch.cat` 汇合后走原始 loss 逻辑                                  | 保持 loss 与原始 concat 版本等价       |
| `validator.py`    | `_rebuild_pt2e_predictions`     | per-scale list + end2end 路由修复                                                               | QAT 验证适配                           |
| `export.py`       | 多个导出包装器 + ONNX 修复函数  | `DetectOne2OneWrapper`/`DetectOne2ManyWrapper`；`_fix_qdq_qdq_mismatch`、`_merge_adjacent_dq_q` | per-scale 多输出导出 + ONNX 图质量修复 |
| `tasks.py`        | `DetectionModel` stride 初始化  | `isinstance(output, list)` 分支                                                                 | 适配 per-scale feats 提取              |
| `block.py`        | `SPPF.forward()`                | 生成器表达式 → 显式 for 循环                                                                    | PT2E `export_for_training` 兼容        |
| `ax_quantizer.py` | `init_regional()`               | matmul/gridsample 硬编码 S16                                                                    | 部署工具要求 S16 MatMul                |

### head.py 详细状态

#### `forward_head(concat_flag=True)`

```python
def forward_head(self, x, cv2, cv3, concat_flag=True, **extra):
    """Per-scale detection head: always returns list[...]; concat_flag=True merges back to single dict."""
    boxes, scores = [], []
    for i in range(self.nl):
        b = cv2[i](x[i]).view(bs, 4 * self.reg_max, -1)
        s = cv3[i](x[i]).view(bs, self.nc, -1)
        boxes.append(b)
        scores.append(s)

    if concat_flag:
        return {"boxes": torch.cat(boxes, dim=-1), "scores": torch.cat(scores, dim=-1), "feats": x}
    return [{"boxes": boxes[i], "scores": scores[i], "feats": [x[i]]} for i in range(self.nl)]
```

- `concat_flag=True`（默认，向后兼容）：返回单 dict（原始行为），用于 one2one 优化路径
- `concat_flag=False`：返回 per-scale list，用于 one2many QAT 训练/导出

#### `forward()`

```python
def forward(self, x):
    preds = self.forward_head(x, **self.one2many, concat_flag=False)
    if self.end2end:
        x_detach = [xi.detach() for xi in x]
        one2one = self.forward_head(x_detach, **self.one2one, concat_flag=False)
        preds = {"one2many": preds, "one2one": one2one}
    if self.training:
        return preds

    # 推理路径：内联 concat 后 decode
    if "one2one" in preds:
        preds["one2many"]["boxes"] = (
            torch.cat(preds["one2many"]["boxes"], dim=-1)
            if isinstance(preds["one2many"]["boxes"], list)
            else preds["one2many"]["boxes"]
        )
        preds["one2many"]["scores"] = (
            torch.cat(preds["one2many"]["scores"], dim=-1)
            if isinstance(preds["one2many"]["scores"], list)
            else preds["one2many"]["scores"]
        )
        preds["one2one"]["boxes"] = (
            torch.cat(preds["one2one"]["boxes"], dim=-1)
            if isinstance(preds["one2one"]["boxes"], list)
            else preds["one2one"]["boxes"]
        )
        preds["one2one"]["scores"] = (
            torch.cat(preds["one2one"]["scores"], dim=-1)
            if isinstance(preds["one2one"]["scores"], list)
            else preds["one2one"]["scores"]
        )
    else:
        preds["boxes"] = torch.cat(preds["boxes"], dim=-1)
        preds["scores"] = torch.cat(preds["scores"], dim=-1)
```

### 已删除的代码

- `_merge_per_scale_preds()`：已删除，替换为 `forward_head` 的 `concat_flag` 参数和 `forward()` 中内联 `torch.cat`

### 设计演进

| 版本          | 核心机制                        | 问题                                            |
| ------------- | ------------------------------- | ----------------------------------------------- |
| v1 (exp24-28) | 原始 Concat head                | ONNX 中三尺度 scale 冲突（4.9x），大量冗余 Q-DQ |
| v2 (exp29)    | `_merge_per_scale_preds()` 汇合 | 间接调用增加图复杂度                            |
| v3 (exp30+)   | `concat_flag` + 内联 concat     | 简洁，训练永远 per-scale，推理内联合回          |

---

## 原始 v1 改动记录（2026-06-05，已过时）

> 以下为初版 per-scale 实现时文档，部分内容已不再适用。当前准确描述见上方。

### 改动概览

| 文件               | 函数/位置                      | 改动内容                                                                 | 原因                                    |
| ------------------ | ------------------------------ | ------------------------------------------------------------------------ | --------------------------------------- |
| `head.py:67-83`    | `_merge_per_scale_preds()`     | ~~新增（已删除）~~                                                       | 推理时将 per-scale list 重汇合为单 dict |
| `head.py:135-144`  | `Detect.forward_head()`        | `torch.cat` → **per-scale list**                                         | 消除 head 内三尺度 Concat 量化域冲突    |
| `head.py:146-167`  | `Detect.forward()`             | 训练返回 list，推理调用 `_merge_per_scale_preds` → **已改为内联 concat** | 训练/推理路由适配                       |
| `loss.py:489-516`  | `v8DetectionLoss.loss()`       | 支持 `list[dict]` 格式                                                   | per-scale 逐尺度计算 loss 求和          |
| `tasks.py:408-412` | `DetectionModel` stride 初始化 | `isinstance(output, list)` 分支                                          | 适配 per-scale feats 提取               |
| `block.py:232-239` | `SPPF.forward()`               | 生成器表达式 → **显式 for 循环**                                         | PT2E `export_for_training` 兼容         |

### 详细改动

#### 1.1 `_merge_per_scale_preds()` — 已删除

~~将 per-scale list 重汇合为单 dict。~~ 当前由 `forward()` 中内联 `isinstance` + `torch.cat` 替代。

#### 1.2 `Detect.forward_head()` — 移除 torch.cat（保持）

```python
# 上游（有 torch.cat）：
boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * reg_max, -1) for i in range(self.nl)], dim=-1)
scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
return {"boxes": boxes, "scores": scores, "feats": x}

# 当前（无 torch.cat，concat_flag=False）：
return [
    {
        "boxes": box_head[i](x[i]).view(bs, 4 * reg_max, -1), "scores": cls_head[i](x[i]).view(bs, self.nc, -1), "feats": [x[i]]
    }
    for i in range(self.nl)
]
```

#### 1.3 `Detect.forward()` — 已更新

```python
# 训练路径：直接返回 per-scale list/嵌套 dict
if self.training:
    return preds  # {"one2many": [...], "one2one": [...]}

# 推理路径：内联合后再 decode
if "one2one" in preds:
    preds["one2many"]["boxes"] = torch.cat(...)
else:
    preds["boxes"] = torch.cat(...)
y = self._inference(preds["one2one"] if self.end2end else preds)
```

**兼容性：** 预训练权重 708/708 完全匹配，参数形状未变。

### 2. loss.py（保持）

#### 2.1 `v8DetectionLoss.loss()` — 支持 per-scale list

当前实现：`isinstance(preds, list)` 检查后，通过 `torch.cat` 将 per-scale list 汇合为单 dict，再走原始 loss 逻辑。保证 loss 与原始 concat 版本完全等价，避免 per-scale 直接训练带来的量化误差暴增（4x）。

```python
def loss(self, preds, batch, teacher_preds=None):
    if isinstance(preds, list):
        # torch.cat 汇合为单 dict 后走原始逻辑
        ...
```

### 3. tasks.py、block.py（保持）

Stride 初始化适配 per-scale feats；SPPF 显式循环。无变化。

---

## 影响分析

| 项目       | 说明                                                |
| ---------- | --------------------------------------------------- |
| 预训练权重 | **完全兼容**（708/708 匹配）                        |
| 推理输出   | 格式不变 `(B, 300, 6)`                              |
| ONNX 导出  | per-scale 版本输出数量变化（2→6 或 5→9）            |
| 部署工具   | 需适配 per-scale 多输出格式；matmul 必须 S16        |
| 训练 loss  | 与原始 concat 版本等价（torch.cat 汇合后走原 loss） |
| 导出质量   | exp32 one2many: 零 Cast、零冗余 DQ-Q                |

---

## 相关实验

| 实验  | matmul | head 版本                  |        最佳 mAP        | ONNX 导出质量        |
| ----- | :----: | -------------------------- | :--------------------: | -------------------- |
| exp28 |   S8   | v1 原始 Concat             |         39.63          | 冗余 Q-DQ            |
| exp29 |   S8   | v2 \_merge_per_scale_preds |         39.64          | 需 strict=False 加载 |
| exp30 |   S8   | v3 concat_flag             |      39.60 (3ep)       | Good                 |
| exp31 |  S16   | v3 concat_flag             |      39.74 (3ep)       | Perfect (零冗余)     |
| exp32 |  S16   | v3 concat_flag             | **39.82** (50ep运行中) | Perfect (零冗余)     |
