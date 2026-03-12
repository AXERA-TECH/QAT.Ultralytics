# Debug QAT 量化配置说明

本目录仅保留当前后续实验计划仍直接相关的量化配置。

## 文件用途

- `config_hist_act.json`
  - 当前 observer 主线实验使用：激活 observer 为 `Histogram`，权重 observer 保持 `moving_avg_per_channel`。
