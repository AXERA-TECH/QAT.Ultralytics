# Debug QAT 量化配置说明

本目录用于归档调试阶段使用过的量化配置，避免实验文件散落在仓库根目录。

## 文件用途

- `config_before_tune10.json`
  - `tune10` 之前的默认非对称量化配置（全局 `U8/S8`，无区域覆盖）。
- `config_asym_backup.json`
  - 默认非对称量化配置备份，与 `config_before_tune10.json` 等价保留。
- `config_headfp32.json`
  - `tune10` 使用：将检测头 `conv2d_63~conv2d_86` 输入设为 `FP32`。
- `config_neckhead_fp32.json`
  - `tune11` 使用：将 Neck+Head `conv2d_40~conv2d_86` 输入设为 `FP32`。
