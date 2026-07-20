# QAT.Ultralytics → torch 2.10 迁移进度

> **状态(2026-07-15):迁移完成并端到端验证 ✅。** 真实 YOLO11 QAT 在 torch 2.10
> 训练 1 epoch → convert → 导出 `qat_slim.onnx`,结构体检 11 PASS/0 FAIL/1 良性 WARN。
> 唯一未做:pulsar2 编译 axmodel 部署 NPU(需 pulsar2 工具链,单独步骤)。
> 采用 **drop-in 覆盖**:utils 量化文件与 QAT.axera 同源,直接覆盖一举完成 ①②④,
> 项目专属 glue(engine 调用点/capture/pt2e_bn_patch/内联 optimize)逐项处理。
>
> 目标:把本项目(YOLO11 PT2E QAT,现锁 torch 2.6)切到 torch 2.10 跑通。
> 方法:遵循 QAT.axera 的 skill `qat-migrate-2_10`,以 QAT.axera 的 `utils/` 为
> **可复用 drop-in 实现**。**策略:2.10-only 切换**,但兼容层 `_compat.py` 沿用
> QAT.axera 的双轨写法(零成本保留 2.6 路径,严格按 major.minor 路由)。
>
> 本文件在 QAT.Ultralytics 仓库内(该仓库在 QAT.axera 的 cache/ 下,已 .gitignore)。

## 第 0 步:活/死代码甄别(已完成)

追入口(export.py / eval.py)→ engine(exporter/model/predictor/validator)调用链:

- **活路径(唯一)**:`ultralytics/utils/ax_quantizer.py` 的 `AXQuantizer`(config 驱动)
  + `torch.ao.quantization.quantize_pt2e` + `torch.export.export_for_training`。
  连带活文件:`ax_quantizer_utils.py`(注解器)、`quant_utils.py`、`train_utils.py`、
  `quantized_decomposed_dequantize_per_channel.py`、`pt2e_bn_patch.py`
  (`utils/__init__.py:134` 模块级激活)。
- **死代码(不改)**:`quantizer.py`(nn_module_stack 老式,仅自引 + model.py:40 注释掉的
  import)、`quantizer_utils.py`(仅被前者引)、`ax_quantizer_lsq.py`(LSQ,全仓无 import)。

## torch 2.10 实测证伤(迁移前基线)

- 真实 AXQuantizer:`prepare_qat_pt2e → _annotate_conv` 报
  `ImportError: gm_using_training_ir from torch._export`(2.10 已删)——prepare 就炸。
- 原厂 XNNPACKQuantizer 隔离:`convert_pt2e` 报 `KeyError: 'source_fn_stack'`
  ——torch.ao PT2E 内核在 2.10 已坏。

## 迁移清单(按 skill 四类坑)

> **最快路径已采用:直接覆盖(drop-in)**。经对比,本项目 utils 的量化文件与
> QAT.axera `utils/` 同源(注解器集合、OPS 列表完全一致,分歧 100% 是迁移本身,
> 零 YOLO 专属逻辑),故直接用 QAT.axera 的 5 个文件**覆盖**,一举完成 ①②④。
> 只有项目专属 glue(engine/export 调用点、pt2e_bn_patch、内联 optimize)逐项处理。

| # | 坑 | 涉及文件 | 状态 |
|---|----|---------|------|
| 0 | 活/死代码甄别 | — | ✅ |
| ① | torch.ao → torchao(整包) | **覆盖** ax_quantizer/ax_quantizer_utils/quant_utils/train_utils + 新建 `_compat.py`;外部 export.py + engine 四处 PT2E import/move/allow 路由 `_compat` | ✅ |
| ② | 元数据换代 | **覆盖**带来 aten 直匹配注解器 + gru/mha 守卫 + remove_reused_bn_hack(num_batches_tracked 版);cat 注解已实测生效 | ✅ |
| ②' | 捕获入口 | export_for_training → `_compat.capture_for_training`(双轨),export.py + engine 五处 | ✅ |
| ③ | BN patch 搬家 | pt2e_bn_patch.py import 改版本感知(2.10→`torchao.quantization.pt2e`);实测 patch 落到 torchao 且保留 YOLO BN 超参 eps=1e-3/mom=0.03 | ✅ |
| ④ | 导出后处理 | **覆盖** train_utils.dynamo_export(optimize=False+内联+ir)/quant_utils.simplify(双格式+castlike+ir);另去掉 export.py+exporter.py 内联 `onnx_program.optimize()` | ✅ |
| V | 验证 | 小闭环(conv-bn-relu+cat)✅;**真实 YOLO11 端到端**:装全栈依赖(torch 未降级)→ import 迁移包 OK → train 1 epoch(`use qat optimizer`,loss 正常)→ convert → 导出 `qat_smoke_qat_slim.onnx` → 结构体检 11 PASS/0 FAIL/1 良性 WARN ✅;NPU 编译部署待 pulsar2 | ✅ |

## 下一步计划(详细)

### V 端到端验证(唯一剩余)

- 装全栈依赖(**不动 torch/torchvision,防降级到 2.6**):清华源装
  opencv-python-headless/pandas/matplotlib/scipy/psutil/requests/pillow 等,或
  `pip install -e . --no-deps` 后补缺失;
- 小批量 `train.py` 1 epoch → `eval.py` 对齐 → `export.py` 出 `qat_slim.onnx`
  → 结构体检(qat-check)→ 按 `compile/qat_deployment.md` 出 axmodel 部署 NPU。
  **先打通链路再谈精度**。

---

<details><summary>已完成项的原始计划(存档)</summary>

### ② 注解器 aten 直匹配(从 QAT.axera `utils/ax_quantizer_utils.py` 移植)

- 活列表 `OPS` 含 avgpool2d/concat/layernorm/groupnorm 等;`get_source_partitions`
  在 2.10 返回空 → 这些注解器**静默 no-op**(不崩,但配置到的算子不被量化)。
- **concat(cat,L1308)对 YOLO11 检测头关键**(neck 大量 Concat),必迁;
  avgpool2d(L751)分类头用到;layernorm(L803)/groupnorm(L852)YOLO11 不用但
  注解器仍会跑——四个都按我们已迁移的 **aten 直匹配版**逐个对拍移植
  (注意两仓库注解器签名/融合处理差异,不能整段覆盖)。
- gru(L697)/mha(L1405)**不在 OPS**(死路径)→ 加 `NotImplementedError` 守卫,
  防将来被纳入 OPS 时静默漏注解。
- `ax_quantizer.remove_reused_bn_param_hack`(读 source_fn_stack)→ 改按
  `num_batches_tracked` buffer 名识别的版本无关实现(YOLO11 未必复用 BN,防御性改)。
- **验证**:桩加载跑一个两分支 `torch.cat` 的小网络,确认 cat 被注解(注解数 > 0)。

### ③ pt2e_bn_patch 搬家

- 现 patch `torch.ao.quantization.pt2e.export_utils._replace_batchnorm`;迁 torchao 后
  BN 折叠走 `torchao.quantization.pt2e.export_utils`。做法:`pt2e_bn_patch.py` 内按
  `_compat.IS_TORCH_210` 选目标模块;先确认 torchao 版是否仍需此 patch(可能已修),
  需要才打。**验证**:import 后 patch 落在正确命名空间不报错;BN 超参在
  `move_exported_model_to_eval` 后不被重置。

### ④ 导出后处理(从 QAT.axera `train_utils.py`/`quant_utils.py` 移植)

- `train_utils.py`:①遗留 `move_exported_model_to_eval`(L173,torch.ao)→ 路由 `_compat`;
  ②`dynamo_export` 的 `optimize()`(train_utils + export.py:66 + exporter.py:1344 三处)
  → 改 `optimize=False` + `onnx.inliner.inline_local_functions` + ir_version 回写 10。
- `quant_utils.simplify_and_fix_4bit_dtype`:namespace 单格式解析(L41)→ 优先 fx_node
  regex 提 target + namespace 回退(双格式);补 `_castlike_to_cast`(FP32 区零 bias
  残链)+ ir_version 回写。

### V 端到端验证

- 装全栈依赖(**不动 torch/torchvision,防降级到 2.6**):清华源装
  opencv-python-headless/pandas/matplotlib/scipy/psutil/requests/pillow 等,或
  `pip install -e . --no-deps` 后补缺失;
- 小批量 `train.py` 1 epoch → `eval.py` 对齐 → `export.py` 出 `qat_slim.onnx`
  → 结构体检(qat-check)→ 按 `compile/qat_deployment.md` 出 axmodel 部署 NPU。
  **先打通链路再谈精度**(呼应 qat-new-model)。

</details>

## 执行日志

- **2026-07-15 ①+②-conv+②'(已验证)**:新建 `ultralytics/utils/_compat.py`(移植 QAT.axera 双轨兼容层,严格 major.minor 路由,含 `get_aten_graph_module_for_pattern` 包装 + `capture_for_training` 双轨 helper)。utils 三文件 torch.ao import 全路由 `_compat`;ax_quantizer_utils 两处 `gm_using_training_ir` 调用点改用包装。外部 export.py + engine 四文件 PT2E import/move/allow 路由 `_compat`,五处 `export_for_training` → `capture_for_training`。
  - **验证**:桩加载真实 AXQuantizer + torchao PT2E 跑 conv-bn-relu:prepare OK(注解 4)、convert OK(quant 节点 5)——迁移前两处报错(gm_using_training_ir ImportError / source_fn_stack KeyError)均消除。
  - py_compile:9 个改动文件全过。
  - 复现脚本:scratchpad/repro_ultra_210_v3.py。
- **2026-07-15 改用 drop-in 覆盖(完成 ①②④)**:经结构对比确认 utils 量化文件与
  QAT.axera 同源(注解器/OPS 完全一致,分歧纯迁移),遂用 QAT.axera `utils/` 的 5 个
  文件(ax_quantizer/ax_quantizer_utils/quant_utils/train_utils/quantized_decomposed_
  dequantize_per_channel)直接**覆盖**。① step1 的手工路由被覆盖为干净版本;② 注解器
  自动变 aten 直匹配(get_source_partitions 弃用);④ dynamo_export/simplify 得迁移版。
  - **验证**:conv-bn-relu + cat 双分支闭环 prepare/convert 全绿,**cat 被注解 1/1**
    (② concat 注解器生效);复现 scratchpad/repro_ultra_210_v4.py。
- **2026-07-15 ③ + ④内联(已验证)**:pt2e_bn_patch.py import 改版本感知
  (2.10→torchao,保留 YOLO BN eps=1e-3/mom=0.03 patch);去掉 export.py + exporter.py
  内联 `onnx_program.optimize()`。**验证**:patch 实测落到 `torchao.quantization.pt2e.
  export_utils._replace_batchnorm`;py_compile 全过。补丁脚本 migrate_step2/step3.py。
- **状态**:①②②'③④ 全部完成并验证(utils 层 + 调用点)。
- **2026-07-15 V 端到端验证 ✅**:全栈依赖装入 `torch2.10` env(清华源 + 约束钉死
  torch/torchvision/torchao/onnx 防降级;opencv 用 headless;补 onnxsim)。
  `import ultralytics` 在 2.10 跑通(③ pt2e_bn_patch 激活落 torchao)。
  `smoke_train.py`:yolo11n + coco(fraction 0.003)+ 1 epoch + GPU2 →
  `use qat optimizer!`(prepare_qat_pt2e 成功)→ QAT 训练 89 iter loss 正常 →
  convert_pt2e → 导出 `runs/qat_smoke_qat_slim.onnx`(2.84MB)。
  onnxscript 0.6.2 保留权重 DQ(`node_dequantize_per_channel_* preserved`)。
  **结构体检**(env_check --sim):11 PASS/0 FAIL/1 良性 WARN——opset21/ir10、
  233 激活 Q 成对、94 权重 DQ 全 per-channel、Concat 被量化(② 生效)、bias 浮点。
  **迁移全线打通,仅 pulsar2→axmodel NPU 部署待做(单独工具链)。**
