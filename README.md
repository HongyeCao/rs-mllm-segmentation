# 遥感分割 V1：安全标签监督 + 数据集专属分类头

本文件夹使用你上传的 `train.py`、`eval.py`、`train_dataset.json`、`val_dataset.json` 的数据约定，**不会覆盖原工程文件/掩码**。V1 是基础实验原型，不包含论文拟议的类别语义对齐 M1 或多尺度融合 M2。Python 代码已在离线环境执行语法和单元检查，但无法在本环境加载用户 WSL 的模型权重与真实 PNG，更未完成 GPU 训练验证。

## 重要判断

- `LoveDA` 原始 `0` 为 no-data，应忽略；原始 `1..7` 对应 7 个有效类别，应映射成 `0..6`。
- `Potsdam`、`Vaihingen` 按 ISPRS 官方任务均为 6 个语义类，但**本地的标签编号未核实**。`configs/label_maps.json` 为它们设置了 `verified=false`，正式训练自动拒绝这两套数据，直到确认本地掩码的编号和语义关系。
- `train_dataset.json` 36440 条，`val_dataset.json` 23635 条；JSON 里 `mask` 指向 Windows WSL 机器上绝对路径，必须在同一台 WSL 主机运行审计。
- 检查点从 V0 的六类输出改成 V1 三个独立输出头，**不可直接恢复 V0 头权重作为正式 V1**。

## 步骤 0：备份

```bash
cd ~/rs_project
mkdir -p experiments/v0_initial/checkpoints
cp train.py eval.py inference_single.py build_multi_dataset_json.py experiments/v0_initial/
cp -a checkpoints/step_10000 experiments/v0_initial/checkpoints/
```

## 步骤 1：安装新脚本到项目根目录

把压缩包解压后，将 `configs/` 目录和 `label_pipeline.py`、`v1_core.py`、`train_v1.py`、`eval_v1.py` 放到 `~/rs_project/`（不要替换旧的 `train.py`/`eval.py`）。

```bash
cd ~/rs_project
conda activate rs_mllm
python -m py_compile label_pipeline.py v1_core.py train_v1.py eval_v1.py
python tests/test_labels.py
```

## 步骤 2：审核标签，禁止直接猜测本地编号

```bash
python label_pipeline.py audit --limit 200 --out label_audit_200.json
python label_pipeline.py audit --limit 0 --out label_audit_all.json
```

`--limit` 是**每个数据集**分别抽取样本数。完整审计要读取 60,075 张掩码，耗时可能较长。请人工检查 Potsdam/Vaihingen 的原始彩色标签转换脚本及各 raw ID 所代表的语义，并核对 PNG、可视化和官方类别顺序。确认后才编辑 `configs/label_maps.json`：填入映射并设 `verified=true`。

可选：用原来的 `label_pipeline.py build` 生成另存标签，但 V1 默认采用**加载时映射**，因此不需要重新生成 60,075 张掩码。

## 步骤 3：标签正确后的小样本过拟合实验

**第一个实验只用已验证的 LoveDA：**

```bash
python train_v1.py --datasets LoveDA --limit-samples 16 --epochs 6 --max-steps 0 --accum 8 --output experiments/v1_loveda_16
```

这是一项 16 张样本、重复 6 轮的过拟合诊断；应观察 Loss 下降和分割图越来越接近 GT。`--max-steps 0` 表示无步数截断。它不能用于论文泛化性能报告。

如果需要先做最短烟雾测试：

```bash
python train_v1.py --datasets LoveDA --limit-samples 8 --epochs 1 --max-steps 8 --output experiments/v1_smoke
```

### 完成分类映射后，才能执行联合训练

```bash
python train_v1.py --datasets LoveDA Potsdam Vaihingen --limit-samples 0 --max-steps 0 --epochs 1 --output experiments/v1_joint_baseline
```

这个版本为单样本微批、固定梯度累积和常规样本比例随机混训；**不是三数据集均衡采样**。论文需要另外设计均衡采样和独立基线对照。

## 步骤 4：使用已有验证集进行独立评估

```bash
python eval_v1.py --checkpoint experiments/v1_loveda_16/epoch_06_step_0000096 --dataset LoveDA --limit 32 --visualize 6 --output results/v1_debug
```

以上用于演示脚本调用方式。若训练时跳过全 255 掩码导致实际步数不同，需按输出目录真实名称替换 `--checkpoint`。正式评估使用 `--limit 0`，并分别对三个数据集运行命令。结果输出为 `metrics.json`、`per_class_iou.csv`、`compare_*.png`。

`mIoU_present` 表示本次评估中有效出现类别的平均 IoU；正式论文须按各数据集官方协议确定类别纳入规则，并报告所有类别的 IoU。

## 关键技术限制和后续检查

1. 真实 WSL 的 `transformers`/`peft` 版本可能与本环境不同；**先进行 8-step smoke test**，尤其是 Qwen2-VL 的 `image_grid_thw`、`pixel_values` 以及 `hidden_states` 维度。
2. `v1_core.py` 从 `image_grid_thw` 和 `spatial_merge_size` 计算视觉网格，并对 image token 数进行强制断言，不再任意截取前 324 个 Token。该方式仍需在用户具体 transformers 版本中确认空间排序。
3. Batch 固定为 1，避免不同图片分辨率和不同数据集 head 的批处理冲突；数据尺寸是 512×512，Qwen2-VL 的处理器可能会进行额外尺寸调整，以处理器真实网格为准。
4. 为了不超出显存，`--save-optimizer` 默认为关闭。若希望支持精确断点续训，必须额外完善 optimizer、scheduler、梯度和采样器状态的保存/恢复；当前脚本只保存了状态，不提供完整 resume 命令。
5. 训练/推理统一用户指令，不使用训练 JSON 中的固定助手回答 `Sure, [SEG].`。类别语义对齐 M1 尚未实现。
6. `LoveDA` 原始数据集许可限制学术用途；按具体数据集许可使用与引用。

## 原始官方定义

- LoveDA: https://github.com/Junjue-Wang/LoveDA
- ISPRS Potsdam/Vaihingen: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx
