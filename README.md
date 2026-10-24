# 基于预训练 ViT 的 CIFAR-10 图像分类

这个项目使用 PyTorch 在本机 GPU 上微调预训练的 Vision Transformer（ViT Tiny），完成 CIFAR-10 十分类。项目按数据、训练、评估三个阶段组织，便于学习完整的图像分类工作流程。

## 第 1 步：准备本地环境

在 VS Code 中打开本文件所在文件夹，并打开终端。此电脑的 Miniconda 基础环境已安装可用的 GPU 版 PyTorch。下面的命令建立一个项目虚拟环境，复用基础环境中的 PyTorch，并把新依赖安装在项目环境中：

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/check_environment.py
```

最后一条命令应显示 `CUDA available: True` 和 `GPU calculation: 6.0`。在 VS Code 中用 `Ctrl+Shift+P` → `Python: Select Interpreter` 选择本项目的 `.venv`。

`--system-site-packages` 意味着这个虚拟环境能够读取基础环境的软件包，因此它不是完全隔离的。换电脑复现时，建议使用普通 `python -m venv .venv`，然后按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 为该环境安装 GPU 版 PyTorch，再安装 `requirements.txt`。本机已验证 PyTorch 2.7.0+cu128、torchvision 0.22.0+cu128、timm 1.0.30 与 RTX 5060 Laptop GPU 可用。

## 第 2 步：认识数据集

[CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) 有 10 类、60,000 张 32×32 彩色图片。官方划分为 50,000 张训练图和 10,000 张测试图；本项目从官方训练集再固定划出 5,000 张作为验证集，因此实际使用 45,000 张训练、5,000 张验证、10,000 张最终测试。测试集不参与模型选择。

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts/preview_data.py
```

首次运行会下载约 170 MB 的数据，完成后生成 `runs/data_preview.png`：每类一张原始图片。`data/` 和 `runs/` 已被 Git 忽略，不会误传到 GitHub。

如果官方下载很慢，可使用 [Zenodo 上的原始文件副本](https://zenodo.org/records/10089977) 下载。下面的命令支持接着已有 `.part` 文件下载；数据集已经就绪时无需执行：

```powershell
curl.exe -L --fail --retry 5 -C - -o data\cifar-10-python.tar.gz.part "https://zenodo.org/records/10089977/files/cifar-10-python.tar.gz?download=1"
$hash = (Get-FileHash data\cifar-10-python.tar.gz.part -Algorithm MD5).Hash.ToLower()
if ($hash -ne 'c58f30108f718f92721af3b95e74349a') { throw "数据文件校验失败：$hash" }
Move-Item data\cifar-10-python.tar.gz.part data\cifar-10-python.tar.gz
.\.venv\Scripts\python.exe scripts/preview_data.py
```

下载中断时，重新执行第一行即可续传。只有校验通过才执行后面的 `Move-Item`；若使用 PowerShell 逐行运行，校验失败后不要继续执行最后两行。

## 第 3 步：训练模型

模型为 [`vit_tiny_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k)：先在 ImageNet-21k 预训练、再在 ImageNet-1k 微调的 ViT Tiny。我们将分类头替换为 10 类输出后，对全部参数进行微调。实际模型约 553 万参数：224×224 输入切成 196 个 16×16 图像块，使用 192 维嵌入、12 层 Transformer、每层 3 个注意力头，最后通过分类头输出 10 类分数。

训练时先对 32×32 原图随机裁剪（边缘填充 4 像素）和水平翻转，再放大到 224×224；验证和测试只调整大小。三个通道均使用 mean=0.5、std=0.5 归一化，与该预训练权重配置一致。

先进行小规模试跑，检查整个流程；**试跑准确率不是最终结果**：

```powershell
.\.venv\Scripts\python.exe -m src.train --epochs 1 --batch-size 8 --workers 0 --max-train-batches 2 --max-val-batches 2 --run-dir runs/smoke
```

确认试跑完成后运行完整训练。下面第一条是基线设置，第二条是优化后的设置，两者的区别见「实测结果」：

```powershell
# 基线：恒定学习率、5 轮、随机验证集划分
.\.venv\Scripts\python.exe -m src.train --epochs 5 --batch-size 32 --workers 0 --warmup-epochs 0 --run-dir runs/vit-tiny

# v2：warmup 2 轮 + 余弦衰减、20 轮、分层验证集、梯度裁剪
.\.venv\Scripts\python.exe -m src.train --epochs 20 --batch-size 32 --workers 2 --run-dir runs/vit-tiny-v2

# v3：v2 的全部设置 + RandAugment
.\.venv\Scripts\python.exe -m src.train --epochs 20 --batch-size 32 --workers 2 --rand-augment-ops 2 --rand-augment-magnitude 9 --run-dir runs/vit-tiny-v3
```

程序会逐轮输出学习率、训练与验证损失、准确率，保存最佳验证模型到 `best.pt`，并保存每轮记录到 `history.json`。如遇 GPU 显存不足，把 `--batch-size 32` 改为 `16` 或 `8`。

可调参数：`--lr`（峰值学习率）、`--min-lr`（余弦衰减终点）、`--warmup-epochs`（预热轮数）、`--clip-grad-norm`（梯度裁剪阈值）、`--val-per-class`（每类验证样本数）。把 `--workers 2` 改为 `0` 也能跑，但数据加载会成为瓶颈，每轮从约 1.5 分钟变成约 10 分钟。

## 第 4 步：独立测试与图表

完整训练后执行下面的命令，加载最佳验证模型，评估 10,000 张测试图片。它不会重新训练或下载预训练权重：

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --checkpoint runs/vit-tiny-v3/best.pt --output-dir reports-v3
```

程序生成测试指标、混淆矩阵、训练曲线，以及各 8 张预测正确和错误的示例。示例在相应集合内以固定种子随机抽取；图中的百分比是模型预测类别的 softmax 分数，不代表预测一定正确。评估更早的 checkpoint 时把两个路径换成 `runs/vit-tiny-v2/best.pt` + `reports-v2` 或 `runs/vit-tiny/best.pt` + `reports` 即可。

## 实测结果

三轮实验都在同一台 RTX 5060 Laptop GPU 上完成，使用同一种子（42）、同一个 CIFAR-10 官方测试集。测试集只在最后评估时使用一次，不参与模型选择。每一轮只改一项，因此提升可以单独归因。

| 实验 | 训练设置 | 最佳验证准确率 | 独立测试准确率 |
| --- | --- | --- | --- |
| v1 基线 | AdamW lr=3e-4 恒定，5 轮，随机划分验证集 | 92.84% | 92.43% |
| v2 调度与时长 | warmup 2 轮 + 余弦衰减至 1e-5，20 轮，分层验证集，梯度裁剪 1.0 | 96.36% | 96.05% |
| v3 数据增强 | v2 的全部设置 + RandAugment（N=2, M=9） | **96.70%** | **96.41%** |

三轮测试正确数分别为 9,243 / 9,605 / 9,641（共 10,000 张）。累计提升 **+3.98 个百分点**，其中绝大部分来自第二轮。

### 第一轮：学习率调度与训练时长（+3.62pp）

v1 的验证准确率在第 5 轮仍在上升、训练损失仍在下降，模型远未收敛——这是最大的一块未收割收益。

| 改动 | 作用 |
| --- | --- |
| 训练轮数 5 → 20 | 收割未收敛的收益 |
| 余弦学习率衰减（预热 2 轮到峰值 3e-4，再降到 1e-5） | 预热保护预训练特征不被初始大步长破坏；余弦尾部让模型在收敛区精细调整，而不是在恒定学习率附近震荡 |
| 梯度裁剪 1.0 | 抑制梯度尖峰；混合精度下先 `unscale_` 再裁剪 |
| 分层验证集（每类 500 张） | 固定 10 类的比例，验证准确率不再因抽样波动而抖动，选出的 checkpoint 更可靠 |

v2 的 20 轮验证准确率从 90.76% 单调爬升到 96.36%，最佳轮次就是最后一轮——这是余弦调度生效的典型形态。

### 第二轮：RandAugment（+0.36pp，主要收获是泛化）

v2 暴露了明确的过拟合：最后一轮训练准确率 99.94%、训练损失 0.0016，而验证准确率 96.36%，**训练—验证差距 3.58pp**；错误样本的平均置信度 0.902，明显高于 v1 的 0.686——模型对错误答案也更确定了。

加入 RandAugment 后：

| 指标 | v2 | v3 | 变化 |
| --- | --- | --- | --- |
| 训练—验证差距 | 3.58pp | **2.32pp** | 收窄 1.26pp |
| 最后一轮验证损失 | 0.2361 | **0.1652** | 显著下降 |
| 错误样本平均置信度 | 0.902 | 0.866 | 不再那么「自信地犯错」 |
| 测试准确率 | 96.05% | 96.41% | +0.36pp |

测试准确率只涨了 0.36pp，但验证损失大幅下降、过拟合明显收窄，说明模型的**稳健性**改善了。这类收益在单次单种子的实验里不容易体现在准确率上，却是更可靠的改进。

### 错误分析（三轮对比）

| 类别 | v1 召回 | v2 召回 | v3 召回 |
| --- | --- | --- | --- |
| dog | 84.1% | 91.1% | **94.0%** |
| bird | 90.0% | 94.9% | 96.3% |
| cat | 89.7% | 92.1% | 91.7% |
| truck | 94.8% | 97.3% | 96.2% |

猫狗互认从 v1 的 172 张降到 v3 的 94 张，仍是最大错误来源；其次是 automobile↔truck 40 张。原图只有 32×32、外观相近，属于任务本身的难度。truck 和 cat 的召回有小幅回落（−1.1pp、−0.4pp），这是增强带来的正常代价。

### 置信度

| 实验 | 平均置信度 | 错误样本置信度 | conf≥0.9 覆盖率 | 该子集准确率 |
| --- | --- | --- | --- | --- |
| v1 | 0.934 | 0.686 | 81.3% | 98.38% |
| v2 | 0.992 | 0.902 | 97.7% | 97.13% |
| v3 | 0.989 | 0.866 | 96.9% | 97.74% |

v3 在高置信度子集上的准确率最高（97.74%），且错误样本的自信程度最低——如果实际使用中加一道置信度阈值，v3 是最合适的选择。

![v3 训练损失和准确率曲线](reports-v3/training_curves.png)

![v3 测试集混淆矩阵：纵轴为真实类别，横轴为预测类别](reports-v3/confusion_matrix.png)

![v3 预测正确示例](reports-v3/correct_predictions.png)

![v3 预测错误示例](reports-v3/incorrect_predictions.png)

v2 与基线的对应图表分别在 `reports-v2/` 和 `reports/` 目录。原始指标见 [reports-v3/test_metrics.json](reports-v3/test_metrics.json)，每轮记录见 [reports-v3/training_history.json](reports-v3/training_history.json)。这是固定划分、固定种子下的实验结果；预训练数据和训练方式与从零训练模型不同，比较结果时应注明这些条件。若要进一步验证稳健性，可以更换种子重复实验。

## 文件与复现

- `src/data.py`：图像处理、训练/验证分层划分、可选 RandAugment。
- `src/model.py`：构建预训练 ViT 和十分类头。
- `src/train.py`：训练、验证、学习率调度与最佳模型保存。
- `src/evaluate.py`：独立测试与图表生成。
- `reports/`、`reports-v2/`、`reports-v3/`：三轮实验的结果图表与指标。
- `runs/`、`data/`、`.venv/`：本地权重、数据和环境，Git 已忽略。

模型权重未包含在 Git 仓库内，复现评估前需要先运行训练命令。不同硬件与非确定性 GPU 运算可能造成细小差异。代码测试命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
