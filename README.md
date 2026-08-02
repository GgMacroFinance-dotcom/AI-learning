# AI-learning

基于 PyTorch 的 CIFAR-10 ResNet 复现与学习仓库。
当前代码实现 CIFAR-10 数据加载、ResNet 模型定义、训练脚本和单图推理脚本，适合用来理解 ResNet 训练流程与复现小规模图像分类实验。

---

## 功能说明

- 自动下载并准备 CIFAR-10 数据集
- 支持 CIFAR-10 版本 ResNet：`resnet20`、`resnet32`、`resnet44`、`resnet56`、`resnet110`
- 支持标准 ResNet 变体：`resnet18`、`resnet34`、`resnet50`、`resnet101`、`resnet152`
- 训练脚本 `train.py`：SGD + MultiStepLR，输出训练损失、测试损失、测试准确率和最佳 checkpoint
- 推理脚本 `predict.py`：加载 checkpoint 对单张 RGB 图像进行 CIFAR-10 预测

---

## 当前目录结构

```text
AI-learning/
├── README.md                 # 本说明文档
├── cifar10_data.py           # CIFAR-10 数据下载、Dataset、数据增强
├── model.py                  # ResNet 模型定义与构建函数
├── train.py                  # 训练入口脚本
├── predict.py                # 单图像推理入口脚本
└── 1、ResNet 2015年12月.pdf  # ResNet 论文参考
```

### 主要文件说明

- `cifar10_data.py`：加载 CIFAR-10 训练/测试集，支持随机裁剪和水平翻转增强
- `model.py`：实现 CIFAR stem 的 ResNet 和 ImageNet 风格 ResNet，并暴露 `build_model()`
- `train.py`：按命令行参数训练模型并保存最优 checkpoint
- `predict.py`：读取指定模型权重，预测单张 RGB 图像的 CIFAR-10 类别

---

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd AI-learning
```

### 2. 创建环境并安装依赖

```bash
# 建议使用虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 安装最小依赖
pip install torch torchvision numpy pillow
```

### 3. 训练模型

```bash
python train.py --model resnet20 --epochs 160 --batch-size 128 --lr 0.1
```

如果你只是想快速跑通一次，可以：

```bash
python train.py --epochs 1 --limit-train 512 --limit-test 128
```

### 4. 推理单张图像

```bash
python predict.py --image path/to/image.png --model-path models/resnet20_cifar10.pt
```

---

## 环境要求

- Python 3.9+ 或更高
- PyTorch
- torchvision
- NumPy
- Pillow

## 推荐学习方式

1. **先看模型代码**：阅读 `model.py` 中的 ResNet 结构和 `cifar10_data.py` 中的 CIFAR-10 数据预处理。  
2. **再跑训练**：用 `train.py` 训练一个小模型，观察训练损失和测试准确率。  
3. **再看推理**：运行 `predict.py`，理解如何读取 checkpoint 并对单张图像预测。  
4. **最后做小实验**：修改模型深度、数据增强、学习率等参数，比较结果。

---

## 使用建议

- 每个示例尽量保持“单一目标”，方便复用和回顾
- 数据和大文件不要直接塞进 Git；可用说明文件记录来源与下载方式
- 实验参数尽量放进 `config/`，避免硬编码
- 重要结论写回 `docs/`，形成可检索的知识库

---

## 贡献与更新

这是学习型仓库，欢迎按主题持续补充：

1. 新增笔记时，补充背景、要点、参考链接
2. 新增示例时，写清“依赖、输入、如何运行、预期输出”
3. 做完实验后，记录关键参数和结论，方便以后回看

---

## 许可证

如无特殊说明，仅供学习与交流使用。
