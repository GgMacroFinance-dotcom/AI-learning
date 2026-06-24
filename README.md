# ResNet 论文复现

本目录复现 PDF 中的经典论文 **Deep Residual Learning for Image Recognition**。代码实现了残差连接、BasicBlock、Bottleneck，以及论文中常用的 CIFAR-10 和 ImageNet 风格 ResNet 结构。

## 文件结构

```text
resnet/
|-- 1、ResNet 2015年12月.pdf
|-- model.py          # ResNet-20/32/44/56/110 和 ResNet-18/34/50/101/152
|-- cifar10_data.py   # 不依赖 torchvision 的 CIFAR-10 下载、解析、增强
|-- train.py          # CIFAR-10 训练脚本
|-- predict.py        # 单张图片推理脚本
`-- README.md
```

## 论文要点

- 核心思想：学习残差函数 `F(x) = H(x) - x`，网络输出为 `F(x) + x`。
- 直接收益：更深的网络不再因为退化问题难以优化。
- CIFAR-10 设置：论文使用 `6n + 2` 层的 ResNet，如 ResNet-20、32、44、56、110。
- ImageNet 设置：论文使用 BasicBlock 构建 ResNet-18/34，使用 Bottleneck 构建 ResNet-50/101/152。

## 快速验证

先在项目根目录安装依赖：

```powershell
python -m pip install -r requirements.txt
```

跑一个小样本 sanity check：

```powershell
python .\resnet\train.py --epochs 1 --limit-train 512 --limit-test 256 --model resnet20
```

完整训练 ResNet-20：

```powershell
python .\resnet\train.py --model resnet20 --epochs 160 --batch-size 128
```

训练完成后会保存最佳 checkpoint：

```text
models/resnet20_cifar10.pt
```

## 推理

```powershell
python .\resnet\predict.py --image .\path\to\image.png --model-path .\models\resnet20_cifar10.pt
```

## 可选模型

```text
resnet20, resnet32, resnet44, resnet56, resnet110
resnet18, resnet34, resnet50, resnet101, resnet152
```

其中 `resnet20` 到 `resnet110` 使用 CIFAR-10 论文结构；`resnet18` 到 `resnet152` 使用 ImageNet 风格结构。当前训练脚本默认训练 CIFAR-10，因此建议优先使用 `resnet20` 或 `resnet32` 做本地实验。
