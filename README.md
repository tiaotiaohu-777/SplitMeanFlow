# SplitMeanFlow

本科毕业设计配套代码。针对扩散模型推理速度慢的问题，基于字节跳动研究团队提出的 **Split MeanFlow** 模型，实现了一套两阶段知识蒸馏框架，用于高效少步图像生成。

实验选用 ImageNet 中 **50 个子类**（涵盖鱼类、鸟类、犬类、猫科、熊类、哺乳动物、昆虫、爬行类、食物共 9 大类），在保证类别多样性的同时控制实验规模。

## 方法

算法设计参考了字节跳动研究团队提出的 Split MeanFlow 模型 [Guo et al., 2025]。

两阶段蒸馏框架：
- **Stage 1** — 训练教师模型，学习瞬时速度场
- **Stage 2** — 训练学生模型，引入教师输出作为边界条件，结合 Split MeanFlow 损失函数

**网络架构：** DiT (SiT-XL/2)，28 层 Transformer Block，隐藏维度 1152，16 头注意力，adaLN-Zero 条件注入。使用预训练 SD-VAE 将图像压缩至潜空间。

## 使用

代码不能直接运行，需下载权重并修改 `config.yaml` 中的路径。

```bash
pip install torch torchvision diffusers cleanfid tqdm pyyaml

# 1. 数据准备
python scripts/extract_imagenet_50classes.py
# 2. 训练教师模型
python scripts/train_teacher.py
# 3. 训练学生模型
python scripts/train_student.py
# 4. 采样与评估
python scripts/sample.py
python scripts/evaluate_fid.py
```

需提前下载：`sd-vae-ft-mse-diffusers`（Hugging Face）和 `SiT-XL-2-256.pt`（SiT 官方仓库），下载后修改 `config.yaml` 中所有 `/root/autodl-tmp/...` 路径。

## 项目结构

```
splitmeanflow/
├── config.yaml                 # 全局配置（数据路径、模型参数、训练超参数）
├── scripts/
│   ├── train_teacher.py        # Stage 1: 训练教师模型
│   ├── train_student.py        # Stage 2: 训练学生模型
│   ├── sample.py               # 学生模型采样生成图片
│   ├── sample_teacher.py       # 教师模型采样生成图片
│   ├── evaluate_fid.py         # 计算 FID 评估指标
│   ├── extract_imagenet_50classes.py  # 从 ImageNet 提取 50 个子类
│   └── prepare_latents.py      # 预计算 VAE 潜空间特征
├── src/
│   ├── models.py               # 模型定义（SiT-XL/2 + MeanFlow 修改）
│   └── splitmeanflow.py        # Split MeanFlow 算法核心实现
├── checkpoint/                 # 模型权重保存目录
└── sample/                     # 生成图片输出目录
```

## 引用

```bibtex
@article{guo2025splitmeanflow,
  title={SplitMeanFlow: Interval Splitting Consistency in Few-Step Generative Modeling},
  author={Guo, Y. and Wang, W. and Yuan, Z. and others},
  journal={arXiv preprint arXiv:2507.16884},
  year={2025}
}
```

该工作来自 **字节跳动（ByteDance）** 研究团队。
