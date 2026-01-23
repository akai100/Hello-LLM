## ViT 是什么

ViT（Vision Transformer）把图像切成 patch，当作“词序列”，用 Transformer 来做视觉建模。

## ViT 的整体流程

一张图像

→ Patch Embedding

→ 加位置编码

→ Transformer Encoder（多层）

→ [CLS] token 输出分类结果


## 核心步骤拆解

### 1️⃣ Patch Embedding

输入图像：

$$x \in R^{H \times W \times C}$$

切成大小为 $P \times P$ 的 patch:

$$N=\frac{HW}{P^2}$$

每个 patch 展平 + 线性映射：

$$x_p \in R^{P^2C} => D$$

👉 实现上通常用 Conv2d(kernel=P, stride=P)

### 2️⃣ 加 token & 位置编码

**[CLS] token**

+ 一个可学习向量

+ 用于分类任务

**Positional Embedding**

+ ViT 使用可学习位置编码

+ 必须有（Transformer 不感知顺序）

### 3️⃣ Transformer Encoder

每一层包括：

```
LayerNorm
Multi-Head Self Attention
Residual
MLP (FFN)
Residual
```

Attention 公式：

$$Attention(Q, K, V)=softmax(\frac{QK^T}{\sqrt{d}})V$$

### 4️⃣ 输出层

+ 分类：取 [CLS] token

+ 回归 / dense：用所有 token

## ViT 为什么能 work

**1️⃣ 全局建模能力强**

+ CNN：局部感受野

+ ViT：一层就能看全图

**2️⃣ Scaling 友好**

+ 参数量 ↑ → 性能 ↑

+ 和 NLP Transformer 类似

**3️⃣ 弱 inductive bias**

+ 不强行假设平移不变性

+ 大数据下更灵活

## ViT 的缺点

| 问题     | 原因                 |
| ------ | ------------------ |
| 数据依赖大  | inductive bias 少   |
| 计算复杂度高 | Attention (O(N^2)) |
| 小数据差   | 不如 CNN             |

## ViT vs CNN

| 维度   | CNN | ViT           |
| ---- | --- | ------------- |
| 建模   | 局部  | 全局            |
| Bias | 强   | 弱             |
| 数据需求 | 少   | 多             |
| 可解释性 | 一般  | Attention 可视化 |

## 经典改进 & 变种

**🔹 DeiT**

+ 知识蒸馏

+ 小数据训练 ViT

**🔹 Swin Transformer**

+ 窗口 Attention

+ 层级结构

**🔹 PVT**

+ 金字塔结构

+ 下采样 token

**🔹 ViT-H / ViT-G**

+ 大模型 Scaling
