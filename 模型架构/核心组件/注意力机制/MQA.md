
## 1. MQA 是什么

**MQA（Multi-Query Attention）** 是对**Multi-Head Attention（MHA）** 的一种高效改进：**多个 Query 头，但 Key / Value 在所有头之间共享**

核心目的：显著降低 KV 缓存与内存带宽开销，特别适合大模型推理


## 2. 动机

### 2.1 MHA 的现实瓶颈（尤其是推理）

在自回归生成中（如 GPT）：

+ 每生成一个 token，都要缓存历史的 **K / V**

+ 对 MHA：

  + 每一层
  + 每一个 head
  + 都有独立的 K / V

KV Cache 大小（MHA）

$$KV size ∝ L\times H \times d_{head}$$
​
当：

+ 层数 L ↑

+ head 数 H ↑

+ 上下文长度 ↑

### 2.2 观察：真的需要对每个 head 都有独立的 K/V 吗？

研究发现：

+ 多个 head 的 K/V 表示高度相似

+ Query 的差异更重要

👉 那就共享 K/V，只保留多 Query

这就是 MQA。

## 4. 数学形式

MHA

$$head_i=Attention(QW_{i}^Q, KW_{i}^{K}, VW_{i}^V)$$

MQA

$$head_i=Attention(QW_{i}^Q, KW^K, VW^V)$$

+ $W_{i}^Q$：每个 head 不同

+ $W^k,W^V$：所有head 共享

## 优缺点

### ✅ 优点

1️⃣ 显著降低显存

+ KV Cache ↓ H 倍

+ 上下文更长

2️⃣ 推理更快

+ 内存带宽压力小

+ 更适合 batch decoding

3️⃣ 实现简单

+ 仅改投影方式

### ❌ 缺点

1️⃣ 表达能力略弱于 MHA

+ K/V 表达受限

+ 对复杂对齐任务可能不如 MHA

2️⃣ 训练阶段收益不明显

+ 优势主要在 推理阶段
