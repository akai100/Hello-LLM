
## 1. 什么是 MHA?

**多头注意力（Multi-Head Attention）** 是 Transformer 的核心机制，它通过**并行的多个注意力头**，让模型在**不同子空间**、**不同位置关系**上同时关注信息，从而更好地建模序列依赖关系。

## 2. 工程流程

### 第一步：线性投影

输入通常是一个词向量序列（Embedding）。对于每个“头” $i$，我们都有独立的一组权重矩阵： $W_{i}^{Q}, W_{i}^{K}, W_{i}^{V}$。

我们将输入 $X$分别乘以这些矩阵，得到该头独有的**Query(查询)、key(键)和Value(值)** 向量。

$$Q_i=XW_{i}^Q, K_i=XW_{i}^K, V_i=XW_{i}^V$$

### 第二步：缩放点积注意力

这是注意力的核心计算。每个头独立计算注意力分数。公式如下：

$$Attention(Q_i, K_i, V_i)=softmax(\frac{Q_iK_{i}^T}{\sqrt{d_k}})V_i$$

+ $Q_iK_{i}^T$：计算查询和键的相似度

+ $\sqrt{d_k}$：缩放因子。因为点积结果可能很大，导致softmax 梯度极小（梯度消失），除以维度的平方根可以稳定数值。

+ Softmax：将分数归一化为概率分布（权重和为 1）

+ $\times V_i$：根据权重对 Value 进行加权求和，提取重要信息。

### 第三步：拼接

假设我们有 $h$ 个头，现在我们得到了 $h$ 个输出向量。我们将这些向量在特征维度上拼接起来。

$$\text{Concat}(head_1, head_2, \dots, head_h)$$

### 第四步：最终线性变换 (Final Linear Transformation)

拼接后的向量维度通常比较长，我们需要通过另一个权重矩阵 $W^O$ 将其映射回模型所需的标准维度。

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, \dots, head_h) W^O$$

## 3. 数学形式

$$MultiHead(Q, K, V)=Concat(head_1, ..., head_h)W^O$$
$$where head_i=Attention(QW_{i}^Q, KW_{i}^K, VW_{i}^{V}$$

其中：

+ $W_{i}^{Q}\in R^{d_{model}\times d_k}$

+ $W_{i}^K \in R^{d_{model}\times d_k}$

+ $W_{i}^{V} \in R^{d_{model} \times d_v}$

+ $W_O \in R^{hd_v\times d_{model}}$

通常，$d_k=d_v=d_{model}/h$ 

## 4. 核心优势

+ 子空间表示学习

  不同的头可以将输入映射到不同的子空间，捕捉序列中多种不同类型的依赖关系（如长距离依赖、局部依赖、语法依赖等）

+ 并行计算

  这是相比于 RNN（循环神经网络）最大的优势。RNN 必须按顺序读取单词，而 Attention 机制可以一次性并行处理整个序列的所有 token。

+ 长距离依赖

  无论两个词在句子中相隔多远，Attention 机制都能直接计算它们之间的点积，距离恒为 1。这解决了 RNN 处理长序列时容易“遗忘”的问题。
