
## 1. 什么是 MHA?

**多头注意力（Multi-Head Attention）** 是 Transformer 的核心机制，它通过**并行的多个注意力头**，让模型在**不同子空间**、**不同位置关系**上同时关注信息，从而更好地建模序列依赖关系。

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
