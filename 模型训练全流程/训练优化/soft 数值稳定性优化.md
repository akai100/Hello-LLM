## softcap

softcap(softmax capping，Softmax 截断)，如果开启了 softcap，无法同时使用 dropout。

### 什么是 softcap

softcap 是一种 **Softmax 数值稳定性优化**，用于解决注意力计算中 QK^T 数值过大导致 Softmax 溢出（inf/nan）的问题，尤其在长序列、大模型训练中非常关键。

**1. 核心原理**

标准 Softmax：

$$softmax(x)_i=\frac{e^{x_i}}{\sum_{j}{e^{x_j}}}$$

如果 $x_i$ 很大（如 100），e^{x_i} 会直接溢出为 $inf$，导致训练崩溃。

Softmax Capping（softcap）：

在 Softmax 前：先将 $x_i$ 限制在 [-softcap, softcap] 范围内：

$$x_{i}^{clamped}=clamp(x_i, -softcap, softcap)$$

再对 $x_i^{\text{clamped}}$ 做 Softmax。

这样可以强制控制 $e^x_i$ 的大小，避免数值溢出，提升训练稳定性。

**2. 典型使用场景**

+ 长序列训练（如 128k+ tokens）；

+ 大模型微调（避免梯度爆炸 / NaN）；

+ 低精度训练（FP16/BF16 下数值范围更小，更容易溢出）。
