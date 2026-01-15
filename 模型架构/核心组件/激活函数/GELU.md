## 1. GELU 是什么

GELU 用“概率”来决定一个神经元该被保留多少，而不是简单地阶段。

它不是：

```
x > 0 → 保留
x ≤ 0 → 丢弃
```

而是：

**x 在标准正态分布下有多大概率为正，就保留多少 x**

## 2. 数学定义

### 2.1 精确定义（理论版）

$$GELU(x)=x \cdot \Phi(x)$$

其中：

+ $\Phi(x)$是标准正太分布的 CDF

$$\Phi(x)=\int_{-\infty}^{x}\frac{1}{\sqrt{2\pi}}e^{-t^2/2}dt$$

直觉：

+ $x$ 很大 -> $\Phi(x) \approx 1$ -> 输出 $\approx x$
+ $x$ 很小 -> $\Phi(x) \approx 0$ -> 输出 $\approx 0$
+ $x$ 接近 0 -> 被“软过滤”

## 3. 工程中的近似形式

直接算 $\Phi(x)$太慢，所以实际用近似

### 3.1 tanh 近似（Transformer 原沦为）

$$GELU(x) \approx 0.5x(1+tanh(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)))$$

### 3.2 sigmoid 近似（更快）

$$GELU(x) \approx x \cdot \sigma(1.702x)$$

## 4. GELU 的函数形状

```
输入 x
│
├─ x < 0：不是直接归零，而是“缓慢压小”
├─ x ≈ 0：输出很小
└─ x > 0：近似线性
```

## 5. 为什么大模型都用 GELU / SwiGLU？

### 1️⃣ 梯度更稳定

没有 ReLU 的 “死神经元”

小负值仍然有梯度

### 2️⃣ 更符合语言建模分布

语言模型中：

token embedding 近似高斯分布

GELU 与输入分布假设匹配

📌 这点在 BERT 原论文里明确提到

### 3️⃣ 对 FFN 特别重要

Transformer 的 FFN：

```
x → Linear → GELU → Linear
```

FFN 参数量最大：

+ 决定了模型非线性能力

+ GELU 比 ReLU 表达更细腻

## 6. 面试

GELU 是一种基于高斯分布的平滑激活函数，通过输入在标准正态分布下为正的概率来缩放输入值。相比 ReLU，GELU 在负区间不会硬截断，梯度更平滑，在 Transformer 和大模型的 
FFN 中能提升训练稳定性和表达能力，因此被 BERT、GPT 等模型广泛采用。

**为什么不用 ReLU**

死神经元、硬阈值不适合语言分布
