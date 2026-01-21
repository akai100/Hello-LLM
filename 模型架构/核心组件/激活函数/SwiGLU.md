
## 1. SwiGLU 是什么

SwiGLU 是一种 神经网络激活函数，它结合了：

+ Swish 激活函数（一种平滑、非单调的激活机制）

+ GLU（Gated Linear Unit）门控结构

相比传统激活（如 ReLU/GELU），SwiGLU 使用一个门控机制来动态控制信息流，从而增强模型的表达能力和训练效果。

## 2. 数学定义与原理

### 2.1 GLU 的基本思想

Gated Linear Unit 最初定义如下：

$$GLU(xW, xV)=(xW)\odot \sigma(xV)$$

其中：

+ $xW$ 是一条线性变换
+ $xV$ 经过 sigmoid 作为门控
+ $\odot$ 表示逐元素相乘
+ $\sigma$ 是 sigmoid 函数

该结构是让部分特征决定另一部分是否通过。

### 2.2 Swish 激活函数

Swish 定义为：

$$Swish(z)=z \cdot \sigma(\beta z)$$

通常 $\beta = 1$，使得 Swish 和 Sigmoid Linear Unit（SiLU）等价。Swish 的特点是：

+ 平滑、可导 → 梯度更稳定

+ 对负输入不完全抑制 → 能更灵活表达信息

### 2.3 SwiGLU 的结构

SwiGLU 的主要操作：

$$SwiGLU(x)=Swish(xW_g) \odot (xW_v)$$

即：

1. 通过两个线性映射分别产生“gate” (门）和 “value”（值）路径。

2. 对 gate 路径使用 Swish 激活

3. 用激活后的 gate 对 value 路径进行逐元素乘积

这种乘积形式让激活可以动态根据输入调制输出，而不仅仅是简单非线性变换


## 3. SwiGLU 优势

**🌟 更强的表达能力**

门控结构允许网络根据不同输入选择性放大或抑制信息，从而比单一激活函数（如 ReLU、GELU）更灵活。

**🔁 更好的梯度传播**

Swish 本身具有平滑梯度，有助于缓解传统激活中的梯度消失/爆炸问题，配合门控能够加速学习和提升稳定性。

**🧠 适合大规模模型**

SwiGLU 替代标准 FFN（前馈网络）中的 GELU 等激活，在 Transformer 和大语言模型中普遍提高性能，并成为许多现代模型的标准选择。

## 4. 与其他激活函数对比

| 激活函数           | 结构特性       | 代表模型                  |
| -------------- | ---------- | --------------------- |
| **ReLU**       | 简单、稀疏激活    | 经典 CNN、初期 Transformer |
| **GELU**       | 平滑非线性      | BERT、GPT-2            |
| **Swish/SiLU** | 平滑、非单调     | 中间激活尝试者               |
| **SwiGLU**     | 门控 + Swish | 现代 LLM FFN 标配         |
