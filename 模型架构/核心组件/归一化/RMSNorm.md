
RMSNorm 是当前大语言模型（LLM）的主流归一化方式（LLaMA/LLaMA2/Qwen/PaLM 均采用），核心特点是**只归一化数据的方差（RMS），不中心化均值**，
相比 LayerNorm 计算更高效、数值更稳定。

## 1. RMSNorm 核心定义与数学公式

### 核心思想

RMSNorm 抛弃了 LayerNorm 中 “减去均值（中心化）” 的步骤，仅通过**根均方（RMS）** 归一化数据的幅度（方差），保留数据的相对分布，同时通过可学习的缩放参数 $\gamma$恢复表达能力。

### 数学公式

对输入张量 $x \in R^d$（d 为特征维度），RMSNorm 计算如下：

$$RMSNorm(x)=\gamma \cdot \frac{x}{RMS(x)}$$

其中 RMS（根均方）的计算是核心:

$$RMS(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_{i}^{2}+\epsilon}$$

+ $\gamma in R^d$：可学习的逐维度缩放参数

+ $\epsilon$：极小值（如 1e - 6），避免 计算为 0；

### 与 LayerNorm 的核心对比

LayerNorm 的公式为：

$$LayerNorm(x)=\gamma \cdot \frac{x-μ}{\sqrt{\sigma^2+\epsilon}}+\beta$$

两者的关键差异用表格总结（也是大模型选择 RMSNorm 的原因）：

| 特性 | RMSNorm | LayerNorm |
|------|---------|-----------|
| 均值处理 | 不减去均值（无中心化） | 减去特征维度均值（中心化） |
| 可学习参数 | 仅缩放参数 $\gamma$ ( $d$ 个) | 缩放 $\gamma$ + 偏移 $\beta$ (2d个) | 
| 计算量 |	少 “减均值” 步骤，效率提升～10%	| 多一步均值计算，效率稍低 |
| 数值稳定性	| 无均值抵消，梯度更稳定	| 均值中心化可能导致数值抵消 |
| 大模型效果	| 与 LayerNorm 持平 / 更优	| 小模型效果略优，大模型无优势 |
