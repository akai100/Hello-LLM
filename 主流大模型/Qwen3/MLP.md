```class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

Qwen-3 的 MLP 层实际上由三个线性层（通常命名为 ```gate_proj```, ```up_proj```, 和 ```down_proj```）组成。

其数学表达式如下：

$$MLP(x) = (SiLU(W_{gate}x) \otimes (W_{up}x))W_{down}$$

其中：

+ $W_{gate}$ (Gate Projection)：生成门控向量；
+ $W_{up}$ (Up Projection)：提升特征维度；
+ $SiLU$ (激活函数)：即上一条回答中提到的 Sigmoid Linear Unit；
+ $\otimes$ (Hadamard Product)：逐元素乘法；
+ $W_{down}$ (Down Projection)：将维度投影回原始大小（ $d_{model}$）；


我们可以将 Qwen-3 的 MLP 拆解为以下三个步骤：

**1. 分叉映射 (Splitting)**：输入向量 $x$ 同时进入两个线性层。一个负责提供“内容”（ $W_{up}$），另一个经过 SiLU 激活后负责提供“门控信号”（ $W_{gate}$）。

**2. 门控交互 (Gating)**：将激活后的信号与内容向量进行逐元素相乘。这一步的作用是让模型能够自适应地决定哪些特征维度应该被加强，哪些应该被抑制。

**3. 维度还原 (Down-projection)**：经过高维空间的特征融合后，通过 $W_{down}$ 将中间层的维度（通常是 $d_{model}$ 的 3.5 到 4 倍左右）重新压缩回隐藏层维度，以便与残差连接（Residual Connection）相加。

## 3. 为什么不直接使用“线性 -> 激活 -> 线性”？

Qwen-3 采用这种结构主要有以下三个考量：

+ **非线性增强**：SwiGLU 结构相比普通 MLP 提供了更丰富的非线性变换。

+ **信息筛选**：门控机制允许网络根据上下文动态调整特征流向。

+ **计算效率**：虽然参数量比普通两层 MLP 增加了约 50%（增加了一个线性层），但在同等计算量（FLOPs）下，SwiGLU 的收敛效果通常好于通过增加深度带来的提升。
