```python3
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

对应公式：

$$ \frac{X}{\sqrt{Mean(x^2) + \epsilon}}$$

与通用 RMSNorm 的差异：

1. eps 添加时机：在计算平方和除以维度后、开平方前加ϵ（而非开平方后），避免极小值开平方导致的数值下溢（适配 Qwen3 的 bfloat16/float16 混合精度训练）

2. 维度对齐：严格适配 Qwen3 的张量形状（[batch_size, seq_len, hidden_size] 或 [batch_size, hidden_size, seq_len]），支持 transpose 维度的快速归一化；

3. 无分组默认版：Qwen3 基础版（7B/14B）用标准单维度 RMSNorm，超大版（72B+）可选分组 RMSNorm（Grouped RMSNorm）进一步降低计算量；

4. 算子融合：推理阶段将 “平方和计算 + 开方 + 归一化 + 缩放” 融合为单个 CUDA 算子，提升推理速度（Qwen3-TRT/TF 框架优化）。
