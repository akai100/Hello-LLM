Qwen3 采用 旋转位置编码，实现如下：

## 计算旋转参数

```python3
def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
) -> tuple["torch.Tensor", float]:

    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor

```

```torch.arange(0, dim,  2, dtype=torch.int64)``` 实现位置信息： $[0, 2, 4, ..., dim]$

```inv_freq``` 则是计算的角速度： $w_i=\frac{1}{base^{2i/d_{model}}}$

## ```Qwen3RotaryEmbedding```

```python3
class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

```Qwen3RotaryEmbedding```主要功能为实现位置和角速度的乘积，然后计算 cos 和 sin。

```inv_freq_expanded``` 扩展成 $R^{B \times d_{dim} \times 1}$

```position_ids_expanded```扩展成 $R^{B \times 1 \times SEQ_LEN}$


$$ cos = 
\begin{pmatrix}
W0 p0  & w_0 p1 & ... & w_0 p_{SEQ_LEN}\\
w_1 p0 & w_1 p2 & ... & w_1 p_{SEQ_LEN}\\
... & ... & ...\\
w_{d_{dim}/2} p0 & w_{d_{dim}/2} p_1& ... & w_{d_{dim}/2} p_{SEQ_LEM}
\end{pmatrix}
$$

### 强制在 float32 下计算

```python
with torch.autocast(device_type=device_type, enabled=False):
```

+ 在 混合精度训练（AMP） 或 bfloat16 推理 中，x.dtype 可能是 bfloat16 或 float16

+ 但 $cos(mθ) / sin(m\theta)$ 在长序列下对数值精度敏感：
  + float16 的精度不足 → 高频部分（大 m）三角函数值失真
  + 导致 RoPE 失效，注意力混乱

### 计算频率

```
freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
```

+ 矩阵乘：```(B, d//2, 1) @ (B, 1, S)``` → (B, d//2, S)```
+ ```.transpose(1, 2)``` → ```(B, S, d//2)```

### 构造完整的 embedding(偶数+基数)

```python
emb = torch.cat((freqs, freqs), dim=-1)  # (B, S, d)
```

+ 将 ```(B, S, d//2)``` 拼接成 ```(B, S, d)```


### 应用缩放因子

```python
cos = emb.cos() * self.attention_scaling
sin = emb.sin() * self.attention_scaling
```

## 与传统 RoPE 缓存方式的区别

| 方式 | 传统 RoPE（LLaMA 原始） | 你提供的动态 RoPE（Qwen3 风格） |
|------|------------------------|-------------------------------|
| 缓存策略 | 预计算 `max_seq_len` 的 cos/sin | 每次 forward 动态计算 |
| 内存占用 | 高（需存 2 × max_len × d） | 低（按需计算） |
| 灵活性 | 固定长度 | 支持任意 `position_ids`（如非连续、滑动窗口） |
| 适用场景 | 标准自回归生成 | 长上下文、稀疏注意力、工具调用等复杂场景 |


## 应用到Q、K

```python3
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
```

