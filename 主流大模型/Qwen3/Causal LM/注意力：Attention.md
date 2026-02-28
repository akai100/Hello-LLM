```python3
class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

```

## 2. ```forward```

```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

### 2.1 输入形状准备

```python
input_shape = hidden_states.shape[:-1]            # (B, S)
hidden_shape = (*input_shape, -1, self.head_dim)  # (B, S, num_heads, head_dim)
```

+ ```hidden_states```: ```(B, S, d_model)```
+ ```hidden_shape``` 用于后续 ```.view()``` 拆分为多头：
  (B, S, d_model) → (B, S, num_heads, head_dim)

### 2.2 Q/K/V 投影 + 归一化

```python
query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
```

+ ```self.q_proj```

  线性层变换，$X^{B \times S \times HiddenSize} \times Q^{Hiddensize \times \HiddenSize} + b(如果有)$

+ ```.view()```

  将经过Q映射后的形状转换为 ```(B, S, H, D)```

+ ```.transpose(1, 2)``` → (B, H, S, D)

  变为后续 attention 计算：```(B, H, S, D) x (B, H, D, S_k)```

### 2.3 应用旋转位置编码

```python
cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

### 2.4 KV Cache 更新

```python
if past_key_values is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
```

### 2.5 动态选择注意力实现后端

```python
if past_key_values is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
```

### 2.6 调用注意力内核

```python
attn_output, attn_weights = attention_interface(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    dropout=0.0 if not self.training else self.attention_dropout,
    scaling=self.scaling,
    sliding_window=self.sliding_window,  # ← Qwen3/Mistral 特有！
    **kwargs,
)
```

### 2.7 输出投影

```python
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = self.o_proj(attn_output)
```

关键创新： QK-Norm

+ 在计算注意力前对 Q 和 K 做归一化：
  + 提升训练稳定性
  + 缓解梯度爆炸（尤其在 deep / wide 模型中）
  + 注意力分数分布更平滑
