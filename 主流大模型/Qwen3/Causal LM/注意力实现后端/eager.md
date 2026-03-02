该函数实现了一种带有 分页 KV 缓存（paged KV cache）、组查询注意力（GQA）、滑动窗口注意力（sliding window） 和 attention sinks（注意力汇点） 的自定义注意力前向传播逻辑，常用于大语言模型推理优化。

## ```eager_paged_attention_forward```

```python
def eager_paged_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],  # shape [seqlen_q, seqlen_k]
    scaling: float,
    **kwargs,
):
```

+ ```module```

  当前注意力层的 PyTorch 模块（如 LlamaAttention），包含配置信息（如 layer_idx, num_key_value_groups, sliding_window, sinks 等）;

+ ```query, key, value```: 输入的 Q/K/V 张量，通常形状为 ```[batch_size=1, num_heads, seqlen, head_dim]```;

+ ```attention_mask```: 注意力掩码，用于控制哪些 token 可以被关注。可能是普通 tensor，也可能是字典（支持不同注意力类型）;

+ ```scaling```: 缩放因子，通常是 ```1 / sqrt(head_dim)```，用于缩放点积结果以稳定 softmax

### 1. 集成分页 KV Cache

```python
cache = kwargs.pop("cache", None)
if cache is not None:
    key, value = cache.update(key, value, module.layer_idx, **kwargs)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
```

+ cache 是一个 分页缓存对象（类似 vLLM 的 PagedKVCache）

+ 它将物理不连续的 KV 块（page）逻辑上拼接成完整序列

+ ```cache.update()``` 返回 已拼接的历史 + 当前 K/V

+ 后续 ```.transpose().unsqueeze()``` 是为了适配 ```(B=1, H, S, D)``` 标准形状

### 2. 支持 GQA

```python
if hasattr(module, "num_key_value_groups"):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
```

+ Qwen3 / LLaMA-2 使用 GQA：32 个 query head，8 个 key/value head

+ 每个 K/V head 被 重复 4 次 以匹配 Q head 数量

### 3. 动态选择因果掩码（支持滑动窗口）

```python
if isinstance(attention_mask, dict):
    sliding_window = getattr(module, "sliding_window", 1)
    layer_type = "full_attention" if sliding_window == 1 or sliding_window is None else "sliding_attention"
    causal_mask = attention_mask[layer_type]
else:
    causal_mask = attention_mask
```

+ 上层传入的 attention_mask 可能是 字典，包含两种掩码：
  + "full_attention"：标准下三角因果掩码
  + "sliding_attention"：带滑动窗口的稀疏掩码（如只看最近 4096 个 token）
+ 根据当前层是否启用 sliding_window 动态选择

### 4. 计算注意力分数 + 应用掩码

```python
attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
if causal_mask is not None:
    attn_weights = attn_weights + causal_mask
```

+ scaling：通常是 1 / sqrt(head_dim)，但可能被 YaRN 等方法调整

+ causal_mask：值为 -inf 或大负数，确保 softmax 后无效位置为 0

### 5. 处理 Attention Sinks （注意力汇点）

attention sinks（一种用于长上下文推理的技术，保留最早几个 token 的注意力权重，防止信息丢失）

```python
if hasattr(module, "sinks"):
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    attn_weights = torch.cat([attn_weights, sinks], dim=-1)
    attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = attn_weights[..., :-1]
else:
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    
```

+ 在```attn_weights``` 的最后一维（即 key 序列长度维度）追加 sink 偏置，相当于新增一个“虚拟 key”。

+ 现在 attn_weights 的形状变为 [..., seqlen_k + num_sinks];

+ 数值稳定化：减去每行最大值，防止 softmax 溢出（log-sum-exp 技巧）;

+ 在 float32 下计算 softmax（更精确），再转回原始 dtype（如 bfloat16）

+ 丢弃 sink 对应的注意力权重（因为我们只是用它来影响分布，但不希望输出包含 sink 的 value）

+ 如果没有 sinks，正常 softmax
