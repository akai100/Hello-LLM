标准 PyTorch 实现

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
