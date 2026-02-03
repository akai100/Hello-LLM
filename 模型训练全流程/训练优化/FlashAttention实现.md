## 

FlashAttention 是由 Tri Dao 等人开发的高效注意力机制实现，其官方 GitHub 仓库是：🔗 https://github.com/Dao-AILab/flash-attention

### 目录结构

```
flash-attention/
├── csrc/                  # CUDA/C++ 核心实现
│   └── flash_attn/
│       ├── src/
│       │   ├── flash_fwd_kernel.h / .cu     ← 前向核心
│       │   ├── flash_bwd_kernel.h / .cu     ← 反向核心
│       │   └── ... 
│       └── ...
├── flash_attn/            # Python 接口
│   ├── __init__.py
│   ├── flash_attn_interface.py   ← 主要调用入口
│   └── ...
├── setup.py               # 安装脚本（旧版）
├── pyproject.toml         # 新版构建配置（使用 triton + ninja）
└── tests/                 # 测试用例
```

## Python 绑定层实现

### ```flash_attn_func``` 接口

```python3
def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )
```

### ```FlashAttnFunc```

###

