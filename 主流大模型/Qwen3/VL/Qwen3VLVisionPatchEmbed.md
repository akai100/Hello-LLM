
专为多模态大模型（支持图像 + 视频）设计的 3D Patch Embedding 模块，它使用 nn.Conv3d 来统一处理 静态图像和动态视频 的 tokenization。这是 Qwen3-VL 支持 视频理解 的关键组件。

## ```__init__```

```python3
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size                      # 空间 patch 尺寸，如 14 (H/W)
        self.temporal_patch_size = config.temporal_patch_size    # 时间 patch 尺寸，如 2 (帧数)
        self.in_channels = config.in_channels                    # 输入通道数，通常为 3 (RGB)
        self.embed_dim = config.hidden_size                      # 输出 token 维度，如 1024

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)
```

## ```forward```

```python3
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states
````
