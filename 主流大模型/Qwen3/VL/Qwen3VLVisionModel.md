
## ```__init__```

```python3
def __init__(self, config, *inputs, **kwargs) -> None:
    super().__init__(config, *inputs, **kwargs)
    self.spatial_merge_size = config.spatial_merge_size
    self.patch_size = config.patch_size
    self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

    self.patch_embed = Qwen3VLVisionPatchEmbed(
        config=config,
    )
```

## ```forward```

```python3
def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
    hidden_states = self.patch_embed(hidden_states)    # 调用 patch embedding 将输入图像转换为视觉 token 序列
```
