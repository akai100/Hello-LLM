
## 1. ```__init__```

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

## 2. ```forward```

（1）调用 patch embedding 将输入图像转换为视觉 token 序列

```python3
def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
    hidden_states = self.patch_embed(hidden_states)    # 调用 patch embedding 将输入图像转换为视觉 token 序列
```

（2） 调用```fast_pos_embed_interpolate```接口实现

```python3

```

## 3. ```fast_pos_embed_interpolate```

为任意尺寸的输入图像/视频，在固定大小的位置编码表上，通过双线性插值生成高精度的位置嵌入，并适配时间维度和空间合并策略，输出与视觉 token 完全对齐的位置编码。

```python3
    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]     # 解析输入网格尺寸
```

（2）为双线性插值的 4 个邻近格点（左上、右上、左下、右下）分别存储：

+ ```idx_list[i]```: 在位置编码表中的索引

+ ```weight_list[i]```: 对应的插值权重

```python3
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
```

（3）对每个样本进行插值计算

```python3
        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
```


（4）构建目标网格在 “规范坐标系”下的浮点索引

+ 假设位置编码表是 ```num_grid_per_side × num_grid_per_side``` 的网格（例如 32×32）

+ 现在要生成 ```h × w``` 个位置编码，于是将 ```[0, num_grid_per_side-1]``` 均匀映射到 ```h``` 和 ```w``` 个点

+ 这相当于把目标尺寸“对齐”到原位置编码网格的坐标范围

```python3
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
```

（5）计算 floor / ceil 索引（四个角点）

+ 得到每个浮点坐标的整数邻居（向下取整和向上取整），并防止越界

```python3
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
```
（6）计算插值权重（距离比例）

+ ```dh```, ```dw``` 表示在 h/w 方向上的小数部分，用于计算双线性权重。

```python3
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

```

（7）构造 2D 网格的线性索引（flatten 后在一维表中查找）

+ 将 2D 坐标 ```(i, j)``` 转换为一维索引：```i * num_grid_per_side + j```

+ 使用广播（```.T``` 和 ```[None]```）构建 ```h × w``` 网格的所有组合，再 flatten 成一维。

```python3
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
```

（8）对应的双线性权重

+ 标准双线性插值的四个权重。

```python3
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
```

（9）存储所有样本的索引和权重

```python3

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())
```

（10）批量查表 + 加权求和

+ ```self.pos_embed``` 是一个 ```nn.Embedding(num_embeddings, embed_dim)```，其中 ```num_embeddings = num_grid_per_side^2```

+ 通过索引查出 4 个邻近位置的 embedding，乘以对应权重后相加 → 得到插值后的位置编码

```python
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
```

（11）按样本拆分（恢复 batch 结构）

```python3
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])
```

（12） 适配时间维度 + 空间合并（spatial merging）

关键说明：

+ ```spatial_merge_size```：比如 ```merge_size=2```，表示每 2×2 个原始 patch 被视为一个“合并单元”。

+ 原始位置编码是按 h × w 生成的。

+ 但模型实际输入的 token 是经过 空间合并 的（token 数减少为 (h/merge) × (w/merge)，但每个 token 可能由多个 patch 融合而来）。

+ 此处的 view + permute + flatten 实际上是在 保持原始位置编码粒度的前提下，调整排列顺序，以便后续与视觉 token 对齐。

```python3
        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
```

（13）合并所有样本并返回

```python3
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds
```
