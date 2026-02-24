
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

## ```rot_pos_emb```

功能：为 Transformer 生成 RoPE 位置嵌入。

```python
def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
```

+ 输出

  + grid_thw

    形状为 ```(B, 3)`` 的张量，表示每个样本的 ```[T, H, W]```（帧数、高度、宽度）

+ 输出

  形状为 ```(total_tokens, dim)``` 的 RoPE 位置嵌入，其中 ```dim``` 是旋转嵌入的维度

```python
    merge_size = self.spatial_merge_size
```

+ 获取空间合并因子（例如 2），表示每 merge_size × merge_size 个原始像素被合并成一个 token

+ 这常见于 ViT 中的 patch embedding（如 14x14 图像 → 7x7 tokens，若 merge_size=2）

```python
    max_hw = int(grid_thw[:, 1:].max().item())
    freq_table = self.rotary_pos_emb(max_hw)
```

+ ```grid_thw[:, 1:]``` 取所有样本的 ```[H, W]```，找最大值（即最大高/宽）

+ 调用 ```self.rotary_pos_emb(max_hw)``` 生成一个预计算的 RoPE 频率表，形状为 (max_hw, dim // 2)

```python
    device = freq_table.device
```

+ 获取设备（CPU/GPU），后续创建张量时保持一致

```python
    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
```

+ 计算所有样本的总 token 数：对每个样本 ```T * H * W``` 求和

+ 创建一个空的 ```pos_ids``` 张量，用于存储每个 token 的 ```(row, col)``` 坐标（2D 位置）

```python
    offset = 0
```

+ 用于在 ```pos_ids``` 中按样本顺序写入坐标的偏移指针

```python
    for num_frames, height, width in grid_thw:
```

+ 对 batch 中每个视频/图像样本，解包其```(T, H, W)```

```python
        merged_h, merged_w = height // merge_size, width // merge_size
```

+ 计算合并后的网格大小（即 token 网格尺寸）

+ 例如：原始图像 224x224，merge_size=16 → 14x14 tokens

```python
        block_rows = torch.arange(merged_h, device=device)  # block row indices
        block_cols = torch.arange(merged_w, device=device)  # block col indices
        intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
        intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets
```

+ ```block_rows/cols```：每个 token 块在合并网格中的行列索引

+ ```intra_row/col```：每个块内部的像素偏移（0 到 merge_size-1）

```python
        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
```

通过广播机制，生成所有原始像素位置（未合并前）的坐标

+ ```block_rows[:, None, None, None]``` 形状 ```(merged_h, 1, 1, 1)```

+ ```intra_row[None, None, :, None]``` 形状 ```(1, 1, merge_size, 1)```

+ 相加后得到 ```(merged_h, merged_w, merge_size, merge_size)``` 的 ```row_idx```，表示每个 token 对应的原始行坐标

+ 同理 ```col_idx``` 是列坐标
