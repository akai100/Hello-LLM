```python3
class Qwen3VLModel(Qwen3VLPreTrainedModel):
    ......
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)      # 视觉模型
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)  # 大语言模型
```

## ```forward```

```python3
def forward(
        self,
        input_ids: torch.LongTensor = None,                        # 文本输入的 token IDS
        attention_mask: Optional[torch.Tensor] = None,             # 指示哪些 token 是有效（非 padding）的
        position_ids: Optional[torch.LongTensor] = None,           # 每个token 的绝对位置所有（用于 RoPE）
        past_key_values: Optional[Cache] = None,                   # KV Cache，用于自回归生成加速（避免重复计算历史 token 的 K/V）
        inputs_embeds: Optional[torch.FloatTensor] = None,         # 直接提供 embedding 向量，跳过 token lookup
        pixel_values: Optional[torch.Tensor] = None,               # 输入的图像像素值（归一化到 [-1, 1] 或 [0, 1]）
        pixel_values_videos: Optional[torch.FloatTensor] = None,   # 输入的视频帧像素值。
        image_grid_thw: Optional[torch.LongTensor] = None,         # 指定每张图像被划分成多少 patches（即 token 数量的结构信息）
        video_grid_thw: Optional[torch.LongTensor] = None,         # 视频版本的 grid 信息
        cache_position: Optional[torch.LongTensor] = None,         # 在生成过程中，指示当前新 token 在完整序列中的绝对位置
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
```

**图像处理**

```python
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

调用 ```get_image_feature``` 接口，将图像交给视觉模型处理

## ```get_image_features```

```python3
def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
    pixel_values = pixel_values.type(self.visual.dtype)
    image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
    split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    image_embeds = torch.split(image_embeds, split_sizes)
    return image_embeds, deepstack_image_embeds
```
