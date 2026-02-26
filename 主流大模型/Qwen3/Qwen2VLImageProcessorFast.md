## ```Qwen2VLImageProcessorFast```

```python
class Qwen2VLImageProcessorFast(BaseImageProcessorFast):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
    do_rescale = True
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    min_pixels = None
    max_pixels = None
    valid_kwargs = Qwen2VLFastImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]
```

```Qwen2VLImageProcessorFast ``` 继承自transformer库的```BaseImageProcessorFast```

## ```_preprocess_image_like_inputs```

```python
def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        videos: VideoInput,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Optional[Union[str, "torch.device"]] = None,
        **kwargs: Unpack[DefaultFastImageProcessorKwargs],
    ) -> BatchFeature:
```
