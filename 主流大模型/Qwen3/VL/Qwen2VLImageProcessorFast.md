## Qwen2VLImageProcessorFast 声明

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

## ```preprocess```

```python
def preprocess(
    self,
    images: ImageInput,
    videos: Optional[VideoInput] = None,
    **kwargs: Unpack[Qwen2VLFastImageProcessorKwargs],
) -> BatchFeature:
    return super().preprocess(images, videos, **kwargs)
```

实际调用父类```BaseImageProcessorFast``` 的 ```preprocess```方法。


