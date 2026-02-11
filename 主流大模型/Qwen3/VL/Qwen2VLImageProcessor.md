## 定义

```python3
class Qwen2VLImageProcessor(BaseImageProcessor):
    ......
```

## ```preprocess``

**定义：**

```python3
def preprocess(
        self,
        images: ImageInput,
        videos: Optional[VideoInput] = None,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
```


```python3

if images is not None:
    images = self.fetch_images(images)
    images = make_flat_list_of_images(images)

if images is not None and not valid_images(images):
    raise ValueError(
        "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
        "torch.Tensor, tf.Tensor or jax.ndarray."
    )
```

## ```fetch_images```

将传入的图像转换成我们期望的图像格式。

```
def fetch_images(self, image_url_or_urls: Union[str, list[str], list[list[str]]]):
    ......
```

```python3
    if isinstance(image_url_or_urls, list):    # 如果传入的image_url_or_urls 是列表，则遍历
        return [self.fetch_images(x) for x in image_url_or_urls]
```

如果传入的是字符串，则加载图像,转换成 PIL Image

```python3
    elif isinstance(image_url_or_urls, str):
        return load_image(image_url_or_urls)
```

如果已经使解析后的图像：PIL Image 或者 numpy array 或者 torch tensor：

```python3
  elif is_valid_image(image_url_or_urls):
      return image_url_or_urls
```

否则，抛出类型异常：

```python3
    else:
        raise TypeError(f"only a single or a list of entries is supported but got type={type(image_url_or_urls)}")
```


