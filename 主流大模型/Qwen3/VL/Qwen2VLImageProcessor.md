## 1. 定义

```python3
class Qwen2VLImageProcessor(BaseImageProcessor):
    ......
```

## 2. ```preprocess```

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

调用 ```fetch_images``` 将传入的图像数据转换成我们期望的格式，如果传入字符串。则加载图像成 PIL Image格式，字符串可以是路径或者网络图像。

调用 ```make_flat_list_of_images``` 将图像展平。

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

调用 ```validate_preprocess_arguments```` 校验参数

```python3
validate_preprocess_arguments(
    rescale_factor=rescale_factor,
    do_normalize=do_normalize,
    image_mean=image_mean,
    image_std=image_std,
    do_resize=do_resize,
    size=size,
    resample=resample,
)
```

调用 ```_preprocess``` 将每张图像转换成像素值，

```python3
if images is not None:
    pixel_values, vision_grid_thws = [], []
    for image in images:                            # 遍历每个图像
        patches, image_grid_thw = self._preprocess(
            image,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            data_format=data_format,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
        )
        pixel_values.extend(patches)
        vision_grid_thws.append(image_grid_thw)
    pixel_values = np.array(pixel_values)
    vision_grid_thws = np.array(vision_grid_thws
    data.update({"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws})
```

## 3. ```fetch_images```

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

## 4.  ```_preprocess```

**方法定义：**

```python3
def _preprocess(
    self,
    images: Union[ImageInput, VideoInput],
    do_resize: Optional[bool] = None,
    size: Optional[dict[str, int]] = None,
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
    data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
```

**转换成RGB,转换成 numpy array**

```python3
images = make_flat_list_of_images(images)

if do_convert_rgb:
    images = [convert_to_rgb(image) for image in images]

# All transformations expect numpy arrays.
images = [to_numpy_array(image) for image in images]
```


**将所有图像大小调整到第一张图像大小，并对图像做缩放和正则化**

```python3
height, width = get_image_size(images[0], channel_dim=input_data_format)
resized_height, resized_width = height, width
processed_images = []
for image in images:
    if do_resize:                                                # 如果设置了调整图像大小
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=size["shortest_edge"],
            max_pixels=size["longest_edge"],
        )
        image = resize(
            image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
        )

    if do_rescale:
        image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

    if do_normalize:
        image = self.normalize(
            image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
        )

    image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
    processed_images.append(image)
```
