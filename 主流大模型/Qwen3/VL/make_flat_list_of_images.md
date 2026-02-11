## 定义

```python3
def make_flat_list_of_images(
    images: Union[list[ImageInput], ImageInput],
    expected_ndims: int = 3,
) -> ImageInput:
    
```

如果输入的是嵌套list 图像，则 flat 一个列表

```python3
    if (
        isinstance(images, (list, tuple))
        and all(isinstance(images_i, (list, tuple)) for images_i in images)
        and all(is_valid_list_of_images(images_i) or not images_i for images_i in images)
    ):
        return [img for img_list in images for img in img_list]
```

如果输入的是一个list图像，即list中元素是图像数据：

```python3
    if isinstance(images, (list, tuple)) and is_valid_list_of_images(images):
        if is_pil_image(images[0]) or images[0].ndim == expected_ndims:    # 如果是PIL Image 图像或者维度是我们期望的维度（默认为3维），则直接返回图像
            return images
        if images[0].ndim == expected_ndims + 1:                           # 或者是我们期望的维度 + 1
            return [img for img_list in images for img in img_list]
```

如果已经是图像数据：

```python3
    if is_valid_image(images):
        if is_pil_image(images) or images.ndim == expected_ndims:
            return [images]
        if images.ndim == expected_ndims + 1:
            return list(images)
```

