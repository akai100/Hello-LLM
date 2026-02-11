## 类声明

```python3
class Qwen3VLProcessor(ProcessorMixin):
```

## ```__init__```


## ```__call__```

```python3
def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature:
```

```images``` 类型：

```python3
ImageInput = Union[
    "PIL.Image.Image", np.ndarray, "torch.Tensor", list["PIL.Image.Image"], list[np.ndarray], list["torch.Tensor"]
]
```

```text``` 类型：

```python3

```

```python3
    output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
```

**处理图像：**

```python3
if images is not None:
    image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
    image_grid_thw = image_inputs["image_grid_thw"]
else:
    image_inputs = {}
    image_grid_thw = None
```

```self.image_processor` 目前使用```Qwen2VLImageProcessor```。


