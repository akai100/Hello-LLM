```python3
class Qwen3VLModel(Qwen3VLPreTrainedModel):
    ......
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)      # 视觉模型
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)  # 大语言模型
```

## ```forward```



