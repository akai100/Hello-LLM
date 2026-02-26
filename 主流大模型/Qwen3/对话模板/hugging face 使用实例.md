```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

messages = [{"role": "user", "content": "解释一下 attention 机制"}]

# 应用 chat template 并 tokenize
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,      # 返回 token IDs
    add_generation_prompt=True  # 自动加 assistant 开头（推理时必需）
)

input_ids
```

输出：

```
[151644,
 872,
 198,
 104136,
 100158,
 6529,
 220,
 100674,
 151645,
 198,
 151644,
 77091,
 198]
```
