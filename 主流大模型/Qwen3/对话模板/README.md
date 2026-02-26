Chat Template（对话模板） 是大语言模型（LLM）在处理多轮对话时，用于将结构化的对话历史（如用户、助手、系统消息）转换为模型可理解的文本格式的一种标准化方法。

## 核心目的

+ 统一输入格式：确保不同来源的对话数据（如 API 请求、微调数据集）以一致方式喂给模型

+ 区分角色：明确标识每条消息的发送者（user / assistant / system），避免模型混淆上下文

+ 支持多轮对话：正确保留对话历史，使模型能进行上下文感知的回复

+ 兼容训练与推理：同一模板可用于 SFT（监督微调）、推理生成等阶段


## Qwen3 对话模板

```jinja
{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
<|im_start|>assistant
```

输入：

```python
messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "今天天气如何？"}
]
```

输出（模型实际看到的 token 序列前的文本）：

```python
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！<|im_end|>
<|im_start|>user
今天天气如何？<|im_end|>
<|im_start|>assistant
```

模型从 <|im_start|>assistant 之后开始生成回复
