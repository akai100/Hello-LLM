# Hello-LMM

## 目录

+ [模型架构](https://github.com/akai100/Hello-LLM/tree/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84)
  + [核心组件](https://github.com/akai100/Hello-LLM/tree/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6)
    + [位置编码](https://github.com/akai100/Hello-LLM/tree/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81)
      + [正余弦位置编码](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81/%E6%AD%A3%E4%BD%99%E5%BC%A6%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md)
      + [旋转位置编码](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81/%E6%97%8B%E8%BD%AC%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md)
    + [注意力机制](https://github.com/akai100/Hello-LLM/tree/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6)
      + [MHA](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/MHA.md)
      + [MQA](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/MQA.md)
      + [GQA](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/GQA.md)
    + [归一化](https://github.com/akai100/Hello-LLM/tree/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E5%BD%92%E4%B8%80%E5%8C%96)
      + [批归一化](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E5%BD%92%E4%B8%80%E5%8C%96/Batch%20Normalization%EF%BC%88%E6%89%B9%E5%BD%92%E4%B8%80%E5%8C%96%EF%BC%89.md)
      + [层归一化](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E5%BD%92%E4%B8%80%E5%8C%96/Layer%20Normalization%EF%BC%88%E5%B1%82%E5%BD%92%E4%B8%80%E5%8C%96%EF%BC%89.md)
      + [RMSNorm](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E5%BD%92%E4%B8%80%E5%8C%96/RMSNorm.md)
    + [激活函数](https://github.com/akai100/Hello-LLM/tree/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0)
      + [传统激活函数](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0/%E4%BC%A0%E7%BB%9F%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md)
      + [GELU](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0/GELU.md)
      + [GLU](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0/GLU.md)
      + [GeGLU](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0/GeGLU.md)
      + [HardSwish](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0/Hardswish.md)
      + [SwiGLU](https://github.com/akai100/Hello-LLM/blob/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0/SwiGLU.md)
  + [主流变体与优化](https://github.com/akai100/Hello-LLM/tree/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E4%B8%BB%E6%B5%81%E5%8F%98%E4%BD%93%E4%B8%8E%E4%BC%98%E5%8C%96)
    + [MoE](https://github.com/akai100/Hello-LLM/tree/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E4%B8%BB%E6%B5%81%E5%8F%98%E4%BD%93%E4%B8%8E%E4%BC%98%E5%8C%96/MoE)
    + [ViT](https://github.com/akai100/Hello-LLM/tree/main/%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E4%B8%BB%E6%B5%81%E5%8F%98%E4%BD%93%E4%B8%8E%E4%BC%98%E5%8C%96/ViT)
+ [主流大模型](https://github.com/akai100/Hello-LLM/tree/main/%E4%B8%BB%E6%B5%81%E5%A4%A7%E6%A8%A1%E5%9E%8B)
  + [CLIP](https://github.com/akai100/Hello-LLM/tree/main/%E4%B8%BB%E6%B5%81%E5%A4%A7%E6%A8%A1%E5%9E%8B/CLIP)
  + [Qwen3](https://github.com/akai100/Hello-LLM/tree/main/%E4%B8%BB%E6%B5%81%E5%A4%A7%E6%A8%A1%E5%9E%8B/Qwen3)

## 学习资料

https://github.com/mlabonne/llm-course
