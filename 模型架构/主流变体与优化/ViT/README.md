## 整体流程

```mermaid
graph LR
    A("输入图像")
    B("图像分块")
    C("线性嵌入")
    D("添加位置编码与[CLS]")
    E("Transformer编码器")
    F("分类头")
    G("输出类别概率")
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
```

## 图像分块与嵌入

+ 分块操作

  将 H×W×C 的图像分割为 N 个 P×P×C 的 patch，其中 N = (H/P) × (W/P)

+ 线性投影

  每个 patch 通过单层全连接层，将维度从 P²C 映射到 D（模型隐藏维度）

  ```
  z₀ = [x_class; x₁^E; x₂^E; ...; x_N^E] + E_pos
  ```
  其中：
  + x_class：可学习的分类 token 嵌入
  + x_i^E：第 i 个 patch 的线性嵌入
  + E_pos：位置嵌入矩阵

## Transformer 编码器

ViT 使用标准 Transformer 编码器（BERT 风格），由 L 层相同的 Transformer Block 堆叠而成：
