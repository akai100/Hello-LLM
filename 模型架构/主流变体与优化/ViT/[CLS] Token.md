## 设计动机

ViT 的输入是多个 patch token 的序列，Transformer 编码器输出的是每个 token 的特征向量。但图像分类任务需要一个单一的全局特征向量来预测类别，而不是多个 patch 的特征。

[CLS]（Classification）Token 就是为解决这个问题设计的：它是一个特殊的、可学习的 token，在序列最开头加入，经过所有 Transformer 层的自注意力计算后，
会聚合所有 patch 的全局信息，最终用它的输出作为分类的依据（类比 BERT 在 NLP 分类任务中的 [CLS] Token）。

## 实现方式

1. 初始化一个可学习的向量cls_token，形状为(1, D)（D 与 patch embedding 维度一致）；

2. 将cls_token拼接在所有 patch embeddings 的最前面，形成完整的输入序列：[cls_token; patch_1; patch_2; ...; patch_N]；

3. 这个序列与位置编码相加后，输入 Transformer 编码器；

4. 经过所有层的自注意力后，只取cls_token对应的输出向量，送入分类头（线性层）得到类别概率

## 关键特点

+ 全局信息融合

  [CLS] Token 在每一层都会与所有 patch token 进行自注意力交互，因此它能逐步聚合整个图像的全局信息，而不是局部信息；

+ 对比均值池化

  ViT 论文中对比了 “用 [CLS] Token” 和 “对所有 patch token 做均值池化” 两种方式，结果显示 [CLS] Token 的分类效果更好；

+ 仅用于分类任务

  ViT 论文中对比了 “用 [CLS] Token” 和 “对所有 patch token 做均值池化” 两种方式，结果显示 [CLS] Token 的分类效果更好；
