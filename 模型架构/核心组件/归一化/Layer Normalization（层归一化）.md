
## 原理

**核心目标**：对单条样本的特征维度进行归一化，让每个样本内部的特征分布均值为 0，方差为 1，从而提高训练稳定性。

+ 归一化维度：隐藏层或特征维度

+ 不依赖 batch size -> 特别适合小 batch 或序列长度可变的 NLP / Transformer

## 公式

给定输入向量 $x = [x_1, x_2, ..., x_H]$，其中 $H$是特征维度：

$$\mu=\frac{1}{H}\sum_{i=1}{H}{x_i}$$

$$\sigma^2=\frac{1}{H}\sum_{i=1}^{H}{(x_i - \mu)^2}$$

$$\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2 + \epsilon}}$$

+ $\gamma, \beta$ 是可学习缩放平移参数：

$$y_i=\gamma\hat{x}_i+\beta$$

** 特点**

+ 归一化只针对单条样本的特征

+ 每条样本独立处理，不依赖 batch

+ 训练和推理行为一致

## LayerNorm 在大模型中的作用

### 在 Transformer 中的位置

+ Pre-LN Transformer（推荐）

```
x → LayerNorm → Multi-Head Attention → Add(x) → Feed Forward → Add
```

+ Post-LN Transformer

```
x → Multi-Head Attention → Add(x) → LayerNorm → Feed Forward → Add → LayerNorm
```

**Pre-LN**

+ 训练更稳定

+ 梯度消失 / 爆炸问题更少

+ 大模型 GPT / LLaMA / BLOOM 都采用

### 解决问题

+ **训练稳定性**：防止激活值过大或过小

+ **梯度稳定性**：防止梯度爆炸或消失

+ **序列长度可变**：适合 NLP / Transformer / RNN

+ **推理一致性**：不依赖 batch → 推理 batch size = 1 也稳定
