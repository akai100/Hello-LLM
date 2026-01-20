
## 1. GQA 是什么

**GQA（Grouped-Query Attention，分组查询注意力）** 的核心思想是：**多个 Query head 共享一组 Key / Value，但不是所有 head 都共享同一组**

也就是：

+ Query 仍然是 多头

+ Key / Value 被划分为 G 组

+ 每组 K/V 被多个 Query head 共享

## 2. 为什么需要 GQA

| 方案      | 特点               | 问题          |
| ------- | ---------------- | ----------- |
| **MHA** | 每个 head 独立 Q/K/V | KV Cache 太大 |
| **MQA** | 所有 head 共用一组 K/V | 表达能力下降      |

👉 GQA 的目标：

+ 接近 MQA 的效率

+ 接近 MHA 的性能

### 2️⃣ 核心观察

Query 的多样性最重要，而 K/V 的多样性存在冗余

因此：

+ 不必为每个 Query 都配独立 K/V

+ 但也不能完全共享

## 数学形式

### MHA

$$head_i=Attention(QW_{i}^Q, KW_{i}^{K}, VW_{i}^{V})$$

### MQA

$$head_i=Attention(QW_{i}^Q, KW^K, VW^V)$$

### GQA

$$head_i=Attention(QW_{i}^Q, KW_{g(i)}^K, VW_{g(i)}^V)$$

其中：

+ $g(i)$: head i 所属的 KV 分组编号

+ 每组有独立的 $W^K, W^V$

## 优缺点

### ✅ 优点

+ 显著减少 KV Cache

+ 推理吞吐提升明显

+ 几乎不损失模型性能

+ 已成为工业标准

### ❌ 缺点

+ 实现略复杂于 MHA

+ KV 分组数需调参
