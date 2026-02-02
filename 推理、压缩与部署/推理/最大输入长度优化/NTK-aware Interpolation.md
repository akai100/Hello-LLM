
NTK-aware Interpolation 是 Position Interpolation (PI) 的关键改进，通过差异化缩放 RoPE 的不同频率维度，
在扩展上下文长度的同时保留高频位置信息。尽管名称含"NTK"（Neural Tangent Kernel），其实际与严格 NTK 理论关联较弱，更多是启发式命名——核心在于**分析位置编码对不同频率的敏感度**。

## PI 的根本缺陷：高频信息丢失

标准 PI 对所有维度统一缩放因子 $s$:

$$m'=\frac{m}{s}$$

导致：

+ 高频维度（ $\theta$大，对应精细位置区分）：被过度压缩 → 相邻 token 位置混淆；
+ 低频维度（ $\theta$小，对应粗粒度位置）：缩放不足 -> 长距离关系建模能力弱

### 实验对比

+ 2K → 8K：PI 可用，但相邻 token 注意力权重波动增大 15%+
+ 2K → 32K：PI 性能崩塌，首尾 token 位置混淆率 >40%

## NTK-aware 核心思想：高频外推 + 低频内插

### 1. 频率感知缩放

NTK-aware 的核心洞察：不同频率维度对位置扰动的敏感度不同

+ 高频维度：对微小位置变化敏感 → 应减少缩放（保留精细区分）
+ 低频维度：对大范围位置变化敏感 → 应增加缩放（增强长距离建模）

### 2. 数学实现：修改 RoPE 基数

原始 RoPE 基频：

$$\theta_i = b^{-2(i-1)/d} (b=10000, d = hidden size)$$

NTK-aware 将基数 $b$ 调整为：

$$b'=b \cdot s^\alpha, \alpha=\frac{d}{d-2}$$

等价于新基频：

$$\theta_{i}^{'}=\theta_i \cdot s^{-2(i-1)/(d-2)}$$

关键特性：

+ 当 i→1（高频维度）： $\theta_{i}^{'}$缩放因子 $\approx 1$ → 几乎不缩放；
+ 当 $i \rightarrow d/2$： $\theta_{i}^{'}$缩放因子 $\approx 1/s$ → 充分缩放；


## 3. Dynamic NTK：避免短文本性能下降

### 3.1 问题

标准 NTK-aware 始终使用新基数，导致：

+ 短序列（<训练长度）性能下降 3–5%

+ 原因：高频维度过度保留，破坏原有位置分布 

### 3.2 解决方案：动态切换

```python3
def dynamic_ntk_rope(q, k, positions, base=10000, max_train_len=2048, dim=4096):
    seq_len = positions.max().item() + 1
    
    if seq_len <= max_train_len:
        # 短序列：使用原始基数（零开销）
        new_base = base
    else:
        # 长序列：动态计算缩放因子
        scale_factor = seq_len / max_train_len
        alpha = dim / (dim - 2)
        new_base = base * (scale_factor ** alpha)
    
    # 后续与 NTK-aware 相同...
```
