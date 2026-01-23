## 1. DPO 是什么

全称：Direct Preference Optimization

目标：用“人类偏好数据”直接训练大语言模型（LLM）生成更符合人类期望的回答。

+ 与 RLHF（Reinforcement Learning with Human Feedback）不同：

  + RLHF 先训练奖励模型（RM），再用 PPO 优化策略
 
  + DPO 跳过 RL，直接用偏好数据优化策略

+ 本质上是一种偏好学习 + 最大似然优化的结合

## 2. DPO 的背景

在训练大型语言模型（LLM）时，直接用最大似然训练（MLE）生成的文本**可能不符合人类偏好**：

+ 例子：

  + 对同一个问题，模型可能生成技术上正确但不够礼貌/简洁的回答
 
  + 生成内容可能偏向训练数据分布而非用户期望

传统方法：

+ RLHF：

  1. 训练奖励模型（RM）来预测哪条回答更好
 
  2. 用 RL（PPO）优化策略，使生成回答的奖励最大化

问题：

+ RLHF 复杂、训练不稳定、计算成本高

+ PPO 涉及采样、clip、策略梯度、value function…

DPO 提出：**直接从偏好数据学习，不用 RL，训练简单且稳定**

## 3. DPO 的核心思想

核心思路：

1. 给定人类偏好数据：

$$(x, y_{preferred}, y_{disprederred})$$

+ $x$: 问题/提示

+ $y_{preferred}$：人类认为更好的回答

+ $y_disprederred$：人类认为差的回答

2. 通过一个**概率模型**直接训练，使模型更倾向生成“更好的回答”

公式化：

+ 对一个提示 x 和回答对 $y^{+}, y^{-}$，定义**偏好概率**：

$$P_{\theta}(y^+ > y^- | x)=\frac{exp(\beta log \pi_{\theta}(y^+|x))}{exp(\beta log\pi_{\theta}(y^+|x))+exp(\beta log \pi_{\theta}(y^-|x))}$$

+ 其中：

  + $\pi_{\theta}$：当前策略/模型概率
 
  + $\beta$：温度参数，控制偏好敏感度
 
  + 这个公式和 Bradley-Terry 模型类型

+ 优化目标：

$$L(\theta)=-\sum_{i}{log P_{\theta}(y_{i}^{+}>y_{i}^{-}|x_i)}$$
