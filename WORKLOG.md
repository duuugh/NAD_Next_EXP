# NAD_Next 工作日志（汇报精简版）

最后更新时间：2026-04-08

## 1. 任务目标

目标是在 `best_of_n` 任务中，从每道题的 64 个候选回答里选出最优答案，并提升 leaderboard 表现。

本质上，这不是重新生成答案，而是：

- 对 64 个候选做打分 / 排序
- 最终选一个作为输出

---

## 2. 任务形式理解

当前任务可以理解为：

- `scorer`：给每个候选答案打一个分数
- `selector`：根据这些分数选出最终答案

因此可以把问题表述成：

- 设计一个更好的候选排序规则
- 再把这个规则变成可提交的选择结果

---

## 3. 现有基线方案：mixed baseline

首先构造并提交了一份 `mixed` 方案。

它的特点是：

- 不同数据集使用不同规则
- 主要混合了：
  - `knn`
  - `max_activation`
  - 部分 fallback 规则

实验结果：

- 可以成功提交
- leaderboard 成绩不高
- 但它成为了当前阶段的 **baseline**

结论：

- 虽然这个方案不强，但它比后续某些新尝试更稳
- 因此适合作为后续改进的出发点

---

## 4. 新方案尝试：hybrid_v1

之后实现并测试了一个统一的 hybrid scorer：

- `hybrid_v1 = 0.6 * knn_norm + 0.4 * selfcert_norm`

所用信号：

- `knn` 相似度
- token-level `tok_selfcert`

核心想法：

- 既看“和其它候选的结构相似性”
- 也看“答案自身的置信度”

实验结果：

- 成功生成并提交
- 但 leaderboard 表现 **比 mixed baseline 更差**

结论：

- 简单地把 `knn` 和 `selfcert` 线性混合，并没有带来提升
- 至少在当前权重和定义下，这条路线不如 baseline

---

## 5. 论文方法尝试：E_M-regularized Best-of-N

阅读论文：

- *Revisiting the (Sub)Optimality of Best-of-N for Inference-Time Alignment*

并实现了论文中的方法：

- `E_M-regularized Best-of-N`

方法要点：

1. 已有 N 个候选及其分数
2. 设 `k = ceil(N / M)`
3. 按分数排序
4. 从前 `k` 个候选中均匀随机选一个

本次做法：

- 直接把当前 best submission 中的 `{sid: float}` 当作 reward score
- 不做额外训练
- 不做在线估计
- 只依赖当前候选及其分数

已生成三个版本：

- `M = 2`
- `M = 4`
- `M = 8`

目的：

- 比较“更保守”与“更随机”的选择策略在任务中的表现差异

当前状态：

- 三个版本都已生成
- leaderboard 暂时卡住，尚未完成结果比较

---

## 6. 当前最重要结论

截至目前，有三个关键结论：

1. **mixed baseline 是当前可靠基线**
   - 它虽然不强，但比 `hybrid_v1` 更稳

2. **简单的 `knn + selfcert` 混合没有提升**
   - `hybrid_v1` 提交后效果更差

3. **论文方法值得继续验证**
   - `E_M-regularized BoN` 已经实现
   - 但还需要 leaderboard 结果来判断它是否优于 baseline

---

## 7. 下一步计划

当前最合理的后续路线是：

### 路线 A：比较论文方法三版结果

比较：

- `M = 2`
- `M = 4`
- `M = 8`

目标：

- 判断论文方法在当前任务上是否有效
- 判断更保守还是更随机更合适

### 路线 B：如果论文方法不优，则回到 baseline 做小步改良

原则：

- 不再做大一统大改
- 以 `mixed baseline` 为基础做 `mixed_v2`
- 只替换一个局部模块或少量数据集

这样做的好处是：

- 能清楚知道是哪一处改动有效
- 避免整份 submission 一起漂移，难以分析

---

## 8. 汇报建议

如果用于 PPT，可以围绕下面的主线展开：

1. 任务目标：从 64 个候选中选最优
2. baseline：mixed 方案
3. 新尝试：hybrid_v1
4. 论文方法：E_M-regularized BoN
5. 当前结论：baseline 更稳，论文方法待验证
6. 下一步：继续做局部改良或比较 `M` 的影响

---

## 9. 新一轮小步实验：mixed_v2 局部 selector 改良

在确认 `hybrid_v1` 不如 baseline 后，这一轮不再继续做统一大改，而是回到 `mixed baseline`，只做 very small selector ablation。

核心原则：

- 保持旧 `mixed` 主体不变
- 只改 `aime24/aime25`
- 其他数据集完全不动
- 只在 baseline 很接近的候选里做二次选择

### 9.1 第一版 mixed_v2：top-3 + selfcert 重选

实现了一个 submit-safe 的 `mixed_v2` 构造脚本：

- 基于 `best_of_n_nad_mixed_v1_complete.json`
- 仅对：
  - `DS-R1/aime24`
  - `DS-R1/aime25`
  - `Qwen3-4B/aime24`
  - `Qwen3-4B/aime25`
- 保持原 baseline 排序逻辑
- 当 baseline 的 `top1-top2 <= 0.002` 时：
  - 取 `top-3` 候选
  - 用 `tok_selfcert` 做二次重选
- 输出仍保留完整 `{sid: float}` 结构，满足 leaderboard 提交要求

实验结果：

- `HIT@1`：从 `0.6719` 提升到 `0.6747`
- 绝对提升：`+0.0028`
- 但：
  - `HIT@3` 持平
  - `SELACC@10%` 持平
  - `PAIRWISE ACC` 持平

结论：

- 说明“小范围 selector 修正”是有可能带来一点收益的
- 但当前收益主要体现在“最后第一名”的局部修正
- 并没有改善整体排序能力

### 9.2 更保守版本：top-2 + gap=0.001

为了进一步降低风险，又生成了两个更保守的版本：

1. `top-2 + max-gap=0.001 + tok_selfcert`
2. `top-2 + max-gap=0.001 + tok_logprob`

共同特点：

- 仍然只改 `aime24/aime25`
- 仍然只在 baseline 非常接近时触发
- 触发范围比前一版更小

提交结果：

- `selfcert` 版：
  - leaderboard 表现不如前面版本
  - 平均排名下降
- `logprob` 版：
  - 平均排名和四项指标都与提交前一致
  - 也就是基本持平

结论：

- `tok_selfcert` 作为 tie-break 信号不够稳，容易伤平均排名
- `tok_logprob` 更安全，但增益较弱
- 说明当前这条“token-level 指标做 top-k 重选”的路线：
  - 不是完全没用
  - 但上限暂时不高

---

## 10. 当前阶段最新判断

截至现在，可以把结论更新为：

1. **`mixed baseline` 仍然是主干方案**
   - 目前依然是最稳的参考基线

2. **selector 小修正比统一大改更合理**
   - `hybrid_v1` 这种统一 scorer 风险大
   - `mixed_v2` 这种 very small ablation 更容易带来稳定信息

3. **`selfcert` 不适合继续作为主推 tie-break 信号**
   - 它可能在局部题目有帮助
   - 但整体上不够稳

4. **`logprob` 可以作为更安全的轻量 tie-break 备选**
   - 但目前看增益不明显

5. **当前瓶颈不在“能不能微调 selector”**
   - 而在于：现有 tie-break 信号太弱，难以显著提升整体排序

---

## 11. 下一步计划（更新版）

基于最新实验结果，下一步不建议再继续扩大全局改动，而应做更小、更可归因的拆分实验。

当前最值得做的是：

### 路线 A：只改单个模型侧

原因：

- 当前总分变化很小
- 很可能是一侧有收益、另一侧把收益抵消了

优先实验：

1. 只改 `DS-R1/aime24,aime25`
2. 只改 `Qwen3-4B/aime24,aime25`

### 路线 B：继续保守 selector，而不是继续换很多 token 指标

原则：

- 继续只在 baseline 最犹豫的 case 上触发
- 不再扩大到更多数据集
- 不再尝试统一新 scorer

如果后续单模型侧版本仍然没有明显收益，则可以基本判断：

- 当前这条“top-k 内 token-level tie-break”路线的边际价值已经接近耗尽
- 后续应考虑回到 `mixed baseline` 的更结构化局部改造，而不是继续在同一类轻量信号上反复试

