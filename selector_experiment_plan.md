# Best-of-N scorer / selector 实验计划

最后更新时间：2026-04-07

## 先说结论

当前 submission 成绩差，最可能不是因为格式，而是因为**打分规则太弱**。

这份提交本质上做的是：

- 对每道题的 64 个候选 run 打分
- 然后取分数最高的那个

但当前规则的问题是：

1. `aime24/aime25` 主要依赖 `knn` 相似度
2. `lcb_v5` 直接用 `length / max_length`
3. Qwen 的 `brumo25/gpqa/hmmt25` 原计划想用 `vote`，但因为缺原始评测报告，最后 fallback 成了 `knn`
4. 整体上缺少 token-level 置信度信号的利用
5. 不同数据集使用的规则不统一，而且没有认真校准

所以它更像一个“能提交的拼装版”，不是一个真的强 scorer。

---

## 当前可用信号

从本地 cache 可以直接拿到这些信号：

- run-level / geometry：
  - 距离矩阵 `D`
  - 相似度 `S = 1 - D`
  - `medoid`
  - `knn-medoid`
  - `dbscan-medoid`
  - 聚类一致性
- activation-level：
  - `lengths`
  - `kmax_lengths`
- token-level：
  - `tok_conf`
  - `tok_selfcert`
  - `tok_neg_entropy`
  - `tok_logprob`
  - `tok_gini`
- cache 支持 row bank / position window：
  - 可以尝试位置分段 scorer，而不一定只看全序列

---

## 为什么当前方法大概率垫底

通俗版：

- 你现在这个裁判，主要只会看“谁和别人更像”或者“谁更长”
- 但它不会认真看“这个答案自己有没有把握、有没有稳定性、是不是像一个高质量解”
- 于是它很容易把：
  - 冗长但不对的答案
  - 跟一群错答案很相似的答案
  - 或者只是神经元形状很像别人的答案
  选成第一名

换句话说：

- 当前 scorer 对“正确性”这个目标的代理太弱
- 更像是在抓“群体相似”而不是“真实可靠”

---

## 最值得先试的 3 个方案

### 方案 1：KNN + DeepConf 混合分数（最推荐先做）

核心思想：

- `knn` 管“和好答案群体像不像”
- `deepconf` 管“这个答案自己有没有把握”
- 两者加权，比单独用其中一个更稳

建议形式：

`final_score = a * knn_score + b * selfcert_score + c * neg_entropy_score`

或者更保守一点：

`final_score = a * knn_score + b * deepconf_quality`

其中：

- `knn_score`：每个 run 的 top-k 相似度均值
- `deepconf_quality`：
  - `- mean(tok_conf)` 或 `- least_grouped(tok_conf)`
  - 或直接用 `tok_selfcert`

为什么它值得先做：

- 已有实现和数据都支持
- 不需要额外外部文件
- 逻辑最清楚
- 比当前 submission 多利用了 token-level 信息

风险：

- 权重需要调
- 单纯线性加权不一定最优

优先级：**最高**

---

### 方案 2：一致性优先 + 自信度重排

核心思想：

先用几何结构找“主群体”，再在主群体里找最自信的答案。

可以理解成两步：

1. 先筛候选：
   - 取 `knn-medoid` / `medoid` / `dbscan-medoid` / `consensus-*` 的候选集合
   - 或者直接取最大簇里的样本
2. 再重排：
   - 在候选集合中，用 `tok_selfcert` 或 `deepconf` 选最好那个

通俗讲：

- 先问“谁像主流靠谱答案的一员”
- 再问“这几个人里谁自己最有把握”

为什么它值得试：

- 比简单加权更稳一点
- 可以减少离群但高置信错误答案被选中
- 和 `consensus-min/max` 的思路兼容

风险：

- 如果主群体本身就是错的，会一起错
- 候选集合大小和筛法需要调

优先级：**第二**

---

### 方案 3：位置分段 scorer（前半段看结构，后半段看置信度）

核心思想：

答案的前中后段提供的信息可能不一样：

- 前段：更像解题路径 / 思路展开
- 后段：更像最终定型、收束、给答案

所以不要只看全序列，可以做：

- 前半段用 `knn / medoid`
- 后半段用 `selfcert / neg_entropy`
- 最后合并分数

例如：

`final_score = a * early_knn + b * late_selfcert`

为什么它值得试：

- NAD cache 支持 position window
- 这个方向可能比“整段平均”更敏感
- 对推理题特别可能有效

风险：

- 实现稍复杂
- 需要先选一个简单窗口方案，不然实验空间太大

优先级：**第三**

---

## 我不建议现在先做的事

### 1. 不建议先做超复杂模型融合

比如一下子做：

- 多数据集不同权重
- 多窗口多阶段加权
- 很多手调规则

原因：

- 现在还没有强 baseline
- 容易把自己绕晕
- 很难知道到底是哪一部分起作用

### 2. 不建议先追求“真概率校准”

当前最重要的是：

- 能不能把正确答案排前面

不是：

- 分数是不是严格等于答对概率

所以现阶段更重要的是排序能力，不是 calibration。

---

## 建议实验顺序

### 第一轮：做最小可用强 baseline

目标：

- 先做一个比当前 submission 明显更靠谱的 scorer

建议直接做：

- `hybrid_v1 = knn_score + selfcert_score`

更具体一点：

- 每题对 64 个 run 计算：
  - `knn_score`
  - `selfcert_score = mean(tok_selfcert)` 或 `least_grouped(tok_selfcert)`
- 分别做题内归一化
- 最后：
  - `final = 0.6 * knn_norm + 0.4 * selfcert_norm`

如果结果还不行，再试：

- `0.7 / 0.3`
- `0.5 / 0.5`

### 第二轮：做候选重排

目标：

- 验证“两阶段选择”是否优于简单线性加权

建议：

- 候选集合 = `medoid / knn-medoid / dbscan-medoid / consensus-max`
- 在这几个候选中用 `selfcert` 选最优

### 第三轮：做位置窗口版本

目标：

- 验证“答案不同阶段用不同信号”是否有效

建议只试最简单版本：

- 早期窗口：前 25%
- 后期窗口：后 25%
- 不要一开始就搞很多窗口

---

## 实现建议

最务实的做法有两个：

### 做法 A：先写独立实验脚本

优点：

- 最快
- 最适合先验证 scorer 思路
- 不用一上来就嵌进 selector registry

建议文件名：

- `scripts/experiment_best_of_n_hybrid.py`

### 做法 B：验证有效后，再封装成 selector

优点：

- 可以更自然接到 NAD 分析流程里
- 后续复用更方便

建议文件名：

- `plugins/hybrid_selector.py`

---

## 现在最推荐的下一步

直接开始做这个：

**先实现 `hybrid_v1 = knn + selfcert` 的离线实验脚本。**

原因：

- 成本最低
- 最容易解释
- 最有希望比当前 submission 立刻强一截
- 如果它没效果，也能最快排除一条路线
