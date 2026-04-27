请先读取：

- `/home/jovyan/work/NAD_Next/WORKLOG.md`
- `/home/jovyan/work/NAD_Next/selector_experiment_plan.md`
- `/home/jovyan/work/NAD_Next/scripts/experiment_best_of_n_hybrid.py`

然后继续当前任务。

## 当前背景

- 这是 `best_of_n` leaderboard 任务。
- 提交格式已经完全摸清楚。
- 这轮已经做过一个 `hybrid_v1 = knn + selfcert` 的实验版。
- 用户已经提交过 hybrid 版本，结果比几小时前那份还差。

## 已知最重要结论

- validator 接受的正确结构是：
  - `scores[cache_key][problem_id] = {sid: float}`
- `hybrid_v1` 虽然能成功提交，但成绩比旧方案更差。
- 所以下一步不应该继续大换血，而应该：
  - 以旧 mixed submission 为 baseline
  - 做 very small ablation / 局部改良

## 当前最可能的正确方向

请从下面这件事开始：

**基于旧 mixed 方案，设计一个 `mixed_v2`，只改一个局部。**

优先思路：

- 只改 `aime24/aime25`
- 或只改某个模型侧（DS-R1 或 Qwen3-4B）
- 其他部分保持原样

## 要求

- 解释保持通俗直白
- 不要再做大一统大改
- 优先做最小改动、最容易验证的版本
- 如有必要，直接生成新的 submission 文件
