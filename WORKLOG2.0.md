# NAD_Next 工作日志 2.0

最后更新时间：2026-04-13

## 0. 已有的基本数据

这一节先不讲新实验，只记录当前这个仓库里已经明确、已经能直接用的基础信息，后面做任何讨论都默认以这里为背景。

### 0.1 这个仓库现在是干什么的

- `NAD_Next` 本质上是一个：
  - 读 activation cache
  - 跑 selector
  - 做可视化
  - 做 batch 分析
  - 生成 best-of-n / selector 实验结果
  的工作目录。

- 这里面既有：
  - 核心代码
  - 实验脚本
  - 本地网页工具
  - 已经产出的结果文件

所以它不是一个“只放脚本”的目录，而是一个完整实验工作区。

### 0.2 顶层目录结构

当前顶层最重要的目录有：

- `nad/`
  - 核心 Python 包
  - 真正负责读取 cache、计算 selector、跑分析流程

- `cookbook/`
  - 一套按步骤组织的操作手册和启动脚本
  - 包括环境安装、cache 浏览、可视化、批量分析、位置 ablation、deepconf 分析

- `plugins/`
  - 自定义 selector 插件目录
  - 目前这里放的是实验性或外接式 selector

- `scripts/`
  - 各类实验脚本、画图脚本、构造 submission 的脚本

- `tools/`
  - 杂项工具
  - 包括 cache browser、accuracy 计算、ground truth 生成等

- `minimal_visualization_next/`
  - 当前新版可视化网页服务

- `result/`
  - 已有实验输出
  - 包括 mixed / hybrid / EM regularized / mixed_v2 / A1 等结果文件

- `token_per_kink_out/`
  - kink 相关分析产物和图片

- `MUI_HUB/`
  - 不是普通目录，而是一个软链接
  - 当前指向：`/home/jovyan/public-ro/MUI_HUB`
  - 也就是外部 cache 数据入口

### 0.3 顶层重要文件

当前顶层最重要的文件有：

- `README.md`
  - 整个项目的总说明

- `WORKLOG.md`
  - 上一阶段工作记录

- `WORKLOG2.0.md`
  - 当前阶段工作记录

- `NEXT_PROMPT.md`
  - 上一轮交接 prompt

- `selector_experiment_plan.md`
  - best-of-n scorer / selector 实验计划

- `selector_rankings_20260330_023531.csv`
- `selector_rankings_20260330_023531.json`
  - selector 排名结果

- `selector_comparison_en.png`
  - selector 对比图

- `nad_config.json`
  - 可视化时查找 tokenizer / model path 用的配置文件

- `pyproject.toml`
- `requirements.txt`
  - Python 项目配置与依赖

### 0.4 核心代码分工

`nad/` 目录可以简单理解成下面几块：

- `nad/api.py`
  - 高层 API

- `nad/cli/`
  - 命令行入口

- `nad/core/selectors/`
  - selector 实现与注册表

- `nad/io/`
  - cache 加载、catalog、可视化数据读取

- `nad/ops/`
  - accuracy、selector ranking、统计操作

- `nad/pipeline/`
  - 分析流程编排

### 0.5 当前仓库里有哪些 selector

目前可以分成三类来理解：

#### A. 内置主 selector

- `min-activation`
  - 选激活最少的 run

- `max-activation`
  - 选激活最多的 run

- `medoid`
  - 选几何中心样本

- `knn-medoid`
  - KNN 版本 medoid

- `dbscan-medoid`
  - 先聚类，再选主簇 medoid

- `consensus-min`
  - 多种结构化 selector 的候选集合里，偏向更短/更少激活的版本

- `consensus-max`
  - 多种结构化 selector 的候选集合里，偏向更长/更多激活的版本

- `deepconf`
  - 用 token-level confidence 信号做选择

- `con64@`
  - 基于 64 跑结果做共识

- `avg64@`
  - 基于 64 跑结果做平均

#### B. legacy 兼容 selector

- `legacy-knn-medoid`
- `legacy-medoid`
- `legacy-dbscan-medoid`
- `legacy-consensus-min`
- `legacy-consensus-max`

这些主要用于复刻或对齐旧版 NAD 行为。

#### C. 当前仓库里的插件 selector

- `plugins/kink_selector.py`
  - 基于 token-per-kink 的 selector

- `plugins/medoid_activation_tiebreak.py`
  - `A1 = medoid + activation tie-break`
  - 已经做过单 cache 实验，但当前结果不成立

- `plugins/em_regularized_bon_selector.py`
  - 论文里的 `E_M-regularized Best-of-N` selector 实现

### 0.6 当前涉及的数据集 / 题目集

当前要区分两层：

#### A. 框架里常见、文档里明确支持的数据集

- `aime24`
- `aime25`
- `gpqa`
- `humaneval`
- `livecodebench`
- `mbpp`

#### B. 当前 best_of_n 实验脚本里实际在用的数据集

- `aime24`
- `aime25`
- `brumo25`
- `gpqa`
- `hmmt25`
- `lcb_v5`

补充说明：

- `lcb_v5` 基本对应 `livecodebench_v5`

### 0.7 当前 best_of_n 里涉及的模型侧

当前脚本里明确使用了两侧：

- `DS-R1/*`
- `Qwen3-4B/*`

对应的 cache key 结构是：

- `DS-R1/aime24`
- `DS-R1/aime25`
- `DS-R1/brumo25`
- `DS-R1/gpqa`
- `DS-R1/hmmt25`
- `DS-R1/lcb_v5`
- `Qwen3-4B/aime24`
- `Qwen3-4B/aime25`
- `Qwen3-4B/brumo25`
- `Qwen3-4B/gpqa`
- `Qwen3-4B/hmmt25`
- `Qwen3-4B/lcb_v5`

### 0.8 当前本地能看到的 cache 数据

当前通过 `MUI_HUB/cache` 这条链路，已明确可见的是：

- 模型：
  - `DeepSeek-R1-0528-Qwen3-8B`

- 数据集：
  - `aime24`
  - `aime25`
  - `brumo25`
  - `gpqa`
  - `hmmt25`
  - `livecodebench_v5`

当前已知数量：

- `aime24`：2 个 cache
- `aime25`：1 个 cache
- `brumo25`：1 个 cache
- `gpqa`：1 个 cache
- `hmmt25`：1 个 cache
- `livecodebench_v5`：1 个 cache

总计：

- 1 个模型
- 6 个数据集
- 7 个 cache

### 0.9 当前可直接用的本地网页 / 可视化入口

当前最重要的网页工具有两个，它们都不是公网网站，而是本地启动的服务。

#### A. Cache Browser

- 入口文件：
  - `tools/cache_browser.py`

- 启动脚本：
  - `cookbook/01_cache_browser/cache_browser.sh`

- 默认地址：
  - `http://localhost:5003`

- 作用：
  - 看当前有哪些 cache
  - 看它们属于哪个 model / dataset
  - 看样本数、题目数、accuracy、temperature、build date

- 更通俗地说：
  - 这是“总览页”
  - 用来先盘点数据仓库里到底有什么

#### B. Visualization Server

- 入口文件：
  - `minimal_visualization_next/app.py`

- 启动脚本：
  - `cookbook/02_visualization/visualization.sh`

- 默认地址：
  - `http://localhost:5002`

- 作用：
  - 深入看单个 cache
  - 看题目级别 run 分布
  - 看 activation 轨迹
  - 看 token-level confidence / entropy
  - 看 selector 在具体题上的表现

- 更通俗地说：
  - 这是“精看页”
  - 用来研究某个 cache 里面到底发生了什么

### 0.10 当前已经存在的重要结果产物

`result/` 目录里已经有多类实验结果，当前最重要的包括：

- `best_of_n_nad_mixed_v1_complete*.json`
  - mixed baseline 相关结果

- `best_of_n_hybrid_v1*.json`
  - `hybrid_v1 = knn + selfcert` 相关结果

- `best_of_n_em_regularized_m2/m4/m8*.json`
  - 论文 `E_M-regularized BoN` 三个版本结果

- `best_of_n_nad_mixed_v2_aime_*.json`
  - 只在 `aime24/aime25` 上做 very small selector ablation 的结果

- `a1_medoid_activation_aime24*.json`
  - `A1 = medoid + activation tie-break` 的单 cache 实验结果

### 0.11 当前这份基础盘点的用途

后面继续写实验记录时，默认基于下面这套共识：

1. 这个仓库同时包含：
   - 核心代码
   - 实验脚本
   - 可视化工具
   - 已产出的结果

2. 当前 best_of_n 工作不是从零开始：
   - 已有 mixed baseline
   - 已有 hybrid_v1
   - 已有 EM-regularized BoN
   - 已有 mixed_v2 小步实验
   - 已有 A1 activation tie-break 单 cache 验证

3. 后续如果再提“网页”“缓存”“selector”“数据集”，默认就是以上这些对象。

## 1. 当前决定

- 旧的推进方式先暂停。
- 不继续沿着之前那条 prompt 里的既定路线往下做。
- 从现在开始，单独启用这份 `WORKLOG2.0.md` 记录新的工作方式。

---

## 2. 已确认背景

- 当前任务仍然是 `best_of_n` leaderboard。
- 提交格式已经明确：
  - `scores[cache_key][problem_id] = {sid: float}`
- 旧 `mixed baseline` 是一个可提交、相对稳定的参考版本。
- `hybrid_v1` 已经验证过，效果不如旧方案。
- 因此当前阶段不再优先做“大一统大改”，而是优先做小范围、可归因的实验。

---

## 3. 从现在开始的新工作方式

这一版不急着直接做新 submission，而是先换工作节奏：

1. 先把新思路说清楚  
2. 再定义一个最小可验证实验  
3. 每次只改一个核心点  
4. 每次都记录：
   - 为什么这样改
   - 具体改了什么
   - 预期会改善什么
   - 最终结果如何
5. 尽量拆成小任务推进，避免一次性做太多、太重的请求

目标是：

- 少做大而散的尝试
- 少做不容易归因的改动
- 让每一步都更容易判断“到底有没有用”

---

## 4. 新路线记录模板

后续每轮实验按下面格式补：

### 方案名

- 待填写

### 核心假设

- 待填写

### 改动范围

- 待填写

### 使用信号 / 特征

- 待填写

### 预期收益

- 待填写

### 风险

- 待填写

### 输出产物

- 待填写

### 结果

- 待填写

### 结论

- 待填写

---

## 5. 当前状态

- 旧 `WORKLOG.md` 保留，作为上一阶段记录。
- 新阶段从这份 `WORKLOG2.0.md` 开始。
- 已经看过：
  - `activation_61.png`
  - `activation_70.png`
- 已经确认：
  - `61` 和 `70` 是题号，不是 layer 编号。
- 已经确认：
  - `selector_rankings_20260330_023531.csv` 里，`max-activation` 排名最后。
- 已经做过第一轮最小真实实验：
  - `A1 = medoid + activation tie-break`
- 当前结论是：
  - activation 图里确实能看到一些现象
  - 但还没有得到一个可以直接采用的新 selector

---

## 6. 当前主线

当前主线不是直接重新发明一个新 selector，而是先回答下面这个问题：

- activation 图里看到的现象，能不能作为一个很小的局部规则，安全地挂到强一点的 selector 上？

因此当前优先级是：

1. 先把图里的现象解释清楚  
2. 再判断它能不能变成可计算特征  
3. 再看它能不能作为局部 tie-break 使用  
4. 先做单 cache、单 selector 的最小实验  

---

## 7. 对 activation 图的重新理解

这次讨论后，已经明确：

- 不能只凭“点更密/更散”来判断对错。
- 必须先看图的横轴、纵轴分别表示什么。

当前对图的解释是：

- 横轴：`Token Position (Slice ID)`
  - 表示生成过程走到了第几个 token
- 纵轴：`Cumulative Unique Neurons`
  - 表示到当前 token 为止，累计激活过的不同 neuron 数量

因此图中的“左/右、下/上”不是抽象二维坐标，而有明确含义：

- 更靠左：答案更早结束 / token 更少
- 更靠右：答案更长 / token 更晚结束
- 更靠下：累计 unique neurons 更少
- 更靠上：累计 unique neurons 更多

这意味着，图里看到的规律不能写成简单的“左下正确、右上错误”口号，
更准确的表达应该是：

- 在观察到的图上，某些正确答案更早结束
- 并且到结束时，累计 unique neurons 更少
- 某些错误答案则更容易更长，且累计 unique neurons 更高

---

## 8. 当前能确认的图像观察范围

当前主要看了：

- `activation_61.png`
- `activation_70.png`

当前能确认的是：

1. 在题 `61` 的图上，正确 / 错误候选之间能看到比较明显的轨迹差异  
2. 在题 `70` 的图上，也能看到类似趋势，但没有题 `61` 那么明显  

但同样需要明确限制：

- 这些观察目前只对应具体题目
- 还不能直接写成跨所有题都成立的通用规律
- 更不能直接写成“某一层固定有效”的结论

---

## 9. 为什么不再以 `max-activation` 为底座

这一点现在已经比较明确：

1. 从现有 selector 排名看：
   - `max-activation` 在 `selector_rankings_20260330_023531.csv` 中是最后一名

2. 从图像观察看：
   - 当前看到的现象更接近“某些正确答案更短、终点累计 unique neurons 更低”
   - 这与 `max-activation` 的“越多越好”方向并不一致

因此当前不再采用下面这条旧判断：

- “优先从 `max-activation` 出发做融合”

当前更新后的判断是：

- 如果要做 activation-aware 规则，应优先挂在更强、更稳的 selector 上
- `medoid`、`consensus-max` 这类 selector 比 `max-activation` 更适合作为实验底座

---

## 10. 第一轮最小真实实验：A1

### 方案名

- `A1 = medoid + activation tie-break`

### 核心假设

- 在 `medoid` 很犹豫的 case 上
- 如果用 activation 终点特征做二次判别
- 也许能把一部分选错的题修正回来

### 改动范围

- 不改主框架逻辑
- 只新增一个外部 selector 插件：
  - `plugins/medoid_activation_tiebreak.py`

### 使用信号 / 特征

- 先用 `medoid` 做主选择
- 只在 top medoid 候选很接近时触发 tie-break
- tie-break 里使用：
  - 终点 token 位置
  - 终点累计 unique neurons

### 输出产物

- 插件文件：
  - `plugins/medoid_activation_tiebreak.py`
- 分析结果：
  - `result/a1_medoid_activation_aime24.json`
- 准确率结果：
  - `result/a1_medoid_activation_aime24_accuracy.json`

### 结果

在 `DS-R1/aime24` 单 cache 上的结果为：

- `medoid`：`24/30 = 80.00%`
- `A1`：`23/30 = 76.67%`

补充观察：

- A1 一共改动了 `14/30` 道题的选择
- 这批改动里没有带来明确净收益
- 还出现了至少一题从原本正确变成错误

### 结论

- 当前这个版本的 `A1` 不成立
- 至少在这个真实 cache 上，它比原始 `medoid` 更差
- 所以不能直接继续扩大跑更多 cache，更不能直接往 submission 里接

---

## 11. 当前阶段的工作原则

基于目前的结果，当前原则更新为：

1. 不把题 `61` / `70` 上的现象直接当成全局规律  
2. 不再以 `max-activation` 作为默认融合底座  
3. 如果继续做 activation 规则，只能挂在更强 selector 上  
4. 触发条件必须足够保守，避免一次改太多题  
5. 每次先做单 cache 验证，没收益就及时停止  

当前明确不做的事：

- 不再用“疏密本身”判断对错
- 不把图粗暴理解成普通二维聚类图
- 不在还没验证前，直接把观察推广成 submission 规则

---

## 12. 下一步（待讨论）

当前还没有锁定下一步具体怎么改。

接下来需要讨论的是：

- 是不是继续沿着 `A1` 这条线，把触发条件收得更紧
- 还是先退一步，不急着写 selector，先继续找更可泛化的 activation 特征
- 又或者改成只分析少量具体题目，先积累更多可解释案例

在这一步讨论清楚之前，不继续扩大实验范围。

---

## 13. 新补充观察：尾段 activity warning

这次又额外看了几张从 `5002` 可视化网页导出的图：

- `activation_78.png`
- `activation_80.png`
- `activation_82.png`
- `activation_85.png`

前面的 `61` / `70` 已经讨论过，新的观察是：

- 某些错误答案在最后的“思考尾段”里
- 曲线还没有完全结束，但已经明显不再继续引入新的 neuron
- 更准确地说，不是“完全没有神经元活动”，而是：
  - **尾段新增的 unique neurons 很少**
  - **曲线过早平台化**

这一点需要和之前的观察一起理解，而不是单独理解成一个口号。

### 13.1 当前对这个新现象的解释

当前更合适的表述不是：

- “低就是对”
- “平就是错”

而是：

- 错答案可能有不止一种异常形态
- 一种是：尾段继续拖长、继续膨胀、越想越散
- 另一种是：尾段虽然还在输出，但已经发空，不再引入新的 unique neurons

也就是说：

- 错答案可能既会“过热”
- 也可能会“过冷”

所以当前不再追求“某一种理想形状”，而是更倾向于：

- 把 **尾段异常** 当成 warning signal

### 13.2 为什么这个信号不能直接做主规则

当前判断是：

- 这个新观察是有价值的
- 但不适合直接写成主 selector 规则
- 更不适合写成“只要尾段低就判错”这种单向规则

原因是：

1. 有些正确答案本来也会在最后自然收尾，曲线也可能变平  
2. 当前观察仍然来自少量具体题目，不足以直接推广成全局定律  
3. 如果把它直接当主规则，风险会和 A1 一样，改动太多题，破坏已有稳 selector  

因此它更适合作为：

- 一个 **保守的小提醒**
- 或者一个 **warning / veto 型信号**

### 13.3 当前对 activation 规则的更新理解

当前更稳的思路不是：

- 奖励某一种固定曲线形状

而是：

- 识别尾段有没有明显异常

也就是把 activation signal 从“主裁判”降级成：

- 副裁判
- 边裁
- 小警报

这比直接让 activation 决定最终选择更稳。

### 13.4 为什么先做 `medoid` 版本，而不是直接上排名前两名

这一步也已经讨论清楚：

1. `selector_rankings_20260330_023531.csv` 里，前 3 名确实是：
   - `con64@`
   - `consensus-max`
   - `medoid`

2. 但这个排序是按综合 `Score` 排的，不是只按单一 accuracy 排的  
   - `medoid` 的 `micro_accuracy` 实际上略高于 `consensus-max`

3. `con64@` 不是最适合直接挂 activation 插件的那种普通 selector 底座  

4. `consensus-max` 本身已经是一个组合规则：
   - 先综合 `knn-medoid`
   - `medoid`
   - `dbscan-medoid`
   - 再在候选里偏向更长的那个

所以如果直接在 `consensus-max` 上挂 activation 规则，会更难判断：

- 到底是 activation warning 有用
- 还是底座自身结构在起作用

因此当前先选择：

- 用 `medoid` 做一个更干净、更容易归因的版本 A

这不表示 `medoid` 一定是全局最优底座，
而是表示：

- 它更适合作为第一轮验证 activation warning 是否值得保留的实验底座

### 13.5 版本 A 的当前定义

当前准备先做：

- `Version A = medoid + tail warning`

核心思路：

1. 主选择仍然是 `medoid`  
2. 只在 `medoid` top2 非常接近时触发  
3. activation 不直接拍板选人  
4. activation 只做一件事：
   - 给“尾段明显异常”的候选一个 warning

当前最重要的实现原则是：

- **保守触发**
- **少改题目**
- **先做单 cache 验证**
- **先验证 warning 有没有净收益，再考虑换更强底座**

### 13.6 Version A 之后的自然下一步

如果 `Version A` 没有明显收益，则说明：

- activation tail warning 作为 `medoid` 的轻量修正，价值有限

如果 `Version A` 有一点净收益，则下一步再考虑：

- 用同样 warning 规则挂到 `consensus-max` 上做版本 B

也就是说，当前顺序明确为：

1. 先做 `Version A = medoid + tail warning`  
2. 只有 A 有正信号，再考虑 `Version B = consensus-max + tail warning`

---

## 14. Version A 实验：medoid + tail warning

### 14.1 方案名

- `Version A = medoid + tail warning`

### 14.2 核心假设

- 某些错误答案在尾段虽然还在输出 token
- 但已经几乎不再引入新的 unique neurons
- 如果这种“尾段过早平台化”现象刚好出现在 `medoid` 犹豫的 top2 里
- 那么也许可以作为一个保守的 veto 信号

### 14.3 改动范围

- 不改 NAD 主框架
- 新增一个独立插件文件：
  - `plugins/medoid_tail_warning.py`

与 A1 的区别是：

- A1 用 activation 终点特征直接做 tie-break
- Version A 不直接拍板选人
- Version A 只在 `medoid` top2 极接近时，尝试否掉“尾段明显过早平台化”的 base 候选

### 14.4 Version A 的实现原则

当前实现是一个非常保守的版本：

1. 主选择仍然是 `medoid`  
2. 只比较 top2 medoid 候选  
3. 只有当 top2 gap 足够小时才触发  
4. 只在 base 候选被 warning、runner-up 没被 warning 时才可能切换  
5. warning 不直接打分，只做 veto  

### 14.5 输出产物

- 插件文件：
  - `plugins/medoid_tail_warning.py`

- 严格版结果：
  - `result/versionA_medoid_tail_warning_aime24.json`
  - `result/versionA_medoid_tail_warning_aime24_accuracy.json`

- 放宽版结果：
  - `result/versionA_medoid_tail_warning_aime24_loose.json`
  - `result/versionA_medoid_tail_warning_aime24_loose_accuracy.json`

### 14.6 单 cache 结果

在 `DS-R1/aime24` 上，先跑了一个严格版，再跑了一个稍微放宽参数的版本。

结果两版一致：

- `medoid`：`24/30 = 80.00%`
- `Version A`：`24/30 = 80.00%`

补充观察：

- 严格版改动题数：`0/30`
- 放宽版改动题数：`0/30`

也就是说：

- 当前这个 `medoid + tail warning` 版本在这份 cache 上是 **零触发 / 零改动**
- 没有伤到 baseline
- 但也没有带来任何净收益

### 14.7 为什么会零触发

为了确认是不是 signal 根本不存在，又额外对用户重点观察的题做了定点检查：

- `78`
- `80`
- `82`
- `85`

检查内容包括：

- `medoid` top2 的 gap
- top2 两个候选的 tail metrics
- 它们是否被 tail warning 命中

结果显示：

- 这几题里，`medoid` top2 的候选尾段都不属于“明显过早平台化”
- 所以 Version A 根本没有切换条件

也就是说：

- 当前不是“warning 信号完全不存在”
- 而是“warning 没出现在 medoid 真正在犹豫的那两个候选里”

### 14.8 进一步诊断

继续往下看后发现：

- 在 `78 / 82 / 85` 这些题里
- 确实都能找到至少一个被 warning 命中的错误 run
- `80` 里也能找到被 warning 命中的错误 run

但关键问题是：

- 这些被 warning 命中的错误 run，**不在 medoid top2 里**

这说明当前 Version A 的局限不是“tail warning 完全没信号”，而更像是：

- 这个信号出现的位置太靠外层
- 当前把它只挂在 `medoid` top2 上，触发得太晚、太窄

### 14.9 当前结论

目前可以更新判断为：

1. `tail warning` 作为现象本身，**不是完全假的**  
2. 但把它写成 `medoid top2 veto` 后，当前版本过于保守，基本不触发  
3. 所以 Version A 没有带来收益，但这个失败更像是“挂载位置不对”，而不完全是“信号无效”  

### 14.10 对下一步的启发

当前更合理的下一步不应该是：

- 直接把 warning 做得更激进，硬改更多题

而应该优先考虑：

- 让 warning 进入一个稍微更宽一点、但仍可控的候选集合

例如未来可讨论的方向是：

- 不是只看 `medoid` top2
- 而是看 `medoid` 的一个 very small candidate set
- 或者看 `consensus` 形成的 very small candidate set

但这已经不再是当前这版 Version A 的定义。

当前 Version A 可以视为：

- 一次成功完成的最小验证
- 结果是：**信号存在，但以当前挂法没有转化成有效选择器收益**

---

## 15. 最新 leaderboard 反馈更新

### 15.1 这次提交的外部结果

用户已经把下面这版文件提交到了 leaderboard：

- `result/best_of_n_versionA_loose_aime24_submit.json`

当前反馈是：

- 这版结果 **不算差**
- 但截至目前，表现最好的方法仍然是：
  - `nad_mixed_v2_aime_top2_gap1e3_logprob`

### 15.2 这条结果意味着什么

这次 leaderboard 反馈进一步说明：

1. `activation` 相关现象不是完全没信息  
2. 但单独沿着 `medoid + activation warning` 这条线往前推，当前还不是最优主线  
3. 到目前为止，`logprob` 仍然是更稳、更强的 tie-break 信号  

也就是说：

- 当前最佳方案不是 activation 系 selector
- 而是更保守、更加 submit-safe 的：
  - `top2 + gap=1e-3 + logprob`

### 15.3 对当前阶段主线的更新

基于这次反馈，当前主线应该更新为：

- **以 `nad_mixed_v2_aime_top2_gap1e3_logprob` 作为新的最强参考基线**

这比继续把精力放在 `medoid` 底座上更合理。

更通俗地说：

- `medoid + activation` 这条线到现在更像是“帮我们理解现象”
- `logprob mixed_v2` 才是当前真正值得继续小步改良的主干方案

### 15.4 当前对 activation signal 的更新定位

截至现在，可以把 activation signal 的定位进一步说清楚：

- 它更像是一个 **辅助 warning**
- 不是当前最好的主 tie-break
- 也不是当前最应该单独扩大的路线

因此后面如果还要继续利用 activation，原则应该是：

- 不让它单独主导选择
- 只把它挂在更强、更稳的 baseline 上
- 作为 very small veto / warning 使用

### 15.5 当前最合理的下一步方向

在这个节点上，最自然的下一步不再是：

- 继续做 `medoid` 版本的扩展实验

而更应该是：

- 以 `nad_mixed_v2_aime_top2_gap1e3_logprob` 为底座
- 再考虑是否给它挂一个非常轻的 activation warning

也就是说，后续如果继续做 activation 融合，应该转成类似下面这种方向：

- `logprob baseline + very small activation veto`

而不是：

- `activation-driven selector`

### 15.6 当前阶段总结

到这一刻为止，可以把阶段结论总结成：

1. `nad_mixed_v2_aime_top2_gap1e3_logprob` 是当前最优参考方法  
2. `Version A / loose` 这条线没有超过它  
3. activation signal 有现象、有解释价值，但当前更适合做边裁，不适合做主裁判  
4. 后续工作应优先围绕最强 baseline 做更小、更稳的改动，而不是再单独扩展 `medoid` 主线

---

## 16. 新一轮小步实验：mixed_v3 = logprob baseline + tail warning veto

### 16.1 方案名

- `mixed_v3 = top2 + gap=1e-3 + logprob + tail warning veto`

### 16.2 为什么从这里继续，而不是继续做 `medoid`

上一轮 leaderboard 反馈已经说明：

- `nad_mixed_v2_aime_top2_gap1e3_logprob` 仍然是当前最强参考方法
- activation signal 更适合作为辅助 warning，而不是主 tie-break

因此这轮不再以 `medoid` 为主干，而是直接挂在当前最强 baseline 上。

### 16.3 核心假设

当前假设是：

- 在 `logprob mixed_v2` 已经判成 top1 的候选里
- 有极少数 case 其实带有明显的尾段 warning
- 如果 top2 候选没有这个 warning，而且尾段更健康
- 那么可以在 very small 的范围内做一次 veto

换句话说：

- 主裁判仍然是 `logprob mixed_v2`
- activation 只在极少数 case 上当边裁

### 16.4 改动方式

这轮新增了一个 submit-safe 构造脚本：

- `scripts/build_mixed_v3_logprob_tail_veto.py`

输入基底是：

- `result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json`

输出文件是：

- `result/best_of_n_nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto_submit.json`
- `result/best_of_n_nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto_submit_notes.json`

### 16.5 当前规则（非常保守）

当前版本的规则是：

1. 仍然只看：
   - `DS-R1/aime24`
   - `DS-R1/aime25`
   - `Qwen3-4B/aime24`
   - `Qwen3-4B/aime25`

2. 仍然只在 `top1-top2 <= 1e-3` 时考虑触发  

3. 只比较当前 submission score 的 top2 候选  

4. 只有在满足下面全部条件时才翻转：
   - 当前 top1 被 `tail warning` 命中
   - top2 没被命中
   - top2 的尾段健康度明显更好

这意味着：

- 这轮不是让 activation 重新选一遍
- 而是在当前最强 baseline 上做 very small veto

### 16.6 先做触发性诊断

在真正构造 submission 之前，先检查了当前最强 `logprob` baseline 里，warning 到底能不能碰到 top2 决策边界。

结果是：

- `DS-R1/aime24`：有 `3` 道题可触发
- `DS-R1/aime25`：`0` 道
- `Qwen3-4B/aime24`：`0` 道
- `Qwen3-4B/aime25`：`0` 道

可触发的 `DS-R1/aime24` 题号是：

- `62`
- `80`
- `85`

这说明：

- 把 activation warning 挂在 `logprob mixed_v2` 上以后
- 它终于能碰到少量真正处在 top2 边界上的题
- 比挂在 `medoid top2` 上更接近“可用的小修正”

### 16.7 本地结果

对可触发题的本地 correctness 做了检查：

- `62`：原本错 -> 新版仍错
- `80`：原本错 -> 新版改成对
- `85`：原本错 -> 新版仍错

因此在 `DS-R1/aime24` 上：

- baseline：`21/30`
- mixed_v3：`22/30`

在 `DS-R1/aime25` 上：

- baseline：`24/30`
- mixed_v3：`24/30`

合并看 `DS-R1/aime24 + aime25`：

- baseline：`45/60`
- mixed_v3：`46/60`

### 16.8 改动范围

当前生成的 `mixed_v3` submission 文件里：

- `DS-R1/aime24`：改动 `3/30`
- `DS-R1/aime25`：改动 `0/30`
- `Qwen3-4B/aime24`：改动 `0/30`
- `Qwen3-4B/aime25`：改动 `0/30`

也就是说这轮确实满足了当前原则：

- 改动非常小
- 触发范围很窄
- 没有把 submission 整体搅动太大

### 16.9 当前限制

当前还不能对 Qwen 两个 cache 做同样的本地 correctness 检查，原因是：

- 本地缺少对应 cache 的 `evaluation_report_compact.json` / `evaluation_report.json`

所以目前本地可验证收益只来自：

- `DS-R1/aime24`
- `DS-R1/aime25`

### 16.10 当前结论

目前可以把 mixed_v3 的判断写成：

1. 这次 activation signal 终于不是“完全碰不到决策边界”了  
2. 挂在 `logprob mixed_v2` 上，比挂在 `medoid` 上更有希望  
3. 当前版本改动仅 `3` 道题，且本地至少带来 `+1` 的净增益  
4. 因此这版已经比之前的 `medoid` 线更值得考虑继续提交验证  

### 16.11 当前产物

当前已经生成的候选提交文件：

- `result/best_of_n_nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto_submit.json`
- `result/best_of_n_nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto_submit_notes.json`

如果要继续推进，当前最自然的动作就是：

- 直接把这版提交到 leaderboard 看真实反馈

---

## 17. mixed_v3 的 leaderboard 反馈

### 17.1 外部结果

用户已经提交：

- `result/best_of_n_nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto_submit.json`

反馈结果是：

- 这版 **不好**
- 并且让 **平均排名下降** 了

### 17.2 这条结果说明什么

这说明当前这版：

- `top2 + gap=1e-3 + logprob + tail warning veto`

虽然在本地 `DS-R1/aime24` 上看起来有 `+1` 的小收益，
但这个局部收益没有转化成 leaderboard 的整体收益，反而伤到了平均排名。

因此当前可以明确更新判断为：

1. `mixed_v3` 这条线 **暂时不成立**  
2. activation warning 挂在当前最强 `logprob` baseline 上，也仍然不够稳  
3. 即使触发范围很小，也依然可能带来负面外部效果  

### 17.3 对 activation 融合路线的进一步判断

到这一步，可以把 activation 融合路线的结论再收紧一点：

- activation signal 可能有解释价值
- 也可能在个别题上看起来“像有帮助”
- 但只要把它真正接到 submission 里，当前还是容易伤平均排名

所以当前不应再继续默认：

- “只要把 activation 挂在更强 baseline 上，就会稳定变好”

目前的真实结论更接近：

- activation signal 目前更像分析工具
- 还不像一个已经成熟、可稳定上线的 submission signal

### 17.4 当前最强方法的状态

截至当前，最稳、最好的方法仍然是：

- `nad_mixed_v2_aime_top2_gap1e3_logprob`

因此当前应该把它视为：

- 默认主干
- 当前最强 baseline
- 后续一切改动的参考起点

### 17.5 当前阶段的工作建议

基于这次失败反馈，当前最合理的动作是：

1. 暂停继续把 activation 接到 submission 上  
2. 暂停沿着 `tail warning veto` 这条线继续做更多上线版变体  
3. 先回到 `nad_mixed_v2_aime_top2_gap1e3_logprob` 作为稳定主干  
4. 后续如果继续实验，应优先考虑：
   - 更稳的非 activation 小修正
   - 或只做离线分析，不直接接 submission

### 17.6 当前结论

到这一刻为止，可以把结论写得更直接：

- `mixed_v3` 已被 leaderboard 否定
- `activation -> submission rule` 这条线目前没有证据表明值得继续主推
- 当前应回到：
  - `nad_mixed_v2_aime_top2_gap1e3_logprob`
  作为最优参考方案

