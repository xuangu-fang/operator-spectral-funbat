# 两周投稿计划：叙事、主实验与止损点

> 版本：2026-08-19。本文是**决策文档**，不是结果文档。所有数字若出现，均标注来源；探针结果标为探针。

---

## 1. 核心战略判断：把定位从「模型」改成「先验」

当前所有麻烦都源于一个隐含定位——把 Operator-Spectral FunBaT 当成一个**与 FunBaT 竞争的模型**。这带来两个无法在两周内解决的问题：

1. 我们的实现是周期 Fourier + rank 固定的 functional CP，模型容量远不如成熟的 functional tensor / neural operator，正面比容易输；
2. 输了之后没有退路，因为贡献和模型绑死了。

**改成先验定位后，这两个问题同时消失：**

> **贡献 = 一套从 PDE 形式（不需要系数）导出的、逐 mode 的合法 GP kernel，可以直接替换任何 functional tensor 模型里的通用 kernel。**

于是 FunBaT 不再是竞争者，而是**宿主**。主张变成配对比较：

$$
\text{FunBaT} + \text{operator kernels} \;>\; \text{FunBaT} + \text{generic kernels}
$$

这个比较：同模型、同容量、同优化预算，唯一变量是 kernel 来源。**它是我们已经在 POC 里 5/5 验证过的那个比较（P5）在真实数据上的翻版**，而不是一个全新赌注。

对应地，论文的一句话摘要是：

> 只要你知道场大致服从哪个方程（**不需要知道系数**），就能把该算子的联合谱非负投影成逐维合法 GP kernel，直接替换 functional tensor 模型中的通用 kernel，在极稀疏观测下获得稳定增益。

---

## 2. 数据策略：先筛后投，并把筛选表写进论文

### 2.0 原则

**不在任何单个数据集上死磕。** 主故事应在多个数据集上尝试，取其中匹配机制的。任何方法都不是万能的——但"选取有用的"必须以**预先声明的筛选判据 + 公开被拒项**来实现，否则就是 cherry pick。

因此筛选表（含被拒数据集与拒绝理由）**原样进论文**，作为适用性刻画的一部分。审稿人看到的是"作者知道自己的方法什么时候不管用"。

### 2.1 预先声明的筛选判据

对每个候选数据集构造一个候选三阶张量，在**完全观测**下计算：

| 判据 | 定义 | 阈值 | 为什么 |
|---|---|---|---|
| **S1 低秩可行性** | 满观测下 CP rank-10 相对误差 | $\le 0.35$ | 满观测都拟合不了，则 1–5% 观测对**任何**低秩方法都不可行，全体落在 NRMSE≈1，比较无意义 |
| **S2 非平凡性** | 至少一个 mode 的 rank95 | $\ge 3$ | rank-1 任务谁都能解，没有区分度 |
| **S3 机制匹配** | 算子形式已知，且符号 even / 逐轴可分 | 定性 | 实 Fourier 特征只能表示逐轴 even 分量（见技术报告 §6.1） |
| **S4 周期性** | 边界是否周期 | 加分项 | 周期 Fourier 先验无边界失配 |

S1/S2 是**便宜且事前**的：一次满观测 CP 拟合，几分钟。S3 由方程形式判定，不需要跑实验。

### 2.2 筛选结果（2026-08-19，本地数据）

| 数据 / 候选张量 | mode rank95 | CP r10 | 判定 | 理由 |
|---|---|---:|---|---|
| Kolmogorov Re40 `[t,x,y]` stride 8 | 20/10/10 | 0.602 | **拒** | S1。湍流谱宽，算子不压制高频 |
| Kolmogorov Re40 stride 1 | 6/9/8 | 0.358 | 边缘 | S1 临界 |
| KS `[traj=32, t, x]` | 24/9/8 | 0.628 | **拒** | S1。轨迹维互相独立 $\Rightarrow$ 高秩 |
| KS `[traj=8, t, x]` | 8/9/7 | 0.473 | 边缘 | S1 |
| CFDBench cavity 单 case `[t,x,y]` | 1/2/2 | 0.003 | **拒** | S2。定常流，rank-1 |
| **CFDBench cavity `[case=116, x, y]`** | **7/6/4** | **0.236** | **收** | S1+S2 通过 |
| CFDBench cavity `[case, t, xy]` | 7/1/7 | 0.195 | 收（次选） | t 维 rank-1，信息量低 |
| The Well active matter | 待筛 | 待筛 | 待筛 | **x/y 周期**（S4 加分），但输运主导，S3 存疑 |
| PDEBench 2D diff-react | 待筛（下载中） | 待筛 | 待筛 | S3 匹配度最高 |

**所有拒绝理由都是机制性的、可事前判定的，不是"跑了效果不好"。** 这是这张表能进论文的前提。

### 2.3 承诺规则

- 通过 S1+S2+S3 的数据集，按 S3 匹配度排序，**投入前 2–3 个**做主实验；
- 边缘项（Kolmogorov stride 1、KS traj=8）**只作为 E4 适用性边界的数据点**，不做主表；
- 被拒项在论文中以筛选表形式出现，**Kolmogorov 额外给出完整 rank 曲线**作为定量的失效证据；
- 任何数据集在 G1/G2 止损点未过，**立即换下一个，不加预算**。

---

## 3. 各数据集的机制判断（含一个被探针否掉的主选项）

### 3.1 被否掉：Kolmogorov 湍流作为主数据

探针（`2D_NS_Re40.npy`，单轨迹 `[32帧, 32, 32]`，**完全观测**）：

| CP rank | 相对误差（stride 8，时间解相关） | 相对误差（stride 1，时间相干） |
|---:|---:|---:|
| 2 | 0.870 | 0.729 |
| 5 | 0.713 | 0.490 |
| 10 | 0.602 | 0.358 |
| 20 | 0.466 | 0.242 |
| 40 | 0.298 | — |

**满观测下 rank-40 才到 0.30**，则 1–5% 观测对任何低秩方法都不可行，所有方法会一起落在 NRMSE≈1——这是 `SHARED_PROTOCOL.md` §6 明令拒绝的情形。

这不是偶然而是机制：**湍流谱宽，算子几乎不压制高频**，因此场既不低秩、算子先验也不含信息。我们的方法只在**耗散主导、算子压掉大部分谱**时才有增益。

> **但这个否定结果要写进论文。** 它正是审稿人会问的"什么时候不管用"的答案，而且是量化的。见 §3 的 E4。

### 3.2 主数据：PDEBench 2D diffusion-reaction（需下载，DaRUS 可达）

选它的理由是机制匹配，不是流行：

- **耗散主导**，算子强烈压制高频 $\Rightarrow$ 场低秩且先验含信息；
- 方程形式公开、生成系数（$D_u, D_v$）已知 $\Rightarrow$ **K2/K1 档位可以干净地做**；
- 是神经算子文献的主流 benchmark，有官方 baseline；
- 真值**不来自我们的 atom family**，这是 `DATASETS_AND_RESOURCES.md` 列的头号 gate。

### 3.3 备份数据：CFDBench cavity（已在本地，零下载风险）

探针（`cavity_target_functional/train.npz`，`fields (25, 16, 64, 64, 2)`，`reynolds` 10–20000）：

| 张量 | mode rank95 | CP rank 5 | CP rank 10 |
|---|---|---:|---:|
| 单 case `[t=16, x, y]` | 1 / 2 / 2 | 0.011 | 0.003 |
| `[Re=25, x=32, y=32]`（25-case split） | 3 / 4 / 3 | 0.202 | 0.117 |
| **`[case=116, x=32, y=32]`（116-case split）** | **7 / 6 / 4** | 0.339 | **0.236** |

单 case 是 rank-1（定常流），**太容易、没有区分度，不用**。`[case, x, y]` 落在有意义区间，且是一个**「物理参数 × 空间 × 空间」的多面张量**——正好对应"知道形式、参数未知"。优先用 **116-case** 版本，参数维统计更可靠。

> **注意**：`prepared/` 下的 split 属于 ai-physical-dynamics 项目的实验设计（含 `mechanisms` 标签与 sealed_test）。我们必须**从 `extracted/` 自建 split**，不复用也不消费别人的封存测试集。

**它的诚实弱点**：物理只给空间两个 mode 的谱，Re 那一维没有 PDE 导出的频谱，只能用通用 kernel。这反而是个可写的"部分知识"设定，但不能包装成全知识。

---

## 4. 主实验矩阵（4 个实验 + 消融）

| 编号 | 实验 | 回答什么 | 数据 |
|---|---|---|---|
| **E1** | 稀疏补全主表 vs baselines，观测率 1%/2%/5% | 方法在真实耗散 PDE 上是否有效 | PDEBench diff-react |
| **E2** | **drop-in 配对**：FunBaT + operator kernels vs FunBaT + generic kernels | **核心贡献**——先验是否可迁移到别人的模型 | PDEBench + CFDBench |
| **E3** | 部分知识设定：`[Re, x, y]`，物理只覆盖空间 mode | 只有部分 mode 有物理时是否仍有增益 | CFDBench cavity |
| **E4** | **适用性边界**：谱宽度 vs 低秩性 vs 增益 | 什么时候**不**管用（含 Kolmogorov 负结果） | Kolmogorov + 前三者 |

**E2 是论文的核心**，不是 E1。E1 只需要证明我们不是在一个所有人都失败的任务上比较。

### Baselines（按优先级，做不完就砍尾部）

| 层级 | 方法 | 回答 |
|---|---|---|
| 必须 | 全局/逐 mode 均值、最近邻、线性/RBF 插值 | 是否学到任何结构；增益是否只是局部平滑 |
| 必须 | 离散 CP / Tucker（EM-ALS 补缺失） | 连续函数因子是否必要 |
| 必须 | **本方法的 generic-kernel 版本（参数匹配）** | 增益是否来自物理来源 |
| 必须 | **FunBaT（官方实现）** | 与成熟 functional tensor 的关系 |
| 尽量 | Fourier-feature MLP / INR | 显式低秩是否有价值 |
| 可选 | 学习型 RBF/Matérn GP 插值 | tensor sharing 是否优于单纯 kernel 平滑 |

所有方法共享 mask、噪声、归一化、seeds；主表必须并列 **可训练参数量** 与 **wall-clock**。

### UQ 表（差异化项）

多数 baseline 只给点预测。我们报 posterior predictive coverage / NLL / interval width，是一个低成本的加分项，且代码已经有。

---

## 5. 十四天排期与三个止损点

| 天 | 内容 | 产出 |
|---|---|---|
| 1 | 启动 PDEBench 下载（后台）；CFDBench 补全任务 pipeline（loader、mask 协议、split） | 可跑的 E3 骨架 |
| 2 | 本方法 + trivial baselines 端到端跑通 | **G1** |
| 3–4 | PDEBench pipeline；从已知方程形式构造算子谱；K2/K1 bank | 早期 3-seed 结果 |
| 5 | E1 早期 gate | **G2** |
| 6–7 | FunBaT 接入（clone 官方实现，替换其 kernel 构造） | E2 骨架 |
| 8 | 离散 CP/Tucker、INR、RBF baselines | **G3** |
| 9–11 | 主表：2 数据 × 3 观测率 × 5 seeds，paired 统计；UQ 表 | E1/E2/E3 完成 |
| 12 | 消融（知识档位、wrong family、floor、routing、分离 rank）——**全部复用已有 POC 代码** | 消融表 |
| 13 | E4 适用性边界图（谱宽度 vs 增益，含 Kolmogorov 负点） | E4 完成 |
| 14 | 缓冲 + 写作 | — |

### 止损点（避免沉没成本）

- **G1（第 2 天）**：CFDBench 上本方法 NRMSE ≤ 0.8 且相对最强插值 baseline 的 MSE skill ≥ 20%。不过 $\Rightarrow$ 张量语义选错，立即换（例如改用 `[case, t, 空间]`），不加预算。
- **G2（第 5 天）**：PDEBench 上 operator bank 在 ≥4/5 seeds 上胜过**参数匹配的 generic bank**。不过 $\Rightarrow$ 核心主张不迁移，退回"带物理初始化的 spectral-mixture 先验"这一更弱定位，并把 E2 提为唯一主实验。
- **G3（第 8 天）**：FunBaT 接入成功。若 FunBaT 单体大幅优于我们的 CP $\Rightarrow$ **这正是先验定位的价值所在**，主表改以 E2 的配对比较为中心，我们自己的 CP 降为消融载体。

---

## 6. 降级为消融的内容（不再是主线）

以下全部有现成代码与结果，**只进消融表，不占正文篇幅**：

- 知识档位 K2/K1/K0/K−1 与宽度扫描（`results/knowledge_ladder_development/`）；
- wrong-family 对照（advection / wave）；
- 固定 generic support floor 与 strict support deletion；
- routing 粒度 global / per-mode / per-mode-rank；
- 非负分离 rank 曲线、signed vs octant 审计；
- collapsed vs expanded 参数化公平性。

**理由**：这些回答的是"机制为什么成立"，而审稿人先问的是"在真实数据上是否有效、比谁强"。顺序必须倒过来。

---

## 7. 已知风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 周期 Fourier 先验 vs 非周期边界（PDEBench 为 no-flow，cavity 为壁面） | 边界区误差偏大 | 所有谱方法共用同一 basis，比较仍公平；**单独报告边界区误差**，并写进 limitation |
| tensor rank 需从 2 提到 10–20 | 计算增加，但仍小 | 先做 rank 扫描确定，冻结后不再调 |
| CFDBench 只有 25 个 case | Re mode 统计弱 | 只作次要实验；不在其上做主张 |
| PDEBench 下载耗时 | 挤压排期 | 第 1 天即启动后台下载；CFDBench 为零风险备份 |
| FunBaT 官方实现接入成本 | 可能吃掉 2 天 | G3 设在第 8 天；接不通则退为"我们自己的 generic-kernel 版本"作为配对宿主 |

---

## 8. 一句话总结

**不再打磨机制 POC。** 把已有的正面证据（P5：同预算下 operator kernel 逐 seed 胜过通用 kernel）搬到耗散主导的真实 PDE 数据上，以**"先验可 drop-in 到别人模型"**为核心主张，用一张诚实的适用性边界图（含湍流负结果）划定范围。
