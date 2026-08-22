# 投稿计划：一句话主线与最简主实验

> 版本：2026-08-19（第三版）。本文是**决策文档**。
> **⚠ 数据决策部分已被实验推翻并替换**：计划里的主选（PDEBench 2D diff-react）与备选（CFDBench cavity）都未通过低秩可行性判据，最终采用**独立数值求解器生成的随机强迫线性 PDE 场**。当前有效的结果见 [`RESULTS_LOG_ZH.md`](RESULTS_LOG_ZH.md)，formulation 修订见 [`PAPER_TECHNICAL_REPORT_ZH.md` 第 14 节](PAPER_TECHNICAL_REPORT_ZH.md#14-formulation-修订记录2026-08-19)。
>
> **计划中被证明正确的部分**：先筛后投的判据（S1/S2/S3）确实拦下了两个会浪费一周的数据集；把贡献定位为"先验而非模型"确实让 formulation 可以随数据修订而叙事不变。
> **计划中被证明过于乐观的部分**：假定存在一个"既真实、又低秩、又有已知算子形式"的现成数据集。三个候选全部落空——真实场要么高秩（湍流、图灵斑图），要么平凡（定常腔体流 rank-1）。

---

## 1. 一句话主线

> **复杂物理场的稀疏重构。除了稀疏观测点，我们唯一额外使用的信息是这个场所服从的 PDE 的形式；从这个形式里挖出谱，构造张量分解各维度的先验 kernel。**

论文只需要证明一件事：

$$
\mathcal{L} \;\longrightarrow\; S_{\mathrm{op}} \;\longrightarrow\; \lbrace k_d \rbrace_{d=1}^{D}\;\longrightarrow\;\hat{\mathcal{X}}
$$

**不需要**证明它对任意边界、任意高频、任意算子族都成立。那些是 ablation，不是主线。

### 1.1 定位：贡献是「先验」，不是「模型」

把自己当成与 FunBaT 竞争的模型，两周内赢不了（周期 Fourier、固定 rank 的 CP 容量不足），而且输了没有退路。改成先验定位后：

> **贡献 = 一套从 PDE 形式导出的逐 mode 合法 GP kernel，可直接替换任何 functional tensor 模型里的通用 kernel。**

于是唯一变量是 **kernel 的来源**，模型、容量、优化预算全部相同。这也正是我们已经在合成 POC 上 5/5 验证过的那个比较（P5）在真实数据上的版本，不是新赌注。

---

## 2. 最简主实验（E1）

### 2.1 setting

| 项 | 选择 | 为什么是最简的 |
|---|---|---|
| 数据 | PDEBench **2D diffusion-reaction** | 三个 mode **全是 PDE 坐标** $(t,x,y)$，每一维的 kernel 都直接来自方程，无需解释"部分知识" |
| 张量 | 单个样本 $\to$ `[t, x, y]` | 不需要跨样本/跨参数的复杂张量语义 |
| 任务 | 随机观测 $p\%$ 的 entry，重构其余 | 最标准的 tensor completion，人人看得懂 |
| 观测率 | 1% / 2% / 5% | 一条曲线足够 |
| 指标 | held-out NRMSE | 单一主指标 |

选 diffusion-reaction 的理由是**机制**：耗散主导 $\Rightarrow$ 算子强烈压制高频 $\Rightarrow$ 场低秩**且**先验含信息。这两件事是同一个原因，正是我们方法起作用的条件。

### 2.2 对比方法

| 方法 | 作用 |
|---|---|
| **本方法**：kernel 由 diffusion-reaction 方程形式导出 | 主张 |
| **同模型 + 通用 kernel（参数完全匹配）** | **最关键的对照**——增益是否来自 PDE 形式 |
| 全局/逐 mode 均值、最近邻、RBF 插值 | 是否学到结构；是否只是局部平滑 |
| 离散 CP / Tucker（EM-ALS 补缺失） | 连续函数因子是否必要 |
| FunBaT（官方实现） | 与成熟 functional tensor 的关系 |

主表并列 **可训练参数量** 与 **wall-clock**。所有方法共享 mask、噪声、归一化、seeds。

### 2.3 支撑实验（E2，锦上添花，不是必需）

**drop-in 配对**：FunBaT + 我们的 kernel vs FunBaT + 通用 kernel。它把"先验可迁移"这一定位坐实。若第 8 天接不通 FunBaT，则退回用我们自己的模型做同一配对（E1 里已经有），主线不受影响。

---

## 3. 全部降级为 ablation 的内容

以下**都有现成代码和结果**，只进消融表：

- 系数敏感性（知识档位 K2/K1/K0、宽度扫描）——已完成，结论是**只需形式、不需系数**；
- 错误算子族（advection / wave）与谱距离风险轴；
- 固定 generic support floor 与 strict support deletion；
- routing 粒度、非负分离 rank、signed vs octant；
- **边界泛用性**（周期先验用在非周期域上的代价）；
- **高频/宽谱泛用性**（湍流失效案例）；
- collapsed vs expanded 参数化公平性。

**理由**：审稿人先问"在真实数据上是否有效、比谁强"，再问"为什么成立、什么时候不成立"。顺序不能倒。

---

## 4. 数据筛选（附录材料，非主线）

### 4.1 原则

不在任何单个数据集上死磕。主故事在多个数据集上尝试，取匹配机制的。但"选取有用的"必须以**预先声明的判据 + 公开被拒项**实现，否则是 cherry pick。筛选表连同被拒数据集**原样进附录**，作为适用性刻画。

### 4.2 预先声明的判据（都在投入之前）

| 判据 | 定义 | 阈值 |
|---|---|---|
| **S1 低秩可行性** | 满观测下 CP rank-10 相对误差 | $\le 0.35$ |
| **S2 非平凡性** | 至少一个 mode 的 rank95 | $\ge 3$ |
| **S3 机制匹配** | 算子形式已知，符号逐轴 even | 定性 |

S1 不过 $\Rightarrow$ 稀疏补全对**任何**低秩方法都不可行，全体落在 NRMSE≈1，比较无意义。S2 不过 $\Rightarrow$ rank-1 任务，无区分度。

### 4.3 筛选结果（2026-08-19，探针）

| 数据 / 候选张量 | mode rank95 | 满观测 CP r10 | 判定 | 理由 |
|---|---|---:|---|---|
| Kolmogorov Re40 `[t,x,y]` stride 8 | 20/10/10 | 0.602 | **拒** | S1；湍流谱宽，算子不压制高频 |
| Kolmogorov Re40 stride 1 | 6/9/8 | 0.358 | 边缘 | S1 临界 |
| KS `[traj=32, t, x]` | 24/9/8 | 0.628 | **拒** | S1；轨迹互相独立 $\Rightarrow$ 高秩 |
| KS `[traj=8, t, x]` | 8/9/7 | 0.473 | 边缘 | S1 |
| CFDBench cavity 单 case `[t,x,y]` | 1/2/2 | 0.003 | **拒** | S2；定常流，rank-1 |
| CFDBench cavity `[case=116, x, y]` | 7/6/4 | 0.236 | 备选 | 通过，但 case 维不是 PDE 坐标 |
| **PDEBench 2D diff-react `[t,x,y]`** | 待筛 | 待筛 | **主选** | S3 匹配度最高；三维全是 PDE 坐标 |
| The Well active matter | 待筛 | 待筛 | 待筛 | x/y 周期，但输运主导，S3 存疑 |

被拒理由全部是**机制性的、事前可判的**，不是"跑了效果不好"。这是这张表能进论文的前提。

> Kolmogorov 的完整 rank 曲线（满观测 rank-40 仍为 0.298）作为**定量失效证据**进 ablation，回答"什么时候不管用"。

---

## 5. 排期与止损点

| 天 | 内容 |
|---|---|
| 1 | PDEBench 下载（已启动）；通用稀疏补全 harness（张量 + mask + 方法 runner） |
| 2 | 筛 PDEBench；本方法 + 通用 kernel 对照 + trivial baselines 跑通 $\Rightarrow$ **G1** |
| 3–5 | 主表：3 观测率 × 5 seeds；离散 CP/Tucker、RBF baselines $\Rightarrow$ **G2** |
| 6–8 | FunBaT 接入（E2） $\Rightarrow$ **G3** |
| 9–11 | 主表定稿、UQ 表、图 |
| 12–13 | ablation（全部复用已有代码） |
| 14 | 缓冲 + 写作 |

- **G1（D2）**：本方法 NRMSE $\le 0.8$ 且相对最强插值 baseline 的 MSE skill $\ge 20\%$。不过 $\Rightarrow$ 换数据/换张量语义，不加预算。
- **G2（D5）**：本方法在 $\ge 4/5$ seeds 上胜过**参数匹配的通用 kernel**。**这是全文唯一不能失败的比较。** 不过 $\Rightarrow$ 换下一个通过 S1–S3 的数据集重试一次；再不过则主张不成立。
- **G3（D8）**：FunBaT 接通。接不通 $\Rightarrow$ E2 用自有模型完成，不阻塞主线。

---

## 6. 已知风险

| 风险 | 缓解 |
|---|---|
| 周期 Fourier 先验 vs 非周期边界 | 所有谱方法共用同一 basis，比较仍公平；边界泛用性移到 ablation，**不在主线上承诺** |
| tensor rank 需从 2 提到 10–20 | 先做一次 rank 扫描确定，冻结后不再调 |
| 单一数据集风险 | CFDBench cavity 为已通过 S1/S2 的本地备份，零下载风险 |

---

## 7. 一句话总结

**不再打磨机制 POC，也不追求任何泛用性。** 用最简单的 `[t,x,y]` 稀疏补全 setting，在耗散主导的真实 PDE 数据上证明一件事：**kernel 来自 PDE 形式，比来自通用字典更好**。其余全部是 ablation。
