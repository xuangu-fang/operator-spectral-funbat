# 四方向共享数据、Baseline 与测试审计协议

## 1. 先区分两个不同任务

### Task C：给定新域少量观测的 completion/adaptation

测试域提供少量 target observations，模型重建其余 entries。适合比较：

- observed mean、nearest neighbor、RBF interpolation；
- graph harmonic/GP interpolation；
- discrete CP/Tucker；
- functional CP/Tucker；
- per-instance INR；
- Bayesian posterior/UQ。

所有方法必须读取完全相同的测试域观测。该任务最能验证“低观测率是否降低过拟合”，但当前方向 3/4 的共享 POC **尚未实现这一 protocol**。

### Task O：新域零 target observation 的 operator/surrogate generalization

训练时读若干 geometry-solution pairs，测试域不提供 target。适合比较：

- joint coordinate/SDF INR；
- conditional neural field；
- functional neural CP/Tucker；
- GINO、DAFNO、Geo-FNO、Transolver；
- domain-kernel/geometry-conditioned surrogate。

当前 irregular elliptic POC 属于 Task O，但训练监督又只有每个训练张量的 1% entries，是“sparse-supervision operator learning”。大型 neural operator 在仅 4 个训练几何上通常没有合理统计规模，不能把其失败直接解释为 proposed method 优势。

两类任务必须分表报告，不得混用 observation ratio 或 baseline 排名。

## 2. 共享 dataset cards

| 数据 | 几何/物理 | 适用方向 | 当前用途 | 主要缺陷 |
|---|---|---|---|---|
| controlled operator CP/Tucker | 人工 mode operators 与可控低秩真值 | 1 | 实现与机制 sanity | simulator 与 prior 同源，不是外部证据 |
| irregular-boundary elliptic | 6 种域、2 分辨率、screened elliptic、Neumann/reflecting boundary | 1/3/4 | 孔洞、边界、跨分辨率 POC | 仅 6 个几何；forcing 与域生成均为自建；只有一个 hole test |
| randomized multi-hole elliptic | 48 train、8 ID-val、8 双孔 OOD-val、8 冻结未读 test specs | 4 | 稀疏监督下 geometry-NO/CP、topology shift | 自建 screened-elliptic；尚无外部 strong operator |
| geometry-kernel dictionary | 3 train、2 unseen val、1 frozen hole test；matched heat-GP、perturbed near-match、elliptic mismatch | 3 | kernel family recovery、ELBO evidence selection、失配边界 | matched/near-match 是方法友好 sanity，不能替代外部数据 |
| irregular-boundary wave | 同一几何族的独立数值波求解 | 2 | phase 方法独立 solver smoke | 小规模、源点和传播结构理想化 |
| independent wave smoke | 多障碍/墙体、独立 wave solver | 2 | 几何传播 sanity | 仍为自建，未等同 WaveBench |
| The Well acoustic scattering | 公共数据 | 2 | 外部失败证据 | 当前抽取任务所有方法 NRMSE≈1，已拒绝 |
| WaveBench | time-harmonic 与 time-varying wave tasks | 2 | 候选外部应用 | 必须先确认具体任务是否提供本方法需要的 source/geometry/travel-time 语义 |
| AirfRANS | 1000 个 NACA airfoil RANS simulations | 3/4 | 候选外部不规则边界数据 | 默认是 full-field surrogate，不是极低 entry completion；需另定义 sparse supervision protocol |

## 3. Baseline 必须承担的因果角色

| 类别 | 最低 baseline | 回答的问题 |
|---|---|---|
| 绝对有效性 | global/per-mode mean、persistence | 模型是否学到任何有效结构？ |
| 非参数插值 | nearest/RBF、graph harmonic | 收益是否只是局部平滑？ |
| 离散低秩 | CP、Tucker | 连续函数因子是否必要？ |
| 方法匹配连续低秩 | coordinate/SDF functional CP 与 Tucker | 收益来自 geometry、core 还是网络容量？ |
| 几何因果消融 | coordinate-only、wrong geometry、intrinsic-kernel replacement | 模型是否真的使用正确几何？ |
| GP/kernel | Euclidean RBF GP、graph/domain GP | tensor sharing 是否优于单独 kernel smoothing？ |
| 黑盒函数 | joint INR / conditional neural field | 显式低秩限制是否有价值？ |
| 强 operator | GINO/DAFNO/Geo-FNO/Transolver | 在足够训练 geometries 下是否仍具竞争力？ |

公平性至少包含两列：trainable parameters 与 wall-clock/peak memory。更关键的是输入信息预算：full fields 训练的 operator 不能与只看 1% entries 的模型混在同一“同预算”排名中。

## 4. 共享 mask taxonomy

- entry-random：仅作统计上最容易的首轮；
- fixed spatial sensors：固定节点观察全部或部分参数/时间；
- missing fibers/slices：整条 source、parameter、time fiber 缺失；
- region missing：连续空间块或孔洞 shadow 完全不可见；
- boundary-biased：传感器靠近或远离边界；
- geometry few-shot：新域提供 0、少量或完整 target observations。

每次必须同时记录 entry count、sensor count、覆盖的 modes 和是否对测试域提供观测。

## 5. 共享测试层级

1. **Unit tests**：tensor contraction、kernel sign invariance、SDF/距离特征、posterior finite；
2. **Data audits**：PDE residual、hash、shape、connectedness、hole count、train/test geometry grouping；
3. **Mechanism tests**：correct/wrong geometry、SDF/coordinate、CP/Tucker、kernel replacement；
4. **Optimization tests**：完整 observed loss checkpoint，不按单个 minibatch 选择；多 seed、rank/capacity matched；
5. **Early screening**：默认 3 seeds、300--500 updates；若绝对门槛或相对 baseline 均失败，立即停止扩预算；
6. **Statistical confirmation**：只有通过 early gate 的方案才增加独立 geometries/seeds，报告 paired effect + CI；
7. **External validation**：作者数据/官方 split/官方 baseline 实现，且先过 absolute-effect gate。

## 6. 冻结门槛

- 外部任务 proposed method macro NRMSE ≤ 0.8；
- 相对最强 trivial/interpolation baseline 的 MSE skill ≥ 20%；
- 几何主张必须在 correct-vs-wrong/erased 对照上跨 seeds 稳定；
- Tucker 主张必须超过参数匹配 CP，而不仅是超过 discrete tensor；
- Bayesian 主张必须报告 posterior predictive、NLL、coverage-width；只有 weight decay 不能叫 Bayesian inference；
- 主会候选至少覆盖 3 个数据族、多个 topology-held-out geometries 和一个官方强 baseline。

## 7. 当前统一早期预算（2026-08-15 起）

- 默认 `3 seeds × 300--500 gradient steps`；不再用 10 seeds 筛方法。
- checkpoint 只看完整 observed training loss 或预先划分的 validation，不按随机 minibatch 选择。
- 一个方法在 seed 0 已明显输给 method-matched baseline 时，只允许一次有明确机制依据的最小修正；再次失败即停止。
- test geometry、test field 和 test mask 在模型/超参数冻结前不得读取。
- 训练误差接近零但 validation NRMSE 接近或高于 1，记为稀疏过拟合，不记为“模型已收敛”。

## 8. 一手 baseline 入口

- GINO / NeuralOperator: <https://github.com/neuraloperator/neuraloperator>
- DAFNO: <https://github.com/ningliu-iga/DAFNO>
- Geo-FNO: <https://github.com/neuraloperator/Geo-FNO>
- Transolver: <https://github.com/thuml/Transolver>
- AirfRANS: <https://github.com/Extrality/airfrans_lib>
- WaveBench: <https://github.com/wavebench/wavebench>
