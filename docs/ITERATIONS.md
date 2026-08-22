# Iteration log

<!-- STATUS BANNER -- keep at the top -->
> ## ⚠ 历史档案，不是当前口径
>
> **当前有效的技术口径见 [`HANDOVER_TECHNICAL_ZH.md`](HANDOVER_TECHNICAL_ZH.md)。**
>
> 本文写于主线转向**受限传感器布局**之前，其中的结论有一部分已被后续实验推翻：
>
> | 本文可能出现的说法 | 现状 |
> |---|---|
> | 随机掩码下相对通用字典 **+5.5% / +8.0%** | **打平**。该 margin 建立在只扫了 3 个长度尺度的 Matérn 上；网格放宽后 baseline 自己的最优在 $\ell = 2.4$ |
> | 单面墙 **+17.2%** | **已撤回**，同一原因。修正后 0.5448 vs 0.5449 |
> | "只需要方程的形式" | 不准确。主表用的是**单个点估计（错 1.5 倍）走 K2 的机器**，真正的 K1（范围 + 池化）见交接文稿 §2.7 |
> | 主指标是随机掩码下的重构 | 主指标已改为**受限布局**下的重构，核心量是**调参代价** |
>
> 保留本文是为了记录推导与失败路径，**不要从这里取数字**。


## R10 — 知识档位 development sweep：三条预注册预测被否证（2026-08-19）

- 环境重建后先做保真度检查：重跑已发表的 anisotropic diffusion seed 201，与存档 JSON 逐方法最大差 `1.1e-07`，分离误差完全一致，故新旧数字可比。
- 主线各向异性扩散、dev seeds 101–105、2% 观测、400 steps。所有方法变分系数恒为 156；K1 与 atom 数匹配 generic 同为 231 个可训练参数。
- **P2 通过（强）**：K1 保留 K2 收益的 100.7%（换稳定基线重算 103%）。只知方程形式、不知系数，相对知道系数没有损失。
- **P5 通过 5/5**：同 12 atoms、同参数量，operator 池化 bank 逐 seed 胜过通用字典（`0.0438` vs `0.0509`），且其 bank 冗余得多（两两 cosine 0.933 vs 0.614）。目前唯一干净支撑「物理来源有用」的证据。
- **P6 名义通过 4/5 但很弱**：advection 池化 `0.0469`，仅差 0.0031，仍优于 generic。收益主要来自「算子形状的低通谱」而非「正确的算子」。
- **P3 否证，且预测本身写错**：floor 在每档都是纯代价（+0.018/+0.018/+0.007），且档位越低代价越小。知识档位下降不制造 support 缺失，必须与删频轴交叉才能检验 floor。
- **P4 初判失败后被 width sweep 推翻**：K0 的 `0.0550` 是一次不走运的 LHS 抽样；$\times1.5\to\times20$ 的池化 bank 平坦在 `0.0437–0.0446`，退化边界在 ×20 内没有出现。教训：bank 本身是随机对象，单一 LHS seed 的抽样方差可与被测效应同量级。
- **P1 按预注册容差通过，但脚本判据写严了**（用了严格 `<=`，忽略了预注册的「允许 K2 ≈ K1」），输出 `false`。记录在案，不事后改判据。
- **未预料 (a)**：`K1-single`（单个错误系数、4 atoms）`0.0440`，与 6 组池化、与真系数完全打平。池化对均值无贡献；width sweep 显示它买的是方差（池化 std 恒 ~0.0078，单点 0.0073–0.0335，两端均值劣化）。「routing 做软参数推断」的叙事应删除。
- **未预料 (b)**：真正的风险轴是算子族的谱距离，不是系数误差。`K1 0.0438 < advection 0.0469 < generic 0.0509 < wave 0.0544`——谱上遥远的错误算子比不用物理还差。这才是 floor 最该保的场景。
- **未预料 (c)**：a-priori 判据中 reachability 无效（Spearman 0.28），prior concentration 有效（0.818）。可达性只度量表达力；好先验必须既近又紧。n=11，启发性而非已确立。
- 代码：新增 `extended_generic_dictionary`（前 4 个与冻结版逐位相同）、暴露 wave 族参数（默认值复现原 literals，有测试保证）。测试 11→13 全过。
- 定位收缩：K1 不需要参数族池化机器，正文改为「代入任一合理系数」，池化降为可选方差控制。

## 结构重构（非实验轮次）— 知识档位 setting 与 Method 瘦身（2026-08-19）

- 没有跑任何新实验；全部数值仍是 2026-08-16 冻结的 K2 档位确认。
- **Setting 改写**：从"假定算子已知"改为**算子知识档位** K2/K1/K0/K−1。K1（只知方程形式、系数未知）被定为论文主打档位；现有全部结果被重新标注为 K2，即最不现实的一档。
- **Method 瘦身**：8 个小节压到 2 个组件 M1（算子知识→合法逐维 kernel bank）与 M2（mixture + support floor）。routing 粒度降为消融，collapsed 参数化移入附录 B（公平性控制），数值细节与 basis bug 移入附录 C。
- **主线锁定各向异性扩散/反应扩散族**，K1 的未知量即未知扩散张量与反应系数；数学理由是它的 symbol 由各轴平方项构成，与实轴向 even/separable 表示严格匹配（full signed rank-4 误差 0.0043，advection 为 0.18）。
- **新主实验设计（§10，待执行）**：横轴为知识档位的降级曲线，替代原先"matched / 删频"两个人工端点；含 6 条预注册可证伪预测（P1–P6），其中 P4/P5/P6 最可能失败也最有信息量；新增两个关键公平性对照——atom 数匹配的 generic bank、wrong-family 池化 bank。
- **seed 纪律**：201–205 已随文稿公开，视为已看过；新主实验须用全新未接触的 301–305。
- LaTeX 骨架（`paper/`）同步重构为相同框架，含 M1/M2、知识档位表与 P1–P6。

## 工程记录（非实验轮次）— 文稿与 LaTeX 基础设施（2026-08-19）

- 没有重跑任何实验，没有改动 `src/` 或 `experiments/`；所有数值口径仍是 2026-08-16 投稿确认版。
- 重写 `docs/PAPER_TECHNICAL_REPORT_ZH.md`：补符号表、把三个算子 symbol 表格化、把 signed-vs-octant 审计和 paired-win 汇总提升为独立小节、新增附录 A 完整数值表（含 spectrum cosine / rel. L2 / coverage / NLL）。
- 修复公式渲染：两份技术报告的 `\[ \]` / `\( \)` 全部改为 GitHub 支持的 `$$ $$` / `$ $`，`\rm` 统一为 `\mathrm`，math 内的中文字符移出。全部公式已用 pdflatex 编译验证通过。
- 本机安装 TeX Live（latexmk / pdflatex / xelatex / biber）与 CJK 字体；新增 `paper/` LaTeX 工程（`make` 编译论文，`make report-zh` 把中文报告渲染成 PDF）。

## R9 — atom-independent Fourier basis 与 strict-support 最终确认（2026-08-16）

- 最终代码审查发现 collapsed 公共 Fourier basis 错误地从第一个 atom 反除得到；第一个 operator atom 被严格删频时，高频 basis 也被置零，导致 generic floor 实际无法产生高频 feature。
- 改为从坐标解析构造 `[1, sqrt(2)cos, sqrt(2)sin]` basis；新增 zero-support feature、Gram PSD、finite-gradient tests。
- 只重跑 strict-support 两个 controls；matched 201–205 主表保持冻结，不重训。
- development：wrong operator `0.6195±0.2149` → floor-robust `0.0480±0.0079`，5/5 wins。
- final reference/shifted/anisotropic：分别 `0.672→0.040`、`0.632→0.085`、`0.615→0.130`，全部 5/5 wins。
- robust 在 matched anisotropic 上从 `0.118` 变为 `0.131`，形成明确的 robustness--specificity tradeoff；恢复为 secondary contribution。

## R8 — untouched 投稿确认与主线收缩（2026-08-16）

- 冻结 proposed method 为 operator per-mode/rank；operator-global、generic-global/per-mode-rank 和 oracle 为主 baselines。
- 使用从未检查过的 seeds 201–205，固定 2% observation、rank 2、frequency 0–6、400 steps。
- 新增 shifted advection 与 strongly anisotropic diffusion；每个算子报告 nonnegative separation rank 1–6 曲线。
- robust 与 wrong-support 初始作为独立审计；R9 修复后 robust floor 通过 strict-support gate，升级为 secondary contribution。
- anisotropic diffusion：operator per-mode/rank `0.1183±0.0582`，generic per-mode/rank `0.1567±0.0990`，5/5 paired wins；operator-global `0.1212±0.0606`，说明收益主要来自 operator-derived mode kernels，per-rank routing 只有弱增益。
- reference/shifted advection 没有一致均值 winner；full signed rank-4 separation error `0.180/0.181`，远高于 positive-octant `0.032/0.024`。
- 决策：以 anisotropic diffusion/even operator spectra 为主线的条件 GO；advection 与 per-rank routing 不作为独立贡献。Robust floor 的最终判断见 R9。

## R7 — generic floor 最小重试与 provisional NO-GO（后被 R9 作废）

- uniform 8-way robust routing 在 matched development seeds 上高度 seed-unstable；这促使固定 generic floor。该轮 strict-support 结论因后发现的 basis bug 无效。
- 只做一次预声明修正：generic 总质量 floor 25%，operator-centered routing initialization；没有继续搜索 floor。
- development seeds 101–105：operator per-mode/rank `0.0396±0.0066`，generic `0.1292±0.1390`，robust `0.0467±0.0085`。
- 当时 strict wrong-support 得到 operator `0.6195±0.2149`、robust `0.6220±0.2165`；该 negative result 后被定位为 atom-derived zero-basis bug，不是科学结论。
- 修复、测试和只重跑 strict controls 的最终结论见 R9。

## R6 — collapsed spectral-mixture 公平性与数值审计（2026-08-16）

- 发现旧 expanded POC 给每个 atom 一套独立 posterior coefficients，导致 8-atom robust bank 比 4-atom baselines 参数更多。
- 新增 canonical collapsed 参数化：先形成 routed spectrum，再用单套 Fourier coefficients；所有 bank 的 coefficient posterior 固定为 156 个参数。
- 发现 strict zero support 在 `sqrt(0)` 的梯度奇异，加入数值 floor、non-finite ELBO fail-fast 和 finite-gradient 单测。
- collapsed 后一度观察到旧 escape 正信号消失；R9 证明原因是另一个 zero-basis 实现 bug，而不是公平参数化本身。
- 101–105 已因上述审计暴露，全部标为 development seeds，不进入最终确认。

## R1 — planted mode-wise kernel sanity（2026-08-15）

- 实现 nonnegative spectrum → real Fourier features → PSD kernel。
- 实现 rank-2 variational functional CP，Gaussian coefficient posterior 通过 MC ELBO+SGD 训练。
- 在 $24^3$、1%/2%/5%、3 seeds 的方法友好数据上比较 global/per-mode/per-mode-rank/oracle/swap。
- 正信号：2% oracle 0.047 vs global 0.088；swap 0.897。说明正确的逐 mode/rank prior 有实质样本效率收益。

## R2 — routing controls 与 identifiability audit（2026-08-15）

- 5% 时 learned per-mode/rank 0.0325，已追平 oracle 0.0326。
- 但 atom top-1 只有 22%--33%；不能把预测成功包装成 kernel 发现。
- 新增 induced prior spectrum cosine/L2 审计。高度相关 atoms 的标签不是可识别参数。

## R3 — operator joint spectrum 非负分离（2026-08-15）

- 实现 diffusion、advection--diffusion、damped-wave 的 $|\widehat L|^{-2}S_w$。
- rank-4 相对谱误差：diffusion 0.0028、advection 0.0325、wave 0.1079。
- 结论：谱可分离性本身应成为适用性指标；波动 dispersion surface 是明确较难/负信号。

## R4 — 简单字典与高级方案的桥（2026-08-15）

- operator atoms 与 smooth/Matérn/oscillatory/broadband atoms 合并为 robust bank。
- 新增 global + shrunk mode deviation：前100 steps global warm start，后300 steps释放 deviation，且加 Gaussian shrinkage。
- matched 2% 下 hybrid per-mode/rank 0.0651 最好；hierarchical 0.0747 方差更低但均值不优。保留为稳定 baseline，不升级为主方法。

## R5 — mismatch 与否证（2026-08-15）

- advection truth / diffusion prior 的简单幅值失配没有稳定伤害：有限 Fourier span 相同，posterior coefficient 可以补偿。这是否证了“核权重可直接识别 PDE”。
- **历史 expanded 结论：** 删除高频 support 的严格错先验使 NRMSE 升至 0.631；加入 generic dictionary 得到 0.068。
- R9 最终在 collapsed 公平 coefficient budget 下确认了相同机制，因此方向正确；投稿只使用 R9 数值，不混用 expanded 数值。

## Migration baseline — 2026-08-15

- Split from the Physics-Informed Tensor Learning Hub.
- Preserved the four-kernel dictionary, full-covariance finite-feature variational GP, ELBO+SGD, and matched/near-matched/mismatched evidence layers.
- The global dictionary is now Stage 0. The advanced POC will test operator-derived positive spectral separation and per-mode/per-rank kernel routing.
