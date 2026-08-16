# Iteration log

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
