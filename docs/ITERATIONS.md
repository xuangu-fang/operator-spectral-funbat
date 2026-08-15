# Iteration log

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
- 删除高频 support 的严格错先验使 NRMSE 升至 0.631；加入 generic dictionary 后恢复到 0.068。
- 推荐故事收敛为：operator-derived atoms 给物理中心，generic atoms 给 misspecification escape route。

## Migration baseline — 2026-08-15

- Split from the Physics-Informed Tensor Learning Hub.
- Preserved the four-kernel dictionary, full-covariance finite-feature variational GP, ELBO+SGD, and matched/near-matched/mismatched evidence layers.
- The global dictionary is now Stage 0. The advanced POC will test operator-derived positive spectral separation and per-mode/per-rank kernel routing.
