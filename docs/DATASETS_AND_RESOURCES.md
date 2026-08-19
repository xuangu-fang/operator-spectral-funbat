# 方向 3：数据集与实现资源

更新时间：2026-08-19

中心总目录见 [共享物理数据与公开资源目录](https://github.com/xuangu-fang/Geo-Aware-Tensor/blob/master/SHARED_DATASETS.md)。方向 3 的数据选择必须检验 operator-induced kernel，而不能只重复从同一有限 atom bank 采样的自洽 sanity。

## 1. 当前证据的角色

- 当前 anisotropic-diffusion、reference/shifted/support-deletion suites 是机制与优化 sanity。
- 它们适合检验 ELBO+SGD、per-mode kernel、fixed generic support floor 和 operator mismatch。
- 因 truth 与 learner 共享可控谱结构，它们不能单独支撑“对真实 PDE 数据有普遍优势”。
- tilted advection 的 full signed separation error 明显更高，应保留为 limitation。

## 2. 本机资源

| 路径 | 适合的问题 | 风险 |
|---|---|---|
| /mnt/data/xuangu-fang/ai-physical-dynamics/datasets/kolmogorov_mno/raw | 周期域上的 operator spectrum、Re shift、规则网格时空场 | 不含不规则几何；需要定义 completion 而非 forecasting split |
| /mnt/data/xuangu-fang/ai-physical-dynamics/datasets/openfwi_curvefault_a | wave kernel、速度模型失配、source/receiver/time modes | full signed phase 与非平稳介质可能超出当前实值可分 kernel |
| /mnt/data/xuangu-fang/ai-physical-dynamics/datasets/cfdbench/{raw,extracted} | generic CFD stress 与 boundary-condition shift | 不应把未知精确算子描述成已知 operator prior |
| /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/Geo-Aware-Tensor/data | The Well acoustic 小子集和历史 smoke test | 使用前固定 manifest，不按结果选择 trajectory |

## 3. 外部数据优先级

1. [PDEBench](https://github.com/pdebench/PDEBench)：先选 diffusion/reaction-diffusion，再选 advection。由官方生成器和 PDE 参数构造 spectrum；truth solutions 不能来自当前 finite atom bank。
2. 本地 Kolmogorov：周期 Fourier spectrum 清楚，可测试 learned kernel 随 Reynolds number 的偏移；需预注册 train/test Re。
3. [OpenFWI](https://openfwi-lanl.github.io/)：作为 wave/high-frequency stress，允许得到明确负结论。
4. [The Well](https://github.com/PolymathicAI/the_well) acoustic/active matter：统一数据接口方便扩展，但先用固定小子集。
5. [RealPDEBench](https://github.com/AI4Science-WestlakeU/RealPDEBench)：适合最终 sim-to-real stress，不应在当前 kernel 选择阶段消费真实 test。

## 4. 每个数据集必须构造的对照

- operator kernel：由预先声明的 PDE/operator metadata 得到；
- generic dictionary：相同 feature budget；
- oracle kernel：允许读取 truth generator 参数，只作为上界；
- swapped/wrong operator：相同参数量但错误 PDE 参数或错误 mode assignment；
- fixed-support hybrid：预先固定 generic spectral floor，不能根据 test error 改比例；
- neural functional tensor 或 neural mean + GP residual：防止只与弱 GP baseline 比较。

所有模型共享 observations、noise、normalization、feature count、steps 和 seeds。主表报告 held-out NRMSE；Bayesian 表另报 NLL、coverage、interval width。NRMSE 约 1 时不讨论 kernel weight 的漂亮可解释性。

## 5. 推荐 manifest 附加字段

除中心公共字段外，必须记录 operator_family、operator_parameters_visible_to_learner、spectrum_construction、signed_or_even_spectrum、separation_rank、separation_error、feature_budget、generic_support_floor 和 oracle_information。

下一 session 应先在 PDEBench diffusion 上完成一个不依赖 finite atom generator 的 3-seed/400-step gate，再决定是否扩到 advection 和 wave。
