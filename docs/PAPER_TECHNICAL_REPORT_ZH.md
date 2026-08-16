# 方向 3 论文级技术报告：从物理算子频谱到稳健的 functional tensor prior

> 版本：2026-08-16 投稿确认版。本文面向论文的 Introduction、Related Work 与 Method，不混入仓库中早期的不规则域 kernel dictionary 支线。当前实验仍是可控合成机制验证，不能表述成真实 PDE benchmark 的最终结果。

## 1. 一句话主线

我们研究极稀疏连续张量补全：已知物理场大致受某个线性算子控制时，先把算子诱导的**不可分联合频谱**压缩成少量合法的一维 GP 频谱，再把这些频谱分配给不同 tensor mode/rank；同时保留少量通用频谱，使算子先验遗漏真实频率时模型仍能恢复。

这项工作的核心不是“GP + tensor”，也不是“把 PDE 塞进 loss”。核心是三个连续步骤：

1. 从算子得到联合解频谱，并用非负低秩分解投影为合法的一维 GP kernels；
2. 允许不同 tensor mode/rank 使用不同的物理频谱，而不是共享一个通用 kernel；
3. 在算子 support 错误时，用一个受约束的通用频谱库提供逃生通道。

## 2. Introduction 草稿

### 2.1 问题背景

多维物理场经常只在少量传感器、少量时间或少量工况上被观测。低秩 CP/Tucker 能利用多维相关性，functional tensor 模型进一步把离散 factor table 替换成连续函数，因此可以在未观测坐标上插值。然而，当观测率降低到 1%–5% 时，单靠低秩并不足够：模型仍需决定每个连续因子应当多平滑、是否振荡、以及不同坐标方向是否应共享同一种变化尺度。

普通 functional Bayesian tensor 通常预先指定 Matérn、RBF 等通用 GP kernel。这个选择没有利用一个常见事实：即使我们不知道完整解，控制物理场的算子往往是已知或近似已知的。扩散、输运和波动算子对不同联合频率的放大或抑制不同，因此它们天然给出了比通用平滑 kernel 更具体的归纳偏置。

直接使用算子联合频谱却有一个结构冲突。物理频谱通常耦合空间、时间或参数频率，不能写成各维一维谱的简单乘积；functional CP/Tucker 则依赖逐 mode 的一维函数。若强行给所有 mode 使用同一个 kernel，会丢掉算子的各向异性；若为每个 mode 任意学习一个大字典，又很容易退化为缺少物理来源的 kernel search。

### 2.2 方法直觉

设线性平稳算子满足

\[
\mathcal L u=w.
\]

在周期常系数近似下，解的功率谱为

\[
S_u(\boldsymbol\omega)
=\frac{S_w(\boldsymbol\omega)}{|\widehat{\mathcal L}(\boldsymbol\omega)|^2}.
\]

我们不直接把这个高维谱当成一个巨大 GP kernel，而是做非负低秩分解：

\[
S_u(\omega_1,\ldots,\omega_D)
\approx\sum_{q=1}^{Q_{\rm op}}\lambda_q
\prod_{d=1}^{D}s^{\rm op}_{dq}(\omega_d),
\qquad \lambda_q,s^{\rm op}_{dq}\ge 0.
\]

每个一维非负谱都对应一个半正定平稳 kernel。于是，不可分的物理联合结构被转换为一个小型、合法且有算子来源的一维 kernel bank。对于第 \(d\) 个 mode 的第 \(r\) 个 tensor factor，我们学习

\[
S_{dr}(\omega)=\sum_q\pi_{drq}s_{dq}(\omega),
\qquad \pi_{drq}\ge0,\quad\sum_q\pi_{drq}=1.
\]

若近似算子遗漏了真实频率，仅在 operator bank 中调权无法创造新的 frequency support。我们因此加入少量 generic spectra，但不让自由 softmax 把它们全部关闭：一个预先固定的 generic support floor 始终保留可用频率。确认实验表明，它能在 operator support 被严格删除时恢复预测；代价是在算子正确时会牺牲少量 specificity。

### 2.3 预期贡献

论文当前可以检验以下四点，但每一点都有明确边界。

1. **算子联合频谱的合法低秩投影。** 将不可分的 \(D\) 维物理谱通过非负分解转成逐 mode GP kernels；非负性保证每个 kernel PSD。
2. **逐维物理 kernel 与可选的 per-rank 适配。** 非负分解天然给不同坐标维度不同的一维 spectra；进一步的 per-rank routing 是可检验组件。最终确认支持前者，对后者只有小幅信号。
3. **可审计的有限谱变分推断。** 使用一套不随 atom 数增长的 Fourier 系数参数化和 ELBO+SGD，分别报告点预测、诱导频谱和 mean-field posterior predictive calibration。
4. **错先验下的 support safeguard。** 自由字典并不保证稳健性；固定非零 generic floor 能修复 operator prior 缺失的频率，并形成可量化的 robustness--specificity tradeoff。

我们**不主张**从 2% 数据中识别每个 atom 的真实标签，也不主张 posterior routing 等于 PDE discovery。相关 atoms 高度相关，softmax top-1 不可识别；因此实验只把 induced spectrum cosine/L2 作为连续诊断。

## 3. 与近邻工作的区别

| 工作 | 主要问题 | GP/tensor 结构 | 与本文的关键差别 |
|---|---|---|---|
| [FunBaT](https://arxiv.org/abs/2311.04829) | 连续坐标张量补全 | GP latent functions + Tucker；SDE/message passing | 使用通用逐 mode GP；没有从不可分算子联合谱构造 mode atoms 或逐 mode/rank 分配 |
| [RR-FBTC, 2025](https://arxiv.org/abs/2512.21486) | functional Bayesian tensor completion 与自动 rank learning | 多输出 GP + rank-revealing Bayesian tensor | 贡献在函数逼近能力和 rank 学习；本文固定 rank，研究 operator spectrum 的非负投影、mode/rank 分配与错先验稳健性 |
| [Tensor Gaussian Processes, 2025](https://arxiv.org/abs/2510.13772) | 高效求解非线性 PDE | 一维 GP factors + tensor decomposition；Newton/ALS | 它通过 collocation/PDE 方程直接求解；本文做有噪极稀疏张量补全，物理进入 prior spectrum，并显式研究 operator misspecification |
| [EPGP, ICML 2023](https://proceedings.mlr.press/v202/harkonen23a.html) | 为常系数线性 PDE 构造 exact-solution GP prior | Ehrenpreis–Palamodov 原理得到满足 PDE 的 GP 样本 | EPGP 的样本严格满足 PDE；本文只近似解的二阶频谱，不保证样本严格满足 PDE，但可直接形成低秩 functional tensor factors |
| [Spectral Mixture Kernels, ICML 2013](https://proceedings.mlr.press/v28/wilson13.html) | 从数据学习表达力强的平稳 kernel | 参数化频谱混合 | 本文的 operator atoms 来自 \(|\widehat{\mathcal L}|^{-2}S_w\) 的非负低秩投影，且按 tensor mode/rank 分配；generic mixture 仅作失配兜底 |

因此，论文标题和摘要不能把“GP + tensor”或“PDE-informed GP”本身写成创新。最窄且可守住的定位是：

> **不可分 operator joint spectrum 的非负低秩合法投影、逐维 GP kernels，以及固定 generic support floor 下的错先验稳健性。** Per-rank routing 是次要适配组件，不单独作为主 claim。

## 4. Problem formulation

令连续 \(D\) 阶场为

\[
y_i=u(x_{i1},\ldots,x_{iD})+\epsilon_i,
\qquad \epsilon_i\sim\mathcal N(0,\sigma^2),
\]

其中只有索引集合 \(\mathcal O\) 上的 \(N_{\rm obs}\) 个值被观测，\(N_{\rm obs}/N_{\rm full}\le 5\%\)。目标是在其余坐标上恢复场，并在可能时输出 posterior predictive uncertainty。

我们假定一个近似线性平稳算子 \(\widetilde{\mathcal L}\) 可用。它可以与真实算子一致，也可以存在系数偏差甚至缺失 frequency support。本文不假设算子参数必须从观测中被精确识别。

POC 使用三阶连续 CP：

\[
u(x_1,x_2,x_3)
=\sum_{r=1}^{R}c_r
f_{1r}(x_1)f_{2r}(x_2)f_{3r}(x_3).
\]

选择 CP 是为了隔离 kernel prior 的贡献。扩展到 Tucker 时，只需把对角权重 \(c_r\) 换成小 core \(G_{r_1r_2r_3}\)；本轮没有用更大 core 换取结果。

## 5. Method

### 5.1 从算子到联合解频谱

对周期常系数算子，Fourier 域中

\[
\widehat u(\boldsymbol\omega)
=\widehat w(\boldsymbol\omega)/\widehat{\mathcal L}(\boldsymbol\omega).
\]

若随机激励 \(w\) 的谱为 \(S_w\)，则

\[
S_{\rm op}(\boldsymbol\omega)
=|\widehat{\mathcal L}(\boldsymbol\omega)|^{-2}S_w(\boldsymbol\omega).
\]

当前实现支持三类机制：

- anisotropic diffusion：分母由 reaction 与不同轴的 \(D_d\omega_d^2\) 构成；
- advection–diffusion：实部为扩散/反应，虚部为 \(\omega_t+v_x\omega_x+v_y\omega_y\)；
- damped wave：频谱集中在近似 dispersion surface 周围。

当前实现是周期、常系数 frequency-domain sanity，不处理不规则边界。训练 atoms 只使用非负 frequency-magnitude octant，再用实 cosine/sine features 对每个轴独立对称化。对 diffusion 这与完整 signed spectrum 基本一致；对有倾斜输运项的 advection，它会丢掉不同符号组合之间的耦合。若数据来自有限非周期域，\(S_{\rm op}\) 还只是局部/近似 prior；当前实验已经说明，遇到这些失配不能假设一个自由 generic bank 会自动修复。

### 5.2 非负低秩谱分离

在离散频率网格上求

\[
\min_{\lambda_q\ge0,\,s_{dq}\ge0}
\left\|S_{\rm op}
-\sum_{q=1}^{Q_{\rm op}}\lambda_q
s_{1q}\circ\cdots\circ s_{Dq}\right\|_F^2.
\]

实现采用确定性 seed 的 nonnegative CP multiplicative updates。每个一维 factor 随后归一化到单位 marginal variance。联合分量尺度 \(\lambda_q\) 被单独记录；在当前 CP likelihood 中，整体幅值与 tensor core 存在尺度不识别，因此没有把 \(\lambda_q\) 重复乘入各 mode kernel。

谱分离误差

\[
e_Q=\|S_{\rm op}-\widehat S_Q\|_F/\|S_{\rm op}\|_F
\]

是方法适用性的先验诊断。低 \(e_Q\) 表示 operator spectrum 适合少量逐 mode atoms；较高误差的 wave dispersion surface 是预期困难案例，而不是隐藏的失败。

必须分别审计 positive octant 与 full signed grid。rank 4 下，reference/shifted advection 的 octant error 为 `0.0323/0.0242`，但 full signed error 为 `0.1799/0.1812`；anisotropic diffusion 则为 `0.0030/0.0043`。因此当前 advection 实验只能证明 magnitude-spectrum prior 有效，不能声称完整恢复了 tilted transport spectrum。完整版本需要 signed-conjugate components、complex factors 或显式 cross-mode phase；本轮没有伪装成已实现。

### 5.3 非负谱为何保证合法 kernel

在一维周期网格上，对任意非负单边谱 \(s(k)\) 定义

\[
\phi_s(x)=\left[
\sqrt{s(0)},
\sqrt{2s(k)}\cos(2\pi kx),
\sqrt{2s(k)}\sin(2\pi kx)
\right]_{k=1}^{K}.
\]

于是

\[
k_s(x,x')=\phi_s(x)^\top\phi_s(x')
\]

必然半正定。对非负 convex mixture

\[
s_{dr}(k)=\sum_q\pi_{drq}s_{dq}(k)
\]

同样成立。因此从 nonnegative CP、routing softmax 到最终 kernel 的每一步都不破坏 PSD。

这里有一个必须写进 Method 而不是藏在附录里的边界：当前 \(s_{dq}(k)\) 是单边 magnitude spectrum，实 Fourier features 会把 \(+k\) 与 \(-k\) 成对。因此当前 hypothesis class 表示的是逐轴 even、separable covariance component。anisotropic diffusion 的 symbol 由各轴平方项组成，与这个表示匹配，是论文最干净的主例。advection 的 \(\omega_t+v_x\omega_x+v_y\omega_y\) 含 cross-sign coupling；只分解正 octant 后再逐轴对称，并不等价于完整 transport spectrum。本文把它作为 limitation/stress test，而不是主理论例子。

### 5.4 Operator bank 与固定 generic support floor

最终候选谱为

\[
\mathcal B_d=
\{s^{\rm op}_{d1},\ldots,s^{\rm op}_{dQ_{\rm op}}\}
\cup
\{s^{\rm gen}_{1},\ldots,s^{\rm gen}_{Q_{\rm gen}}\}.
\]

POC 冻结 \(Q_{\rm op}=4\)、\(Q_{\rm gen}=4\)，generic atoms 分别覆盖 smooth、Matérn-like、oscillatory 与 broadband。generic-only 是主要 baseline。robust 版本约束

\[
\pi_{drq}=\rho_q+(1-\textstyle\sum_j\rho_j)
\operatorname{softmax}(\alpha_{dr})_q,
\]

其中 operator atoms 的 \(\rho_q=0\)，四个 generic atoms 均分总计 25% 的固定 floor。routing 以 operator logits 0、generic logits -2 初始化。25% 是在 development seeds 上只做一次的预声明最小重试，没有继续搜索。

这不是“自由字典会自动适配”。floor 明确用一部分 matched-prior 精度换取 support safety：development matched NRMSE 从 operator-only `0.0396±0.0066` 变为 robust `0.0467±0.0085`，但 strict wrong-support 从 `0.6195±0.2149` 恢复到 `0.0480±0.0079`，5/5 paired wins。

### 5.5 Global 与 mode/rank routing

比较两种主要参数共享方式：

\[
\text{global: }\pi_{drq}=\pi_q,
\qquad
\text{mode/rank: }\pi_{drq}=\operatorname{softmax}(\alpha_{dr})_q.
\]

global baseline 强制三个 mode 和所有 CP rank 共享一个谱。mode/rank 版本允许每个连续因子具有不同的物理尺度。\(\alpha\) 是通过 ELBO 优化的确定性 kernel hyperparameter；当前没有为 routing 构造额外变分 posterior，因此 routing softmax 不能被解释成 Bayesian model probability。

### 5.6 本轮关键公平性修正：collapsed spectral mixture

旧 POC 为每个 atom 分配一套独立系数：

\[
f_{dr}(x)=\sum_q\sqrt{\pi_{drq}}\,\phi_{dq}(x)^\top a_{drq}.
\]

这在 prior covariance 上是正确的，但 8-atom robust bank 比 4-atom bank 多一倍变分系数，导致方法比较混入参数预算差异。

投稿确认实验改用等价的 canonical 表示：先计算

\[
s_{dr}=\sum_q\pi_{drq}s_{dq},
\]

再构造单套 features \(\phi_{s_{dr}}\) 与系数

\[
f_{dr}(x)=\phi_{s_{dr}}(x)^\top a_{dr},
\qquad a_{dr}\sim\mathcal N(0,I).
\]

因为所有 atoms 使用相同离散 Fourier support，这两个 prior covariance 相同；collapsed 版本去掉了重复 features。其变分系数数量固定为

\[
2DR(1+2K),
\]

不随 bank atom 数量变化。robust 版本只比 operator baseline 多 routing logits，不再多 GP coefficients。旧 expanded 结果不能与本轮混表；collapsed 结果证明 escape 不依赖额外 GP coefficient budget。

旧 expanded POC 与新 collapsed confirmation 必须分表报告，不能把两者的数值直接合并。

### 5.7 Variational inference

每个 Fourier coefficient 采用 mean-field Gaussian：

\[
q(a_{dr})=\mathcal N(\mu_{dr},\operatorname{diag}(\sigma_{dr}^2)),
\qquad p(a_{dr})=\mathcal N(0,I).
\]

CP core 权重、routing logits 与 observation noise 是点估计。训练最大化

\[
\mathcal L
=\mathbb E_q\left[
\sum_{i\in\mathcal O}\log p(y_i\mid f,c,\sigma_y)
\right]
-\sum_{d,r}\operatorname{KL}[q(a_{dr})\|p(a_{dr})].
\]

实现用 3 个 reparameterized MC samples 估计期望 log likelihood，Adam 优化 400 steps。当前观测率不超过 2% 的确认实验使用完整 observed batch，因此代码中的 minibatch scaling 等于 1；loss 除以 \(N_{\rm obs}\)，相当于优化 per-observation negative ELBO。所有方法使用相同学习率、gradient clipping、初始化随机流和 step 数。

### 5.8 Predictive uncertainty 的准确表述

给定 mean-field coefficient posterior，代码可以精确计算 latent marginal 的一二阶矩，也可以从

\[
q(a)p(y_*\mid a)
\]

采样。确认实验在固定训练结束后，用 64 个 posterior predictive samples，在 1024 个未观测位置报告：

- 95% posterior predictive interval coverage；
- Monte-Carlo posterior predictive NLL；
- interval width。

评估目标是额外独立加噪的 held-out observations。这个 UQ 是**有限 Fourier、mean-field variational posterior predictive**，不是解析 exact GP posterior；64-sample NLL 也带有 Monte-Carlo 误差。论文必须使用这一限定。

## 6. 复杂度

记 mode 数为 \(D\)，tensor rank 为 \(R\)，单边最高频率为 \(K\)，Fourier feature 数 \(F=1+2K\)，bank atoms 为 \(Q\)，观测数为 \(N_{\rm obs}\)。

- 变分系数与尺度参数：\(2DRF\)，与 \(Q\) 无关；
- mode/rank routing 参数：\(DRQ\)；global routing 为 \(Q\)；
- 每个 MC sample 的观测预测：约 \(O(N_{\rm obs}DRF)\)；
- operator joint spectrum 网格：\(O((K+1)^D)\) 存储；
- nonnegative CP 分离：随分离迭代数、\(Q_{\rm op}\) 和联合谱网格线性增长。

当前 POC 的主要扩展瓶颈不是 tensor completion，而是高维联合 spectrum grid。未来可用解析 symbol、稀疏频率采样或 tensor train 表示替代完整 \((K+1)^D\) 网格。

## 7. 投稿确认实验设计

### 7.1 冻结协议

- development/audit seeds：101–105，仅用于发现数值问题与冻结协议，不进入确认表；
- untouched confirmation seeds：201–205，共 5 个；
- 网格：\(24^3\)；观测率：2%；训练观测约 276 个；
- functional CP rank：2；operator atoms：4；generic atoms：4；
- frequency support：\(k=0,\ldots,6\)，每个 factor 13 个 Fourier features；
- optimizer：Adam；400 steps；3 个 ELBO samples；无 validation、无 early stopping；
- 同一 case/seed 的所有方法共享场、mask、训练噪声和 UQ targets；
- 所有未观测值只在 400 steps 完成后用于报告。

### 7.2 三个算子设置

1. reference advection–diffusion：复现原始机制；
2. shifted advection–diffusion：改变两个扩散系数、两个速度分量、反应和 forcing scale；
3. strongly anisotropic diffusion：三个轴扩散系数差异显著。

每个 setting 都单独报告 rank-4 joint-spectrum separation error。它们不是从 test labels 拟合得到的超参数，而是在数据生成前冻结。

### 7.3 方法矩阵

- operator-only：global / per-mode-rank；
- generic-only：global / per-mode-rank；
- oracle operator component route；
- robust operator+generic：global / per-mode-rank，固定 25% generic support floor；
- strict wrong-support operator-only：删除所有 \(k\ge2\) operator support；
- strict wrong-support + generic escape。

所有 learned 方法保持相同 coefficient posterior 参数数和 400-step 预算。oracle 用于测量 prior 路由上限，不属于可部署方法。

### 7.4 主要指标与 gate

主要指标是 held-out clean-field NRMSE。辅助指标包括 induced-spectrum cosine/L2、predictive coverage/NLL 和逐 seed paired wins。

投稿主线 gate 被收缩为两个条件：

1. matched setting 中 operator-derived per-mode/rank prior 相对 generic-only 和 operator-global 至少有一项稳定、逐 seed 可复现的优势；
2. anisotropic diffusion 的 **full signed** joint spectrum 在小 rank 下可压缩；advection 必须分别报告 positive octant 与 full signed error，只作为 cross-sign coupling stress test，不进入通过 gate 的正证据。

第二个 gate 是：strict support mismatch 中 floor-robust 必须稳定改善 wrong-support operator-only，同时 matched case 的代价可量化且不过大。

## 8. 本轮确认结果

### 8.1 Development audit：support floor 的代价与收益

| 方法 | matched NRMSE | strict wrong-support NRMSE |
|---|---:|---:|
| operator per-mode/rank | 0.0396±0.0066 | 0.6195±0.2149 |
| generic per-mode/rank | 0.1292±0.1390 | — |
| operator + generic（25% floor） | 0.0467±0.0085 | **0.0480±0.0079** |

robust 在 matched case 0/5 战胜 operator-only，平均代价为 `+0.0071` NRMSE；但在 wrong-support case 5/5 改善错误先验，平均收益为 `-0.5714`。因此它是明确的 robustness--specificity tradeoff。

工程审计曾产生一版错误的 negative result：collapsed basis 最初通过“第一个 atom feature ÷ 第一个 atom amplitude”恢复公共 Fourier basis。当第一个 operator atom 的 \(k\ge2\) support 被设为零时，公共高频 basis 也错误地变成零，所以 generic floor 虽有非零谱却无法产生高频 feature。修复为直接从坐标解析构造 \([1,\sqrt2\cos,\sqrt2\sin]\) basis 后，只重跑 strict 两个 controls；matched 主表没有重训。新增测试覆盖 zero-support feature、Gram PSD 和 finite routing gradient。修复前后审计保存在 `results/strict_support_basis_fix_audit.json`。

### 8.2 Untouched 201–205 confirmation

所有数值均来自协议冻结后一次性运行的 seeds 201–205。

| setting | operator-global | operator per-mode/rank | generic-global | generic per-mode/rank | oracle operator route |
|---|---:|---:|---:|---:|---:|
| reference advection | 0.0411±0.0144 | 0.0462±0.0332 | 0.0474±0.0237 | **0.0343±0.0077** | 0.0325±0.0106 |
| shifted advection | 0.1006±0.0586 | 0.0995±0.0590 | **0.0892±0.0369** | 0.2312±0.2580 | 0.1042±0.1041 |
| anisotropic diffusion | 0.1212±0.0606 | **0.1183±0.0582** | 0.3761±0.5235 | 0.1567±0.0990 | 0.1183±0.0588 |

最清楚的确认信号来自数学上与 real separable spectrum 匹配的 anisotropic diffusion：operator per-mode/rank 相对参数匹配的 generic per-mode/rank 平均 NRMSE 降低约 24.5%，5/5 paired wins，并追平 oracle mean。相对 operator-global 只有约 2.4% 均值改善和 3/5 wins。因此当前证据强烈支持 **operator-derived mode kernels**，但不足以把自由 per-rank routing 单独包装成第二个主要贡献。

advection 两个 setting 不形成均值上的一致 winner。reference case 中 operator per-mode/rank 虽对 global 和 generic per-mode/rank 都有 4/5 paired wins，但 seed 203 出现 `0.111` 的优化 outlier，使均值反转；shifted case 的 generic-global 均值最好。结合 full signed-spectrum audit，这两个结果更适合作为“当前 even/magnitude kernel 对倾斜输运不完整”的边界证据。

anisotropic diffusion 的连续诊断也与预测一致：

| 方法 | induced-spectrum cosine ↑ | spectrum relative L2 ↓ | predictive NLL ↓ | 95% coverage |
|---|---:|---:|---:|---:|
| operator-global | 0.972 | 0.269 | -0.642 | 0.956 |
| operator per-mode/rank | **0.977** | **0.204** | **-0.679** | 0.947 |
| generic per-mode/rank | 0.926 | 0.427 | -0.451 | 0.960 |
| oracle | 1.000 | 0.000 | -0.678 | 0.946 |

这里的 NLL 和 coverage 是 64-sample mean-field variational posterior predictive，不是 exact GP posterior。operator per-mode/rank 的 coverage 接近名义 95%，NLL 与 oracle 基本相同。

strict support 的确认结果为：

| setting | wrong-support operator | 25% floor robust | paired wins |
|---|---:|---:|---:|
| reference advection | 0.6723±0.1938 | **0.0402±0.0096** | 5/5 |
| shifted advection | 0.6318±0.0883 | **0.0847±0.0414** | 5/5 |
| anisotropic diffusion | 0.6149±0.1798 | **0.1301±0.0621** | 5/5 |

在 matched anisotropic diffusion 上 robust 为 `0.1308±0.0624`，比 operator-only `0.1183±0.0582` 差约 0.0124；在 prior 被删频时却恢复到几乎相同的 `0.1301`。这正是固定 support floor 的预期行为。

### 8.3 投稿 gate 判断

本轮给出的是**收缩后的条件 GO**：

- GO：anisotropic diffusion 上，operator-derived kernels 相对 generic kernels 有 5/5 paired prediction wins、更好的 induced spectrum 和 NLL；rank-4 full signed spectrum error 仅 0.0043。
- 弱信号：per-mode/rank routing 只小幅优于 operator-global，不能单独作为大贡献。
- GO（secondary）：固定 25% generic support floor 在三类 strict mismatch 上全部 5/5 wins，并在 matched setting 付出可测的小幅代价。
- NO-GO：当前 real separable features 不能把 advection positive-octant 低误差解释成完整 tilted transport spectrum。

因此推荐论文主线为：**针对 even/axis-separable operator spectra（以 anisotropic diffusion 为主例），把联合谱非负投影成逐维合法 GP kernels，并用固定 generic support floor 对抗错先验。** Mode/rank routing 是实现组件和消融；advection 仍是限制。

## 9. 工程与统计审计清单

### 已通过

- 每个 atom 与 routed mixture 的频谱非负；Fourier Gram matrix PSD；
- one-sided spectrum 按 \(s_0+2\sum_{k>0}s_k=1\) 归一化；
- operator、generic 和 robust 使用完全相同的 \(k=0,\ldots,6\) support 上限；
- collapsed coefficient shape 为 `[mode, rank, feature]`，不依赖 atom 数；
- exact posterior moments 与大样本 MC 在单元测试容差内一致；
- full-observed-batch ELBO 的 likelihood 与 KL scaling 可直接审计；
- train mask 后的所有索引都不进入 optimizer、early stopping 或超参选择；
- fixed route shape 为 `[mode, rank, atom]`，并显式归一化。

### 仍有限制

- synthetic field 从同一个有限 operator atom family 采样，属于机制 sanity；
- mean-field posterior 忽略 CP factors 间 posterior correlation；
- routing 是 point estimate，UQ 未包含 kernel-routing uncertainty；
- 周期 Fourier prior 不处理边界条件与非平稳系数；
- rank 固定为 2，没有与 RR-FBTC 的 rank learning 做正面竞争；
- nonnegative CP 可能存在局部最优和 component permutation；论文只使用重构误差与 induced spectrum，不解释 component label。
- joint-spectrum component weights \(\lambda_q\) 当前只被记录，因与 CP core scale 不识别而未进入 routing prior；真实数据版本应检验用 \(\lambda_q\) 初始化/正则化 routing 是否更合理。
- 训练 kernel 使用非负频率 octant；full signed audit 显示 advection 的 cross-sign coupling 更难分离，当前 real axis-wise factors 不能表示完整 transport phase。

## 10. 可证伪 claims 与下一步

### 本轮已经验证或否证的 claims

1. **验证：** even/axis-separable 的 anisotropic-diffusion full signed spectrum 可由少量一维 PSD atoms 高精度表示；
2. **验证：** 2% 观测下，operator-derived mode kernels 在 anisotropic diffusion 上 5/5 战胜参数匹配 generic kernels；
3. **弱信号：** per-mode/rank routing 相对 operator-global 仅 3/5 wins 和约 2.4% 均值改善，不能独立成为主 claim；
4. **验证：** 自由选择不是 support 保证，但固定 25% generic floor 在 strict deletion 下三类 setting 均 5/5 恢复；
5. **验证：** 前述结论是在 bank-size-independent coefficient budget 下得到的。

### 下一阶段的真正 gate

- 在不是从相同 finite atoms 采样的 PDE solutions 上，operator-derived kernels 仍需稳定优于 FunBaT/generic functional tensor；
- observation ratio 1%/2%/5% 与 structured sensor masks 下，优势需形成可解释 phase diagram；
- 非周期边界、变系数和 operator coefficient mismatch 下，优势不能完全消失；
- 若要重新纳入 advection，必须实现并验证 signed-conjugate/complex 或 cross-mode phase factors；
- predictive UQ 需在独立 PDE 数据上维持合理 coverage/NLL，而不是只在 planted data 上成立。

### 投稿前仍需要

1. 至少一个不由本模型 atom 直接生成的 PDE solution dataset；
2. 与 FunBaT、RR-FBTC 和 neural/functional CP 的统一 sparse-mask baseline；
3. operator coefficient misspecification 连续曲线，而不只有 matched/删频两个端点；
4. nonnegative separation rank、generic atom 数、tensor rank 和 observation ratio ablation；
5. 若进入真实非周期域，明确边界误差以及 periodic-prior failure case。

## 11. 复现入口

- 核心模型：`src/geoaware/operator_spectral_funbat.py`
- 旧 expanded POC：`experiments/run_operator_spectral_poc.py`
- 新 collapsed 投稿确认：`experiments/run_submission_confirmation.py`
- 旧结果：`results/advanced_poc_r1_r5/`
- 新结果：`results/submission_confirmation/`
- 单元测试：`tests/test_operator_spectral_funbat.py`
