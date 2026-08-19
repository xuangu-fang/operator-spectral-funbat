# Operator-Spectral FunBaT 论文级技术报告

> **版本**：2026-08-19 重写版（数值口径 = 2026-08-16 投稿确认版，未重训任何实验）。
> **范围**：面向论文的 Introduction、Related Work、Method、实验协议与确认结果。
> **纪律声明**：本轮全部实验仍是**周期、常系数的可控合成机制验证**，不能表述为真实 PDE benchmark 的最终结果。仓库中早期的不规则域 kernel dictionary 支线不并入本文，见 [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md)。
> **公式渲染**：本文全部使用 `$ ... $` / `$$ ... $$`，可在 GitHub 网页端直接渲染。

---

## 目录

1. [一句话主线](#1-一句话主线)
2. [Introduction 草稿](#2-introduction-草稿)
3. [符号表](#3-符号表)
4. [与近邻工作的区别](#4-与近邻工作的区别)
5. [问题设定](#5-问题设定)
6. [方法](#6-方法)
7. [复杂度](#7-复杂度)
8. [投稿确认实验协议](#8-投稿确认实验协议)
9. [确认结果](#9-确认结果)
10. [工程与统计审计清单](#10-工程与统计审计清单)
11. [可证伪 claims 与下一步](#11-可证伪-claims-与下一步)
12. [复现入口](#12-复现入口)
13. [附录 A：确认实验完整数值表](#附录-a确认实验完整数值表)

---

## 1. 一句话主线

我们研究**极稀疏观测下的连续张量补全**：当物理场大致受某个已知或近似已知的线性算子控制时，先把算子诱导的**不可分联合功率谱**压缩成少量**合法的一维 GP 谱**，再把这些谱分配给不同的 tensor mode / rank；同时保留一小块**不可关闭的通用谱**，使得算子先验遗漏真实频率时模型仍能恢复。

核心不是"GP + tensor"，也不是"把 PDE 塞进 loss"。核心是三个连续步骤：

$$
\underbrace{\mathcal{L}u=w\;\Longrightarrow\;S_{\mathrm{op}}(\boldsymbol{\omega})}_{\textbf{(i)}}
\;\Longrightarrow\;
\underbrace{S_{\mathrm{op}}\approx\sum_{q}\lambda_q\,s_{1q}\circ\cdots\circ s_{Dq}}_{\textbf{(ii)}}
\;\Longrightarrow\;
\underbrace{s_{dr}=\sum_{q}\pi_{drq}\,s_{dq}}_{\textbf{(iii)}}
$$

| 步骤 | 做什么 | 可检验断言 |
|---|---|---|
| **(i)** | 算子 $\to$ 联合解频谱 | 算子给出比通用平滑核更具体的归纳偏置 |
| **(ii)** | 非负低秩投影 $\to$ 逐维合法 kernel | 非负性保证每一步都不破坏半正定性 |
| **(iii)** | mode/rank 路由 + generic floor | 自由字典**不是** support 保证，固定 floor 才是 |

---

## 2. Introduction 草稿

### 2.1 问题背景

多维物理场经常只在少量传感器、少量时间点或少量工况上被观测。低秩 CP/Tucker 能利用多维相关性；functional tensor 模型进一步把离散 factor table 替换成连续函数，因而可以在未观测坐标上插值。

但当观测率降到 1%–5% 时，单靠低秩结构并不足够。模型仍需决定：

- 每个连续因子应当**多平滑**；
- 是否应当**振荡**、在哪个频段振荡；
- 不同坐标方向是否应当**共享同一种变化尺度**。

普通 functional Bayesian tensor 通常预先指定 Matérn、RBF 等通用 GP 核。这个选择浪费了一个常见事实：**即使我们不知道完整解，控制物理场的算子往往是已知或近似已知的**。扩散、输运与波动算子对不同联合频率的放大/抑制模式完全不同，因此天然给出了比通用平滑核更具体的先验。

### 2.2 直接使用算子谱的结构冲突

设线性平稳算子满足 $\mathcal{L}u=w$。在周期常系数近似下，解的功率谱为

$$
S_u(\boldsymbol{\omega})=\frac{S_w(\boldsymbol{\omega})}{\bigl\lvert\widehat{\mathcal{L}}(\boldsymbol{\omega})\bigr\rvert^{2}} .
$$

直接把 $S_u$ 当成一个巨大的 $D$ 维 GP 核会遇到结构冲突：

| 冲突项 | 说明 |
|---|---|
| **不可分性** | 物理谱通常耦合空间/时间/参数频率，写不成各维一维谱的简单乘积；而 functional CP/Tucker 依赖逐 mode 的一维函数。 |
| **各向异性丢失** | 若强行让所有 mode 共享同一个核，算子的方向性信息被抹平。 |
| **退化为核搜索** | 若为每个 mode 自由学一个大字典，方法很容易退化成缺少物理来源的 kernel search，失去"物理先验"这一卖点。 |

### 2.3 方法直觉

我们不直接使用 $S_u$，而是对它做**非负低秩分解**：

$$
S_{\mathrm{op}}(\omega_1,\ldots,\omega_D)\;\approx\;\sum_{q=1}^{Q_{\mathrm{op}}}\lambda_q\prod_{d=1}^{D}s^{\mathrm{op}}_{dq}(\omega_d),
\qquad \lambda_q\ge 0,\;\; s^{\mathrm{op}}_{dq}\ge 0 .
$$

每一个一维非负谱都通过 Wiener–Khinchin 对应一个**半正定的平稳核**。于是，不可分的物理联合结构被转换成一个**小型、合法、且有算子来源**的一维核库（bank）。第 $d$ 个 mode 的第 $r$ 个 tensor factor 再学习一个凸组合：

$$
s_{dr}(\omega)=\sum_{q}\pi_{drq}\,s_{dq}(\omega),
\qquad \pi_{drq}\ge 0,\quad \sum_q \pi_{drq}=1 .
$$

**关键的失败模式**：若近似算子遗漏了真实频率，仅在 operator bank 内部调权**无法创造新的 frequency support**——凸组合的 support 永远是各 atom support 的并集的子集。我们因此加入少量 generic 谱，但**不允许**自由 softmax 把它们全部关闭：一个预先固定的 generic support floor 始终保留可用频率。确认实验表明它能在 operator support 被严格删除时恢复预测，代价是在算子正确时牺牲少量 specificity。

### 2.4 预期贡献与各自的边界

| # | 贡献 | 边界（必须写进论文，不能藏进附录） |
|---|---|---|
| C1 | **算子联合谱的合法低秩投影**：将不可分的 $D$ 维物理谱通过非负分解转成逐 mode GP 核，非负性保证每个核 PSD。 | 只对 **even / axis-separable** 谱成立。含倾斜输运项的 advection 存在 cross-sign 耦合，当前实 Fourier 表示无法覆盖。 |
| C2 | **逐维物理核 + 可选 per-rank 适配**：非负分解天然给不同坐标维度不同的一维谱。 | 确认实验强支持"逐维"，对"per-rank routing"只有弱信号（3/5 wins，均值 ~2.4%），不能单独作为主 claim。 |
| C3 | **可审计的有限谱变分推断**：一套**不随 atom 数增长**的 Fourier 系数参数化 + ELBO/SGD，分别报告点预测、诱导谱、posterior predictive calibration。 | UQ 是 mean-field、有限 Fourier、64-sample MC 的 variational posterior predictive，不是解析 exact GP posterior。 |
| C4 | **错先验下的 support safeguard**：固定非零 generic floor 能修复 operator prior 缺失的频率，形成可量化的 robustness–specificity tradeoff。 | 这是 secondary contribution，**不是** automatic kernel discovery；floor 比例（25%）是在 development seeds 上只做一次的预声明最小重试。 |

我们**明确不主张**：从 2% 数据中识别每个 atom 的真实标签；posterior routing 等于 PDE discovery。相关 atoms 高度相关，softmax top-1 不可识别，因此实验只把 induced spectrum 的 cosine / 相对 L2 作为**连续诊断**。

---

## 3. 符号表

| 符号 | 含义 |
|---|---|
| $D$ | tensor 阶数（POC 中 $D=3$，轴为 $x,y,t$） |
| $R$ | functional CP rank（确认实验固定 $R=2$） |
| $K$ | 单边最高频率（确认实验 $K=6$，即 $k=0,\dots,6$） |
| $F=1+2K$ | 每个 factor 的 Fourier feature 数（$F=13$） |
| $Q_{\mathrm{op}},\,Q_{\mathrm{gen}}$ | operator / generic atom 个数（各 4） |
| $Q$ | bank 总 atom 数 |
| $\mathcal{O}$, $N_{\mathrm{obs}}$ | 观测索引集合与观测数 |
| $S_{\mathrm{op}}(\boldsymbol\omega)$ | 算子诱导的联合功率谱 |
| $s_{dq}(k)$ | 第 $d$ 个 mode、第 $q$ 个 atom 的一维单边非负谱 |
| $\pi_{drq}$ | mode $d$、rank $r$ 对 atom $q$ 的路由权重 |
| $\rho_q$ | atom $q$ 的固定 support floor（operator atoms 为 0） |
| $a_{dr}$ | Fourier 系数向量，变分对象 |
| $c_r$ | CP core 权重（点估计） |
| $e_Q$ | 秩 $Q$ 的谱分离相对误差 |

---

## 4. 与近邻工作的区别

| 工作 | 主要问题 | GP / tensor 结构 | 与本文的关键差别 |
|---|---|---|---|
| [FunBaT (2023)](https://arxiv.org/abs/2311.04829) | 连续坐标张量补全 | GP latent functions + Tucker；SDE / message passing | 使用**通用**逐 mode GP；没有从不可分算子联合谱构造 mode atoms，也没有逐 mode/rank 分配 |
| [RR-FBTC (2025)](https://arxiv.org/abs/2512.21486) | functional Bayesian tensor completion 与自动 rank 学习 | 多输出 GP + rank-revealing Bayesian tensor | 贡献在函数逼近能力与 rank 学习；本文**固定 rank**，研究算子谱的非负投影、mode/rank 分配与错先验稳健性 |
| [Tensor GPs (2025)](https://arxiv.org/abs/2510.13772) | 高效求解非线性 PDE | 一维 GP factors + tensor decomposition；Newton/ALS | 它通过 collocation/PDE 方程**直接求解**；本文做**有噪极稀疏张量补全**，物理只进入 prior spectrum，并显式研究 operator misspecification |
| [EPGP (ICML 2023)](https://proceedings.mlr.press/v202/harkonen23a.html) | 为常系数线性 PDE 构造 exact-solution GP prior | Ehrenpreis–Palamodov 原理，样本严格满足 PDE | EPGP 的样本**严格**满足 PDE；本文只近似解的**二阶频谱**，不保证样本满足 PDE，但可直接形成低秩 functional tensor factors |
| [Spectral Mixture Kernels (ICML 2013)](https://proceedings.mlr.press/v28/wilson13.html) | 从数据学习表达力强的平稳核 | 参数化频谱混合 | 本文的 atoms 来自 $\lvert\widehat{\mathcal L}\rvert^{-2}S_w$ 的**非负低秩投影**且按 mode/rank 分配；generic mixture 仅作失配兜底 |

因此，标题与摘要**不能**把"GP + tensor"或"PDE-informed GP"本身写成创新。最窄且可守住的定位是：

> **把不可分的 operator joint spectrum 做非负低秩、保 PSD 的投影，得到逐维 GP kernels；并用固定 generic support floor 换取错先验下的稳健性。** Per-rank routing 是次要适配组件，不单独作为主 claim。

---

## 5. 问题设定

令连续 $D$ 阶场的观测为

$$
y_i=u(x_{i1},\ldots,x_{iD})+\epsilon_i,
\qquad \epsilon_i\sim\mathcal{N}(0,\sigma^{2}),
\qquad i\in\mathcal{O},
$$

其中 $N_{\mathrm{obs}}=|\mathcal{O}|$ 且 $N_{\mathrm{obs}}/N_{\mathrm{full}}\le 5\%$。目标是在其余坐标上恢复场，并在可能时输出 posterior predictive uncertainty。

我们假定存在一个**近似**线性平稳算子 $\widetilde{\mathcal{L}}$。它可以与真实算子一致、可以有系数偏差，甚至可以**缺失 frequency support**。本文不假设算子参数必须从观测中被精确识别。

POC 使用三阶连续 CP：

$$
u(x_1,x_2,x_3)=\sum_{r=1}^{R}c_r\,f_{1r}(x_1)\,f_{2r}(x_2)\,f_{3r}(x_3).
$$

选择 CP 而非 dense Tucker 是为了**隔离 kernel prior 的贡献**——避免把收益混入额外 core 参数。扩展到 Tucker 只需把对角权重 $c_r$ 换成小 core $G_{r_1r_2r_3}$；本轮没有用更大 core 换取结果。

---

## 6. 方法

### 6.1 从算子到联合解频谱

对周期常系数算子，在 Fourier 域中 $\widehat{u}(\boldsymbol\omega)=\widehat{w}(\boldsymbol\omega)/\widehat{\mathcal{L}}(\boldsymbol\omega)$。若随机激励 $w$ 的谱为 $S_w$，则

$$
S_{\mathrm{op}}(\boldsymbol\omega)=\bigl\lvert\widehat{\mathcal{L}}(\boldsymbol\omega)\bigr\rvert^{-2}S_w(\boldsymbol\omega).
$$

当前实现（`operator_joint_spectrum`）支持三类机制，轴序为 $(\omega_x,\omega_y,\omega_t)$，激励取各向同性高斯 $S_w=\exp(-\alpha\Vert\boldsymbol\omega\Vert^2)$：

| 机制 | $\lvert\widehat{\mathcal L}\rvert^{2}$ | 谱结构 |
|---|---|---|
| anisotropic diffusion | $\bigl(\kappa+D_x\omega_x^2+D_y\omega_y^2+D_t\omega_t^2\bigr)^2$ | 逐轴平方项之和，**even 且接近可分** |
| advection–diffusion | $\bigl(\kappa+D_x\omega_x^2+D_y\omega_y^2\bigr)^2+\bigl(\omega_t+v_x\omega_x+v_y\omega_y\bigr)^2$ | 含**跨符号耦合**的输运项 |
| damped wave | $\bigl(c_x\omega_x^2+c_y\omega_y^2-\omega_t^2\bigr)^2+\bigl(\gamma_0+\gamma_1\lvert\omega_t\rvert\bigr)^2$ | 能量集中在倾斜 dispersion surface 附近 |

**必须写清的三条边界：**

1. 这是**周期、常系数**的 frequency-domain sanity，不处理不规则边界；
2. 训练 atoms 只使用**非负频率 octant**，再用实 cosine/sine features 对每个轴独立对称化——对 diffusion 与完整 signed spectrum 基本一致，对倾斜 advection 会丢掉不同符号组合之间的耦合；
3. 若数据来自有限非周期域，$S_{\mathrm{op}}$ 只是局部/近似 prior。

### 6.2 非负低秩谱分离

在离散频率网格上求解

$$
\min_{\lambda_q\ge 0,\;s_{dq}\ge 0}\;
\Bigl\Vert\,S_{\mathrm{op}}-\sum_{q=1}^{Q_{\mathrm{op}}}\lambda_q\, s_{1q}\circ\cdots\circ s_{Dq}\,\Bigr\Vert_F^{2}.
$$

实现（`nonnegative_cp_spectrum`）采用**确定性 seed 的非负 CP 乘性更新**（Euclidean loss，MTTKRP 形式），每轮把各 mode factor 的范数收进 $\lambda_q$ 以稳定迭代。每个一维 factor 随后归一化到单位 marginal variance。

> **关于 $\lambda_q$ 的诚实说明**：联合分量尺度 $\lambda_q$ 被单独记录，但**没有**重复乘入各 mode kernel。原因是在当前 CP likelihood 下，整体幅值与 tensor core $c_r$ 存在尺度不可识别。真实数据版本应检验用 $\lambda_q$ 初始化/正则化 routing 是否更合理。

谱分离误差

$$
e_Q=\frac{\bigl\Vert S_{\mathrm{op}}-\widehat{S}_Q\bigr\Vert_F}{\bigl\Vert S_{\mathrm{op}}\bigr\Vert_F}
$$

被当作**方法适用性的先验诊断**：低 $e_Q$ 表示该算子谱适合少量逐 mode atoms 表示。

**关键审计：必须分别报告 positive octant 与 full signed grid。** 秩 4 时：

| 机制 | positive octant $e_4$ | full signed grid $e_4$ |
|---|---:|---:|
| reference advection | 0.0323 | **0.1799** |
| shifted advection | 0.0242 | **0.1812** |
| anisotropic diffusion | 0.0030 | **0.0043** |

因此当前 advection 实验只能证明 **magnitude-spectrum prior 有效**，不能声称完整恢复了倾斜输运谱。完整版本需要 signed-conjugate components、复数 factors 或显式 cross-mode phase——本轮没有伪装成已实现。

### 6.3 为什么非负谱保证合法核

在一维周期网格上，对任意非负单边谱 $s(k)$ 定义特征映射

$$
\phi_s(x)=\Bigl[\;\sqrt{s(0)},\;\;\bigl\lbrace\sqrt{2s(k)}\cos(2\pi kx)\bigr\rbrace_{k=1}^{K},\;\;\bigl\lbrace\sqrt{2s(k)}\sin(2\pi kx)\bigr\rbrace_{k=1}^{K}\;\Bigr].
$$

于是

$$
k_s(x,x')=\phi_s(x)^{\top}\phi_s(x')=s(0)+2\sum_{k=1}^{K}s(k)\cos\bigl(2\pi k(x-x')\bigr)
$$

必然半正定。对非负凸组合 $s_{dr}(k)=\sum_q \pi_{drq}s_{dq}(k)$ 同样成立。因此从**非负 CP → routing softmax → 最终核**的每一步都不破坏 PSD。

> **必须写进 Method 而不是附录的边界**：当前 $s_{dq}(k)$ 是**单边 magnitude spectrum**，实 Fourier features 会把 $+k$ 与 $-k$ 成对。因此当前 hypothesis class 表示的是**逐轴 even、separable 的协方差分量**。
> anisotropic diffusion 的 symbol 由各轴平方项组成，与这个表示匹配，是论文**最干净的主例**。
> advection 的 $\omega_t+v_x\omega_x+v_y\omega_y$ 含 cross-sign coupling；只分解正 octant 再逐轴对称化，**不等价于**完整 transport spectrum。本文把它作为 limitation / stress test，而不是主理论例子。

### 6.4 Operator bank 与固定 generic support floor

最终候选谱库为

$$
\mathcal{B}_d=\bigl\lbrace s^{\mathrm{op}}_{d1},\ldots,s^{\mathrm{op}}_{dQ_{\mathrm{op}}}\bigr\rbrace\;\cup\;\bigl\lbrace s^{\mathrm{gen}}_{1},\ldots,s^{\mathrm{gen}}_{Q_{\mathrm{gen}}}\bigr\rbrace.
$$

POC 冻结 $Q_{\mathrm{op}}=Q_{\mathrm{gen}}=4$，generic atoms 分别覆盖 smooth、Matérn-like、oscillatory、broadband。**generic-only 是主要 baseline**。robust 版本把路由约束为

$$
\pi_{drq}=\rho_q+\Bigl(1-\sum_{j}\rho_j\Bigr)\,\mathrm{softmax}(\alpha_{dr})_q,
$$

其中 operator atoms 的 $\rho_q=0$，四个 generic atoms **均分总计 25% 的固定 floor**；routing 以 operator logits $=0$、generic logits $=-2$ 初始化。

> 25% 是在 development seeds 上**只做一次的预声明最小重试**，此后没有继续搜索 floor 值。

这**不是**"自由字典会自动适配"。Floor 明确用一部分 matched-prior 精度换取 support safety：

| development seeds 101–105 | matched NRMSE | strict wrong-support NRMSE |
|---|---:|---:|
| operator per-mode/rank | **0.0396 ± 0.0066** | 0.6195 ± 0.2149 |
| generic per-mode/rank | 0.1292 ± 0.1390 | — |
| operator + generic（25% floor） | 0.0467 ± 0.0085 | **0.0480 ± 0.0079** |

matched 上 robust 0/5 战胜 operator-only（平均代价 $+0.0071$），wrong-support 上 5/5 改善（平均收益 $-0.5714$）。这是明确的 **robustness–specificity tradeoff**。

### 6.5 Global 与 mode/rank routing

比较两种主要参数共享方式：

$$
\text{global:}\quad \pi_{drq}=\pi_q,
\qquad\qquad
\text{mode/rank:}\quad \pi_{drq}=\mathrm{softmax}(\alpha_{dr})_q .
$$

global baseline 强制所有 mode 和所有 CP rank 共享同一个谱；mode/rank 版本允许每个连续因子具有不同的物理尺度。

> $\alpha$ 是通过 ELBO 优化的**确定性 kernel hyperparameter**；当前没有为 routing 构造额外变分 posterior，因此 routing softmax **不能**被解释成 Bayesian model probability，UQ 也不包含 kernel-routing uncertainty。

### 6.6 本轮关键公平性修正：collapsed spectral mixture

旧 POC（expanded）为每个 atom 分配一套独立系数：

$$
f_{dr}(x)=\sum_{q}\sqrt{\pi_{drq}}\;\phi_{dq}(x)^{\top}a_{drq}.
$$

这在 prior covariance 上是正确的，但 8-atom robust bank 比 4-atom bank **多一倍变分系数**，方法比较因此混入了参数预算差异。

投稿确认实验改用等价的 canonical（collapsed）表示：先合成谱，再用**单套** features 与系数

$$
s_{dr}=\sum_q \pi_{drq}s_{dq},
\qquad
f_{dr}(x)=\phi_{s_{dr}}(x)^{\top}a_{dr},
\qquad
a_{dr}\sim\mathcal{N}(0,I).
$$

因为所有 atoms 使用相同的离散 Fourier support，两者的 prior covariance **相同**；collapsed 版本只是去掉了重复 features。其变分系数数量固定为

$$
2DR(1+2K),
$$

**不随 bank atom 数变化**（确认实验中恒为 156 个参数）。robust 版本只比 operator baseline 多 routing logits，不再多 GP coefficients。

> **分表纪律**：旧 expanded 结果与新 collapsed 确认结果**必须分表**，数值不得合并。collapsed 结果证明 escape 效应不依赖额外 GP coefficient budget。

### 6.7 变分推断

每个 Fourier 系数采用 mean-field Gaussian：

$$
q(a_{dr})=\mathcal{N}\bigl(\mu_{dr},\mathrm{diag}(\sigma_{dr}^{2})\bigr),
\qquad
p(a_{dr})=\mathcal{N}(0,I).
$$

CP core 权重 $c_r$、routing logits $\alpha$ 与观测噪声 $\sigma_y$ 是点估计。训练最大化

$$
\mathcal{F}=\mathbb{E}_{q}\Bigl[\sum_{i\in\mathcal{O}}\log p(y_i\mid f,c,\sigma_y)\Bigr]-\sum_{d,r}\mathrm{KL}\bigl[q(a_{dr})\,\Vert\,p(a_{dr})\bigr].
$$

实现细节：3 个 reparameterized MC samples 估计期望 log likelihood；Adam；400 steps。观测率 $\le 2\%$ 时使用**完整 observed batch**，因此代码中的 minibatch scaling 恒等于 1；loss 除以 $N_{\mathrm{obs}}$，相当于优化 per-observation negative ELBO。所有方法共享相同的学习率、gradient clipping、初始化随机流与 step 数。

> **数值细节**：strict wrong-support control 会产生**精确为零**的谱分量，而 $\sqrt{\cdot}$ 在 0 处导数发散。实现只在**构造 feature 时**做 `clamp_min(1e-12)`，既保持预期的零协方差（到数值精度），又避免 NaN routing 梯度。这一点有单元测试覆盖。

### 6.8 Predictive uncertainty 的准确表述

给定 mean-field coefficient posterior，代码既可精确计算 latent marginal 的一二阶矩，也可从 $q(a)p(y_\ast\mid a)$ 采样。确认实验在**训练完全结束后**，用 64 个 posterior predictive samples、在 1024 个未观测位置报告：95% 区间 coverage、MC posterior predictive NLL、区间宽度；评估目标是额外独立加噪的 held-out observations。

> 这个 UQ 是**有限 Fourier + mean-field variational posterior predictive**，不是解析 exact GP posterior；64-sample NLL 带有 Monte-Carlo 误差。论文必须使用这一限定。

---

## 7. 复杂度

| 项 | 规模 |
|---|---|
| 变分系数与尺度参数 | $2DRF$，**与 $Q$ 无关** |
| mode/rank routing 参数 | $DRQ$（global 为 $Q$） |
| 每个 MC sample 的观测预测 | $O(N_{\mathrm{obs}}DRF)$ |
| operator joint spectrum 网格 | $O\bigl((K+1)^{D}\bigr)$ 存储 |
| 非负 CP 分离 | 随分离迭代数、$Q_{\mathrm{op}}$ 与联合谱网格线性增长 |

当前 POC 的主要扩展瓶颈**不是** tensor completion，而是高维联合 spectrum grid 的 $O((K+1)^D)$ 存储。未来可用解析 symbol、稀疏频率采样或 tensor-train 表示替代完整网格。

---

## 8. 投稿确认实验协议

### 8.1 冻结协议

| 项 | 设定 |
|---|---|
| development / audit seeds | 101–105，**仅**用于发现数值问题与冻结协议，不进确认表 |
| untouched confirmation seeds | 201–205（5 个），协议冻结后**一次性**运行 |
| 网格 / 观测率 | $24^3$；2%（约 276 个训练观测） |
| functional CP rank | 2 |
| operator / generic atoms | 4 / 4 |
| frequency support | $k=0,\ldots,6$，每个 factor 13 个 Fourier features |
| optimizer | Adam，400 steps，3 个 ELBO samples；**无 validation、无 early stopping** |
| 共享项 | 同一 case/seed 的所有方法共享场、mask、训练噪声与 UQ targets |
| test 使用时点 | 所有未观测值**只在** 400 steps 完成后用于报告 |

### 8.2 三个算子设置

1. **reference advection–diffusion**：复现原始机制；
2. **shifted advection–diffusion**：改变两个扩散系数、两个速度分量、反应项与 forcing scale；
3. **strongly anisotropic diffusion**：三个轴的扩散系数差异显著。

每个 setting 都单独报告 rank-4 联合谱分离误差。它们在数据生成前就已冻结，**不是**从 test labels 拟合的超参数。

### 8.3 方法矩阵

| 组 | 方法 |
|---|---|
| operator-only | global / per-mode-rank |
| generic-only | global / per-mode-rank |
| oracle | oracle operator component route（**测量 prior 路由上限，非可部署方法**） |
| robust | operator+generic，固定 25% generic floor，global / per-mode-rank |
| strict mismatch | wrong-support operator-only（删除所有 $k\ge 2$ 的 operator support） |
| strict mismatch + escape | wrong-support + generic floor |

所有 learned 方法保持相同的 coefficient posterior 参数数与 400-step 预算。

### 8.4 主要指标与 gate

主要指标是 **held-out clean-field NRMSE**。辅助指标：induced-spectrum cosine / 相对 L2、predictive coverage / NLL、逐 seed paired wins。

**主线 gate（已收缩为两条）：**

- **G1**：matched setting 中，operator-derived per-mode/rank prior 相对 generic-only 与 operator-global 至少有一项**稳定、逐 seed 可复现**的优势；
- **G2**：anisotropic diffusion 的 **full signed** 联合谱在小 rank 下可压缩；advection 必须**分别**报告 positive octant 与 full signed error，只作为 cross-sign coupling stress test，**不进入**通过 gate 的正证据。

**次要 gate**：strict support mismatch 中，floor-robust 必须稳定改善 wrong-support operator-only，同时 matched case 的代价可量化且不过大。

---

## 9. 确认结果

### 9.1 Development audit：support floor 的代价与收益

数值见 [6.4 节表](#64-operator-bank-与固定-generic-support-floor)。结论：robust 在 matched case 0/5，平均代价 $+0.0071$；在 wrong-support case 5/5，平均收益 $-0.5714$。

> **一次被推翻的负结果（必须保留在论文/附录中）**：collapsed 参数化最初通过"第一个 atom 的 feature ÷ 第一个 atom 的 amplitude"反推公共 Fourier basis。当第一个 operator atom 的 $k\ge 2$ support 被置零时，公共高频 basis 也被错误地置零，于是 generic floor 虽有非零谱却**无法产生高频 feature**，产生了一个假的 negative result（robust $\approx$ wrong-support operator）。修复方式是直接从坐标解析构造 $[\,1,\;\sqrt{2}\cos,\;\sqrt{2}\sin\,]$ basis，并新增 zero-support feature、Gram PSD、finite routing gradient 三类单测。修复后**只重跑两个 strict controls**，matched 主表没有重训。修复前后对照保存在 `results/strict_support_basis_fix_audit.json`。

### 9.2 Untouched seeds 201–205 确认（主表）

所有数值来自协议冻结后一次性运行的 seeds 201–205，指标为 held-out NRMSE（mean ± std）。

| setting | operator-global | operator per-mode/rank | generic-global | generic per-mode/rank | oracle route |
|---|---:|---:|---:|---:|---:|
| reference advection | 0.0411 ± 0.0144 | 0.0462 ± 0.0332 | 0.0474 ± 0.0237 | **0.0343 ± 0.0077** | 0.0325 ± 0.0106 |
| shifted advection | 0.1006 ± 0.0586 | 0.0995 ± 0.0590 | **0.0892 ± 0.0369** | 0.2312 ± 0.2580 | 0.1042 ± 0.1041 |
| anisotropic diffusion | 0.1212 ± 0.0606 | **0.1183 ± 0.0582** | 0.3761 ± 0.5235 | 0.1567 ± 0.0990 | 0.1183 ± 0.0588 |

![2% 观测下的 frozen confirmation](../results/submission_confirmation/submission_confirmation_nrmse.png)

**读法：**

- **最清楚的确认信号来自 anisotropic diffusion**（数学上与 real separable spectrum 匹配）：operator per-mode/rank 相对参数匹配的 generic per-mode/rank，平均 NRMSE 降低约 **24.5%**，**5/5 paired wins**，并追平 oracle mean（0.1183 vs 0.1183）。
- 相对 operator-global 只有约 **2.4%** 均值改善、**3/5** wins。因此证据强烈支持 **operator-derived mode kernels**，但**不足以**把自由 per-rank routing 单独包装成第二个主要贡献。
- **advection 两个 setting 不形成一致 winner**。reference case 中 operator per-mode/rank 对 global 和 generic per-mode/rank 都有 4/5 paired wins，但 seed 203 的 $0.111$ 优化 outlier 使均值反转；shifted case 的 generic-global 均值最好。结合 signed-spectrum 审计，这两个结果更适合作为"当前 even/magnitude kernel 对倾斜输运不完整"的**边界证据**。

**anisotropic diffusion 的连续诊断与预测一致：**

| 方法 | induced-spectrum cosine ↑ | spectrum 相对 L2 ↓ | predictive NLL ↓ | 95% coverage |
|---|---:|---:|---:|---:|
| operator-global | 0.972 | 0.269 | −0.642 | 0.956 |
| operator per-mode/rank | **0.977** | **0.204** | **−0.679** | 0.947 |
| generic per-mode/rank | 0.926 | 0.427 | −0.451 | 0.960 |
| oracle | 1.000 | 0.000 | −0.678 | 0.946 |

operator per-mode/rank 的 coverage 接近名义 95%，NLL 与 oracle 基本相同。（再次提醒：这是 64-sample mean-field variational posterior predictive。）

### 9.3 Strict support mismatch 确认

| setting | wrong-support operator | 25% floor robust | paired wins |
|---|---:|---:|---:|
| reference advection | 0.6723 ± 0.1938 | **0.0402 ± 0.0096** | 5/5 |
| shifted advection | 0.6318 ± 0.0883 | **0.0847 ± 0.0414** | 5/5 |
| anisotropic diffusion | 0.6149 ± 0.1798 | **0.1301 ± 0.0621** | 5/5 |

在 matched anisotropic diffusion 上 robust 为 $0.1308\pm0.0624$，比 operator-only 的 $0.1183\pm0.0582$ 差约 $0.0124$；但在 prior 被删频时恢复到几乎相同的 $0.1301$。这正是固定 support floor 的**预期行为**：牺牲少量 specificity，换取 support 不可被优化关闭的保证。

### 9.4 Signed vs octant 可分性审计

![positive octant 与 full signed spectrum 分离误差](../results/signed_spectrum_audit/signed_vs_octant_separability.png)

| rank | ref. adv (oct / signed) | shift. adv (oct / signed) | aniso diff (oct / signed) |
|---:|---:|---:|---:|
| 1 | 0.2374 / 0.4552 | 0.1064 / 0.5528 | 0.0450 / 0.0713 |
| 2 | 0.1012 / 0.3664 | 0.0525 / 0.4036 | 0.0131 / 0.0203 |
| 3 | 0.0886 / 0.2218 | 0.0266 / 0.2042 | 0.0058 / 0.0078 |
| 4 | 0.0323 / 0.1799 | 0.0242 / 0.1812 | 0.0030 / 0.0043 |
| 5 | 0.0251 / 0.1280 | 0.0197 / 0.1742 | 0.0021 / 0.0029 |
| 6 | 0.0120 / 0.1233 | 0.0165 / 0.1666 | 0.0016 / 0.0021 |

advection 在单个 octant 上看起来可分，但对称扩展后仍存在**无法被当前逐轴实核表示的耦合**（signed 误差在 rank 6 仍 $>0.12$）；anisotropic diffusion 则没有这个缺口（rank 4 已达 0.0043）。

### 9.5 投稿 gate 判断：**收缩后的条件 GO**

| 判断 | 内容 |
|---|---|
| **GO** | anisotropic diffusion 上，operator-derived kernels 相对 generic kernels 有 5/5 paired prediction wins、更好的 induced spectrum 与 NLL；rank-4 full signed 谱误差仅 0.0043。 |
| **弱信号** | per-mode/rank routing 仅小幅优于 operator-global（3/5 wins，~2.4%），不能单独作为大贡献。 |
| **GO（secondary）** | 固定 25% generic support floor 在三类 strict mismatch 上全部 5/5 wins，matched setting 的代价可测且小。 |
| **NO-GO** | 当前 real separable features **不能**把 advection 的 positive-octant 低误差解释成完整倾斜输运谱。 |

**推荐论文主线：**

> 针对 **even / axis-separable** 的 operator spectra（以 anisotropic diffusion 为主例），把联合谱**非负投影**为逐维合法 GP kernels，并用**固定 generic support floor** 对抗错先验。Mode/rank routing 是实现组件与消融；advection 仍是明确 limitation。

---

## 10. 工程与统计审计清单

### 已通过

- 每个 atom 与 routed mixture 的频谱非负；Fourier Gram matrix PSD；
- 单边谱按 $s_0+2\sum_{k>0}s_k=1$ 归一化；
- operator、generic 与 robust 使用完全相同的 $k=0,\ldots,6$ support 上限；
- collapsed coefficient shape 为 `[mode, rank, feature]`，**不依赖 atom 数**；
- exact posterior moments 与大样本 MC 在单测容差内一致；
- full-observed-batch ELBO 的 likelihood 与 KL scaling 可直接审计；
- train mask 之后的所有索引都不进入 optimizer、early stopping 或超参选择；
- fixed route shape 为 `[mode, rank, atom]` 并显式归一化；
- zero-support feature、finite routing gradient 有单测覆盖。

### 仍有限制（必须写进论文 Limitations）

1. synthetic field 从**同一个有限 operator atom family** 采样，属于机制 sanity，不是外部证据；
2. mean-field posterior 忽略 CP factors 之间的 posterior correlation；
3. routing 是 point estimate，UQ **不包含** kernel-routing uncertainty；
4. 周期 Fourier prior 不处理边界条件与非平稳系数；
5. rank 固定为 2，没有与 RR-FBTC 的 rank learning 做正面竞争；
6. 非负 CP 可能存在局部最优与 component permutation；论文只使用重构误差与 induced spectrum，**不解释 component label**；
7. 联合谱 component weights $\lambda_q$ 当前只被记录，因与 CP core scale 不可识别而未进入 routing prior；
8. 训练核使用非负频率 octant；full signed 审计显示 advection 的 cross-sign coupling 更难分离。

---

## 11. 可证伪 claims 与下一步

### 本轮已验证或已否证

| # | claim | 结论 |
|---|---|---|
| 1 | even/axis-separable 的 anisotropic-diffusion **full signed** 谱可由少量一维 PSD atoms 高精度表示 | **验证**（rank-4 误差 0.0043） |
| 2 | 2% 观测下，operator-derived mode kernels 在 anisotropic diffusion 上胜过参数匹配的 generic kernels | **验证**（5/5 paired wins） |
| 3 | per-mode/rank routing 相对 operator-global 有独立贡献 | **弱信号**（3/5 wins，~2.4%），不独立成 claim |
| 4 | 自由选择即 support 保证 | **否证**；固定 25% generic floor 在 strict deletion 下三类 setting 均 5/5 恢复 |
| 5 | 上述结论在 bank-size-independent coefficient budget 下成立 | **验证**（collapsed 参数化，156 个系数恒定） |
| 6 | 核权重可直接识别 PDE / atom label | **否证**（R2：atom top-1 仅 22–33%；R5：相同 Fourier span 下 posterior 可补偿幅值失配） |

### 下一阶段的真正 gate

- 在**不是**从相同 finite atoms 采样的 PDE solutions 上，operator-derived kernels 仍需稳定优于 FunBaT / generic functional tensor；
- observation ratio 1% / 2% / 5% 与 structured sensor masks 下，优势需形成可解释的 phase diagram；
- 非周期边界、变系数与 operator coefficient mismatch 下，优势不能完全消失；
- 若要重新纳入 advection，必须实现并验证 signed-conjugate / 复数 / cross-mode phase factors；
- predictive UQ 需在独立 PDE 数据上维持合理 coverage / NLL，而不是只在 planted data 上成立。

### 投稿前仍需要

1. 至少一个**不由本模型 atom 直接生成**的 PDE solution dataset；
2. 与 FunBaT、RR-FBTC 和 neural / functional CP 的统一 sparse-mask baseline；
3. operator coefficient misspecification 的**连续曲线**，而不只有 matched / 删频两个端点；
4. 非负分离 rank、generic atom 数、tensor rank 与 observation ratio 的 ablation；
5. 若进入真实非周期域，明确边界误差以及 periodic-prior 的 failure case。

---

## 12. 复现入口

| 内容 | 路径 |
|---|---|
| 核心模型 | `src/geoaware/operator_spectral_funbat.py` |
| 新 collapsed 投稿确认 | `experiments/run_submission_confirmation.py` |
| signed vs octant 审计 | `experiments/audit_signed_operator_spectrum.py` |
| basis 修复后重跑 strict controls | `experiments/rerun_strict_support_after_basis_fix.py` |
| generic floor development | `experiments/run_escape_floor_development.py` |
| 旧 expanded POC | `experiments/run_operator_spectral_poc.py` |
| 新结果（主表来源） | `results/submission_confirmation/summary.json` |
| signed 审计结果 | `results/signed_spectrum_audit/summary.json` |
| 旧 expanded 结果（**不得与主表合并**） | `results/advanced_poc_r1_r5/` |
| 单元测试 | `tests/test_operator_spectral_funbat.py` |
| 论文 LaTeX 工程 | `paper/` |

---

## 附录 A：确认实验完整数值表

seeds 201–205，2% 观测，400 steps，rank 2，$k\le 6$。格式：mean ± std。

### A.1 reference advection–diffusion

| 方法 | NRMSE | spectrum cosine | spectrum rel. L2 | 95% coverage | predictive NLL |
|---|---:|---:|---:|---:|---:|
| operator-global | 0.0411 ± 0.0144 | 0.8337 ± 0.0149 | 0.6936 ± 0.0325 | 0.9963 | −1.1080 |
| operator per-mode/rank | 0.0462 ± 0.0332 | 0.8175 ± 0.0641 | 0.6278 ± 0.1765 | 0.9949 | −0.9443 |
| generic-global | 0.0474 ± 0.0237 | 0.8413 ± 0.0011 | 0.6867 ± 0.0065 | 0.9932 | −1.0311 |
| generic per-mode/rank | 0.0343 ± 0.0077 | 0.8239 ± 0.0138 | 0.6923 ± 0.0273 | 0.9973 | −1.1414 |
| robust-global | 0.0392 ± 0.0120 | 0.8340 ± 0.0110 | 0.6885 ± 0.0232 | 0.9965 | −1.1244 |
| robust per-mode/rank | 0.0384 ± 0.0100 | 0.8568 ± 0.0915 | 0.5876 ± 0.2170 | 0.9945 | −1.1624 |
| oracle route | 0.0325 ± 0.0106 | 1.0000 | 0.0000 | 0.9811 | −1.2810 |
| wrong-support operator | 0.6723 ± 0.1938 | 0.8467 ± 0.0877 | 0.5958 ± 0.2302 | 0.8197 | +1.1223 |
| wrong-support robust | 0.0402 ± 0.0096 | 0.8583 ± 0.0655 | 0.6111 ± 0.1674 | 0.9955 | −1.1388 |

### A.2 shifted advection–diffusion

| 方法 | NRMSE | spectrum cosine | spectrum rel. L2 | 95% coverage | predictive NLL |
|---|---:|---:|---:|---:|---:|
| operator-global | 0.1006 ± 0.0586 | 0.9281 ± 0.0106 | 0.5210 ± 0.0448 | 0.9838 | −0.6995 |
| operator per-mode/rank | 0.0995 ± 0.0590 | 0.9248 ± 0.0326 | 0.5172 ± 0.1488 | 0.9811 | −0.7226 |
| generic-global | 0.0892 ± 0.0369 | 0.9243 ± 0.0004 | 0.4973 ± 0.0013 | 0.9834 | −0.7315 |
| generic per-mode/rank | 0.2312 ± 0.2580 | 0.8613 ± 0.0621 | 0.5509 ± 0.0696 | 0.9531 | −0.1977 |
| robust-global | 0.0832 ± 0.0379 | 0.9322 ± 0.0065 | 0.4969 ± 0.0154 | 0.9881 | −0.7895 |
| robust per-mode/rank | 0.0796 ± 0.0384 | 0.9453 ± 0.0334 | 0.4160 ± 0.1411 | 0.9848 | −0.8442 |
| oracle route | 0.1042 ± 0.1041 | 1.0000 | 0.0000 | 0.9742 | −0.8129 |
| wrong-support operator | 0.6318 ± 0.0883 | 0.9071 ± 0.0300 | 0.6297 ± 0.1340 | 0.8436 | +1.0208 |
| wrong-support robust | 0.0847 ± 0.0414 | 0.9349 ± 0.0328 | 0.4602 ± 0.1479 | 0.9873 | −0.7661 |

### A.3 strongly anisotropic diffusion（主例）

| 方法 | NRMSE | spectrum cosine | spectrum rel. L2 | 95% coverage | predictive NLL |
|---|---:|---:|---:|---:|---:|
| operator-global | 0.1212 ± 0.0606 | 0.9718 ± 0.0066 | 0.2694 ± 0.0510 | 0.9561 | −0.6415 |
| **operator per-mode/rank** | **0.1183 ± 0.0582** | **0.9768 ± 0.0103** | **0.2043 ± 0.0655** | 0.9467 | **−0.6794** |
| generic-global | 0.3761 ± 0.5235 | 0.8946 ± 0.1091 | 0.4547 ± 0.1633 | 0.9158 | −0.0032 |
| generic per-mode/rank | 0.1567 ± 0.0990 | 0.9259 ± 0.0356 | 0.4268 ± 0.0868 | 0.9600 | −0.4510 |
| robust-global | 0.1314 ± 0.0628 | 0.9731 ± 0.0039 | 0.3071 ± 0.0205 | 0.9549 | −0.5768 |
| robust per-mode/rank | 0.1308 ± 0.0624 | 0.9757 ± 0.0088 | 0.2788 ± 0.0424 | 0.9535 | −0.5819 |
| oracle route | 0.1183 ± 0.0588 | 1.0000 | 0.0000 | 0.9463 | −0.6783 |
| wrong-support operator | 0.6149 ± 0.1798 | 0.9731 ± 0.0076 | 0.2714 ± 0.0480 | 0.8477 | +1.0177 |
| wrong-support robust | 0.1301 ± 0.0621 | 0.9782 ± 0.0073 | 0.2721 ± 0.0337 | 0.9578 | −0.5800 |

### A.4 Paired-win 汇总（5 seeds）

| 对比 | ref. adv | shift. adv | aniso diff |
|---|---:|---:|---:|
| operator per-mode/rank vs operator-global | 4/5 | 3/5 | 3/5 |
| operator vs generic（matched） | 4/5 | 4/5 | **5/5** |
| robust vs operator（matched） | 2/5 | 3/5 | 0/5 |
| robust escape vs wrong-support | **5/5** | **5/5** | **5/5** |

> 注：`operator vs generic` 在 advection 上虽有 4/5 wins，但均值被单个 seed 的优化 outlier 反转（reference case），因此不作为正证据；只有 anisotropic diffusion 的 5/5 同时满足均值与逐 seed 一致性。
