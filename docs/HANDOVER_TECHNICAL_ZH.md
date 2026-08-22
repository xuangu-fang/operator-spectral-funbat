# 交接技术文稿：算子谱先验用于受限传感器布局下的场重构

写给接手这个项目的人。目标是让你**不需要读完 71 个实验脚本**就能知道：
这个方法在做什么、每一步公式是怎么来的、哪些实验是可信的、
数据从哪来、以及哪些坑我已经踩过。

**想先把东西跑起来** → [`GETTING_STARTED_ZH.md`](GETTING_STARTED_ZH.md)（约 20 分钟出第一张主表）。
**想知道未来做什么** → [`FUTURE_BRANCHES_ZH.md`](FUTURE_BRANCHES_ZH.md)。

> **这份是唯一的主文档。** 此前的 11 份文档已合并到这里或删除
> （历史版本在 git 里，`git log --diff-filter=D --name-only -- docs/` 可以找回）。
> 删掉的多是会话总结、迭代日志、以及主线转向前写的投稿计划——
> 它们里面的有效内容都已并入本文，而其中的数字有一部分已被推翻，
> 留着只会误导。
>
> **公式渲染**：本文所有公式已通过 `tools/check_github_math.py`。
> 改动后请重新运行——GitHub 上 `$$` 块前面没有空行就**不会渲染**，
> 这个仓库为此坏过 54 处公式，源码看着好好的，网页上是散着美元符号的散文。

---

## 1. 背景与动机：这个问题为什么存在

### 1.1 一个具体场景

一个厂房里有气体泄漏。你想知道**整个房间**的浓度分布——哪里最浓、往哪个方向扩、
要不要疏散。但你能装传感器的地方由不得你选：

- 房间中央是设备和通道，**不能布线也不能悬挂**；
- 能开孔的只有**墙面**，或者某一面可达的墙；
- 燃烧室只能在壁面开测点，管道监测只能贴外壁，
  地下水污染监测只能打有限几口井。

于是观测集中在域的一小块**连续区域**里，而你要重建的是**整个场**。

这个约束在很多物理场重构里是常态而非例外：**传感器的位置是由工程可达性决定的，
不是由信息量决定的**。

### 1.2 为什么这不是标准的张量补全设定

张量补全文献默认的缺失模式是**均匀随机**。这两种设定的差别不是程度问题，是**性质**问题：

| | 随机缺失 | 受限布局 |
|---|---|---|
| 未观测点的邻居 | **总是有**观测过的邻居 | 大部分点**一个都没有** |
| 任务本质 | 邻居之间**内插** | 向从未观测的区域**外推** |
| 光滑先验够不够 | **够**——"邻近点相似"是充分描述 | **不够**——见 §1.3 |
| 实测（1% 观测） | 三种方法打平在 0.053 | 通用方法崩到 0.67–0.96 |

最后一行是我们自己测的：随机掩码下我们的方法和通用核**完全打平**。
这不是失败，是**应该的**——那个设定里物理确实没用。

### 1.3 光滑先验在外推时说了什么

平稳核（Matérn / RBF）说的只有一句话：**相关性随距离衰减**。

在内插时这句话够用：目标点旁边就有观测，"和邻居相似"直接给出了答案。

但在外推时，同一句话变成：离观测区一远，相关性趋于零，**所以场趋于其均值**。

这是一个关于**我们的无知**的陈述，不是关于**场**的陈述。它没有说错，但它什么也没说。

实测能看到这一点：把传感器限制在一面墙上，用在传感器数据上调出来的 Matérn 重建，
**离墙一远整个场就塌回 0**（见 `results/leak/figure_reconstruction.png` 第三列）。

### 1.4 物理提供的恰好是缺的那句话

这里的关键观察是：**方程的形式几乎总是已知的**，哪怕系数不知道、解不知道。

一个工程师往往不知道场长什么样，却知道自己面对的是**扩散、输运还是波动**。
而对耗散算子，方程直接给出了**场沿每个轴如何衰减**——

$$
\mathcal{L} u = w \quad\Longrightarrow\quad S_u(\boldsymbol{\xi}) = \bigl\lvert \hat{\mathcal{L}}(\boldsymbol{\xi}) \bigr\rvert^{-2} S_w(\boldsymbol{\xi}) .
$$

算子的符号 $\hat{\mathcal{L}}$ 逐频率地说明**哪些频率被压制、压制多少**。
这正是外推所缺的那条信息：

> 光滑核只知道"远处相关性低"；
> 算子知道"**远处的值应该是多少**"。

举个能直接对上的例子。各向异性扩散 $\partial_t u = D_x \partial_{xx} u + D_y \partial_{yy} u - r u$
的符号是 $i\omega + r + D_x k_x^2 + D_y k_y^2$。$D_x \neq D_y$ 意味着两个空间方向的
**衰减律不同**——沿 $x$ 平滑、沿 $y$ 粗糙。一个共享单一长度尺度的通用核**没有办法表达这件事**，
而它是方程免费给出的。

### 1.5 更尖锐的问题：受限布局下连"调参"都做不到

上面那条还可以反驳："通用核也有超参，调一调不就行了？"

**在这个设定里调不了，而且原因是结构性的。**

调参需要**与预测目标相像的验证数据**。传感器全在一面墙上时，
你能留出来做验证的每一个点**也在那面墙上**。于是：

- 验证衡量的是**条带内部的内插**；
- 部署要求的是**横跨整个房间的外推**。

两者要的核不是一回事，而验证集**看不到**这个差别。

实测（同一网格、同一宿主，只差谁选的 $\ell$）：

| 布局 | 验证选出的 $\ell$ | 测试集实际需要的 $\ell$ | 调参代价 |
|---|---|---|---|
| 随机 | 0.8 | 0.8 | **+0.0003** |
| 单面墙 | 0.5, 0.5, 0.8 | **1.6, 2.4, 2.4** | **+0.1760** |
| 3D 单面墙 | 0.12 | **3.5** | **+0.2412** |

散布传感器时验证选得**准**，调参零代价；受限时它一致地选出**短 3–5 倍**（3D 上短 30 倍）的尺度。
这不是噪声，是系统性的——因为它衡量的本来就是另一件事。

**所以这个 setting 里，"能不能不用调参数据就给出核"本身就是价值。**

### 1.6 我们做的事，一句话

> 把算子族**不可分**的联合谱做**非负、保半正定**的低秩投影，
> 得到 functional tensor 每个 mode 的一维 GP kernel；
> 只需要方程的**形式**和系数的一个**范围**，不需要任何调参数据。

技术上要跨过的坎在 §2.2：算子给出的联合谱 $S_{\mathrm{op}}(\omega, k_x, k_y)$
**天然不可分**，而 functional tensor 要的是每个 mode 一条一维谱。
本方法的核心就是这个投影怎么做才**既可分又仍然是合法的核**。

### 1.7 三次转向（每次都是被自己的实验逼的）



**转向一：从"随机稀疏"到"受限布局"。**
在随机掩码下，我们和通用核**打平**（0.053 vs 0.053）。
这不是失败而是应该的：随机采样下每个未观测点都有邻居，重构是**内插**，
光滑先验已经是充分描述。物理只在**外推**时才有话说，
而外推来自**传感器只能装在墙上/某个局部**这一现实约束。

**转向二：从"我们赢 baseline"到"这里没人能调参"。**
一度报出"单面墙 +17.2%"。后来发现那是我把 Matérn 的长度尺度网格
`(0.12, 0.32, 0.8)` 扫窄了——它自己的最优在 $\ell = 2.4$。
放宽后 Matérn 从 0.6568 降到 0.5449，我们 0.5448，**打平**。
这个头条**已撤回**。

真正站得住的是另一件事：受限布局下**超参无法从数据中选出**。
你能留出来做验证的点全在传感器所在的那一小块里，
它衡量的是**块内内插**，而任务是**横跨房间外推**——两者要的核不是一回事。

**转向三：承认"固定一个常数"也不需要调参。**
一个实践者可以既不调参、也不用方程，直接固定一个中庸的 $\ell$。
实测 2D 上 $\ell = 1.6$ 相对我们最差只差 0.0135。
所以在 2D 上，**物理相对"随手固定一个合理常数"几乎没买到东西**。
剩下的价值集中在：跨场景时那个"合理常数"是多少会变（3D 上需要的 $\ell$ 跨 20–30 倍），
而方程自动给出它。

### 1.8 现在的 claim（只剩这三条）

1. **调参代价随传感器受限程度单调上升**，且与算子族无关：
   散布传感器 $+0.000\ldots+0.003$，单面墙 $+0.20\ldots+0.34$。
   这是 **setting 的性质**，与我们的方法好坏无关。
2. **同一份物理，当先验比当残差惩罚值钱一个数量级**：
   单面墙上谱先验值 $0.26$，PINN 残差值 $0.02$。
3. **只需要方程形式 + 系数的一个范围**：
   知道 $\theta^\star \in \times[1/3, 3]$ 相对知道真值只差 $0.0018$，
   而 PINN 和 AutoIP **必须承诺一个具体的 $\theta$**。

---

## 2. 核心推导（完整展开）

### 2.1 从算子到解的功率谱

设场 $u$ 由常系数线性算子 $\mathcal{L}$ 驱动，受随机强迫 $w$：

$$
\mathcal{L} u = w .
$$

做时空 Fourier 变换，记 $\hat{\mathcal{L}}(\boldsymbol{\xi})$ 为算子的**符号**
（$\boldsymbol{\xi} = (\omega, k_x, k_y)$）。线性系统的输出谱是

$$
S_u(\boldsymbol{\xi}) \;=\; \bigl\lvert \hat{\mathcal{L}}(\boldsymbol{\xi}) \bigr\rvert^{-2} \, S_w(\boldsymbol{\xi}) .
$$

取 $w$ 为白噪（$S_w \equiv \sigma^2$），得到本文一切的出发点：

$$
S_{\mathrm{op}}(\boldsymbol{\xi}) \;\propto\; \bigl\lvert \hat{\mathcal{L}}(\boldsymbol{\xi}) \bigr\rvert^{-2} .
$$

**主线算子**是各向异性反应-扩散

$$
\partial_t u \;=\; D_x \partial_{xx} u + D_y \partial_{yy} u - r u + w ,
$$

其符号为 $\hat{\mathcal{L}} = i\omega + r + D_x k_x^2 + D_y k_y^2$，于是

$$
S_{\mathrm{op}}(\omega, k_x, k_y) \;=\; \frac{1}{\omega^2 + \bigl(r + D_x k_x^2 + D_y k_y^2\bigr)^2} .
$$

**离散版本**（代码里真正用的）。在 $n$ 个格点、无流边界下，
二阶差分算子的特征值不是 $k^2$ 而是

$$
\lambda(k) \;=\; \frac{2 - 2\cos(\pi k / n)}{h^2}, \qquad h = 1/n .
$$

> 实现：`run_leak_sensors.neumann_eigenvalues`。
> **这一步曾经出过大错**：早期版本用手写的 index-space 系数 $(1.0, 0.3)$，
> 而物理系数是 $(0.02, 0.006)$，导致先验衰减快了约 4 倍。
> 必须从**离散算子的特征值**构造，不能从连续 $k^2$ 直接套。

时间维取 $\omega_j = \pi j / (T \Delta t)$，$T$ 是记录帧数。合起来：

$$
S_{\mathrm{op}}[j, a, b] \;=\; \Bigl(\omega_j^2 + \bigl(r + D_x \lambda_x(a) + D_y \lambda_y(b)\bigr)^2\Bigr)^{-1} .
$$

### 2.2 结构冲突：为什么不能直接用这个联合谱

$S_{\mathrm{op}}$ 是三维张量，而 functional tensor 模型要的是**每个 mode 一条一维谱**
（每个 mode 一个独立的 GP kernel）。二者不兼容，因为

$$
S_{\mathrm{op}}(\omega, k_x, k_y) \;\neq\; S_t(\omega)\, S_x(k_x)\, S_y(k_y)
$$

对任何一组一维函数都成立——分母里的 $\bigl(r + D_x k_x^2 + D_y k_y^2\bigr)^2$
**天然不可分**。

这就是本方法要解决的技术核心：**把一个不可分的联合谱投影成可分的、且仍然合法（非负）的每维谱**。

### 2.3 非负 CP 分离

用秩 $Q$ 的**非负** CP 分解逼近联合谱：

$$
S_{\mathrm{op}} \;\approx\; \sum_{q=1}^{Q} c_q \; s^{(t)}_q \otimes s^{(x)}_q \otimes s^{(y)}_q ,
\qquad s^{(d)}_q \ge 0 .
$$

非负是**必须的**而非美观：谱必须非负才对应一个合法的（半正定）核。
若用普通 CP，因子可以取负，得到的"核"不是核。

求解用乘性更新（Euclidean loss），第 $d$ 个 mode 的更新为

$$
s^{(d)} \;\leftarrow\; s^{(d)} \odot \frac{X_{(d)} \, K_{(d)}}{s^{(d)} \bigl(K_{(d)}^\top K_{(d)}\bigr) + \varepsilon} ,
$$

其中 $X_{(d)}$ 是沿 mode $d$ 的展开矩阵，$K_{(d)}$ 是其余因子的 Khatri–Rao 积。

> 实现：`nonnegative_cp_spectrum`。已泛化到**任意阶**（3D 场景需要 4 阶）。

**掩码版本。** 数据做了去均值，这恰好抹掉了**联合模态** $(0,0,0)$。
早期实现是把该元素**置零**再分离，这是个真实的 bug：

一个秩 $Q$ 的**可分**模型无法只在一个角上取零，它只能通过压掉**某个因子的 $k=0$** 来实现，
而它会压掉最"便宜"的那个。实测压掉的是时间维——
真实场沿时间近乎常数（$k=0$ 占 0.837 的能量），而置零版先验只给了 0.107，
把 0.514 压在 $k=1$ 上，**先验在和数据对着干**。

正确做法是让 CP **忽略**该元素而不是拟合一个零。带掩码 $M$ 时 Gram 捷径失效，
必须显式加权：

$$
s^{(d)} \;\leftarrow\; s^{(d)} \odot \frac{\bigl(M_{(d)} \odot X_{(d)}\bigr) K_{(d)}}{\bigl(M_{(d)} \odot (s^{(d)} K_{(d)}^\top)\bigr) K_{(d)} + \varepsilon} .
$$

逐模态 KL（与真实场的经验谱相比）：时间维 $1.578 \to 0.455$，总和 $2.358 \to 1.711$
（对照：调好的 Matérn 是 4.204）。重构 $0.5496 \to 0.5448$。

### 2.4 从谱到核：Mercer 与有限特征

给定一维谱 $s_d$ 和一组正交基 $\lbrace \phi_{dq} \rbrace$，核由 Mercer 展开给出：

$$
k_d(z, z') \;=\; \sum_{q} s_d(q) \, \phi_{dq}(z) \, \phi_{dq}(z') .
$$

关键点：当 $\lbrace \phi_{dq} \rbrace$ **就是算子的本征函数**时，
这个有限展开不是近似而是**精确的** Mercer 特征映射。
所以这里用 inducing points 反而是**严格的降级**——
它会用低秩近似去逼近一个本来就精确的有限秩表示。

实现上，特征取

$$
\Phi_d(z) \;=\; \bigl[\, \sqrt{s_d(0)}\,\phi_{d0}(z), \; \ldots, \; \sqrt{s_d(Q)}\,\phi_{dQ}(z) \,\bigr] ,
$$

于是 $k_d(z,z') = \Phi_d(z)^\top \Phi_d(z')$ 自动半正定。

> 实现：`ModeAdaptiveVariationalTucker._collapsed_features`。
> $\sqrt{\cdot}$ 在 0 处导数无穷，代码只在**构造特征时**做 $10^{-12}$ 截断，
> 保持"严格零支撑"控制实验的语义，同时避免 NaN 梯度。

### 2.5 边界条件决定本征基（这一条是 load-bearing 的）

算子的本征基由**边界条件**决定，不是由符号决定：

- 周期边界 $\Rightarrow$ 复指数 $e^{i 2\pi k x}$；
- **无流（Neumann）边界** $\Rightarrow$ 余弦 $\phi_k(x) = \sqrt{2}\cos(\pi k x)$，$\phi_0 \equiv 1$。

房间的墙是无流的，初值数据沿时间也**不是周期的**。
强行施加 $f(0) = f(1)$ 是一个很大的非物理约束。

**实测**（单面墙，3 seeds，只换基）：周期 Fourier **0.5781** vs 无流余弦 **0.5448**。

归一化也随之改变。周期基下平均逐点方差是 $s_0 + 2\sum_{k>0} s_k$；
余弦基下每个特征的均方都是 1，所以是 $\sum_k s_k$。

> 实现：`real_cosine_basis`、`normalize_spectrum_cosine`。用错归一化会引入一个
> 系统性的整体尺度偏差，而它看起来像"方法不好"。

### 2.6 宿主模型：为什么必须是 Tucker

真实二维场是**一堆 blob**，不是外积。它的 multilinear rank 小，但 CP rank 不小——
稀疏观测能识别的秩下根本表示不了。所以宿主用 Tucker：

$$
u(t, x, y) \;\approx\; \sum_{p,q,r} G_{pqr} \, f^{(t)}_p(t) \, f^{(x)}_q(x) \, f^{(y)}_r(y) ,
$$

每个 $f^{(d)}_\bullet$ 是上面构造的核下的 GP。**逐 mode 独立的秩**是刻意的：
multilinear rank 为 $[2, 11, 11]$ 的场需要 $2 \cdot 11 \cdot 11 = 242$ 的核，
而强行取等秩 11 需要 $1331$——这个差别决定模型在真实观测量下是否可辨识。

**collapsed 参数化**（公平性关键）。若每个 atom 各配一套系数，
8-atom bank 的变分参数是 4-atom 的两倍，方法对比就混入了参数预算差异。
collapsed 形式先混合谱、再用**一套**系数，变分系数量 $2DR(1 + 2K)$
**与 bank 大小无关**。所有表都用 collapsed。

### 2.7 混合权重与知识档位

每个 mode 的谱是 bank 内 atom 的凸组合，权重 $\pi$ 由 softmax 参数化并从数据学：

$$
s_d \;=\; \sum_{q} \pi_{dq} \, s_{dq}, \qquad \pi_{dq} \ge 0, \quad \textstyle\sum_q \pi_{dq} = 1 .
$$

`routing="global"` 表示三个 mode 共享一组 $\pi$。
（`per_mode` / `per_mode_rank` 在 1% 观测下**过拟合**，实测更差，主表一律用 global。）

**知识档位**由 bank 怎么造来定义：

| 档 | 学习者知道什么 | bank 构造 |
|---|---|---|
| K2 | 真值 $\theta^\star$ | 分离 $S_{\mathrm{op}}(\theta^\star)$，得 $Q$ 个 atom |
| K1 | $\theta^\star \in \Theta_1$（预先声明的范围） | 从 $\Theta_1$ 采 $M$ 组，各自分离，atom **池化** |
| K0 | $\theta^\star \in \Theta_0 \supset \Theta_1$ | 同上，范围更宽 |
| K$-$1 | 无算子信息 | generic dictionary |

在 K1/K0 下，$\pi$ 实际在做**软参数推断**——它在"物理可达的谱"张成的集合内选择，
而不是在无约束核空间里搜索。

**这正是与 PINN / AutoIP 的结构性差别**：那两者必须先承诺一个具体的 $\theta$
才能写出残差或虚拟观测，没有在参数族上摊开的机制。

### 2.8 推断

变分下界，因子系数取对角高斯后验，重参数化采样：

$$
\mathcal{F} \;=\; \mathbb{E}_{q}\bigl[\log p(\mathbf{y} \mid \mathbf{a})\bigr] \;-\; \mathrm{KL}\bigl(q(\mathbf{a}) \,\Vert\, p(\mathbf{a})\bigr) .
$$

先验是标准正态（尺度已经吸进特征的 $\sqrt{s_d}$ 里），所以 KL 有闭式。
只在观测到的 entry 上计算，$64^3$ 的张量在训练中从不被物化。

---

## 2A. 算法表：公式、代码与输入输出的对应

下面三段伪代码把 §2 的推导落到可执行的粒度。**每一行都标了对应的公式和函数名**，
接手时可以逐行对照，不必猜。

**三段的分工**：算法 1 只用方程，**不碰任何数据**；算法 2 只用观测数据拟合；
算法 3 是评估协议，它决定了主表里那三档是怎么来的。
一个常见的误解是"物理在训练时起作用"——**不是的，物理全部在算法 1 里，
在看到任何数据之前就固定下来了**，算法 2 只学混合权重和变分系数。

### 记号

| 符号 | 含义 | 代码里的东西 |
|---|---|---|
| $D$ | mode 数（2D 场为 3，3D 场为 4） | `len(shape)` |
| $n_d$ | 第 $d$ 个 mode 的格点数 | `shape[d]` |
| $K_d$ | 第 $d$ 个 mode 的频率数 | `BINS[d]` |
| $R_d$ | 第 $d$ 个 mode 的 Tucker 秩 | `RANKS[d]` |
| $Q$ | bank 里的 atom 数 | `atoms`，主线为 4 |
| $\theta$ | 算子系数 $(D_x, D_y, r)$ | `NOMINAL` |
| $S_{\mathrm{op}}$ | 联合谱，$K_1 \times \cdots \times K_D$ | `joint` |
| $s_{dq}$ | 第 $d$ 个 mode 的第 $q$ 条谱，长 $K_d$ | `spectra[d][q]` |
| $\Phi_d$ | 第 $d$ 个 mode 的本征基，$n_d \times K_d$ | `bases[d]` |
| $\pi_{dq}$ | 混合权重 | `routing_weights()` |
| $\mathbf{a}^{(d)}$ | 变分系数，$R_d \times K_d$ | `variational_mean[d]` |
| $\mathcal{G}$ | Tucker 核，$R_1 \times \cdots \times R_D$ | `core` |
| $\Omega$ | 观测索引集合，$\lvert \Omega \rvert \times D$ | `observed` |
| $\bar{\Omega}$ | 留出索引集合 | `test` |

**主线的具体数值**：$D = 3$，$(n_t, n_x, n_y) = (64,64,64)$，$(K_t,K_x,K_y) = (12,12,12)$，
$(R_t,R_x,R_y) = (8,5,5)$，$Q = 4$，$\lvert \Omega \rvert = 2621$（$1\%$）。
3D 场是 $D = 4$，$32^4$，$K = (8,8,8,8)$，$R = (5,4,4,4)$。

**参数量**：变分系数 $\sum_d R_d K_d = 8{\cdot}12 + 5{\cdot}12 + 5{\cdot}12 = 216$，
乘 2（均值与方差）；Tucker 核 $8{\cdot}5{\cdot}5 = 200$；混合权重 $Q = 4$。
合计约 $636$ 个参数去拟合 $2621$ 个观测。
**注意混合权重只有 4 个——"物理省下的是整个核，只留 4 个数要学"就是指这里。**

---

### 算法 1：从算子构造逐维核 bank

> **输入** 算子族与系数 $\theta$（或范围 $\Theta$）、网格 $\lbrace n_d \rbrace$、
> 频率预算 $\lbrace K_d \rbrace$、atom 数 $Q$、时间步长 $\Delta t$
> **输出** 每个 mode 一组非负谱 $\lbrace s_{dq} \rbrace_{q=1}^{Q}$，以及本征基 $\lbrace \Phi_d \rbrace$
> **不需要** 任何观测数据

```
 1  for d in spatial modes:                        # §2.1 离散算子的特征值
 2      λ_d[k] ← (2 − 2 cos(π k / n_d)) / h_d²         [K_d]      neumann_eigenvalues
 3  ω[j]  ← π j / (T Δt)                               [K_t]
 4
 5  for each (j, a, b):                            # §2.1 联合谱，不可分
 6      S_op[j,a,b] ← 1 / ( ω[j]² + (r + D_x λ_x[a] + D_y λ_y[b])² )
 7                                                     [K_t,K_x,K_y]
 8  M ← ones_like(S_op);  M[0,…,0] ← 0             # §2.3 掩码而非置零
 9  {s_dq} ← NonnegCP(S_op, rank=Q, mask=M)            D 组 [Q,K_d]
10  for d: s_dq ← s_dq / Σ_k s_dq[k]               # §2.5 余弦基下的归一化
11
12  for d:                                         # §2.5 边界条件决定本征基
13      Φ_d ← [ 1 , √2·cos(π k x) ]                     [n_d,K_d]  real_cosine_basis
14  return {s_dq}, {Φ_d}
```

| 行 | 为什么这样写 | 写错会怎样 |
|---|---|---|
| 2 | 用**离散**算子的特征值，不是连续的 $k^2$ | 早期版本用手写 index-space 系数 $(1.0,0.3)$ 而物理值是 $(0.02,0.006)$，先验衰减快了 4 倍 |
| 6 | 分母里 $\bigl(r + D_x\lambda_x + D_y\lambda_y\bigr)^2$ **天然不可分**，这是整个方法要解决的问题 | 直接按 mode 边缘化会丢掉轴间耦合，实测比分离差 0.025 |
| 8 | 数据去均值只抹掉**联合模态** $(0,0,0)$；掩码让 CP **忽略**它 | 置零会迫使可分模型压掉某个因子的 $k{=}0$，实测压掉了时间维（KL $1.578\to0.455$） |
| 10 | 余弦基下平均方差是 $\sum_k s_k$，不是周期基的 $s_0 + 2\sum_{k>0}s_k$ | 引入系统性整体尺度偏差，看起来像"方法不好" |
| 13 | 无流边界 $\Rightarrow$ 余弦；房间的墙就是无流的 | 用周期 Fourier：单面墙上 $0.5781$ vs $0.5448$ |

**代价**：$O(\text{steps} \cdot Q \prod_d K_d)$，主线约 $1200 \times 4 \times 12^3 \approx 8\times10^6$ 次乘加，
**不到一秒，且整个实验只做一次**（与 seed、布局、观测量都无关）。

**K1 档的唯一改动**（§2.7）：把第 5–9 行放进对 $\Theta_1$ 的采样循环，
每组 $\theta_m$ 分离出 $Q_1$ 个 atom，最后把 $M \cdot Q_1$ 个 atom **池化**成一个 bank：

```
 5' for m = 1 … M:
 6'     θ_m ← LogUniform(Θ₁)                        确定性 seed，Latin hypercube
 7'     S_op^(m) ← 式 (2.2) with θ_m
 8'     {s_dq^(m)} ← NonnegCP(S_op^(m), rank=Q₁, mask=M)
 9' {s_d·} ← concat over m                                   # atom 数变成 M·Q₁
```

> **公平性**：与 K1 bank 比较的 generic 字典必须有**完全相同的 atom 数**。
> collapsed 参数化保证变分系数量与 bank 大小无关，所以这个比较不被参数预算污染。

---

### 算法 2：拟合（变分推断）

> **输入** 观测索引 $\Omega$ 与读数 $\mathbf{y}$、bank $\lbrace s_{dq} \rbrace$、
> 本征基 $\lbrace \Phi_d \rbrace$、秩 $\lbrace R_d \rbrace$、步数 $N$、学习率 $\eta$
> **输出** 后验均值参数 $\lbrace \mathbf{a}^{(d)} \rbrace$、$\mathcal{G}$、混合权重 $\pi$
> **不使用** 任何留出数据（我们这一臂没有验证集）

```
 1  初始化 a^(d) ~ N(0, 0.12²),  log σ^(d) ← −2.5,  G ~ N(0, 1/√∏R_d),  logits ← 0
 2  for step = 1 … N:
 3      π_d ← softmax(logits)                                  routing="global": 三个 mode 共享
 4      s̄_d ← Σ_q π_dq · s_dq                                 # §2.7 混合
 5      Ψ_d ← Φ_d ⊙ √s̄_d          (广播到 [n_d, K_d])         # §2.4 Mercer 特征
 6      for each observed entry n ∈ minibatch:
 7          f_r^(d) ← Σ_k Ψ_d[i_d(n), k] · ã^(d)[r, k]         ã = a + σ·ε  重参数化
 8          û(n)   ← Σ_{r₁…r_D} G[r₁,…,r_D] · Π_d f_{r_d}^(d)      _contract
 9      L ← −Σ_n log N(y_n ; û(n), σ_noise²)  +  KL(q‖N(0,I))  # §2.8 ELBO
10      反向传播，梯度裁剪到 10，Adam(η) 更新 a, log σ, G, logits, log σ_noise
11  return 参数
```

**几个必须知道的实现细节**

| 行 | 细节 | 为什么 / 写错会怎样 |
|---|---|---|
| 3 | `routing="global"`：三个 mode 共享一组 $\pi$ | `per_mode_rank` 在 1% 观测下过拟合，对通用字典的伤害（$0.042$）大于对算子 bank（$0.023$），**在这个设置下比较会虚高约 1.6 倍** |
| 5 | $\sqrt{\cdot}$ 前把谱截断到 $10^{-12}$，**只在构造特征时** | $\sqrt{\cdot}$ 在 0 处导数无穷。只在特征处截断保住了"严格零支撑"控制实验的语义，同时避免 NaN 梯度 |
| 5 | 特征 $\Psi_d = \Phi_d \odot \sqrt{\bar{s}_d}$ 使 $k_d = \Psi_d\Psi_d^\top$ **自动半正定** | 这是"非负分离"必须非负的原因：谱取负就不是核 |
| 7 | 重参数化采样，3 个样本 | 样本太少梯度噪声大；实测 3 个够用 |
| 8 | 只在**观测到的** entry 上求值 | $64^3$ 的张量在训练中**从不物化**；换成先重建整场会 OOM |
| 9 | 先验是标准正态 | 尺度已经吸进 $\sqrt{s_d}$，所以 KL 有闭式，不需要采样 |

**代价**：每步 $O\bigl(\lvert\Omega\rvert \sum_d R_d K_d\bigr)$，主线约 $2621 \times 216 \approx 5.7\times10^5$，
1000 步在 A100 上约 2 秒、CPU 上约 25 秒。**与网格分辨率无关**——这是低秩函数式表示相对
全 GP（$O(n^3)$）的结构性优势。

---

### 算法 3：三档评估协议（主表怎么来的）

> **输入** 场生成器、布局、观测比例、seed 列表、长度尺度网格 $\mathcal{E}$
> **输出** 每个臂的 held-out NRMSE，以及**调参代价**

```
 1  for seed in seeds:
 2      X      ← Solve(seed)                                   独立求解器，非模型先验采样
 3      Ω, Ω̄  ← SensorMask(layout, budget, seed)               同 seed 内所有臂共用
 4      y      ← X[Ω] + noise                                  同 seed 内所有臂共用
 5
 6      # 臂一：我们，零调参数据
 7      e_ours ← Fit(Ω, y, bank=算法1(θ_nominal)) 在 Ω̄ 上评估
 8
 9      # 臂二：可部署 Matérn —— 实践者唯一能做的
10      Ω_tr, Ω_val ← 把 Ω 随机切成 75% / 25%
11      ℓ_dep ← argmin_{ℓ∈ℰ}  Fit(Ω_tr, y_tr, Matérn(ℓ)) 在 Ω_val 上的误差
12      e_dep ← Fit(Ω, y, Matérn(ℓ_dep)) 在 Ω̄ 上评估
13
14      # 臂三：oracle Matérn —— 上界，标 ★，没人跑得了
15      ℓ_orc ← argmin_{ℓ∈ℰ}  Fit(Ω, y, Matérn(ℓ)) 在 Ω̄ 上的误差
16      e_orc ← 该最小值
17
18  调参代价 ← mean(e_dep) − mean(e_orc)          ← 本文的核心量
19  报告 ℓ_dep 与 ℓ_orc 的实际取值           ← 机制证据，比差值更难反驳
```

**三档的严格定义**（这是主表的全部内容，值得逐字读）：

| 档 | $\ell$ 由哪份数据选出 | 现实中能做吗 | 在表里怎么标 |
|---|---|---|---|
| ours | **不选**。谱由算子给出，系数取错 1.5 倍的名义值 | ✅ | 无标记 |
| Matérn 可部署 | 传感器读数的 25% 留出来做验证 | ✅ 这是实践者唯一能做的 | 无标记 |
| Matérn oracle | **真实留出区域**（$\bar{\Omega}$，即测试集） | ❌ | $\star$ |

**为什么要把一个作弊的档放进主表。** 它是**上界**，回答"假如你事先就知道最佳超参，
通用核能做到多好"。三个数字要一起读：

- **ours vs oracle** → 我们的先验**准不准**（单面墙 0.5387 vs 0.5438，追平）
- **ours vs 可部署** → 相对**真实对手**有没有用（0.5387 vs 0.8130，$+26\%$）
- **可部署 $-$ oracle** → **调参代价**，即这个 setting 本身有多惩罚调参
  （$+0.27$）。**这一列与我们的方法好坏完全无关**，是最难被推翻的数字

> **第 15 行是作弊的，这正是重点。** oracle 用真实留出区域选超参，
> 现实中拿不到。它存在是为了回答"假如你事先知道最佳超参，通用核能做到多好"。
> **第 18 行的差值是 setting 的性质，与我们的方法好坏无关**——
> 这是本文最难被推翻的那个数字。
>
> **第 11 行和第 15 行的网格 $\mathcal{E}$ 必须夹住最优。**
> 检查方法：如果 $\ell_{\mathrm{orc}}$ 落在 $\mathcal{E}$ 的端点，网格不够宽，**结果不可用**。
> 这个项目唯一一次撤回就是因为没检查。

---

## 3. 相关工作（以及本文**不**主张什么）

| 工作 | 它做什么 | 关键差别 |
|---|---|---|
| [FunBaT (2023)](https://arxiv.org/abs/2311.04829) | 连续坐标张量补全，GP latent functions + Tucker，message passing 推断 | 用**通用**逐 mode 核；不从算子谱构造 atom。我们把宿主、秩、预算全部锁死，只换谱的来源 |
| [RR-FBTC (2025)](https://arxiv.org/abs/2512.21486) | functional Bayesian tensor completion + 自动 rank 学习 | 贡献在逼近能力与 rank 学习；本文**固定 rank** |
| [EPGP (ICML 2023)](https://proceedings.mlr.press/v202/harkonen23a.html) | Ehrenpreis–Palamodov 构造 GP，样本**严格满足**常系数线性 PDE | 需要**完整已知**的方程（含系数）；本文只近似解的二阶谱，换来对 K1/K0 的支持 |
| [Tensor GPs (2025)](https://arxiv.org/abs/2510.13772) | 一维 GP factors + 张量分解**求解**非线性 PDE | collocation 求解，需要完整方程；本文做**有噪极稀疏补全** |
| [AutoIP (ICML 2022)](https://proceedings.mlr.press/v162/long22a.html) | 把微分方程作为 collocation 点上的**虚拟观测**并入 GP | **最近的亲戚**：物理同样进先验而非损失。差别在承载者——它保留全 GP，要分解 $(n+m)\times(n+m)$ 稠密矩阵；我们放进低秩张量的逐维谱，从不形成该矩阵。**且它必须承诺单个 $\theta$** |
| [Spectral Mixture (ICML 2013)](https://proceedings.mlr.press/v28/wilson13.html) | 参数化谱密度，在平稳核类中**稠密** | 因为稠密，任何固定构造都不可能在**表达力**上胜出，只可能在**样本效率**上。所以我们把它作为 baseline，而不是声称谱表示是新的 |
| PINN 类 | 把 PDE 残差加进损失 | 见 §5：同一份物理，当先验和当惩罚差一个数量级 |

**本文不主张**"GP + tensor"或"PDE-informed GP"是新的。可守的最窄定位是：

> 把算子族**不可分**的联合谱做**非负、保 PSD** 的低秩投影，得到逐维 GP kernel，
> 使得在"只知方程形式、不知系数"时仍可用；并给出何时该用、何时不该用的判据。

---

## 4. Baseline：每一个是什么、给了什么待遇

**通用原则：所有 baseline 的待遇都比我们好。** 我们这一臂永远是：
一组配置、系数错 50%、零调参数据。

### 4.1 核类（跑在**我们的**宿主里，因此是消融不是 baseline）

这一点很重要，我一度弄混过：这些臂用的是我们的余弦本征基、Tucker 宿主、
collapsed 参数化、逐维特征预算——**本文论证的每一项设计都无偿给了它们**。
所以它们隔离的是"谱从哪来"这一项，是**消融**。

| 臂 | $\ell$ 由什么数据选 | 可部署？ |
|---|---|---|
| Matérn 可部署 | 传感器读数的 1/4 留出验证 | ✅ |
| Matérn oracle$^\star$ | **真实留出区域**（测试集） | ❌ 上界 |
| Matérn 固定 | 不选，固定 $\ell = 1.6$ | ✅ |
| 逐维常数 oracle$^\star$ | 27 种组合在测试集上选 | ❌ 上界 |

谱密度 $s(k) = (1 + (\ell k)^2)^{-2}$，即一维 Matérn-$3/2$。

> **网格必须夹住 baseline 自己的最优**。原网格 `(0.12, 0.32, 0.8)` 没夹住
> （最优在 2.4），直接导致一个被撤回的头条。现网格
> `(0.12, 0.32, 0.8, 1.6, 2.4, 3.5)`，最优 2.4 是内点。
> **任何新网格都要检查最优是否落在端点。**

### 4.2 神经类（真正的对手）

| 臂 | 结构 | 给的待遇 |
|---|---|---|
| **CoSTCo** (KDD'19) | 每 index 嵌入 → 跨 rank 轴卷积 → 跨 mode 卷积 → MLP。**刻意非多线性** | 3 架构 × 2 学习率，**测试集选**；6 倍预算 |
| **Fourier-MLP** | 位置编码 + MLP，**无低秩假设** | 同上 |
| LRTFR/SIREN | sine 因子 + Tucker 核（Continuous-Tensor-Toolbox） | 6 倍预算 |

> 实现：`neural_baselines.py`、`neural_functional_tucker.py`、`run_leak_neural.py`。
> 旧版只有 LRTFR，它在受限布局上都在 0.9 附近，**根本没在竞争**——
> 留它当唯一神经对照会让论文看起来"赢了容量"，其实只赢了一个小模型。

### 4.3 物理类（最该比的）

| 臂 | 怎么用方程 | 给的待遇 |
|---|---|---|
| **PINN** | 损失加 $\lVert \mathcal{L}_\theta u \rVert^2$ 于 collocation 点，autograd 求导 | 5 权重 × 2 学习率**测试集选**，4 倍预算，**权重 0 在候选里**（可以拒绝物理） |
| **AutoIP 式** | $\mathcal{L}u = 0$ 作为虚拟观测并入 GP | 3 组长度尺度 × 4 residual noise **测试集选**，含"等于关闭"的档 |

> 实现：`pinn_baseline.py`、`physics_informed_gp.py`。
> AutoIP 需要核的**四阶混合导数**。解析式**必须对拍自动微分**
> （`verify_against_autograd`，现吻合到 $2\times10^{-16}$），
> 而且**光验证零件不够**——我的分块转置写错过，
> 所以另加了"在自己的无噪观测上条件化必须复现观测"的端到端检验。

---

## 5. 主实验与完整 setting

### 5.1 共享 setting（所有表）

```python
FIELD = dict(grid=(64, 64), diffusivity=(0.02, 0.006), reaction=0.04,
             sources=((0.30, 0.65, 0.09, 15.0), (0.70, 0.35, 0.07, 25.0),
                      (0.55, 0.75, 0.06, 40.0)),
             dt=0.6, burn_in=200, record_steps=64, background_noise=0.02)
NOMINAL = dict(diffusivity=(0.03, 0.012), reaction=0.06)  # 故意错 1.5 倍
LENGTH_SCALES = (0.12, 0.32, 0.8, 1.6, 2.4, 3.5)
BINS   = (12, 12, 12)     # 每 mode 频率数
RANKS  = (8, 5, 5)        # Tucker 逐 mode 秩
```

- 张量 $[t, x, y] = 64 \times 64 \times 64$，去均值并除以标准差；
- 观测比例 $1\%$（2621 个点），观测噪声 std $0.05$；
- 优化：Adam，lr $0.02$，1000 步，ELBO 3 样本，梯度裁剪 10；
- **同一 seed 内所有臂共享场、掩码、噪声**，所以每个比较都是配对的；
- 指标：held-out NRMSE，按留出集标准差归一化，**所以 $1.0$ 恰好是预测均值的分数**。

### 5.2 布局定义（`sensor_mask`）

| 名称 | 区域 | 可达 |
|---|---|---|
| `random` | 全域 | 100% |
| `wall_ring` | $\mathrm{depth} < 2$ | 12% |
| `near_wall` | $3 \le \mathrm{depth} < 8$ | 26% |
| `one_wall_strip` | $x < 5$ | 8% |
| `one_wall_strip_y` | $y < 5$ | 8% |
| `two_walls_lr` / `two_walls_tb` | $x < 5$ 或 $x \ge n_x - 5$ / 同理 $y$ | 16% |
| `two_walls_adjacent` | $x < 5$ 或 $y < 5$ | 15% |
| `corner_block` | $x < 20$ 且 $y < 20$ | 10% |
| `four_corners` | 四角各 $10 \times 10$ | 10% |

**所有布局的观测预算相同**，只有排布不同。

### 5.3 可信的主实验清单

| 实验 | 脚本 | 结果文件 | seeds |
|---|---|---|---|
| **主表**：5 布局 × 3 档 | `run_leak_sensors.py` | `leak_main3tier_summary.json` | 5 |
| **算子网格**：3 族 × 5 布局 | `run_leak_operators.py --tag operator_grid` | `operator_grid_summary.json` | 3 |
| **3D 房间** $32^4$ | `run_leak_3d.py` | `leak3d_fixed_summary.json` | 2 |
| **知识档位** K2/K1/K0/K$-$1 | `run_leak_knowledge_ladder.py` | `knowledge_ladder_leak_summary.json` | 3 |
| **强神经 baseline** | `run_leak_neural.py` | `neural_strong_summary.json` | 3 |
| **PINN 对比** | `run_leak_physics_baselines.py` | `physics_baselines_summary.json` | 3 |
| **固定核对照** | `run_leak_fixed_kernel.py` | `fixed_kernel_summary.json` | 3 |
| **配对几何** | `run_leak_sensors.py --tag leak_geometry` | `leak_geometry_summary.json` | 3 |
| **稳健性**（样本量/噪声/系数） | `run_leak_robustness.py` | `robustness_summary.json` | 3 |
| **剖面机制** | `run_leak_profile_mechanism.py` | `profile_mechanism_summary.json` | 3 |

表与图**全部由脚本从 JSON 生成**，不手抄：
`make_paper_tables.py`、`make_leak_figures.py`、`make_headline_figure.py`、
`make_reconstruction_figure.py`、`make_appendix_tables.py`。
**这张主表被撤回过一次，手抄正是撤回过的数字混进修订版的典型途径。**

### 5.4 三个最重要的数字

**（一）调参代价随受限程度上升，与算子族无关**

| 布局 | 反应-扩散 | 扩散主导 | 平流-扩散 |
|---|---|---|---|
| 随机 | $+0.0000$ | $+0.0008$ | $+0.0031$ |
| 单面墙 | $+0.2281$ | $+0.3411$ | $+0.1996$ |

**（二）同一份物理，先验 vs 残差**

| 布局 | ours | PINN$^\star$ | 同一网络关掉物理 | 残差买到 |
|---|---|---|---|---|
| 随机 | 0.0547 | **0.0522** | 0.1174 | $+0.0652$ |
| 单面墙 | **0.5387** | 0.8008 | 0.8229 | $+0.0220$ |

单面墙上三个 seed 有两个**自己选了残差权重 0**。

**（三）知识档位（单面墙）**

| 档 | NRMSE |
|---|---|
| K2 真系数 | 0.5450 |
| K1 bank $\times[1/3,3]$ | **0.5468** |
| K0 bank $\times[1/10,10]$ | 0.5783 |
| K$-$1 generic（同 16 atom） | 0.7408 |

### 5.5 完整主表（5 seeds，1% 观测）

| 布局 | 可达 | ours（不调参） | Matérn 可部署 | Matérn oracle$^\star$ | mixture | neural CP | 调参代价 |
|---|---|---|---|---|---|---|---|
| 房间内任意 | 100% | 0.0553 | 0.0539 | 0.0536 | **0.0537** | 0.0770 | +0.0003 |
| 四面墙 | 12% | **0.4767** | 0.5012 | 0.4468 | 0.5282 | 0.5151 | +0.0544 |
| 贴墙带 | 26% | **0.3142** | 0.3861 | 0.3069 | 0.3342 | 0.3361 | +0.0791 |
| **单面墙** | 8% | **0.5396** | 0.8130 | 0.5433 | 0.7334 | 0.8935 | **+0.2697** |
| 单个角块 | 10% | 1.0212 | 0.9568 | 0.9023 | **0.9236** | 0.9465 | +0.0544 |

粗体标的是**实践者能实际部署的最好那一臂**。oracle 列打星因为没人跑得了。

### 5.6 固定核对照：一个必须知道的削弱

一个既不调参、也不用方程、直接固定一个光滑先验的实践者，表现如何：

| 布局 | ours | $\ell$=0.12 | 0.32 | 0.8 | **1.6** | 2.4 | 3.5 |
|---|---|---|---|---|---|---|---|
| 随机 | 0.0534 | 0.0539 | 0.0530 | **0.0513** | 0.0581 | 0.1705 | 0.3038 |
| 四面墙 | 0.4791 | 0.5946 | 0.5402 | 0.4813 | 0.4776 | **0.4498** | 0.4976 |
| 贴墙带 | 0.3086 | 0.4321 | 0.4100 | **0.3019** | 0.3121 | 0.3487 | 0.4028 |
| 单面墙 | 0.5448 | 0.8899 | 0.8179 | 0.6573 | 0.5514 | **0.5438** | 0.5592 |
| 角块 | 1.0368 | 0.9340 | **0.8982** | 0.9233 | 1.0503 | 1.1221 | 1.1315 |

事先承诺一个值（不知道会遇到哪个布局）时相对我们的最差表现：

| $\ell$ | 0.12 | 0.32 | 0.8 | **1.6** | 2.4 | 3.5 |
|---|---|---|---|---|---|---|
| 最差差距 | +0.3452 | +0.2731 | +0.1125 | **+0.0135** | +0.1172 | +0.2505 |

**结论对我们不利，照实写**：$\ell = 1.6$ 相对我们最差只差 0.0135。
**2D 上物理相对"随手固定一个中庸常数"几乎没买到东西。**

但同一张表里活下来一个更反直觉的结论：单面墙上固定 $\ell = 1.6$ 得 0.5514，
而**在传感器数据上调参**得 0.8130——**调参比不调参差 0.26**。
受限布局下，留出验证集选超参这个标准做法是**主动有害**的。

### 5.7 一个被自己的对照推翻的机制解释

**预先登记的预测**：场是各向异性的（$D_x = 0.02$、$D_y = 0.006$），
若增益来自"知道各轴衰减律"，则沿粗糙轴（$y$）外推时增益应更大。

**实测（5 seeds）**：

| 场 | 传感器 | ours | Matérn$^\star$ | margin |
|---|---|---|---|---|
| 各向异性 | $x$ 墙 | 0.5428 | 0.5433 | +0.1% |
| 各向异性 | $y$ 墙 | **1.2652** | 0.8445 | **−49.8%** |
| 各向同性对照 | $x$ 墙 | 0.7350 | 0.7374 | +0.3% |
| 各向同性对照 | $y$ 墙 | **1.2240** | 0.8944 | **−36.8%** |

$y$ 墙不是增益更大而是**惨败**，且**各向同性对照没有消除这个差距**——
所以"逐轴衰减律"这个解释**被撤回**。

后续诊断给出了部分答案：条带贴 $x$ 墙时**张满整个 $y$ 轴**，
于是"只随 $y$ 变化"的分量被完全观测、无需外推——而它占 64% 的方差；
贴 $y$ 墙时只能确定 $x$-剖面，那只占 4.4%。
把扩散系数交换后主导剖面翻到 $x$，差距从 +0.598 塌到 +0.091（**解释了约 85%**），
但**没有反转**，所以严格说这个后续解释也未完全成立。

**重要区分**：oracle 也表现出同样的不对称（$x$ 墙 0.5438 vs $y$ 墙 0.8534），
所以 $y$ 墙对**所有方法**都更难——那是**场的性质**；
我们额外退化得更厉害，才是**我们的问题**（见 §5.8）。

### 5.8 必须一并报告的负面结果

- **角块布局**：反应-扩散场上我们 1.0346，**比预测均值还差**，输给所有 baseline。
  但在扩散主导场上我们 0.7037 **赢 oracle 0.095**——所以这是**场的性质**，不是方法的普遍缺陷。
- **失败不体面**：全部实验里唯一超过 NRMSE 1.0 的臂始终是我们，baseline 最差只到 0.95。
  长尺度 Matérn 打不过时会收缩到均值并停下；我们的尺度被方程钉死，会继续外推。
  **部署前需要一个"检测正在超出观测约束范围并转为收缩"的回退机制，目前没有。**
- **2D 上固定常数几乎追平我们**（$\ell = 1.6$，最差差 0.0135）。
- **AutoIP 在 2D 尺度上比我们快**（1.7–8.0 秒 vs 20–25 秒）。
  "标准 GP 不 scale"在这个规模上**不成立**，不要这样写。

---

## 6. 数据从哪来

### 6.1 主线数据：自己解，不需要下载

**当前论文的全部主实验都不依赖任何外部数据集。** 场由本仓库的求解器生成：

```bash
# 2D 主线场（每个 seed 约 1 秒）
python -c "
import sys; sys.path[:0]=['experiments']
from forced_pde_solver import solve_multi_leak
f = solve_multi_leak(seed=0, grid=(64,64), diffusivity=(0.02,0.006), reaction=0.04,
                     sources=((0.30,0.65,0.09,15.0),(0.70,0.35,0.07,25.0),
                              (0.55,0.75,0.06,40.0)),
                     dt=0.6, burn_in=200, record_steps=64, background_noise=0.02).field
print(f.shape)"

# 3D 房间（约 1.4 秒，32^4 ≈ 105 万格点）
python -c "
import sys; sys.path[:0]=['experiments']
from forced_pde_solver import solve_multi_leak_3d
print(solve_multi_leak_3d(seed=0).field.shape)"
```

求解器：`experiments/forced_pde_solver.py`
- DCT 谱方法 + 指数积分器做扩散-反应（无流边界，余弦基对角化）；
- 平流用**算子分裂**：谱步之后在实空间做一阶迎风，**CFL 条件运行时检查**；
- `_smooth_noise` 用高斯滤波白噪。
  > **踩过的坑**：早期用 `np.kron` 复制块来造强迫，其谱是 sinc、**带硬零点**，
  > 导致先验与数据在"强迫"上不一致——而我一度以为是算子错了。

**为什么用自己的场是合理的**：真值来自**独立的**有限差分/谱积分，
**不是**从模型自己的先验里采样；而且给模型的名义系数是**故意错 1.5 倍**的。

### 6.2 若要换真实数据集

本项目评估过以下外部来源。**每一个都必须先过 §6.3 的筛查**。

| 数据集 | 链接 | 适合什么 | 已知风险 |
|---|---|---|---|
| **PDEBench** | https://github.com/pdebench/PDEBench | diffusion / reaction-diffusion / advection，官方生成器给出确切 PDE 参数 | 2D diffusion-reaction 已实测**不通过筛查**：各空间轴需 63% 的模态才够 95% 能量，全观测秩-40 CP 仍留 0.298 相对误差 |
| **Kolmogorov flow (MNO)** | https://github.com/neuraloperator/markov_neural_operator | 周期域算子谱清楚，可测 Reynolds shift | $\mathrm{Re}=40$ 已实测**不通过**：同样 63%，broad-spectrum turbulence |
| **OpenFWI** | https://openfwi-lanl.github.io/ | wave / 高频压力测试 | full signed phase 与非平稳介质可能超出实值可分核；允许得到明确负结论 |
| **The Well** | https://github.com/PolymathicAI/the_well | 统一接口，acoustic / active matter | 先用固定小子集，**manifest 要先冻结**，不按结果选 trajectory |
| **CFDBench** | https://github.com/luo-yining/CFDBench | 通用 CFD 与边界条件 shift | 精确算子未知，**不能**描述成"已知 operator prior" |
| **RealPDEBench** | https://github.com/AI4Science-WestlakeU/RealPDEBench | 最终 sim-to-real | 不应在核选择阶段消费真实 test |

处理脚本入口：`experiments/pdebench_data.py`（PDEBench 读取与筛查）、
`src/geoaware/the_well_pilot.py`（The Well 小子集）、
`src/geoaware/tensor_data.py`（通用张量化与掩码）。

### 6.3 上数据集之前必须跑的三个筛查（预先声明，先跑再比方法）

1. **低秩可行性**：按拟采用的秩做**全观测** Tucker 拟合，相对误差须**明显低于 1**。
   否则任何核都无关紧要，所有臂都会落在平凡基线附近。
2. **谱集中度**：各轴携带 95% 能量所需的模态比例。实测 $\ge 63\%$ 的都不通过。
3. **边界可观测性**（受限布局特有）：

$$
\rho \;=\; \frac{E_{\partial} / E_{\text{total}}}{N_{\partial} / N_{\text{total}}} \;\ge\; 0.5 ,
$$

   其中 $E_{\partial}$ 是边界区域的能量、$N_{\partial}$ 是其格点数。

   > **阈值必须用被拒设计来标定**。原来写的是 $\rho \ge 1$，
   > 而主线场自己是 $0.986$——那个阈值会把它一并判掉，**没有区分力**。
   > 用真正被拒的设计（单个窄羽流，$\rho = 0.160$，墙面 SNR 7.6）标定后，
   > 在用的场是 $0.986$ / $1.035$（SNR 19），阈值取在空档里才有意义。

---

## 7. 代码地图

```
src/geoaware/operator_spectral_funbat.py   核心：谱构造、非负CP、Tucker宿主（998 行）
  ├─ operator_joint_spectrum        算子符号 → 联合谱
  ├─ nonnegative_cp_spectrum        非负CP分离（任意阶，支持掩码）
  ├─ real_cosine_basis              无流边界本征基
  ├─ normalize_spectrum_cosine      余弦基下的归一化
  └─ ModeAdaptiveVariationalTucker  宿主（任意阶，global/per_mode/per_mode_rank routing）

experiments/
  forced_pde_solver.py              场生成（2D/3D，可选平流）
  run_leak_sensors.py               主表 + 布局定义 + fit_gp（其他脚本都 import 它）
  run_leak_operators.py             算子族 × 布局网格
  run_leak_3d.py                    3D 房间
  run_leak_knowledge_ladder.py      K2/K1/K0/K−1
  run_leak_neural.py                CoSTCo / Fourier-MLP / LRTFR
  run_leak_physics_baselines.py     PINN
  run_leak_autoip.py                AutoIP 式物理信息 GP（精度 + 代价）
  make_*.py                         全部表与图的生成器
tools/check_github_math.py          公式渲染检查（改文档后必跑）
```

### 7.1 配置系统（换实验不要改 Python）

所有会变的设置都在 `configs/*.yaml`，变体用 `inherits` 只写差异：

```yaml
# configs/my_study.yaml
inherits: base
field:
  reaction: 0.004
nominal:
  reaction: 0.006
```

```bash
python experiments/run_leak_sensors.py --config my_study --tag my_study
python experiments/run_leak_sensors.py --config base --set evaluation.ratio=0.02
```

三条刻意的设计：

- **未知键报错**，不静默忽略——超参名打错和"没效果"在结果上无法区分。
  这个检查上线时立刻抓到 `base.yaml` 从未声明 `drift`，导致平流变体根本覆盖不了它。
- **覆盖可追溯**：`config_name`、`overrides` 和展开后的完整配置都写进结果 JSON。
- **不原地修改**：`operator_spectra` 和 `fit_gp` 把系数与预算作为参数传入。
  重构前有三个脚本靠**改写另一个模块的全局变量**来切换算子族再改回去——
  两个研究共用一个进程时就会出错，而且看不出某次运行到底用了什么。

现成变体：`base`、`diffusion_dominated`、`advection_diffusion`、`room3d`、`isotropic_control`。

### 7.2 复现主表

```bash
python experiments/run_leak_sensors.py --config base --tag leak_main3tier
python experiments/make_paper_tables.py
```

> 重构后与已落盘结果**逐位一致**（单面墙 seed 0：0.523953 vs 0.523953，差 0）。
> 任何进一步的重构都应该这样验证，而不是假设。

---

## 8. 给接手者的忠告

1. **任何超参网格都要检查最优是否在端点。** 这个项目最大的一次撤回就是因为没检查。
   反过来的错误（把 baseline 调得过强）大家都会防；
   **调得不够**产生的数字**看起来像测量结果**，没有任何东西会提示你。
2. **机制解释先写进脚本，再跑。** 本项目至今有 4 次机制预测被自己的对照推翻，
   每一次事后都很容易重新叙述。判定条件要写成代码里的布尔值。
3. **报告负面结果。** 角块超过 1.0、$y$ 墙崩溃、2D 上固定常数追平——
   这些都在文档和论文里。掩盖它们会在审稿时以更大的代价回来。
4. **改完文档跑 `tools/check_github_math.py`。**
5. **未来方向见 `docs/FUTURE_BRANCHES_ZH.md`**（泄漏源定位、归纳式表征、
   以及两个尚未解释的现象）。
