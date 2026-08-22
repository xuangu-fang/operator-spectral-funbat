# 从零跑通

假设你刚拿到这个仓库，什么都没装。跟着做，大约 20 分钟能看到第一张主表。

**先读哪个**：本文只管"跑起来"。方法为什么长这样、公式怎么来的、
哪些结论已被推翻，见 [`HANDOVER_TECHNICAL_ZH.md`](HANDOVER_TECHNICAL_ZH.md)。

---

## 0. 你不需要下载任何数据集

主线的场由本仓库的求解器现算，2D 约 1 秒、3D 约 1.4 秒一个 seed。
只有当你想换到真实数据时才需要外部数据集（见交接文稿 §6.2）。

---

## 1. 环境

```bash
git clone <repo>
cd operator-spectral-funbat

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch numpy scipy matplotlib pyyaml pytest
```

**GPU 是可选的但强烈建议**：所有脚本默认 `--device cuda`（有卡时自动用），
在 A100 上比 CPU 快约一个数量级。没有卡就自动退回 CPU，结果相同、只是慢。

验证装好了：

```bash
.venv/bin/python -m pytest tests/ -q          # 应该 19 passed
```

---

## 2. 三分钟确认核心组件是活的

```bash
# 场生成器
.venv/bin/python -c "
import sys; sys.path[:0]=['experiments']
from forced_pde_solver import solve_multi_leak
f = solve_multi_leak(seed=0).field
print('field', tuple(f.shape), 'std %.3f' % float(f.std()))"
```

预期：`field (64, 64, 64) std 1.000`

```bash
# 配置系统
.venv/bin/python -c "
import sys; sys.path[:0]=['src']
from geoaware.config import load_config
c = load_config('base')
print('reaction', c.get('field.reaction'), '| ranks', c.get('model.ranks'))"
```

预期：`reaction 0.04 | ranks [8, 5, 5]`

```bash
# 文档公式检查（改文档后必跑）
.venv/bin/python tools/check_github_math.py
```

预期：`total: 0`

---

## 3. 跑一格，看数字对不对

先跑**一个 seed、一个布局**，确认你的环境复现得出已落盘的数字：

```bash
.venv/bin/python experiments/run_leak_sensors.py \
    --config base --seeds 0 --layouts one_wall_strip --tag mycheck
```

预期输出里 `ours` 应该是 **0.5240**。如果差得远，先别往下跑——
检查 torch 版本和是否用了 GPU（不同硬件在第 4 位小数上可能有差异，
但第 2 位应该一致）。

---

## 4. 跑完整主表

```bash
.venv/bin/python experiments/run_leak_sensors.py \
    --config base --tag leak_main3tier          # 5 seeds x 5 布局
.venv/bin/python experiments/make_paper_tables.py
.venv/bin/python experiments/make_leak_figures.py
```

产出：

| 文件 | 是什么 |
|---|---|
| `results/leak/leak_main3tier_summary.json` | 原始数值，含完整配置 |
| `paper/sections/table_layouts.tex` | 主表（LaTeX，自动生成） |
| `results/leak/figure_layouts.png` | 布局缩略图 + 柱状对比 |

**表和图一律由脚本从 JSON 生成，不要手抄。** 这张主表被撤回过一次，
手抄正是撤回过的数字混进修订版的典型途径。

---

## 5. 换一个场：写 YAML，不要改 Python

这是这次重构的重点。想跑一个新的算子族或新的房间：

```yaml
# configs/my_study.yaml
inherits: base
field:
  reaction: 0.004        # 只写和 base 不同的部分
nominal:
  reaction: 0.006
```

```bash
.venv/bin/python experiments/run_leak_sensors.py --config my_study --tag my_study
```

改单个值不必新建文件：

```bash
.venv/bin/python experiments/run_leak_sensors.py \
    --config base --set evaluation.ratio=0.02 --set optimisation.steps=2000 \
    --tag denser
```

`--set` 的内容会**写进结果 JSON**，所以任何数字都能追溯到产生它的确切配置。

**打错超参名会报错，不会静默忽略**：

```bash
.venv/bin/python experiments/run_leak_sensors.py --config base --set model.rank=8
# KeyError: override 'model.rank' names a key that does not exist;
#           known keys there: ['atoms','banks',...,'ranks',...]
```

现成的变体：

| 配置 | 是什么 |
|---|---|
| `base` | 各向异性反应-扩散，论文主线 |
| `diffusion_dominated` | 吸收项砍到 1/10 |
| `advection_diffusion` | 平流主导，Péclet ≈ 5 |
| `room3d` | 三维房间，$32^4$ |
| `isotropic_control` | 各向同性负对照 |

---

## 6. 跑其他主实验

| 想问什么 | 命令 |
|---|---|
| 换算子族还成立吗 | `run_leak_operators.py --tag operator_grid` |
| 三维呢 | `run_leak_3d.py` |
| 需要多少方程知识 | `run_leak_knowledge_ladder.py` |
| 打得过神经网络吗 | `run_leak_neural.py` |
| 物理当损失更好吗 | `run_leak_physics_baselines.py` |
| 和 AutoIP 比呢 | `run_leak_autoip.py` |
| 不调参的固定核呢 | `run_leak_fixed_kernel.py` |

每个都写一个 `results/leak/<tag>_summary.json`。

---

## 7. 加一个新实验时的检查清单

这几条都是这个项目**真实付出过代价**才学到的：

1. **超参网格必须夹住 baseline 自己的最优。**
   跑完检查选中值是否落在端点；落在端点就是网格太窄，**结果不可用**。
   这个项目唯一一次撤回就是因为没检查。
2. **机制预测先写进脚本再跑。** 判定条件写成代码里的布尔值并存进 JSON。
   本项目至今有 4 次机制预测被自己的对照推翻，事后都很容易重新叙述。
3. **上新数据集前先跑三个可行性筛查**（交接文稿 §6.3），跑完再比方法。
4. **同一 seed 内所有臂共享场、掩码、噪声**，否则比较不是配对的。
5. **负面结果照实写。** 角块超过 1.0、$y$ 墙崩溃、2D 上固定常数追平——
   这些都在文档和论文里。掩盖它们会在审稿时以更大的代价回来。
6. **改完文档跑 `tools/check_github_math.py`。**

---

## 8. 常见问题

**跑得很慢。** 确认在用 GPU：`nvidia-smi` 看得到进程；脚本会打印用的 device。
CPU 上一次拟合约 25 秒，A100 上约 2 秒。

**显存不够。** 减小 `evaluation.ratio` 或 `model.bins`；3D 场把
`--max-test` 调小（它只影响评估用的留出点数，不影响训练）。

**结果和文档里的对不上。** 先跑 §3 那一格。若第 2 位小数就不同，
检查是否改了 `configs/base.yaml`；若只是第 4 位不同，那是硬件差异，正常。

**想复现某个已有结果，但不知道它用了什么设置。**
打开对应的 `results/leak/*_summary.json`，`config` 字段里有完整的
`config_name`、`overrides` 和 `resolved`（展开后的全部配置）。
