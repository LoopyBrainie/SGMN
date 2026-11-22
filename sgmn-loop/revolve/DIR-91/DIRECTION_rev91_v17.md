# DIRECTION · DIR-91 · rev 1.7 — 闭式 $W^\star$ 端到端 4 选 1

**本卡实现 task.md 的 `DIR-91-rev2.3`（机制文档 `PROBE91-MECHANISM.md` rev2.3 计划的下一步）。**
**只新增一节判据（§1′–§4′）与五条守卫；DIR-91 的既有主判据 $\Delta_{\rm content}$、`A91-11b`
（$0.75$）、`A91-15`、$0.40$ 门、S91 门**一字未改**，本卡**不产出**也不替代它们。**

---

## §0 本卡要回答的唯一问题

`DEF-91-23` 已定案：装置终态存储 $\mathrm{rel}=0.16\sim0.27$（相对 $W^0$ 基线 $2.28\sim2.54$
改善 $8.59\sim14.03\times$，逐层口径），但距**同目标闭式地板** $8.03\times10^{-4}$ 仍差
$\sim10^2\sim10^3$。

**问题**：这个存储差距**是否约束端到端 4 选 1 准确率**？
- 若是 ⇒ 只是**优化预算**问题，$T$（写步数）扫描**有意义**；
- 若否 ⇒ **表征受限**，任何 $T$ 都救不回来，$T$ 扫描是浪费算力，应改攻**键表征**。

**为什么必须现在问**：它**分叉全部分支**（task.md 原文）。不先跑它，$T$ 扫描的方向都无法确定。

---

## §1′ 主假说与证伪

**主假说 `H91-2.3`**：在该装置、该 $\mathrm{MOUNT}=[8,12,16]$、该基座、**同一批 $K,V$**、
同 $(\alpha,\lambda)=(1.0,0.05)$ 下，存储差距是**优化预算受限**的 ⇒ 把可塑状态直接置为
**装置自身目标的精确不动点 $W^\star$**（而非 599 步有限 $\eta$ 的部分收敛态）会**实质性提高**
4 选 1 准确率。

**证伪条款**：若 $W^\star$ 臂的按题准确率**未能实质高于**「不写入」$W^0$ 臂（见 §2′ 数值门），
则**存储差距不是约束**⇒ 判 `FALSIFIED`（表征受限），并**据此否决** $T$ 扫描。

**第三态**：任一守卫 `A91-22/23/25/26` FAIL ⇒ 记 `INCONCLUSIVE-<守卫名>`，
**不得**记作 `FALSIFIED`；`A91-24` 只报告不否决（见 §4′）。

---

## §2′ $W^\star$ 的推导（**它是装置自身目标的不动点，不是我另立的目标**）

装置写规则（`sgmn_module.py:86`）：

$$G=\alpha\,\mathrm{outer}(Wk-v,\,k)+\lambda\,(W-W^0),\qquad W\leftarrow W-\eta G$$

$G$ 恰是下式的梯度：

$$L(W)=\tfrac{\alpha}{2}\lVert Wk-v\rVert^2+\tfrac{\lambda}{2}\lVert W-W^0\rVert^2$$

对**冻结**的 $n$ 对 $\{(k_t,v_t)\}$ 求和（$\gamma{=}0$ 下 $k,v$ **不随 $W$ 漂移**，
`DEF-91-20`(b) 已实测 $\max|\Delta|=0$），令 $\partial L/\partial W=0$：

$$\alpha W K^{\top}K+\lambda W=\alpha V^{\top}K+\lambda W^0
\;\Longrightarrow\;
\boxed{W^\star=(\alpha V^{\top}K+\lambda W^0)(\alpha K^{\top}K+\lambda I)^{-1}}$$

其中 $K\in\mathbb{R}^{n\times d_k}$、$V\in\mathbb{R}^{n\times d_v}$ 行分别为 $k_t,v_t$。

**⇒ $W^\star$ 与装置下降的是同一个目标、同一批数据、同一 $\alpha,\lambda$**；
装置 599 步 @ $\eta=3\times10^{-5}$ 只是**朝它走了一部分**。故本卡测的是
「若预算无限（精确解）能否救回准确率」，**不是**换目标。

**$K,V$ 的来源（必须与存储指标同源）**：`key_matrices()` 每事实一条
（第 `owner[i]+1` 块、`pk_scale` 已乘），$n=32$。存储指标 $\mathrm{rel}$ 就用这 32 对定义
⇒ $W^\star$ 与之**同源可比**。**不**用 599 个块级对（那会换成另一个目标，不可比）。

---

## §3′ 判据与数值门（**开跑前冻结**）

**裁决统计量**：`acc_q` = **按题计**（同题 `N_DRAWS=3` 次**全对**才命中），$n_q=16$
（32 事实 − 16 训练键 = 16 评测键），沿用 `acc_by_question` 的既有裁决口径。
**辅助**：`acc_item`（$n=48$）。**只在 `eval_keys`（留出键）上裁决。**

**零假设基准**：4 选 1 ⇒ 精确 $p_0=0.25$。$X\sim\mathrm{Bin}(16,0.25)$：
$P(X\ge8)=0.0271$、$P(X\ge7)=0.0796$。

| 判定 | 条件（**两 seed 须一致**） |
|---|---|
| **`SUPPORTED`**（预算受限，$T$ 扫描有意义） | $\mathrm{acc}_q(W^\star)\ge\mathbf{0.50}$ **且** $\mathrm{acc}_q(W^\star)-\mathrm{acc}_q(\text{real})\ge\mathbf{0.125}$ **且** $\mathrm{acc}_q(W^\star)-\mathrm{acc}_q(W^0)\ge\mathbf{0.125}$ |
| **`FALSIFIED`**（表征受限，否决 $T$ 扫描） | $\mathrm{acc}_q(W^\star)<\mathbf{0.40}$ **或** $\mathrm{acc}_q(W^\star)-\mathrm{acc}_q(W^0)<\mathbf{0.125}$ |
| **`INCONCLUSIVE`** | 其余（$0.40\le\mathrm{acc}_q(W^\star)<0.50$），或守卫 FAIL |

**门限的来源（透明声明）**：$0.50$ 与 $0.40$ 是 $n_q{=}16$、$p_0{=}0.25$ 下的**二项切点**
（单侧 $2.7\%$ 与 $8.0\%$），**先于任何读数写定**。$0.125=2/16$ 是「至少多答对 2 题」的
最小实质幅度。**透明声明**：`eval_keys` 上的准确率**从未被测过**（rev 1.6 §4.4：
$\Delta_{\rm content}$ 至今未计算）⇒ 这三个数**不可能**由结果反推。

**两 seed 处置**：`20261028`、`7777` 各自独立判；两 seed 同为 `SUPPORTED` 才记 `SUPPORTED`，
同为 `FALSIFIED` 才记 `FALSIFIED`，否则 `INCONCLUSIVE`。

---

## §4′ Stage A 守卫（**可失败**，任一 FAIL 阻断正式轮）

| 编号 | 断言 | 失败含义 |
|---|---|---|
| **`A91-22`** | 不动点残差 $\dfrac{\lVert\alpha(W^\star K^{\top}K-V^{\top}K)+\lambda(W^\star-W^0)\rVert_F}{\lVert\lambda W^0\rVert_F}\le10^{-10}$（**float64**，逐层） | 我推导/求解错了，读数无意义 |
| **`A91-23`** | 置入 `Wfull`（float64→**float32**）后，用 float32 重算 $\mathrm{rel}(W^\star)\le10^{-3}$（闭式地板 $8.03\times10^{-4}$） | 收益在 dtype 往返中丢失 |
| **`A91-24`** | 从 $W^0$ 出发**装置的一步**严格降低 $L(W)$（$\eta<2/L_{\rm Lip}$）；并**报告** `steps_to_cover` $=\lVert W^\star-W^0\rVert_F/(\eta\lVert G_1\rVert_F)$ | 目标方向与装置下降方向不一致（**只报告不否决**其步数） |
| **`A91-25`** | 三臂仅**初值 $W$** 不同：同 seed、同 `Random(1000+seed%100)` 读批、同 `READ_STEPS=600`、同 `Random(555)` 评测项、同 `n_item` | 无法归因（预算不齐） |
| **`A91-26`** | $\gamma{=}0$ 下「无写」与「写后」的 $K,V$ **逐位相同**（逐层 $\max\lvert\Delta\rvert=0$） | 前提（$k,v$ 不漂移）在本轮不成立，$W^\star$ 定义失效 |

**并`A91-19`/`A91-20`/`A91-21`**（`DEF-91-23`/`24`，上一轮已固化）一并先跑。

---

## §5′ 三臂（同读侧预算、$\gamma{=}0$）

| 臂 | $W$ 初值 | 写阶段 | 说明 |
|---|---|---|---|
| `W0` | $W^0$ | **不做**（`no_write=True`） | 下界；`Wfull` 初始化即 $W^0$（`sgmn_module.py:53`） |
| `real` | $W^0$ | `write_phase`（599 次，$\eta{=}3\times10^{-5}$） | 装置真实可达态 |
| `Wstar` | $W^\star$ | **不做**，直接写入 `Wfull` | 闭式不动点 |

三臂**同一** `train_read_task`（`READ_STEPS=600`、`READ_LR=10^{-3}$、`Adam`、`CLIP=1.0`）、
同一 `eval_keys`、同一评测项。写冻结（`no_write=True`）。

**$\gamma$ 的可训练性（须如实记录）**：`read_params` 含 `gamma` ⇒ 读侧训练会移动 $\gamma$
（初值 $0$）。三臂**同初值、同步数**，故**臂间可比**；但「漂移恒 0」只对**写阶段**成立，
**不外推**到读侧训练之后。

---

## §6′ 边界

1. 本卡**不产出** $\Delta_{\rm content}$、不改变 `A91-11b`/`A91-15`/S91 任何门限，**不替代** DIR-91 主判据。
2. 本卡准确率**不是**「架构上限」：读侧只训 600 步、`MOUNT` 只有三层、基座为 Qwen0.5B 级。
3. $W^\star$ 用 32 个**事实级**对求闭式解；**不得**外推为「599 块级对上的最优」。
4. $W^\star$ 的 float64 可解性**不等于**装置可承载性（装置步长 $\eta=3\times10^{-5}$ 能否到达，
   由 `A91-24` 的 `steps_to_cover` **报告**，不作否决）。
5. `SUPPORTED` 只支持「存储差距**是**约束之一」，**不排除**键表征同时受限；
   `FALSIFIED` 才是否决 $T$ 扫描的依据。
6. 须报 `dtype`；引用 $W^\star$ 的数字必须注明 float64 求解 + float32 落地。

## 修订史

- **rev 1.7**（Cycle 155）：本卡首版，专为 `DIR-91-rev2.3` 冻结判据与守卫；既有门限一字未改。
