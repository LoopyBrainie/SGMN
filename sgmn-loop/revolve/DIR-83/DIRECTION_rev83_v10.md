# DIRECTION · DIR-83 · rev 1.0（判据一次冻结）

**方向**：`target` 的缺口 $\rho_\lambda/\eta_{\rm th}$ 为何随 $\gamma$ 单调增长？——**机制裁决**
**出处**：`D-103` 派生①（= `D-102` 派生① 的后续；框架 §6.1「各层独立稳定 ⟹ 多层稳定？」的定量残余）。**非自造。**
**日期**：2026-11-18（Cycle 137） · **状态**：卡已冻结，Stage A 未跑

---

## §1 已冻结的事实（`D-79`/`D-80`/`D-99`/`D-103`，本卡不复测）

| $\gamma$ | $\eta_{\rm th}$ | $\rho_\lambda$（gfix 代表） | 缺口 $\rho/\eta_{\rm th}-1$ | $\lVert W^\ast\rVert_F$ |
|---|---|---|---|---|
| 3 | 0.064974679 | 0.068867243 | **5.99%** | 5.373535 |
| 10 | 0.0073034051 | 0.0091167253 | **24.83%** | 14.809432 |
| 20 | 0.0015740607 | 0.0025007558 | **58.87%** | 28.280904 |
| 30 | 0.00057804665 | 0.001147340655 | **98.49%** | 41.750654 |

$\lVert W^\ast\rVert_F = \sqrt2\,u_1 \propto \gamma$（实测 4 点，$u_1$ 与 `ref70.pred_target` 相对差 $4\times10^{-6}$）。
$\eta_{\rm th}$ 的定义（`dir79/formal79.bisect79`，**未改**）：对 `ref70.run_stack(n=2e5, init=0.1, seed=20261118)` 是否 `_div` 做 log 二分（`rel=0.005`）；`_div` = 任一层的 $\lVert W\rVert>10^3$ 或非有限。

## §2 开卡前的代数/数值预言（`DEF-64-1` 义务，已完成）

探针 `dir83/probe83_pre{,2,3}.py`（**只读**沿用 `ref70/step72/jac79/jac81/jac82`，不改装置）在冻结的不动点上（源规则 `warm78.warm_fp`，$\eta_w{=}0.1\eta_{\rm th}$，逐格 npz 缓存）测了四件事。**它们排除三条候选机制，并锁定第四条。**

### 2.1 排除机制①（非正规/瞬态放大）：$\Lambda$ 几乎恒等于 1.39

$\Lambda(\eta,T{=}2\times10^4)=\sup_{t\le T}\lVert J(\eta)^t\rVert_2$，在 **gfix 规范代表**上：

| $\gamma$ | $\Lambda(\eta_{\rm th})$ | $\Lambda(0.5\rho_\lambda)$ | 峰时 |
|---|---|---|---|
| 3 | **1.3925** | 1.234 | 1 |
| 10 | **1.349** | 1.348 | 4 |
| 20 | **1.380** | 1.380 | 8 |
| 30 | **1.391** | 1.391 | 13 |

极差 **3.2%**，且**与 $\eta$ 无关**（$\eta\le\rho_\lambda$ 时）。⇒ 瞬态放大上限 $\approx1.4$，而越阈需要 $10^3/\lVert W^\ast\rVert_F=24\text{–}186$ 倍。**差 2 个数量级**。机制①**由代数排除**。
> **顺带更正**：`D-101`/`D-79` 把 $\sup_t\lVert J^t\rVert$ 推荐为「更稳健的证书候选」。实测 $\Lambda\equiv1.39$、与 $\gamma$ 无关 ⇒ 它**几乎不携带信息**，不是有用的证书。

### 2.2 排除机制（谱提前穿越）：$\mu(\eta_{\rm th})<0$ 四格全成立

$\mu(\eta)=\ln\max_{\text{非中性}}|\lambda(J(\eta))|$：

| $\gamma$ | 3 | 10 | 20 | 30 |
|---|---|---|---|---|
| $\max_{\text{非中性}}\lvert\lambda\rvert$ at $\eta_{\rm th}$ | 0.99677365 | 0.99963516 | 0.99992132 | 0.99997110 |
| $\mu(\eta_{\rm th})$ | $-3.232\times10^{-3}$ | $-3.649\times10^{-4}$ | $-7.869\times10^{-5}$ | $-2.890\times10^{-5}$ |

**全部为负** ⇒ 在 $\eta=\eta_{\rm th}$（以及整个 $(\eta_{\rm th},\rho_\lambda)$，因为 $\mu$ 在 $\rho_\lambda$ 处才过零）不动点是**线性吸引**的。所以「缺口」**不是**谱提前穿越。
> 附带量：探测下限 $\mu_{\rm floor}=\ln(10^3/\lVert W^\ast\rVert_F)/N$ = 2.613e-5 / 2.106e-5 / 1.783e-5 / 1.588e-5。$\lvert\mu(\eta_{\rm th})\rvert/\mu_{\rm floor}$ = **123.7 / 17.3 / 4.41 / 1.82** ⇒ 在 $\gamma{=}3$ 衰减比探测下限快 124×，在 $\gamma{=}30$ 只快 1.8×。**该裕度的塌缩值得记，但 §2.4 显示它不是缺口主因。**

### 2.3 排除机制③（探测阈值伪像）：发散是**真发散**

$\eta>\eta_{\rm th}$ 的发散格上 $\lVert W\rVert$ 直接变**非有限**（`nan=True`），不是仅仅越过 $10^3$：

| $\gamma$ | $\eta/\eta_{\rm th}$ | $\max_t\lVert W_t\rVert/\lVert W^\ast\rVert$ | $t_{\rm div}$ | 非有限 |
|---|---|---|---|---|
| 30 | 1.00 | 1.18 | —（收敛，$\mu_{\rm tail}=-4.8\times10^{-9}$） | 否 |
| 30 | 1.25 | $\to\infty$ | 2322 | **是** |
| 20 | 1.50 | $\to\infty$ | 1125 | **是** |
| 10 | 1.10 | $3.8\times10^{110}$ | 592 | **是** |

⇒ 绝对阈值 $10^3$ 与 $\gamma$ 增长的 $\lVert W^\ast\rVert$ 之间的相对口径差（$\ln$ 意义下 1.75×）**不足以**造成缺口；**机制③排除**。

### 2.4 ★ 锁定第四条：**有限幅度（吸引域）逃逸**

在**同一个** $\eta=1.10\,\eta_{\rm th}$ 上，从 **`gfix_qr`($x^\ast$)** 出发加 $\varepsilon\hat\delta$（$\hat\delta$ 固定随机单位向量，seed 7777）：

| $\gamma$ | $\varepsilon{=}0$ | $10^{-6}$ | $10^{-3}$ | $3\times10^{-2}$ | $10^{-1}$ | 1 |
|---|---|---|---|---|---|---|
| 10 | 收敛 | 收敛 | 收敛 | **收敛** | **发散**($t{=}8$) | 发散 |
| 20 | 收敛 | 收敛 | 收敛 | **收敛** | **发散**($t{=}5$) | 发散 |
| 30 | 收敛 | 收敛 | 收敛 | **收敛** | **发散**($t{=}5$) | 发散 |

而**同一个 $\eta$** 上、从 `init=0.1` 出发的轨迹**真发散**（§2.3）。⇒ **同一 $\eta$、同一系统：小扰动收敛、固定初值发散** ⇒ 逃逸由**吸引域边界**决定，与谱和瞬态都无关。**且 $\varepsilon^\ast$ 的区间 $(3\times10^{-2},\ 10^{-1}]$ 在 $\gamma{=}10/20/30$ 上完全一致。**

**注（$\eta_{\rm mid}$ 的定义）**：正式轮的判据 ③ 用 $\eta_{\rm mid}:=(\eta_{\rm th}+\rho_\lambda)/2$，由构造严格落在缺口内部：

| $\gamma$ | 3 | 10 | 20 | 30 |
|---|---|---|---|---|
| $\eta_{\rm mid}$ | 0.066920961 | 0.0082100652 | 0.0020374083 | 0.00086269365 |
| $\eta_{\rm mid}/\eta_{\rm th}$ | 1.0299 | 1.1242 | 1.2944 | 1.4925 |
| $\eta_{\rm mid}/\rho_\lambda$ | 0.9718 | 0.9006 | 0.8147 | 0.7519 |

## §3 主假设与判据（**阈值已冻结，不得改动**）

### 主假设 **H83**
`target` 的缺口 $\rho_\lambda/\eta_{\rm th}$ 是**有限幅度（吸引域）现象**：在 $\eta\in(\eta_{\rm th},\rho_\lambda)$ 上不动点**线性稳定**且**瞬态放大为 $O(1)$**，但有限幅度的初值仍会逃出吸引域并**真发散**。

**否证条款**：若在 $\eta_{\rm mid}$ 上，$\varepsilon=3\times10^{-2}$ 也**发散**，或 $\varepsilon=10^{-1}$ 也**收敛**（二者任一发生，即在任一 $\gamma$ 上越出 $[3\times10^{-2},\,10^{-1}]$ 这个括号），则 H83 的「有限幅度」刻画**被证伪**。

### 判据（四项，全部为门禁）

| # | 判据 | 阈值（冻结） |
|---|---|---|
| **①** | **线性稳定**：四个 $\gamma$ 上 $\mu(\eta_{\rm mid})<0$，即 $\max_{\text{非中性}}\lvert\lambda(J(\eta_{\rm mid}))\rvert<1$ | 四格**全** $<1$ |
| **②** | **瞬态有界**：四个 $\gamma$ 上 $\Lambda(\eta_{\rm mid})\le2.0$（$T{=}2\times10^4$，gfix 代表） | 四格**全** $\le2.0$ |
| **★③** | **有限幅度**：在 $\eta_{\rm mid}$ 上，从 `gfix_qr`($x^\ast$)$+\varepsilon\hat\delta$ 出发、$N{=}2\times10^5$ 步内 —— $\varepsilon=3\times10^{-2}$ 的 $\lVert W\rVert$ **始终有限**（收敛），且 $\varepsilon=10^{-1}$ 的 $\lVert W\rVert$ **变为非有限**（真发散） | 四格**全**满足；括号 $[3\times10^{-2},10^{-1}]$ 一个都不越出 |
| **④** | **阴性对照**（装置有效性）：$\eta=0.5\,\rho_\lambda$ 上 $\varepsilon=1$ 也**收敛**（$\lVert W\rVert$ 有限，且 $\max_t\lVert W\rVert/\lVert W^\ast\rVert\le3$） | 四格**全**满足 |

**「$p$ 符号约定（纪律 194）」**：本卡的指数一律取 $p=-\text{slope}$；本卡不设指数判据。

### 次级读数（非门禁，如实报）
- $b$：$\lvert\mu(\eta_{\rm th})\rvert/\mu_{\rm floor}$ 的 $\gamma$ 律（探针报 123.7→1.82）。
- $c$：$\Lambda(\eta)$ 的 $\eta$ 依赖（探针报 $\eta\le\rho_\lambda$ 时基本常数）。
- $d$：$\varepsilon^\ast$ 的 $\gamma$ 依赖（探针在 $3\times10^{-2}$/$10^{-1}$ 的分辨率下**看不出** $\gamma$ 依赖）。

## §4 测量对象与算例（开卡前已逐位核过）

**测量对象**：在**规范参数化代表** `dir82/jac82.gfix_qr` 上（`D-103` 引理：$\rho_\lambda$ 在 $O(2)$ 上精确不变，且实测 `gfix_qr($x^\ast$)` 本身仍是残差 $1.0\times10^{-15}$ 的不动点），
- $\mu(\eta)=\ln\max\{|\lambda|:\ |\lambda-1|>10^{-6}\}$（**不用** `eig` 的特征向量基，`DEF-81-6`）；
- $\Lambda(\eta)=\sup_{1\le t\le T}\lVert J(\eta)^t\rVert_2$，$T=2\times10^4$；
- $\varepsilon^\ast$：沿固定 $\hat\delta$（seed 7777）的**真发散**（非有限）判据，$N=2\times10^5$。

**算例（已核）**：$\gamma{=}30$、$\eta{=}1.00\,\eta_{\rm th}$：从 `init=0.1` 出发 $\max_t\lVert W_t\rVert/\lVert W^\ast\rVert=1.18$、$\mu_{\rm tail}=-4.79\times10^{-9}$、不发散；同一 $\eta$ 上 $\Lambda=1.391$。$\gamma{=}30$、$\eta{=}1.25\eta_{\rm th}$：$t_{\rm div}=2322$、非有限。两点读数已逐位复核。

## §5 Stage A 守卫断言（硬门禁，任一 FAIL 阻断正式轮）

| # | 断言 | 阈值 |
|---|---|---|
| A1 | `gfix_qr($x^\ast$)` 的固定点残差 $\lVert\text{step}(x)-x\rVert/\lVert x\rVert$ | $\le10^{-13}$ |
| A2 | $\lVert W^\ast\rVert_F/\bigl(\sqrt2\,u_1^{\rm pred}\bigr)-1$（四格，$\lVert\cdot\rVert_F$ 对 $u_1$ 的 $\sqrt2$ 因子） | $\le10^{-4}$ |
| A3 | `gfix_qr` 后 $V_1,V_2$ 的正交误差 $\lVert V^\top V-I\rVert$ | $\le10^{-13}$ |
| A4 | 算例：$\gamma{=}30,\eta{=}\eta_{\rm th}$ 的 $\mu_{\rm tail}$ 与本卡 §4 记的 $-4.79\times10^{-9}$ 相对差 | $\le5\%$（**需 $T$ 加倍自洽**，`D-101` 边界 4） |
| A5 | 二分 $\varepsilon^\ast$ 的每个判定两端都**有限**（NaN 必须走 `not finite` 分支，不得当数值比较，`DEF-81-1`） | 结构断言 |
| A6 | $\eta_{\rm mid}$ 必须严格满足 $\eta_{\rm th}<\eta_{\rm mid}<\rho_\lambda$（四格） | 布尔 |
| A7 | $\Lambda$ 的 SVD 在 $\lVert J^t\rVert>10^{12}$ 时必须走 overflow 分支而非抛异常 | 结构断言 |
| A8 | 从历史 JSON 取 $\eta_{\rm th}/\rho_\lambda$ 一律**数值匹配**，不得用 `%g` 拼键（纪律 192） | 结构断言 |

## §6 已知缺陷与本次开卡更正（`DEF-83-1…3`）

| # | 内容 |
|---|---|
| `DEF-83-1` | 探针第 1 部分把 $\lVert W_1\rVert_F$ 与标量 $u_1$ 直接比，报「rel 4.1e-01」——那是 $\sqrt2-1$：$\lVert W^\ast\rVert_F=\sqrt2 u_1$。**读数没错，标签错**（`DEF-81-2` 同类）。Stage A 的 A2 已改为显式带 $\sqrt2$。 |
| `DEF-83-2` | 探针第 2 部分 E 段用 $\eta=1.25\eta_{\rm th}$，但 $\gamma{=}3$ 的 $\rho_\lambda/\eta_{\rm th}$ **仅 1.0599** ⇒ 1.25$\eta_{\rm th}$ 在 $\rho_\lambda$ **之上**，测的是线性不稳，与缺口无关。该段 $\gamma{=}3$ 数据**作废**；正式轮改用 $\eta_{\rm mid}$（由构造落在缺口内）。 |
| `DEF-83-3` | 探针第 2 部分 F 段在 $\eta\ge\rho_\lambda$ 上 $\lVert J^t\rVert$ 溢出致 `SVD did not converge` 抛异常中断。改为 $10^{12}$ 上限 + 异常捕获（A7）。 |

## §7 证据边界（提前声明，不得外推）
1. 只有 $c{=}D{=}6$、$\lambda{=}0.05$、$a{=}0.1$、$L{=}2$、单种子抽签（20261118）、$\gamma\in\{3,10,20,30\}$；$R\ge3$ 需重数。
2. 只测 `target`（`keys` 的缺口已在 `D-103` 证为**代表伪像**，不在本卡范围）。
3. 「有限幅度」= **值域**判据（真发散 vs 收敛），**不是** basin 的几何体积；本卡**不**测 basin 半径的 $\gamma$ 律。
4. $\hat\delta$ 只取**一个**固定方向（seed 7777）。$\varepsilon^\ast$ 可能方向依赖；本卡**不**扫方向，故 $\varepsilon^\ast$ 只作**该方向**的读数。
5. `init=0.1` 是 `D-79`/`D-80` 的冻结口径；本卡**不**扫 `init`（`DIR-81` 已扫过，见 `D-102`）。
6. 不改装置（`ref70`/`step72`/`jac79`/`jac81`/`jac82` 一字未改）；不重跑轨迹求 $\eta_{\rm th}$（沿用冻结值）。
7. 不含 $S$/LLM；不改 Freeze A–E；不给安全 $\eta$ 建议。
8. 本卡**解释不了**「为什么 $\varepsilon^\ast$ 与 $\gamma$ 无关而缺口随 $\gamma$ 增长」——那是派生问题（§8-①）。

## §8 派生（下一方向候选）
**①（最优先）** 既然 $\varepsilon^\ast$ 在固定 $\eta/\eta_{\rm th}$ 上与 $\gamma$ **无关**（§2.4），而缺口随 $\gamma$ 增长，那么**缺口必须由「沿实际接近路径的过冲」而非「围绕 $x^\ast$ 的 basin 半径」决定**。可测：从 `init=0.1` 出发的轨迹在**何时、朝哪个方向**离开吸引域（记录首次越过 $\lVert W\rVert=\lVert W^\ast\rVert$ 的时刻与当时的 $(W_1,W_2)$ 谱）；并以 `ref70.pred_target` 解析式核对 runaway 通道（$W_2\to B_1\to W_1\to C_2\to W_2$ 的正反馈，增益 $\propto\gamma$）。
**②** 在**规范代表**下重测 `keys` 的 $\Lambda$（`D-103` §4-3 的欠账；本卡已证 $\Lambda$ 与代表的关系尚未测）。
**③** $\varepsilon^\ast$ 的**方向依赖**：$\hat\delta$ 扫 $GL(2)\times GL(2)$ 的若干切向。