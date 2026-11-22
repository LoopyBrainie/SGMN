# DIR-78 · 方向卡 rev 1.0 —— **操作域是 gauge 的性质**：同一 $W^\ast$ 的 bal/warm 两因子化各有自己的精确线性阈值

> 状态：`FROZEN`。出处：**§7.2「因子化随机在线更新的操作域」** + D-98 §2（$\eta_{\rm th}=\rho_\lambda$，5/5，$\le0.5\%$）+ D-95 §3「gauge 诊断」+ D-94「代表点依赖」。
> 冻结时刻 2026-11-18 · Cycle 131。**不是新议题**：D-98 证明 bal 上"逃逸阈值 = 精确线性边界"；D-94 的 warm 读数（target 30 **0.9965**、keys 10 **0.5679**）与 bal（1.0539 / 1.1481）差**近 2 倍** ⇒ 若同一 $W$ 的两个因子化代表点各自有自己的线性阈值，"操作域"就**不是 $W$ 的性质而是 gauge 的性质**。

## 1 · 主假设 H78 与否证条款

**H78（gauge 依赖）**：同一 $W^\ast$（同一不动点流形上的两个因子化代表点 bal 与 warm）有**各自的精确线性阈值** $\rho_\lambda$；warm 的 $\rho_{\rm div}$ **也**等于它的 $\rho_\lambda$；且 $\rho_\lambda^{\rm warm}<\rho_\lambda^{\rm bal}$。

- **否证 H78**：① warm 上 $\rho_{\rm div}$ 与 $\rho_\lambda$ 差异 $>2\%$ 的格 $\ge2$ 个（即"阈值 = 精确线性边界"这条**不跨 gauge 成立**）**或** ② $\rho_\lambda^{\rm warm}<\rho_\lambda^{\rm bal}$ 不严格 5/5。

## 2 · 装置（逐字派生）

`dir77/jac77.py`（48 维展平 + 前向差分 $h{=}10^{-9}$ + 模最大特征对）与 `dir77/diag77.py` 的"**首个 $\lambda_{\max}>1+10^{-6}$** 细扫 + 二分"**逐字照用**；`near73.{eta_lin, freeze_fp, perturb, run_from, bisect_near}`、`step72.step`、`ref70.{pred_target,pred_keys}` **全原样**。$\delta{=}10^{-3}$、$n{=}2\times10^5$、`rel=0.02`、`perturb` 种子**固定 20261118**（与 D-98 参照点同一颗）。

| gauge | 代表点 | 来源 |
|---|---|---|
| **bal** | $U_1{=}V_1{=}\sqrt{u_1^\ast}E$、$U_2{=}V_2{=}\sqrt{u_2^\ast}E$，$E{=}[e_1\,e_2]$ | `ref70.pred_target/pred_keys` |
| **warm** | `near73.freeze_fp(mode, $\gamma$, $0.3\,\eta^\ast_{\rm lin}$)` | 同装置 |

**唯一自变量 = gauge**。5 个格：target 3/10/30、keys 3/10。**bal 行不重测**，**引用 D-98 冻结值**（$\rho_\lambda$ 与 $\rho_{\rm div}$ 皆已落盘，写明"引用"）；本轮只跑 **warm 行 5 个二分**（括号 $[0.30,\,1.50]\cdot\eta^\ast_{\rm lin}$，端检失败 ⇒ 该格 `CENSORED`）。

## 3 · 判据（冻结）

| # | 判据 | 阈值 |
|---|---|---|
| **①** | **warm 一致（主判据）**：5 格的 $\left|\rho_{\rm div}^{\rm warm}-\rho_\lambda^{\rm warm}\right|/\rho_\lambda^{\rm warm}$ | $\le2\%$ 者 **$\ge4/5$** |
| **②** | **gauge 分裂**：$\rho_\lambda^{\rm warm}<\rho_\lambda^{\rm bal}$（bal 取 D-98 冻结值） | 严格 **5/5** |
| **③** | 报告（不判）：每格两 gauge 的 $\rho_\lambda$、$\rho_{\rm div}$、括号、$n_{\rm eval}$、$\lambda_{\max}$ 曲线、$\|W_{\rm bal}-W_{\rm warm}\|/\|W_{\rm warm}\|$、穿越点导数 $d\lambda/d\rho$；并与 D-94 冻结 warm 值（target 30 $\mathbf{0.9965}$、keys 10 $\mathbf{0.5679}$、target 3 $\mathbf{0.9940}$）**并列**（只作对照，不进判据） | — |
| **④** | 删失处置：任一 `CENSORED` ⇒ 明写删失哪格；$\ge3$ 格 CENSORED ⇒ **整轮 `CENSORED`** | — |
| **⑤** | 若 ② 成立而 ① 不成立 ⇒ **`PARTIAL`**，并明写"gauge 分裂存在，但'阈值 = 精确线性边界'不跨 gauge 成立" | — |

## 4 · 诊断（不进判据，`DEF-77-1` 的更正测法）

在 **target 30 · bal**、$\eta{=}1.10\,\eta^\ast_{\rm lin}$（$\lambda_{\max}{=}1.0876>1$）上跑 5 颗种子（20261118–20261122），量 $t_{\rm div}$ 与 $\ln(1/\left|\cos\right|)$ 的关系。预测斜率 $=1/\ln\lambda_{\max}{=}1/\ln(1.0876)=11.8$（$\pm15\%$）、$R^2\ge0.95$。**这是"方向效应住在时间里、不住在阈值里"的直接检验**（D-98 §3 的推导）。

## 5 · Stage A 守卫（全过才开正式轮）

| # | 断言 | 阈值 |
|---|---|---|
| **A1** | **warm 是（装置自己场里的）不动点**：$\|{\rm step}(x_w,10^{-9})-x_w\|/(10^{-9}\|x_w\|)$ | $\le10^{-6}$，5/5 |
| **A2** | **同一 $W^\ast$**：$\|W_{\rm bal}-W_{\rm warm}\|/\|W_{\rm warm}\|$ | $\le10^{-9}$，5/5 |
| **A3** | **展平/反展平排序正确**：$J(\eta{=}0)=I_{48}$ | $\max\left|J-I\right|\le10^{-9}$，10/10 点 |
| **A4** | **特征值层交叉验证**：dense `eig` 与 20 次幂迭代（同一 $J$）的 $\lambda_{\max}$ 相对差 | $\le10^{-6}$，3 点（warm target 30、warm keys 10、bal target 30） |
| **A5** | **窗口可穿越**：5 格 warm 在 $1.30\,\eta^\ast_{\rm lin}$ 处 $\lambda_{\max}>1+10^{-6}$ | 5/5（否则该格 CENSORED） |
| **A6** | **正控/负控**：warm 在 $0.30\,\eta^\ast_{\rm lin}$ 走 2000 步不 `div`；在 $1.50\,\eta^\ast_{\rm lin}$ 于 $\le2000$ 步 `div` | 5/5 |
| **A7** | **二分单调**（warm target 30，`rel=0.5`）：`ok=True`、`lo<η<hi`、每次试探 = 当前 log 括号中点、括号严格收缩 | 全成立 |
| **A8** | **测度对象的闭式工作量例**：标量松弛 $x'=(1-\eta b)x$ 的 $J=1-\eta b$，用**同一条细扫+二分路径**求 $\lambda{=}1$ 的穿越 $\rho$，与解析 $\rho^\ast=1/b$ 比 | 相对差 $\le10^{-12}$（$b{=}3.7$） |

## 6 · 预言（带宽 $\ge0.15$）

| # | 预言 |
|---|---|
| **P1** | ① 通过：$\ge4/5$ |
| **P2** | ② 通过：5/5 |
| **P3** | $\rho_\lambda^{\rm warm}$(target 30) $\in[0.85,\,1.10]$（宽 0.25） |
| **P4** | $\rho_\lambda^{\rm warm}$(keys 10) $\in[0.40,\,0.75]$（宽 0.35） |
| **P5** | ①② ⇒ H78 **`SUCCEEDED`** |
| **P6** | 诊断斜率 $\in[10.0,\,13.6]$（$11.8\pm15\%$）且 $R^2\ge0.95$ |

## 7 · 边界（写作前声明）

1. 单 $\delta$、单 $n$、单种子（20261118）、单 `rel`；5 个格只覆盖 target 与 keys 两模式的 $\gamma\le30$。
2. bal 行是**引用** D-98 冻结值（同装置、同种子、括号略异：$[0.85,1.50]$ vs D-96 的 $[0.90,1.30]$）⇒ 其 $\pm1\%$ 终止分辨率**带入** ② 的比较。
3. "warm"= `freeze_fp(…,0.3η*_lin)` 这一条具体路径所得的平衡代表点，**不是**所有非 bal 代表点；不动点流形是 8 维 ⇒ **代表点的总体分布未测**（D-94 §7 已自陈）。
4. 若 warm 的 $\rho_\lambda$ 落在 $\eta^\ast_{\rm lin}$ **之下**（如 keys 10 预测 0.57），那只说明"`ref70` 的界不是**跨 gauge** 的安全上界"，**不**说明该界在 bal 上失效（D-98 已证 bal 上它是上界但松 5–15%）。
5. 诊断（§4）只在 **一个格、一个 $\eta$** 上测斜率，**不**据此建立普适"时间律"。
6. 不改 v2.4 Freeze A–E；不改写 D-94/D-95/D-96/D-97/D-98 任何一行。
