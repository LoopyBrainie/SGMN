# DIR-87b · 方向卡 rev 1.0（**冻结**）—— `S` 的容量对 $W^0$ 的依赖

> **出处（非自造）**：框架 `framework/SGMN_v2.4_架构设计.md` §7.2（**行号 526**）「$S$ 的更新、检索、**容量**和遗忘」；
> 其中「**$W^0$ 的具体构造**」在同节**留白**（`dir23/R1_scope.py:44` 已记该留白为 `BASES` 第三轴）。
> **派生自 `D-116` 的显式残项**：`D-116` 把 hebbian 的瓶颈定位到 $\lVert W^0\rVert=\sqrt d$ 淹没 $\eta\sum G$（相对 0.14%），
> 并明写「**容量对 $W^0$ 的依赖 = 本轮未测的显式残项**」。本卡只补这一项，**不另立议题**。

---

## §1 主假说与否证条款（冻结）

**H87b**：hebbian `S` 的容量瓶颈**是**初始权重范数 $\lVert W^0\rVert$，**不是**键间干扰。
⇒ 推论：在 `base="empty"`（$W^0\equiv0$）下，**同一条** $\eta{=}0.1$、一次写的 hebbian 写规则应表现出
**可用容量**，且该容量**随 $d$ 增长**。

**否证条款（frozen，两条任一成立即否证）**：
- **F1**：`empty` 下 hebbian 的命中曲线在**两个 $d$** 上对所有 $N\ge4$ 都**不离机会水平**（判据 `C87b-NOTMEM`）⇒ H87b 否证，结论改为「hebbian `S` 在本族参数下**不是记忆**」。
- **F2**：`empty` 下 hebbian 有容量但 **$N_{0.5}$ 不随 $d$ 增长**（`C87b-SCALE` FAIL）⇒ H87b 的「$d$ 标度」那一半否证。

★ **两种结局都闭合 §7.2-4 的容量子项**（这是本卡被立项的理由）。

---

## §2 MVP 标准（冻结 · 数值阈值此后不得改动）

**最小可裁定产物** = 一张 $6\text{ arm}\times5\text{ seed}\times20\ N$ 的命中率表，加上由它算出的
**$N_{0.5}$（半衰点）** 与 **$\Delta$hit(empty,rand)**。除此之外不需要任何东西。

### §2.1 测量对象（逐字沿用 `dir87/R1_task87.py`，只加 `base` 轴）

- **装置**：`dir22/sgmn_module_hebb_v7.py` md5 `1a242d02c530307f6c76646ca13469ef`（一字**不改**）。
- **写**：`core.write(k_i, v_i, sig=None)`，**`oneshot` 单遍**（$E{=}1$，逐事件一次写）。★ 本卡**不跑** `rehearse`（`D-116` 已证明它在 `rand` 下无效；`empty` 下的排练属新自变量，留待另轮）。
- **读**：`Y = core.read(K[:N])`（**整批**，`(N,d_v)`），`read` 内 `y = sg(W) q`。
- **解码**：`decode_hits`，$\hat j=\arg\max_j\cos(y,v_j)$，命中 $\iff\hat j=i$。**与 `D-116` 逐字相同**。
- **`base` 轴**（本卡唯一新自变量）：
  - `empty`：`HebbianCore(..., factorized=False)` 后 `core.W.zero_(); core.W0.zero_()`（逐字沿用 `dir23/R1_scope.py:123-126`）；
  - `rand`：`HebbianCore(..., factorized=True)`（装置默认口径，$W^0=U_0V_0^\top$ 随机）。
  - **`explicit` 不吃 `base`**（`ExplicitKVCore` 无 $W^0$）⇒ 记为 `base="n/a"`，**不得**伪造 empty/rand 对比。
- **网格**：`d ∈ {8,64}` · `seed ∈ {20261028,20261104,7,20261111,20261118}`（= `dir23:33`）·
  `N ∈ {1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,512,900,1024,2048}`（**20 点**；小 $N$ 加密以分辨 $N_{0.5}$）。
- **arm（6 条）**：`hebbian/empty/d8`、`hebbian/empty/d64`、`hebbian/rand/d8`、`hebbian/rand/d64`、`explicit/n-a/d8`、`explicit/n-a/d64`。

### §2.2 Stage A 硬门断言（★ 全部**先手算**再冻结 —— `CORR-24`）

记 $r\equiv\eta\alpha=0.1$、$\delta\equiv\eta\lambda=0.005$，$\alpha{=}1.0$、$\lambda{=}0.05$、$\eta{=}0.1$（`dir23:36`）。

| 编号 | 断言 | **手算值（推导见 §2.3）** | 容差 |
|---|---|---|---|
| `A87b-00a` | `empty`/`d=8`/`N=1`：$\lVert W - r\,v_0k_0^\top\rVert_F$ | $W$ **精确等于** $0.1\,v_0k_0^\top$ | $\le10^{-6}$ |
| `A87b-00b` | 同上：$\lVert W\rVert_F$ | $r\lVert v_0\rVert\lVert k_0\rVert=\mathbf{0.1}$ | $\le10^{-6}$ |
| `A87b-00c` | 同上：$\cos(y_0,v_0)$ | $\mathbf{1.0}$（$y_0=r\,v_0(k_0^\top k_0)=r\,v_0$） | $\le10^{-5}$ |
| `A87b-00d` | 同上：`data_norm`（最后一步数据项范数） | $\lVert\alpha\,\mathrm{outer}(W_{\rm prev}k_0-v_0,k_0)\rVert=\lVert v_0\rVert\lVert k_0\rVert=\mathbf{1.0}$ | $\le10^{-5}$ |
| `A87b-00e` | 同上：`reg_norm` | $\lambda\lVert W_{\rm prev}-W^0\rVert=\mathbf{0}$（$W_{\rm prev}{=}W^0{=}0$） | **精确 0** |
| `A87b-01` | `N=1` 后 $\sigma_1/\sigma_0$（第二/第一奇异值） | $\mathbf{0}$（$W$ 是**单外积**，rank **恰为 1**） | $\le10^{-6}$ |
| `A87b-02` | `empty`/`d=8`/`N=2`：$\cos(y_0,v_0)$ vs §2.3 闭式 | $\dfrac{A+Bt}{\sqrt{A^2+B^2+2ABt}}$，$A{=}0.0995-0.01s^2$、$B{=}0.1s$、$s{=}k_0{\cdot}k_1$、$t{=}v_0{\cdot}v_1$ | $\le10^{-5}$ |
| `A87b-03` | $W^0$ 范数：`empty` $\lVert W^0\rVert_F$；`rand` $\lVert W^0\rVert_F$ | `empty` **精确 0**；`rand` **$>0$**（只断言符号，**不断言** $\sqrt d$ —— 见注） | `empty` 精确 0 |
| `A87b-04` | 跨 `base` 数据不变性：`sha256(K)` | 与 `base` **无关**（`make_data` 只依赖 $(d,\mathrm{seed})$） | 逐位相同 |
| `A87b-05` | **复现 `D-116`**：`hebbian/rand` 与 `explicit` 在**共有 $N$** 上的命中率 | 与 `dir87/out/D87_formal_*.json` 的 `curves` **逐位同值** | $\le10^{-9}$ |
| `A87b-06` | 越界五条：源内无 `torch.optim`；无 `LayerMem`/`SGMNStack` 实例化；写后 `(alpha,lam,eta)` 不变；写后 `W0` 不变；不调 `write_side_loss` | 全为**否** | 全过 |
| `A87b-07` | 网格逐格有读数 | $6\times5\times20=\mathbf{600}$ 格 | 全有 |
| `A87b-08` | `N=1` 命中（全部 6 arm） | $\mathbf{1.0}$（单候选，解码器平凡正确） | $\ge0.999$ |

> **`A87b-03` 为什么不断言 $\lVert W^0_{\rm rand}\rVert=\sqrt d$**：`dir22:260` 是 `randn/sqrt(d_k)`，故
> $\mathbb E\lVert W^0\rVert_F^2=d_v$。但 $d{=}8$ 时只有 $d_vd_k{=}64$ 个元素，相对涨落 $\sim1/\sqrt{128}\approx\mathbf{8.8\%}$，
> **大于**可安全手算的门 ⇒ 按 `CORR-10`（门不得设在我不能手算验证的地方）只断言**符号**，量值**只报告**。
> 实测值 `D-116` 诊断已给：$d{=}8\Rightarrow2.828$、$d{=}64\Rightarrow8.0$（$\sqrt d$ 吻合但**不作门**）。

### §2.3 算例（worked example，全部可纸面复算）

**（a）$N{=}1$。** $W_{\rm prev}{=}0$、$W^0{=}0$ ⇒
$G=\alpha\,\mathrm{outer}(W_{\rm prev}k_0-v_0,k_0)+\lambda(W_{\rm prev}-W^0)=\mathrm{outer}(-v_0,k_0)$；
$W\leftarrow W_{\rm prev}-\eta G=r\,v_0k_0^\top$。于是 $\lVert W\rVert_F=r$、
$y_0=Wk_0=r v_0(k_0^\top k_0)=r v_0$ ⇒ $\cos=1$。`data_norm`$=\lVert-\mathrm{outer}(v_0,k_0)\rVert=1$、`reg_norm`$=0$。⇒ `A87b-00a…e`、`A87b-01`。

**（b）$N{=}2$。** 记 $s=k_0{\cdot}k_1$。$W_1=r\,v_0k_0^\top$，
$W_1k_1=r s\,v_0$ ⇒ $G_2=\mathrm{outer}(r s\,v_0-v_1,k_1)+\lambda W_1$ ⇒
$$W_2=(1-\delta)W_1-\eta r s\,v_0k_1^\top+\eta\,v_1k_1^\top
=0.0995\,v_0k_0^\top-0.01s\,v_0k_1^\top+0.1\,v_1k_1^\top .$$
读 $q{=}k_0$：$y_0=0.0995\,v_0-0.01s^2v_0+0.1s\,v_1=(0.0995-0.01s^2)v_0+0.1s\,v_1$。
因 $v_0,v_1$ **未必正交**（$t\equiv v_0{\cdot}v_1$ 一般 $\ne0$），
$$\cos(y_0,v_0)=\frac{A+Bt}{\sqrt{A^2+B^2+2ABt}},\quad A=0.0995-0.01s^2,\ B=0.1s.$$
⇒ `A87b-02`。**数值感**：$d{=}8$ 时 $\mathbb E|s|\approx d^{-1/2}{=}0.354$ ⇒ $B/A\approx0.36$；
$d{=}64$ 时 $\mathbb E|s|{=}0.125$ ⇒ $B/A\approx0.125$ ⇒ **干扰随 $d$ 降**，即 H87b 的方向。

**（c）为何 `empty` 可能翻转 `D-116` 的结论。**`D-116` 的 `rand` 下读出 $y\approx W^0q$，
而 $W^0$ 与「写了什么」无关 ⇒ 命中贴机会水平。`empty` 下 $y=\sum_i c_i\,v_ik_i^\top q$ 与写入**同相**；
且 $\lVert y\rVert=r\approx0.1$ **虽小，但解码器对余弦是尺度无关的** ⇒ 幅度小本身不损害命中。
这正是「瓶颈是 $\lVert W^0\rVert$ 而非干扰」的可判据化。

---

## §3 判据（冻结 · 三支互斥，优先级自上而下）

**$N_{0.5}$ 定义（`CORR-23`：比值/半衰点，**不用**绝对命中门）**：
在 $N$ 递增的击中率曲线上找**首个** $h(N)<0.5$，与前一格 $h(N_{\rm prev})\!\ge\!0.5$ 做**线性插值**：
$N_{0.5}=N_{\rm prev}+(0.5-h(N_{\rm prev}))\cdot\dfrac{N-N_{\rm prev}}{h(N)-h(N_{\rm prev})}$。
若 $h(2)<0.5$ 则按 $N_{\rm prev}{=}1,h(1){=}1$ 插值（$N_{0.5}\in(1,2)$）。
若直到网格末 $h(2048)\ge0.5$ ⇒ **右删失**，记 $N_{0.5}\ge2048$（**只作下界**）。

| 判据 | 冻结阈值 | 为何这个数（手算依据） |
|---|---|---|
| **`C87b-NOTMEM`**（否证器，**最先判**） | 若 `empty`/hebbian 在**两个 $d$** 上对所有 $N\ge4$ 都有 $h\le\max(3/N,\,0.10)$ ⇒ **NOT-MEMORY** 成立 | $3/N$ = 三倍机会水平（`CORR-23` 要求「离开机会水平」）；$0.10$ 为其地板。两者取大 ⇒ 不惩罚大 $N$ 处 $3/N$ 过小 |
| **`C87b-GAIN`**（机制：$W^0$ 主导？） | $\max_{N\ge4}\big[h_{\rm empty}(N)-h_{\rm rand}(N)\big]\ \ge\ \mathbf{0.20}$，**两个 $d$ 都要满足** | 手算：$d{=}64$ 干扰比 ${\approx}0.125$ ⇒ $h_{\rm empty}(4\ldots16)\approx1$ vs $h_{\rm rand}\approx0$（`D-116` 实测 $d64/N{=}4{:}0.000$）⇒ 差 ${\approx}0.98$；$d{=}8$ 干扰比 ${\approx}0.36$ ⇒ $h_{\rm empty}(4){\approx}0.8\text{–}1.0$ vs $h_{\rm rand}{=}0.350$ ⇒ 差 ${\approx}0.5$。**0.20 是这两者的保守地板** |
| **`C87b-SCALE`**（$d$ 标度，主假说的后半） | $N_{0.5}^{\rm empty}(d{=}64)\big/N_{0.5}^{\rm empty}(d{=}8)\ >\ \mathbf{1.5}$ | 理论 $N^\ast\sim d/(2\ln N)$：$d{=}8\Rightarrow{\sim}2$、$d{=}64\Rightarrow{\sim}10\text{–}16$ ⇒ 比 ${\sim}5\text{–}8$。**1.5 是保守地板** |

**删失处理**：若 $d{=}64$ 右删失（$N_{0.5}\ge2048$）而 $d{=}8$ 有限 ⇒ 用下界比，比值**仍成立则 PASS**（删失只让真比值更大）；若**两者都删失** ⇒ `C87b-SCALE` **不可计算**（按规则**不得**记 PASS/FAIL），如实报「网格不足以分辨」。

**判定合成**：`NOTMEM` 成立 ⇒ H87b **否证（F1）**，结论「hebbian `S` 在本族参数下不是记忆」；否则 `GAIN ∧ SCALE` 全过 ⇒ H87b **证实**；否则 **部分**（逐条报哪一支失败，含 F2）。

---

## §4 越界裁定（开卡前必须做）

- **裁定：不越界。** 本卡纯 inference-only：**无优化器**、**不实例化** `LayerMem`/`SGMNStack`、不调 `write_side_loss`、不改装置一行（`A87b-06` 实测守）。
- 框架第 **219** 行禁止的「外层通道」指**带参 $S$**（`D-30` 的警告）；本卡**不引入任何新参数**，`W`/`W^0` 只是互易记忆状态本身 ⇒ 不触该禁令。
- `base` 是框架 §7.2 自己留白的「$W^0$ 构造」，**不是我加的自由度**。

## §5 边界（不算结论的部分）

1. 只测**一次写**（`oneshot`）；排练/多遍是另一个自变量，本卡**不覆盖** $W^0{=}0$ 下排练的效果。
2. `explicit` 不吃 `base`，其曲线只作**装置不变性参照**，**不得**写成「$W^0$ 对 explicit 有/无影响」的发现。
3. 合成随机键/值 ⇒ 结论是**装置几何/代数**产物，**不得**写成内容性收益。
4. 命中率**不是** LLM 任务收益（§7.2-7 是另一行）。
5. hebbian `W` **没有槽位驱逐**（`W` 无限累加，rank $\le\min(N,d_v)$），故 $N>1024$ 时 hebbian 与 explicit 的机制**不同**，不可混谈「容量」。
6. 不得外推到 `factorized=False` 以外的其它写规则族。

## §6 冻结网格与产物

- 网格见 §2.1；读数为 `dir87b/out/D87b_{stageA,formal}_*.json`。
- 产物：`dir87b/R1_task87b.py`（逐字派生自 `dir87/R1_task87.py`，**只**加 `base` 轴、加密网格、换判据）、`revolve/DIR-87b/VERDICT.md`、证据行 **`D-117`**。

## §7 缺陷表（rev 1.0 起，追加式）

*(暂无；实现/Stage A 中发现的缺陷逐条追加，**不回改**上面已冻结的阈值)*
