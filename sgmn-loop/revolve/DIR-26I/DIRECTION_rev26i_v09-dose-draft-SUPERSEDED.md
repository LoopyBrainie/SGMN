# ⚠️ 已作废的草案（**从未冻结、从未开跑**）—— 保留留档，勿据此开跑

**作废理由（advisory review, 2026-11-09）**：本草案的主判据（载荷比）**在冻结前的锚点条款下已被确定** —— 锚点要求 s=0 等价于 D-40 的 Fu 臂、s=1 等价于 Ku 臂且逐位相等，那么该比值必然等于 0.0300/0.1064 = 0.282，落在 [0.70,1.43] 之外 ⇒ 通道 (i) 永不可能成立，正式轮只能重放 D-40。
**另两处**：① §1 用名义 s、§6 却说「判据用实测 A」，两套口径未合并；② N=10 的 |Δ|=0.030 与 se≈0.029 同阶，Λ 在那里无定义。
**保留价值**：§0 的盘上排除（候选②「剂量解释」被否）与 §4 的算例仍有效，已并入 `DIRECTION_rev26i_v10.md` §0。

---

# DIR-26-I · 方向卡片草案 —— 剖面效应的**剂量-响应**：载荷到底改的是「剂量」还是「敏感度」？

**卡片 rev 1.0（冻结）** · 2026-11-09 · 承接 **D-40**（DIR-26-H）与框架 §7.2「$\alpha$ 的产生规则及内容选择收益」
命名空间 `DIR26I` · 装置 `dir22/sgmn_module_hebb_v7.py`（md5 `1a242d02c530307f6c76646ca13469ef`，**零改动**）

---

## §0 为什么要这一轮（D-40 留下的量纲问题）

D-40 的读数：$\lvert\Delta(F-K)\rvert$ 在 $N{=}10$ 只有 $0.030$，$N{=}35$ 是 $0.106$（**3.5 倍**），$N\ge20$ 后进平台。但"$N$ 改了什么"没有被分离。开跑前已用**盘上数据**排除了三条候选解释中的两条：

| 候选解释 | 判据量 | 实测 | 结论 |
|---|---|---|---|
| ② 低载荷时 $\bar q$ 更大 ⇒ 剖面被 cap 压平、名义剂量更弱 | 实现剂量 $A=\tfrac12\big[(p_2+p_3)-(p_0+p_1)\big]$（逐轮平均发放数的"后重减前重"对比） | $N{=}10$：$A_F=-1.580$、$A_K=+1.580$ ⇒ 摆幅 **3.16**；$N{=}20$ 3.49、$N{=}35$ **3.21**、$N{=}45$ 3.52、$N{=}60$ 3.22（**极差仅 11%**） | **② 被否**：剂量几乎与载荷无关，而 $\lvert\Delta\rvert$ 变了 3.8 倍 ⇒ 差异只能来自**敏感度** |
| ③ 读侧 $n$ 小 ⇒ 竞争者少 ⇒ 写侧差异被压缩 | 读集大小 | 见下（**本轮不测**，但已在 §2(f) 记为保留解释） | 未决 |
| ① 每条目有效方向数 vs 支撑维数的竞争强度 | — | $N{=}10$ 时 40 个方向已挤 8 维 ⇒ "过订阅比"不自洽（D-40 §2 M8 的自评） | 未决 |

★ 由此**唯一还没被直接测量**的量是：**单位剂量能换来多少精度差**（敏感度 $S=\partial\Delta/\partial A$）。本轮把它测出来，而且是在**同一条剖面轴上**用连续剂量测——这既给出 $S(N)$，也给出 $\Delta$ 关于剂量的**形状**（线性 / 阈值 / 饱和）。
★ 每一条剂量点都由**同一套构造**（`build_keep`）实现，只改轮权重向量 $\mathbf w(s)$；镜像对 $\mathbf w(s)$ 与 $\mathrm{rev}\,\mathbf w(s)$ 同时跑，逐位检验反演对称。

---

## §1 主假说与判定通道（**判据；冻结后不得改**）

**H26I（主假说）**：剖面效应**与剂量成正比、且单位剂量的敏感度与载荷无关** —— 即 $\Delta(s)\approx s\cdot\Delta(1)$，且 $\lvert\Delta(1)\rvert$ 在 $N{=}10$ 与 $N{=}35$ 上相差不超过 1.43 倍。

**剂量轴**：$\mathbf w(s)=(1-s)\,\mathbf w_F+s\,\mathbf w_K$，$s\in\{0,\tfrac14,\tfrac12,\tfrac34,1\}$（$\sum w\equiv10.64$ 恒定；$s{=}0$ ⇒ 前重、$s{=}1$ ⇒ 后重）。
**主口径**：$\Delta(s)=\overline{\text{ho8}+\text{mean}}\big(\mathbf w(s)\big)-\overline{\text{ho8}+\text{mean}}\big(\mathbf w(0)\big)$，5 种子。$\Delta(0)\equiv0$ 是构造恒等，不作判据。
**镜像臂**：$\mathbf w^{\rm m}(s)=\mathrm{rev}\,\mathbf w(s)$，其剂量为 $-A(s)$；只用于 Stage A 的镜像恒等与 (e) 的对称性读数。

| 通道 | 数值判据（全部冻结） | 结论 |
|---|---|---|
| **(i) CONFIRMED** | 线性指数 $\Lambda=\big\lvert\Delta(0.5)-0.5\Delta(1)\big\rvert/\lvert\Delta(1)\rvert\le0.30$ **且** 载荷比 $\lvert\Delta(1;10)\rvert/\lvert\Delta(1;35)\rvert\in[0.70,1.43]$ **且** 单调（$\Delta(s)$ 在 $s$ 上 5 点 Spearman $\ge0.9$，两个 $N$ 都成立）且 $\lvert\Delta(1;35)\rvert\ge0.05$ | H26I 成立 ⇒ `SUCCEEDED` +1 |
| **(ii) FALSIFIED** | 载荷比 $<0.70$ 或 $>1.43$（**即剂量解释不成立、载荷改的是敏感度**），且线性指数 $\Lambda\le0.30$ | H26I 否证；但**剂量-响应是线性的** ⇒ 结论"敏感度随载荷变"成为硬读数 |
| **(iii) 非线性** | $\Lambda>0.30$（$\Delta$ 关于剂量弯折：阈值或饱和） | H26I 否证，**且**剂量-响应本身非线性 ⇒ 独立成判（与 (ii) 互斥，优先于 (ii)） |
| **(iv) INDETERMINATE** | 构造恒等 (d) 失败，或锚点 (c) 失败，或天花板 (f) 触发（任一 $N$ 上任一臂的 `ho8` 均值 $\ge0.95$） | 不计 |

**合成规则**（逐字执行）：先查 (iv) ⇒ 是则 INDETERMINATE；否则查 (iii) ⇒ 是则 **非线性**；否则查 (ii) ⇒ 是则 **FALSIFIED**；否则查 (i) ⇒ 是则 **CONFIRMED**；都不满足 ⇒ **WEAK**。
**`SUCCEEDED` 只在 (i) 时加一。**

---

## §2 Stage A 护栏（任一 FAIL 则不得开正式轮）

| 编号 | 断言 |
|---|---|
| A26I-1 | 装置 md5 == `1a242d02c530307f6c76646ca13469ef` |
| A26I-2 | 12 个复用对象（`build_stream`/`drive`/`make_core`/`metrics_plas`/`stream_idx`、`keep_positions`/`item_counts`、`ret_jitter`/`ret_jitter_bruteforce`、`calibrate_w`/`theta_of`、`_pos_idxs`、`round_quota`/`build_keep`/`realized_profile`/`even_counts`）**与 `dir26g/R1_gating7.py` 同一个对象**（`is` 断言） |
| A26I-3 | 标定后模块全局复位：`R1_gating5.N_DIST == 35`；五个 $N$ 上 $\rho{=}0.4$ 的 `skip_rate=0.400±0.05`、`gap≥0.05`、`repeat≥0.85`、`gate_ok` |
| A26I-4 | **剂量构造恒等**：每格点 $\sum$ 发放 $=k_B=\sum_i q_i$（逐位）；逐条目发放 $=q_i$（逐位）；$\sum_\rho \mathbf w(s)_\rho\equiv10.64$ |
| A26I-5 | **镜像恒等**：$\lvert\mathrm{keep}_{\mathbf w}\rvert=\lvert\mathrm{keep}_{\mathrm{rev}\mathbf w}\rvert$；实测剖面 $\mathbf w$ 与 $\mathrm{rev}\,\mathbf w$ **逐位镜像**；$A(\mathrm{rev}\,\mathbf w)=-A(\mathbf w)$ |
| A26I-6 | **锚点**：$s{=}0$ ⇒ 与 D-40 的 `Fu` 臂、$s{=}1$ ⇒ 与 `Ku` 臂，在 $N{=}10$ 与 $N{=}35$ 上、5 种子、`{ho4_mean,ho4_maj,ho8_mean,clean}` **逐位相等**（2×5×4=40 个值对/臂） |
| A26I-7 | 载荷无关性**如实报告**：$\bar q(N)$ 与实测剂量 $A(s)$ 在每个 (N,s) 上必须打印；不设阈值（D-40 的 DEF-26H-1 教训：护栏阈值不得用探针子集定） |
| A26I-8 | 稳定性 $\rho_{\rm stab}=1-\eta(S_{\rm key}+\lambda/2)>0$；`NONE` 臂 0 次发放且 $\lVert\Delta W\rVert=0$ |
| A26I-9 | 抖动指标逐位：`ret_jitter` 与 `ret_jitter_bruteforce` 在 $r_q\in\{4,8\}$ 一致，`maj_need` 3/5 |
| A26I-10 | 读后不变：每次读数后 $W$ 逐位不变 |
| A26I-11 | **流-臂同 $N$**：每处"臂 + 写流"必须断言 `D["n"] == 该臂的 N`（DEF-26H-2 第三次复发的教训） |
| A26I-12 | 逐条目配额上限：$\max_i q_i\le16$；`q_max` 逐格点打印 |

---

## §3 装置 / 测量 / 定标（逐字沿用）

- 装置：`dir22/sgmn_module_hebb_v7.py`，零改动。$d{=}896$，$\alpha$、$\lambda$、$\eta$、$M{=}2$、$S_{\rm key}{=}5$ 全部沿用。
- 构造：`build_stream(seed, d, n, R_PRES, θ)`（$N$ 条目 × $R{=}4$ 轮，$\cos\theta_{\rm w}{=}\cos\theta_{\rm r}{=}0.9$）、`keep_positions`、`item_counts`、`round_quota`、`build_keep`、`realized_profile`（**同一个对象**）。
- 测量：`ret_jitter(P, K, vb, n, r_q)`（读流独立抖动键 `seed+999`）、`metrics_plas`（clean 读）。
- 定标：`calibrate_w(r0=8, ρ=0.40, R_PRES, θ≻0.9)`，**必须在该格点的同一个 $n$ 上取分位数**，调用后复位（D-40 A26H-3）。
- 臂：`A`（全写，容量参照）、`w(s)` 与 `w^m(s)` 各 5 个剂量点（$s\in\{0,\frac14,\frac12,\frac34,1\}$；其中 $s{=}0$ 的镜像与 $s{=}1$ 同剂量 ±、互为反演，**只跑一份**并逐位对照）、`B`（因果基线，仅诊断）、`P`（位置基线，仅诊断）。
- 格点：$N\in\{10,35\}$ × 5 种子 × (A + B + P + 5 剂量臂 + 3 镜像臂) = $2\times5\times11=110$ 次驱动。

---

## §4 测量层的算例核对（冻结前必须过，`worked`）

用 $N{=}10$、seed $20261028$、$s{=}\tfrac12$ 手算并核对：
1. $\mathbf w(0.5)=(2.46,2.86,2.86,2.46)$、$\sum=10.64$；
2. $\mathrm{rev}\,\mathbf w(0.5)=(2.46,2.86,2.86,2.46)$ ⇒ **$s{=}\tfrac12$ 的镜像就是它自己**（剂量 $A=0$）⇒ 该格点只需跑一份，且 $A$ 必须为 $0$（浮点 $\lvert A\rvert\le10^{-9}$）；
3. `round_quota` 在 $q\in[0,16]$ 全表上 $\sum$ 成立、每轮 $\le4$；$\mathbf w$ 与 $\mathrm{rev}\,\mathbf w$ 的配额表互为反演；
4. $s{=}0.25$ 与 $s{=}0.75$ 的剂量必须严格互为相反数（逐位）；
5. 实测剖面（`realized_profile`）与手算配额表的逐轮平均一致。

---

## §5 判据量清单（必须全部落盘）

$\Delta(s)$（$s\in\{0,\frac14,\frac12,\frac34,1\}$）× 2 个 $N$、逐种子值、sd、正号计数；线性指数 $\Lambda$（两个 $N$）；载荷比 $\lvert\Delta(1;10)\rvert/\lvert\Delta(1;35)\rvert$；Spearman($s,\Delta$)；实测剂量 $A(s)$；$\bar q$、$q_{\max}$、$k_B$；六臂电平（`ho4`/`ho8`/`clean`）；镜像臂电平与剂量；$\Delta_{\rm ho4}$ 副口径；天花板检查。

---

## §6 已知风险与边界（开跑前声明）

- $\mathbf w(s)$ 的**形状**在中途会经过"碗形"（$s{=}\frac12$ 时两端高、中段高、$w_0{=}w_3{<}w_1{=}w_2$）⇒ $\Delta(0.5)$ 不是"均匀剖面"，而是"碗形 vs 前重"的对比。**这是剂量插值的必然结果，必须在报告里写明**，不得把 $\Delta(0.5)$ 当成"均匀基线"。
- 剂量 $A$ 是**实测**（`realized_profile`），不是名义 $s$；判据用实测 $A$ 拟合，$s$ 只作索引。
- 本轮**不**回答"读侧竞争者数"（解释 ③）：它需要另一套干预（固定写侧、改读集），留作下一方向。
- 天花板检查只看 `ho8` 均值；$N{=}10$ 的 $A$ 臂在 D-40 是 0.82、$N{=}35$ 是 0.71 ⇒ 预期不触发，但必须实测。
