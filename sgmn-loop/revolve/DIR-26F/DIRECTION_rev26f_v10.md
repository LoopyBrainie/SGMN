# DIR-26-F · **B 的惩罚机制** —— 内容门控的收益与负债来自「内容选择」还是「发放不均」？（rev 1.0，**冻结**）

> **来源（不得自造）**：`framework/SGMN_v2.4_架构设计.md` §7.2「写入映射与门控 …… $\alpha$ 的产生规则及**内容选择收益**」；§5.2 原文「**内容相关门控是否带来独立于速率效应的收益，仍属 EXP-07 的开放问题**」。本卡判的正是这条开放项里最低层的分解问题：**B 相对 P 的效应到底由"选了哪些轮次"承载，还是由"各条目拿到多少写"承载。**
> **既有账**：D-34/D-35（DIR-26/DIR-26-B/C）确认真实格点上 B 优于 P（$+0.15$）；D-36（读侧错位）与 D-37（写侧错位）都在惩罚格上看到 **B < P**（读侧 $r{=}8$、$\cos\theta_{\rm r}{=}0.5$：$-0.0657$，**0/5**；写侧 $\cos\theta_{\rm w}{=}0.3$：$r{=}8$ $-0.0386$、$r{=}12$ $-0.0543$，均 **0/5**）。D-37 **M6** 给出的解释是"错位把'地址可分辨'退化成'随机方向覆盖'，此时**覆盖游戏**占优"——**该解释从未被直接检验**。装置：`dir22/sgmn_module_hebb_v7.py`（md5 `1a242d02c530307f6c76646ca13469ef`，**零改动**）。

---

## §1 主假说与证伪

**H26F（覆盖假说）**：B 相对 P 的**增益与惩罚都由「逐条目发放数不均」承载**；在逐条目发放数被拉平的条件下，**条目内的内容排序选择不产生可分辨的效应**。

**判据形式（2×2 因子设计，$N{=}35$ 条目 × $R{=}4$ 轮）**

|  | 逐条目计数 **不均** | 逐条目计数 **均** |
|---|---|---|
| 条目内选择 **内容** | **B** | **E**（=B-even） |
| 条目内选择 **位置** | **U**（=P-uneven） | **P** |

四个主效应（每个格点、每个种子各自算，再对种子取均值）：
$$\begin{aligned}
\delta_{\rm sel\mid uneven}&=\Delta(B-U), & \delta_{\rm sel\mid even}&=\Delta(E-P) \quad(\star)\\
\delta_{\rm cnt\mid content}&=\Delta(B-E), & \delta_{\rm cnt\mid pos}&=\Delta(U-P)
\end{aligned}$$
其中 $\Delta(X-Y)=\mathrm{ret}_1(X)-\mathrm{ret}_1(Y)$（`ho4`+mean）。恒等式 $\Delta(B-P)=\delta_{\rm cnt\mid content}+\delta_{\rm sel\mid even}=\delta_{\rm sel\mid uneven}+\delta_{\rm cnt\mid pos}$ 逐位成立（构造审计 A26F-5）。

**H26F 主张**：$(\star)$ 两列 —— 即"内容选择"在两个计数水平上的主效应 —— **都不显著**：
$$\bigl|\delta_{\rm sel\mid even}\bigr|\le0.05\quad\text{且}\quad\bigl|\delta_{\rm sel\mid uneven}\bigr|\le0.05\qquad \forall\ \text{5 个判据格点}$$
而 $\Delta(B-P)$ 的全部量（健康档 $+0.10\sim0.16$、惩罚档 $-0.04\sim-0.07$）由 $\delta_{\rm cnt}$ 承接（(b2)）。

**证伪**：任一判据格点上 $\bigl|\delta_{\rm sel\mid even}\bigr|\ge0.10$ 或 $\bigl|\delta_{\rm sel\mid uneven}\bigr|\ge0.10$ ⇔ **内容选择即使均分也显著有效/有害** ⇒ 覆盖假说不成立，须报出**方向**（内容选择带来正效应=健康档仍有内容收益；负效应=内容选择在错位下本身有害）与所在格点。

**不存在"更强即更好"的读法**：若健康档上 $\delta_{\rm sel\mid even}\ge+0.10$，那是**对 H26F 的证伪**、但对"内容选择有效益"是**好消息**——本卡只判 H26F，两种读法都要如实记入裁决。

## §2 测量对象

**指标（唯一判据口径）**：**import** `R1_gating3.ret_jitter`（同一对象）。主口径 `ho4`+mean（$r_q{=}4$）；副口径 `ho8`+maj（$r_q{=}8$，多数票门槛 5/8）。`clean` 读（原型键、`metrics_plas`）为报告项。

**六格点（$r$ × 读侧 $\cos\theta_{\rm r}$ × 写侧 $\cos\theta_{\rm w}$）**

| 格点 | $r$ | $\cos\theta_{\rm r}$ | $\cos\theta_{\rm w}$ | 角色 | 冻结参考 $\Delta(B-P)$（`ho4`+mean） |
|---|---|---|---|---|---|
| **H8** | 8 | 0.9 | 0.9 | 健康（正对照） | $+0.150$（D-35） |
| **H12** | 12 | 0.9 | 0.9 | 健康（正对照） | $+0.159$（D-35） |
| **PR8** | 8 | **0.5** | 0.9 | **读侧惩罚** | $-0.0657$（D-36） |
| **PW8** | 8 | 0.9 | **0.3** | **写侧惩罚** | $-0.0386$（D-37） |
| **PW12** | 12 | 0.9 | **0.3** | **写侧惩罚** | $-0.0543$（D-37） |
| NR12 | 12 | 0.5 | 0.9 | 无惩罚（**只报告**） | $+0.0086$（D-36） |

★ **判据格点 = 前 5 个**（2 健康 + 3 惩罚），NR12 **不参与判定**（它没有可解释的惩罚，只作"效应不存在时 $\delta$ 应为 0"的旁证）。★ **每个条款都在这 5 个格点上全报**，不得单选一格（纪律 (xiii)）。

**流的构造（与 D-36/D-37 逐位同一，锚点靠它）**
- 写流：`build_stream(seed, 896, 35, 4, theta_of(cosθ_w))` —— $\cos\theta{=}0.9$ 档**必须**用 `THETA_W0=radians(25.8419)` 本身，不用 `acos(0.9)`（DEF-26C-1；`theta_of` 从 `R1_gating5` **import**）。
- 读流：`ho4 = build_stream(seed+999, …, r_pres=4, theta_of(cosθ_r))`、`ho8 = … r_pres=8 …`（同 D-36 的 `HO_OFFSET`）。
- $c_\tau$：按 $(\rm r0,\cos\theta_{\rm w})$ 定标（`R1_gating5.calibrate_w`，**import**），4 次 = $\{8,12\}\times\{0.9,0.3\}$。**跳过率固定 0.40**。

### 工作算例（先跑即核，不是判据）
`dir26f/probe26f_arms.py`（$d{=}16$、$N{=}8$、$R{=}4$、$k$ 取真实比例）：核 (i) E 臂 $\sum$发放 $=k_B$ 且逐条目 $\max-\min\le1$；(ii) U 臂逐条目计数与 B 逐位相同；(iii) E 与 P 的 keep 集合 Jaccard $<0.9$（E 不是 P 的复制品）；(iv) E 保留轮次的 $c_{\rm now}$ 严格不高于其丢弃轮次（内容升序）。

## §3 臂的构造（**冻结，逐字实现**）

- **A**：`drive(core, D, "A")` —— 全写（$k_A{=}T{=}560$）。
- **B**：`drive(core, D, "B", tau=c_tau)` —— 内容门控，$c_{\rm now}>c_\tau$ 则跳过。其 log 逐条给 $(\rho,i,m_{\rm br},m,c_{\rm now},{\rm issued})$，按 `stream_idx` 顺序与呈现一一对应。
- **E（B-even）**：keep 集合 $S_E$ 由 **B 的 log** 生成：
  1. $q_i\in\{\lfloor k_B/N\rfloor,\lceil k_B/N\rceil\}$，$\sum_i q_i=k_B$；
  2. 条目内把该条目的 $2RM$ 个呈现按 B 的 $c_{\rm now}$ **升序**排列，取前 $q_i$ 个；
  3. 差额 $r_B=k_B-N\lfloor k_B/N\rfloor$ 个 $+1$ 名额给"**下一候选的 $c_{\rm now}$ 最低**"的那 $r_B$ 个条目（**内容决定谁多拿一个**）；
  4. 以 `drive(core, D, "R", keep=S_E)` 执行。
  ⇒ E 与 B **总发放数逐位相同**、与 P **逐条目计数只差 $\pm1$**；E 与 P 的差别**只剩"条目内选哪些轮次"**（内容 vs 位置）。
  ★ 构造理由（写死）：E 的策略读**B 的在线前向记录**（同一信息集：内容余弦），不引入未来信息；这正是"保留内容门控的判断，只把配额拉平"的字面实现。
- **P**：`drive(core, D, "R", keep=keep_positions(N, R, k_B))` —— **import** `R1_gating2.keep_positions`，与 D-35/D-36/D-37 同一函数同一参数（锚点靠它）。
- **U（P-uneven）**：keep 集合 $S_U$ 的**逐条目计数 $=c_i^B$ 逐位相同**（$c_i^B$ 由 B 的发放集合按 `item_counts` 数出），条目内用**位置规则**（与 `keep_positions` 同款 `np.linspace` 均匀取点）选 $c_i^B$ 个 ⇒ U 与 B 只差"条目内选哪些轮次"，U 与 P 只差"逐条目计数"。

★ **五臂同流、同种子、同 $W^0$、同读流**；臂间的唯一差别是 keep/门控规则。

## §4 判据（**阈值一次冻全**）

| 子句 | 内容 | 阈值 | 依据 |
|---|---|---|---|
| **(c)** 前置复现 | ① 健康档 $\Delta(B-P)\ge+0.10$（两 $r$）；② 3 个惩罚子格 $\Delta(B-P)\le-0.03$；③ **逐位锚点**：H8 的 A/B/P 与 D-35、PR8 的 B/P 与 D-36、PW8/PW12 的 B/P 与 D-37 **逐位相同**（5 种子全查） | ①② 如上；③ 逐位 | 机制问题只在效应复现后才有意义；③ 是"同装置同口径"的唯一凭据 |
| **(a)** ★ 主判据 · 选择主效应为零 | $\bigl|\delta_{\rm sel\mid even}\bigr|\le0.05$ **且** $\bigl|\delta_{\rm sel\mid uneven}\bigr|\le0.05$，5 个判据格点**全部**满足（共 10 个数值） | $\le0.05$ | 5 种子配对差的 SEM 实测 $0.016\sim0.035$（D-36/D-37），取 **2× 最大 SEM** ⇒ "不可分辨" |
| **(b)** 计数主效应承接 | **(b1)** 记账恒等逐位（$\le10^{-12}$）；**(b2)** $\bigl|\delta_{\rm cnt\mid content}-\delta_{\rm cnt\mid pos}\bigr|\le0.10$（内容选择下的计数效应与位置选择下的计数效应同量级） | b1 逐位；b2 $\le0.10$ | b2 阈值 = 健康档主效应的量级；两者若差到一个主效应，说明"计数效应"依赖选择规则、不能单独归因 |
| **(e)** 位置不均复现（报告，不判） | $\bigl|\delta_{\rm cnt\mid pos}-\Delta(B-P)\bigr|$ 与 $\bigl|\delta_{\rm cnt\mid content}-\Delta(B-P)\bigr|$ 逐格点 | — | 若两者都 $\approx0$，则"位置不均即可复现内容不均"，内容完全无关 |
| **(f)** 副口径一致（报告+判） | `ho8`+maj 上的 $\delta_{\rm sel\mid even}$ 与 `ho4`+mean 上同号且差 $\le0.10$ | $\le0.10$ | 防止主口径的多数票假象 |
| **(g)** 发放分布（报告，不判） | 每格点逐臂：逐条目计数 $\max-\min$、$\mathrm{entropy}$、$k$；E/U 与 B 的 keep 集合 Jaccard | — | 构造审计的可读版 |
| **(h)** `clean` 读（报告，不判） | 五臂的原型键读数 | — | 区分"读出映射受益"与"存储结构受益"（用户规则：内容性收益 vs 通用注入收益不得混报） |
| **(i)** 护栏 | 纯标量不变性（U 臂，$c\in\{0.37,1,2.5\}$） | 逐位 | 指标的数学性质 |

**判定合成（冻结）**
- **CONFIRMED** ⇔ (c) PASS **且** (a) 10/10 **且** (b1) **且** (b2) **且** (f)。
- **FALSIFIED** ⇔ 任一判据格点上 $\bigl|\delta_{\rm sel\mid even}\bigr|\ge0.10$ 或 $\bigl|\delta_{\rm sel\mid uneven}\bigr|\ge0.10$（须报方向与格点）。
- **INDETERMINATE** ⇔ (c) ③ FAIL（锚点不逐位 ⇒ 口径坏了，本轮的 $\delta$ 不可信）。
- **WEAK** ⇔ 其余（含 $0.05<|\delta_{\rm sel}|<0.10$ 的边界情形，或 (b2)/(f) FAIL）。

**为什么这个分解能定案**：$\Delta(B-P)$ 只告诉我们"内容门控整体好/坏"。要判"覆盖游戏"，必须把"**谁拿到写**（$\delta_{\rm cnt}$）"与"**什么时候写**（$\delta_{\rm sel}$）"分开——四臂 2×2 是分离它们所需的最小设计，且 $(\star)$ 列在两个计数水平上各测一次，互为复核。

## §5 Stage A 断言（硬门）

| 编号 | 断言 |
|---|---|
| **A26F-1** | 装置 md5 == `1a242d02c530307f6c76646ca13469ef` |
| **A26F-2** | `unit, make_core, build_stream, drive, metrics_plas, stream_idx` → `"R1_gating"`；`keep_positions, item_counts` → `"R1_gating2"`；`ret_jitter, ret_jitter_bruteforce` → `"R1_gating3"`；`calibrate_w, theta_of` → `"R1_gating5"`（全为同一对象） |
| **A26F-3** | ★ **四臂构造审计**（$N{=}8$ 最小实例 + 一个真实 $k_B$）：E 臂 $\sum$发放 $=k_B$ 且逐条目 $\max-\min\le1$；U 臂逐条目计数 $=c_i^B$ 逐位、$\sum=k_B$；P 臂 $\max-\min\le1$；**E 与 P 的 keep 集合 Jaccard $<0.9$**（E 不是 P 的复制品） |
| **A26F-4** | E 臂**确实读内容**：逐条目（$q_i>0$）核"保留轮次的 $c_{\rm now}$ 最大值 $\le$ 丢弃轮次的 $c_{\rm now}$ 最小值"，全部条目成立；报告平均 gap |
| **A26F-5** | 记账恒等：真实格点上 $\Delta(B-P)=\delta_{\rm cnt\mid content}+\delta_{\rm sel\mid even}=\delta_{\rm sel\mid uneven}+\delta_{\rm cnt\mid pos}$，偏差 $\le10^{-12}$ |
| **A26F-6** | ★ **三锚点逐位复现**（1 种子，DEF-26E-2 的最小实例原则）：H8 的 A/B/P 与 D-35 的 `Jout`（`ho4` mean 与 maj）；PR8 的 B/P 与 D-36 `reads["0.5"]`；PW8 与 PW12 的 B/P 与 D-37 |
| **A26F-7** | 抖动指标逐位正确（$d{=}16$、矩阵式 vs 暴力 $\cos$；`maj_need`：$r_q{=}4\Rightarrow3$、$8\Rightarrow5$） |
| **A26F-8** | 护栏：纯标量不变性在 U 臂的 `ho4` 指标上逐位成立 |
| **A26F-9** | 稳定性 $\rho_{\rm stab}=1-\eta(\alpha S+\lambda/2)>0$；零发放臂 $k{=}0$ 且 $\lVert\Delta W\rVert=0$ |
| **A26F-10** | 读流与臂无关（同格点五臂共用同一读出对象）；$\overline{\cos({\rm ho4},{\rm proto})}=\cos\theta_{\rm r}\pm0.01$（$0.9$ 与 $0.5$ 两档各核一次） |
| **A26F-11** | A 臂实发 $=T=560$；读不改写（计算前后 $W$ 逐位不变） |

## §6 缺陷与纪律（本轮必须遵守并留档）

- **纪律 (xiii)**：判据格点不得单选一个 free 单元 ⇒ 5 个判据格点全报。
- **纪律 (xiv)**：应随独立变量变化的诊断量只能作结果不能作门槛 ⇒ 本卡 (g) 发放分布、(h) `clean` 读、(e) 位置不均均为**报告项**。
- **DEF-26E-2**：Stage A 里结构性必真的断言用最小实例 + 论证（A26F-3/A26F-6 只用 $N{=}8$ 与 1 种子）。
- 判据前**先落盘 rows**；`json.dump` 前先 `os.makedirs`；`cosine_similarity` 必写 `dim=-1`；不使用 `nanmax/nanmean`（DEF-26E-3）。

## §7 运行预算（冻结）

5 臂 × 6 格点 × 5 种子 = **150 次运行**（每次 1 次驱动，约 $4\sim5$ s）＋ 4 次定标 ⇒ 预计 $12\sim20$ 分钟。无 GPU、纯前向。

## §8 修订记录

| 版本 | 变更 | 是否影响判据 |
|---|---|---|
| rev 1.0 | 冻结：H26F、2×2 四臂、6 格点（5 判据 + 1 报告）、(a) 0.05 / (b2) 0.10 / (c) 三锚点、(f) 0.10 | — |
