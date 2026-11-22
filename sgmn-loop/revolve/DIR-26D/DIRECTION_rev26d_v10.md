# DIR-26-D · **读侧错位阈值** —— 内容收益在多大的查询错位下归零？（rev 1.0，**冻结**）

> **来源（不得自造）**：`framework/SGMN_v2.4_架构设计.md` §7.2「数值与结构选择 …… $\mathcal Q_l,\mathcal O_l$ 的内部结构、参数共享及**查询坐标对齐方式**」。
> **既有账**：D-35（DIR-26-C）把"读键是否参与写"拆开了：`clean`（原型键）→ `Jin`（写用实例键）几乎免费（$\le0.05$），`Jout`（**未参与写的独立抖动**，$\cos\theta_{\rm read}{=}0.9$）才付代价（B $-0.113$ @ $r{=}8$），但收益仍在（$\Delta^{\rm out}_{BP}{=}+0.1500/+0.1586$ @ $r{=}8/12$）。
> ⇒ §7.2 的**剩余空白就是错位幅度轴**：$\cos\theta_{\rm read}{=}0.9$ 是唯一测过的一档。本卡把 $\theta_{\rm read}$ 作为**唯一独立变量**（写侧固定 $\cos\theta_{\rm w}{=}0.9$），找收益归零的错位阈值；顺带测"**少写多读**"（D-35 §6-2）。
> 装置：`dir22/sgmn_module_hebb_v7.py`（md5 `1a242d02c530307f6c76646ca13469ef`，**零改动**）。测量层、P 臂、$\rho$ 定标、抖动指标全部 **import**（同一对象）。

---

## §1 主假说与证伪

**H26D**：内容门控的收益随**查询错位**单调衰减，且**归零阈值落在 $\cos\theta_{\rm read}\in[0.3,0.7)$**：
$$\Delta(\theta_{\rm r}):=\mathrm{ret}^{{\rm ho4},\rm mean}_1(B)-\mathrm{ret}^{{\rm ho4},\rm mean}_1(P)\quad(\rho{=}0.4,\ r{=}8)$$
$$\text{预测：}\ \Delta(0.7)\ \ge\ 0.10\quad\text{且}\quad \Delta(0.3)\ \le\ 0.05,\qquad \Delta(0.9)\ge\Delta(0.7)\ge\Delta(0.5)\ge\Delta(0.3)$$
**证伪（两个方向，都要如实记）**：
- **方向一（归零更早）**：$\Delta(0.7)\le0.05$ ⇒ 收益在 $45.6^\circ$ 错位前已死。
- **方向二（归零更晚/不存在）**：$\Delta(0.3)\ge0.10$ ⇒ 近正交读下仍存活，预测的区间错。
⇒ 两个方向都记 **FALSIFIED**（假说预测错），但方向二须在证据行注明"**收益比预测更稳健**"，不得写成"机制失效"。

## §2 测量对象

**写侧**（不变）：$R_{\rm w}$ 次呈现、$\theta_{\rm w}{=}25.8419^\circ$（$\cos{=}0.9$）、$k_{i,\rho}=\sqrt5\,{\rm norm}(\cos\theta_{\rm w}e_i+\sin\theta_{\rm w}u_{i,\rho})$。

**读侧（本卡的唯一独立变量）**：四条**独立流**（`build_stream(seed+999, …, θ_r)`，$e_i$ 为确定构造 ⇒ 各流 `proto` 逐位相同），
$$k^{\rm read}_{i}={}\sqrt5\,{\rm norm}(\cos\theta_{\rm r}e_i+\sin\theta_{\rm r}u'_{i}),\qquad \cos\theta_{\rm r}\in\{0.9,\ 0.7,\ 0.5,\ 0.3\}\ (\theta_{\rm r}=25.84^\circ,45.57^\circ,60^\circ,72.54^\circ)$$
每条 $\theta_{\rm r}$ 建**两条**读流：`ho4`（$r_q{=}4$，供 mean/maj4）与 `ho8`（$r_q{=}8$，供 maj8）。★ $\theta_{\rm r}{=}0.9$ 的 `ho4` 与 D-35 的 `Jout` 流**同种子同构造 ⇒ 逐位相同**（A26D-4），故 (e) 是硬复现校验而非"近似"。

**指标**（同 D-35，import 使用）：单键命中 $h(i,\rho)=\mathbf1[\arg\max_j\cos(Pk^{\rm read}_{i,\rho},v_j)=i]$，$P=W-W^0$；
$$\mathrm{ret}^{\rm mean}_1=\frac1{N r_q}\sum_{i,\rho}h,\qquad \mathrm{ret}^{\rm maj}_1=\frac1N\sum_i\mathbf1\Big[\textstyle\sum_\rho h\ge\lfloor r_q/2\rfloor+1\Big]$$
（$r_q{=}4\Rightarrow\ge3$；$r_q{=}8\Rightarrow\ge5$）。

### 工作算例（先跑即核，不是判据）
`dir26d/probe26d_align.py`：在 $d{=}16,N{=}8$ 上把 $\cos\theta_{\rm r}$ 四档各跑一遍，逐位核对 (i) $\cos(k^{\rm read}_i,{\rm proto}_i)=\cos\theta_{\rm r}\pm0.01$；(ii) 抖动指标与暴力 $\cos$ 循环逐位相同；(iii) 同一 $W$ 上 $\Delta(\theta_{\rm r})$ 随错位**非增**（侦察用，不作判据）。

## §3 设计（冻结）

**流**：$d{=}896$、$N_{\rm dist}{=}35$、$m{=}2$、$\lambda{=}0.05$、$\eta{=}0.1$、$S{=}5$、$\rho{=}0.4$、种子 `20261028, 20261104, 7, 20261111, 20261118`（定标 pilot `20261041`）。
**格点** $r\in\{8,12\}$（D-35 的两个 free 格点；$r{=}6$ 已被 D-35 判定为跨情节读的容量边界，**本卡不跑**）。

| 组 | $R_{\rm w}$ | 写步 $T$ | 臂 | 读 |
|---|---|---|---|---|
| **主组**（对齐轴） | 4 | 560 | A · **B**($\rho{=}.4$) · **P** | `ho4`/`ho8` × 4 档 $\theta_{\rm r}$ |
| **少写组** | **2** | **280** | A · **B**($\rho{=}.4$) · **P** | `ho4`/`ho8` × $\theta_{\rm r}\in\{0.9,0.7\}$ |

**运行数**：2 个 $r$ × 5 种子 × 6 次 = **60 次**（主组 3 次/格 + 少写组 3 次/格）。
**$\tau$ 定标**：$\rho{=}0.4$ 各 $r$ 一次（`R1_gating2.calibrate`，$R_{\rm w}{=}4$）；少写组 $R_{\rm w}{=}2$ 用本卡新增的 `calibrate_at(r0,ρ,R_w)`（**同一 quantile 规则与同一 gate_ok 判据**，只把 $R$ 参数化），并断言 $R_{\rm w}{=}4$ 时与 `calibrate` **逐位相同**（A26D-7）。

## §4 判据（**阈值一次冻全**）

| 子句 | 内容 | 阈值 |
|---|---|---|
| **(G)** 锚点闸门 | $\theta_{\rm r}{=}0.9$ 的 `ho4`+mean 下 $\{A,B,P\}$ 极差 | $<0.10$ ⇒ **INDETERMINATE**（整条轴不可解释） |
| **(a)** ★ 存活 | $\Delta(0.7)$ @ $r{=}8$ | $\ge0.10$ |
| **(b)** ★ 归零 | $\Delta(0.3)$ @ $r{=}8$ | $\le0.05$ |
| **(c)** 单调 | 相邻比较 $\Delta(0.9)\!\ge\!\Delta(0.7)\!\ge\!\Delta(0.5)\!\ge\!\Delta(0.3)$，每处容差 0.05 逆差 | $r{=}8$ **与** $r{=}12$ 两处都成立 |
| **(d)** 符号一致 | $\Delta(0.7)$ 的 5 种子符号 @ $r{=}8$ | **5/5 同号**（$p{=}0.031$） |
| **(e)** 锚点复现 | $\theta_{\rm r}{=}0.9$ 的 `ho4` 读数 vs D-35 的 `Jout` 读数（同 $W$、同读流） | **逐位相同**（若不同须给出原因并降级为报告） |
| **(f)** 多数票 | $\lvert\Delta^{\rm maj8}(0.7)-\Delta^{\rm mean4}(0.7)\rvert$，且同号 | $\le0.10$ |
| **(g)** 少写多读 | $R_{\rm w}{=}2$、$r_q{=}8$（$\ge5$）下 $\Delta_{w2}(0.7)$ | $\ge0.10$ |
| **(h)** 诚实附注（报告） | $\theta_{\rm r}{=}0.3$ 逐臂 $\mathrm{ret}_1$ vs 随机基线 $1/N{=}0.0286$ | 必须区分"塌到随机"与"持平但高于随机" |

**判定合成（冻结）**
- **CONFIRMED** ⇔ (G) PASS **且** (a)(b)(c)(d)(e)(f)(g) 全 PASS。
- **FALSIFIED** ⇔ (a) FAIL 或 (b) FAIL（方向一/方向二，须写明）。
- **WEAK** ⇔ (a)(b) 过但 (c) 或 (f) 或 (g) FAIL（衰减律/多数票/少写多读未成立，阈值结论仍记）。
- **INDETERMINATE** ⇔ (G) FAIL。

## §5 Stage A 断言（硬门）

| 编号 | 断言 |
|---|---|
| **A26D-1** | 装置 md5 == `1a242d02c530307f6c76646ca13469ef` |
| **A26D-2** | `unit, make_core, build_stream, drive, metrics_plas, stream_idx` 的 `__module__=="R1_gating"`；`keep_positions, calibrate` 的 `__module__=="R1_gating2"`；`ret_jitter, ret_jitter_bruteforce` 的 `__module__=="R1_gating3"`（全部同一对象） |
| **A26D-3** | **读键对齐度**：四档 $\theta_{\rm r}$ 上 $\overline{\cos(k^{\rm read}_i,{\rm proto}_i)}=\cos\theta_{\rm r}\pm0.01$，且 $\cos(k^{\rm read}_i,e_{j\ne i})$ 的绝对值 $\le0.05$ |
| **A26D-4** | ★ **锚点流逐位同一**：$\theta_{\rm r}{=}0.9$ 的 `ho4` 的键与 D-35 的 `Jout` 流**逐位相同**（`torch.equal`），且其读数与 `dir26c/out/DIR26Cr11_formal.json` 中对应格点的 `Jout` 读数**逐位相同** |
| **A26D-5** | 抖动指标逐位正确：$d{=}16$ 上矩阵式与暴力 $\cos$ 循环相同；`maj_need`：$r_q{=}4\Rightarrow3$、$r_q{=}8\Rightarrow5$ |
| **A26D-6** | 四条读流的 `proto` 彼此**逐位相同**、`va/vb` 也逐位相同（读侧只许改 $K$）⇒ 读侧差异**只**来自 $\theta_{\rm r}$ |
| **A26D-7** | `calibrate_at(r0,0.4,4)` 与 `R1_gating2.calibrate(r0,0.4)` 的 `c_tau`、`skip_rate` **逐位相同**；$R_{\rm w}{=}2$ 的实测跳过率 $\in0.4\pm0.05$ |
| **A26D-8** | 写步计数：主组 $T{=}560$、少写组 $T{=}280$；A 臂实发 $=T$；读不改写（计算前后 $W$ 逐位不变） |
| **A26D-9** | 稳定性 $\rho_{\rm stab}{=}1-\eta(\alpha S+\lambda/2)>0$；零发放臂 $W$ 逐位不变 |
| **A26D-10** | P 臂等量：两组 $R_{\rm w}$ 下 $k_P{=}k_B$ 逐位 |
| **A26D-11** | 护栏：纯标量不变性在**最坏错位**（$\theta_{\rm r}{=}0.3$）的指标上逐位成立 |

## §6 边界（不得外推）
1. 读侧错位仍**与写侧抖动同族**（同构造、独立实现）；真实 LM 的错位结构不同（D-19 的病根是"同批相关向量"）⇒ 本卡的阈值**不得**当 LM 的可用阈值。
2. 只扫 4 档 $\cos\theta_{\rm r}$（0.9/0.7/0.5/0.3）；归零点只在**这四档之间**定位，不得报连续曲线。
3. 写侧 $\theta_{\rm w}$ 固定在 0.9；"写错位也一起变"**未测**。
4. 只测 $\rho{=}0.4$；错位轴上 $\rho$ 交互**未测**（D-34/D-35 只测了 $\rho$ 在 $\theta_{\rm r}{=}0.9$ 的曲线）。
5. 少写组只测 $R_{\rm w}{=}2$ 与 $r_q{=}8$（$\ge5$）一档；$R_{\rm w}{=}1$、$r_q{=}16$ **未测**。
6. 聚合：mean 与多数票二种；分数裕度/top-$k$ **未测**。
7. $r\in\{8,12\}$ 单尺度；$r{=}6$ 已被 D-35 判为跨情节读的容量边界，本卡**不跑**、不得引用为"收益已死"。
8. 主判据在 $\mathrm{ret}_1$；$\mathrm{rowm}/\cos P$ 仅诊断，方向不同必须报告。
