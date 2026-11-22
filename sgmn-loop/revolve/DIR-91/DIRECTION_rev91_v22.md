# DIRECTION · DIR-91 · rev 2.0 — 内容寻址定位：瓶颈在 `$c\mapsto q$` 还是 `$W_o$`？

**任务**：`task.md` 顶部「DIR-91-rev2.4」。**来源**：`D-103` 机制 2
（$W^\star$ 由全部 32 事实**含 16 评测键**解出 ⇒ 对评测键近乎完美查表、**仍为 chance**
⇒ 瓶颈不在存储容量）。**派生自 v2.4 §7.2-7 与 §2.2 的读通路定义**，不新造议题。

---

## §0 · 装置事实（本轮**全部**推理的地基，逐条有出处）

`sgmn_module.py:65-76` 的读通路，逐字：

```
c = sg(h_{l-1}^+)      # (B,L,d_model)  层输入
q = Wq @ c             # (N,d_k)
y = sg(W) @ q          # (N,d_v)
m = Wo @ y             # (N,d_model)    注入 h_l^+ = h_l + gamma*m
```

`sgmn_module.py:82-89` 的写通路，逐字：

```
k = mean_t(Pk h_{l,t-1}) * pk_scale        # (d_k,)  跨 batch 与 token 取均值
v = mean_t(Pv h_{l,t})                     # (d_v,)
G = alpha*outer(Wk - v, k) + lam*(W - W0)  # 全矩阵分支更新 Wfull
```

⇒ **两个不同的投影**：写入用 `$P_k$`（× `pk_scale`），读取用 `$W_q$`。二者**无从属关系**
—— 除非读侧训练把它们对齐。**这正是本轮要测的东西。**

**尺度**：$R{=}8$、$d_k{=}d_v{=}896$、`pk_scale` 校准使单位方差输入下 $\mathbb E[k^2]{=}5$
⇒ $\lVert k\rVert^2\approx5\cdot896=4480$。$\alpha{=}1.0$、$\lambda{=}0.05$、
$\eta{=}3\times10^{-5}$、`MOUNT`$=[8,12,16]$、`READ_STEPS`$=600$、`READ_LR`$=10^{-3}$、
$N_{\rm facts}{=}32$、$N_{\rm train}{=}16$、$N_{\rm cand}{=}4$、$N_{\rm draws}{=}3$、
$n_q{=}16$（chance $0.25$）。

**已知读数（`D-103`，作为本轮的输入而非重复测量）**：$\gamma{=}0$ 下真实写入态
$\mathrm{rel}{=}0.1628/0.1914/0.2771$；闭式 $W^\star$ 把 $\mathrm{rel}$ 推到
$7.90/7.88/4.80\times10^{-4}$，**端到端仍为 chance**（$0.000\sim0.062$）。

---

## §1 · 主假说与否证条款（**冻结**）

**主假说 H1**：SGMN 端到端 4 选 1 在留出键上的失败，**主因是内容寻址**
（$c\mapsto q$ 无法把 $q_i$ 对准已存的 $k_i$），**而非输出映射**（$W_o$，$v\to$ 答案 token）。

**否证条款**：若在**寻址被完全修正**的条件下（$q_i$ 直接取已存的 $k_i$），读侧
**仍**学不到任务（$\mathrm{acc}_q\le0.25$，即不高于机会水平），则 H1 **被否证**：
瓶颈在下游（$W_o$ 的输出映射或其线性泛化），**不在寻址**。

> ★ **为什么用「干预」而不是「相关」**：`task.md` 原拟测**对角优势**
> $q_i^{\top}k_i$ vs $\max_{j\ne i}q_i^{\top}k_j$。但该量**只能说明"读不到"**，
> **不能排除**"读到了但 $W_o$ 用不了"。而 $y_i=Wq_i=\sum_j w_{ij}v_j$（$w_{ij}{=}q_i^{\top}k_j$）
> 只是**近似**叠加（真实解含 $\lambda$ 修正），故"对角优势大 ⇒ 必然成功"**不成立**。
> ⇒ 必须**直接干预** $q$，才能把寻址与输出映射**分离**。对角优势降为**机制读数**（§3）。

---

## §2 · MVP 标准与**数值门**（冻结；改动须新开 revision）

### 2.1 三臂（**仅 q 的来源不同**，其余逐位相同）

| 臂 | $q$ 的来源 | 读侧训练 | 作用 |
|---|---|---|---|
| `real` | $q=W_qc$（装置原样） | 是，600 步 | 基线，须复现 chance |
| `oracle_frozen` | $q\leftarrow k_i$（**仅评测时替换**） | **否**，直接用 `real` 臂训好的参数 | 「已训好的 $W_o$ 能否用上正确的 $y$」 |
| `oracle_trained` | $q\leftarrow k_i$（**训练与评测都替换**） | 是，600 步（同预算） | **★ 主判决臂**「寻址完美时读侧能否学到」 |

★ `oracle_frozen` 与 `oracle_trained` 用 `real` 臂**同一份** `Wq/Wo/gamma` 为初值（`oracle_frozen`
完全不训；`oracle_trained` 从同初值再训 600 步）。三臂**同一** `items`（`random.Random(555)`）。

### 2.2 判决规则（**主判决量 = `acc_q(oracle_trained)`，$n_q{=}16$，chance $0.25$**）

| 分支 | 条件（**两条同时**） | 含义 |
|---|---|---|
| **B（寻址）** | $\mathrm{acc}_q(\text{oracle\_trained})\ge0.40$ **且** 相对 `real` 提升 $\ge0.125$ | 寻址是瓶颈 ⇒ 改 `$c\mapsto q$` |
| **A（输出映射）** | $\mathrm{acc}_q(\text{oracle\_trained})\le0.25$ | 寻址完美也学不到 ⇒ 改 `$W_o$`/下游 |
| **INCONCLUSIVE** | 其余（$0.25<\cdot<0.40$，或提升 $<0.125$） | **不选分支**，如实报 |

`0.40` / `0.25` / `0.125` **沿用 `D-103` 已冻结的同一组门**（同 $n_q{=}16$、同 $p_0{=}0.25$），
理由：同一裁决口径便于**跨轮直接比较**；`0.40` 是 $\mathrm{Bin}(16,0.25)$ 的上尾切点
（$P(X\ge7)\approx2.7\%$），`0.25` 即机会水平本身。

### 2.3 一致性检查（**必报；不一致 ⇒ 判决降为 INCONCLUSIVE**）

机制读数 $\mathrm{share}_i$（§3.1）与判决分支必须相容：

| 判决 | 要求 | 理由 |
|---|---|---|
| **A** | median $\mathrm{share}\ge0.25$ | A 说"寻址没坏"，若 share 也显示没坏 ⇒ 相容 |
| **B** | median $\mathrm{share}\le0.25$ | B 说"寻址坏了"，share 须确实低 |

**若不相容 ⇒ 判决记为 `INCONCLUSIVE` 并写明冲突**——不得择一而报。

---

## §3 · 测量对象

### 3.1 机制读数（非干预，在 `real` 臂的终态上测）

对每个事实 $i$（$N{=}32$，含训练键与留出键，**分开报**）：

- $\bar c_i=\mathrm{mean}_t\big(h^{\rm in}_{l,t}\big)$（与 `key_matrices` **同一块、同一 token 集**）
- $q_i=W_q\bar c_i$；$k_i$、$v_i$ 取自 `key_matrices`（$\gamma{=}0$ 口径）
- $\mathrm{share}_i=\dfrac{\lvert q_i^{\top}k_i\rvert}{\sum_j\lvert q_i^{\top}k_j\rvert}$
  （完美寻址 $\to1$；随机 $\to1/32{=}0.031$）
- $\cos(q_i,k_i)$、$\max_{j\ne i}\lvert\cos(q_i,k_j)\rvert$
- $y_i=Wq_i$：$\cos(y_i,v_i)$、$\lVert y_i-v_i\rVert/\lVert v_i\rVert$
- **对照**：$\mathrm{share}^{\rm oracle}_i$ 用 $q_i\leftarrow k_i$ 计算（须 $\to1$，见 §4 A91-32）

### 3.2 判决量

`acc_q`（按题计，同题 $N_{\rm draws}{=}3$ 次全对才算命中）与 `acc_item`，三臂 × 两 seed × 三层。

### 3.3 测量对象的**已算例**（Stage A `A91-32` 固化）

取事实 0，令 $q{=}k_0$：则 $\mathrm{share}_0=|k_0^{\top}k_0|/\sum_j|k_0^{\top}k_j|$。
**由定义** $\mathrm{share}_0\ge\lVert k_0\rVert^2/(\lVert k_0\rVert^2+(N{-}1)\max_j|k_0^{\top}k_j|)$，
且当 $k_j\perp k_0$ 时 $\to1$。`A91-32` 断言该解析式与代码计算一致（相对误差 $\le10^{-6}$），
**从而证明 share 的实现不是"看起来像"**。

---

## §4 · Stage A 守卫断言（任一 FAIL ⇒ 阻断正式轮）

| 编号 | 断言 |
|---|---|
| `A91-30` | ★ **打补丁的读通路在无替换时与原读通路逐位相同**：`max|m_patched − m_orig| = 0`（否则三臂不可比） |
| `A91-31` | ★ **`oracle` 注入确实生效**：设 $q{=}k_i$ 后 $y_i$ 与 $W k_i$ 逐位相同（$\max|\Delta|{=}0$） |
| `A91-32` | 测量对象已算例（§3.3）：`share` 实现与解析式一致（$\le10^{-6}$），且 $q{=}k_i$ 时 $\mathrm{share}\to1$ |
| `A91-33` | $k_i$ 与 `key_matrices` 逐位相同（复用同一函数，$\gamma{=}0$）；写入数 $=[599,599,599]$ |
| `A91-34` | 三臂 `items` 完全相同（同 `Random(555)`、同 $n_{\rm item}{=}48$、同 $n_q{=}16$） |
| `A91-35` | `oracle_frozen` 与 `oracle_trained` 的**初值** `Wq/Wo/gamma` 与 `real` 臂逐位相同（$\max|\Delta|{=}0$） |
| `A91-36` | $\gamma{=}0$ 口径：$K,V$ 在"无写"与"写后"下逐位相同（沿用 `A91-26` 的做法，顺序建栈，`DEF-91-28`） |

---

## §5 · 证据边界（**先写死，防止事后解释**）

1. **不得**把 `oracle_*` 的准确率当"架构上限"：$q{=}k_i$ 需要**测试时不可得**的
   事实编号 ⇒ 它是**上界诊断**，不是可实现配置。
2. **不得**由 `oracle_trained` 成功推出"只需改 $c\mapsto q$ 就能达标"：它仍是**线性**读侧、
   $0.5$B 基座、600 步、16 训练键。
3. **不得**由 `oracle_frozen` 失败推出"$W_o$ 无用"：其输入分布与训练时不同（分布位移），
   失败可能只是位移所致；**该臂只作辅助读数，不作判决**。
4. **不得**把本轮的 A/B 结论外推到 $\gamma\neq0$、其他基座、其他 `MOUNT`。
5. `D-103` 已否决 $T$ 扫描；本轮**不**再涉写预算。
6. 训练键与留出键**必须分开报**；不得用训练键结果代替留出键（`D-103` 已证差一个泛化）。

---

## §6 · 预测（**先写死**，供事后核对，不作判据）

- **P1**：`real` 臂 $\mathrm{acc}_q\approx0.00\sim0.06$（复现 `D-103`）。
- **P2**：`oracle_frozen` $\mathrm{acc}_q\le0.25$（分布位移 + 只训 16 键的线性 $W_o$）。
- **P3**：median $\mathrm{share}$（留出键）$<0.10$（寻址确实坏）。
- **P4**：`oracle_trained` $\mathrm{acc}_q\ge0.40$ ⇒ 判 **B**。（**这是我对 H1 的赌注**。）

若 P4 落空（$\le0.25$）⇒ 判 A，H1 否证，**我上一轮"改攻键表征"的指向被推翻**，
须改攻 $W_o$/输出映射，并在 `revisions.md` 记为我的判断错误。

---

# 附 · rev 2.1（Cycle 156，Stage A 首轮两处守卫 FAIL 后）—— **只订正 §2.3 与 A91-30/32**

**触发（均为**我的**判据/守卫缺陷，不是装置缺陷）**

**`DEF-91-29`（A91-30 守卫无效：两次前向之间夹了一次写入）**
`A91-30` 断言「打补丁的读通路在无替换时与原生逐位相同」，实测 `max|Δm|=7.3e-02` **FAIL**。
根因：守卫里第一次 `model(chunks[0])` 会在 `no_write=False` 下触发 `mem.write()`
⇒ **W 被更新**；第二次前向用的是**另一个 W** ⇒ 两次 `m` 必然不同。
补丁本身在 `_q_oracle is None` 时**直接 `return _ORIG_READ(...)`**，不可能有差异。
⇒ 订正：两次前向都置 `no_write=True`（与三臂的评测口径一致），并**额外**做一次
「快照 W → 原生前向 → 还原 W → 补丁前向」的交叉校验。

**`DEF-91-30`（★★ §2.3 的 raw `share` 是**退化**判据：真键也只有 $1/N$）**
`A91-32` 的已算例暴露：令 $q{=}k_0$（**真键**），实测 $\mathrm{share}_0=\mathbf{0.0314}$，
而随机水平 $1/32=\mathbf{0.03125}$ —— **两者几乎相等**。
机理：32 个 $k_i$ 由一个**巨大公共模**主导（$k_i\cdot k_j$ 对所有 $j$ 量级相近），故
$$\mathrm{share}_0=\frac{\lVert k_0\rVert^2}{\sum_j\lvert k_0\cdot k_j\rvert}\approx\frac{\lVert k_0\rVert^2}{32\lVert k_0\rVert^2}=\frac1N .$$
⇒ **raw share 的天花板就是 $1/N$**，任何「$\ge0.25$」门**永远无法达到** ⇒ 若照 v2.0 §2.3 执行，
分支 **A 永不可能被判出**（系统性偏向 B）。**这是我设的门错。**
⇒ 订正：判别量改为**去均值（centered）share**，与 `R2_task91.py:369` 的 `key_cos`
（`A91-13` 已用 `cen_max`/`cen_mean`）**同一惯例**：
$$\tilde k_j=k_j-\bar k,\quad \mathrm{share}^{\rm cen}_i=\frac{\lvert q_i^{\top}\tilde k_i\rvert}{\sum_j\lvert q_i^{\top}\tilde k_j\rvert}.$$
**raw share 仍全量报告**（它本身是有价值的机制读数：说明键的近共线程度），
但**只作机制**，**不再承担一致性门**。

**订正后的 §2.3（冻结）**

| 判决 | 要求（centered） | 理由 |
|---|---|---|
| **A** | median $\mathrm{share}^{\rm cen}$（留出键）$\ge0.25$ | A 说"寻址没坏"，须确有身份结构 |
| **B** | median $\mathrm{share}^{\rm cen}$（留出键）$\le0.25$ | B 说"寻址坏了" |

**不变的**：§1 主假说与否证条款、§2.1 三臂、**§2.2 判决门 `0.40`/`0.25`/`0.125`**、
§3.2 判决量、§5 边界、§6 预测 P1–P4 —— **一字未改**。

**新增预测（先写死）**
- **P5**：raw $\mathrm{share}$（真键 $q{=}k_i$）$\approx1/N\pm20\%$（已验证：0.0314 vs 0.03125）。
- **P6**：median $\mathrm{share}^{\rm cen}$（留出键，$q{=}W_q\bar c$）$<0.10$（寻址确实坏）。

---

# 附 · rev 2.2（Cycle 156，`A91-32` FAIL 后）—— **§2.3 降为「只报告不否决」；A91-32 删去无效算例**

**触发（又是我自己的缺陷）**

**`DEF-91-31`（A91-32 的 centered 正交算例在数学上无效）**
我断言「正交基的 centered share == 1」。**错**：`centering(I_N)` 把每列变成
对角的 $1-1/N$、非对角的 $-1/N$ ⇒ **去均值后不再正交**，实测 centered share $=0.5$（$N{=}2$ 时的真值）。
⇒ 该子断言**前提不成立**，与实现无关。**订正**：删掉它；对 **raw 与 centered 都改用**
有效的解析下界 $\mathrm{share}(q{=}k_0)\ \ge\ \dfrac{\lVert\tilde k_0\rVert^2}{\lVert\tilde k_0\rVert^2+(N-1)\max_j\lvert\tilde k_0\cdot\tilde k_j\rvert}$。

**`DEF-91-32`（★★ 绝对 share 门**第二次**不可达 ⇒ 该门设计本身是错的）**
`A91-32` 实测：**真键**（$q{=}k_0$）的 **centered** share 只有 $\mathbf{0.0847}$。
⇒ 我 rev 2.1 定的「A 须 centered share $\ge0.25$」**同样不可达** —— 因为**完美寻址的上限就是 0.0847**，
不是 1。**连续两次把绝对门设在天花板之上，说明"绝对 share 门"这个设计是错的**，不是阈值选错。

**订正后的 §2.3（冻结）—— 降为「只报告，不否决」**

理由：**判决量是 oracle 干预臂**（直接替换 $q{=}k_i$，§2.2），它**不需要** share；
把 share 设成否决门只会引入**系统性偏向**（两次都偏 B），且不增加信息。
故 §2.3 改为**纯机制报告**，逐层报告四个量：
1. $\mathrm{share}^{\rm raw}(q{=}W_q\bar c)$ —— 仅机制（天花板≈$1/N$，`DEF-91-30`）；
2. $\mathrm{share}^{\rm cen}(q{=}W_q\bar c)$ —— 判别结构；
3. $\mathrm{share}^{\rm cen}(q{=}k_i)$ = **该层的完美寻址上限（ceiling）**；
4. $\rho=\mathrm{median}\big(\text{2}\big)/\mathrm{median}\big(\text{3}\big)$ —— **相对上限的达成度**。

$\rho\approx1$ ⇒ 实际寻址**已达该层几何允许的最好水平**（则失败在 $W_o$，与 A 相符）；
$\rho\ll1$ ⇒ 寻址有**可改进空间**（与 B 相符）。**两者只用于解释判决，不改判决。**

**不变的**：§1 主假说、§2.1 三臂、**§2.2 判决门 `0.40`/`0.25`/`0.125`**、§3.2、§5、§6 P1–P6 ——
**一字未改**。（§2.2 是唯一决定 verdict 的门，自 rev 2.0 冻结以来**从未改动**。）

**新增预测（先写死）**
- **P7**：$\mathrm{share}^{\rm cen}$ 的**上限**（$q{=}k_i$）逐层在 $0.05\sim0.15$ 量级
  （即键在该层几何下**本就不易分离**）；若实测 $\rho$ 接近 1，则"寻址已尽力"。
