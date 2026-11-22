# DIR-91 · 测量口径审计（`DIR-91-meta-A`）

**目的**：`DEF-91-16` / `DEF-91-18` / `DEF-91-22` 连续三次同类错误（**用未与当前协议对齐的量支撑结论**）⇒
不再逐个手改，改为**一次性审计全部已归档读数的口径**，并把不变量固化为 Stage A 守卫。

**为什么在本方向范围内**：这是**产出证据的装置的测量层验证**，服务于 `SGMN_v2.4` §7.2 既有开放行
（DIR-91 的证据行 `D-91` 由这些读数构成）。记入 `decisions/`。

---

## 一 · ★ 根因：`2.95` 是**冻结 `W`** 的读数（`DEF-91-23`）

装置 `sgmn_module.py:78`：

```python
W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
```

而 `R2_task91.make_stack()`（第 107 行）**恒为 `use_factorized=False`**；`W_of(m)` 返回 `m.Wfull`（正确）。

⇒ 在 dir91 的配置下，`U`、`V` **永不更新**（因子化分支从不执行），因此
**`m.W0 + m.U @ m.V.T` 是一个常量**。用它算 `rel`，读数**必然与学习无关**（全程平坦），且**≈ `W0` 基线**。

`probe91_storage.py` 正是这样：`Wb`（第 53 行，喂 `traj` 的逐次 rel）与 keymat 的 `W`（第 87 行，
喂 `keymat.rel_mean`）**都**用了这个常量表达式。

**⇒ 观测与预测完全吻合**：`traj` rel 在 600 次写入中恒为 $2.91\sim2.97$（**平坦**），
`keymat.rel_mean` $=2.95$，而 $W^0$ 基线为 $2.28\sim2.54$ —— 一个**略高于基线、不随学习变化**的常数。

**⇒ `DEF-91-21` 的 $\gamma$ 归因与「移动靶」机制全部作废**（已由 `probe91_gamma.py` 独立证伪：
$\gamma{=}0$ 与 $0.05$ 只差第三位小数）。**真正成因是冻结 `W`，与 $\gamma$ 无关。**

**⇒ 推论**：凡该轮从 `traj`/`keymat` 读出的「Hebb 更新近乎惰性」结论**作废**——
它不是写规则的性质，而是**测量读了不动的矩阵**。

---

## 二 · 受污染的读数（逐行核实）

**判据**：一行若把 `W0 + U@V.T` **无条件**用于读数（无 `if use_factorized else Wfull`），即受污染。

| 脚本:行 | 表达式 | 供给的读数 | 状态 |
|---|---|---|---|
| `probe91_storage.py:53` | `Wb` | `traj` 逐次 rel/cos（写入瞬间） | **污染** |
| `probe91_storage.py:87` | `W` | `keymat.rel_mean` $=2.95$ | **污染** |
| `probe91_gate.py:101` | `W` | `cos_Wk_v`/`cos_Wq_v`/`cos_Wq_eans` | **污染** |
| `probe91_align2.py:101` | `W` | 该轮对齐读数 | **污染** |
| `probe91_write.py:73` | `W` | 该轮写入诊断 | **污染** |
| `probe91_stub.py:103`, `:235` | `W` | 该轮桩读数 | **污染** |

**未污染（有 `if use_factorized else Wfull` 守卫）**：
`probe91_fulltie.py:33` · `probe91_identity.py:165` · `probe91_overfit.py:48,54,76,80,83,88` ·
`probe91_scale.py:33` · `probe91_tied.py:57,122`。

**未污染（用 `W_of` 或完全不碰 `W`）**：
`probe91_gamma.py` · `probe91_realpairs.py` · `probe91_decod2.py` ·
`probe91_lambda.py` · `probe91_lsfloor.py` · `probe91_conv.py` · `probe91_coexist.py` ·
`probe91_etafloor.py` · `probe91_spectral.py` · `probe91_premise.py` · `probe91_div.py` · `R2_task91.py`。

> **注**：`probe91_gate.py` 的解码门（第 123–127 行）只用 `st._hs_in` 与 `e_ans`，**不碰 `W`**
> ⇒ **该门的读数不受冻结 `W` 影响**（只受 $\gamma$ 影响，见 §四）。

---

## 三 · 聚合口径：`improve_ratio` 本身是跨层之比（`DEF-91-24`）

`R2_task91.storage_gamma0()` 原实现只报**每层中位数**，再取
`improve_ratio = median(基线层值) / median(写入层值)` —— **跨层中位数之比**。
这正是 `DEF-91-22`(A) 认定的聚合错误，且它**长在装置助手里**，故会反复诱发同一个错。

**处置（不删不改旧键，只新增）**：新增 `written_mean` / `baseline_mean` /
`ratio_per_layer` / `ratio_median_of_layer_ratios` / `ratio_min` / `ratio_max`；
`improve_ratio` **保留原样**（A91-17 依赖它，且已归档），但**不得再被引用为改善幅度**。

**正确读数**（逐层比值，两 seed）：
$14.26/13.33/8.80$（seed $20261028$）、$14.39/13.40/9.73$（seed $7777$）
⇒ 改善 **$8.8\sim14.4\times$**，**不是**单一 $12.44\times$。

---

## 四 · $\gamma$ 混淆已排除（`probe91_decod2.py`，三臂同一脚本）

| 臂 | `prompt_last` z（$L8/L12/L16$） | `answer_last` z |
|---|---|---|
| $\gamma{=}0$，真写（599 次） | $+0.01\ /{-}0.64\ /{+}0.14$ | $+15.28\ /{+}15.59\ /{+}14.69$ |
| $\gamma{=}0$，`no_write` | $\mathbf{+0.01\ /{-}0.64\ /{+}0.14}$ | $\mathbf{+15.28\ /{+}15.59\ /{+}14.69}$ |
| $\gamma{=}0.05$，真写 | $+0.01\ /{-}0.70\ /{+}0.30$ | $+15.28\ /{+}15.58\ /{+}14.55$ |

**两条硬结论**：
1. **守卫 PASS**：$\gamma{=}0$ 下「真写」与「不写」的隐态 **逐位相同**（$\max|\Delta|=\mathbf{0.000e{+}00}$）
   ⇒ 漂移恒为 0 的解析结论**端到端成立**（`DEF-91-20`(2)）。
2. **rev1.9 门 FAIL 的理由 (a) 在 $\gamma{=}0$ 下【成立】**：`prompt_last` 的 $z$ 为
   $+0.01/{-}0.64/{+}0.14$（均落在置换零分布内），而 `answer_last` 的 $z$ 高达 $+14.7\sim{+}15.6$
   ⇒ **答案在答案位可线性解出，在查询位不可**。该结论**稳健于 $\gamma$**。

⇒ **rev1.9 的门 FAIL 由「未验证」恢复为「成立」，且理由 (a) 现已在 $\gamma{=}0$ 范围内实测。**
（其理由 (b) 仍作废：来自污染读数。）

---

## 五 · 已固化为 Stage A 守卫（`R2_task91.stage_a()`）

| 守卫 | 内容 | 拦截的历史错误 |
|---|---|---|
| `A91-19` | `use_factorized` 全为 False，且 `W0+U@V.T` 与 `W_of` 相对差 $>10^{-6}$（可分辨） | `DEF-91-23` 冻结 `W`（$2.95$） |
| `A91-20` | `write_phase` 入口 `prev_hs is None`；写入次数 $==$ 块数$-1$（$599$） | 600 v.s. 599（`DEF-91-20`(3)） |
| `A91-21` | `storage_gamma0` 必须同时报 mean 与 median，且改善用**逐层比值** | `DEF-91-22`(A) 跨层之比 |

**仍为人工纪律（无法自动守卫）**：⑦ 跨脚本比较须在**同一脚本内**实现两臂。
**本轮两次采纳此纪律**（`probe91_gamma.py`、`probe91_decod2.py`），且**两次都靠它抓到了错**。

---

## 六 · 边界

- 本审计**只覆盖 `dir91/`** 的脚本；更早方向（`dir21`–`dir90`）的装置若同样用
  `use_factorized=True`，则不受此污染——**未逐一核实**，须在引用旧读数时单独查。
- 本审计**不改任何冻结判据的阈值与文字**；`improve_ratio` 旧口径保留。
- 600 块只承载 **32 个不同输入**，此事实与 $\cos\approx0.997$ 的条件数故事相关，外推须注意。

---

## 七 · 守卫实测值（`probe91_guards.py`，2026-11-19）

```
[A91-19] L8  use_factorized=False  rel_diff=8.5701e-01
[A91-19] L12 use_factorized=False  rel_diff=8.3878e-01
[A91-19] L16 use_factorized=False  rel_diff=8.6116e-01
[A91-20] pre_nonnull=[]  writes=[599, 599, 599]  expect=599  => PASS
[A91-21] per_layer_ratios={8: 14.0334, 12: 13.2450, 16: 8.5921}
[A91-21] median_of_ratios=13.245  min=8.592  max=14.033
[A91-21] OLD cross-layer improve_ratio=12.440
```

**要点**：
1. **两种 `W` 候选相差 86%**（`rel_diff` $0.84\sim0.86$）⇒ 冻结 `W` 的错**不是**细微偏差，
   而是**换了一个完全不同的矩阵**，因此它产生的读数（$2.95$）**毫无信息量**。这也解释了为何
   `traj` 能在 600 次写入中"稳定"到小数点后两位——它读的是一个常量。
2. `A91-20` 复算 PASS：入口 `prev_hs` 全为 `None`，写入次数 $599=$ 块数$-1$。
3. **`A91-21` 在装置内部复现了更正**：逐层比值 $14.033/13.245/8.592$，
   其中位数 $13.245$；**旧跨层口径 `improve_ratio`$=12.440$** —— 与 `DEF-91-22`(A) 的认定**逐位一致**。
   ⇒ 改善的**正确**幅度为 **$8.59\sim14.03\times$**（逐层，median 口径）；
   `12.44` 是跨层中位数之比，**不得再引用**。
4. `written` 中位数 $0.1628/0.1914/0.2771$ 与 `probe91_realpairs.py` **独立复现一致**。
