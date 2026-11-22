# DIR-20 · 跨会话状态语义 —— **rev 1.1（冻结）**

> **卡片 rev 1.0（冻结，2026-11-06）** 是本方向的第一版冻结卡片（`DIRECTION.md`，md5 `e4b0ba36d3eb414ff54a2c5eb871a1ab`）。
> **本文件 rev 1.1（冻结，2026-11-06，正式轮开跑前）** —— **只做「协议可执行化」的澄清**：
> **判据的阈值与文字一字未改**；改的是「怎么把它算出来」。
> **装置**：`revolve/DIR-20/sgmn_module_hebb_v4.py`（v3 的逐字派生 + `export_state` / `import_state` / `state_fingerprint`），md5 `ad7c1eb621397a37c590b003576e6169`。
> **脚本**：`revolve/DIR-20/R1_xsession.py`（`revolve/DIR-19/R1_protect.py` 的逐字派生）。

---

## 0. rev 1.1 为什么必须存在（两条装置级实测约束）

### 0.1 ★ 约束一：`hebbian_full` 的单矩阵装不下 60 篇（时间视界）

写一步（v2.4 §3.2，装置逐字沿用）：

```
W ← W − η ( α (W k − v) kᵀ + λ (W − W⁰) )
```

⇒ ⇒ **每写一步，`(W − W⁰)` 乘 `(1 − ηλ)`**。本卡片的可用配置 `η = 1e−1`、`λ = 0.05`
⇒ ⇒ `(1 − ηλ) = 0.995` ⇒ ⇒ **内容的时间视界 ≈ 1/(ηλ) ≈ 200 步 ≈ 2.7 篇**。
⇒ ⇒ 一篇 ≈ 68 步（目标 34 + 干扰 34）⇒ ⇒ **60 篇 ≈ 4080 步 ⇒ 内容被衰减 `0.995^4080 = e^{−20} ≈ 2×10⁻⁹`**。

⇒ ⇒ ⇒ **⇒ 后果**：若按 rev 1.0 写「会话 A 写 60 篇 → 导出一个状态 → 会话 B 读 60 篇」，
⇒ ⇒ 则会话 B 读到的**内容几乎为零** ⇒ ⇒ **H1″ 会被协议假象判 FAIL** ⇒ ⇒ **这是协议缺陷，不是机制结论**。

⇒ ⇒ ⇒ **⇒ 修订 20-A（核心）**：会话 A 的第二阶段对**每一篇评测文档单独走一遍「写 → 导出状态」**
（写前 `restore_all(snap0)` ⇒ 该篇的状态与 DIR-19 的评测协议**逐字同构**）；
会话 B 对每篇**导入对应状态并只读读出**。
⇒ ⇒ ⇒ **⇒ 唯一变量仍然是「模型实例是否被重建 + 状态是否经磁盘搬运」** ⇒ ⇒ **判据文字不变**。
⇒ ⇒ ⇒ 统计量：60 篇 × 3 种子 = **n = 180** ⇒ ⇒ **与 DIR-19 的 n=180 口径可比**。

### 0.2 ★ 约束二：槽表回绕（`N_ENTRIES`）

`N_ENTRIES = 128`（沿用 DIR-15…19）≪ 60 篇 × 68 条 = 4080 条 ⇒ ⇒ 若整段会话导出一个状态，
⇒ ⇒ **只剩最后 ≈ 1.7 篇**的条目 ⇒ ⇒ 同样会让 H1″ 假 FAIL。
⇒ ⇒ ⇒ 修订 20-A 天然解决（每篇导出 ⇒ 每篇的表只有 68 条 ⇒ **不绕** ⇒ ⇒ `true_slot == true_idx`）。

### 0.3 ★ 约束三：`hebbian_full` 的**内容在 `W` 里，不在 `vals` 里**

读路径：`q = keys[addr_slot]` → `y = sg(W)·q` → `m = W_o·y`。
⇒ ⇒ **`core.vals` 只用于写保真度测量 `cos(W k_j, v_j)`，不参与读路径。**

⇒ ⇒ ⇒ **⇒ 后果**：若按 rev 1.0 在**导出状态上**置换 `vals` ⇒ ⇒ **对读出零影响**
⇒ ⇒ ⇒ H2″ 将**结构上永不可能 PASS**（与 DIR-19 的 D82/D83 同型 ⇒ ⇒ **纪律 #38**）。

⇒ ⇒ ⇒ **⇒ 修订 20-B**：`X4_content_wrong` 的置换施加在**会话 A 的写阶段**
（沿用 DIR-18 修订 68 / DIR-19 A19-10 **已验证**的 `content_wrong` derangement 钩子：
文档内**无不动点**地置换「值」的来源位置，**键 `k` 与签名 `sig` 逐位不动**）
⇒ ⇒ ⇒ `W` 编码**错内容**、`keys`/`sigs` **逐位不变** ⇒ ⇒ **A20-6 逐字成立**、
⇒ ⇒ ⇒ **A20-7（无不动点置换确实是排列）由该钩子保证** ⇒ ⇒ **判据阈值与文字一字未改**。

### 0.4 ★ 约束四：`linear` 键的 η 窗口（`X8_linear`）

DIR-18 的 D50/D51 实测：`linear` 键的 ‖k‖² ≈ **1.7×10⁴**（`ortho`/`renorm` 恒为 5）
⇒ ⇒ 稳定上界 `η < 2/(λ + α‖k‖²) ≈ 1.2×10⁻⁴` ⇒ ⇒ **`η = 1e−1` 会让写规则发散**
⇒ ⇒ ⇒ 「ortho vs linear」的差会被**发散**污染 ⇒ ⇒ **测不到「键尺度」**。

⇒ ⇒ ⇒ **⇒ 修订 20-C**：`X8_linear` 的 **η = 1e−5**（线性键的尺度适配档：稳定但惰性）
⇒ ⇒ ⇒ ⇒ 于是 H4″ 读的是「**尺度归一 vs 原尺度（惰性写）**」
⇒ ⇒ ⇒ ⇒ **⇒ H4″ 即使 PASS，也只能读成「原尺度下写规则不起作用 ⇒ 内容取不回」**，
⇒ ⇒ ⇒ ⇒ **⇒ 不得读成「跨会话时尺度失稳」** ⇒ ⇒ ⇒ ⇒ **⇒ 且卡片 §4.1 的写死条款照抄执行**：
⇒ ⇒ ⇒ ⇒ **H4″ FAIL ⇒ 不得把 Freeze B 的「键必须逐写归一化」扩展到跨会话语义**。

---

## 1. ★ 冻结的两段式协议（rev 1.1 的确切形式）

```
【会话 A】(模型实例 A；每臂每种子一个)
  0. 建装置（key_mode = 该臂的写配置；η = 该臂的 η；mem_kind = 该臂的核）
  1. 训练阶段：对 200 篇**训练**文档逐篇做
        restore_all(snap0) → write_both(目标+干扰) → freeze_write → 读 → 损失 → backward → step
     ★ 与 DIR-19 的 N1_frozen_base **逐字同口径**（只训读侧 W_o/γ；Adam(lr=1e-3)；基座全冻结）
  2. 写阶段：对 60 篇**评测**文档逐篇做
        restore_all(snap0) → write_both(目标+干扰) → 记录真值槽 → export_state(...) → torch.save
     ★ 写入前的 restore_all ⇒ 该篇的 W 与地址表 = 「干净底 + 该篇的写」⇒ 与 DIR-19 评测协议同构
     ★ 导出文件：.ops/state/DIR20_{arm}_{seed}_{doc_idx}.pt  ⇒ 记录 **文件 md5** + **state_fingerprint**
  3. 释放装置（会话 A 的模型实例按臂的会话模式决定是否保留）

【会话 B】
  · 同会话臂（X0 / X10）：**不重建实例、不导入**
      X10_same_session：在会话 A 写完该篇后**立刻只读读出**（不导出、不重建）★ 同会话基线
      X0_no_state     ：在会话 A 用**一个空装置**（全新栈、无任何状态）只读读出 ★ 退化下界
  · 跨会话臂（X1…X8 / X11）：**重建模型实例**（from_pretrained）⇒ 建栈 ⇒ import_state(该篇状态) ⇒ **只读读出**
      X11_newinst_noexport：**不导入**（全新实例 + 空栈）★ 隔离「新实例」与「状态搬运」
```

⇒ ⇒ **唯一变量**：会话 B 的模型实例是否被重建 + 状态是否经磁盘搬运（`import_state`）。
⇒ ⇒ **只读评测的定义**：只前向、**不写、不训练** ⇒ ⇒ `core.count` 与 `write_calls` 必须不变（Stage A A20-9 断言）。

### 1.1 只读评测的指标（全部可从导入状态算出）

| 指标 | 定义 | 与 DIR-19 的关系 |
|---|---|---|
| `nll` | 码 token 的 NLL（读阶段最后位置） | 同 |
| `nll_dcode` | 干扰码的 NLL | 同 |
| `nll_ctrl` | 交换性对照 token 的平均 NLL | 同 |
| `hit` | `argmax_j cos(core.vals[j], y) == 真值槽` | 同（表内只有该篇的 68 条 ⇒ 语义等同） |
| `addr` | `last_cue_idx == 真值写入序` | 本协议下 **恒等于** `addr_slot`（构造使然 ⇒ Stage A 断言） |
| `addr_slot` | `last_cue_idx == 真值槽` | 同 |
| `fid_true` | `core.fidelity(真值槽)` = cos(W k, v) | 同 |
| `fid` | `core.fidelity(last_cue_idx)` | 同 |
| `cos_y_vtrue` | `cos(core.vals[真值槽], y)` | 同 |

---

## 2. ★ 臂表（11 臂，rev 1.1 冻结）

字段：`(name, mem_kind, eta, key_mode, variant, export_mode, session_mode)`
`export_mode`：`full` | `no_readmap` | `no_addr`
`session_mode`：`cross`（重建实例 + 导入） | `same`（同实例，读完即测） | `same_empty`（同实例 + 空栈） | `cross_noimport`（重建实例 + 不导入）

| # | 臂 | `mem_kind` | `η` | `key_mode` | `variant` | `export_mode` | `session_mode` | 作用 |
|---|---|---|---|---|---|---|---|---|
| 1 | `X0_no_state` | hebbian_full | 1e-1 | ortho | normal | — | **same_empty** | 退化下界（同实例、无记忆） |
| 2 | `X1_full_state` | hebbian_full | 1e-1 | ortho | normal | **full** | **cross** | ★ **主臂** |
| 3 | `X2_no_readmap` | hebbian_full | 1e-1 | ortho | normal | **no_readmap** | cross | 读出映射必须搬运吗 |
| 4 | `X3_no_addr` | hebbian_full | 1e-1 | ortho | normal | **no_addr** | cross | 地址表必须搬运吗 |
| 5 | `X4_content_wrong` | hebbian_full | 1e-1 | ortho | **content_wrong** | full | cross | ★ 内容特异对照 |
| 6 | `X5_wrong_session` | hebbian_full | 1e-1 | ortho | normal | full | cross（**导入另一 seed 的状态**） | ★ 会话归属对照 |
| 7 | `X6_verbatim` | **explicit_kv** | — | linear | normal | full | cross | ★ 零效度对照（按构造内容正确） |
| 8 | `X7_renorm` | hebbian_full | 1e-1 | **renorm** | normal | full | cross | 键尺度归一（只修尺度） |
| 9 | `X8_linear` | hebbian_full | **1e-5** | **linear** | normal | full | cross | 原尺度（惰性写）对照 |
| 10 | `X10_same_session` | hebbian_full | 1e-1 | ortho | normal | — | **same** | ★ 同会话基线（跨会话折损的参照） |
| 11 | `X11_newinst_noexport` | hebbian_full | 1e-1 | ortho | normal | — | **cross_noimport** | 隔离「新实例」与「状态搬运」 |

⇒ ⇒ **`X6_verbatim` 的 `explicit_kv`**：其读为 `y = softmax(β·cos(q,K))·V` ⇒ ⇒ **内容在地址表的 `vals` 里**
⇒ ⇒ ⇒ **按构造内容正确** ⇒ ⇒ 作为「零效度门槛」对照 ✔
⇒ ⇒ ⇒ **注意**：`explicit_kv` 的读**对键尺度不敏感**（余弦）⇒ ⇒ ⇒ **H4″ 只在 `hebbian_full` 臂系上读**（`X1`/`X7`/`X8`）
⇒ ⇒ ⇒ ⇒ ⇒ ⇒ **`X6` 不参与 H4″**（写死）。

⇒ ⇒ **`X5_wrong_session` 的实现**：种子 `s` 导入**另一个种子 `s'`** 的同编号状态文件。
⇒ ⇒ ⇒ 为此，主循环对每个臂**先把三个种子的会话 A 全部跑完并落盘状态**，再进入会话 B ⇒ ⇒ ⇒
⇒ ⇒ ⇒ ⇒ **`X5` 的会话 B 读的是 `s' = 下一个种子` 的状态文件**（最后一个种子用第一个种子的）⇒ ⇒ **循环错位 ⇒ 会话归属被破坏**。

---

## 3. ★ 冻结判据（7 条，**阈值与文字一字未改**；只写死「读什么」）

> 口径（写死，防漂移）：**NLL 口径，越低越好**；
> `Δ(Δnll)(A,B) = mean( nll_B − nll_A )`（>0 ⇒ A 更好）⇒ ⇒ 与 DIR-19 的 `paired()` 同义；
> `p` 用 `paired()` 的**符号检验**（n=180）；「三种子同向」= 逐种子的均值差**全部 > 0**。

| 编号 | 判据 | 阈值 | **落点（rev 1.1 写死）** |
|---|---|---|---|
| **H1″** | 跨会话可用 | Δ(Δnll) > **0.5 nats**，p < 0.05，三种子同向 | `X1_full_state` vs **`X11_newinst_noexport`**（同一「新实例」，唯一差别 = 有没有状态） |
| **H2″** | 内容特异（记忆内） | Δ(Δnll) > **0.5 nats**，p < 0.05，三种子同向 | `X1_full_state` vs **`X4_content_wrong`**（只差内容；两臂地址逐位相同 ⇒ 天然无零效度问题） |
| **H3″** | 跨会话折损有界 | 折损 ≤ **1.0 nats** | 折损 = `benefit(X10_same_session) − benefit(X1_full_state)`；`benefit(arm) = NLL(X11) − NLL(arm)` |
| **H4″** | 键尺度归一 ⇒ 跨会话寻址稳定 | Δ(Δnll) > **0.5 nats**，p < 0.05，三种子同向 | **`X1_full_state`(ortho) vs `X8_linear`**（`X7_renorm` vs `X8` 作为**报告项 R-4**）⇒ ⇒ **★ 只在 `hebbian_full` 臂系上读**（`X6` 不参与）⇒ ⇒ **★ PASS 的读法见 §4.3** |
| **H5″** | 会话归属 | Δ(Δnll) > **0.5 nats**，p < 0.05，三种子同向 | `X1_full_state` vs **`X5_wrong_session`** |
| **H6″** | 零效度门槛 | Δ(Δnll) > **0.5 nats**，p < 0.05 | **`X6_verbatim`** 相对 `X11_newinst_noexport` 的跨会话收益 > 0 |
| **H7″** | 状态成分的必要性 | 至少一条：Δ(Δnll) > **0.5 nats**（缺失臂相对主臂**更差**） | `X1_full_state` vs `X2_no_readmap` / `X3_no_addr`，**至少一条**成立 |

### 3.1 ★ 报告项（不进入 PASS/FAIL）

| 编号 | 内容 | 目的 |
|---|---|---|
| R-1 | `X1_full_state` 的 `hit` / `addr_slot` / `fid_true` / `cos_y_vtrue`（跨会话后） | 寻址是否幸存、是否取回内容 |
| R-2 | `X8_linear` 的 `hit` / `addr_slot` / `fid_true` | 与 `X1` 对比 ⇒ 原尺度下的直接证据 |
| R-3 | `X11_newinst_noexport` 相对 `X0_no_state` 的差 | 「新实例」与「状态搬运」两个变量的分解（基座冻结 ⇒ 预期 ≈ 0） |
| R-4 | `X7_renorm` vs `X8_linear` 的差 | H4″ 的第二个归一配置（`renorm`）的读数 |
| R-5 | `X1` 与 `X10` 的 `hit` / `addr_slot` 逐种子读数 | 跨会话的折损落在哪个环节（寻址 or 内容） |

### 3.2 ★ 证伪条件（rev 1.0 §4.1 **照抄执行**，一字不改）

- **H1″ FAIL** ⇒ 「状态**不可**跨会话搬运」⇒ ⇒ §7.2 的「跨会话状态语义」写成「**不可行**」。
- **H2″ FAIL** ⇒ 跨会话后**内容不再驱动输出** ⇒ ⇒ 该子项降级为「跨会话的**通用能力提升**」，不是记忆。
- **H5″ FAIL** ⇒ 收益**不来自**本会话写入的信息 ⇒ ⇒ 「搬的是模型，不是记忆」⇒ ⇒ 该子项不成立。
- **H4″ FAIL** ⇒ 键尺度归一对跨会话**没有**额外作用 ⇒ ⇒ **不得**把 Freeze B 的「键必须逐写归一化」**扩展**到跨会话语义。
- **H3″ FAIL** ⇒ 跨会话的折损**超过事先声明的界** ⇒ ⇒ 「跨会话」在任务层**不可接受** ⇒ ⇒ 只能写成「有代价」。

### 3.3 ★ 主假说四子句与判据的对应（rev 1.0 §1 照抄）

| 子句 | 判据 |
|---|---|
| (a) 状态可跨会话搬运并在全新实例上产生收益 | **H1″** |
| (b) 收益仍由记忆内容驱动 | **H2″**（+ H6″ 作零效度门槛） |
| (c) 收益折损有界且事先声明 | **H3″**（界 = 1.0 nats，开跑前冻结） |
| (d) 可用性依赖具体机制成分 | **H7″**（+ H5″ 作会话归属） |

⇒ ⇒ **裁决规则**：四子句全部 ✔ ⇒ **COMPLETE / SUCCESS**；任一 ✘ ⇒ 按 §3.2 的对应条款降级，
⇒ ⇒ **主假说被证伪 ⇒ 不计入 SUCCEEDED**（沿用 DIR-11…14 的计数纪律）。

---

## 4. ★ Stage A（装置校验门，开跑前必须全 PASS）

| # | 检查 | 判据 |
|---|---|---|
| **A20-1** | `export_state` → `import_state` 往返一致 | `state_fingerprint` **逐位相同** |
| **A20-2** | 往返后读出逐位相同 | 同一前缀下新旧装置的读出向量余弦 ≥ **0.999** |
| **A20-3** | `import_state(strict=True)` 的形状校验生效 | 喂形状不符的状态 ⇒ **必须抛错** |
| **A20-4** | `include_read_map=False` ⇒ `W_o`/`γ` 未搬运 | 导入后 `W_o` 仍为随机初值（与导出前**不等**） |
| **A20-5** | `include_addr=False` ⇒ 地址表未搬运 | 导入后 `core.count == 0`、`keys` 全零 |
| **A20-6** | 内容置换只动**值**、不动**键与签名** | `X1` 与 `X4` 的导出状态：`keys`/`sigs` **逐位相同**、`vals` **逐位不同** |
| **A20-7** | 无不动点置换**确实是排列**（DIR-18 D62 的教训） | `content_wrong` 钩子的置换：索引集合 = 原集合，且**无不动点** |
| **A20-8** | 会话 B 的模型实例**确实是新的** | 会话 B 的 `model` 对象 id **≠** 会话 A 的；且会话 A 的模型已释放（显存回落） |
| **A20-9** | 只读评测**不写** | 评测前后 `core.count` **不变**、`write_calls` **不变** |
| **A20-10** | 继承 DIR-19 的 Stage A 全部 30 项 | **逐项 PASS**（不重写） |
| **A20-11** | ★★ `hebbian_full` 的 **`vals` 不进入读路径**（修订 20-B 的装置级依据） | 把 `core.vals` 全部置零 ⇒ **读出逐位不变**；把 `core.W` 置零 ⇒ 读出**改变** |
| **A20-12** | ★ 本协议下 `addr == addr_slot`（构造使然） | 每篇的 `true_slot == true_idx`（表在写前清零 ⇒ 不回绕） |
| **A20-13** | ★ `X8_linear` 的键尺度实测 ≫ 5（复现 D50/D51） | `linear` 的实测 ‖k‖² **> 100**（`ortho`/`renorm` ≈ 5） |
| **A20-14** | ★ 状态文件往返：落盘后 `torch.load` 的指纹与内存中的指纹相同 | `state_fingerprint(load(file)) == state_fingerprint(in_memory)` |

⇒ ⇒ **纪律 ①**：Stage A 任一 FAIL ⇒ **不得开主循环**（脚本内部硬闸 + 启动脚本双保险）。
⇒ ⇒ **纪律 ㉞ / #37**：增量与状态产物按「**方向 + 臂 + 种子**」命名空间隔离
（`formal_incremental_DIR20_{smoke,full}.json` / `.ops/state/DIR20_{arm}_{seed}_{doc}.pt` / `R20_results.json`）。

---

## 5. ★ 执行环境与规格

- **服务器**：`ec2-3-27-212-181.ap-southeast-2.compute.amazonaws.com`（Tesla T4 15 GB），用户 **`ec2-user`**，密钥 `/workspace/.ssh/SGMN.pem`。
- **模型**：`Qwen/Qwen2.5-0.5B-Instruct`（fp32）。
- **语料**：`~/sgmn-loop/corpus/{train,valid}.txt`（md5 `34a77ebe59551b3a3b3d8c0a2797c976` / `d756d22ab2310e7e60c094bea5c5a8eb`）。
- **正式轮规格**：**200 训练篇 / 60 评测篇 / 3 种子** ⇒ **11 臂 × 3 种子 = 33 组合** ⇒ **每臂 n = 180**。
- **smoke 规格**：40 / 12 / 2 种子。
- **状态文件**：`.ops/state/DIR20_{arm}_{seed}_{doc}.pt`（导入后即删 ⇒ 磁盘占用可控 ⇒ **md5 与指纹留档**）。
- **磁盘**：服务器余量有限 ⇒ ⇒ **状态文件在成功导入后立即删除** ⇒ ⇒ 只保留 md5 与指纹记录。

---

## 6. ★ 证据边界（rev 1.1 追加）

1. 本方向测的是「**状态经磁盘搬运 + 模型实例重建**后是否仍可用」，**不是**「无限长的会话记忆」。
2. 本轮的 `hebbian_full` 臂在**单篇粒度**上做跨会话比较（修订 20-A）⇒ ⇒
   **不能外推为「一个会话里能跨会话记住 60 篇」**（那受 §0.1 的时间视界限制）。
3. `X8_linear` 的 η 经过**适配**（修订 20-C）⇒ ⇒ **H4″ 的结论只覆盖「归一 vs 原尺度（惰性写）」**，
   **不能外推为「跨会话时尺度失稳的定量刻画」**。
4. `X6_verbatim` 的 `explicit_kv` 核**对键尺度不敏感** ⇒ ⇒ **不能**用它支持任何关于键尺度的结论。
