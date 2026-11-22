# DIR-52 · VERDICT —— **`EARLY-LOCALIZED`**（`H52` 被否证）

> 方向：框架 §7.2「稳定性推广 · **非 iid 与 AR(1) 修正**」；直接后续 = D-67。
> 卡片：`revolve/DIR-52/DIRECTION_rev52_v10.md`（rev 1.0 冻结）· `v11/v12/v13.md`（**只订正成本估算、n、功效代数与精度勘误，阈值与判据文字不动**）。
> 读数**唯一**来源：`dir52/out/DIR52r12_adjudication.json`（由 `dir52/adjudicate52.py` 生成）；机制分解：`dir52/out/DIR52r12_mechanism.json`（**非判据**）。

---

## §1 卡片与装置（抄自卡）

| 项 | 内容 |
|---|---|
| 主判据量 | `SHAPE = ΔG(2) − ΔG(4)` 的种子均值；`ΔG(k) = G_k(写入) − G_k(无写入对照)`，`k∈{2,4}` 为 AR(1) 轮数 |
| 冻结值 | `δ* = 0.000430` · P4 闸门 `0.000861` · `n = 200` · 种子段 `62000..62199` · `min_n_eff = 160` · 剔除率上限 `0.20`（`out/DIR52r12_adjudication.json` → `frozen`） |
| 装置 | 逐字派生 D-67 的 `R1_dose_ar1_v3.run_unit_dose`；`N_KV = 60`；AR(1) 流 `R1_ar1.build_stream_ar1`（`torch.Generator().manual_seed(seed+1)`）；`c_tau` 由 `K12.cal_at(N_KV)` 现场标定 |
| 实现 | `dir52/{R1_task52.py, stageA52.py, adjudicate52.py, mechanism52.py, ready52.py}` |

## §2 主判据量与冻结值

- **`P1_late_localized`** ⇔ `mean(SHAPE) < −max(δ*, 3se)`；**`P2_early_localized`** ⇔ `mean(SHAPE) > +max(δ*, 3se)`。
- **`P3_valid`** ⇔ `indet ≤ 0.20` 且 `n_eff ≥ 160`；**`P4_resolvable`** ⇔ `3se ≤ 0.000861`。
- 分支序 **①INVALID → ②INFEASIBLE-POWER → ③PASS → ④EARLY-LOCALIZED → ⑤NOT-LOCALIZED → ⑥INCONCLUSIVE**，不得改序。

## §3 正式轮读数（全部取自 `DIR52r12_adjudication.json`）

| 字段 | JSON 路径 | 值 |
|---|---|---|
| 行数 / 期望 | `n_rows` / `n_expected` | **196** / 200 |
| 有效 `n_eff` | `n_eff` | **196** |
| 剔除率（冻结口径） | `indeterminate_frac` | **0.0000** |
| 缺失行 / 按期望计剔除率 | `n_missing_vs_expected` / `indet_vs_expected` | **4** / **0.0200**（纯报告；P3 用冻结口径） |
| 无效原因 | `invalid_reasons` | `{"nonfinite": 0, "identity": 0}` |
| `SHAPE` 均值 / sd / se / 3se | `SHAPE.mean/.sd/.se`，`SHAPE.threshold` | **+0.004057** / **0.003421** / **0.000244** / **0.000733** |
| `max(δ*,3se)` 阈值 | `SHAPE.threshold` | **0.000733**（= 3se > δ*） |
| `SHAPE` 正/负计数 | `SHAPE.n_pos` / `.n_neg` | **175 / 21** |
| `EARLY` / `LATE` / `TOTAL` 均值 | 同名键 `.mean` | **+0.002644** / **−0.001413** / **+0.001231** |
| `TOTAL` 与 D-67 交叉核对 | `D67_cross_check` | 本轮 `TOTAL = +0.001231` vs D-67 `dG = 0.000860862`（**只作一致性报告，不改 D-67**） |
| 一致性 | `derive_checked` / `derive_mismatch_detail` | **196** / `{}`（逐行重算 `SHAPE` 与行内字段一致） |
| 完整性 | `integrity` | 196 行全部可解析 · 无重复种子 · 无越界 · `n_lines_ge_expected=False` |

## §4 判定表（按序，分支由代码决定）

| 序 | 谓词 | 读数 | 过？ |
|---|---|---|---|
| ① | `indet > 0.20` 或 `n_eff < 160` | `0.0000`；`196` | 否（`P3_valid=True`） |
| ② | `3se > 0.000861` | `0.000733` | 否（`P4_resolvable=True`） |
| ③ | `mean(SHAPE) < −0.000733` | `+0.004057` | 否 |
| ④ | `mean(SHAPE) > +0.000733` | `+0.004057 > +0.000733` | **是** ⇒ **`EARLY-LOCALIZED`** |

`decision.outcome = "EARLY-LOCALIZED"`，`decision.P2_early_localized = true`（`DIR52r12_adjudication.json` → `decision`）。

## §5 机制（`DIR52r12_mechanism.json`，**非判据**）

- **分解到两臂**：`F` 臂 **+0.005126**（`3se=0.000493`，`t=+31.19`）主导；`K` 臂 **−0.001069**（`3se=0.000916`，`t=−3.50`）微负。
  分解残差 `max|F+K−SHAPE| = 0.000e+00`（逐行恒等，无残差）。
- **逐层量**：`ΔF(2) = −0.004194`、`ΔF(4) = −0.003262`、`ΔK(2) = −0.001549`、`ΔK(4) = −0.002030`；
  ⇒ `EARLY = ΔG(2) = +0.002644`（`t=+23.40`）、`LATE = ΔG(4)−ΔG(2) = −0.001413`（`t=−8.83`）。
- **读法**：`SHAPE > 0` 的**全部来源**是 `F` 臂在**早段（2 轮）**的收益更大；`K` 臂在晚段反而略差。
  探索性相关（不作判据）：`corr(keepK, SHAPE) = +0.120`、`corr(keepF, SHAPE) = +0.083`；保留量 `keepF2=373.3 / keepF4=556.5 / keepK2=183.1 / keepK4=556.5`。

## §6 固定措辞（逐字）

**④ `EARLY-LOCALIZED`**：

> 「`H52` **被否证**：收益**集中在早段轮次**（`SHAPE = +0.004057`，正向显著）。**不计入** `SUCCEEDED`。」

读数支撑（引号外）：`3se = 0.000733` · `t = +16.61` · 正/负 **175/21**（`DIR52r12_adjudication.json` → `SHAPE`）。

★ **不得**写"反向成立""D-67 错了"——D-67 的**总效应**（`+0.000861`）未被质疑；被否证的只是**它的口头机制解释**（"收益在晚段"）。

## §7 边界（不得外推）

1. **单装置、单参数组**：`N_KV=60`、`a = 0.776` 实测相关结构、`δ*`/闸门为 `n=200` 冻结值；换 `n_kv`、换 `a`、换轮数 `{2,4}` 均**未测**。
2. **`SHAPE` 是"早/晚段对比量"，不是"总收益"**：`TOTAL = +0.001231` 才是总效应量级；不得用 `SHAPE` 的符号替代总效应的符号。
3. **不得外推到完整 LLM / 真实 token 流**：本装置是 AR(1) 修正的**玩具动力学**，`k_t` 为合成 AR(1) 流。
4. **196/200**：缺的 4 行是**装置事故**（`DEF-52-5`，见 §9），不是删失、不是窗口外剔除；`indet_vs_expected = 0.02` **不参与**任何分支。
5. **行级不可跨进程逐位复现**（`DEF-52-6`）：行长量相对差 ~1e-6 量级，**≪** 判据尺度（`3se 0.000733` 的 1/700）；故对分支无影响，但**不得**声称"逐位复现"。

## §8 证明义务与硬门

- **Stage A 10/10 全过**：`dir52/out/DIR52_stageA.json` → `stage_a_pass = true`（`A52-1..A52-10`）。
- **就绪门**：`ready52.py` 输出 **`DONE-BUT-SHORT: 缺 4 个种子`**、`n_eff 上限=196`（门槛 160），退出码 0 ⇒ 走预注册短缺口路径，**非** `NOT-READY`。
- **裁决器零魔数**：阈值全部命令行传入（`--delta-star 0.000430 --power-gate 0.000861 --n-expected 200 --max-indet-frac 0.20 --min-n-eff 160 --seed-segment 62000 62199`）。

## §9 缺陷与纪律（本轮新增，全部留档）

- **`DEF-52-5`（装置事故，已抢救）**：Cycle 84 的一条本地→远程 `rsync -a dir52/ … trainer: --relative` 把**本地陈旧镜像**覆盖到权威输出 ⇒ 远程 shard 路径换成新 inode，4 个写手继续向**旧（未链接）inode** 追加。
  **数据未丢**：`watch_fd_snapshot.py` 每 5 s 从 `/proc/<pid>/fd` 快照（9 个快照）；证 **毒化文件 = 恢复数据的逐字节前缀（4/4 `True`）**。
  恢复：`restore85.py --allow-short` 把 **196/200** 放回并留档毒化文件于 `out/_POISONED_by_rsync/`；记录 `out/RESTORE_RECORD.json`（`shortfall=[62049,62099,62149,62199]`、`sha256_16`）。
- **`DEF-52-6`（行级不可逐位复现）**：同种子在**新进程**重算，与原写手行**逐字段不等**（`EARLY/LATE/SHAPE/TOTAL/dG/phi`），相对差 `phi 3.9e-7`、`TOTAL 2.1e-6`；但**两次同条件新进程 A==B 逐字段全等**。
  ⇒ 判定为"**同条件可复现，跨进程历史/并发不可逐位复现**"，成因**未验证**（RNG 全部 `manual_seed(seed+…)`，无未播种抽样；`dmesg` 不可读）。
  **后果**：末 4 行**不可补算**（补算门 `ctrl85.py` 判 `FAIL`）⇒ 本轮按 196 行定论并**如实披露**。
- **`DEF-52-7`（守护盲区，设计性）**：轮询式 fd 快照对"写末行后立即退出"**结构性失效**（末行窗口毫秒级）⇒ 4 片**各缺最后 1 个种子**完全符合该设计预期，**不是**运气问题。
- **纪律候选 121**：**补算必须有"复现对照门"**——任何"补跑丢失数据"的动作，先用**已在库的种子**重算并**逐字段比对**；`FAIL` 即禁止并入（本轮正是靠它避免了混入异源行）。
- **纪律候选 122**：**结果目录只出不进** —— 本地镜像永不回升（`out/` 单向）；推送只走 `.ops/scripts/push52.sh`（拒绝目录、拒绝 `out/`，自测 3 拒 1 放行）。
- **诚实声明**：本轮曾见 `seed 62000` 行的 `phi/mu/F/K` **原始数值**（为核实 21 行文件是否真数据）；**未见**任何判据量分布；`adjudicate52.py` 的输出是本轮**首次**判据计算。

## §10 台账

- 证据行：`D-75`（`evidence/table.md`，追加在表末，不改旧行）。
- decisions：`decisions/DECISIONS.md` **决定八十七**（事故与抢救）· **八十八**（补算与合并门）· **八十九**（定论与披露）。
- 打包与一致性：`rebuild_full.sh` / `repack.sh` / `.ops/scripts/audit_consistency.py`（见 task.md 下一步）。