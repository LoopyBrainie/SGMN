# DIR-107 · rev **1.1**（修订 §4.2 的派生化）—— 其余一切照 rev 1.0

> **生效版本**：rev 1.1。**rev 1.0 全文留档**（`DIRECTION_rev107_v10.md`，不删改）。
> 本文件**只**修订 rev 1.0 的 **§4.2「实现（逐字派生）」** 一段，并登记实施期缺陷。
> ★★ **阈值、判据、支定义、守卫编号、格计划、边界 —— 一字未改。**
> 具体：`CHANCE=0.25`、`GATE_FIT=0.50`、`GATE_GEN=0.50`、`N_TRAIN_SET=(16,20,24,28)`、
> `N_SPLIT=4`、`SEEDS=(20261028,7777,60013)`、`M107-1`、`M107-2`、`F107-a`、`F107-b`、
> `C107-SAMPLE-LIMITED` / `C107-NOT-SAMPLE` / `C107-PARTIAL` / `INCOMPLETE`、
> `A107-1`…`A107-11`、`EXPECT_NQ_MAIN`、`EXPECT_CELLS`、§6 边界 —— **全部与 rev 1.0 完全相同**。

---

## §9 修订内容（rev 1.0 → rev 1.1）

### §9.1 rev 1.0 §4.2 的原文（**错误，已废**）

> - **键生成 / 句子 / `val_ids` / `prompt_ids`**：抄 `dir93/R1_task93.py:build_facts` 的同一段
>   （键生成与 `R2_task91` 的 rng 消耗**完全一致** ⇒ 键与 `D-121`/`D-136` **同键**）。

### §9.2 为何错（**Stage A 实测抓到，不是推测**）

rev 1.0 把 `dir93` 的**整段** `build_facts` 抄了过来，于是同时继承了
`dir93` 的**单一句子模板**（`"The access code for item {key} is {val}."`，
32 条事实**模板完全相同**），而 `dir106`/`R2_task91` 用的是 **V3 多样模板**
（`sentence_of(i, f)`：32 话题 × 6 动词 × 长注记）。

⇒ **这等于偷偷引入了第二个自变量（模板统一 vs 多样）**，违反
「只许改一个自变量」与本卡 §4.1「自变量（唯一）」的自我约束。

**实测证据（Stage A，`n=16`，`split=0`，`seed=20261028`）**

| 量 | rev 1.0 装置（dir93 模板） | 预期（dir106 装置） |
|---|---|---|
| `acc_train` | **0.4375** | `D-136` 的 `A106-2` 在 `n=16` 为 **0.9375** |
| `acc_item_train` | 0.5417 | — |
| `margin_train` | +2.529 | — |
| `acc_held` | **0.0625**（= `D-121` 锚，`A107-1` **PASS**） | 0.0625 |
| `key_maxcos`（白化后） | **1.75e-06** | ~1.99e-06 |

★★ **关键判读**：`key_maxcos` 只有 `1.75e-06` ⇒ **键不是共线的**，所以问题**不在**
键几何（`D-114` 的 `cos≈0.997` 那条解释在这里**不成立**，白化已把它消掉）。
真正塌掉的是 **`acc_train` 0.4375 < `GATE_FIT` 0.50** ⇒ **读出器连 16 条训练事实都没拟合好**。

⇒ ★★★ **若带着这个装置开正式轮，`acc_held = 0.0625` 将不可解释**：
它无法区分「没有泛化能力」与「**根本没拟合**」。这正是 `P107-0` 审计（`C-137`）
对本项目此前几轮的批评，我**不能**在修完那个之后再自己犯一次。

⇒ **`A107-10`（`acc_train ≥ GATE_FIT`）按设计阻断正式轮 —— 它起作用了，不是故障。**

### §9.3 修正后的 §4.2（**rev 1.1 生效**）

- **值分配** ← `dir93/R1_task93.py:build_facts` 的**类覆盖赋值那 4 行**（**只抄这 4 行**）：
  ```python
  _C = VALUE_POOL[:SPLIT_CLASSES]                      # SPLIT_CLASSES = 16
  vals = [_C[i % SPLIT_CLASSES] for i in range(N_FACTS)]
  assert len(set(vals)) == SPLIT_CLASSES
  ```
- **事实构造（键 / 句子 / `val_ids` / `prompt_ids`）** ← **`R2_task91.build_facts(tok)` 原文**
  （= `dir106` 的装置，**V3 多样模板**），**然后只把 `val` 换掉**并重算派生字段：
  ```python
  facts = R91.build_facts(tok)                 # ★ 键与 dir106 逐字同源（同 seed=9091）
  for i, f in enumerate(facts):
      f["val"]     = vals[i]                   # ★ 唯一改动
      f["sent"]    = R91.sentence_of(i, f)     # 用新 val 重算（sentence_of 按 i 确定、吃 f["val"]）
      f["prompt"]  = f["sent"].split(" " + f["val"] + ",")[0]
      assert f["sent"].startswith(f["prompt"] + " " + f["val"]), (f["sent"], f["prompt"])
      f["val_ids"]    = tok(" " + f["val"], add_special_tokens=False)["input_ids"]
      f["prompt_ids"] = tok(f["prompt"], add_special_tokens=False)["input_ids"]
  ```
- **块构造** ← `R2_task91.build_chunks`（不变，rev 1.0 已如此）。
- **测量层** ← 直接调用 `dir106/R1_task106.py:run_cell`（**不变**）。
- **顺序生成** ← `_make_order_class`（**不变**；类分层，`A107-2`/`A107-8` 已 PASS）。

★ 由此**自变量回到唯一**：`n_train`。值分配（类覆盖）是 `D-121` 的装置参数、不是本卡的变异；
模板/键/句子/块/测量层全部与 `dir106` **逐字同源**。

### §9.4 rev 1.1 的复验要求（**必须重跑，不得沿用 rev 1.0 的读数**）

- `A107-1`（`n=16` 复现 `D-121` 的 `acc_held = 0.0625`，容差 ±0.0625）—— rev 1.0 已 PASS，
  rev 1.1 **必须再 PASS**（换了模板 ⇒ 读数须重验）。
- **`A107-10`（`acc_train ≥ 0.50`）必须由 FAIL 转为 PASS** —— 这是 rev 1.1 的**唯一动机**。
  ★ 若换模板后 `A107-10` **仍** FAIL，则说明「拟合不足」来自**类覆盖本身**（而非模板），
  届时**不得**降低阈值 ⇒ 必须把「类覆盖下读出器无法拟合训练集」本身作为读数登记，
  并按 §6 边界另行裁定该轴是否可测（新开 rev）。
- 其余 `A107-2/3/4/5/6/7/8/9/11` 全部重跑。

---

## §10 实施期缺陷（**不追认**，全部留档）

| ID | 缺陷 | 定位/处置 |
|---|---|---|
| **`DEF-107-01`** | `dir106.run_cell` 的 `finally` 只在 **`dir106.ORIG_WRITE is not None`** 时还原 `LayerMem.write/read`；而这两个是 **`dir106` 的模块全局**，只在 `dir106.main()` 里赋值。⇒ **直接调用 `run_cell` 会永久留下 `LayerMem.write = D100._write_tf` 的 monkey-patch**，污染后续所有格（且**不报错**）。 | `run_one` 里先 `D106.ORIG_WRITE/ORIG_READ = LayerMem.write/read`，并在 `finally` 里比对前后对象，泄漏则强制还原；`A107-11` 断言已还原为 `ORIG_WRITE_LOCAL`。 |
| **`DEF-107-02`** | ★ **我的设计错误**（rev 1.0 §4.2）：把 `dir93` 的**整段** `build_facts` 抄来，**连单一句子模板一起继承** ⇒ 悄悄引入第二个自变量 ⇒ `acc_train` **0.4375** < `GATE_FIT` 0.50（`dir106` 同点 **0.9375**）⇒ `A107-10` FAIL。 | rev 1.1 §9.3：只抄**赋值那 4 行**，事实构造回归 `R2_task91.build_facts` + `sentence_of`。★ 元教训：**「逐字派生」指的是派生"被声明的那个装置层"，不是把上游脚本整段搬过来**；搬迁范围越大，越容易夹带未声明的自变量。 |
| **`DEF-107-03`** | `A107-5` 的详情串里有字面量 **`i%27`**，而该串走 `%` 格式化 ⇒ `ValueError: unsupported format character '+' (0x2b)`，Stage A 中途崩。 | 字面量改为 `i%%27`。★ 已在全文件扫过其余字面量（余下命中一处 docstring 与一处合法 `%-8s`）。 |
| **`DEF-107-04`** | 以脚本方式跑 `dir107/R1_task107.py` 时 `sys.path[0]` 是 `dir107/`，而 `sgmn_module.py` 在**项目根** ⇒ `ModuleNotFoundError: No module named 'sgmn_module'`。★ 顺带查明 **`dir106` 有同样脆弱性**：它的 `sys.path` 只插 `dir91`/`dir100`，而 `import sgmn_module` 在 `import R2_task91` **之前**（`R2_task91:29` 才插根）⇒ `dir106` 当年能跑是因为**外部 shell 设了 `PYTHONPATH`**。 | `dir107` 的 `sys.path` 循环加入 `""`（= 项目根），使脚本**自足**、不依赖外部 `PYTHONPATH`。★ `dir106` 未改（本卡不碰既有方向；如需修另立任务）。 |

---

## §11 边界（补充，与 rev 1.0 §6 并列，不替代）

1. rev 1.1 **只**改派生化与实现，**不改**任何冻结阈值与判据文字。
2. rev 1.0 的读数（`acc_train` 0.4375 那格）**留档但不作正式轮读数**：该装置已被 rev 1.1 取代。
3. 若 `A107-10` 在 rev 1.1 下**仍** FAIL ⇒ **不得**下调 `GATE_FIT`；按 §9.4 末段处置。
4. 本卡**不得**因 rev 1.1 而把 `§7.2-7b` 记 `已足够`。