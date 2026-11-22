# DIR-20 · 跨会话状态语义 —— **rev 1.2（冻结，正式轮开跑前）**

> **卡片 rev 1.1（冻结，2026-11-06）** 见 `../revolve/DIR-20/DIRECTION_rev11.md`（17 896 B）。
> **本文件 rev 1.2（冻结，2026-11-07，正式轮开跑前）** —— ★ **只做「协议可执行化」**：
> **判据 H1″…H7″ 与报告项 R-1…R-5 的阈值与文字一字未改**；改的只是「怎么把它算出来」。
>
> **装置**：`../sgmn_module_hebb_v4.py`（v3 的逐字派生 + `export_state` / `import_state` / `state_fingerprint`），
> md5 `ad7c1eb621397a37c590b003576e6169`（与 rev 1.1 一致，未改）。
> **脚本**：`~/sgmn-loop/dir20/R1_xsession.py`（`revolve/DIR-19/R1_protect.py` 的逐字派生），
> md5 `00f6d1b17bcf44842a28c5199c77c8d8`。

---

## 0. ★ 本 rev 的五处改动（逐条给依据；**没有一处改判据**）

### 0.1 缺陷 **D84** —— 只读评测**必须冻结写**（实现缺陷；判据不动）

- **现象**：交接包里的 `eval_arm_readonly` 只前向、**未显式冻结写** ⇒ 前缀的每个位置都会触发写钩子 ⇒
  ① `core.count` / `write_calls` 必增 ⇒ Stage A 的 **A20-9 必 FAIL**；
  ② 读出 `last_yr` 会混进「**评测自己写的条目**」⇒ 指标被污染 ⇒ 跨会话比较失真。
- **依据**：DIR-19 的 `eval_arm` **逐字同款**（写阶段结束后 `st.freeze_write()` 才前向读出）。
- **修法**：新增**外层包装函数** `eval_arm_readonly` ⇒ 读前 `freeze_write()`、读后**按原样恢复**调用方的冻结状态；
  读出逻辑本体（改名 `_eval_arm_readonly_body`）**一字未改**。
- **对判据的影响**：**无**。H1″…H7″ 与 R-1…R-5 的阈值与文字未变；只是让「只读」名副其实。

### 0.2 缺陷 **D85** —— 真值槽必须**跨会话传递**（实现缺陷；判据不动）

- **现象**：会话 B **重建**了 `ev` 文档（`build_docs` 重新构造 dict）⇒ 重建出的文档**没有** `true_slot`
  ⇒ `eval_arm_readonly` 读 `d["true_slot"]` ⇒ **KeyError** ⇒ 会话 B 整段不可执行。
- **且**：`X0_no_state` / `X11_newinst_noexport` 的 `export_mode="none"` ⇒ **不导出状态**
  ⇒ 会话 A 若只在导出时记录真值槽，这两个臂就没有真值槽（而它们同样要读出）。
- **修法**：① 会话 A 对**所有**臂记录 `meta[i]["true_slot"]`（**精确槽号映射**，D54 口径）；
  ② 会话 B 从 `_smeta[(arm, seed)]` 取回并写进重建的文档 ⇒ 会话归属对照臂 `X5_wrong_session` 也用同一映射。
- **对判据的影响**：**无**。

### 0.3 ★ **A20-12 的表述按事实订正**（Stage A 检查项；**判据不动**）

- **rev 1.1 的原文**：A20-12 = 「每篇的 `true_slot == true_idx`（表在写前清零 ⇒ 不回绕）」；
  §1.1 还写「`addr` 恒等于 `addr_slot`（构造使然 ⇒ Stage A 断言）」。
- **事实**：本协议（**与 DIR-18/19 同构**）在写阶段前执行 `restore_all(snap1)` ⇒ 地址表带**训练残留**
  ⇒ 真值槽 = **写前 count + true_idx**（**不是** `true_idx`）。
- **实测依据**（DIR-19 正式轮，从实验卷恢复的 `.ops/formal_incremental_DIR19_full.json`，seed 20261028）：

  | 臂 | 训练 | `nll` | `hit` | `addr`（写入序口径） | `addr_slot`（槽空间口径） | `n_entries` |
  |---|---|---|---|---|---|---|
  | `N1_frozen_base` | 有 | 9.8724 | 0.9111 | **0.0000** | **0.9111** | 68 |
  | `N1_content_wrong` | 有 | 11.6235 | 0.9889 | **0.0000** | **0.9889** | 68 |
  | `N1_linear_ctrl` | 有 | 11.1072 | 0.0000 | **0.0000** | 0.7500 | 68 |
  | `E5_base_mem` | **无** | 15.3458 | 1.0000 | **1.0000** | **1.0000** | 68 |

  ⇒ **两个口径相等**只发生在**零训练**臂（表里无残留）⇒ 偏移的**唯一来源就是训练残留**
  ⇒ 与 DIR-18 的 **D46 双口径**同源（DIR-18 起就同时报两个口径）。
- **为何不改成「清表」**：DIR-18 的 **D55 / probe18h** 已实测「每篇评测前清空槽表」把寻址命中
  从 **0.8833 打到 0.3722** ⇒ **清表会换一个已知会破坏寻址的协议** ⇒ 与「跨会话语义」这个待测问题**混淆**。
  ⇒ 保持 DIR-18/19 的协议（**表带残留**），只把**检查项的表述**改成真正的不变量。
- ⇒ **改写为**：

  > **A20-12′〔真值槽精确可预测且不回绕〕** 写前 `count = C0` ⇒ 逐层
  > `slot_of_write[true_idx] == C0 + true_idx`（**唯一确定**）且 `C0 + 68 ≤ N_ENTRIES`（**不回绕**）
  > ⇒ 真值槽口径（`addr_slot`）有效；并**显式记录** DIR-20 协议下 `addr`（写入序口径）**恒为 0**
  > ⇒ 只有 `addr_slot` 可读（与 DIR-18/19 的 D46 双口径一致）。

- **附带口径记明（不进入判据）**：DIR-20 的 `n_entries` = **表内已用槽数**（含训练残留 ⇒ 136）；
  DIR-19 的 `n_entries` = **本篇写入条数**（68）⇒ 两者的 `n_entries` **不可直接对比**；
  任何进入判据的量（`nll` / `hit` / `addr_slot` / `fid_true` / `cos_y_vtrue`）**口径不变**。

### 0.4 `N_ENTRIES`：128 ⇒ **256**

- **依据**（rev 1.1 §0.2 + 写阶段实测）：写阶段 = **训练残留 68 条 + 本篇 68 条 = 136**
  ⇒ `N_ENTRIES = 128` 会**回绕** ⇒ 真值槽不可预测 ⇒ 指标口径错位。
- ⇒ **`N_ENTRIES = 256`** ⇒ 136 ≤ 256 ⇒ 不回绕（与 A20-12′ 的 `C0 + 68 ≤ N_ENTRIES` 配对）。

### 0.5 脚本与环境（记录用）

| 项 | 值 |
|---|---|
| 脚本 | `~/sgmn-loop/dir20/R1_xsession.py`，md5 `00f6d1b17bcf44842a28c5199c77c8d8`（2094 行） |
| 拼装来源 | `R1_xsession.py` 中间态（md5 `96f4f680126d6831fad8fec9b605bbe7`）+ `_blk_arms.py` / `_blk_readonly.py` / `_blk_main.py` / `_blk_crit.py`（交接包）+ 新写 `_blk_stageA.py` |
| 派生关系 | `revolve/DIR-19/R1_protect.py`（md5 `2ac1d5624869a767357fc1734db48199`）的逐字派生 |
| 工作区 | 训练机 **`/dev/sdf`**（实验卷 `sgmn-exp`，UUID `7a34d754-d2b4-45c1-9b9f-7f6f70d1a942`；`/etc/fstab` **按 UUID** 挂载 ⇒ 设备名漂移/重启都不丢）；`~/sgmn-loop` → `/dev/sdf/home/ec2-user/sgmn-loop` |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct`（离线缓存于实验卷，`dtype=torch.float32` 加载） |
| 语料 | `~/sgmn-loop/corpus/{train,valid}.txt`，md5 `34a77ebe59551b3a3b3d8c0a2797c976` / `d756d22ab2310e7e60c094bea5c5a8eb`（**逐位一致**） |
| 运行环境 | `/opt/pytorch/bin/python`（Python 3.13 / torch 2.14.1+cu130 / **transformers 5.21.0** / numpy 2.3.5），Tesla T4 15 GB |

---

## 1. ★ 冻结判据（**照抄 rev 1.1 §3；阈值与文字一字未改**）

> 口径（写死，防漂移）：**NLL 口径，越低越好**；`Δ(Δnll)(A,B) = mean(nll_B − nll_A)`（>0 ⇒ A 更好）；
> `p` 用 `paired()` 的**符号检验**（n=180）；「三种子同向」= 逐种子的均值差**全部 > 0**。

| 编号 | 判据 | 阈值 | 落点 |
|---|---|---|---|
| **H1″** | 跨会话可用 | Δ(Δnll) > **0.5 nats**，p < 0.05，三种子同向 | `X1_full_state` vs `X11_newinst_noexport` |
| **H2″** | 内容特异（记忆内对照） | Δ(Δnll) > **0.5 nats**，p < 0.05，三种子同向 | `X1_full_state` vs `X4_content_wrong` |
| **H3″** | 跨会话折损有界 | 折损 ≤ **1.0 nats** | 折损 = `benefit(X10_same_session) − benefit(X1_full_state)` |
| **H4″** | 键尺度归一 ⇒ 跨会话寻址稳定 | Δ(Δnll) > **0.5 nats**，p < 0.05，三种子同向 | `X1_full_state`(ortho) vs `X8_linear`（**只读 `hebbian_full` 臂系**） |
| **H5″** | 会话归属 | Δ(Δnll) > **0.5 nats**，p < 0.05，三种子同向 | `X1_full_state` vs `X5_wrong_session` |
| **H6″** | 零效度门槛 | Δ(Δnll) > **0.5 nats**，p < 0.05 | `X6_verbatim` 相对 `X11_newinst_noexport` 的跨会话收益 > 0 |
| **H7″** | 状态成分的必要性 | 至少一条：Δ(Δnll) > **0.5 nats** | `X1_full_state` vs `X2_no_readmap` / `X3_no_addr` |

**报告项（不进入 PASS/FAIL）**：R-1 `X1` 的 hit/addr_slot/fid_true/cos_y_vtrue · R-2 `X8_linear` 的寻址读数 ·
R-3 `X11` vs `X0` 的差 · R-4 `X7_renorm` vs `X8_linear` · R-5 `X1` vs `X10` 的寻址读数。

## 2. ★ 证伪条件（**照抄 rev 1.1 §3.2**）

- **H1″ FAIL** ⇒ 「状态**不可**跨会话搬运」⇒ §7.2 的「跨会话状态语义」写成「**不可行**」。
- **H2″ FAIL** ⇒ 跨会话后**内容不再驱动输出** ⇒ 该子项降级为「跨会话的**通用能力提升**」，不是记忆。
- **H5″ FAIL** ⇒ 收益**不来自**本会话写入的信息 ⇒ 「搬的是模型，不是记忆」⇒ 该子项不成立。
- **H4″ FAIL** ⇒ 键尺度归一对跨会话**没有**额外作用 ⇒ **不得**把 Freeze B 的「键必须逐写归一化」**扩展**到跨会话语义。
- **H3″ FAIL** ⇒ 跨会话的折损**超过事先声明的界** ⇒ 「跨会话」在任务层**不可接受** ⇒ 只能写成「有代价」。

## 3. ★ 证据边界（rev 1.1 §6 + 本 rev 追加）

1. 本方向测的是「**状态经磁盘搬运 + 模型实例重建**后是否仍可用」，**不是**「无限长的会话记忆」。
2. `hebbian_full` 臂在**单篇粒度**上做跨会话比较（修订 20-A）⇒ **不能**外推为「一个会话里能跨会话记住 60 篇」。
3. `X8_linear` 的 η 经过**适配**（修订 20-C）⇒ H4″ 只覆盖「归一 vs 原尺度（惰性写）」。
4. `X6_verbatim` 的 `explicit_kv` 核**对键尺度不敏感** ⇒ **不能**用它支持任何关于键尺度的结论。
5. ★ **（本 rev 追加）** DIR-20 协议下 **`addr`（写入序口径）恒为 0** ⇒ 报告里**只有 `addr_slot` 可读**；
   与 DIR-19 正式轮的读数**只能按 `addr_slot` 对比**。
6. ★ **（本 rev 追加）** 本方向**不测**「会话 B 是否重新训练过读出映射」⇒ 会话 B 的读出映射**就是会话 A 训练好的那份**
   （`X2_no_readmap` 臂除外）⇒ ⇒ 「跨会话」的含义是「**状态**跨会话」，**不是**「能力**跨会话**」。

## 4. ★ Stage A 清单（rev 1.1 §4 的 14 项；**A20-12 按 §0.3 改写为 A20-12′**）

**A20-10 = 继承 DIR-19 的 30 项**（逐字保留，不重写）；新增 13 项：

`A20-1` export→import 往返指纹逐位相同 · `A20-2` 往返后读出余弦 ≥ 0.999 · `A20-3` `strict=True` 喂错形状**必须抛错** ·
`A20-4` `include_read_map=False` ⇒ `W_o`/`γ` 未搬运 · `A20-5` `include_addr=False` ⇒ `count=0`、keys 全零 ·
`A20-6` `X1` vs `X4` 的导出状态：`keys`/`sigs` 逐位相同、`vals` 逐位不同 · `A20-7` `content_wrong` 的置换**是排列且无不动点** ·
`A20-8` 会话 B 的 model 对象 id ≠ 会话 A 的，且会话 A 的模型已释放（显存回落） · `A20-9` 只读评测前后 `core.count` 与 `write_calls` 不变 ·
`A20-11` `vals` 置零 ⇒ 读出逐位不变；`W` 置零 ⇒ 读出改变 · **`A20-12′`** 真值槽 = `C0 + true_idx` 且 `C0 + 68 ≤ N_ENTRIES` ·
`A20-13` `linear` 的实测 ‖k‖² > 100（`renorm` ≈ 5） · `A20-14` 落盘后 `torch.load` 的指纹 == 内存中的指纹。

⇒ **纪律 ①**：Stage A 任一 FAIL ⇒ **不得开主循环**（脚本内部硬闸 + 启动脚本双保险）。
⇒ **纪律 #37**：增量与状态产物按「**方向 + 臂 + 种子**」命名空间隔离
（`formal_incremental_DIR20_{smoke,full}.json` / `.ops/state/DIR20_{arm}_{seed}_{doc}.pt` / `R20_results.json`）。
