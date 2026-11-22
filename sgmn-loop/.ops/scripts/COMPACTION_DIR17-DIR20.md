# SGMN Loop · COMPACTION（DIR-17 → DIR-20 开卡）—— 供另一个 session 续跑

> **生成时间**：2026-11-06 20:35 UTC+8
> **工作区根**：`/workspace/sgmn-loop/`（**NAS 持久卷**；沙箱重建会清 `/usr` 但**保留** `/workspace`）
> **唯一框架来源**：`/workspace/attachments/a12eb0ce5766d0b4/SGMN_v2.4_架构设计.md`（**不得引入其他框架**）
> **本文件用途**：把 DIR-17 至今（并向前覆盖 DIR-01…DIR-16 的结论）的全部可续跑状态**打包成一份**，
> 使新 session 无需重新检索环境与产物即可直接接着 loop。

---

## 0. 一页速览（续跑者先读这页）

| 项 | 状态 |
|---|---|
| 任务 | 用 v2.4 唯一框架，按「方向 → 冻结判据 → Stage A 装置校验门 → Stage B 最小验证 → 对抗批评裁决」逐方向探索 SGMN 架构 |
| 目标 | 「至少 3 个 loop 宣告成功」—— **已达成并超额**：`SUCCEEDED = 14/14` |
| 用户指令 | **「继续 looping，keep looping until finish」**（多次重复）⇒ **必须持续开新方向，直到无剩余开放项** |
| 已闭合方向 | **DIR-01 … DIR-19 全部 COMPLETE**（DIR-11/12/13/14 主假说被证伪 ⇒ **不计入** +1；DIR-18 证实 ⇒ +1） |
| 当前进行中 | **DIR-20 · 跨会话状态语义**（卡片 rev 1.1 已冻结；脚本 `R1_xsession.py` 派生**未完成**） |
| 剩余开放项 | v2.4 §7.2 表中「持久化与 LLM 协同」子项已全部覆盖；**「时间与情节状态」**尚未覆盖 ⇒ ⇒ DIR-20 之后可开 **DIR-21** |
| 纪律条数 | **38 条**（机制级，①–㉟ 连续 + 平铺 1–38） |
| 证据表 | `evidence/table.md`，**只增行**，已到 **D-26**（DIR-20 完成后追加 **D-27**） |
| 关键纪律 | #37 产物按方向命名空间隔离；#38「判据从未被计算」必须与「判据算出 FAIL」严格区分 |

---

## 1. ★ 每个方向的完整台账（DIR-01 … DIR-20）

> `SUCCEEDED` 只在**主假说被证实**时 +1。**主假说被证伪 = COMPLETE 但不计入 +1**（DIR-11/12/13/14 皆如此）。

| 方向 | 主题 | 裁决 | 关键读数 / 结论 |
|---|---|---|---|
| **DIR-01** | 因子化随机操作域与临界步长 | COMPLETE | 因子化 ⇒ 有界最优解；临界步长与谱半径口径分离 |
| **DIR-02** | AR(1)/非 iid 修正 | COMPLETE | 非 iid ⇒ 均方噪声底随相关长度放大；$\varphi=0$ 必须复现 DIR-01 |
| **DIR-03** | 跨层耦合稳定性 | COMPLETE | 零耦合时必须复现 DIR-01 单层域（跨方向回归门） |
| **DIR-04** | **Freeze C 梯度级接口验收**（真实预训练 LLM） | COMPLETE | sg 隔离恒等式成立 ⇒ Freeze C 可实现（**全 loop 的地基**） |
| **DIR-05** | 低秩信息寿命分解与干扰 | COMPLETE | 干扰项主导寿命；$\alpha=0$ 时 $\tau_{\rm ref}$ 有闭式 |
| **DIR-06** | 记忆的因果效应与条件敏感性 | COMPLETE | 写调用计数门 + 梯度门双证 |
| **DIR-07** | 写通道的可行工作窗口 | COMPLETE | $\eta_{\rm learn}\propto1/\mu$ **精确成立**（跨 4 个量级）；稳定天花板同样 $\propto1/\mu$ |
| **DIR-08** | 长视窗下的写通道因果效应复核 | COMPLETE | token 级在线步映射，N=1024，3 种子，640 对/层 |
| **DIR-09** | 写通道的前向任务收益 | COMPLETE | 阴性 ⇒ 写通道本身不产生任务收益 |
| **DIR-10** | **读取侧联合训练下的前向任务收益** | **COMPLETE / SUCCESS (+1)** | 收益**必须**来自读侧；⇒ 打开「前向任务收益」子项 |
| **DIR-11** | 写-读键空间一致性与可检索性 | COMPLETE / **FALSIFIED**（不计 +1） | 共享键空间**只是坐标兼容契约**，不保证检索有效 |
| **DIR-12** | 读取侧检索算子、键一致性与「写是否终于有用」 | COMPLETE / **FALSIFIED**（不计 +1） | 竞争式检索算子下写仍无用 |
| **DIR-13** | 可检索性的结构前提：键去相关、正交键 | COMPLETE / **FALSIFIED**（不计 +1） | 正交键**不足以**救回检索 |
| **DIR-14** | 替代写规则：多槽 / 稀疏字典 / 显式键值表 | COMPLETE / **FALSIFIED**（不计 +1） | 换结构**不能**救回检索 ⇒ ⇒ 瓶颈不在写侧 |
| **DIR-15** | **内容寻址式读取**（查询由「要取回的内容」构造） | **COMPLETE / SUCCESS (+1)** ⇒ 11/11 | 4 种查询构造 ⇒ 可寻址读取成立 |
| **DIR-16** | 可寻址性的泛化（线索换说法） | **COMPLETE / SUCCESS (+1)** ⇒ 12/12 | 4 类上下文变体 ⇒ 寻址对措辞鲁棒 |
| **DIR-17** | 可寻址的读取 + 读出映射训练 ⇒ 记忆内容能否产生任务收益 | COMPLETE / SUCCESS (+1) ⇒ 13/13 | 读出映射训练 ⇒ 内容**能**产生任务收益；D36（状态复位）/D42（判定自洽）两条纪律的来源 |
| **DIR-18** | v2.4 的**实际写规则**（Hebbian）能否支撑「寻址→读出→收益」 | **COMPLETE / SUCCESS (+1)** ⇒ **14/14** | 证据行 **D-25-d**；正式轮 200/60/3 种子、16 臂=48 组合、180 对/臂；Stage A 21/21 PASS、Stage B 12/22 PASS；**H13′ 零效度门槛 +3.2135 PASS**；**H14 记忆内内容对照 +1.7189 PASS**（p≈0，n=180，三种子 +1.7510/+1.7077/+1.6981） |
| **DIR-19** | **受保护联合训练**（定义与首测） | **COMPLETE / FALSIFIED**（**不计 +1** ⇒ 维持 14/14） | 证据行 **D-26**；正式轮 200/60/3 种子、12 臂=36 组合；Stage A **30/30 PASS**、判据 **37/39 PASS**。PASS：H1′ +3.1644、H2′★（记忆内内容对照）**+1.7951**（与 DIR-18 的 H14 互证）、H4′（`N2_kl` ΔPPL 差 +0.5196，保留率 0.972）、H6′。**FAIL：H5′**（旋钮最优保留率 0.9722 < 早停对照 1.0000 ⇒ **保护 ≈ 早停**）、**H7′**（`E5_base_cap` 越界 +0.3021 ⇒ 按预注册 19-J 读作「任务训练本身有代价」） |
| **DIR-20** | **跨会话状态语义**（§7.2 字面上完全未覆盖的子项） | **进行中**（卡片 rev 1.1 已冻结；脚本派生未完成） | 详见 §5 |

### 1.1 ★ DIR-19 的裁决文本（**必须照抄进 v2.4**）

- 证伪的是「**保护 ⊃ 早停**」这一**强形式**；三个旋钮（KL / 低秩 / 前 6 层）**都确实把退化压到界内**（H4′ PASS，ΔPPL 全为负）。
- ⇒ 准确表述：「**未找到能打败早停的保护旋钮**」⇒ **不得**写成「受保护联合训练不可行」。
- ⇒ v2.4 §7.2 的「受保护联合训练」应**降级为「训练步数超参」**。
- ★ **H4′ 的 PASS 有混淆须写下**：旋钮臂 ΔPPL 改善（−0.52 nats）**部分来自「基座在任务分布（wikitext-2 train）上被训练」**（保持探针量的正是 valid 的 NLL）⇒ 只能读成「**放开基座训练不伤基座**」，**不能**读成「保护项起了作用」。

### 1.2 ★ DIR-17…DIR-19 的产物 md5（复核用）

**DIR-18**：`D18_R4_formal.log` `68d79f9be493626fde254cc5ae3d25c6` · `R28_results.json` `58bf810aec9ea26b7d9e8111f333e8fb` · `formal_incremental_full.json` `4edf159f2e2c1bc58b8e14e4e8cfae08`（与服务器逐位一致）
脚本/装置：`sgmn_module_hebb.py` `d62f833b…` · `rounds/R2_write.py` `12dc22f6…` · `rounds/R3_cpu.py` `dd631cf8801987c3bd8c7e6da92243b2` · `rounds/R4_formal.py` `20656a2199b2482b1dff57033b5dd5ca` · `sgmn_module_hebb_v2.py` `3f136ce8…`

**DIR-19**：`D19_R2_formal.log` `18c2b81d99b17d75a1e6a02ea6a398d9` · `D19_R3_reeval.log` `03126fa01480522d2d25c92632bd19b3` · `R19_results.json` `eecc16d6bfb58d639f7b22ab71b9fa2c` · `formal_incremental_DIR19_full.json` `8e25110983dae102fe2484a142601427` · `D19_R1_smoke.log` `a51c0f1bf07cf041dc4937555fc97580` · `D19_R2_formal_attempt1_OOM.log` `fad1bb572dbea46a72272996d506e0fd` · `D19_R2_formal_attempt2_sigbug.log` `070c8df1376c9d8bd9b920eb420e5b86` · `D19_R3_reeval_D82.log` `84b6f0071e89c79fb5728aa44c29e9fb`
脚本/装置：`R1_protect.py` `2ac1d5624869a767357fc1734db48199` · `sgmn_module_hebb_v3.py` `54a140b0882d8dcc2d740ca951a6c29d`

**DIR-20（进行中）**：`sgmn_module_hebb_v4.py` `ad7c1eb621397a37c590b003576e6169` · `DIRECTION.md`(rev 1.0) `e4b0ba36d3eb414ff54a2c5eb871a1ab` · `DIRECTION_rev11.md`（本轮新写，**未记 md5**） · `R1_xsession.py`（派生中，**当前 md5 `96f4f680126d6831fad8fec9b605bbe7` = 仅完成小替换的中间态**）

---

## 2. ★ 机制级纪律（38 条；续跑者必须遵守，不得删除既有条目）

> 完整原文在 `LOOP-MECHANISM.md` §「机制级纪律（续编；全 loop 合计 38 条）」（24–38 条）与前段（1–23 条）。
> **纪律清单跨文件必须双写**（活文档 `LOOP-MECHANISM.md` + 已冻结报告 `LOOP_REPORT_DIR04-DIR17.md`）⇒ 由 #32 要求。
> **每次新增纪律必须同步 4 处**：`LOOP-MECHANISM.md` / `SUMMARY.md` / `evidence/ledger.md` / `revolve/DIR-18/.ops/audit_consistency.py` 的 `CIRC` 串。

**最关键的 12 条（续跑者必读）**：

| # | 一句话 |
|---|---|
| 24 | 判据 FAIL 前必须先排除「任务平凡/饱和」；所有臂都相同 ⇒ 该判据无信息量 |
| 25 | 装置的可变状态必须**每轮复位**（否则写阶段静默空转，伪装成三类不同现象） |
| 26 | 判据 FAIL 必须先自检「**数字方向与判定是否自洽**」⇒ FAIL 可能来自实现缺陷 |
| 27 | ★ 装置的**标定量**必须按**真实输入分布**校准（DIR-18 D50/D51：`pk_scale` 按随机向量标定 ⇒ 真实 hidden 的 ‖k‖² 超标 ~60× ⇒ 「机制不可用」是**假象**） |
| 28 | ★ 收益归因必须有「**同一可用配置**下的对照」＋ 必须报告**机制量**（否则「差值=0」与「机制没生效」不可区分） |
| 29 | ★ 收益必须用「与目标**同分布、可交换**」的对照判定内容性；只报 $\Delta_{\rm code}$ **不构成**内容证据 |
| 30 | ★ 协议修订不得只看「指标更干净」，必须同时报告**机制量**（「更干净」≠「更正确」） |
| 31 | ★ 执行环境的持久性属于实验设计：沙箱重建后 `/usr` 丢失、`/workspace` 保留 ⇒ 模型/语料/装置/脚本/台账**全部落持久卷** |
| 32 | ★ 跨文件的纪律清单必须做「编号连续性 + 双文件对照」审计 |
| 33 | ★ 跨轮复现必须**逐臂**比对全字段，给出「一致 N/M」的**计数式**结论 |
| 34 | ★ 「可恢复性机制」本身必须被**单独测试**（失效不产生错误数值，只浪费时间 ⇒ 只能靠回归测试发现） |
| 35 | ★ 写侧/读侧的配置必须**逐写归一化**（键尺度），否则稳定窗口与学习窗口不相交 |
| 36 | ★ 装置派生必须有**独立回归测试**（derangement 的 `p[i],p[j] = p[j],p[i]` 因 torch 视图别名会写坏排列 ⇒ 必须先取标量副本） |
| **37** | ★ 增量/续跑产物必须按「**方向 + 轮次 + 规格**」命名空间隔离，且必须做「跨方向不会互相复用」的**测试**（DIR-19 D78 曾静默复用 DIR-18 的 `E5_base` ⇒ 整轮作废） |
| **38** | ★ 「**判据从未被计算**」必须与「判据算出 FAIL」严格区分开 —— 空判据先当**实现缺陷**处理，修好后重算，**修前/修后读数同时留档**，**判据阈值与文字不动** |

**其它硬约束**：
- 证据表**只增行、不改行**；失败轮**必须留档**（不删改、不追认）。
- **Never invent missing project details**；证据边界必须写「**支持范围**」与「**不能外推为**」两栏。
- **无代码复用**：每次实验**最小且独立、自包含单文件**；复用必须是前序装置的**逐字派生 + md5 记录 + Stage A 全面复验**。
- 装置缺陷必须由 **Stage A 当场抓出**，不得事后追认；smoke/正式轮后发现缺陷 ⇒ 留档 + 修实现 + 加断言 + 重跑。
- 判据**开跑前冻结**；改动必须**新开 revision**（全部留档）。
- 报告要**尽可能详细完备**；引用时**直接附上原文源码**；正文放文件交付。

---

## 3. ★ DIR-17 → DIR-20 的缺陷留档（失败轮全留档；**不得删除**）

### 3.1 DIR-17
`D36` 循环内状态未复位（写阶段静默空转）⇒ 纪律 25 · `D42` 判据判定与数字方向不自洽 ⇒ 纪律 26

### 3.2 DIR-18（16 个）
`D46` `addr` 两口径混用 ⇒ 双口径（写入序 / 槽空间）+ 精确槽号映射 · `D47` Freeze C 下 $P_v$ 梯度恒 `None` ⇒ `P_v` 不可训 ⇒ 写侧自监督形态 · **`D48`** 评测期清空槽表 ⇒ **被 probe18h 证伪并回退**（清空使寻址命中 0.8833→0.3722）⇒ 纪律 30 · `D49` $d_k=d_v=64$ 声明错误 ⇒ 订正为 hidden_size · **`D50`/`D51`** ★ `pk_scale` 标定口径错 ⇒ 真实 ‖k‖²≈1.7e4（max 1.19e7）⇒ 稳定窗口与学习窗口不相交 ⇒ **「写规则不可用」是假象** ⇒ 纪律 27 · `D52` 槽表清空 ⇒ 同 D48 · `D53` 对照取最稀有 token ⇒ 不可交换 ⇒ 纪律 29 · `D54` 真值槽「值内容反查」歧义 ⇒ 改精确槽号映射 · `D55` 执行环境持久性 ⇒ 纪律 31 · `D56` ★ 续跑跳过判据用 `str(seed)` 查 int 键 ⇒ 续跑形同虚设（**失效不产生错误数值**）⇒ 纪律 34 · `D57` ★ 判据从未被计算（同型，见纪律 38）· `D58` 增量文件 seed 键类型 · `D59` NaN 落盘成 `None` · `D60`/`D61` 打包静默缩水（368→39 文件，零报错）⇒ 打包改自足重建 · `D62` ★ derangement 因 torch 视图别名写坏排列 ⇒ 纪律 36 · `D63` CPU 降级轮与 GPU 轮的逐臂逐位复现（15/15 臂全字段一致 ⇒ 确定性）⇒ 纪律 33 · `D64` 低秩投影残差 · `D65`–`D83` 见 DIR-19

### 3.3 DIR-19（19 个，D65–D83；**全部当场抓出 + 修复 + 留档**）
| 缺陷 | 现象 | 修法 |
|---|---|---|
| D65 | `for R in KNOBS` 遮蔽全局读秩 ⇒ `UnboundLocalError` | 改名 `_Rk` |
| D66 | A19-4 用 token-id 交集（普通词必然重叠）⇒ 必 FAIL | 改**文档窗口 md5 集合**（交集 0） |
| D67 | 低秩投影存 GPU 2.9 GB | 存 CPU |
| D68 | `kv_drift` 未摘钩 ⇒ 零训练 D_kv 5.269e-04 ≠ 0 | 摘钩后取 hidden |
| **D69** | A19-9 期望写反（「KL 应下降」） | 改**双向**生效性检查 |
| **D70** | ★★ `kl_to_reference` 返回 `float` ⇒ **无梯度** ⇒ 旋钮 N2 是**空操作** | 返回带图张量 |
| D71 | 任务损失与 KL 同一 backward ⇒ 两图并存 | 分开 |
| D72 | θ₀ 快照常驻 GPU 2 GB + 梯度跨臂累积 | CPU pristine state_dict + 逐臂清 `p.grad` |
| **D73** | ★★ 全序列 logits ≈ 520 MB ⇒ OOM 直接触发点 | `logits_to_keep=1`（↓860×） |
| **D74 / D74a** | 臂循环分支条件 ⇒ `E5_base_cap` 落进挂记忆分支 ⇒ `ValueError`；且 `train=False` ⇒ 该臂与 `E5_base` 完全相同 ⇒ 判据落空 | 修分支条件 + 改 `train=True` |
| D75 | shell 用全日志 `grep -c "^\[FAIL\]"` 当 Stage A 闸 ⇒ 误杀正式轮 | 只统计 Stage A 段 |
| D76 | 收尾 `passed` 未定义 ⇒ 结果 JSON **从未写出** | 修 |
| **D77** | ★★ 正式轮 `N2_kl` OOM（参考模型常驻 2 GB + 两张图） | 参考 logits 预计算后立刻释放 + 任务图先 backward（**口径仍摘钩，不改为同图**） |
| **D78** | ★★ 增量/结果文件未按方向隔离 ⇒ 实测 DIR-19 **静默复用 DIR-18 的 `E5_base`** ⇒ 基准臂污染 ⇒ **整轮作废** | 文件名加 `DIR19` 前缀 ⇒ **纪律 #37** |
| D79 | `res["dir"]` 仍是 `"DIR-18"` | 改 |
| D80 | 包装层加 `ref_logits` 未加到 `_train_arm_impl` ⇒ 首臂 `NameError` | 两层签名一致 + 新增静态 lint |
| D81 | 收尾又一处 `passed` 未定义 | 修 |
| **D82** | ★★ H4′ 分支条件 `if R not in agg`（`R` 是全局读秩）⇒ 每旋钮被 `continue` 跳过 ⇒ 「无可用旋钮读数」⇒ **H4′ 从未被计算** | 改 `_Rk` + 改逐文档 `paired()`（n=180） |
| **D83** | H4′ 符号判反（`paired(A,B)` 返回 B−A）⇒ 结构上永不可能 PASS | 判据条件改 `_mR > 0`（**阈值与文字未动**） |

### 3.4 ★★ 三条「元教训」（续跑者最该记住）

1. **「写配置」是方向能否成立的前提（D51 第三次复现）**：同一张卡片、同一批判据、同一台服务器，仅写配置不同 ⇒ `linear`+η=1e-7 ⇒ `fid_true` −0.0004、内容判据 ≈ 0（**测的是空记忆**）；`ortho`+η=1e-1 ⇒ `fid_true` +0.9817、内容判据 **+1.7951** ⇒ ⇒ **新方向开跑前必须先对齐写配置**。
2. **「断言失败 ⇒ 必须同时追问『是断言错还是实现错』」**：D69/D70/D81/D82/D83 五个缺陷里**三个**是「判据从未被真正计算」这一类（与 DIR-18 D57 同型）⇒ ⇒ 纪律 #38。
3. **判据实现的修复不得改阈值与文字**：D82/D83 只改实现（变量名、配对口径、符号），判据文本一字未改；修前/修后读数同时留档。

---

## 4. ★ 执行环境（续跑者直接用）

### 4.1 GPU 服务器（正式轮用）
- `ec2-3-27-212-181.ap-southeast-2.compute.amazonaws.com`，**登录用户 `ec2-user`**，密钥 `/workspace/.ssh/SGMN.pem`
- Tesla T4 15 GB / 4 核 / 15 GB RAM / 磁盘 25 G（**余 ~7.2 G**，状态文件必须用完即删）
- `/opt/pytorch/bin/python`（3.13.15 / torch 2.14.1+cu130 / transformers 5.21.0 / numpy 2.3.5）；**transformers 支持 `logits_to_keep=1`**（已实测 shape (1,1,151936)）
- 工作目录：`~/sgmn-loop/dir18/`、`~/sgmn-loop/dir19/` 完好；**DIR-20 尚未创建**（计划 `~/sgmn-loop/dir20/`）
- ★ **落盘路径陷阱**：`HERE.parent` = `~/sgmn-loop/` ⇒ `R19_results.json`/`R20_results.json` 在 `~/sgmn-loop/`；增量在 `~/sgmn-loop/.ops/`（**必须带 `DIR19`/`DIR20` 前缀**）
- 连接模板：
  ```
  ssh -i /workspace/.ssh/SGMN.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 ec2-user@ec2-3-27-212-181.ap-southeast-2.compute.amazonaws.com '<cmd>'
  ```
  scp 加同样选项；后台长任务：
  ```
  setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /opt/pytorch/bin/python -u <script>.py > <unique>.log 2>&1 < /dev/null &
  ```

### 4.2 沙箱（CPU / 分析 / 打包）
- 2 核 / cgroup 上限 3 GB / 无 swap；**沙箱重建会清 `/usr` 但保留 `/workspace`（NAS）**
- torch CPU 持久副本 `/workspace/pylibs/`（2.15.0+cpu）；另有 `/workspace/venv/`
- 模型 `/workspace/models/qwen05b`（Qwen2.5-0.5B-Instruct，fp32）
- 语料 `/workspace/sgmn-loop/corpus/{train,valid}.txt`（md5 `34a77ebe59551b3a3b3d8c0a2797c976` / `d756d22ab2310e7e60c094bea5c5a8eb`）
- fp32 模型 RSS 2.3 GB、加载峰值 2.92 GB ⇒ 用 `safetensors.load_file` + `load_state_dict`；`st.remove()` 只摘钩 ⇒ `_free_stack` + `malloc_trim(0)`；Stage A 与主循环峰值叠加 ⇒ 三阶段拆分
- PyPI：`https://pypi.tuna.tsinghua.edu.cn/simple`；torch CPU wheel `--index-url https://download.pytorch.org/whl/cpu`；HF 权重 `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`；GitHub raw 被封 ⇒ jsDelivr

### 4.3 ★ 显存管理三原则（T4 15 GB）
1. θ₀ 只存 **CPU pristine state_dict**（不占显存）
2. 参考模型按臂临时建并**训练前**预计算 logits 后**立即释放**
3. **任一时刻只有一张前向图**（任务图先 backward）+ `logits_to_keep=1` + 逐臂清 `p.grad` + `empty_cache()`

---

## 5. ★★★ DIR-20 · 跨会话状态语义（**当前进行中；续跑者的第一任务**）

### 5.1 方向定义
- **source**：v2.4 §7.2「持久化与 LLM 协同」⇒「**跨会话状态语义**」—— §7.2 表里**字面上完全未覆盖**的子项（前 19 个方向**全部单会话内**运行：训练 200 篇 + 评测 60 篇，同进程同实例）。
- **主假说 H 四子句**：(a) 状态可跨会话搬运并在全新实例上产生收益；(b) 收益仍由**记忆内容**驱动；(c) 收益折损**有界且事先声明**；(d) 可用性**依赖具体机制成分**。
- **7 条判据**（阈值冻结）：H1″ 跨会话可用（> 0.5 nats，参照 `X11_newinst_noexport`）· H2″ 内容特异（`X1` vs `X4`）· H3″ 折损 ≤ **1.0 nats**（vs `X10_same_session`）· H4″ 键尺度归一（`X1` vs `X8`）· H5″ 会话归属（`X1` vs `X5`）· H6″ 零效度门槛（`X6` vs `X11`）· H7″ 成分必要性（`X1` vs `X2`/`X3` 至少一条）。
- **报告项**：R-1 `X1` 的 hit/addr_slot/fid_true · R-2 `X8` 的 hit/addr_slot · R-3 `X11` vs `X0` 的差 · R-4 `X7` vs `X8` · R-5 `X1` vs `X10` 的寻址读数。
- **11 臂**：`X0_no_state` / `X1_full_state`(主臂) / `X2_no_readmap` / `X3_no_addr` / `X4_content_wrong` / `X5_wrong_session` / `X6_verbatim`(explicit_kv) / `X7_renorm` / `X8_linear`(η=1e-5) / `X10_same_session` / `X11_newinst_noexport`。

### 5.2 ★ 卡片 rev 1.1 的四条修订（**已冻结，2026-11-06**；全文 `revolve/DIR-20/DIRECTION_rev11.md`，17 896 B）

| 修订 | 内容 | 根因 |
|---|---|---|
| **20-A**（核心） | 会话 A 第二阶段**逐篇**「写 → 导出状态」；会话 B **逐篇导入并只读读出** ⇒ 60 篇 × 3 种子 = **n=180**（与 DIR-19 同口径） | ★ `hebbian_full` 的 **λ 衰减**：`W ← W − η(α(Wk−v)kᵀ + λ(W−W⁰))` ⇒ 每步 `(W−W⁰) × (1−ηλ)` = **0.995** ⇒ 时间视界 ≈ **1/(ηλ) ≈ 200 步 ≈ 2.7 篇** ⇒ 60 篇（4080 步）内容衰减 `e^{−20} ≈ 2e−9` ⇒ 整段导出会让 H1″ **假 FAIL**。另：`N_ENTRIES=128` ≪ 4080 条 ⇒ 槽表回绕 ⇒ 只剩最后 ≈1.7 篇 |
| **20-B** | `X4_content_wrong` 的置换施加在**会话 A 的写阶段**（沿用 DIR-18 修订 68 / DIR-19 A19-10 已验证的 `content_wrong` derangement 钩子：文档内无不动点置换值来源位置，**键与签名逐位不动**） | ★ `hebbian_full` 的**内容在 `W` 里，不在 `vals` 里**（`vals` 只用于写保真度测量）⇒ 在导出状态上置换 `vals` 对读出**零影响** ⇒ H2″ 将**结构上永不可能 PASS**（D82/D83 同型 ⇒ 纪律 #38） |
| **20-C** | `X8_linear` 的 **η = 1e-5**（尺度适配档），`η=1e-1` 会让写规则**发散** | `linear` 键实测 ‖k‖² ≈ **1.7e4** ⇒ 稳定上界 `η < 2/(λ+α‖k‖²) ≈ 1.2e−4` ⇒ ⇒ 「ortho vs linear」会测成「发散 vs 正常」⇒ ⇒ H4″ **PASS 也只能读成「原尺度下写规则惰性 ⇒ 内容取不回」**，**不得**读成「跨会话尺度失稳」 |
| **20-D/E/F** | 臂表口径（`X6` 用 `explicit_kv`、**不参与 H4″**；`X5` 导入**另一个种子**的状态；`X0`/`X11` 的分解）+ Stage A 增补 A20-11…A20-14 + 判据「读什么」的落点写死 | 见卡片 |

⇒ ⇒ **判据的阈值与文字一字未改**；改的只是「怎么把它算出来」。

### 5.3 ★ 本 session 新发现的、**尚未写进任何文档**的事实（续跑者必须知道）

1. **DIR-19 的 `N1_linear_ctrl`（linear + η=1e-7）读数**：`nll=10.8713`（benefit **+1.5317**）、`hit=0.0296`、`addr_slot=0.9037`、`fid_true=0.0016`、`cos_y_vtrue=0.0015`。
   ⇒ ⇒ ★ **该臂的记忆里没有内容（fid_true≈0、hit≈0），但仍有 +1.53 nats 的收益** ⇒ ⇒ ⇒ **DIR-19/DIR-18 的「收益」有相当一部分并非来自内容召回**。
   ⇒ ⇒ ★ **对 DIR-20 的直接影响**：`H1″`（`X1` vs `X11`）测的是「**整个记忆效应**」（内容 + 通用注入）⇒ 仍有效；但 **H2″（`X1` vs `X4`）才是内容性的判据** ⇒ ⇒ ⇒ **DIR-20 的裁决文本必须把这两件事分开写**，否则会重复 DIR-18/19 的过度解读。
2. **`hebbian_full` 的内容在 `W` 里**（已由阅读装置源码确认：`read` 是 `y = sg(W)·q`；`vals` 只进 `fidelity()`）⇒ ⇒ 修订 20-B 的依据；并新增 Stage A **A20-11**（把 `vals` 置零 ⇒ 读出逐位不变；把 `W` 置零 ⇒ 读出改变）当场断言。
3. **`hebbian_full` 的 λ 衰减时间视界 ≈ 2.7 篇**（公式见上）⇒ ⇒ 这是**机制本身的性质**，**不是**跨会话的性质 ⇒ ⇒ DIR-20 只能做**单篇粒度**的跨会话比较 ⇒ ⇒ 证据边界必须写明「**不能外推为『一个会话里能跨会话记住 60 篇』**」。
4. **`N_ENTRIES=128` 在 DIR-20 协议下会绕**：会话 A 训练后表的残留 = 74 条（最后一篇训练文档），再加本篇的 68 条 = **142 > 128** ⇒ ⇒ **DIR-20 必须把 `N_ENTRIES` 改为 256**（否则真值槽不可预测 ⇒ 指标口径错位）⇒ ⇒ **这是脚本里尚未做的一处编辑**（见 §6 步骤 3）。
5. **状态文件体积**：`N_ENTRIES=256` ⇒ 每个状态文件 ≈ W(3.2 MB) + keys/vals/sigs(3×0.9 MB) + Wo(3.2 MB) ≈ **21 MB** ⇒ 60 篇 × 3 种子 = **3.8 GB/臂** ⇒ ⇒ **必须逐篇导入后立即删除**（脚本已实现「用完即删」，但 **`X5_wrong_session` 需要「另一个种子」的文件 ⇒ 删除必须放在「该臂全部种子跑完」之后** ⇒ 脚本已按臂末尾删除 ✔）。
6. **`explicit_kv` 的读对键尺度不敏感**（余弦）⇒ ⇒ `X6_verbatim` 只能作零效度门槛对照，**不能**支持任何键尺度结论（卡片已写死）。

### 5.4 ★ DIR-20 脚本的**当前状态（未完成！）**

`revolve/DIR-20/R1_xsession.py` 目前**只完成了小替换**（import v4、`res["dir"]="DIR-20"`、`_DIR_TAG="DIR20"`、`R20_*.json`），**主循环 / 判据 / 只读评测 / 臂表尚未拼进去**。
已写好、**等待拼装**的代码块（都在 `revolve/DIR-20/`）：

| 文件 | 内容 | 拼装位置 |
|---|---|---|
| `_blk_arms.py` | DIR-20 的 11 臂臂表 | 替换 `# ★ DIR-19 臂表` … 到 `def make_stack(` 之前 |
| `_blk_readonly.py` | `eval_arm_readonly(st, d, ctrl_ids)` | 插入到 `def eval_base(` 之前 |
| `_blk_main.py` | 两段式会话主循环（阶段 1 会话 A / 阶段 2 会话 B） | 替换 `    # ---------------- 主循环` … 到 `    # ---------------- 汇总` 之前 |
| `_blk_crit.py` | 判据 H1″…H7″ + 报告项 R-1…R-5 | 替换 `    # ══════════════════ ★★【DIR-19】判据` … 到 `    # —— 汇总报告 ——` 之前 |
| **（未写）** | Stage A 的 **A20-1…A20-9 + A20-11…A20-14** | 插入到 `    # ---------------- ★ Stage A 硬闸` 之前 |
| **（未写）** | 文件头 docstring 的 DIR-20 段 | 插到模块 docstring 开头 |

⇒ ⇒ ★ **拼装完还必须补**：
- `import gc`、`import hashlib`、`import os`（`os` 已有？⇒ 需检查）
- `N_ENTRIES = 256`（★ 见 §5.3 第 4 点）
- 头部的 `print("=== DIR-17 R1 …")` 文案改成 DIR-20
- `train_arm` 在 DIR-20 只用 `base_mode="none"`/`kl_lambda=0`（**DIR-19 的保护旋钮逻辑在 DIR-20 不用**，但函数逐字保留即可）

### 5.5 ★ DIR-20 的 Stage A 清单（卡片 rev 1.1 §4）

**继承 DIR-19 的 30 项**（A1'…A19-10，**逐字保留**）+ 新增：
**A20-1** export→import 往返指纹逐位相同 · **A20-2** 往返后读出余弦 ≥ 0.999 · **A20-3** `strict=True` 喂错形状**必须抛错** · **A20-4** `include_read_map=False` ⇒ `W_o` 未搬运 · **A20-5** `include_addr=False` ⇒ `count=0`、keys 全零 · **A20-6** `X1` vs `X4` 导出状态：`keys`/`sigs` 逐位相同、`vals` 逐位不同 · **A20-7** `content_wrong` 的置换**是排列且无不动点** · **A20-8** 会话 B 的 model 对象 id ≠ 会话 A 的，且会话 A 的模型已释放（显存回落）· **A20-9** 只读评测前后 `core.count` 与 `write_calls` 不变 · **A20-11** `vals` 置零 ⇒ 读出逐位不变；`W` 置零 ⇒ 读出改变 · **A20-12** 本协议下 `true_slot` 精确可预测（`addr == addr_slot`）· **A20-13** `X8_linear` 的实测 ‖k‖² ≫ 5 · **A20-14** 落盘后 `torch.load` 的指纹 == 内存中的指纹。

---

## 6. ★ 续跑者的执行清单（照做即可）

### 步骤 1：拼装 DIR-20 脚本
在 `/workspace/sgmn-loop/revolve/DIR-20/` 下用 Python 脚本把 §5.4 的代码块按表拼接进 `R1_xsession.py`，并补齐 `import gc / hashlib / os`、`N_ENTRIES = 256`、头段文案。
⇒ 语法自检：`python3 -c "import ast;ast.parse(open('R1_xsession.py').read())"` ⇒ 记录 md5。

### 步骤 2：写 Stage A 块（A20-1…A20-9 + A20-11…A20-14）并插入
⇒ 沿用 DIR-19 的 `.ops/test_static_lint.py`（包装层↔实现层签名一致 / 判据块无裸 `R` / 同路径 `rec()` 名不重复 / 负向对照）与 `.ops/test_signature_consistency.py` ⇒ ★ **必须为 DIR-20 复制并复验**（纪律：派生必须有独立回归测试）。

### 步骤 3：上传服务器 + Stage A + smoke + 正式轮
```
mkdir -p ~/sgmn-loop/dir20/.ops
scp -i /workspace/.ssh/SGMN.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  R1_xsession.py sgmn_module_hebb_v4.py ec2-user@ec2-3-27-212-181...:~/sgmn-loop/dir20/
```
⇒ Stage A 必须 **全 PASS**（否则不得开主循环，纪律 ①）⇒ smoke（40/12/2 种子）⇒ 正式轮（**200/60/3 种子**，11 臂 × 3 种子 = **33 组合**，n=180/臂）。
⇒ ★ 建议脚本 `.ops/run_dir20_formal.sh`：**只统计 Stage A 段**的 `grep -c "^\[FAIL\]"`（D75 的教训）。

### 步骤 4：DIR-20 读数后
1. 按卡片 §3 的判据 + §3.2 的证伪条件判读 ⇒ **追加证据行 D-27**（`evidence/table.md` **只增行**）
2. 写 `revolve/DIR-20/VERDICT.md`（**必须**把「内容性收益」与「通用注入收益」分开写 ⇒ 见 §5.3 第 1 点）
3. 更新 `SUMMARY.md` / `LOOP-MECHANISM.md` / `DIRECTION-REGISTER.md` / `revisions.md` / `evidence/ledger.md` / `REPORT.md`
4. 若主假说被证伪 ⇒ **不计入 SUCCEEDED**（维持 14/14）；若四子句全 ✔ ⇒ **SUCCEEDED = 15/15**

### 步骤 5：决定是否开 DIR-21
v2.4 §7.2 中**尚未覆盖**的子项：**「时间与情节状态」** ⇒ ⇒ 这是 DIR-21 的天然来源（须从 §7.2 的具体开放项派生，**不得自造议题**）。

### 步骤 6：最终重打包与审计
```
bash revolve/DIR-18/.ops/repack.sh          # ★ 需先加 DIR-20_artifacts.zip + MANIFEST 扩 DIR-20
bash revolve/DIR-18/.ops/rebuild_full.sh     # README 文案：SUCCEEDED = 14/14（DIR-20 闭合后改 15/15）
python3 revolve/DIR-18/.ops/audit_consistency.py   # 六项须全通过；纪律计数一致；证据表唯一
```
⇒ 交付用 `<deliver-assets>` 附：`SGMN_LOOP_full.zip`、`LOOP_REPORT_full.zip`、`DIR-19_artifacts.zip`、`DIR-20_artifacts.zip`、`REPORT.md`、`SUMMARY.md`、`revolve/DIR-19/VERDICT.md`、`revolve/DIR-20/DIRECTION_rev11.md`、`evidence/table.md`、`MANIFEST.md`（**须核验当前状态**）。

---

## 7. ★ 文件地图（`/workspace/sgmn-loop/`）

| 路径 | 内容 |
|---|---|
| `SUMMARY.md` | 阶段总结（§1 三个方向互补 / §2 方法论教训 / §3 对 v2.4 的反哺 22 条 / §4 不宣称什么 / §5 剩余方向 / §2.15 DIR-19 裁决） |
| `LOOP-MECHANISM.md` | **机制级纪律 38 条**（活文档，纪律续编从第 495 行起） |
| `DIRECTION-REGISTER.md` | 方向登记表（DIR-01…DIR-16 表 + DIR-17…DIR-19 详段 + 下一方向提示） |
| `revisions.md` | 卡片修订留档（R18-A…R19-B） |
| `evidence/table.md` | **证据表（只增行）**，已到 D-26 |
| `evidence/ledger.md` | 实验台账（133 KB，最详细） |
| `REPORT.md` | 总报告（1562 行；§2.19 DIR-18 / §2.20 DIR-19 进行中 / §2.21 DIR-19 裁决） |
| `MANIFEST.md` | 打包清单 |
| `LOOP_REPORT_DIR04-DIR17.md` | DIR-04…DIR-17 的冻结阶段报告（纪律双写用） |
| `revolve/DIR-XX/` | 每方向的 `DIRECTION*.md`（卡片全部 revision 留档）/ `VERDICT.md` / 装置 / 脚本 / `.ops/`（含产物与日志） |
| `revolve/DIR-18/.ops/` | `repack.sh` / `rebuild_full.sh` / `audit_consistency.py` / `test_static_lint.py` / `test_signature_consistency.py` / `formal/`（正式轮产物与日志） |
| `revolve/DIR-19/.ops/formal/` | DIR-19 正式轮全部日志与结果 JSON |
| `revolve/DIR-20/` | **进行中**：`DIRECTION.md`(rev 1.0) / `DIRECTION_rev11.md`(rev 1.1 冻结) / `sgmn_module_hebb_v4.py` / `R1_xsession.py`(派生中) / `_blk_*.py`(待拼装) |
| `*.zip` | `SGMN_LOOP_full.zip` / `LOOP_REPORT_full.zip` / `DIR-10…DIR-19_artifacts.zip` |

---

## 8. ★ 交付与核验规范（写给续跑者）

- 最终回复**必须**用 `<deliver-assets><media src="..."/></deliver-assets>` 附文件；**须核验文件当前状态**（`unzip -t`、`md5sum`、`ls -l`）；失败/未核验的输出**如实报告**，不得声称已交付。
- ★ **大段 markdown 不要用 heredoc 写入**（会生成损坏内容）⇒ 用 `write` 工具写临时文件再 append。**本 session 已两次出现工具调用被截断**，事后核对 `REPORT.md` 完好 ⇒ ⇒ **写入后必须核对**。
- 打包暂存区 `/workspace/.pkg/SGMN_LOOP_PACKAGE`（**NAS 持久卷**，沙箱重建不丢）。
- `audit_consistency.py` 的六项：纪律连续 / 双文件对照 / 证据表编号唯一 / SUCCEEDED 计数 / 3 个关键 md5 / 纪律计数跨文件一致。

---

## 9. ★ 一句话总结（给下一个 session 的你）

**DIR-01…DIR-19 已全部闭合，`SUCCEEDED = 14/14`；机制级纪律 38 条；证据表到 D-26。
接下来的第一件事是把 DIR-20 的脚本拼装完成（§5.4 + §6 步骤 1–3）⇒ 跑 Stage A → smoke → 正式轮（33 组合，n=180/臂）⇒ 追加 D-27 ⇒ 写 VERDICT ⇒ 决定是否开 DIR-21（「时间与情节状态」）。**
**永远记住：「keep looping until finish」；失败轮留档；判据开跑前冻结；产物按方向隔离；判据「从未被计算」≠「算出 FAIL」。**
