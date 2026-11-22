# 环境与装置笔记（常驻，供各 cycle 查阅；不再塞进 handoff.md）

> 由 Cycle 182 从 handoff 抽出（handoff 只留一段话）。**这里只记可复用的装置/脚本事实**，
> 不是实验结论 —— 结论一律在 `evidence/table.md` 与 `revolve/DIR-xx/VERDICT.md`。

## 装置常量（已核于 trainer）
- `sgmn_module.py` md5 = `80e334a5a6751a07d9e03162ebf17ff0`（DIR-95/96 均未改）。
- `dir91/R2_task91.py`：`DEV`、`MOUNT=[8,12,16]`（**层下标**）、`ALPHA=1.0`、`LAM=0.05`、
  `ETA_WRITE=3e-5`、`READ_STEPS=600`、`READ_LR=1e-3`、`CLIP=1.0`、`N_FACTS=32`、
  `N_TRAIN_KEYS=16`、`R=8`（仅 factorized 用）。
- 模型：`hidden_size = d_model = d_k = d_v = 896`；`W` 形状 `(d_v, d_k)`；
  `‖W⁰‖_F = sqrt(896) = 29.9333`（实测 29.9249–29.9462）。
- `R2_task91.make_stack(use_factorized=False)` ⇒ 该装置里 `W = m.Wfull`（不是 `W0+U Vᵀ`）。

## 脚本事实（踩过的坑）
- `nll_of_candidate` / `make_eval_items` / `score_and_accuracy` / `acc_by_question`
  **都在 `dir91/R2_task91.py`**，不在 `probe91_womap.py`。
- `items` 结构：`{key_idx, prompt_ids, cand: [(ids_list, is_correct)]}`，候选**单 token**（有断言）。
- `acc_by_question` 要求**同题 3 次抽样全对** ⇒ 按题机会 `(1/4)³ = 0.0156`（**不是** 0.25）。
- `probe91_womap.eval_keys` 自己会设 `no_write=True`；但 **`train_read_task` 末尾还原为 `False`**
  ⇒ 训练后评测仍须**显式重封**（`DEF-95-01`）。
- `probe91_kpool` 原生臂 `K_mean_V_mean` ⇒ `RULE=("mean","mean")`、`need_vt=False`
  （`vph_by_fact` 可传 `None`）。
- `make_stack(gamma=0.0, ...)` 是 `run_arm` 的口径（γ 从 0 训起）。
- 写入**每个 chunk 一次前向一次写**；`ΔW` 的 `ν` 实测**与块数无关**（60 块 ≈ 600 块），
  但**共因机制未定**（「饱和」「衰减平衡」两解释均已在 `DIR-96` 撤回）。

## 运维
- **一次一栈**：多栈并存会 autograd 版本冲突（`DEF-94-03`/`DEF-95-02`）。
- 长跑用 `setsid nohup ... > log 2>&1` 后台 + 轮询；**不要**用 `grep` 直接过滤 ssh 输出
  （会吞 traceback，`DEF-95-03`）。
- 产物 JSON 必须**先聚合、再落盘、后抛异常**（`DEF-95-04`）。
- `push52.sh` **拒绝 `out/` 目录**（"只出不进"）⇒ 远端须 `ssh ... mkdir -p dirNN/out` 自建；
  结果 JSON 用 `tar | base64` 拉回本地复核。
- `repack.sh` 会调用 `rebuild_full.sh`（第 104 行）⇒ 其"完整包重建结束"即证明重建已跑。
- ssh 期间常掉线 ⇒ 每条远端命令都套重试循环。
- 审计脚本 `audit_consistency.py` 的 **④ 用 `re.search` 首匹配**会打印**过期**的
  `SUCCEEDED = 39/39`（历史快照）且只断言 `n==d`；**权威项是 ④b**（读 `当前计数：SUCCEEDED = N / M` 规范标记）。
  ★★ **但 ④b 曾经也是空的（`DEF-112-07`，Cycle 209 修）**：`_derive_count` 要求 **`n == d`**，
  而本项目所有真实行都写成 `SUCCEEDED = N / 55`（分母是**方向总数**，`N ≠ 55`）⇒
  旧实现**只看得见早期 `N / N` 的历史快照**（实测命中 `{23,38,39,48,49}`）⇒
  「头部 vs 证据表」交叉校验**两侧都停在旧快照上而"通过"**。
  ⇒ 已改为「取所有 `SUCCEEDED = N / M` 提及中的**最大 N**」（依据：SUCCEEDED **单调不减**）。
  ⇒ ★ **教训（可推广）**：交叉校验只能证明"两边一致"，**不能证明"两边都新"**。
     凡自检脚本，务必拿**一个已知会变的量**喂进去验证它真的会变 —— 否则它永远是绿的。

## 典型耗时
- 算术前置（纯线性代数，不训练）≈ 40 s；`--stage smoke`（1 seed、写入砍 1/10）≈ 1.5–2 min；
  `--stage formal`（2 seed × 4 条件 × 2 键集）≈ 9 min；`DIR-96` 正式轮（含 8 次随机 logit 矩阵/seed）≈ 9–10 min；
  `probe96_attrib.py`（重跑写+训 + 秩/归因）≈ 6 min；审计 + repack ≈ 1 min。

## ★ SSH 偶发「banner 后被断」（Cycle 187 实测，务必照此处置）
- **症状**：`ssh -vv` 显示 `Connection established` → 随即 `kex_exchange_identification: Connection closed by remote host`；
  `nc -zv <ip> 22` **成功**（TCP 通、sshd 立刻断）。`rsync` 报 `unexplained error (code 255)`。
- **成因（最可能）**：**短时间内大量 SSH 连接**触发 sshd 的 `MaxStartups`/限速，或 fail2ban 类临时封禁。
  本循环的 `for i in 1..N; do ssh ...; sleep 25; done` **重试循环正是诱因** —— 每次重试都可能**延长**封禁。
- **重试预算（Cycle 188 收紧）**：**每个 cycle 默认只在开头单发 1 次**；若仍有富余时间最多再 1 次
  （**合计 ≤2 次/cycle**），间隔 ≥8 min。**不要**用 3 次以上的连发 —— 6 次/小时足以让 fail2ban 类封禁**持续续期**。
- **正确处置**：
  1. **立刻停止重试**，静默等待 **10–30 min**，然后**只发一次**连接尝试；
  2. **不要**用 `sleep 15/25` 的高频重试循环；单次失败就转去**不需要 SSH 的本地工作**（写卡片/缺陷表/证据行草稿）；
  3. 恢复后先跑一次 `echo OK` 确认，再继续推送。
- **反例（本循环曾犯）**：`for i in 1 2 3 4; do push52.sh ...; sleep 25; done` —— 4 次连发，无效且可能加剧封禁。
- **判据纪律**：此类中断属「**从未被计算**」，依 round standard **不得**记 PASS/FAIL，也不得据此改写缺口。

### ★ 缺陷类「追加式日志 + 首匹配取数」（`DEF-96-11` / `DEF-188-01`）
`SUMMARY.md` 这类文件是**追加式**的：裸 `SUCCEEDED = N / N` 会随轮次不断新增，
**历史上最早的那个快照永远排在前面** ⇒ 任何"取第一个"的读法都必然读到**过期值**。
- `DEF-96-11`（Cycle 188 修）：`audit_consistency.py` §④ 用 `re.search(...)` 首匹配 ⇒ 读到 `39 / 39`（真值 `49 / 49`）。
  旧断言只有 `n == d`，**每个历史快照都满足** ⇒ 缺陷被**静默通过**。
- `DEF-188-01`（Cycle 188 修）：`repack.sh:66` 用 `grep -o ... | head -1` ⇒ **同一缺陷的 shell 写法**，
  且更严重：它把过期数字**烧进交付包里的 `MANIFEST.md`**（没有 ④b 那样的交叉核）。
- **统一修法**：**优先读规范标记**`当前计数：SUCCEEDED = N / M`，**仅在无标记时**退回裸匹配，
  并**如实说明走的是哪条路**。`repack.sh` 用 `${_SUCC#当前计数：}` 去掉前缀以保持 MANIFEST 原格式。
- **检查纪律**：以后任何新增脚本要读计数，**必须先问「这个文件是追加式的吗」**；
  是 ⇒ 必须用规范标记，不得用 `head -1` / `re.search` / 正则首匹配。

## ★★ DEF-189-01 · 「trainer 连不上」的真因是**本机 FlClash 代理**（Cycle 189 定案）

**这一段请优先读**：Cycle 186–188 连续三个 cycle 把 `kex_exchange_identification: Connection closed by remote host`
判成「trainer 宕机 / 被封禁」，并据此向用户升级 **Q3**（请求重启实例、解封 IP）。**该判断是错的**：
`trainer` 的 `uptime` 全程 **5 天 8 小时**、load 0.00 —— 实例一直是好的。**错的在本机网络栈**。

### 症状（误导性极强，务必记住这套组合）
| 观测 | 读数 | 误导方向 |
|---|---|---|
| `ping 1.1.1.1` | **0% 丢包，size 1472 -M do 也 OK** | ⇒ 像是"网络完全正常、所以问题在远端" |
| `nc -zv 13.238.13.194 22` | **成功** | ⇒ 像是"TCP 通、只有 sshd 拒绝" |
| `ssh` 实际结果 | `Connection established` → 立刻 `Connection closed by remote host` | ⇒ 像是远端 sshd 的 `MaxStartups`/fail2ban |
| `curl https://github.com` | `SSL: UNEXPECTED_EOF_WHILE_READING`（等价症状！） | 若早注意到 HTTPS 也断，就能立刻排除"只有 SSH 坏" |
| `aws sts get-caller-identity` | 同样 `SSL: UNEXPECTED_EOF_WHILE_READING` | 同上 |

### 真因（机械证据）
- 本机 **FlClash** 代理在跑，TUN 接口名 `FlClash`，策略路由在 **table 2022**：
  - `ip route get 13.238.13.194` ⇒ `via 198.18.0.2 dev FlClash table 2022 src 198.18.0.1`
  - `ip route get 8.8.8.8` ⇒ 同上；**所有**外网目的地都被改道进 FlClash。
- FlClash 的**上游节点已死**：连 `curl -x http://127.0.0.1:7890 https://github.com` 也是 `000`
  （7890 在 `ss -ltnp` 里确实在 LISTEN，但代理转发不出去）。
- ⇒ **TCP 能建连（SYN 由 TUN 本地代答）但随后被切断**，于是 SSH 死在 kex、HTTPS 死在 TLS 握手。
- ICMP 不受该链路影响 ⇒ `ping` 全绿，形成"网络正常"的假象。

### 正确解法（**不需要 root、不改系统路由、不动用户代理**）
**仅对外层一跳绑定真实出口源地址**，让内核据此选中真实网卡的路由，从而绕开 FlClash 的策略表：
```bash
ip route get 13.238.13.194 from 10.29.221.115     # ⇒ via 10.29.0.1 dev wlp1s0  ✅
ssh -b 10.29.221.115 -i ~/.ssh/Bastion.pem ec2-user@13.238.13.194   # ⇒ BASTION-OK
```
已封装成脚本，**以后一律用它们连 trainer**：
- `.ops/scripts/tr.sh <host>` —— 到 bastion 的 `-W` 转发（含 `-b` 绑定）
- `.ops/scripts/run_on_trainer.sh '<远端命令>'` —— 一步到 trainer 执行命令
- **不要**再用裸 `ssh trainer`（会走 FlClash 而失败）

### 纪律（新增）
1. **任何"远端连不上"的排查，第一件事是测一个无关的外部 HTTPS**（如 `curl -s -o /dev/null -w %{http_code} https://github.com`）。
   若 HTTPS 也断 ⇒ **先怀疑本机代理，不要怀疑远端**。
2. **`ping` + `nc -zv` 全绿完全不能证明链路可用** —— 它们走的是 SYN/ICMP，代理 TUN 会本地代答。
3. **判"远端宕机"必须有一条远端自身的证据**（如本轮的 `uptime`），否则只能记「**链路不可用，成因未定**」。
4. 源地址 `10.29.221.115` 随 WiFi 重连会变 ⇒ 失效时用 `ip route show default` 取新 `src`，
   或直接设 `SGMN_LAN_SRC=<新地址>`。
5. **禁止**为了修这个去 `kill` 用户的 FlClash 或改系统路由 —— 越界，且用户的代理可能是他们自己要用的。

---

## ★ 多格实验的数值可复现性（`DEF-105-02`，Cycle 196 实测；**★ 边界见文末 Cycle 209 追加**）

**结论**：**同一配置在同一进程内与不同配置混跑后，读数不可比。**

**实测证据**（`DIR-105`，seed 7777 `lin@600`，同代码同数据）：

| 运行方式 | `acc_all32` |
|---|---|
| 隔离单跑（重复 2 次） | **0.8750**（= `D-133`） |
| 同进程、前一格为同配置 | **0.8750** |
| 同进程、前有 `lin@3000`+`gate`+`mlp1` 共 6 格 | **0.7812** |

**排除项**（`dir105/diag105_state.py` 逐格实测）：
- 基座模型全部参数+buffer 的 sha1 **恒为 `26f966c3b98ccc30`**（12 格全程不变）⇒ **非模型污染**；
- `W_drift == 0.000e+00` 逐位 ⇒ **非写入污染**；
- `make_stack(seed=…)`、`random.Random(555)` 逐格新建 ⇒ **非 RNG 泄漏**。

**根因**：cuBLAS workspace 的分配受**前序格**影响 ⇒ GEMM 归约顺序变化 ⇒
末位差异经 600–3000 步 Adam 混沌放大 ⇒ 决策边界附近的 item 翻面。

**两种修法与取舍**：

| 修法 | 效果 | 代价 |
|---|---|---|
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 可复现 | **3.2× 慢**（65 s → 208 s/格） |
| **一格一进程**（推荐） | 可复现（分配器初态恒同） | 仅 ~35 s/格启动 |

**做法**：写一个 `run_formalXXX.sh` 驱动，对每个 `(family, steps, seed)` 单独起进程，
每格落盘为 `out/cells/<tag>.json`（**存在即跳过 ⇒ 可断点续跑**），最后用 `--stage merge` 合成并裁决。
参考实现：`dir105/run_formal105.sh` + `R1_task105.py --stage merge`。

**★ 同时必须带的守卫**：每次多格实验都要有一条「**基座 sha1 不变**」的检查，
否则无法区分「代码层状态泄漏」（如 `DEF-105-01`）与「纯数值混沌」。
`diag105_state.py` 可直接复用（改 `--fams/--steps/--seeds`）。

**★ 另一条教训（`DEF-105-05`）**：**只对有余量的读数主张逐位复现**。
`acc_all32`（有真实 margin）可逐位复现；`acc_W0` 落在 chance 附近（0.06–0.31），
同配置连跑两次得 0.1875 / 0.0625 ⇒ 它是**不可复现量**，只能作**粗对照**用，
且其判据阈值必须**远离观测带**（本卡的 `0.50` 距上界 0.31 仍有余量）。

## ★★ 运维陷阱：`run_on_trainer.sh` 带 **`-n`**，**不能用 stdin 管道改远端文件**（Cycle 202 实测踩到）

`run_on_trainer.sh` 的 ssh 带 **`-n`**（stdin ← /dev/null）。因此：

```bash
# ✗ 危险：远端文件被**截断为 0 字节**，且**不报错**
run_on_trainer.sh 'cat > ~/sgmn-loop/evidence/table.md' < evidence/table.md
# ⇒ 远端 table.md = 0 字节（md5 d41d8cd98f00b204e9800998ecf8427e）
```

★ 本次实测后果：`evidence/table.md` 与 `decisions/DECISIONS.md` **双双被清空**；
因**本地副本完好**且**当场 md5 双向核对**，已用直连 ssh（**不带 `-n`**）完整恢复并复核
（`b5c409c2746affc4a1e527b6a34a6914` / `47e192ae46d3b5ea9bee8c1400746f2e` 两侧逐位相同）。

**正确做法（二选一）**：
1. **改文件一律用 `rsync -a` 或直连 ssh**（不带 `-n`）：
   ```bash
   ssh -o ProxyCommand="$HOME/.sandbox/experiment/1107/sgmn-loop/.ops/scripts/tr.sh %h" \
       -i "$HOME/.ssh/Training.pem" ec2-user@trainer.loopybrainie.aws \
       'cat > ~/sgmn-loop/evidence/table.md' < evidence/table.md
   ```
2. 若坚持用 `run_on_trainer.sh`，**只用远端命令自带内容**（heredoc 在**远端**展开），
   **不要**从本地喂 stdin。

★ **纪律（本次是它救的）**：**每次改完远端文件，当场 `md5sum` 双向核对**。
本次正是「本地 b5c409 vs 远端 e0482…」的不一致才暴露了问题；
若只看「命令没报错」，会把两个**空文件**当成推送成功。

★ 另一条：`rsync -a` 对这两个文件曾**报告无需传输却不生效**（`<f` 项、两侧 md5 均不变）；
加 **`-I`（`--ignore-times`）**或**直连 ssh** 才可靠。⇒ **推送后必须 md5 核，不看 rsync 的退出码**。

### ★★★ 追加（Cycle 203，第二次复现）：`rsync` 对 `evidence/table.md` 与 `decisions/DECISIONS.md` **静默失效**（连 `--ignore-times` 也无效）

**现象（两次完全一致，可复现）**：用
`rsync -a --ignore-times --itemize-changes ... 本地文件 trainer:~/sgmn-loop/`
推送这两个文件时，rsync **不报错**、**不传输**，且 itemize 输出把方向标成 **`<f`（本应 `>f`）**：

```
<f......... DECISIONS.md
<f......... table.md
```

`<` 在 `--itemize-changes` 里表示「**收到**（发往本机）」⇒ rsync 认为自己在**拉取**。
**两侧 md5 均不变、两侧大小持续不同**（实测 本地 883047 vs 远端 875279），
所以命令"成功"了而文件**根本没到**。★ 同一次 `rsync` 里的 `COVERAGE-v24.md`、
`DIRECTION-REGISTER.md`、`revolve/DIR-108/*`、`dir108/*` **全部正常** ⇒ 只有这两个文件。

**已验证可行的做法**（Cycle 202 与 203 都用它救回来）：
```bash
ssh -o ProxyCommand="$HOME/.sandbox/experiment/1107/sgmn-loop/.ops/scripts/tr.sh %h" \
    -i "$HOME/.ssh/Training.pem" ec2-user@trainer.loopybrainie.aws \
    'cat > ~/sgmn-loop/evidence/table.md' < evidence/table.md
```
★ **必须直连 ssh**（**不带 `-n`**，见上一节）；推完**当场 md5 双向核对**。

**纪律（本 cycle 再次由它兜住）**：**任何推送到远端的文件，不看退出码，只看两侧 md5。**
对本项目而言，`evidence/table.md` 与 `decisions/DECISIONS.md` 这两个文件**默认假定 rsync 不可靠**，
一律走直连 ssh + md5 核对。

## ★★ `READ_LR` 是**按装置**的，不是一个常数（Cycle 205 核；补 `D-139` 勘误的适用范围）

| 装置 | `READ_LR` | 用于 |
|---|---|---|
| `dir90/R1_task90.py:47` | **`1e-2`** | `DIR-104`（`dir104/R1_task104.py` 经 `D90`）· `dir92` · `dir93` |
| `dir91/R2_task91.py:51` | **`1e-3`** | `DIR-105` 起的全部（`dir105/106/107/108` 经 `R91`） |

- 机械核：`dir108/out/premise.json` 记 `{"readlr_fact": 0.001, "argparse_lr_claimed_by_P105_1": 0.01}`。
- ⇒ `D-138`/`D-139` 的「`COVERAGE:969` 写 `Adam(lr=1e-2)` 而实测 `1e-3`」这条更正，
  **适用范围是 `DIR-105`（dir91 系）**，**不**适用于 `DIR-104`（dir90 系 —— 那里 `1e-2` 是对的）。
- ★ **纪律：引用 `READ_LR` 必须连装置一起引。** 单写「`READ_LR=1e-3`」而不说装置 = 埋一个反向漂移。
- ★ 同型风险还有 `N_TRAIN_KEYS`/`CHUNK`/`MOUNT` 等：本表只锁了 `READ_LR`，**其余尚未逐一核对**，
  引用前**先看该装置的源文件**，不要跨装置照抄。


## ★ bastion 的 DNS 对 `trainer.loopybrainie.aws` 会**间歇性**失败（Cycle 206 实测）

- **症状**：`channel 0: open failed: connect failed: Name or service not known` + `stdio forwarding failed`
  ⇒ 看起来像"远端挂了"，但**先别下结论**。
- **鉴别顺序**（`DEF-189-01` 的纪律要接着往下走一步）：
  1. 先测**无关外部 HTTPS**：`curl -sS -m 12 -o /dev/null -w '%{http_code}' https://github.com`
     —— 本 cycle 实测 **200** ⇒ **不是** FlClash 代理（`DEF-189-01` 不适用）。
  2. 再看报错文本：如果是 `Name or service not known`（**解析失败**），而不是 `UNEXPECTED_EOF`/kex 断
     ⇒ 问题在 **bastion 侧 DNS**，不在本机、不在代理。
- **★ 处置**：
  - **重试**（本 cycle 实测：连续失败 2 次后，隔几秒连试 6 次**全绿**）—— 这是**抖动**，不是故障。
  - **稳定兜底**：用 trainer 的 VPC 内网 IP 直连，绕开 bastion 的 DNS。
    `trainer` = **`10.0.140.43`**（主机名 `ip-10-0-140-43.ap-southeast-2.compute.internal`，Cycle 206 复核）。
    把 ssh 目标换成 IP 即可（`ProxyCommand` 里的目标也一并换成 IP）：
    ```bash
    ssh -o ProxyCommand="$PWD/.ops/scripts/tr.sh 10.0.140.43" -i ~/.ssh/Training.pem \
        -o StrictHostKeyChecking=accept-new ec2-user@10.0.140.43
    ```
  - **★ 纪律**：`rsync`/`ssh` 等**批量步骤要包一层重试**（3–4 次，间隔 4 s）；
    但**不要把会报错的 python 命令也包进重试** —— 本 cycle 我这么做了，
    结果同一个 `TypeError` 被**重复跑了 4 次**，白等 4 倍时间，还容易让人误以为是"不稳定"而不是"代码错"。
    **重试只用于传输，不用于逻辑。**
- **★ 还有一个坑**：`rsync` 之前**先** `ssh mkdir -p` 建目录；否则 `--ignore-times` 也可能因目标目录不存在而失败。

## ★ 正式轮跑完了但 **SSH 回传是空的** —— 先去查落盘，别急着重跑（Cycle 207 实测）

- **症状**：`run_on_trainer.sh '... | grep -vE ... | tail -48'` 返回 **完全空白**，
  看起来像"命令没跑/连接断了"。
- **真相**：远端 **36 格已全部跑完**，`out/stage_formal.json`（255847 B）与
  `out/stage_verdict.json` 在 **20:40** 就写好了。丢失的只是**回传的那一段 stdout**（bastion DNS 抖动）。
- **★ 鉴别顺序（照做，能省一整轮）**：
  1. **先看远端 `out/` 的 mtime 与文件大小** —— 有 `stage_verdict.json` 且时间是刚才 ⇒ **跑完了**。
  2. 再看 `cells` 计数是否等于 `plan_n`：
     `python -c "import json;d=json.load(open('out/stage_formal.json'));print(len(d['cells']))"`
  3. 计数齐 ⇒ **直接 `rsync` 拉回来**，不要在远端重跑（重跑=白烧 20 min）。
- **★ 为什么脚本能扛住**：`dump(...)` 是**每格之后**都写盘（`stage_formal.json` 增量落盘），
  且正式轮 `for ... if k in RES["cells"]: continue` 会**跳过已有格** ⇒ 中断可续跑。
  ⇒ **凡是要跑 >5 min 的正式轮，务必确认脚本是逐格 dump 的**；这次正是它救回来的。
- **★ 纪律**：`| tail -N` **只在回传成功时才可信**。判断"跑没跑"**永远看远端文件**，不看 stdout。

## ★★★ `DEF-105-02` 的**边界**：它是**设备/量特定**的，不是普遍律（Cycle 209 实测）

`DEF-105-02` 原文说「**对所有多格方向适用**」—— **这句话过强，已改**。

**Cycle 209 实测（`dir90` 系装置）**：同一 `(γ, arm, seed)` 格，**一格一进程**重跑 vs
`DIR-111` 的**整批同进程**读数，深度比对 `state` 全层全量 + `scalars` 全字段 +
`acc_item`/`acc_q`/`n_q`/`S`/`tgt`/`item_ok`/`gamma_final`/`eval_writes`/`finite_params`：

**2880 个字段逐一比较，不等 = 0**（8 格 × 4 臂 × 2 γ，`dir111` 与 `dir112/out/iso/` A/B）。

| 装置/量 | 同进程多格 vs 隔离 | 结论 |
|---|---|---|
| `dir91` 系 · `acc_all32`（`DIR-105`） | 0.7812 vs 0.8750 | **不可比**（`DEF-105-02` 成立） |
| `dir90` 系 · `state.rank_eff` / `state.cos_mean` / `S`（`DIR-111`） | 逐位相同 | **可比**（该缺陷不成立） |

⇒ **正确做法**：**每轮实测一次**（8 格 A/B 约 5 min），**不要照搬**结论；
更不要因为「本轮不需要」就跳过 —— 需要的时候已经晚了。

**怎么快速实测**：拿目标方向的主判据量，挑 **2 个端点配置 × 4 个臂 × 1 个 seed**，
各起一个独立进程重跑，与整批读数**逐字段**比对。
探针模板：`dir112/probe_isolation.py` + `dir112/run_iso112.sh`（照改 `--gamma/--arm/--seed`）。

**未解释（留档，不主张机理）**：为何本装置的 `acc_item`（离散项计数）也逐位相同，
而 `DIR-105` 的 `acc_all32` 会翻面？两者都是离散计数。差异可能在设备形状/步数
（600 步、896 维、48 items）。**不主张机理**，只报事实。

**另两条相关事实（Cycle 209）**：
- **`T-NONE`（未训练臂）的 `state` 逐位 γ-不变**（3 seed × 3 γ × 2 量全等），
  但 `kappa` **随 γ 线性变**（0.001022 → 0.010222 恰 10×）⇒ **γ 改增益、不改结构读数**。
  ⇒ 配对设计 `δ=(X_arm−X_NONE)/(1−X_NONE)` 的 `X_NONE` **每 seed 只需 1 格**（跨 γ 共用），省 25% 格。
  ★ 但 `acc_item` **不是** γ-不变的（seed 1125：0.2917 → 0.2708）⇒ 该不变性**只对 `state` 成立**。
- **`--stage formal` 会 `setdefault` 静默继承 Stage A 的格** ⇒ 同一"正式轮"的数据混了两种进程史，
  而**产物本身读不出来源**（`DIR-111`：γ=0.001 的 6 格来自 Stage A 进程）。
  本轮实测证明无后果，但**建议以后给 `cells` 加 `src_stage` 字段**（`DEF-112-03`）。

### ★★★ 「独立单位」纪律（Cycle 211 · `DIR-113` 前置检查一 + `DEF-112-04` 同族）

**凡是"一条记录里有多个子观测"的量，先问：这些子观测真的独立吗？独立单位是哪个？**

- 实例：`acc_item = mean(item_ok)` 有 **48 个 item**，看似可以做 48×12=576 对配对，
  但 **48 个 item 共享同一个训练好的 `W`** ⇒ 正确性通过 `W` 相关 ⇒
  **独立单位是「训练出来的那一格」(seed×arm)，不是 item**。
- 后果（实测）：把 item 当独立 ⇒ `p = 9.760e-06`；按 seed 聚类 ⇒ `p = 0.01305`
  ⇒ **虚高 1337×**。上轮 `DIR-112` 是 **74×** ⇒ **倍数不是常数** ⇒
  ★ **池化值必须整体禁用，不能"打个折再用"**。
- 上一轮的 `DEF-112-04` 是同一族的另一形式：`rank_eff` 与 `cos_mean` 是**镜像**
  （ρ=−1.0000），把两个量当两个检验就把 p 平方了一次。
- **操作规则**：
  1. 多量判据：**先算量间相关系数**；ρ 接近 ±1 ⇒ 只报一个，**不得**合成联合 p。
  2. 多子观测判据：**先定独立单位**（通常是"一次训练/一次运行"），按它算 n。
  3. 若两者都要报，**必须标注哪个进判据、哪个是报告块**，并在产物里写死这句话。
  4. **自检脚本要能变红**（`DEF-112-07`）：拿一个已知会变的量喂进去，确认它真的会报错。

### ★★★ 「共享装置带地雷」：`build_facts` 的留出答案值覆盖（Cycle 212 · `CORR-30` 复现）

**任何从 `dir90/R1_task90.py` 的 `D90.build_facts` 继承事实的方向，都继承同一个坏划分。**

- 机制：`VALUE_POOL` **27** 个词、`N_FACTS = 32` ⇒ 5 个值重复；`vals=[VALUE_POOL[i%27] …]` 后 `rng.shuffle(vals)`；
  划分 = `Random(777).shuffle(order)` 取前 16 为训练键。⇒ **留出 16 条的答案值里只有 1 条在训练集出现过**
  （`class_seen_frac_held = 1/16 = 0.0625`；12 个不同留出值从未作为训练答案出现）。
- ⇒ 受约束：**`D-125`/`D-129`/`D-130`/`D-136`/`D-143`/`D-144`**（全部跑在该划分上）。`D-133`/`D-135` 无留出划分，不受影响。
- ⇒ `D-121`（`dir93:SPLIT_CLASSES=16`）已把上界修到 `1.0000`，**但该修法从未回流到共享 `build_facts`**。
- ★★ **复算时必须复刻 rng 消耗顺序**：`build_facts` 里 `rng = Random(9091)` **先**被键生成吃掉（32 个 5 字符键），
  **之后**才 `shuffle(vals)`。漏掉键生成 ⇒ 算成 `3/16`（`COR-114-01` 的实际事故）。
  ⇒ **守卫必须从实际 `build_facts` 输出算，不许重实现**。
- **规则**：凡「留出集」类判据，开卡前先算 `class_seen_frac_held` 并写进卡片；
  若它 `< 0.95`，该判据的**绝对水平**受结构上界约束，**不得**用绝对阈值下结论（只能用划分内配对差）。

### ★★ 「注入正确内容」必须沿**输出**方向，不能用输入嵌入（`DEF-115-06`，Cycle 215 · `DIR-115`）

- **现象**：想让「记忆把**正确**答案注入进去」当**上界/正对照**时，直觉做法是取
  `model.get_input_embeddings().weight[t]` 当 `m` 注入（`dir104.install_ceil` 就是这么做的）。
- **实测**（`DIR-115`，`SPLIT` d=600，`γ=1e-3`）：`acc = 0.3541666666666667`
  在 `mult ∈ {0.25, 1, 4}` **三点完全相同**，且**恰等于关闭注入的先验 `17/48`**；
  而同格真实（学到的）注入是 `0.2708`。
- **真因**：transformer 的**输出**方向由 `lm_head`（unembedding）决定，**它不是输入嵌入的转置**
  ⇒ 沿 `emb[t]` 移动残差流**不抬高** token `t` 的 logit ⇒ 该向量在 argmax 上近似**中性**。
- **规则**：
  1. 凡要做「注入正确内容」的**正对照**，必须沿**输出方向**（`lm_head` 对正确 token 的那一行/列）
     或**直接加 logit bias**；**不得**用 `input_embeddings` 冒充。
  2. 若只能用输入嵌入，则该臂**只能**当**中性向量对照**（"同量级可无害"），
     **不得**当上界、**不得**据此说「连正确内容也救不回来」。
  3. 见到「oracle 注入后准确率**恰等于先验**、且对强度**完全不敏感**」⇒ **先怀疑方向不对**，再怀疑装置。

### ★★★ 更正：Qwen2.5-0.5B-Instruct 的 `lm_head.weight` **就是**输入嵌入（Cycle 217 · `DIR-116`）

**上面那条 `DEF-115-06` 的「输入嵌入 ≠ 输出方向」在本装置上是【错的】，特此更正（旧条目保留不改）。**

- **实测**（`dir116/probe116_premise.py` → `dir116/out/premise116.json`）：
  `config.tie_word_embeddings = **True**`；`lm_head.weight.data_ptr() == embed.weight.data_ptr()` = **True**；
  `torch.equal(lm_head.weight, embed.weight)` = **True**；`cos(emb[t], lm_head[t])` = **1.0**（16 个值 token）；
  `‖emb[t]‖ = ‖lm_head[t]‖ = 0.425116`（形状均 `[151936, 896]`）。
- **⇒** 取 `lm_head` 对 token `t` 的那一行注入，与取 `input_embeddings[t]` **逐位相同**。
  `DIR-115` 的 ORACLE 臂**方向本来就是对的**，它"看不出效果"的真因是**幅度太小**：
  `Δlogit_t(mult=1) = 1.571e-2`，而 `logit_absmax ≈ 16.7`；`mult∈{0.25,1,4}` 全落在 `mult≤300 ⇒ 17/48` 的平台内。
- **规则（改写前一条的第 1、3 点）**：
  1. 想做「注入正确内容」的**正对照**，**先查 `tie_word_embeddings`**：
     - **绑定**（如 Qwen2.5-0.5B）⇒ `lm_head` 行 ≡ 输入嵌入，**方向已对**；此时问题一定是**幅度**，
       要做的是**扫倍率网格直到效果出现**（本装置转变点在 `mult ∈ (300, 1e4)`）或改用**直接 logit bias**。
     - **未绑定** ⇒ 才需要区分输入嵌入与输出行。
  2. **识别信号**：若"oracle 注入"的准确率**恰等于先验**且对小幅**不敏感** ⇒ **先怀疑幅度，不要先怀疑方向**。
     ★ 真正的方向错，症状是"**越大越差**"，不是"**始终不动**"。
  3. ★ **注入→logit 的转移不是 `‖emb‖²` 的线性外推**：实测比闭式大 **29×**，且**强次线性**
     （`mult×100 → Δlogit×4.93`），根因是 **final RMSNorm 会重标定 `‖h‖`**。
     ⇒ 估增益必须**实测**，且要扫到 `Δlogit ≳ 典型 logit 间隙`的量级才有判别力。


---

## 9. 实验归档：git 远端与分支（Cycle 230 起生效）

**远端**：`origin = https://github.com/LoopyBrainie/SGMN.git`（**PUBLIC** 仓库，`gh repo view` 可复核）。
**分支**：每个实验轮次一条 `experiment/<开始日期>`，本轮 **`experiment/1107`**。
**已完成**：`experiment/1107` 基于远端 `main`（`acb79cf`）建立，首提交 `e41b508`（2510 文件），已 `push -u`；
`refs/heads/main` **未动**（仍 `acb79cf`）。

### 9.1 每个 cycle 收尾必须做（用户 2026-11-22 指令：从下一轮开始直接 commit）
```bash
cd /home/brra1n0/.sandbox/experiment/1107
git add -A
git commit -m "<cycle N>: <一句话>"        # 只提交，不碰 main
GIT_TERMINAL_PROMPT=0 git -c credential.helper='!gh auth git-credential' push
```
★ **凭据**：`gh` 已登录（`LoopyBrainie`，scopes 含 `repo`），但**未配置 credential.helper**
⇒ 必须带上上面那行 `-c credential.helper='!gh auth git-credential'`，否则会**卡在询问用户名**。
（**不要**为此改全局 git 配置；用一次性 `-c` 即可。）
★ **绝不** `git push origin main`、**绝不** 在 `experiment/*` 上 merge main —— 分支互相独立。

### 9.2 两条已知陷阱（都实测踩过）
1. **远端 `.gitignore` 是 GitHub 的 Python 模板，`*.log` 会静默丢掉全部实验日志**（实测 487 个）。
   已在 `.gitignore` 末尾加 `!*.log` 取消。**新增/覆盖 `.gitignore` 时必须保留该行。**
2. **`.pi/` 必须保持排除**：内含 `.pi/loop/inbox.md`（**与用户的私人对话原文**），
   而本仓库 **PUBLIC**。代价：云端 agent 只拿仓库**看不到** `task/inbox/handoff` ⇒ 无法直接续跑 loop
   （可见状态只有 `00_HANDOFF.md`）。**若要云端可续跑，须用户明确授权**。
3. 本地根提交 `37e7a10` 与远端 `main`（`acb79cf`）**历史无关**（两条孤儿根）⇒
   **不要**以 `37e7a10` 为基建分支（公开仓库会出现两个不相连的根）；以 `origin/main` 为基。

### 9.3 ★★ 追加型文件的同步纪律（因一次真实事故而写）
**事故**：上一 cycle 把「决定二百一十九」用 `ssh cat >>` 追加到**训练机**副本，
随后又用 `ssh cat >` 把**本地**副本（从未含该决定）整体覆盖过去 ⇒ **训练机上刚写入的决定被抹掉**，
两边都只剩 46 条；而当时的 **md5 双向核对报「✓ 一致」** —— 因为**两边都错**。
**教训**：**md5 只能证明「两份相同」，不能证明「内容正确」**。
**规则**：
- 「追加型编辑」只作用于**唯一权威副本**；禁止「先追加到远端、再整文件覆盖远端」。
- 覆盖式同步**之前**，必须证明 **本地 ⊇ 远端**（比对 `^### 决定` / `^\| \*\*D-` 这类**结构性计数与关键串**），
  不是只比 md5：
  ```bash
  comm -23 <(远端 heads | sort) <(本地 heads | sort)   # 必须为空
  ```
- 同步后**再**比 md5（此时 md5 才只用于「同步是否完成」）。
