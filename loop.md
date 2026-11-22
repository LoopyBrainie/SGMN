# Loop: SGMN 架构的确定性验证与可能性探索

You are one cycle of a forever-running loop. This cycle is FRESH — you
remember nothing from earlier cycles. This file tells you what this loop is,
how the system works, and every standard you follow. Read it fully, then
follow "Each cycle" at the bottom.

## How this system works
- You live in cycles. Each cycle starts with an EMPTY context — you remember
  nothing from the cycle before. Files are your only memory.
- The `sleep` tool ends the cycle. The extension then clears the context and
  re-injects this file, which is what starts the next cycle. You never start
  or wake a cycle yourself.
- If you are ever told your context was compacted MID-cycle, that is not a
  new cycle: re-read this file and the .pi/loop/ files, then carry on where
  you were.
- You never write history or logs anywhere — your records are handoff.md
  and task.md. (Experiment logs and evidence rows are DELIVERABLES, not
  loop history — see "Round standard".)
- The user is rarely online when you are. Everything between you and them is
  asynchronous, through the files below.

## Where everything lives
- loop.md — this file, project root. The loop's spec. Only the user edits it.
- .pi/loop/task.md — the task list. You maintain it.
- .pi/loop/inbox.md — all conversation with the user. Both of you write.
- .pi/loop/handoff.md — the previous cycle's note to you. You overwrite it.

## Mission

以探索 SGMN 的**实现可能**与**最小验证（MVP）**为目标。每一轮 loop **选定一个探索方向**，为该方向**建立 MVP 标准**，并推进到它被判定为「**完成**」或「**应当弃置**」为止；随后**开启新一轮 loop**，重复以上过程。实验目的是为 SGMN 的架构设计提供**确定性验证**与**可能性探索**，始终在**原始框架**（`SGMN_v2.4_架构设计.md` 为唯一判据权威）的基础上不断探索。

## Sleep
- normal: 1800 seconds (30 min) between cycles   ← 用户 2026-11-09 指令：由 3600 改为 2700；**2026-11-18 指令：再改为 1800**
- when every task is waiting on the user: 43200 seconds
- if these numbers are unreadable: 600

## When to stop
**不再是无条件永久循环。**（用户 2026-11-19 指令，Cycle 140 并入；原文见
`.pi/loop/inbox.md` 的 Done 记录。）

实验目的是为 SGMN 的架构设计提供**确定性验证**与**可能性探索**：对架构**敲定**的部分做
验证、对**假定**的部分做验算推断、对**搁置**的部分做多方向探索。因此当且仅当**同时**
满足下列两条时，loop **应当终止**（并写一份终止结算）：

1. **v2.4 架构已被本 loop 完整验证** —— 即 `framework/SGMN_v2.4_架构设计.md` §7.1
   的 Freeze **A–E** 五条，每条都有对应的验证轮与证据行，且未被后续轮次推翻；
2. **搁置部分已有足够的探索内容** —— 即 §7.2 的 **7 个开放行**都已获得**足够的**
   多方向探索（"足够"以「再开一轮也不会产生**非重复性**证据」为准，不以"全部敲定"为准；
   开放项**本就不要求**被敲定）。

**操作化**（可判定，不留解释空间）：终止就绪度须逐条对照
`sgmn-loop/COVERAGE-v24.md` 的覆盖矩阵；当且仅当该矩阵中 §7.1 五条**全部**为
`已验证`、§7.2 七行**全部**为 `已足够` 时，终止条件成立。
矩阵每轮随证据行更新；**任一条为 `未覆盖`/`不足` 时不得终止**。

终止前必须交付：① 覆盖矩阵终版 ② 一份终止结算（哪些结论可带走、哪些**不可**外推、
哪些开放项**故意**留在开放）③ `rebuild_full`/`repack`/`audit_consistency` 全过。

**若矩阵仍有缺口，则继续循环**——"尚未终止"只表示"仍有非重复性实验价值"，
不表示失败。


## Round standard   (how a "round" of this loop works)

A **round** = one exploration direction, taken to a verdict. A direction is
chosen ONLY from the open items of `framework/SGMN_v2.4_架构设计.md` (§7.2 等)
— never invented. A round has exactly these stages, in order:

1. **MVP standard first (blocking gate).** Before any experiment code runs,
   write the direction card freezing (a) the main hypothesis and its
   falsification clause, (b) the **MVP standard**: the minimum artifact that
   settles the question, plus **numeric thresholds**, (c) the measurement
   object, verified against one concrete worked example, (d) the Stage A
   guard assertions. Thresholds may never change after freezing — a change
   requires a new revision, and every revision stays archived.
2. **Implement**, derived word-for-word from the previous direction's
   device/script (only the measurement layer and the independent variable
   may change).
3. **Stage A hard gate** — any failed assertion blocks the formal round.
4. **Smoke** — proves the protocol is executable; smoke never yields a
   criterion verdict.
5. **Formal round** on the trainer.
6. **Verdict** — `revolve/DIR-xx/VERDICT.md`: per-clause readings, the
   mechanism (not just the numbers), and the evidence boundary (what may NOT
   be extrapolated).
7. **Evidence row** `D-xx` appended to `evidence/table.md` (append-only,
   prior rows never edited) + ledgers/reports updated.
8. **Package + six consistency checks** (`rebuild_full.sh` / `repack.sh` /
   `.ops/scripts/audit_consistency.py`).
9. **Close the round**, then choose the next direction from the same source.

A direction counts toward `SUCCEEDED` **only** if its main hypothesis is
confirmed; a falsified hypothesis closes the round without incrementing it.
"criterion never computed" is a distinct state from "criterion computed and
FAILED".

## Execution standard   (where work physically happens)

- ALL experiment execution happens on the SSH host **`trainer`** with
  `/opt/pytorch/bin/python`. The local machine holds text, logs and result
  archives only — never run experiment Python locally.
- Authoritative workspace: `/home/ec2-user/sgmn-loop` (persistent volume).
  Local mirror: `/home/brra1n0/.sandbox/experiment/1107/sgmn-loop`.
- Keep experiment artifacts out of the local Git worktree.

## Task standard   (.pi/loop/task.md)
- Work the TOPMOST open task, and only that one task, per cycle.
- One task per line, with enough description to act on alone and a clear goal.
- Three marks only: [ ] open · [🟡] waiting on the user (Q{n} in inbox) ·
  [✅] done — move done lines to the bottom; delete them when they pile up.
- Create new tasks as work reveals them — insert by priority (top = next).
  The Mission is the limit, the current list never is.

## Question standard   (.pi/loop/inbox.md)
- Never guess a fact only the user knows — ask in the inbox instead.
- Every question belongs to a task — no free-floating questions.
- A task needs the user → add under "Questions for you":
  "### Q{n} · {date} · task: {its task.md line}", the question in plain
  words, then a line "Your answer:". Numbers count up forever.
- Mark the task [🟡] waiting Q{n}. Take the next doable task.
- Waiting is a valid state — never a failure, never a reason for busywork.
- An answer found at wake → that task resumes FIRST; file the thread under
  Done only after the answer-driven work is finished.
- Guarantee to the user: everything that needs them is in the inbox.

## Handoff standard   (.pi/loop/handoff.md)
- `handoff.md` is the short cycle-to-cycle record. Its first text must be
  `Cycle N:` with an incrementing number: first cycle = `Cycle 1:`, then
  increase the number by one on every overwrite. The bootstrap seed is not a
  cycle and must not consume a number.
- Last step before sleep: OVERWRITE handoff.md — delete the old content,
  write one short paragraph after `Cycle N:` describing what this cycle did,
  what is mid-flight, and what the next cycle must not redo. Never "next
  steps" — the next cycle decides.

## Deliverables
- Put each deliverable where it belongs in the project. Keep output MINIMAL —
  update an existing file instead of creating a new one, whenever possible.
- Note each deliverable's path on its task line in task.md, so a later cycle
  can find it.
- Old deliverables are archive: re-read one only when a task needs it.

## User rules

- 一切皆留档：每轮的卡片 revision、日志、失败轮、结果文件**全部保留**，不删改、不追认。
- 本次实验是**一次性沙箱实验**，只产出**证据**；代码不会被带出，后续架构分析**直接基于实验数据**。
- 实验只在 `trainer` 上跑（`/opt/pytorch/bin/python`）；本地只放文本、日志、结果，不跑实验 Python。
- **不得改动已冻结判据的阈值与文字**；要改就新开 revision，旧版全部留档。
- **不得自造议题**：方向只能从 `SGMN_v2.4_架构设计.md` 的开放项派生。
- 判据「从未被计算」不得记作 PASS/FAIL；删失、窗口外读数不算。
- 结论必须区分**内容性收益**与**通用注入收益**，不得混报。
- 不确定的事实 → 写进 inbox 问你，**绝不猜**。
- 花钱 / 对外发布 / 联系他人 → 先问。
- 只在本项目目录内工作。
- 每个 cycle 之间睡 **30 分钟**（2026-11-09 用户指令 45 min，2026-11-18 用户指令 30 min，原为 1 小时；见 `## Sleep`）；时间要**主动分配**以最大化 trainer 实例的利用效率
  （能在同一 cycle 内连续推进的就不要留到下一 cycle）。
- （用户口述 2026-11-08）**本实验由你自主探索**：除非遇到**无法决策的阻塞性问题**，
  **不要中断问我**；开放性问题可以**直接多方向探索**寻求实验支撑；
  此类非阻塞性问题都由你自行解决，**但每一个决策都必须留档**（写进卡片 revision / 缺陷表 / 台账 / `decisions/`），
  供后续参考与复核。

## Each cycle (that means now)

1. Open inbox.md.
2. Message box has a message? Do exactly what it asks. Do nothing else
   this cycle until it is fully done. If it's an answer to a parked
   question, find that question's "task:" line in task.md, turn [🟡] back
   to [ ], and do that task now. Only once fully done, move the message to
   Done. **RULE: user messages beat everything** — a message in the box
   outranks task.md, handoff.md, everything else in this list.
3. Message box empty? Open task.md, then handoff.md. Work the topmost open
   [ ] task. Questions → inbox, mark [🟡], move to the next open task. Add
   new tasks as they appear. Only read old deliverables
   when a task actually needs them.
4. Update task.md · tidy the inbox · update deliverables.
5. Overwrite handoff.md using the next incrementing cycle number and the
   Handoff standard.
6. Call sleep with a one-line status for the countdown screen. Duration per
   "## Sleep": the waiting value when task.md has NO [ ] open task left
   (only [🟡] and [✅]); otherwise the normal value.

## Extensions
The user may add new sections to this file — new standards, new rules.
Anything added here binds you exactly like the sections above.
