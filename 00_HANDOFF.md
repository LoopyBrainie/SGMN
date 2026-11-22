# 00_HANDOFF · SGMN 持久 loop（接管点）

> ⚠️ **这是 2026-11-07 的带日期快照**（DIR-21 闭合时写下的接管点），**不是当前状态**。
> 当前状态以这些**活文件**为准：`.pi/loop/{task,inbox,handoff}.md` · `sgmn-loop/SUMMARY.md`（SUCCEEDED 计数）· `sgmn-loop/evidence/table.md`（证据行）· `sgmn-loop/DIRECTION-REGISTER.md`（方向队列）。
> 最新闭合：**DIR-26-N**（2026-11-09，COMPLETE / **GATE-ONLY**，H26N 未确认 ⇒ `SUCCEEDED` **21/21** 不变）。

## 当前状态（2026-11-07 · DIR-21 已闭合）

- **`SUCCEEDED = 15/15`**（DIR-21 为 **COMPLETE / FALSIFIED**，按机制 §3.4 **不计入 +1**）
- **21 个方向已闭合**；证据表 **22 行**（D-01…D-28，只增不改）
- **纪律 48 条**（1–48 连续）；六项一致性审计 **全部通过**

## DIR-21 的结论（一句话）

v2.4 §7.2「时间与情节状态」的**可测部分**：
**粒度是真实机制变量**（容量峰值 `K*` 从 4 篇移到 192 篇，**48×**）、**衰减只认写步数**（同 `m` 不同调度 0.0084；半衰期 544/512/448 写步，**1.21×**）；
但 **`η_S ≡ m·η_W` 被证伪**（0.0723 / 0.1840，两处越界；预注册证伪条款）；
`H21-1`（任务层）因卡片 §2.2/§2.3 **内部矛盾**（缺陷 D120）记「**未确认**」，两口径并列（0.376 / 1.444）。

## 交接物（md5）

| 项 | 路径 | md5 |
|---|---|---|
| 裁决书 | `revolve/DIR-21/VERDICT.md` | `0fbca2f7430b51e463a2a7ee211d31e2` |
| 卡片 rev 1.8 | `revolve/DIR-21/DIRECTION_rev21_v18.md` | `38b1a5cdb34da5df1b94750b7834b590` |
| 脚本 | `revolve/DIR-21/R1_granularity.py` | `08b68ecb8859e8bda30911beb4ced43f` |
| 装置 v7 | `revolve/DIR-21/sgmn_module_hebb_v7.py` | `1a242d02c530307f6c76646ca13469ef` |
| 正式轮日志 | `dir21/D21_R2_formal_r18.log` | `0261d97393d18dc156f3ca7e3850a796` |
| 结果 | `R21_results.json` | `37b6cef54d9e7193502bb3e89848479c` |
| 增量 | `.ops/formal_incremental_DIR21r18_full.json` | `f83a4984df31105982dc6f10c9a72064` |
| 全包 | `SGMN_LOOP_full.zip` = `LOOP_REPORT_full.zip` | `3460da01204082900c9b85a741fe3edd` |
| DIR-21 产物包 | `DIR-21_artifacts.zip` | `2346378a9ac1b844bc6fc649fca17eb9` |
| DIR-22 卡片 rev 1.0 | `revolve/DIR-22/DIRECTION_rev22.md` | `f66bd0dee286cae764bd08f5b4c19bde` |

## 下一轮的第一件事

**DIR-22 · rev 1.1 冻结数值阈值**（卡片 rev 1.0 已冻结语义与问题，md5 见上）：
主假说 **H22** = 「单一标量化约 `η_S = f(m)·η_W` **在结构上不存在**」，
可证伪形态 = 两条等值线在 `(ln η, ln m)` 平面上**正交**（衰减线 `k_t = 0`，内容线 `k_cos ≠ 0`）。
⇒ rev 1.1 必须写死：`k_cos`/`k_t` 的可区分性判据、`cos_1` 与闭式预言的允许偏离、等值线斜率容差、种子表、`K` 网格。
⇒ 然后才是装置 v8（只加 `refresh_per_episode`）+ 脚本 `R1_rate.py` + Stage A（继承 58 + A22-1…A22-6）+ smoke + **8×5 二维网格**正式轮。

## 禁止事项（沿用）

- 本地不跑实验 Python（本地只放文本/日志/结果）；一切执行走 `trainer` / `/opt/pytorch/bin/python`。
- 不改已冻结判据的**阈值与文字**；改动 ⇒ 新开 revision（全部留档）。
- 不复用 DIR-21 的 r18 数据回答 DIR-22 的问题（它只有 `f(m)∈{1,1/17}` 两点）。
- 失败轮日志不删（`dir21/` 现有 15 份，含 r17 作废轮）。
