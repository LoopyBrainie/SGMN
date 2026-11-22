#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-42 · **续跑机制自检**（`.jsonl` 增量落盘 + 指纹作废 + 合并优先）

★ **这不是实验**：不导入装置（`dir31`/`dir33`）、不训练、不产生任何实验读数。
  它把 `R1_task42.py` 里 `_row_cfg` / `_append_jsonl` / `_read_jsonl` / `_jsonl_path`
  的**源码本体**抽出来，用**合成行**验证「正式轮断点续跑」这条**保护 300 000 条轨迹算力**的机制。

⇒ ⇒ 为什么必须自检：续跑机制**只在网络中断/作业被抢占时才会被触发** ⇒ ⇒
  那正是**最不该在半夜第一次跑**的路径 ⇒ ⇒ 必须在开跑前用合成数据把它跑通。

用法：  python3 dir42/verify_resume42.py
退出码：0 = 全部通过；1 = 有失败。
"""
import ast
import json
import os
import pathlib
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "R1_task42.py")

NEED = {"_row_cfg", "_append_jsonl", "_read_jsonl", "_jsonl_path"}


def load_plumbing():
    """抽出**源码本体**（不是复制品）⇒ 测的就是真正会跑的代码"""
    src = open(SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    segs = [ast.get_source_segment(src, n) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in NEED]
    if len(segs) != len(NEED):
        raise SystemExit(f"★ 在 {SRC} 里只找到 {len(segs)}/{len(NEED)} 个目标函数")
    out = tempfile.mkdtemp()
    ns = {"os": os, "json": json, "pathlib": pathlib, "OUT": out,
          "STEPS": 250, "LR": 0.03, "N_REP": 16, "SEED0": 1000,
          "TRAIN_WIN": (1, 8), "EVAL_WIN": (9, 15),
          "INST_OFF": list(range(22500, 22500 + 400)), "ROW_SCHEMA": 1}
    exec("\n\n".join(segs), ns)
    return ns, out


def main():
    fails = []

    def check(cond, msg):
        print(("  ✓ " if cond else "  ✗ ") + msg)
        if not cond:
            fails.append(msg)

    ns, out = load_plumbing()
    row_cfg, append, read, jpath = (ns["_row_cfg"], ns["_append_jsonl"],
                                     ns["_read_jsonl"], ns["_jsonl_path"])

    def mk(idx, **over):
        r = {"idx": idx, "seed": ns["SEED0"] + ns["INST_OFF"][idx],
             "L": 1.0 + idx, "L_seen": 2.0, "L_full": 3.0, "cos": 0.5,
             "cfg": row_cfg()}
        r.update(over)
        return r

    print("① 追加 → 读取：往返无损、顺序保持")
    p = jpath("learned", 0)
    for i in (3, 7, 11):
        append(p, mk(i))
    rows = read(p)
    check([r["idx"] for r in rows] == [3, 7, 11],
          f"读回 3 行且顺序一致：{[r['idx'] for r in rows]}")

    print("\n② ★ 末行被截断（追加写被打断）⇒ 只丢该实例，**不丢整片**")
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"idx": 15, "seed": 9999, "L": 1.0, "cfg": {"ste')   # 半个 JSON
    rows2 = read(p)
    check([r["idx"] for r in rows2] == [3, 7, 11],
          f"截断行被跳过，其余 3 行仍在：{[r['idx'] for r in rows2]}")

    print("\n③ ★ 配置指纹：改任一配置维度 ⇒ 旧行**全部作废**")
    for dim, val, label in (("STEPS", 500, "steps"),
                            ("LR", 0.05, "lr"),
                            ("N_REP", 32, "n_rep"),
                            ("TRAIN_WIN", (1, 6), "train_win"),
                            ("EVAL_WIN", (9, 14), "eval_win")):
        old = ns[dim]
        ns[dim] = val
        got = read(p)
        check(got == [], f"改 `{label}` ⇒ 旧行作废（读到 {len(got)} 行）")
        ns[dim] = old

    print("\n④ ★ schema 版本：改行结构 ⇒ 旧行**自动作废**（★ 只靠配置盖不住）")
    ns["ROW_SCHEMA"] = 2
    got = read(p)
    check(got == [], f"改 `ROW_SCHEMA` ⇒ 旧行作废（读到 {len(got)} 行）")
    ns["ROW_SCHEMA"] = 1
    check(len(read(p)) == 3, "改回 ⇒ 旧行**重新有效**（指纹是等式，不是单向锁）")

    print("\n⑤ ★ 种子基址：改 `SEED0` ⇒ 旧行作废（同 `idx` 的种子已不同）")
    old = ns["SEED0"]
    ns["SEED0"] = 2000
    check(read(p) == [], "改 `SEED0` ⇒ 旧行作废")
    ns["SEED0"] = old

    print("\n⑥ ★ 并发追加**原子性**（8 线程 × 50 行 ⇒ 行不串、不丢）")
    p2 = jpath("rand", 1)
    errs = []

    def worker(tid):
        try:
            for k in range(50):
                append(p2, mk(k, L=float(tid), thread=tid))
        except Exception as e:      # pragma: no cover
            errs.append(e)

    ts = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    check(not errs, f"8 线程追加无异常（errs={len(errs)}）")
    got = read(p2)
    check(len(got) == 400, f"400 行**全部完整**（读到 {len(got)}）")
    check(all("L" in r and "idx" in r for r in got), "没有串行/截断的行（每行都是完整 JSON）")

    print("\n⑦ ★ 结构断言：续跑逻辑真的挂在 `formal_arm` 上")
    src = open(SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    fa = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "formal_arm")
    fa_src = ast.get_source_segment(src, fa)
    check("done = {r[\"idx\"]: r for r in _read_jsonl(" in fa_src,
          "`formal_arm` 用 `_read_jsonl` 建 `done` 映射")
    check("if si in done:" in fa_src and "continue" in fa_src,
          "`formal_arm` 命中已算实例就 `continue`（⇒ 只重算未落盘的）")
    check("_append_jsonl(jpath, row)" in fa_src,
          "`formal_arm` 每算完一个实例就追加落盘")
    mg = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "merge")
    mg_src = ast.get_source_segment(src, mg)
    check("_read_jsonl(_jsonl_path(arm, sh))" in mg_src,
          "`merge` 读 `.jsonl` 增量行（⇒ 断网不丢算力）")
    check("get(\"rows\", [])" in mg_src and "cfg" in mg_src,
          "`merge` 对**分片 JSON** 的行也过指纹（⇒ 旧配置不得混入）")

    print("\n⑧ 原子写实现：单次 `os.write` + `O_APPEND`")
    ap = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_append_jsonl")
    ap_src = ast.get_source_segment(src, ap)
    check(ap_src.count("os.write(") == 1, f"`_append_jsonl` 只有一次 `os.write`（实测 {ap_src.count('os.write(')}）")
    check("os.O_APPEND" in ap_src, "`_append_jsonl` 用 `O_APPEND`")

    print("\n★ 续跑机制自检 " + ("全部通过" if not fails else "**有失败**"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
