#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-106 · `§7.2-7b` 的**样本量轴**：共享 `W_o: y→m` 的留出泛化是样本效率还是结构性不可泛化？

卡片：`revolve/DIR-106/DIRECTION_rev106_v11.md`（rev 1.1 冻结）
任务：`.pi/loop/task.md` 的「终止唯一硬缺口 `§7.2-7b`」

★ 派生（逐字）：
  - 装置骨架 / `probe91_womap.py`（`D-91`）：`_read_patched`（可替换 q）、`set_oracle`、
    `train_read_oracle`、`eval_keys`、`snapshot`、`top1`
  - white 键臂 / `dir100/R1_task100.py`（`D-98`→`D-128`）：`build_tf("white", …)`、
    `_write_tf`（写侧过 `key_tf`）、`key_matrices_arm`（K 已过 `key_tf`）
  - 底座 / `dir91/R2_task91.py`：`N_FACTS=32`、`N_CAND=4`、`READ_STEPS=600`、`READ_LR=1e-3`、
    `write_phase`、`make_stack`、`make_eval_items`、`acc_by_question`

★ 本卡**唯一**改动（卡片 §3.3）：
  1. `N_TRAIN_KEYS` 由常量 → 自变量 `n_train ∈ {16,20,24,28}`（外加只作可学性上界的 `n=32`）
  2. `order` 由 `(seed, split)` 派生，**与 `n_train` 无关**（守卫 `A106-8`）
  3. 新增 `--stage merge`：一格一进程产物合成（`DEF-105-02` 强制）

★ 全部臂都用 **oracle `q = k_i`**（卡片 §3.1，`D-130` 已证查询不是瓶颈）⇒ 不需要 `q_tf`。
"""
import argparse, json, math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dir91", "dir100"):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import numpy as np
import torch

import sgmn_module
from sgmn_module import LayerMem, sg
import R2_task91 as R91
import R1_task100 as D100
import probe91_womap as W91
from R2_task91 import (DEV, N_CAND, N_FACTS, build_chunks, build_facts,
                       key_matrices, make_stack, model_and_tok, write_phase)

# ★ `R2_task91.OUT` 指向 **dir91/out/**；本卡必须落自己的目录（`DEF-106-06`）。
#   原稿直接沿用 `OUT` ⇒ 全部格子写进 dir91/out/，会污染另一方向且互相覆盖。
from pathlib import Path as _Path
OUT = _Path(__file__).resolve().parent / "out"

CARD = "DIR-106 rev 1.3"
# ---- 阈值全部沿用 D-91 已冻结值（本卡不新造任何阈值，卡片 §2.1）
CHANCE = 1.0 / N_CAND          # 0.25
GATE_FIT = 0.50
GATE_GEN = 0.50
N_TRAIN_SET = (16, 20, 24, 28)  # 主扫描（M106-1/M106-2 的定义域）
N_FULL = 32                     # 只作 A106-2 可学性上界臂
N_SPLIT = 4                     # 每 (n, seed) 的独立划分
SEEDS = (20261028, 7777, 60013)  # rev 1.1 冻结；60013 实测零碰撞（DEF-106-03）

# ★ 合并后的**完备性**期望（`DEF-106-07`）：`A106-9` 的 64/48/32/16 是**单个划分**的题数，
#   合并 4 划分 × 3 seed 后是 4×3×(32-n) = 192/144/96/48。用单划分数字做完备性检查会
#   把**正确**数据判成缺格。
EXPECT_NQ_MAIN = {16: 4 * 3 * (N_FACTS - 16), 20: 4 * 3 * (N_FACTS - 20),
                  24: 4 * 3 * (N_FACTS - 24), 28: 4 * 3 * (N_FACTS - 28)}
EXPECT_CELLS = {"main": 4 * 4 * 3, "shuf": 4 * 1 * 3, "w0": 4 * 1 * 3}

RES = {"card": CARD, "stage": None, "cells": [], "guards": {},
       "chance": CHANCE, "gate_fit": GATE_FIT, "gate_gen": GATE_GEN,
       "n_train_set": list(N_TRAIN_SET), "n_split": N_SPLIT, "seeds": list(SEEDS)}
FAILS = []


def base_fingerprint(model):
    """基座全部参数+buffer 的 sha1（截 16 位）。`A106-7`／`DEF-105-02` 的纪律：\
    区分「代码层状态泄漏」与「纯数值混沌」——没有这条守卫就无法区分。"""
    import hashlib
    h = hashlib.sha1()
    for n, t in sorted(model.named_parameters()):
        h.update(n.encode()); h.update(t.detach().float().cpu().numpy().tobytes())
    for n, t in sorted(model.named_buffers()):
        h.update(n.encode()); h.update(t.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


ORIG_WRITE = None
ORIG_READ = None


def rec(cid, ok, detail, value=None):
    RES["guards"][cid] = {"ok": bool(ok), "detail": detail, "value": value}
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}", flush=True)
    if not ok:
        FAILS.append(cid)


def rec_only(cid, detail, value=None):
    RES["guards"][cid] = {"ok": None, "detail": detail, "value": value, "report_only": True}
    print(f"[----] {cid}: {detail}", flush=True)


# ---------------------------------------------------------------- 划分（与 n_train 无关）
def make_order(seed, split):
    """`(seed, split)` → 固定置换。★ 与 `n_train` **无关**（守卫 `A106-8`）。

    `DEF-106-02`：若每格各自重新置换，曲线会混合「划分不同」与「样本量不同」两个变化。
    """
    order = list(range(N_FACTS))
    random.Random(seed * 100 + split).shuffle(order)
    return order


def install_white(st):
    """把 `key_tf` / `q_tf` 设为单位（`A106` 无 `q_tf` 需求），由调用方再填 white `key_tf`。"""
    for m in st.mems.values():
        m.key_tf = D100.IdentityTF()
        m.q_tf = D100.IdentityTF()


def build_white(st, Kraw, V, owner):
    """为每层建 white `key_tf` 并回传**白化后的 K**（`key_matrices_arm` 已过 `key_tf`）。"""
    TF = {}
    for li in st.mems:
        TF[li] = D100.build_tf("white", Kraw[li], V[li], li)
        st.mems[li].key_tf = TF[li]
        st.mems[li].q_tf = TF[li].qmap
    return TF


def _shuffled_labels(facts, tr, seed, split):
    """把训练区 `tr` 内的「事实 → 答案值」做一次置换（训练区外的原样保留）。

    只改**训练标签**，不动输入通路 ⇒ 与 `w0` 臂（抹记忆）互补：
    `w0` 问「记忆在不在承重」，`shuf` 问「`Wo` 读不读 `y`」。
    """
    import copy
    f2 = copy.deepcopy(facts)
    vals = [facts[i]["val"] for i in tr]
    perm = list(vals)
    random.Random(seed * 1000 + split + 7).shuffle(perm)
    for i, v in zip(tr, perm):
        f2[i]["val"] = v
    return f2


# ---------------------------------------------------------------- 单格
def run_cell(model, tok, facts, chunks, owner, seed, split, n_train, arm, stage):
    """一格 = `(arm, n_train, split, seed)`；返回该格读数。"""
    t0 = time.time()
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    out = {"arm": arm, "n_train": n_train, "split": split, "seed": seed, "stage": stage}
    try:
        # ---- 基座指纹（A106-7）
        base_fp = base_fingerprint(model)
        out["base_fp_before"] = base_fp
        # ---- 写阶段：white 键（写前先装 key_tf）
        install_white(st)
        LayerMem.write = D100._write_tf
        LayerMem.read = W91._read_patched
        write_phase(model, st, chunks)
        Kraw, V = key_matrices(model, st, facts, chunks, owner)
        TF = build_white(st, Kraw, V, None)
        # ★ 白化后的 K（用于 oracle q）
        K = {li: TF[li](Kraw[li]).detach() for li in TF}
        base = W91.snapshot(st)

        order = make_order(seed, split)
        tr = order[:n_train]
        he = order[n_train:]
        out["n_train"] = n_train
        out["n_held"] = len(he)
        out["order_head"] = order[:8]

        # ---- 守卫 A106-3：white 键两两 |cos|（只记录 + 断言）
        Kn = K[list(K)[0]]
        Kn = Kn / Kn.norm(dim=1, keepdim=True).clamp_min(1e-12)
        C = (Kn @ Kn.T).abs()
        mc = float((C - torch.eye(C.shape[0], device=C.device)).max())
        out["key_maxcos"] = mc

        cls, ncls = W91.class_ids(facts)
        out["n_class"] = ncls
        out["class_seen_frac_held"] = (float(np.mean([cls[i] in {cls[j] for j in tr} for i in he]))
                                       if he else None)

        # ---- 复原同初值（所有臂同初值）
        for li, m in st.mems.items():
            with torch.no_grad():
                m.Wfull.copy_(base[li]["W"]); m.Wq.copy_(base[li]["Wq"])
                m.Wo.copy_(base[li]["Wo"]); m.gamma.copy_(base[li]["gamma"])

        # ---- 对照臂对 W 的处理
        if arm == "w0":
            # W ← W⁰（抹掉全部写入；W⁰ 由 make_stack 记录在 m.W0 体系内 ⇒ 置 U,V 为 0）
            for li, m in st.mems.items():
                with torch.no_grad():
                    m.U.zero_(); m.V.zero_()
        elif arm == "wother":
            for li, m in st.mems.items():
                with torch.no_grad():
                    m.V.copy_(torch.roll(m.V, shifts=1, dims=1))

        # ---- 训练 `Wo`（oracle q）；label-shuffle 臂把标签打乱
        rng = random.Random(1000 + seed % 100)
        # ★ `arm="shuf"` = 零假设臂（`A106-5`）：训练时把「事实→正确答案」的对应**置换掉**，
        #   而 `q`/`y` 通路完全不变 ⇒ 若该臂留出准确率仍高，说明 `Wo` 在背标签而非读 `y`。
        facts_use = facts
        if arm == "shuf":
            facts_use = _shuffled_labels(facts, tr, seed, split)
        losses, viol, div = W91.train_read_oracle(
            model, st, tok, facts_use, tr, rng, K, "readout")
        out["violations"] = int(viol)
        out["diverged"] = bool(div)
        out["final_loss"] = float(np.mean(losses[-50:])) if losses else None

        # ---- 评测：train / held 用**同一份权重**
        fit = W91.eval_keys(model, st, tok, facts, tr, K, True, "readout")
        out["acc_train"] = fit["acc_q"]
        out["margin_train"] = fit["margin_mean"]
        out["acc_item_train"] = fit["acc_item"]
        if he:
            gen = W91.eval_keys(model, st, tok, facts, he, K, True, "readout")
            out["acc_held"] = gen["acc_q"]
            out["margin_held"] = gen["margin_mean"]
            out["acc_item_held"] = gen["acc_item"]
            out["n_held"] = int(gen.get("n_q", len(he)))
        else:
            # ★ A106-12：he 为空 ⇒ 不得产生 acc_held 数值
            out["acc_held"] = None
            out["margin_held"] = None
            out["acc_item_held"] = None
            out["n_held"] = 0

        # ---- W 漂移（A106-4）+ 基座指纹（A106-7）
        # ★ 只比 `W`：`Wq`/`gamma` 是**允许**被读训练更新的参数（`v2.4 §2.2` 只冻结 `W`）。
        #   原稿用 `maxdiff_snap`（含 Wq/Wo/gamma）⇒ 恒为 0.115 的假 FAIL（`DEF-106-04`）。
        dW = 0.0
        for li, m in st.mems.items():
            from R2_task91 import W_of as _Wof
            dW = max(dW, float((_Wof(m) - base[li]["W"]).abs().max()))
        out["W_drift"] = dW
        out["base_fp_after"] = base_fingerprint(model)
        out["sec"] = round(time.time() - t0, 1)
    finally:
        if ORIG_WRITE is not None:
            LayerMem.write = ORIG_WRITE
        if ORIG_READ is not None:
            LayerMem.read = ORIG_READ
        try:
            W91.clear_oracle(st)
        except Exception:
            pass
        try:
            st.remove()
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stageA",
                    choices=["stageA", "smoke", "formal", "merge", "adjudicate"])
    ap.add_argument("--arm", default="main",
                    choices=["main", "w0", "shuf", "full"])
    ap.add_argument("--n-train", type=int, default=-1)
    ap.add_argument("--split", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--cells-dir", default="out/cells")
    ap.add_argument("--from", dest="from_json", default="")
    a = ap.parse_args()
    global ORIG_WRITE, ORIG_READ
    ORIG_WRITE = LayerMem.write
    ORIG_READ = LayerMem.read
    OUT.mkdir(parents=True, exist_ok=True)
    RES["stage"] = a.stage

    model, tok, facts, chunks, owner = _setup()
    print(f"[装置] 事实 {len(facts)}；块 {len(chunks)}；候选 {N_CAND}；chance={CHANCE}", flush=True)

    if a.stage == "stageA":
        stage_a(model, tok, facts, chunks, owner)
        _dump("stage_stageA")
        print(f"\n★ Stage A {'全部通过' if not FAILS else '有 FAIL：' + str(FAILS)}", flush=True)
        raise SystemExit(0 if not FAILS else 1)

    if a.stage == "merge":
        cells = _load_cells(a.cells_dir)
        RES["cells"] = cells
        _dump("stage_formal")
        verdict_main(a.from_json or str(OUT / "stage_formal.json"))
        return

    if a.stage == "adjudicate":
        verdict_main(a.from_json or str(OUT / "stage_formal.json"))
        return

    # ---- 单格（由 run_formal106.sh 一格一进程驱动）
    n_train = a.n_train if a.n_train > 0 else N_TRAIN_SET[0]
    split = a.split if a.split >= 0 else 0
    seed = a.seed if a.seed > 0 else SEEDS[0]
    cell = run_cell(model, tok, facts, chunks, owner, seed, split, n_train, a.arm, a.stage)
    RES["cells"] = [cell]
    print(f"[格] arm={cell['arm']} n={cell['n_train']} split={cell['split']} seed={cell['seed']} "
          f"acc_train={cell['acc_train']:.4f} acc_held={cell['acc_held']} "
          f"{cell['sec']}s", flush=True)
    _dump(f"cell_{cell['arm']}_{cell['n_train']}_{cell['split']}_{cell['seed']}")


def _setup():
    """★ 逐字沿用 `probe91_womap.py:503-505`：`build_chunks` 返回 `(chunks, owner)`。"""
    tok, model = model_and_tok()          # ★ 返回顺序是 (tok, model)（R2_task91:99）
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    return model, tok, facts, chunks, owner


def _dump(name):
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[完成] → {p}", flush=True)


def _load_cells(cd):
    import pathlib
    p = pathlib.Path(cd)
    fs = sorted(p.glob("*.json"))
    cells = []
    for f in fs:
        d = json.loads(f.read_text(encoding="utf-8"))
        cells.extend(d.get("cells", []))
    cells.sort(key=lambda c: (str(c["arm"]), c["n_train"], c["split"], c["seed"]))
    print(f"[merge] 载入 {len(cells)} 格（{p}）", flush=True)
    return cells


# ---------------------------------------------------------------- Stage A
def stage_a(model, tok, facts, chunks, owner):
    print("\n===== Stage A 守卫 =====\n", flush=True)

    # ---- A106-8：order 与 n_train 无关
    o1 = make_order(SEEDS[0], 0)
    o2 = make_order(SEEDS[0], 0)
    rec("A106-8", o1 == o2 and sorted(o1) == list(range(N_FACTS)),
        f"`order` 由 `(seed,split)` 决定、与 n_train 无关（构造上无 n_train 入参）；"
        f"同一 (seed,split) 两次生成逐位相同，且是 0..31 的置换")

    # ---- A106-9：题数账目
    acc = []
    for n in N_TRAIN_SET:
        nho = N_SPLIT * (N_FACTS - n)
        acc.append((n, nho))
    want = [(16, 64), (20, 48), (24, 32), (28, 16)]
    rec("A106-9", acc == want, f"池化留出题数 = {acc}（须 == {want}，且全 >0）", value={"table": acc})

    # ---- A106-11：手算 5 条（卡片 §2.4）
    e = []
    e.append(("chance", abs(CHANCE - 0.25) < 1e-12, CHANCE))
    e.append(("gate_gen", abs(GATE_GEN - 0.50) < 1e-12, GATE_GEN))
    # n=28 池化 16 题，acc=0.50 <=> 8/16
    e.append(("n28_pool", 0.50 * 16 == 8.0, 0.50 * 16))
    # SE 公式在 p=0.25, N=16
    se = math.sqrt(0.25 * 0.75 / 16)
    e.append(("se_16", abs(se - 0.10825317547305482) < 1e-9, se))
    e.append(("chance+2se", abs((0.25 + 2 * se) - 0.46650635094610965) < 1e-9, 0.25 + 2 * se))
    ok = all(x[1] for x in e)
    rec("A106-11", ok, f"手算 5 条：{[(k, v) for k, _, v in e]}")

    # ---- A106-10：不得引用 D-106 论证 7b
    rec("A106-10", True, "本卡论证不引用 `D-106`（`D-103` 已裁定其与 7b 无关）；"
                         "全文仅在本守卫中提及该编号")

    # ---- A106-2：可学性上界（★ rev 1.2：锚在**最易**点 n=16，阈值 0.90 不变；`DEF-106-05`）
    print("\n--- A106-2 可学性上界臂（n_train=16）---", flush=True)
    cell = run_cell(model, tok, facts, chunks, owner, SEEDS[0], 0, 16, "full", "stageA")
    rec("A106-2", float(cell["acc_train"]) >= 0.90,
        f"n_train=16 全拟合 `acc_train` = {cell['acc_train']:.4f}（须 ≥0.90；`D-130` = 0.9375）",
        value={"acc_train": cell["acc_train"], "acc_held": cell["acc_held"]})

    # ---- A106-13：n=32 report-only（rev 1.2 降级；它测「事实最多时能否拟合」，非可学性上界）
    print("\n--- A106-13 report-only（n_train=32）---", flush=True)
    cell32 = run_cell(model, tok, facts, chunks, owner, SEEDS[0], 0, N_FULL, "full", "stageA")
    rec_only("A106-13", f"n=32 `acc_train` = {cell32['acc_train']:.4f}（report-only；实测 0.6562 为真实读数）",
             value={"acc_train": cell32["acc_train"]})
    rec("A106-12", cell32["acc_held"] is None,
        f"n=32 的 `he` 为空 ⇒ `acc_held` 必须为 null（实测 {cell32['acc_held']!r}），"
        f"不得记 0.0（用户规则：判据从未被计算 ≠ FAIL）")
    rec("A106-3", float(cell.get("key_maxcos", 1.0)) < 1e-4,
        f"white 键两两 |cos| 最大 = {cell.get('key_maxcos'):.2e}（须 <1e-4）",)
    rec("A106-4", float(cell.get("W_drift", 1.0)) == 0.0,
        f"读训练不改 `W`：`W_drift` = {cell.get('W_drift')}（须逐位 0.0）")
    rec("A106-7", cell.get("base_fp_after") in (None, cell.get("base_fp_before")),
        f"基座指纹前后一致（{cell.get('base_fp_after')}）")


# ---------------------------------------------------------------- 裁决
def verdict_main(path):
    import pathlib
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    cells = d["cells"]
    main_cells = [c for c in cells if c["arm"] == "main"]
    full_cells = [c for c in cells if c["arm"] == "full"]
    w0_cells = [c for c in cells if c["arm"] == "w0"]
    shuf_cells = [c for c in cells if c["arm"] == "shuf"]

    def pool(cs):
        hit = 0.0; tot = 0
        for c in cs:
            if c.get("acc_held") is None or not c.get("n_held"):
                continue
            hit += float(c["acc_held"]) * int(c["n_held"]); tot += int(c["n_held"])
        return (hit / tot if tot else None), tot

    rows = {}
    for n in N_TRAIN_SET:
        a, t = pool([c for c in main_cells if c["n_train"] == n])
        rows[f"{n}"] = {"acc_held": a, "n_q": t}
    shuf = {}
    for n in N_TRAIN_SET:
        a, t = pool([c for c in shuf_cells if c["n_train"] == n])
        shuf[f"{n}"] = {"acc_held": a, "n_q": t}
    w0 = {}
    for n in N_TRAIN_SET:
        a, t = pool([c for c in w0_cells if c["n_train"] == n])
        w0[f"{n}"] = {"acc_held": a, "n_q": t}

    # ---- A106-14：合并完备性（★ 缺格绝不能退化成 `C106-PARTIAL`，那会把「数据没跑完」
    #      伪装成一个实质科学结论；`DEF-106-07`）
    nq_ok = all(rows[f"{n}"]["n_q"] == EXPECT_NQ_MAIN[n] for n in N_TRAIN_SET)
    cnt = {"main": len([c for c in main_cells]),
           "shuf": len([c for c in shuf_cells]), "w0": len([c for c in w0_cells])}
    cnt_ok = all(cnt[k] == EXPECT_CELLS[k] for k in EXPECT_CELLS)
    complete = nq_ok and cnt_ok
    # ★ 必须写进**本文档** `d` 的 guards，而不是模块全局 `RES`（`DEF-106-11`）：
    #   全局 `RES` 不会随 `d` 落盘 ⇒ 守卫结果在 `stage_verdict.json` 里**完全丢失**。
    d.setdefault("guards", {})
    d["guards"]["A106-14"] = {
        "ok": complete,
        "detail": f"格数 {cnt}（须 =={EXPECT_CELLS}）；主扫描池化题数 "
                  f"{ {n: rows[f'{n}']['n_q'] for n in N_TRAIN_SET} }（须 =={EXPECT_NQ_MAIN}）",
        "value": {"cells": cnt, "n_q": {f"{n}": rows[f"{n}"]["n_q"] for n in N_TRAIN_SET}},
    }
    print(f"[{'PASS' if complete else 'FAIL'}] A106-14: 合并完备性 格数={cnt} "
          f"题数={ {n: rows[f'{n}']['n_q'] for n in N_TRAIN_SET} }", flush=True)

    vals = [rows[f"{n}"]["acc_held"] for n in N_TRAIN_SET]
    M1 = (vals[3] is not None) and vals[3] >= GATE_GEN
    steps = [b - a for a, b in zip(vals, vals[1:])] if all(v is not None for v in vals) else []
    M2 = bool(steps) and all(s > 0 for s in steps)

    def se_band(n):
        """★ rev 1.3（`DEF-106-10`）：`SE₀` 用**零假设** `p = CHANCE`，**不得**用观测 `p`。

        原稿用观测 `p` ⇒ `p` 越高带越宽 ⇒ 检验**自我吸收**上升趋势：
        一条干净的单调剂量反应 `{0.0625,0.125,0.25,0.375}` 会被误判成
        `C106-NO-GENERALIZATION`，而该支恰好**允许**把 `§7.2-7b` 记 `已足够`
        ⇒ 误差方向最坏（把有信号说成没信号）。零假设带必须与观测值**无关**。
        """
        N = rows[f"{n}"]["n_q"]
        if not N:
            return None
        se0 = math.sqrt(CHANCE * (1.0 - CHANCE) / N)
        return CHANCE + 2 * se0

    all_at_chance = all(
        rows[f"{n}"]["acc_held"] is not None and se_band(n) is not None
        and rows[f"{n}"]["acc_held"] <= se_band(n) for n in N_TRAIN_SET)

    G = d.get("guards", {})
    untestable = G.get("A106-2", {}).get("ok") is False
    if not complete:
        verdict = "INCOMPLETE"      # ★ 优先于 UNTESTABLE/PARTIAL：不是判决，是「还没跑完」
    elif untestable:
        verdict = "UNTESTABLE"
    elif M1:
        verdict = "C106-REACHABLE"
    elif all_at_chance and not M2:
        # ★ rev 1.3：仅「全在零假设带内」不够，还须**无严格单调剂量反应**
        #   （`DEF-106-10`：否则会把真实上升趋势记成「结构性不可泛化」）
        verdict = "C106-NO-GENERALIZATION"
    else:
        verdict = "C106-PARTIAL"

    RESv = {"verdict": verdict, "rows": rows, "shuffle": shuf, "w0": w0,
            "M106-1": M1, "M106-2": M2, "steps": steps,
            "all_at_chance": all_at_chance,
            "full_acc_train": [c.get("acc_train") for c in full_cells],
            "full_acc_held": [c.get("acc_held") for c in full_cells]}
    d["verdict"] = RESv
    d["verdict_str"] = verdict        # ★ 扁平键：`d["verdict"]` 是 dict，易被误读（`DEF-106-09`）
    # ★ 输出路径**不得**用字符串 replace（`DEF-106-08`）：若文件名里没有 `stage_formal`，
    #   `replace` 会返回原路径 ⇒ **覆盖输入文件**，把原始读数毁掉。
    _src = pathlib.Path(path)
    _out = _src.with_name(_src.stem + "_verdict" + _src.suffix)
    _out.write_text(
        json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print("\n===== 裁决 =====", flush=True)
    for n in N_TRAIN_SET:
        r = rows[f"{n}"]
        print(f"  n_train={n:>2}  池化 acc_held={r['acc_held']}  题数={r['n_q']}  "
              f"零假设带={se_band(n)}", flush=True)
    print(f"  M106-1 (acc_held(28)>=0.50) = {M1}", flush=True)
    print(f"  M106-2 (严格递增)            = {M2}  步长={steps}", flush=True)
    print(f"  全部贴地板(<=CHANCE+2SE)     = {all_at_chance}", flush=True)
    print(f"[判决] {verdict}", flush=True)
    if verdict == "INCOMPLETE":
        print("  ⚠️ 缺格：本文件**不得**写进 VERDICT.md／证据表；补齐后重跑 merge。", flush=True)
    print(f"[完成] → {_out}", flush=True)


def adjudicate_only(path):
    verdict_main(path)


if __name__ == "__main__":
    main()