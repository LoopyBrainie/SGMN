#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-21 · **写侧容量 / 衰减探针**（★ **诊断用；不是科学结论** ★）

------------------------------------------------------------------
**目的（只两件）**：
  ① 把 H21-1b 的 **测量对象**从「猜」变成「有据」：冻结定义里的
     `R_cnt(K)` = **可检索条目数**（绝对量 ⇒ 容量曲线），`K*` = 该曲线的 argmax。
     ⇒ 探针把**所有候选口径**同时测出来，看哪一条曲线有**可定义的 argmax**：
        (a) `r_cnt`      —— 仍占据槽位且自身签名仍选中该槽的条目数（rev 1.4 的实现口径）；
        (b) `alive_ep`   —— **仍驻留**（未被回绕覆盖）的**情节**数（纯记账，精确）；
        (c) `occupied`   —— `min(累计写次数, N_ENTRIES)`（槽位占用量）；
        (d) `addr_ok`    —— 文档 1 的码条目**寻址是否仍正确**（训练无关）。
  ② 把**衰减曲线**测在**正确的对象**上 ⇒ 定出 H21-3 的半衰期**是否可测**（不被删失）、
     以及扫描网格的**取值范围**。

------------------------------------------------------------------
**为什么必须探针（三个已证实的测量缺陷）**：
  * **D103（步轴不等长）**：现实现 `D(s)=floor(s/m)+1` ⇒ 同一个名义 step 上，
    各臂的**实际写次数不同**（实测 step 96：`G_tok` 136 次 / `G_doc` 98 次）
    ⇒ 「同龄」不成立 ⇒ 跨臂曲线不可比。
  * **D102（保真度测错对象）**：`fid_true` 走 `fidelity(true_slot)` = **当前槽里住的是谁**；
    一旦回绕（`count > N_ENTRIES`），槽换主 ⇒ 测的是**别人的内容**
    ⇒ smoke5 实测 `G_doc` 的 `fid` 在 step 256 从 0.867 **跳回** 0.991（伪回升）。
    ⇒ 正确对象 = **文档 1 当时写下的 (k, v)**：`cos(W·k, v)`（内容是否还在 W 里）。
  * **D107（半衰期删失）**：smoke5 所有臂 `half_life=256` = 网格边缘 ⇒ H21-3 是
    **删失导致的伪 PASS**（「判据被计算了，但值是删失值」）。

------------------------------------------------------------------
**探针边界（纪律）**：
  * **不训练**（`Wo`/`γ` 保持初值）⇒ 读侧**不具代表性** ⇒ `hit` 仅作参考；
    但**写侧动力学**（`W` 的衰减）、**地址动力学**（签名寻址）、**槽位占用**
    **全部与训练无关**（DIR-21 的基座逐位冻结 ⇒ hidden state 不随训练改变）
    ⇒ ⇒ 本探针的写侧读数**对正式轮有效**。
  * **不产生判据、不进证据表**；只用于**开跑前**钉死协议口径（rev 1.6）。
"""

import json
import math
import pathlib
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R1_granularity as RG            # noqa: E402  （复用**逐字同一份**装置与协议件）

SEED = 20261028
KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N_EVAL = KMAX + 2
OUT_JSON = HERE.parent / "dir21_probe_capacity.json"
OUT_TXT = HERE.parent / "dir21_probe_capacity.txt"


def set_arm(st, eta, stride, pick):
    """与 `R1_granularity.main()` 内的 `_set_arm` **逐字同构**（探针自持一份，避免依赖嵌套定义）。"""
    for m in st.mems.values():
        m.eta = float(eta)
        m.core.eta = float(eta)
        m.write_stride = int(stride)
        m.stride_pick = str(pick)


def content_fid(m, k, v):
    """★ **内容保真（正确对象）**：`cos(W·k, v)`，k/v = 该条目**当时写入**的原始键/值。

    ⇒ 与 `fidelity(slot)` 的区别：后者测「当前槽里的住户像不像自己」，
    回绕后两者**指向不同内容** ⇒ 衰减曲线被污染（缺陷 D102）。
    """
    with torch.no_grad():
        W = m.core._W()
        a = W @ k
        b = v
        na = float(a.norm())
        nb = float(b.norm())
        if na < 1e-30 or nb < 1e-30:
            return 0.0
        return float((a @ b) / (na * nb))


def slot_fid(m, j):
    with torch.no_grad():
        return float(m.core.fidelity(int(j)))


def run_arm(model, d, name, eta, stride, pick, ev):
    st = RG.make_stack(model, d, "cue", seed=SEED, mem_kind="hebbian_full",
                       eta=RG.ETA_BASE, key_mode="ortho", variant="normal")
    RG.clear_slots(st)                      # ★ 容量扫描必须从**空地址表**开始（缺陷 D100）
    set_arm(st, eta, stride, pick)
    m0 = st.mems[list(st.mems)[0]]
    N = int(m0.core.N)
    trig = list(m0.trigger_positions(len(ev[0]["ids"])))
    m_doc = 2 * len(trig)                   # 每篇写规则触发总次数（目标文档 + 干扰文档）
    code_i = len(trig) - 1                  # 触发位置升序 ⇒ 码 token (t=L−1) 是最后一个
    t_code = int(trig[code_i])
    true_slot_code = code_i % N             # 写入序号 0 起 ⇒ 槽号 = 写入序号（未回绕时）

    # —— 写文档 1，捕获它的 (k, v)（**逐字原始对象**）——
    st.unfreeze_write()
    RG.write_both(st, ev[0])
    doc1 = []
    for m in st.mems.values():
        ks = [t.detach().clone() for t in m.keys_written[:len(trig)]]
        vs = [t.detach().clone() for t in m.vals_written[:len(trig)]]
        doc1.append(list(zip(ks, vs)))

    spec_code = {"prefix": list(ev[0]["ids"][:t_code]),
                 "code_ids": [int(ev[0]["ids"][t_code])],
                 "true_idx": int(code_i),
                 "true_slot": int(true_slot_code),
                 "dist_code": ev[0].get("dist_code", "0")}

    rows = []
    for K in range(1, KMAX + 1):
        if K > 1:
            st.unfreeze_write()
            RG.write_both(st, ev[K - 1])
        writes = int(m0.core.count)
        # —— 衰减：内容保真（正确对象）vs 槽保真（污染对象）——
        f_code = float(np.mean([content_fid(m, doc1[i][code_i][0], doc1[i][code_i][1])
                                 for i, m in enumerate(st.mems.values())]))
        f_mean = float(np.mean([content_fid(m, k, v)
                                 for i, m in enumerate(st.mems.values())
                                 for (k, v) in doc1[i]]))
        s_code = float(np.mean([slot_fid(m, true_slot_code) for m in st.mems.values()]))
        # —— 容量候选口径 ——
        alive_ep = 0
        for e in range(1, K + 1):
            wi = (e - 1) * m_doc + code_i       # 情节 e 的码条目的**全局写序号**
            if wi + N > writes:
                alive_ep += 1
        r = RG.eval_arm_readonly(st, spec_code, None)
        rows.append({
            "K": K, "writes": writes,
            "fid_content_code": f_code, "fid_content_mean": f_mean, "fid_slot_code": s_code,
            "r_cnt": RG.resident_r_cnt(st), "occupied": min(writes, N),
            "alive_ep": alive_ep,
            "addr_slot": float(r.get("addr_slot", float("nan"))),
            "hit": float(r.get("hit", float("nan"))),
            "nll": float(r.get("nll", float("nan"))),
        })
    st.remove()
    torch.cuda.empty_cache()
    return {"name": name, "eta": float(eta), "stride": int(stride), "pick": str(pick),
            "m_doc": int(m_doc), "n_entries_doc": int(len(trig)), "N": N, "rows": rows}


def summarize(arm):
    """从曲线里读出：内容半衰期（情节 / 写步）、容量 knee、删失情况。"""
    rows = arm["rows"]
    f0 = arm["rows"][0]["fid_content_mean"]
    out = {"name": arm["name"], "m_doc": arm["m_doc"], "n_entries_doc": arm["n_entries_doc"],
           "fid_content_code_K1": arm["rows"][0]["fid_content_code"],
           "fid_content_mean_K1": f0,
           "fid_slot_code_K1": arm["rows"][0]["fid_slot_code"]}
    # —— 内容半衰期：首个 `fid_content_mean / fid_content_mean(K=1) ≤ 0.5` ——
    hl_K = None
    for r in rows:
        if f0 > 0 and (r["fid_content_mean"] / f0) <= 0.5:
            hl_K = r["K"]
            break
    out["half_life_K_content"] = hl_K
    out["half_life_writes_content"] = (hl_K * arm["m_doc"]) if hl_K else None
    out["half_life_censored_content"] = hl_K is None
    # —— 槽保真的半衰期（**污染对象**，用于展示伪回升）——
    s0 = arm["rows"][0]["fid_slot_code"]
    hl_s = None
    for r in rows:
        if s0 > 0 and (r["fid_slot_code"] / s0) <= 0.5:
            hl_s = r["K"]
            break
    out["half_life_K_slotpolluted"] = hl_s
    # —— 容量 knee：三条候选曲线的 argmax / 首个平台点 ——
    for key in ("r_cnt", "occupied", "alive_ep"):
        vals = [(r["K"], r[key]) for r in rows]
        am = max(vals, key=lambda kv: kv[1])
        out["argmax_" + key] = am[0]
        out["peak_" + key] = am[1]
        # 首个「达到最终峰值」的点（平台起点）
        out["plateau_start_" + key] = next((k for k, v in vals if v >= am[1]), None)
    # —— 寻址失效点（文档 1 的码条目不再被正确寻址）——
    out["addr_fail_K"] = next((r["K"] for r in rows if r["addr_slot"] < 0.999), None)
    out["hit_fail_K"] = next((r["K"] for r in rows if r["hit"] < 0.999), None)
    # —— 伪回升检查：内容保真单调性 ——
    fcs = [r["fid_content_code"] for r in rows]
    out["content_code_monotone_noninc"] = all(fcs[i + 1] <= fcs[i] + 1e-6 for i in range(len(fcs) - 1))
    scs = [r["fid_slot_code"] for r in rows]
    out["slot_code_monotone_noninc"] = all(scs[i + 1] <= scs[i] + 1e-6 for i in range(len(scs) - 1))
    out["slot_code_max_rise_after_drop"] = max(
        [scs[i + 1] - scs[i] for i in range(len(scs) - 1) if scs[i + 1] > scs[i] + 1e-9] or [0.0])
    return out


def main():
    print("=== DIR-21 写侧容量/衰减探针（诊断用 · 非科学结论）===", flush=True)
    tok = AutoTokenizer.from_pretrained(RG.MODEL)
    model = AutoModelForCausalLM.from_pretrained(RG.MODEL, dtype=torch.float32).to(RG.DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size
    tr, ev = RG.build_docs(tok, SEED, 1, N_EVAL)
    print(f"[数据] 评测篇数={len(ev)}；文档长度={len(ev[0]['ids'])}", flush=True)
    arms = []
    for (name, eta, stride, pick) in RG.ARMS21:
        arm = run_arm(model, d, name, eta, stride, pick, ev)
        arms.append(arm)
        s = summarize(arm)
        print(f"\n--- {name}  m_doc={arm['m_doc']}  每篇条目={arm['n_entries_doc']}  "
              f"N={arm['N']}  η={eta:.6g} stride={stride} ---", flush=True)
        print(f"    内容半衰期：K={s['half_life_K_content']}  ⇒ 写步={s['half_life_writes_content']}"
              f"  （删失={s['half_life_censored_content']}）", flush=True)
        print(f"    槽保真半衰期（污染对象）：K={s['half_life_K_slotpolluted']}", flush=True)
        print(f"    容量候选：argmax r_cnt={s['argmax_r_cnt']}（峰 {s['peak_r_cnt']:.0f}，"
              f"平台起点 {s['plateau_start_r_cnt']}）", flush=True)
        print(f"              argmax alive_ep={s['argmax_alive_ep']}（峰 {s['peak_alive_ep']:.0f}，"
              f"平台起点 {s['plateau_start_alive_ep']}）", flush=True)
        print(f"              argmax occupied={s['argmax_occupied']}（峰 {s['peak_occupied']:.0f}，"
              f"平台起点 {s['plateau_start_occupied']}）", flush=True)
        print(f"    寻址失效 K={s['addr_fail_K']}；读出命中失效 K={s['hit_fail_K']}", flush=True)
        print(f"    内容保真单调不增={s['content_code_monotone_noninc']}；"
              f"槽保真单调不增={s['slot_code_monotone_noninc']}；"
              f"槽保真最大回升={s['slot_code_max_rise_after_drop']:.4f}", flush=True)
        print("    K  writes  fid_content_code  fid_content_mean  fid_slot_code  "
              "r_cnt  occupied  alive_ep  addr_slot", flush=True)
        show = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96,
                128, 160, 200]
        for r in arm["rows"]:
            if r["K"] in show and r["K"] <= KMAX:
                print(f"   {r['K']:4d} {r['writes']:6d}  {r['fid_content_code']:.4f}"
                      f"        {r['fid_content_mean']:.4f}         {r['fid_slot_code']:.4f}"
                      f"       {r['r_cnt']:4d}  {r['occupied']:5d}   {r['alive_ep']:5d}"
                      f"     {r['addr_slot']:.3f}", flush=True)
    res = {"probe": "DIR-21 write-side capacity/decay probe",
           "note": "DIAGNOSTIC ONLY - not a scientific conclusion",
           "seed": SEED, "kmax": KMAX, "doc_len": len(ev[0]["ids"]),
           "arms": arms, "summary": [summarize(a) for a in arms]}
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
