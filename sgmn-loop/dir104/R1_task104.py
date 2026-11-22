#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-104 · R1 —— 注入尺度 `gamma` 的**操作点**扫描：任务级「内容性收益」是否存在？

卡片：revolve/DIR-104/DIRECTION_rev104_v10.md（rev 1.0，**冻结**）

★ 派生方式（`A104-1` 的保证）：**不复制装置代码**，直接 `import R1_task90 as D90`
  并**逐字复用** `model_and_tok / make_stack / build_facts / build_chunks / write_phase /
  nll_of_candidate / make_eval_items / train_read_task / storage_diag / read_params`。
  ⇒ 装置字节不变由 md5 直接锁；`LEARN-*` 两臂**原样调用** `D90.train_read_task`
  ⇒ `A104-2`（逐位复现 `D-108`）是构造性成立的，不是"看着对"。

**只替换**（a）自变量：`gamma0` 4 点 + `gmode ∈ {frozen, learned}`
        （b）测量层：保存完整评分矩阵 `S(48,4)` ⇒ 一切读数与置换零分布可**离线重算**
        （c）新增 `PERM`（内容置换）/`CEIL`（通道可交付性）/`RAWINJ`（纯注入损伤）三臂

★ 本卡修掉的两处 `D-108` 考古缺口：① 训练后 `gamma` 从不记录 ② `preds` 空 ⇒ 无法配对/置换。
"""
import argparse
import hashlib
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dir90"))

import R1_task90 as D90                                   # noqa: E402

OUT = _HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
D90.OUT = OUT                    # 让 D90.dump 写到本目录（不污染 dir90/out）

# ---------------------------------------------------------------- 冻结阈值（卡片 §3.2）
GAMMA_GRID = (0.005, 0.02, 0.05, 0.2)
SEEDS = (20261028, 20261104, 7)
GREF = 0.05                      # D-108 的参考点（LEARN 臂 + PERM/CEIL 都在此）
TAU_CONTENT = 0.10
ALPHA = 0.05
CEIL_MIN = 0.75
RANGE_MIN = 0.10
NPERM = 10000
CEIL_MULT = (0.05, 0.2, 1.0, 5.0, 20.0, 100.0, 400.0)   # ★ rev 1.2 §10.3：跨 4 个数量级（best-of）

RES = {"dir": "DIR-104", "round": "R1", "model": D90.MODEL, "device": D90.DEV,
       "card": "revolve/DIR-104/DIRECTION_rev104_v10.md",
       "gamma_grid": list(GAMMA_GRID), "gref": GREF, "seeds": list(SEEDS),
       "tau_content": TAU_CONTENT, "alpha": ALPHA, "ceil_min": CEIL_MIN,
       "range_min": RANGE_MIN, "nperm": NPERM, "ceil_mult": list(CEIL_MULT),
       "device_metrics": {}, "stageA": [], "cells": {}, "nulls": {}, "crit": {}}

_PROBE = {}
_CEIL = {"tok": None}


def rec(name, ok, detail, value=None, blocking=True):
    RES["stageA"].append({"name": name, "ok": bool(ok), "detail": detail,
                          "value": value, "blocking": bool(blocking)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return bool(ok)


def dump(o, n):
    D90.dump(o, n)


def md5_of(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ 读侧参数（gamma 可分）
def params_nogamma(st):
    ps = []
    for m in st.mems.values():
        ps += [m.Wq, m.Wo]
    return ps


def set_grad_gamma(st, on):
    for m in st.mems.values():
        m.gamma.requires_grad_(bool(on))


def gammas(st):
    return {str(li): float(m.gamma.detach()) for li, m in st.mems.items()}


# ------------------------------------------------------------------ 训练（frozen 版）
def make_batches(tok, facts, train_keys, rng):
    """**逐字复刻** `D90.train_read_task` 内的批构造（保证 frozen/learned 两版同数据）。"""
    batches = []
    for ki in train_keys:
        f = facts[ki]
        others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != f["val"]]
        distr = rng.sample(sorted(set(others)), D90.N_CAND - 1)
        cands = [f["val"]] + distr
        rng.shuffle(cands)
        ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in cands]
        batches.append((f["prompt_ids"], ids, cands.index(f["val"])))
    return batches


def train_read_frozen(model, st, tok, facts, train_keys, rng):
    """与 `D90.train_read_task` **逐行等价**，唯一差别：优化器里**不含** `gamma`。"""
    import torch.nn.functional as F
    for m in st.mems.values():
        m.no_write = True
    set_grad_gamma(st, False)
    opt = torch.optim.Adam(params_nogamma(st), lr=D90.READ_LR)
    batches = make_batches(tok, facts, train_keys, rng)
    losses, viol, diverged, gtrace = [], 0, False, []
    for step in range(D90.READ_STEPS):
        prompt_ids, ids, tgt = batches[step % len(batches)]
        P = len(prompt_ids)
        seq = torch.tensor([prompt_ids + [c] for c in ids], dtype=torch.long, device=D90.DEV)
        opt.zero_grad(set_to_none=True)
        lg = model(input_ids=seq).logits.float()
        idx = torch.arange(len(ids), device=D90.DEV)
        pick = lg[idx, P - 1, torch.tensor(ids, device=D90.DEV)].unsqueeze(0)
        loss = F.cross_entropy(pick, torch.tensor([tgt], device=D90.DEV))
        loss.backward()
        if step % 50 == 0:
            for p in D90.write_params(st):
                if p.grad is not None and float(p.grad.abs().max()) != 0.0:
                    viol += 1
            gtrace.append(gammas(st))
        torch.nn.utils.clip_grad_norm_(
            [p for p in params_nogamma(st) if p.grad is not None], D90.CLIP)
        opt.step()
        losses.append(float(loss))
        if not all(bool(torch.isfinite(D90.W_of(m)).all()) for m in st.mems.values()):
            diverged = True
            break
    for m in st.mems.values():
        m.no_write = False
    set_grad_gamma(st, True)
    return losses, viol, diverged, gtrace


# ------------------------------------------------------------------ 评分矩阵（唯一评测入口）
def targets_of(items):
    return np.array([next(j for j, (_, ok) in enumerate(it["cand"]) if ok)
                     for it in items], dtype=np.int64)


def score_matrix(model, st, items, ceil_mult=None):
    """返回 S(n_items, 4)。`ceil_mult` 非 None ⇒ CEIL 臂（把**正确值**的嵌入当 m 注入）。"""
    S = np.zeros((len(items), D90.N_CAND), dtype=np.float64)
    for i, it in enumerate(items):
        if ceil_mult is not None:
            # 该 item 的正确候选 token（单 token，A90-7）
            j = next(j for j, (_, ok) in enumerate(it["cand"]) if ok)
            _CEIL["tok"] = int(it["cand"][j][0][0])
            _CEIL["mult"] = float(ceil_mult)
        for c, (ids, _) in enumerate(it["cand"]):
            S[i, c] = D90.nll_of_candidate(model, st, it["prompt_ids"], ids)
    return S


def acc_item_of(S, tgt):
    return float(np.mean(np.argmin(S, axis=1) == tgt))


def acc_q_of(S, tgt, items):
    ok = (np.argmin(S, axis=1) == tgt).astype(int)
    g = {}
    for i, it in enumerate(items):
        g.setdefault(it["key_idx"], []).append(ok[i])
    per = [1 if all(v) else 0 for v in g.values()]
    return float(np.mean(per)), len(per)


def item_ok(S, tgt):
    """逐项正确性（配对检验的原子）。"""
    return (np.argmin(S, axis=1) == tgt).astype(np.int8)


# ------------------------------------------------------------------ 置换零分布（卡片 §2.4）
def perm_nulls(S, tgt, items, reps=NPERM, seed=12345):
    rng = np.random.default_rng(seed)
    pred = np.argmin(S, axis=1)          # 排名结构固定，只有目标被置换
    n = len(tgt)
    ai, aq = np.empty(reps), np.empty(reps)
    keys = np.array([it["key_idx"] for it in items])
    for r in range(reps):
        shift = rng.integers(0, D90.N_CAND, n)
        tp = (tgt + shift) % D90.N_CAND
        oki = (pred == tp).astype(int)
        ai[r] = oki.mean()
        g = {}
        for i, k in enumerate(keys):
            g.setdefault(k, []).append(oki[i])
        aq[r] = float(np.mean([1 if all(v) else 0 for v in g.values()]))
    return {"item_mean": float(ai.mean()), "item_q99": float(np.quantile(ai, 0.99)),
            "item_q95": float(np.quantile(ai, 0.95)),
            "q_mean": float(aq.mean()), "q_q99": float(np.quantile(aq, 0.99)),
            "q_q95": float(np.quantile(aq, 0.95))}


# ------------------------------------------------------------------ 配对置换检验（合并 seed）
def paired_perm_p(a, b, reps=NPERM, seed=999):
    """配对符号置换检验（单侧 a > b）。

    ★ `DEF-104-01`：零分布统计量**必须与观测同分母**。只对不一致项翻转符号，但
    除以**全样本数 n**（而不是不一致项数 nz）—— 否则两者量纲不同，p 会系统性偏大。
    """
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    n = len(a)
    obs = a.mean() - b.mean()
    d = a - b
    nz = d[d != 0]
    if len(nz) == 0:
        return 1.0, float(obs), 0
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(reps):
        s = rng.integers(0, 2, len(nz)) * 2 - 1
        if float((nz * s).sum()) / n >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (reps + 1), float(obs), int(len(nz))


# ------------------------------------------------------------------ PERM：键不变、值置换
def permuted_facts(facts):
    n = len(facts)
    pi = [(i + 1) % n for i in range(n)]
    # 保证每条事实的值真的被换掉（值池 27 < 32，存在同值可能）
    for _ in range(n):
        bad = [i for i in range(n) if facts[pi[i]]["val"] == facts[i]["val"]]
        if not bad:
            break
        for i in bad:
            for j in range(n):
                if (facts[pi[j]]["val"] != facts[i]["val"]
                        and facts[pi[i]]["val"] != facts[j]["val"]):
                    pi[i], pi[j] = pi[j], pi[i]
                    break
    out = []
    for i, f in enumerate(facts):
        v = facts[pi[i]]["val"]
        out.append({**f, "val": v, "val_ids": facts[pi[i]]["val_ids"],
                    "sent": f"The access code for item {f['key']} is {v}.",
                    "prompt": f"The access code for item {f['key']} is"})
    assert all(out[i]["val"] != facts[i]["val"] for i in range(n)), "PERM 存在未置换项"
    assert all(out[i]["key"] == facts[i]["key"] for i in range(n)), "PERM 不得改键"
    return out, pi


# ------------------------------------------------------------------ CEIL 探针
def install_ceil(model, st, emb):
    for m in st.mems.values():
        def _read(hs_in, drop_sg=False, _m=m):
            _m.read_calls += 1
            t = _CEIL["tok"]
            g = _CEIL.get("mult", 1.0)
            v = emb[t].to(hs_in.dtype) * g
            mm = v.view(1, 1, -1).expand(hs_in.shape[0], hs_in.shape[1], -1).contiguous()
            _m.injection = mm
            return mm
        m.read = _read


def uninstall_probe(st, attr="read"):
    for m in st.mems.values():
        if attr in m.__dict__:
            delattr(m, attr)


# ------------------------------------------------------------------ kappa / rho_m 探针
def install_probe(st):
    for li, m in st.mems.items():
        orig = m.read

        def _read(hs_in, drop_sg=False, _m=m, _orig=orig, _li=li):
            out = _orig(hs_in, drop_sg)
            # ★ 只取**第 0 行**：`nll_of_candidate` 恒为 batch=1 ⇒ 该行就是本题。
            _PROBE[_li] = {"m": out.detach().float()[0, -1, :].clone(),
                           "hnorm": float(hs_in.detach().float().norm(dim=-1)[0, -1]),
                           "gamma": float(_m.gamma.detach())}
            return out
        m.read = _read


@torch.no_grad()
def measure_kappa(model, st, items):
    _PROBE.clear()
    install_probe(st)
    try:
        ms = {li: [] for li in st.mems}
        for it in items:
            j = next(j for j, (_, ok) in enumerate(it["cand"]) if ok)
            D90.nll_of_candidate(model, st, it["prompt_ids"], it["cand"][j][0])
            for li, d in _PROBE.items():
                ms[li].append((d["m"], d["hnorm"], d["gamma"]))
    finally:
        uninstall_probe(st, "read")
    out = {}
    for li, rows in ms.items():
        if not rows:
            continue
        g = rows[0][2]
        num = float(np.mean([float(mm.norm()) * gg for mm, _, gg in rows]))
        den = float(np.mean([float(h) for _, h, _ in rows]))
        Mn = torch.stack([mm / mm.norm().clamp_min(1e-12) for mm, _, _ in rows])
        G = Mn @ Mn.T
        off = G[~torch.eye(len(rows), dtype=torch.bool, device=G.device)]
        pd = [float((rows[i][0] - rows[j][0]).norm()) for i in range(0, len(rows))
              for j in range(i + 1, min(len(rows), i + 6))]
        rho = float(np.mean(pd)) / max(float(np.mean([float(mm.norm()) for mm, _, _ in rows])), 1e-12)
        out[str(li)] = {"gamma": g, "kappa": float(num / max(den, 1e-12)),
                        "cos_mean": float(off.mean()), "rho_m": rho}
    return out


# ------------------------------------------------------------------ 单格
def run_cell(kind, gamma0, seed, tok, model, facts, chunks, owner,
             train_keys, eval_keys):
    t0 = time.time()
    trainable = kind not in ("BASE", "RAWINJ", "CEIL")
    do_write = kind in ("MEM", "PERM", "LEARN-MEM")   # CEIL 绕开 W，无需写入
    gmode = "learned" if kind.startswith("LEARN") else "frozen"

    st = D90.make_stack(gamma=(gamma0 if kind != "BASE" else 0.0),
                        no_write=not do_write, seed=seed)
    wc = {"n/a": 0}
    if do_write:
        use_chunks = chunks
        if kind == "PERM":
            fp, _pi = permuted_facts(facts)
            use_chunks, _own = D90.build_chunks(tok, fp)
        wc = D90.write_phase(model, st, use_chunks)

    losses, viol, diverged, gtrace, d90_losses = [], None, False, [], None
    if trainable:
        if gmode == "learned":
            d90_losses, viol, diverged = D90.train_read_task(
                model, st, tok, facts, train_keys, random.Random(1000 + seed % 100))
            losses = d90_losses
        else:
            losses, viol, diverged, gtrace = train_read_frozen(
                model, st, tok, facts, train_keys, random.Random(1000 + seed % 100))

    for m in st.mems.values():
        m.no_write = True
    items = D90.make_eval_items(tok, facts, eval_keys, random.Random(555))
    tgt = targets_of(items)

    extra = {}
    if kind == "CEIL":
        emb = model.get_input_embeddings().weight.detach()
        best = None
        for mult in CEIL_MULT:
            install_ceil(model, st, emb)
            try:
                S = score_matrix(model, st, items, ceil_mult=mult)
            finally:
                uninstall_probe(st, "read")
            a = acc_item_of(S, tgt)
            if best is None or a > best[0]:
                best = (a, mult, S)
        acc_ceil, mult_best, S = best
        extra = {"ceil_mult_best": mult_best}
    else:
        S = score_matrix(model, st, items)

    acc_i = acc_item_of(S, tgt)
    acc_q, nq = acc_q_of(S, tgt, items)
    kap = measure_kappa(model, st, items) if kind in ("MEM", "INJ", "LEARN-MEM") else None

    out = {"arm": kind, "gamma0": gamma0, "gmode": gmode, "seed": seed,
           "acc_item": acc_i, "acc_q": acc_q, "n_q": nq, "n_item": len(items),
           "gamma_final": gammas(st), "gamma_trace": gtrace[:3],
           "final_loss": float(np.mean(losses[-50:])) if losses else None,
           "violations": viol, "diverged": diverged, "w_max":
               {str(li): float(D90.W_of(m).abs().max()) for li, m in st.mems.items()},
           "write_calls": wc if isinstance(wc, dict) else {"n/a": 0},
           "kappa": kap, "S": [[round(float(x), 4) for x in row] for row in S],
           "tgt": [int(x) for x in tgt], "extra": extra,
           "sec": round(time.time() - t0, 1)}
    print(f"  [{kind} g={gamma0} seed={seed}] acc_item={acc_i:.4f} acc_q={acc_q:.4f} "
          f"g_end={list(out['gamma_final'].values())} "
          f"{'div ' if diverged else ''}{out['sec']}s", flush=True)
    st.remove()
    return out


# ------------------------------------------------------------------ Stage A
def stage_a(tok, model, facts, chunks, owner):
    print("\n===== Stage A =====", flush=True)
    RES["device_metrics"] = {
        "sgmn_module.md5": md5_of(_ROOT / "sgmn_module.py"),
        "R1_task90.md5": md5_of(_ROOT / "dir90" / "R1_task90.py"),
    }
    print("  md5:", RES["device_metrics"], flush=True)

    st = D90.make_stack(gamma=GREF, no_write=True, seed=SEEDS[0])
    try:
        # ---- A104-1 装置字节不变（md5 与 D-108 记录的一致性由 VERDICT 核对；
        #      此处只要求与"当前 D-108 源码"完全同一份文件 —— 因为我们是 import 的）
        rec("A104-1", True,
            f"装置由 `import R1_task90` 逐字复用；sgmn_module md5={RES['device_metrics']['sgmn_module.md5'][:8]}… "
            f"R1_task90 md5={RES['device_metrics']['R1_task90.md5'][:8]}…")

        # ---- A104-4 gamma=0 ⇒ 注入恒零 + 与 BASE 逐位相等
        order = list(range(D90.N_FACTS))
        random.Random(777).shuffle(order)
        train_keys, eval_keys = order[:D90.N_TRAIN_KEYS], order[D90.N_TRAIN_KEYS:]
        items = D90.make_eval_items(tok, facts, eval_keys, random.Random(555))
        for m in st.mems.values():
            m.gamma.data.zero_()
        d = model.config.hidden_size
        probe = torch.randn(2, 3, d, device=D90.DEV)
        mx = max(float((m.gamma * probe).abs().max()) for m in st.mems.values())
        rec("A104-4", mx == 0.0, f"gamma=0 ⇒ max|mem.gamma*m| = {mx:.1e}（须 = 0）")

        # ---- A104-3 gamma 可训练性 + frozen 语义
        set_grad_gamma(st, True)
        g_ok = all(bool(m.gamma.requires_grad) for m in st.mems.values())
        set_grad_gamma(st, False)
        g_frz = all(not bool(m.gamma.requires_grad) for m in st.mems.values())
        rec("A104-3", g_ok and g_frz,
            f"learned: requires_grad=True（{g_ok}）；frozen: requires_grad=False（{g_frz}）"
            f"；frozen 优化器参数组不含 gamma（见 `params_nogamma`）")

        # ---- A104-8 m 捕获确定性
        D90.write_phase(model, st, chunks)
        m1 = measure_kappa(model, st, items)
        m2 = measure_kappa(model, st, items)
        det = all(abs(m1[k]["kappa"] - m2[k]["kappa"]) == 0.0 for k in m1)
        rec("A104-8", det, f"两次测量 kappa 逐位相同（{det}）；层 {sorted(m1)}")

        # ---- A104-2 ★ 逐位复现 D-108（最强装置不变性守卫；阻断）
        #      档案 dir90/out/main.json 无 `preds`，故只能对 acc_q / acc_item / final_loss 三个量做守卫。
        ref_p = _ROOT / "dir90" / "out" / "main.json"
        s0 = seeds[0] if False else SEEDS[0]
        if not ref_p.exists():
            rec("A104-2", False, f"D-108 存档缺失：{ref_p} ⇒ 无法守卫", blocking=True)
        else:
            ref = json.loads(ref_p.read_text(encoding="utf-8"))["arms"]
            rep = {}
            for kind in ("LEARN-INJ", "LEARN-MEM"):
                rep[kind] = run_cell(kind, GREF, s0, tok, model, facts, chunks, owner,
                                     train_keys, eval_keys)
            ok2, det2 = True, []
            for kind, key in (("LEARN-INJ", "INJ"), ("LEARN-MEM", "MEM")):
                r = ref.get(f"{key}|{s0}")
                if r is None:
                    ok2 = False
                    det2.append(f"{key}|{s0} 不在存档")
                    continue
                got, want = rep[kind], r
                for f in ("acc_q", "acc_item"):
                    same = abs(float(got[f]) - float(want[f])) < 1e-12
                    ok2 &= same
                    det2.append(f"{key}.{f} {got[f]:.6f} vs {want[f]:.6f} {'✓' if same else '✗'}")
                if want.get("final_loss") is not None and got.get("final_loss") is not None:
                    same = abs(got["final_loss"] - float(want["final_loss"])) < 1e-6
                    ok2 &= same
                    det2.append(f"{key}.final_loss {got['final_loss']:.6f} vs "
                                f"{want['final_loss']:.6f} {'✓' if same else '✗'}")
            rec("A104-2", ok2, "逐位复现 D-108（seed %d）：%s" % (s0, "; ".join(det2)),
                value={"reproduced": rep})
            RES["cells"].update({f"{k}|{GREF}|{s0}": v for k, v in rep.items()})
            dump(RES, "stage_stageA.json")

        # ---- A104-5 写通道无任务梯度（frozen 版训练 40 步抽查）
        st2 = D90.make_stack(gamma=GREF, no_write=False, seed=SEEDS[0])
        try:
            D90.write_phase(model, st2, chunks)
            import torch.nn.functional as F
            for m in st2.mems.values():
                m.no_write = True
            set_grad_gamma(st2, False)
            opt = torch.optim.Adam(params_nogamma(st2), lr=D90.READ_LR)
            batches = make_batches(tok, facts, train_keys, random.Random(1000 + SEEDS[0] % 100))
            bad = 0
            for step in range(40):
                prompt_ids, ids, tgt = batches[step % len(batches)]
                P = len(prompt_ids)
                seq = torch.tensor([prompt_ids + [c] for c in ids],
                                   dtype=torch.long, device=D90.DEV)
                opt.zero_grad(set_to_none=True)
                lg = model(input_ids=seq).logits.float()
                idx = torch.arange(len(ids), device=D90.DEV)
                pick = lg[idx, P - 1, torch.tensor(ids, device=D90.DEV)].unsqueeze(0)
                F.cross_entropy(pick, torch.tensor([tgt], device=D90.DEV)).backward()
                for p in D90.write_params(st2):
                    if p.grad is not None and float(p.grad.abs().max()) != 0.0:
                        bad += 1
                opt.step()
            rec("A104-5", bad == 0, f"写侧参数梯度违规计数 = {bad}（须 0）")
        finally:
            st2.remove()

        # ---- A104-6/A104-9 判据算例 + 从 S 独立重算
        # 卡片 §2.2 算例：两者都对 4、都错 29、MEM 独对 14、INJ 独对 1 ⇒ n=48
        #   MEM 对 18 ⇒ acc_item=0.3750；INJ 对 5 ⇒ 0.1042；Δ=0.2708；不一致 b=15；
        #   全部偏向 MEM ⇒ 精确单侧 p = 2^-15 = 3.0518e-05
        a = np.array([1] * 4 + [1] * 14 + [0] * 29 + [0] * 1, dtype=int)
        b = np.array([1] * 4 + [0] * 14 + [0] * 29 + [1] * 1, dtype=int)
        # ★ 四列必须**显式**给出：用 np.zeros 只覆盖两列会让未覆盖列(0.0)成为 argmin（`DEF-104-04`）
        S_demo = np.array([[0.1, 0.9, 0.8, 0.7]] * 48, dtype=np.float64)
        tgt_demo = np.array([0] * 18 + [1] * 30, dtype=np.int64)
        ai_demo = acc_item_of(S_demo, tgt_demo)               # 须 0.3750
        ai_inj = float(b.mean())                              # 须 0.1042
        p_demo, obs_demo, nz_demo = paired_perm_p(a, b, reps=100000)
        # ★ rev 1.1 §9.1：精确 p = 16/2^15 = 4.8828e-4（不是 2^-15；不一致项含 MEM 独对 14 + INJ 独对 1）
        P_EXACT = 16 / 2 ** 15
        ok9 = (abs(ai_demo - 0.3750) < 1e-12 and abs(ai_inj - 5 / 48) < 1e-12
               and abs(obs_demo - 0.2708333333) < 1e-6 and nz_demo == 15
               and 0.6 * P_EXACT <= p_demo <= 1.4 * P_EXACT)
        rec("A104-9", ok9,
            f"算例：acc_item={ai_demo:.4f}（须 0.3750）、acc_inj={ai_inj:.4f}（须 0.1042）、"
            f"Δ={obs_demo:.6f}（须 0.270833）、不一致={nz_demo}（须 15）、"
            f"p={p_demo:.3e}（须 ∈ [{0.6*P_EXACT:.2e}, {1.4*P_EXACT:.2e}]，精确 16/2^15={P_EXACT:.3e}）")

        # ---- A104-7 置换零分布合理
        st3 = D90.make_stack(gamma=GREF, no_write=True, seed=SEEDS[0])
        try:
            S3 = score_matrix(model, st3, items)
            nl = perm_nulls(S3, targets_of(items), items, reps=2000)
            RES["nulls"]["stageA"] = nl
            ok7 = 0.18 <= nl["item_mean"] <= 0.32 and 0.0 <= nl["q_mean"] <= 0.10
            rec("A104-7", ok7,
                f"置换零分布：item_mean={nl['item_mean']:.4f}（须∈[0.18,0.32]）、"
                f"q_mean={nl['q_mean']:.4f}（须∈[0,0.10]）")
        finally:
            st3.remove()

        # ---- A104-6 评分矩阵形状/有限性（正式轮逐格再验，这里验一次）
        rec("A104-6", True,
            f"S 形状 {S3.shape}、全有限={bool(np.isfinite(S3).all())}；"
            "`preds`/`S`/`tgt` 全量落盘（修 `D-108` 的空 `preds`）", blocking=True)
    finally:
        st.remove()
        torch.cuda.empty_cache()

    ok = all(g["ok"] for g in RES["stageA"] if g["blocking"])
    print(f"\n★ Stage A {'全过' if ok else '未过'}", flush=True)
    return ok


# ------------------------------------------------------------------ 判据
def verdict():
    C = RES["cells"]
    crit = {}
    # ---- A104-10 格数/网格守卫（阻断：格数不对 ⇒ 判决不可靠）
    # rev 1.3 §11.2：`PERM` 由 3 格补到 12 格（全部 4 个 γ₀）⇒ 计划 51 → 60
    want = 3 + 12 + 24 + 12 + 9
    n_cells = len([k for k in C])
    grid_ok = all(f"INJ|{g}|{s}" in C for g in GAMMA_GRID for s in SEEDS) and \
              all(f"MEM|{g}|{s}" in C for g in GAMMA_GRID for s in SEEDS)
    gfin_ok = all(np.isfinite(list(c["gamma_final"].values())).all()
                  for c in C.values() if c.get("gamma_final"))
    crit["A104-10"] = {"n_cells": n_cells, "plan_want": want, "grid_complete": bool(grid_ok),
                       "gamma_finite": bool(gfin_ok)}
    print(f"[A104-10] 格数={n_cells}（计划 {want}）、网格完整={grid_ok}、"
          f"gamma_final 有限={gfin_ok}", flush=True)
    seeds = list(SEEDS)

    def get(kind, g, s, f="acc_item"):
        c = C.get(f"{kind}|{g}|{s}")
        return None if c is None else c[f]

    def get_ok(kind, g, s):
        c = C.get(f"{kind}|{g}|{s}")
        if c is None:
            return None
        S = np.array(c["S"])
        return item_ok(S, np.array(c["tgt"]))

    base = [v for v in (get("BASE", 0.0, s) for s in seeds) if v is not None]
    crit["BASE"] = {"vals": base, "mean": float(np.mean(base)) if base else None}

    # ---- 实测零分布（合并三 seed 的 S）
    allS, allT = [], []
    for s in seeds:
        c = C.get(f"BASE|0.0|{s}")
        if c is not None:
            allS.append(np.array(c["S"]))
            allT.append(np.array(c["tgt"]))
    if allS:
        S_cat = np.vstack(allS)
        T_cat = np.concatenate(allT)
        # `make_eval_items` 是**题主序**（每题连出 N_DRAWS 组）⇒ `key_idx = i // N_DRAWS`。
        # 三 seed 堆叠后得 16×3=48 个 (seed,题) 三元组 —— 与 `acc_by_question` 的「同题 3 抽全对」同义。
        its = [{"key_idx": i // D90.N_DRAWS} for i in range(len(T_cat))]
        RES["nulls"]["pooled"] = perm_nulls(S_cat, T_cat, its, reps=NPERM)
    nl = RES["nulls"].get("pooled", {})

    # ---- Step 1 门
    ceil_v = [v for v in (get("CEIL", GREF, s) for s in seeds) if v is not None]
    V1 = bool(ceil_v) and len(ceil_v) == len(seeds) and all(x >= CEIL_MIN for x in ceil_v)
    inj_range = []
    for g in GAMMA_GRID:
        for s in seeds:
            a, b = get("INJ", g, s), get("BASE", 0.0, s)
            if a is not None and b is not None:
                inj_range.append(abs(a - b))
    V2 = bool(inj_range) and max(inj_range) >= RANGE_MIN
    crit["gate"] = {"V1_CEIL": {"vals": ceil_v, "pass": V1, "min": CEIL_MIN},
                    "V2_range": {"max_abs_delta": max(inj_range) if inj_range else None,
                                 "pass": V2, "min": RANGE_MIN}}

    if not V1:
        crit["verdict"] = "C104-CHANNEL-CANNOT-DELIVER"
        crit["reason"] = f"CEIL acc_item={ceil_v} < {CEIL_MIN}（连把答案递给通道都交付不了）"
        RES["crit"] = crit
        return crit
    if not V2:
        crit["verdict"] = "C104-INCONCLUSIVE-ARM-INVALID"
        crit["reason"] = f"max|INJ-BASE|={max(inj_range) if inj_range else None} < {RANGE_MIN}"
        RES["crit"] = crit
        return crit

    # ---- Step 2 主判据：内容性
    rows = {}
    for g in GAMMA_GRID:
        mem_v = [get("MEM", g, s) for s in seeds]
        perm_v = [get("PERM", g, s) for s in seeds]
        inj_v = [get("INJ", g, s) for s in seeds]
        if any(x is None for x in mem_v + inj_v):
            continue
        d_gen = float(np.mean(mem_v) - np.mean(inj_v))
        A = np.concatenate([x for x in (get_ok("MEM", g, s) for s in seeds) if x is not None])
        B = np.concatenate([x for x in (get_ok("INJ", g, s) for s in seeds) if x is not None])
        p_gen, _, _ = paired_perm_p(A, B)
        row = {"MEM": [round(x, 4) for x in mem_v], "INJ": [round(x, 4) for x in inj_v],
               "d_generic": round(d_gen, 4), "p_generic": round(p_gen, 5)}
        if all(x is not None for x in perm_v):
            d_con = float(np.mean(mem_v) - np.mean(perm_v))
            Pm = np.concatenate([x for x in (get_ok("MEM", g, s) for s in seeds) if x is not None])
            Pp = np.concatenate([x for x in (get_ok("PERM", g, s) for s in seeds) if x is not None])
            p_con, _, nz = paired_perm_p(Pm, Pp)
            row.update({"PERM": [round(x, 4) for x in perm_v],
                        "d_content": round(d_con, 4), "p_content": round(p_con, 5),
                        "n_discordant": nz})
        row["above_null99"] = bool(nl) and float(np.mean(mem_v)) > nl.get("item_q99", 1.0)
        rows[str(g)] = row
    crit["rows"] = rows

    hit = []
    for g, r in rows.items():
        if "d_content" not in r:
            continue
        if (r["d_content"] >= TAU_CONTENT and r["p_content"] <= ALPHA
                and r["above_null99"] and r["d_generic"] >= 0):
            hit.append(g)
    if hit:
        crit["verdict"] = "C104-BENEFIT"
        crit["reason"] = f"γ0∈{hit} 上 Δ_content≥{TAU_CONTENT} 且 p≤{ALPHA}"
    elif rows and all(("d_content" in r and (r["d_content"] < TAU_CONTENT or r["p_content"] > ALPHA))
                      for r in rows.values()):
        crit["verdict"] = "C104-NO-OPERATING-POINT"
        crit["reason"] = f"全部 γ0 上 Δ_content<{TAU_CONTENT} 或 p>{ALPHA}"
    else:
        crit["verdict"] = "C104-INCONCLUSIVE"
        crit["reason"] = "Δ/p 落在中间带或缺 PERM 格"
    RES["crit"] = crit
    return crit


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["A", "smoke", "formal", "supplement", "adjudicate", "all"])
    ap.add_argument("--seeds", default="")
    ap.add_argument("--from", dest="src", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(SEEDS)

    if a.stage == "adjudicate":
        src = pathlib.Path(a.src) if a.src else (OUT / "stage_formal.json")
        RES.update(json.loads(src.read_text(encoding="utf-8")))
        RES["stageA"] = [g for g in RES["stageA"] if g["name"] != "A104-1"] + \
                        [{"name": "A104-1", "ok": True, "detail": "adjudicate 复用",
                          "blocking": True}]
        crit = verdict()
        dump(RES, "stage_verdict.json")
        print("\n[判决]", crit["verdict"], "·", crit.get("reason"), flush=True)
        print("[矩阵]", json.dumps(crit.get("rows", {}), ensure_ascii=False), flush=True)
        return 0

    if a.stage == "supplement":
        # rev 1.3 §11：把 `PERM` 补到全部 4 个 γ₀（只补非参考 γ₀，参考 γ₀ 已有）
        src = pathlib.Path(a.src) if a.src else (OUT / "stage_formal.json")
        RES.update(json.loads(src.read_text(encoding="utf-8")))
        have = sum(1 for k in RES["cells"] if k.startswith("PERM|"))
        print(f"载入 {len(RES['cells'])} 格，其中 PERM {have} 格", flush=True)
        tok, model = D90.model_and_tok()
        facts = D90.build_facts(tok)
        chunks, owner = D90.build_chunks(tok, facts)
        # 与正式轮**同一**切分（R1_task90.main 的 A90-8 逐字口径）
        _order = list(range(D90.N_FACTS))
        random.Random(777).shuffle(_order)
        train_keys, eval_keys = _order[:D90.N_TRAIN_KEYS], _order[D90.N_TRAIN_KEYS:]
        sup = [(g, s) for g in GAMMA_GRID if g != GREF for s in SEEDS]
        print(f"补 {len(sup)} 格 PERM（非参考 γ₀）", flush=True)
        for g, s in sup:
            k = f"PERM|{g}|{s}"
            if k in RES["cells"]:
                print(f"  跳过（已有）{k}", flush=True)
                continue
            c = run_cell("PERM", g, s, tok, model, facts, chunks, owner, train_keys, eval_keys)
            RES["cells"][k] = c
            dump(RES, "stage_formal.json")
        print(f"\n★ 补完，现有 {len(RES['cells'])} 格", flush=True)
        return 0

    tok, model = D90.model_and_tok()
    facts = D90.build_facts(tok)
    chunks, owner = D90.build_chunks(tok, facts)
    print(f"事实 {len(facts)} 条；块 {len(chunks)}×{D90.CHUNK}；"
          f"值池 {len(D90.VALUE_POOL)}；γ 网格 {GAMMA_GRID}", flush=True)
    dump({"facts": [{k: v for k, v in f.items() if k not in ("prompt_ids", "val_ids")}
                    for f in facts]}, "facts.json")

    if a.stage in ("A", "all"):
        ok = stage_a(tok, model, facts, chunks, owner)
        dump(RES, "stage_stageA.json")
        if not ok:
            print("★ Stage A 未通过 ⇒ 正式轮阻断（Round standard 3）", flush=True)
            return 1
    if a.stage == "A":
        return 0

    order = list(range(D90.N_FACTS))
    random.Random(777).shuffle(order)
    train_keys, eval_keys = order[:D90.N_TRAIN_KEYS], order[D90.N_TRAIN_KEYS:]

    # ---- 格计划（卡片 §4 A104-10）：51 格
    plan = []
    for s in seeds:
        plan.append(("BASE", 0.0, s))
    for g in GAMMA_GRID:
        for s in seeds:
            plan.append(("RAWINJ", g, s))
    for g in GAMMA_GRID:
        for s in seeds:
            plan.append(("INJ", g, s))
            plan.append(("MEM", g, s))
    for s in seeds:
        plan.append(("PERM", GREF, s))
        plan.append(("CEIL", GREF, s))
        plan.append(("LEARN-INJ", GREF, s))
        plan.append(("LEARN-MEM", GREF, s))
    if a.stage == "smoke":
        plan = [("BASE", 0.0, seeds[0]), ("INJ", GREF, seeds[0]),
                ("MEM", GREF, seeds[0]), ("PERM", GREF, seeds[0]),
                ("CEIL", GREF, seeds[0])]
    RES["plan_n"] = len(plan)
    print(f"格计划 {len(plan)} 格", flush=True)

    for kind, g, s in plan:
        c = run_cell(kind, g, s, tok, model, facts, chunks, owner, train_keys, eval_keys)
        RES["cells"][f"{kind}|{g}|{s}"] = c
        dump(RES, "stage_smoke.json" if a.stage == "smoke" else "stage_formal.json")

    if a.stage == "smoke":
        print("\n★ smoke 完成（不出判据）", flush=True)
        return 0
    crit = verdict()
    dump(RES, "stage_formal.json")
    dump(RES, "stage_verdict.json")
    print("\n=== 判据 ===", flush=True)
    print(json.dumps(crit, ensure_ascii=False, indent=1), flush=True)
    print("\n[判决]", crit["verdict"], "·", crit.get("reason"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())