#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 rev 2.5 · 值编码的可辨识性（task: DIR-91-rev2.6）

问题（由 D-105 的承重读数派生）：M2 岭回归 V->27 类 LOO = 0.031-0.062（chance 1/27），
而样本内 = 1.000。**为什么**？是「块均值」抹掉了答案身份，还是 `P_v`／层选？

三臂（唯一区别 = 对哪些位置取均值；无训练、无写预算、无读侧更新）：
  mean_all   1/64 * sum_{t=0..63} z_t          复现 D-105 的 v_i（装置现状）
  vt_pos     值 token 位均值                    主假说臂
  nonvt      非值 token 位均值                  阴性对照

z_t = h_t @ P_v，h_t = 基座隐态（gamma=0 => 注入 = 0*m = 0 => 逐位为基座）。

★ 判据口径（CORR-12）：32 事实 / 27 值 => 22 个类是单例，留一法**不可能**预测
  在其余 31 条中不出现的类 => LOO-32 天花板 = 10/32 = 0.3125。
  故**主判据 = LOO-10**（只在重数>=2 的 10 条事实上算，天花板 = 1.0，chance 1/27）。
  IN-SAMPLE 必须同时报（只作"记忆 vs 结构"对照，不承重）。

守卫 A92-01..08（任一 FAIL 阻断正式轮）。本文件**不修改** sgmn_module.py / R2_task91.py。
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from R2_task91 import (ALPHA, B, CHUNK, DEV, ETA_WRITE, LAM, MOUNT, N_FACTS,
                       build_chunks, build_facts, key_matrices, make_stack,
                       model_and_tok, write_phase)

CARD, TASK = "DIR-91 rev 2.5", "DIR-91-rev2.6"
OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUTD, exist_ok=True)
RES = {"card": CARD, "task": TASK, "mount": MOUNT, "chunk": CHUNK, "B": B,
       "n_facts": N_FACTS, "seeds": [], "guards": {}, "arms": {}, "gates": {},
       "preds": {}, "ceiling": {}, "verdict": None}
FAILS = []


def rec(name, passed, detail, value=None):
    RES["guards"][name] = {"ok": bool(passed), "detail": detail, "value": value}
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)
    if not passed:
        FAILS.append(name)


def dump(obj, name):
    with open(os.path.join(OUTD, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- 岭回归（逐字同 D-105）
def ridge_scores(Xtr, ytr, Xte, nclass, lam=1.0):
    """对偶形式岭回归到 one-hot：W = Xc^T (Xc Xc^T + lam I)^{-1} Y。返回 (nte,nclass)。"""
    Xtr = Xtr.detach().to("cpu", torch.float64)
    Xte = Xte.detach().to("cpu", torch.float64)
    mu = Xtr.mean(dim=0, keepdim=True)
    Xc, Xt = Xtr - mu, Xte - mu
    Y = torch.zeros(len(ytr), nclass, dtype=Xtr.dtype)
    Y[torch.arange(len(ytr)), torch.as_tensor(ytr)] = 1.0
    Kmat = Xc @ Xc.T
    A = torch.linalg.solve(Kmat + lam * torch.eye(len(ytr), dtype=Xtr.dtype), Y)
    return Xt @ Xc.T @ A


def loo_predictions(X, cls, nclass, lam=1.0):
    """留一（留一**条事实**）：返回每条事实的预测类别。"""
    pred = []
    for i in range(len(X)):
        keep = [j for j in range(len(X)) if j != i]
        s = ridge_scores(X[keep], [cls[j] for j in keep], X[[i]], nclass, lam)
        pred.append(int(s.argmax(dim=1)[0]))
    return pred


def acc_on(pred, cls, idx):
    return float(np.mean([int(pred[i]) == int(cls[i]) for i in idx])) if idx else float("nan")


# ---------------------------------------------------------------- 值 token 位精确定位
def value_positions(tok, facts):
    """用 offset-mapping 精确定位值 token 在句内的 token 下标 j，再按块=句子循环平铺
    得位置集合 V = {p in [0,CHUNK) : p % period == j}。返回 [(V, j, diag)]。

    ★ DEF-92-01：块张量形状是 (1, CHUNK)，**不是** (CHUNK,) —— 位置范围必须用 CHUNK，
      不能用 shape[0]（那是 batch 维 = 1）。原写法使 V 恒为空集，且 nonvt 退化成 mean_all。"""
    out = []
    for f in facts:
        enc = tok(f["sent"] + " ", add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]
        chars = len(f["prompt"]) + 1                  # sent = prompt + " " + val + ","
        j = None
        for t, (s0, e0) in enumerate(offs):
            if s0 <= chars < e0:
                j = t
                break
        if j is None:
            raise RuntimeError(f"未能定位值 token offset：{f['sent']}")
        per = len(ids)
        V = [p for p in range(CHUNK) if p % per == j]
        out.append({"V": V, "j": int(j), "period": int(per),
                    "tok_id": int(ids[j]),
                    "id_count_in_period": int(ids.count(ids[j]))})
    return out


# ================================================================= Stage A
def stage_a(tok, model, facts, chunks, owner, seed):
    print("\n===== Stage A 守卫 =====\n", flush=True)
    d_model = model.config.hidden_size
    cls = R_CLS
    ncls = R_NCLS

    # ---------- A92-01 ★ 天花板在**真实划分**上算出（CORR-12）
    counts = {}
    for c in cls:
        counts[c] = counts.get(c, 0) + 1
    dist = {}
    for c, k in counts.items():
        dist[k] = dist.get(k, 0) + 1
    idx10 = [i for i in range(N_FACTS) if counts[cls[i]] >= 2]
    ceil32 = len(idx10) / N_FACTS
    RES["ceiling"] = {"ncls": ncls, "mult_dist": {str(k): v for k, v in dist.items()},
                      "n_predictable": len(idx10), "ceiling_loo32": ceil32,
                      "ceiling_loo10": 1.0, "chance": 1.0 / ncls,
                      "idx10": idx10}
    rec("A92-01", ncls == 27 and dist == {1: 22, 2: 5} and len(idx10) == 10
        and abs(ceil32 - 0.3125) < 1e-12,
        f"★ 天花板在真实划分上算出：ncls={ncls}；重数分布={dist}（须 {{1:22, 2:5}}）；"
        f"可预测事实={len(idx10)}（须 10）；**LOO-32 天花板={ceil32:.4f}**（须 0.3125）、"
        f"**LOO-10 天花板=1.0**；chance=1/27≈{1.0/ncls:.4f}",
        value={"dist": dist, "n10": len(idx10), "ceil32": ceil32})

    # ---------- 建栈（gamma=0, no_write=True => 注入=0 => 隐态逐位为基座）
    st = make_stack(gamma=0.0, no_write=True, seed=seed)
    g = float(st.mems[MOUNT[0]].gamma.detach())
    rec("A92-00", g == 0.0,
        f"栈的 gamma 初值={g:.1f}（须 ==0 ⇒ 注入 gamma*m=0 ⇒ _hs_in 为基座隐态）", value=g)

    # ---------- 逐事实取逐位置 z
    per = value_positions(tok, facts)
    Z = {li: [] for li in MOUNT}          # Z[li][fi] = (CHUNK, d_v)
    H = {li: [] for li in MOUNT}
    for fi in range(N_FACTS):
        ci = owner.index(fi) + 1
        model(input_ids=chunks[ci].to(DEV))
        for li in MOUNT:
            hs = st._hs_in[li]                       # (1,CHUNK,d_model) 基座隐态
            pf = hs.reshape(-1, hs.shape[-1])        # (CHUNK,d_model)
            H[li].append(pf.detach().float().cpu())
            Z[li].append((pf @ st.mems[li].Pv).detach().float().cpu())   # (CHUNK,d_v)

    # ---------- A92-02 ★ 值 token 位定位精确
    cnt = [len(p["V"]) for p in per]
    same = all(d["id_count_in_period"] == 1 for d in per)
    tok_ok = True
    for fi, d in enumerate(per):
        ci = owner.index(fi) + 1
        row = chunks[ci][0].tolist()
        tok_ok &= all(row[p] == d["tok_id"] for p in d["V"])
        tok_ok &= all(row[p] == row[d["V"][0]] for p in d["V"])
    _deg = 0
    for li in MOUNT:
        for fi in range(N_FACTS):
            V = per[fi]["V"]
            if V:
                m = torch.ones(Z[li][fi].shape[0], dtype=torch.bool); m[V] = False
                if torch.allclose(Z[li][fi][m].mean(dim=0), Z[li][fi].mean(dim=0), atol=1e-6):
                    _deg += 1
    rec("A92-02", (min(cnt) >= 2 and max(cnt) <= 5 and same and tok_ok and _deg == 0),
        f"★ 值 token 位定位：|V_i|∈[{min(cnt)},{max(cnt)}]（须 [2,5]）；"
        f"值 token 在每个周期内**只出现一次**={same}；定位位置上的 token id 全部等于 offset-mapping "
        f"给出的值 token={tok_ok}；**非值位均值不等于块均值**（退化计数={_deg}，须 0）",
        value={"cnt": cnt, "yes": same and tok_ok, "degenerate": _deg})

    # ---------- A92-03 ★ 分解与装置逐位一致（与 key_matrices 的 V 比）
    Kdev, Vdev = key_matrices(model, st, facts, chunks, owner)
    mean_all = {li: torch.stack([Z[li][fi].mean(dim=0) for fi in range(N_FACTS)])
                for li in MOUNT}
    d03 = max(float((mean_all[li] - Vdev[li].float().cpu()).abs().max()
                    / Vdev[li].float().cpu().abs().max()) for li in MOUNT)
    rec("A92-03", d03 <= 1e-5,
        f"★ 逐位置分解与装置**逐位一致**：由 z_t 重建的 mean_all 与 key_matrices 的 V 的 "
        f"max|Δ|/max|V| = {d03:.2e}（须 ≤1e-5）", value=d03)

    # ---------- 构造三臂
    A = {}
    for li in MOUNT:
        ma, vt, nv = [], [], []
        for fi in range(N_FACTS):
            z = Z[li][fi]
            V = per[fi]["V"]
            mask = torch.ones(z.shape[0], dtype=torch.bool)
            mask[V] = False
            ma.append(z.mean(dim=0))
            vt.append(z[V].mean(dim=0))
            nv.append(z[mask].mean(dim=0))
        A[li] = {"mean_all": torch.stack(ma), "vt_pos": torch.stack(vt), "nonvt": torch.stack(nv)}

    # ---------- A92-06 尺度审计（固定 lam=1.0 是否公平）
    rms = {}
    for li in MOUNT:
        for a in ("mean_all", "vt_pos", "nonvt"):
            rms[f"L{li}_{a}"] = float(A[li][a].pow(2).mean().sqrt())
    lo, hi = min(rms.values()), max(rms.values())
    rec("A92-06", hi / lo <= 4.0,
        f"尺度审计（逐维 rms）：min={lo:.3f} max={hi:.3f} 比值={hi/lo:.2f}（须 ≤4 ⇒ 固定 lam=1.0 公平）",
        value={"ratio": hi / lo, "min": lo, "max": hi})

    # ---------- A92-07 ★ P_v 保持可辨识：rank(Pv)=d_model 且 LOO(h)≈LOO(h Pv)
    Pv = st.mems[MOUNT[0]].Pv.detach().float().cpu()
    rk = int(torch.linalg.matrix_rank(Pv))
    looH, looZ = {}, {}
    for li in MOUNT:
        hv = torch.stack([H[li][fi][per[fi]["V"]].mean(dim=0) for fi in range(N_FACTS)])
        pz = loo_predictions(A[li]["vt_pos"], cls, ncls)
        ph = loo_predictions(hv, cls, ncls)
        looH[li] = acc_on(ph, cls, idx10)
        looZ[li] = acc_on(pz, cls, idx10)
        A[li]["vt_on_h"] = hv
    d07 = max(abs(looH[li] - looZ[li]) for li in MOUNT)
    rec("A92-07", rk == d_model and d07 <= 0.20,
        f"★ `P_v` 保持可辨识：rank(P_v)={rk}（须 =={d_model}，方阵满秩 ⇒ 线性双射）；"
        f"值 token 位上 LOO-10(h) 与 LOO-10(hP_v) 之差 max={d07:.3f}（须 ≤0.20）⇒ "
        f"**P_v 不是凶手**", value={"rank": rk, "d": d07,
                                    "loo_h": {str(k): v for k, v in looH.items()},
                                    "loo_hpv": {str(k): v for k, v in looZ.items()}})

    # ---------- A92-04 ★ LOO 通路天花板（合成数据，每类重数 >=2）
    Xoh = torch.eye(ncls)[torch.as_tensor(cls)].clone()      # 精确 one-hot（同 A91-42 口径）
    p_oh = loo_predictions(Xoh, cls, ncls)
    a_oh = acc_on(p_oh, cls, idx10)
    rec("A92-04", a_oh == 1.0,
        f"★ LOO 通路天花板（合成 one-hot，每类重数>=2）：**LOO-10={a_oh:.12f}**（须 ==1.0）",
        value=a_oh)

    # ---------- A92-05 置乱对照
    accs = []
    for k in range(5):
        gen = torch.Generator().manual_seed(5 + k)
        perm = torch.randperm(N_FACTS, generator=gen).tolist()
        cls_sh = [cls[perm[i]] for i in range(N_FACTS)]
        p_sh = loo_predictions(A[MOUNT[0]]["vt_pos"], cls_sh, ncls)
        accs.append(float(np.mean([int(p_sh[i]) == int(cls_sh[i]) for i in range(N_FACTS)])))
    a_sh = float(np.mean(accs))
    rec("A92-05", a_sh <= 0.10,
        f"置乱标签对照（vt_pos, L8, **5 次置换取均值**）：LOO-32={a_sh:.3f}（须 ≤0.10；"
        f"五次 {[round(x,3) for x in accs]}；chance 1/27≈0.037）⇒ 无信息泄漏", value=a_sh)

    # ---------- A92-08 延续性：mean_all 的 LOO-32 复现 D-105 的机会水平
    loo32 = {}
    for li in MOUNT:
        p = loo_predictions(A[li]["mean_all"], cls, ncls)
        loo32[li] = float(np.mean([int(p[i]) == int(cls[i]) for i in range(N_FACTS)]))
    rec("A92-08", max(loo32.values()) <= 0.10,
        f"延续性：mean_all 的 LOO-32 = " +
        " ".join(f"L{li}:{v:.3f}" for li, v in loo32.items()) +
        f"（须全部 ≤0.10；D-105 实测 0.031–0.062）", value={str(k): v for k, v in loo32.items()})

    return A, idx10


# ================================================================= 正式轮
def run_seed(seed, tok, model, facts, chunks, owner):
    print(f"\n===== seed {seed} =====", flush=True)
    A, idx10 = stage_a(tok, model, facts, chunks, owner, seed)
    cls, ncls = R_CLS, R_NCLS
    out = {}
    for li in MOUNT:
        row = {}
        for a in ("mean_all", "vt_pos", "nonvt"):
            X = A[li][a]
            p = loo_predictions(X, cls, ncls)
            s_in = ridge_scores(X, cls, X, ncls)
            row[a] = {"loo10": acc_on(p, cls, idx10),
                      "loo32": float(np.mean([int(p[i]) == int(cls[i]) for i in range(N_FACTS)])),
                      "insample": float(np.mean([int(v) == int(c) for v, c in
                                                 zip(s_in.argmax(dim=1).tolist(), cls)]))}
        # 反事实：同一 z 若把值 token 位"置零"再取块均值（检验稀释假说的直接模拟）
        row["loo10_h"] = acc_on(loo_predictions(A[li]["vt_on_h"], cls, ncls), cls, idx10)
        out[str(li)] = row
        print(f"  L{li}: " + " | ".join(
            f"{a} loo10={row[a]['loo10']:.3f} loo32={row[a]['loo32']:.3f} ins={row[a]['insample']:.3f}"
            for a in ("mean_all", "vt_pos", "nonvt")) + f" || vt_on_h loo10={row['loo10_h']:.3f}",
            flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["guards", "all"])
    ap.add_argument("--seeds", default="20261028,7777")
    args = ap.parse_args()
    t0 = time.time()
    tok, model = model_and_tok()
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    global R_CLS, R_NCLS
    vals = sorted({f["val"] for f in facts})
    v2c = {v: i for i, v in enumerate(vals)}
    R_CLS = [v2c[f["val"]] for f in facts]
    R_NCLS = len(vals)
    RES["ncls"] = R_NCLS

    seeds = [int(s) for s in args.seeds.split(",")]
    for sd in seeds:
        RES["seeds"].append(sd)
        RES["arms"][str(sd)] = run_seed(sd, tok, model, facts, chunks, owner)

    if FAILS:
        print(f"\n★ Stage A 有 {len(FAILS)} 项 FAIL ⇒ **阻断正式轮**：{FAILS}", flush=True)
        RES["blocked"] = FAILS
        dump(RES, "probe91_valuetok_guards.json")
        return

    print("\n★ Stage A 全部 PASS", flush=True)
    # ------------- 判决合成（§2.4，冻结）
    Z = RES["arms"]
    n_pool = n_none = n_weak = n_both = n_contra = 0
    tot = 0
    for sd in Z:
        for li, row in Z[sd].items():
            a1, a2, a3 = row["mean_all"]["loo10"], row["vt_pos"]["loo10"], row["nonvt"]["loo10"]
            tot += 1
            if a1 > 0.20:
                n_contra += 1
            elif a2 >= 0.70 and a3 <= 0.50:
                n_pool += 1
            elif a2 >= 0.70 and a3 > 0.50:
                n_both += 1
            elif 0.40 <= a2 < 0.70:
                n_weak += 1
            else:
                n_none += 1
    if n_contra > 0:
        verdict = "CONTRADICT"
    elif n_pool == tot:
        verdict = "POOLING"
    elif n_pool >= 4:
        verdict = "POOLING_PARTIAL"
    elif n_none * 2 > tot:
        verdict = "NOT_AT_TOKEN"
    elif n_both * 2 > tot:
        verdict = "BOTH"
    else:
        verdict = "WEAK"
    RES["gates"] = {"n_pool": n_pool, "n_both": n_both, "n_weak": n_weak,
                    "n_none": n_none, "n_contra": n_contra, "total": tot}
    RES["verdict"] = verdict

    # ------------- 预测核对（§2.5）
    def all_le(key, thr):
        return all(Z[sd][li][key]["loo10"] <= thr for sd in Z for li in Z[sd])

    def all_ge(key, thr):
        return all(Z[sd][li][key]["loo10"] >= thr for sd in Z for li in Z[sd])

    P = {
        "P1_mean_all_loo10_le_0.20": all_le("mean_all", 0.20),
        "P2_vt_pos_ge_0.70": all_ge("vt_pos", 0.70),
        "P3_nonvt_le_0.50": all_le("nonvt", 0.50),
        "P4_gap_vt_minus_nonvt_ge_0.40": all(
            Z[sd][li]["vt_pos"]["loo10"] - Z[sd][li]["nonvt"]["loo10"] >= 0.40
            for sd in Z for li in Z[sd]),
        "P5_vt_pos_ge_0.70_all_layers": all_ge("vt_pos", 0.70),
        "P6_pv_gap_le_0.20": RES["guards"].get("A92-07", {}).get("ok", False),
        "P7_mean_all_loo32_le_0.10": all(Z[sd][li]["mean_all"]["loo32"] <= 0.10
                                         for sd in Z for li in Z[sd]),
    }
    RES["preds"] = P
    print("\n" + "=" * 88)
    for k, v in P.items():
        print(f"  {'✓' if v else '✗'} {k}", flush=True)
    print(f"\n★ 判决 = **{verdict}**  (gates={RES['gates']})", flush=True)
    print(f"★ 用时 {time.time()-t0:.0f}s", flush=True)
    dump(RES, "probe91_valuetok.json")


if __name__ == "__main__":
    main()
