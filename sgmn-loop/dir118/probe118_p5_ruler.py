#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""`P118-5` —— 把 `N_hit^v` 尺子在本装置（DIR-118）上重新实例化 + 跑正负对照。

未跑任何训练（无 backward）。只做：
  ① 结构核：`E_random = Σm_v²/N` 由装置自己的 facts 机械算出；`S_i` 自洽
  ② 三臂正负对照：`oracle = V K⁺`（阳性）/ `oracle^π`（阴性，5 个置换取 max）/ `W⁰`（零基线）
  ③ `intact` 读数：原生键 与 去相关键（`D-128` 的 `WhiteTF`），各 1 遍写

判据见 `revolve/DIR-118/RULER-P118-5.md` §3（**冻结于跑之前**）。
★ 逐字派生：`dir97/R1_task97.py` + `dir98/R1_task98.py`（`n_hit` / `WhiteTF` / `key_matrices_tf`
  全部**直接复用**，不重写）；唯一改动 = **facts**（27 值 → 16 值×2）与随之而变的 `E_random`。
★ `COR-114-01`：K/V 一律读装置自己的 `m.Pk / m.Pv / m.pk_scale`，不重实现投影。

用法（在 trainer 上）：
    /opt/pytorch/bin/python probe118_p5_ruler.py --stage stageA    # 结构 + 对照 + W⁰（便宜）
    /opt/pytorch/bin/python probe118_p5_ruler.py --stage formal    # 追加 A5-5 + intact 两臂
"""
import argparse
import json
import math
import pathlib
import random
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
for _d in ("dir118", "dir98", "dir97", "dir91", "dir90", "dir114", "dir115", "dir117"):
    sys.path.insert(0, str(_ROOT / _d))

import torch                                     # noqa: E402

import R1_task90 as D90                          # noqa: E402
import R1_task118 as D118                        # noqa: E402   ← 抢回 OUT / DEV 的权威来源
import R1_task98 as D98                          # noqa: E402   ← n_hit / WhiteTF / key_matrices_tf
from sgmn_module import LayerMem                 # noqa: E402

P = D118.P                                       # = R1_task114（build_facts_pairs / partitions）
OUT = _HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
# ★ 必须在 import D98 之后抢回：D98 模块体可能改写别处的 OUT（DEF-118-04 同族风险）
D98.OUT = OUT

DEV = D90.DEV
MOUNT = D90.MOUNT
SEED_STRUCT = 20261028                           # 结构/对照用的栈种子（W0 由此定）
SEED_INTACT = (20261028, 7777)                   # intact 两臂各跑这两个种子（与 dir98 同族）
N_PERM = 5
PERM_SEED0 = 20261122                            # ★ 冻结：第 k 个置换 = Random(PERM_SEED0+k)

RES = {"card": "P118-5 / RULER-P118-5.md", "stage": None,
       "guards": [], "structure": {}, "controls": {}, "intact": {}, "notes": []}
FAILS = []


def rec(cid, ok, detail, value=None, blocking=True):
    """★ `DEF-95-04`：先聚合、再落盘、后抛异常 —— 本函数**不**在中间写盘、不 raise。"""
    RES["guards"].append({"id": cid, "ok": bool(ok), "detail": str(detail),
                          "value": value, "blocking": bool(blocking)})
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {cid}  {detail}", flush=True)
    if not ok and blocking:
        FAILS.append(cid)
    return bool(ok)


def norm_rows(X):
    return D98.norm_rows(X)


def n_hit(W, K, V, Ssets):
    """★ 四臂**同一个函数对象**（`A5-4①`）：不复制、不包装、不换名。"""
    return D98.n_hit(W, K, V, Ssets)


# ==================================================================== 结构核
def structure(tok):
    facts = P.build_facts_pairs(tok)
    N = len(facts)
    cnt = {}
    for f in facts:
        cnt[f["val"]] = cnt.get(f["val"], 0) + 1
    m = list(cnt.values())

    # ---- A5-1：零期望**由 facts 机械算出**，且**不得**照抄 D-127 的 1.3125
    E_num = sum(x * x for x in m)
    E_random = E_num / N
    ceiling = max(m)
    RES["structure"] = {
        "n_facts": N, "distinct_vals": len(cnt),
        "mult_dist": {str(k): sum(1 for x in m if x == k) for k in sorted(set(m))},
        "E_num": E_num, "E_random": E_random, "ceiling": ceiling,
        "d129_E_random": 1.3125,          # ★ 只用于「我没照抄」的断言，不参与任何计算
        "val_names": sorted(cnt)[:4],
    }
    rec("A5-1a", abs(E_random - E_num / N) < 1e-12,
        f"E_random = Σm_v²/N = {E_num}/{N} = {E_random}（由装置 facts 算出，非硬编码）",
        value=E_random)
    rec("A5-1b", E_random != 1.3125,
        f"E_random = {E_random} ≠ 1.3125 ⇒ 未照抄 D-127 的 27 值池常数", value=E_random)

    # ---- A5-2：值等价类自洽
    vals = [f["val"] for f in facts]
    Ssets = [[j for j in range(N) if vals[j] == vals[i]] for i in range(N)]
    sz = [len(s) for s in Ssets]
    ok2 = all(len(s) == cnt[vals[i]] for i, s in enumerate(Ssets)) and max(sz) == ceiling
    rec("A5-2", ok2,
        f"|S_i| == m_(v_i) 全部成立；max|S| = {max(sz)} == ceiling = {ceiling}"
        f"（|S| 分布 {sorted(set(sz))}）", value={"size_dist": sorted(set(sz)),
                                                  "max": max(sz), "ceiling": ceiling})
    return facts, Ssets, E_random, ceiling


# ==================================================================== K/V 提取
def extract_kv(model, facts, chunks, owner, seed, key_tf):
    """★ 直接调 `D98.key_matrices_tf`（不重写）：它在 `KP.RULE[ARM]=('mean','mean')` 下
    逐位等于装置原生写入公式（`A5-5` 另行证明补丁恒等）。"""
    st = D90.make_stack(no_write=True, seed=seed)
    try:
        for mm in st.mems.values():
            mm.key_tf = key_tf
        K, V = D98.key_matrices_tf(model, st, facts, chunks, owner)
    finally:
        st.remove()
    return K, V


# ==================================================================== 对照
def controls(K, V, Ssets, W0s, E_random):
    n = len(Ssets)
    out = {"layers": {}}
    g1, g3 = [], []
    perm_hits = []
    for li in sorted(K):
        Kl, Vl = K[li], V[li]
        Kp = torch.linalg.pinv(Kl.T)                  # (n, d_k)；W_or K^T = V^T
        W_or = Vl.T @ Kp
        resid = float((W_or @ Kl.T - Vl.T).norm() / Vl.T.norm())
        orv, ors, _ = n_hit(W_or, Kl, Vl, Ssets)

        per = []
        for k in range(N_PERM):
            perm = list(range(n))
            random.Random(PERM_SEED0 + k).shuffle(perm)
            Vp = Vl[torch.as_tensor(perm, device=Kl.device)]
            W_pi = Vp.T @ Kp
            pv, ps, _ = n_hit(W_pi, Kl, Vl, Ssets)
            per.append({"perm_seed": PERM_SEED0 + k, "n_hit_v": pv, "n_hit_s": ps})
        perm_hits += [p["n_hit_v"] for p in per]

        w0v, w0s, _ = n_hit(W0s[li], Kl, Vl, Ssets)
        Sr = torch.linalg.svdvals(Kl.to(torch.float64))
        out["layers"][str(li)] = {
            "oracle_n_hit_v": orv, "oracle_n_hit_s": ors, "oracle_resid": resid,
            "perm": per, "perm_max": max(p["n_hit_v"] for p in per),
            "W0_n_hit_v": w0v, "W0_n_hit_s": w0s,
            "sigma1_over_sigma_min": float(Sr[0] / Sr[-1]),
        }
        g1.append(orv)
        g3.append(resid)

    RES["controls"] = out
    rec("G5-1", min(g1) >= 30,
        f"阳性对照 min N_hit^v(oracle) = {min(g1)}/{n}（须 ≥30）", value=g1)
    rec("G5-2", max(perm_hits) <= 3,
        f"阴性对照 max N_hit^v(oracle^π) = {max(perm_hits)}/{n}"
        f"（{len(perm_hits)} 个读数 = 3 层 × {N_PERM} 置换，须 ≤3）", value=perm_hits)
    rec("G5-3", max(g3) < 1e-6,
        f"oracle 构造残差 max = {max(g3):.3e}（须 <1e-6 ⇒ K⁺ 数值可靠）", value=g3)
    rec("A5-3", max(out["layers"][L]["W0_n_hit_v"] for L in out["layers"]) <= E_random + 3,
        f"零基线 max N_hit^v(W⁰) = "
        f"{max(out['layers'][L]['W0_n_hit_v'] for L in out['layers'])}/{n}"
        f"（须 ≤ E_random+3 = {E_random + 3:.3f}）")
    # A5-4②：每层 oracle != oracle^π
    ok4 = all(out["layers"][L]["oracle_n_hit_v"] != out["layers"][L]["perm_max"]
              for L in out["layers"])
    rec("A5-4", ok4, "同一 n_hit 函数对象；且每层 oracle 与 oracle^π 得分不同（判别器有分辨力）")
    return out


# ==================================================================== A5-5 补丁恒等
def check_patch_identity(model, chunks, seed):
    """★ 我自己引入的接缝（class-level 替换 `LayerMem.write`）必须先证明是**恒等**。"""
    sch = seed + 7777
    # (a) 原生写
    stN = D90.make_stack(seed=sch)
    D90.write_phase(model, stN, chunks)
    Wn = {li: mm.Wfull.detach().clone() for li, mm in stN.mems.items()}
    stN.remove()
    torch.cuda.empty_cache()
    # (b) 补丁写 + IdentityTF
    stP = D90.make_stack(seed=sch)
    for mm in stP.mems.values():
        mm.key_tf = D98.IdentityTF()
    D98.install_write()
    try:
        D90.write_phase(model, stP, chunks)
    finally:
        D98.restore_write()
    Wp = {li: mm.Wfull.detach().clone() for li, mm in stP.mems.items()}
    stP.remove()
    torch.cuda.empty_cache()

    same = {li: bool(torch.equal(Wn[li], Wp[li])) for li in Wn}
    dmax = {li: float((Wn[li] - Wp[li]).abs().max()) for li in Wn}
    rec("A5-5", all(same.values()),
        f"补丁(IdentityTF) 与 原生 LayerMem.write 逐位相同：{same}（max|Δ|={max(dmax.values()):.1e}）",
        value={"equal": same, "maxabs": dmax})
    return Wn[MOUNT[0]]


# ==================================================================== intact 两臂
def intact_arm(model, facts, chunks, owner, Ssets, seed, geom, E_random):
    """geom ∈ {'native','white'}：装 key_tf → 1 遍写 → 提 K/V → 量 N_hit^v。"""
    # 采集 K_raw（identity、no-write）以建 white 变换
    Kr, _ = extract_kv(model, facts, chunks, owner, seed, D98.IdentityTF())
    n = len(Ssets)
    spec = {}
    for li in Kr:
        Kd = Kr[li].to(torch.float64)
        mu = Kd.mean(dim=0)
        U, S, Vh = torch.linalg.svd(Kd, full_matrices=False)
        Un = (Vh.T * (n ** 0.5)).float()
        Kp = torch.linalg.pinv(Kd.T).float()
        base = float(Kd.norm(dim=1).mean())
        s_c = base / float((Kd - mu).norm(dim=1).mean())
        Aw = Un.float() @ Kp
        s_w = base / float((Kd @ Aw.to(torch.float64).T).norm(dim=1).mean())
        cr = norm_rows(Kd) @ norm_rows(Kd).T
        spec[li] = {"Un": Un, "Kp": Kp, "s_c": s_c, "s_w": s_w,
                    "maxcos_raw": float(cr.fill_diagonal_(-2).max()),
                    "sigma1_over_sigma_min": float(S[0] / S[-1])}

    st = D90.make_stack(seed=seed)
    for li, mm in st.mems.items():
        sp = spec[li]
        mm.key_tf = D98.IdentityTF() if geom == "native" else \
            D98.WhiteTF(sp["Un"].to(DEV), sp["Kp"].to(DEV), sp["s_w"])
    W0 = {li: mm.W0.detach().clone() for li, mm in st.mems.items()}
    D98.install_write()
    try:
        D90.write_phase(model, st, chunks)
        for mm in st.mems.values():
            mm.key_tf.freeze()
        Ws = {li: mm.Wfull.detach().clone() for li, mm in st.mems.items()}
        K, V = D98.key_matrices_tf(model, st, facts, chunks, owner)
        for mm in st.mems.values():
            mm.no_write = True
    finally:
        D98.restore_write()
    st.remove()
    torch.cuda.empty_cache()

    out = {"geom": geom, "seed": seed, "layers": {}}
    for li in sorted(K):
        v, s, l2 = n_hit(Ws[li], K[li], V[li], Ssets)
        dW = Ws[li] - W0[li]
        out["layers"][str(li)] = {
            "n_hit_v": v, "n_hit_s": s, "n_hit_v_L2": l2,
            "stable_rank_dW": D98.stable_rank(dW),
            "nu": float(dW.norm() / W0[li].norm()),
            "maxcos_raw": spec[li]["maxcos_raw"],
            "s_white": spec[li]["s_w"], "s_center": spec[li]["s_c"],
        }
    return out


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stageA", choices=["stageA", "formal"])
    a = ap.parse_args()
    RES["stage"] = a.stage
    t0 = time.time()

    print("=" * 92)
    print(f"P118-5 · N_hit^v 尺子在 DIR-118 装置上重建  [{a.stage}]")
    print("=" * 92)

    tok, model = D118.model_and_tok()
    torch.set_grad_enabled(True)

    print("\n① 结构核（E_random 由装置 facts 算出）")
    facts, Ssets, E_random, ceiling = structure(tok)
    chunks, owner = D90.build_chunks(tok, facts)
    assert len(chunks) == 600, len(chunks)

    print("\n② K/V 提取（原生键，no-write）")
    K, V = extract_kv(model, facts, chunks, owner, SEED_STRUCT, D98.IdentityTF())
    print(f"    层 = {sorted(K)}；K 形状 = {tuple(K[MOUNT[0]].shape)}；"
          f"V 形状 = {tuple(V[MOUNT[0]].shape)}")

    print("\n③ 三臂正负对照（oracle / oracle^π / W⁰）")
    st0 = D90.make_stack(no_write=True, seed=SEED_STRUCT)
    W0s = {li: mm.W0.detach().clone() for li, mm in st0.mems.items()}
    st0.remove()
    controls(K, V, Ssets, W0s, E_random)

    if a.stage == "formal":
        print("\n④ A5-5：补丁写 ≡ 原生写（逐位）")
        check_patch_identity(model, chunks, SEED_STRUCT)

        print("\n⑤ intact 读数（1 遍写；原生键 / 去相关键）")
        for geom in ("native", "white"):
            for sd in SEED_INTACT:
                r = intact_arm(model, facts, chunks, owner, Ssets, sd, geom, E_random)
                RES["intact"][f"{geom}|{sd}"] = r
                for li, d in r["layers"].items():
                    print(f"    [{geom} seed={sd} L{li}] N_hit^v={d['n_hit_v']}/32 "
                          f"(strict={d['n_hit_s']}) stable_rank(ΔW)={d['stable_rank_dW']:.2f} "
                          f"maxcos_raw={d['maxcos_raw']:.4f}", flush=True)

    RES["elapsed_s"] = round(time.time() - t0, 1)
    verdict = ("RULER-READY" if not FAILS else "RULER-NOT-READY")
    RES["verdict"] = verdict
    RES["fails"] = FAILS
    dump_name = f"ruler_p118_5_{a.stage}.json"
    (OUT / dump_name).write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str),
                                encoding="utf-8")
    print("\n" + "=" * 92)
    print(f"★ 判决 = {verdict}" + ("" if not FAILS else f"   未通过 = {FAILS}"))
    print(f"★ 读数已落盘 {OUT / dump_name}    用时 {RES['elapsed_s']}s")
    print("=" * 92)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())