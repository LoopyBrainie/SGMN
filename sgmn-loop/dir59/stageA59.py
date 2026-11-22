#!/opt/pytorch/bin/python
r"""DIR-59 · Stage A 硬门（`A59-1..10`，见 `revolve/DIR-59/DIRECTION_rev59_v10.md §7`）

任一失败 ⇒ 禁止正式轮。产物 `dir59/out/DIR59r10_stageA.json` + `DIR59r10_card_md5.json`。
"""
import hashlib
import json
import math
import pathlib
import sys

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir59")
import R1_formal59 as F      # noqa: E402
import R1_task59 as T59      # noqa: E402
import R1_task37 as T37      # noqa: E402
import R1_task40 as T40      # noqa: E402

OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir59/out")
MD5_FROZEN = {"dir31/R1_grad.py": "8fa40f708600bec7e6c484727d2180c7",
              "dir33/R1_task33.py": "96ef89e87ec61556b52e0e464850087e",
              "dir37/R1_task37.py": "9236ac0490be9c413f1e8af163289207",
              "dir40/R1_task40.py": "baab3566891591944efb60a5b6dc9623"}
HISTORY = [(10000, 10399), (11000, 11549), (12000, 13095), (13500, 14511), (15000, 15899),
           (16000, 16411), (17000, 18911), (20400, 20999), (21000, 21111), (21500, 22063),
           (22100, 22103), (22500, 22899), (23000, 23003), (23990, 24399), (25000, 25019),
           (25200, 26199), (26200, 27199), (27400, 28399), (28800, 28819), (29400, 30419),
           (31000, 31199), (31400, 32419), (33000, 36999), (41000, 41039), (51000, 52999),
           (53000, 53019), (55000, 55023), (56000, 56095), (57000, 57095), (58000, 58463),
           (59000, 59095), (59500, 59595), (59600, 59631), (59800, 60299), (61000, 61007),
           (62000, 62399)]


def md5(p):
    return hashlib.md5(pathlib.Path("/home/ec2-user/sgmn-loop/" + p).read_bytes()).hexdigest()


def sha1t(t):
    return hashlib.sha1(t.detach().numpy().tobytes()).hexdigest()


def main():
    g = {}
    # A59-1 装置 md5
    g["A59-1"] = {p: md5(p) for p in MD5_FROZEN}
    g["A59-1"]["pass"] = all(g["A59-1"][p] == v for p, v in MD5_FROZEN.items())
    # A59-2 卡 md5
    card_md5 = md5("revolve/DIR-59/DIRECTION_rev59_v10.md")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DIR59r10_card_md5.json").write_text(json.dumps(
        {"card_md5": card_md5, "frozen_at": "2026-11-17", "expected": F.CARD_MD5},
        ensure_ascii=False, indent=1), encoding="utf-8")
    g["A59-2"] = {"card_md5": card_md5, "driver_expected": F.CARD_MD5,
                  "pass": card_md5 == F.CARD_MD5}
    # A59-3 头初值确定 ⇒ 四臂相同
    p1, p2 = T40.init_params(), T40.init_params()
    keys = sorted(p1)
    same = all(sha1t(p1[k]) == sha1t(p2[k]) for k in keys)
    g["A59-3"] = {"n_params": len(keys), "keys": keys,
                  "sha1": {k: sha1t(p1[k])[:16] for k in keys}, "pass": same}
    # A59-4/A59-9 每格 Pv 两张不同 + 两两不共线
    pvs = {}
    for c in F.CELLS:
        pvs[c["j"]] = (T59._pv1_for(c["pvA"]), T59._pv1_for(c["pvB"]))
    shas = {j: (hashlib.sha256(a.numpy().tobytes()).hexdigest()[:16],
                hashlib.sha256(b.numpy().tobytes()).hexdigest()[:16]) for j, (a, b) in pvs.items()}
    all_shas = [s for pair in shas.values() for s in pair]
    g["A59-4"] = {"shas": {str(j): v for j, v in shas.items()},
                  "shape": list(pvs[1][0].shape),
                  "n_distinct": len(set(all_shas)),
                  "pass": len(set(all_shas)) == 10 and list(pvs[1][0].shape) == [T59.D_H, T59.D_V]}
    worst = 0.0
    for j, (a, b) in pvs.items():
        worst = max(worst, abs(float((a / a.norm() * (b / b.norm())).sum())))
    for j1 in pvs:
        for j2 in pvs:
            if j1 < j2:
                a, b = pvs[j1][0], pvs[j2][0]
                worst = max(worst, abs(float((a / a.norm() * (b / b.norm())).sum())))
    g["A59-9"] = {"max_abs_cos": worst, "pass": worst < 0.9}
    # A59-5 四臂 Pk1 逐位相同（同格同一值；5 格互异）
    pks = {c["j"]: F.pk1_for(c["pk1"]) for c in F.CELLS}
    g["A59-5"] = {"sha1": {str(j): sha1t(v)[:16] for j, v in pks.items()},
                  "n_distinct": len({sha1t(v) for v in pks.values()}),
                  "pass": len({sha1t(v) for v in pks.values()}) == 5}
    # A59-6 钩子咬住（同一实例、未训练的头）
    phi = T40.init_params()
    pk, pvA, pvB, _ = F.cell_spec(1)
    sd = F.EVAL[0]
    Ls = {}
    for arm in F.ARMS:
        w, t = F.role_pv(arm, pvA, pvB)
        Ls[arm] = float(T59.trace59(T37._mk(sd, phi, False), T59.N_REP, "learned", sd, phi,
                                    shared_pk1=pk, shared_pv1=w, target_pv1=t)[0])
    g["A59-6"] = {"L": Ls,
                  "AA_ne_AB": Ls["AA"] != Ls["AB"], "BA_ne_BB": Ls["BA"] != Ls["BB"],
                  "AA_ne_BA": Ls["AA"] != Ls["BA"]}
    g["A59-6"]["pass"] = g["A59-6"]["AA_ne_AB"] and g["A59-6"]["BA_ne_BB"] and g["A59-6"]["AA_ne_BA"]
    # A59-7 shared_pk1 真的落到 P["Pk1"]
    P = T37._mk(sd, phi, False)
    L_none, _ = T59.trace59(P, T59.N_REP, "learned", sd, phi)
    P2 = T37._mk(sd, phi, False)
    L_pk, _ = T59.trace59(P2, T59.N_REP, "learned", sd, phi, shared_pk1=pk)
    g["A59-7"] = {"L_none": float(L_none), "L_shared": float(L_pk),
                  "sha_match": sha1t(P2["Pk1"]) == sha1t(pk),
                  "pass": (sha1t(P2["Pk1"]) == sha1t(pk)) and float(L_none) != float(L_pk)}
    # A59-8 种子段
    bad = lambda s: any(a <= s <= b for a, b in HISTORY)
    segs = {"train": F.TRAIN, "eval": F.EVAL,
            "pk1": [c["pk1"] for c in F.CELLS],
            "pvA": [c["pvA"] for c in F.CELLS],
            "pvB": [c["pvB"] for c in F.CELLS]}
    g["A59-8"] = {"train_eval_overlap": len(set(F.TRAIN) & set(F.EVAL)),
                  "hits": {k: [s for s in v if bad(s)] for k, v in segs.items()}}
    g["A59-8"]["pass"] = (g["A59-8"]["train_eval_overlap"] == 0
                          and all(len(x) == 0 for x in g["A59-8"]["hits"].values()))
    # A59-10 非有限 + 装置回归
    fin = {arm: math.isfinite(v) for arm, v in Ls.items()}
    reg = T59.selftest()
    g["A59-10"] = {"finite": fin, "A59D_all_pass": reg["all_pass"],
                   "pass": all(fin.values()) and reg["all_pass"]}
    g["all_pass"] = all(v["pass"] for k, v in g.items() if k.startswith("A59-"))
    (OUT / "DIR59r10_stageA.json").write_text(json.dumps(g, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(json.dumps({k: v.get("pass") for k, v in g.items() if k.startswith("A59-")},
                     ensure_ascii=False, indent=1))
    print("★ DIR-59 rev1.0 Stage A 全部通过" if g["all_pass"] else "★★ Stage A 失败")
    return 0 if g["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
