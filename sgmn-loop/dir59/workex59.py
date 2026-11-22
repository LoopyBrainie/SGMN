#!/opt/pytorch/bin/python
r"""DIR-59 工作例（卡片 §2.3 用）：seed `59700` 的三臂读数 + **独立重实现**逐位核对

★ 独立重实现 = 不调用 `probe59a.rollout_measure`，从 `dir33` 的公式**重写** 16 步前向 + 写更新 + 读数，
  只为卡住"卡里的测量对象"与"探针的测量对象"是同一个东西（`A59-11`）。
产物：`dir59/out/WORKEX59.json`
"""
import hashlib
import json
import pathlib
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir59")
import R1_grad as M          # noqa: E402
import R1_task33 as T33      # noqa: E402
import probe59a as A         # noqa: E402

T = T33.T
D_H = T33.D_H
OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir59/out")
SD = 59700


def indep(P_src, P_tgt, alpha=1.0):
    r"""独立重写（**不调用** `rollout_measure`）：每步 feedforward 两层 + 写更新 + 读数。

    ★ 与探针实现的两处"易错点"在这里**显式照抄**：① 第 2 层的输入与 $c$ 都是**本步更新后的第 1 层输出**；
      ② `U,V` 的更新**都**用**更新前**的因子（对称式），不得用已更新的 `U` 去算 `V`。
    """
    U1, V1 = P_src["U1"].clone(), P_src["V1"].clone()
    U2, V2 = P_src["U2"].clone(), P_src["V2"].clone()
    hs = T33.hs_of(P_tgt)
    gs = [torch.tanh(hs[t - 1] @ P_tgt["Pv1"]) for t in range(1, T)]
    K, Vw, G = [], [], []
    hprev1 = hprev2 = torch.zeros(D_H)
    for t in range(T):
        c1 = P_src["a"][t]
        h1 = torch.tanh(c1 @ P_src["A1"] + P_src["b1"])
        W1 = M._W(U1, V1)
        m1 = torch.tanh(torch.cat([c1, W1 @ torch.tanh(c1 @ P_src["Wq1"] + P_src["bq1"])], -1)
                        @ P_src["Wo1"] + P_src["bo1"])
        hp1 = h1 + P_src["g1"] * m1
        c2 = hp1
        h2 = torch.tanh(c2 @ P_src["A2"] + P_src["b2"])
        W2 = M._W(U2, V2)
        m2 = torch.tanh(torch.cat([c2, W2 @ torch.tanh(c2 @ P_src["Wq2"] + P_src["bq2"])], -1)
                        @ P_src["Wo2"] + P_src["bo2"])
        _hp2 = h2 + P_src["g2"] * m2                      # noqa: F841（结构对照用）
        k1 = torch.tanh(hprev1 @ P_src["Pk1"])
        v1 = torch.tanh(h1 @ P_src["Pv1"])
        G1 = alpha * torch.outer(W1 @ k1 - v1, k1) + M.LAM * W1
        U1n = U1 - M.ETA * (G1 @ V1)
        V1n = V1 - M.ETA * (G1.t() @ U1)
        k2 = torch.tanh(hprev2 @ P_src["Pk2"])
        v2 = torch.tanh(h2 @ P_src["Pv2"])
        G2 = alpha * torch.outer(W2 @ k2 - v2, k2) + M.LAM * W2
        U2n = U2 - M.ETA * (G2 @ V2)
        V2n = V2 - M.ETA * (G2.t() @ U2)
        U1, V1, U2, V2 = U1n, V1n, U2n, V2n
        if t >= 1:
            K.append(k1)
            Vw.append(v1)
            G.append(gs[t - 1])
        hprev1, hprev2 = h1, h2
    W1f = M._W(U1, V1)
    return {"K1": K, "V1w": Vw, "W1": W1f, "gs": G, "U1": U1, "V1": V1}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P_own, P_for = T33.build(SD, "V0"), T33.build(SD + 1, "V0")
    un = A.null_keys(P_own, P_for)
    res, ind, arms = {}, {}, {}
    for name, src, al in (("OWN", P_own, 1.0), ("FOREIGN", P_for, 1.0), ("NONE", P_own, 0.0)):
        Pw = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in P_own.items()}
        if name == "FOREIGN":
            Pw["Pv1"] = P_for["Pv1"]      # ★ 只换第 1 层的值映射
        r = A.rollout_measure(Pw, P_own, SD, alpha=al)
        res[name] = r
        arms[name] = A.measure(r, un)
        q = indep(Pw, P_own, alpha=al)
        ind[name] = {"max_abs_dW": float((r["W1"] - q["W1"]).abs().max()),
                     "max_abs_dU1": float((r["W1"] - q["W1"]).abs().max())}
        # 独立读数（同一公式、独立计算）
        e_own = statistics.fmean([float((q["W1"] @ k - g).norm()) for k, g in zip(q["K1"], q["gs"])])
        e_wr = statistics.fmean([float((q["W1"] @ k - v).norm()) for k, v in zip(q["K1"], q["V1w"])])
        ind[name]["E_own"] = e_own
        ind[name]["E_written"] = e_wr
        ind[name]["dE_own"] = abs(e_own - arms[name]["E_own"])
        ind[name]["dE_written"] = abs(e_wr - arms[name]["E_written"])

    Vown = [torch.tanh(h @ P_own["Pv1"]) for h in T33.hs_of(P_own)]
    Vfor = [torch.tanh(h @ P_for["Pv1"]) for h in T33.hs_of(P_own)]
    cs = [float(torch.dot(a, b) / (a.norm() * b.norm())) for a, b in zip(Vown, Vfor)]

    out = {"seed": SD, "T": T, "arms": arms,
           "indep": ind,
           "bitwise": {n: bool(ind[n]["max_abs_dW"] == 0.0) for n in res},
           "cos_v": {"per_t": cs, "med": statistics.fmean(cs),
                     "max_abs": max(abs(c) for c in cs)},
           "sha_Pv1_own": hashlib.sha256(P_own["Pv1"].numpy().tobytes()).hexdigest()[:16],
           "sha_Pv1_for": hashlib.sha256(P_for["Pv1"].numpy().tobytes()).hexdigest()[:16],
           "md5_dir31": A.md5("/home/ec2-user/sgmn-loop/dir31/R1_grad.py"),
           "md5_dir33": A.md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py")}
    (OUT / "WORKEX59.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("★ 工作例 seed %d（T=%d）" % (SD, T))
    for n in ("OWN", "FOREIGN", "NONE"):
        a = arms[n]
        print("   %-8s E_own %.17g · E_written %.17g · E_null %.17g · dW %.6f"
              % (n, a["E_own"], a["E_written"], a["E_null"], a["dW"]))
        print("            独立重实现: E_own %.17g (|Δ| %.1e) · E_written %.17g (|Δ| %.1e) · max|ΔW| %.1e"
              % (ind[n]["E_own"], ind[n]["dE_own"], ind[n]["E_written"], ind[n]["dE_written"],
                 ind[n]["max_abs_dW"]))
    print("★ cos(v_own,v_foreign)：逐拍 %s · 均值 %.6f · max|cos| %.6f"
          % ("[" + ", ".join("%.3f" % c for c in cs) + "]", out["cos_v"]["med"], out["cos_v"]["max_abs"]))
    print("★ Pv1 sha own/foreign: %s / %s" % (out["sha_Pv1_own"], out["sha_Pv1_for"]))
    print("★ 产物 → dir59/out/WORKEX59.json")


if __name__ == "__main__":
    main()