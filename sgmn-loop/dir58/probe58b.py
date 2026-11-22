#!/opt/pytorch/bin/python
r"""DIR-58 前提探针 B —— **时间衰减与「写入驱逐」能否在同一读出量上分离**（框架 §7.2「时间与情节状态 · S 的…遗忘」）。

问法（不含判据）：写完 1 对后，
  臂 D（只衰减）：$q$ 对 $\alpha{=}0$ 步 ⇒ $S_{\rm old}^{D}(q)$；
  臂 W（1 条等量异值写入 + 剩余 $\alpha{=}0$）：总步数与 D **相同** ⇒ $S_{\rm old}^{W}(q)$、$S_{\rm new}^{W}(q)$。
  ⇒ 若存在 $q^\ast$ 使 $S_{\rm old}^{D}(q^\ast)=S_{\rm old}^{W}(q)$（$0\le q^\ast\le8$），则**衰减可以冒充一次写入**，
     "寿命分解"在这个读出量上**没有自由度**；若 $S_{\rm old}^{W}$ 落在 $S_{\rm old}^{D}$ 的整个可及区间**之外**，则两者**可分离**。
★ 读出量：$E=\|Wk-v\|$、$E_{\rm null}$=5 把未写键中位、$S=E_{\rm null}-E$（D-79 纪律 140：**不用 `cos`**）。
"""
import json
import os
import statistics as st
import sys

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir51"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task48 as T48      # noqa: E402
import R1_task51 as T51      # noqa: E402

OUT = os.path.join(ROOT, "dir58", "out")
OFF_K, OFF_V, OFF_R, OFF_I = 59501, 59502, 59503, 59504
SEEDS = list(range(59000, 59032))
QS = [0, 1, 2, 4, 8]


def key_batch(g, m, max_cos=0.30):
    ks = []
    while len(ks) < m:
        k = R.KEY_SCALE * T51.gen_vec(g, M.D_K)
        if all(abs(float((k @ q) / (k.norm() * q.norm()))) <= max_cos for q in ks):
            ks.append(k)
    return ks


def read_errors(U, Vm, ks, vP):
    W = M._W(U, Vm)
    return [float((W @ k - vP).norm()) for k in ks]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sd in SEEDS:
        P = R.build(sd, "V0")
        gk = torch.Generator().manual_seed(sd + OFF_K)
        gv = torch.Generator().manual_seed(sd + OFF_V)
        gr = torch.Generator().manual_seed(sd + OFF_R)
        gi = torch.Generator().manual_seed(sd + OFF_I)
        ks = key_batch(gk, 4)
        vP = T51.gen_vec(gv, M.D_V)
        kUn = key_batch(gr, 5)
        kI = key_batch(gi, 1)[0]
        vI = None
        while vI is None:
            v = T51.gen_vec(gi, M.D_V)
            if abs(float(v @ vP)) <= 0.30:
                vI = v
        U1, V1 = T48.step_pair(P["U1"].clone(), P["V1"].clone(), ks[0], vP, 1.0)
        rec = {"seed": sd, "d": {}, "w": {}, "v_cos": abs(float(vI @ vP))}
        for q in QS:                                    # 臂 D：只衰减
            U, Vm = U1.clone(), V1.clone()
            for _ in range(q):
                U, Vm = T48.step_pair(U, Vm, ks[0], vP, 0.0)
            E = read_errors(U, Vm, [ks[0]], vP)[0]
            rec["d"]["q%d" % q] = {"S_old": st.median(read_errors(U, Vm, kUn, vP)) - E}
        for q in QS:                                    # 臂 W：1 条异值写入 + (q−1) 对衰减，总步数相同
            U, Vm = U1.clone(), V1.clone()
            U, Vm = T48.step_pair(U, Vm, kI, vI, 1.0)
            for _ in range(max(q - 1, 0)):
                U, Vm = T48.step_pair(U, Vm, ks[0], vP, 0.0)
            En = st.median(read_errors(U, Vm, kUn, vP))
            rec["w"]["q%d" % q] = {"S_old": En - read_errors(U, Vm, [ks[0]], vP)[0],
                                  "S_new": En - read_errors(U, Vm, [kI], vI)[0],
                                  "E_null": En}
        rows.append(rec)

    def med(f, q):
        return st.median([r[f][q]["S_old"] for r in rows])

    res = {"probe": "58b", "n_seeds": len(rows), "qs": QS,
           "curve_D_med": {q: med("d", "q%d" % q) for q in QS},
           "curve_W_med": {q: med("w", "q%d" % q) for q in QS},
           "S_new_med": {q: st.median([r["w"]["q%d" % q]["S_new"] for r in rows]) for q in QS},
           "E_null_W_med": {q: st.median([r["w"]["q%d" % q]["E_null"] for r in rows]) for q in QS},
           "S_new_pos_frac": {q: sum(1 for r in rows if r["w"]["q%d" % q]["S_new"] > 0) / len(rows) for q in QS},
           "cos_vI_vP_max": max(r["v_cos"] for r in rows),
           "rows": rows}
    # q*：用 D 曲线线性内插，找 S_old^W(q) 落在 D 曲线上的位置
    D = [res["curve_D_med"][q] for q in QS]
    star = {}
    for q in QS:
        w = res["curve_W_med"][q]
        qs_ = None
        for i in range(len(D) - 1):
            if D[i] >= w >= D[i + 1]:
                qs_ = QS[i] + (QS[i + 1] - QS[i]) * (D[i] - w) / (D[i] - D[i + 1])
                break
        star["q%d" % q] = qs_
    res["q_star"] = star
    res["summary"] = {
        "D_at_q0": D[0], "D_at_q8": D[-1],
        "W_at_q8": res["curve_W_med"][8],
        "W_below_D_range": res["curve_W_med"][8] < res["curve_D_med"][8],
        "S_new_positive_all_q": all(res["S_new_pos_frac"][q] == 1.0 for q in QS),
        "S_new_at_q8_med": res["S_new_med"][8]}
    with open(os.path.join(OUT, "PROBE58B.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("探针 58B：%d 种子 · |cos(v_I,v_P)| max = %.4f" % (len(rows), res["cos_vI_vP_max"]))
    print(" q   S_old(只衰减)   S_old(1写入+q−1衰减)   q*     S_new(新键)   S_new>0")
    for q in QS:
        print("%2d     %+8.4f          %+8.4f        %5s     %+8.4f      %.2f"
              % (q, res["curve_D_med"][q], res["curve_W_med"][q],
                 ("%.2f" % star["q%d" % q]) if star["q%d" % q] is not None else "外",
                 res["S_new_med"][q], res["S_new_pos_frac"][q]))
    print("── S_new 在所有 q 上都 >0：%s · q8 时 S_new 中位 %+.4f"
          % (res["summary"]["S_new_positive_all_q"], res["summary"]["S_new_at_q8_med"]))


if __name__ == "__main__":
    main()