#!/opt/pytorch/bin/python
r"""DIR-34 开卡前 pilot（决定二十九，**不进判据**）：**配对** Δ 的 se 与出样本功效。

**为什么必须再跑一次标定**：DIR-34 的主判据是**配对**差 $\Delta_{\rm content}=L^\ast(S)-L^\ast(T)$
（同一种子上两臂之差 ⇒ 实例难度抵消）。标定 `calib34.py` 只存了各臂的**边际** sd，
边际 sd 之和（18.69）给出的 3se（11.4）**严重高估**了配对差的不确定度 ⇒ 必须直接算配对 se。

本 pilot **复现**标定的训练/评估种子（600..623 / 700..723）⇒ 同时是一次**可复现性检查**
（两次独立运行应给出逐位相同的读数）。输出逐种子列表，供开卡时：
  ① 定主判据的 5%×L*(G0) 阈值与**功效**条款；
  ② 定 H34c（样本内溢价）的阈值；
  ③ 定 P4（出样本隔离）阈值。
"""
import json
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
import R1_task33 as R

OUT = "/home/ec2-user/sgmn-loop/dir34/out"
TRAIN = [R.M.SEED0 + 600 + i for i in range(24)]
EVAL = [R.M.SEED0 + 700 + i for i in range(24)]


def floor_Z(seeds):
    zs = []
    for sd in seeds:
        P = R.build(sd, "V0")
        R.period2(P, sd)
        zs.append(sum(float((g ** 2).sum()) for g in R.targets(P)))
    return zs


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m = statistics.fmean(d)
    sd = statistics.stdev(d)
    se = sd / len(d) ** 0.5
    return {"mean": m, "sd": sd, "se": se, "3se": 3 * se}


def main():
    res = {"train": TRAIN, "eval": EVAL}
    res["Z_train"] = floor_Z(TRAIN)
    res["Z_eval"] = floor_Z(EVAL)
    out = {}
    for addr in ("oracle",):
        out[addr] = {}
        for arm in ("N", "S", "T"):
            t0 = time.time()
            phi = R.train_arm(addr, arm, seeds=TRAIN)
            ins = R.per_seed(addr, arm, phi, TRAIN)
            evs = R.per_seed(addr, arm, phi, EVAL)
            out[addr][arm] = {"in": ins, "out": evs, "gamma": float(phi["g1"]),
                              "secs": round(time.time() - t0, 1)}
            print(f"{addr:7s} {arm}: 样本内 {statistics.fmean(ins):8.4f} · 出样本 {statistics.fmean(evs):8.4f}"
                  f" · γ={float(phi['g1']):.3f} · {time.time()-t0:.0f}s", flush=True)
        g0i = R.per_seed(addr, "G0", {}, TRAIN)
        g0o = R.per_seed(addr, "G0", {}, EVAL)
        out[addr]["G0"] = {"in": g0i, "out": g0o}
        Ztr, Ze = statistics.fmean(res["Z_train"]), statistics.fmean(res["Z_eval"])
        g0 = statistics.fmean(g0o)
        dc = paired(out[addr]["S"]["out"], out[addr]["T"]["out"])
        dr = paired(out[addr]["N"]["out"], out[addr]["S"]["out"])
        dc_in = paired(out[addr]["S"]["in"], out[addr]["T"]["in"])
        thr = 0.05 * g0
        print(f"\n  L*(G0,eval) {g0:.5f} ⇒ 5% 阈值 {thr:.5f}")
        print(f"  Δcontent(out) {dc['mean']:+.5f} sd {dc['sd']:.5f} 3se {dc['3se']:.5f}"
              f" ⇒ 效应/阈 {dc['mean']/thr:.2f}× · 功效{'够' if dc['3se']<=thr else '**不够**'}")
        print(f"  Δrand(out) {dr['mean']:+.5f} sd {dr['sd']:.5f}   Δcontent(in) {dc_in['mean']:+.5f}")
        print(f"  地板 Z: train {Ztr:.4f} · eval {Ze:.4f}")
        print(f"  N(out)/Z {statistics.fmean(out[addr]['N']['out'])/Ze:.3f} · N(in)/Z {statistics.fmean(out[addr]['N']['in'])/Ztr:.3f}")
        print(f"  样本内溢价 N: {statistics.fmean(out[addr]['N']['out'])-statistics.fmean(out[addr]['N']['in']):+.5f}"
              f" vs 0.25×L*(G0,train)={0.25*statistics.fmean(g0i):.5f}")
        res["summary"] = {"L_G0_eval": g0, "thr5": thr, "dcontent_out": dc, "drand_out": dr,
                           "dcontent_in": dc_in, "Z_train": Ztr, "Z_eval": Ze,
                           "N_out_over_Z": statistics.fmean(out[addr]["N"]["out"]) / Ze,
                           "N_in_over_Z": statistics.fmean(out[addr]["N"]["in"]) / Ztr}
    res["arms"] = out
    json.dump(res, open(f"{OUT}/DIR34_pilot.json", "w"), indent=1)
    print("\n已写出 DIR34_pilot.json")


if __name__ == "__main__":
    main()
