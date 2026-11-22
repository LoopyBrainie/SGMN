#!/opt/pytorch/bin/python
r"""DIR-34 开卡前标定（决定二十九）：**出样本评估**下的内容性收益有多大？

**问题**：DIR-32/33 的四臂阶梯里，读数头 `Wo1,bo1,g1` 是**在用来评估的那 24 个种子上训练**的
（`train_arm(seeds=...)` 随后 `per_seed(seeds=...)` 用同一组）。目标 $g_t=v(h^{\rm nom}_{t-1})$
里的 $P_v,A,b$ **逐实例随机** ⇒ 目标逐实例特定；而共享读数头只看到 $c=h^{\rm nom}_t$。
⇒ 它**有可能把 48 个（实例 × 相位）的目标在样本内背下来** ⇒ 记忆内容对损失下降**可能不是必需的**。

**本标定做的事**：同一训练只做一次，然后**同时**报两套读数：
  - **样本内**（训练种子）—— 与 DIR-32/33 同口径；
  - **出样本**（从未见过的 24 个种子）—— 读数头无法背诵。
两套之差 = **样本内背诵的幅度**。另报「地板」$Z=\sum_t\lVert g_t\rVert^2$
（读数头输出恒 0 时的损失下界）。

种子段：**600..623**（训练）· **700..723**（出样本）—— 与 100/200/300/400/500 段全不相交（纪律 70）。
**本文件不进任何判据**，只供开卡定阈值（卡 rev 1.0 在本文件跑完之后才冻结）。
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
HELD = [R.M.SEED0 + 700 + i for i in range(24)]
assert not (set(TRAIN) & set(HELD))


def floor_Z(seeds):
    """读数头输出恒 0 ⇒ 损失 = Σ_t‖g_t‖²（逐实例后取均值）。"""
    zs = []
    for sd in seeds:
        P = R.build(sd, "V0")
        R.period2(P, sd)
        gs = R.targets(P)
        zs.append(sum(float((g ** 2).sum()) for g in gs))
    return statistics.fmean(zs)


def main():
    res = {"train_seeds": TRAIN, "held_seeds": HELD}
    res["floor_Z"] = {"train": floor_Z(TRAIN), "held": floor_Z(HELD)}
    res["arms"] = {}
    for addr in ("oracle", "device"):
        for arm in ("N", "S", "T"):
            t0 = time.time()
            phi = R.train_arm(addr, arm, seeds=TRAIN)
            ins = R.per_seed(addr, arm, phi, TRAIN)
            out = R.per_seed(addr, arm, phi, HELD)
            res["arms"][f"{addr}/{arm}"] = {
                "in_sample": statistics.fmean(ins), "out_sample": statistics.fmean(out),
                "in_sd": statistics.stdev(ins), "out_sd": statistics.stdev(out),
                "gamma": float(phi["g1"]), "secs": round(time.time() - t0, 1),
            }
            print(f"{addr:7s} {arm}: 样本内 {res['arms'][f'{addr}/{arm}']['in_sample']:8.4f}"
                  f" · 出样本 {res['arms'][f'{addr}/{arm}']['out_sample']:8.4f}"
                  f" · γ={res['arms'][f'{addr}/{arm}']['gamma']:.3f}"
                  f" · {res['arms'][f'{addr}/{arm}']['secs']:.0f}s", flush=True)
        # G0（无注入、无训练）
        g0i = R.per_seed(addr, "G0", {}, TRAIN)
        g0o = R.per_seed(addr, "G0", {}, HELD)
        res["arms"][f"{addr}/G0"] = {"in_sample": statistics.fmean(g0i),
                                     "out_sample": statistics.fmean(g0o)}
        print(f"{addr:7s} G0: 样本内 {statistics.fmean(g0i):8.4f} · 出样本 {statistics.fmean(g0o):8.4f}", flush=True)
    json.dump(res, open(f"{OUT}/DIR34_calib.json", "w"), indent=1)
    print("\n── 分解（出样本 regime）──")
    for addr in ("oracle", "device"):
        g0 = res["arms"][f"{addr}/G0"]["out_sample"]
        n = res["arms"][f"{addr}/N"]["out_sample"]
        s = res["arms"][f"{addr}/S"]["out_sample"]
        t = res["arms"][f"{addr}/T"]["out_sample"]
        print(f"{addr:7s} Δchan {g0-n:+8.4f} · Δrand {n-s:+8.4f} · Δcontent {s-t:+8.4f}"
              f"  （地板 Z={res['floor_Z']['held']:.4f}）")


if __name__ == "__main__":
    main()
