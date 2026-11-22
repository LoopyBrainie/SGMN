#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""
DIR-42 · ★ **正式轮每实例成本实测**（开卡前；用于回填卡 §8 的成本行）

★ **这不是判据读数**：本脚本**不产出任何实验读数**（只打印耗时），
  只测「**正式轮每实例读器的墙钟成本**」⇒ ⇒ 用于把卡 §8 的成本从**外推**变成**实测**。

★ **逐字镜像 `formal_arm` 的每实例工作**（否则测出来的不是正式轮的成本）：
  `train_reader42(arm, sd, steps=STEPS)` ⇒
  `eval_reader42(EVAL_WIN)` + `eval_reader42(TRAIN_WIN)` + `eval_reader42(None)` ⇒
  （`learned` 臂再加 `eval_reader42(EVAL_WIN, rand_content=True)`）。

★ ★ **探针的成本不能直接外推**！探针每实例只做「训练 + **1** 次评估」，
  而正式轮每实例做「训练 + **3**（`learned` 是 **4**）次评估」
  ⇒ ⇒ 用探针外推会**低估**正式轮成本 ⇒ ⇒ 故本脚本专门实测。

用法（在 trainer 上）：
    /opt/pytorch/bin/python dir42/timing42.py --arm learned --reps 3 --tag learned-0
"""
import argparse
import os
import sys
import time

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir42")
import R1_task42 as T            # noqa: E402


def one_instance(arm, si):
    """★ 与 `formal_arm` 的每实例工作**逐字一致**（含三次/四次评估）"""
    sd = T.SEED0 + T.INST_OFF[si]
    phi, _meta = T.train_reader42(arm, sd, steps=T.STEPS)
    e = T.eval_reader42(arm, phi, sd, win=T.EVAL_WIN)
    T.eval_reader42(arm, phi, sd, win=T.TRAIN_WIN)
    T.eval_reader42(arm, phi, sd, win=None)
    if arm == "learned":
        T.eval_reader42(arm, phi, sd, win=T.EVAL_WIN, rand_content=True)
    return float(e["L"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(T.ARMS))
    ap.add_argument("--reps", type=int, default=3, help="本进程跑几个实例")
    ap.add_argument("--start-idx", type=int, default=0, help="起点（实例在 INST_OFF 里的下标）")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    per = []
    t_all = time.time()
    for k in range(a.reps):
        si = a.start_idx + k
        t0 = time.time()
        L = one_instance(a.arm, si)
        dt = time.time() - t0
        per.append(dt)
        print(f"[{a.tag or a.arm}] instance {si}: {dt:.1f}s  (L={L:.4f})", flush=True)
    wall = time.time() - t_all
    print(f"[{a.tag or a.arm}] arm={a.arm} reps={a.reps} wall={wall:.1f}s "
          f"per_inst={wall / max(a.reps, 1):.1f}s", flush=True)


if __name__ == "__main__":
    main()
