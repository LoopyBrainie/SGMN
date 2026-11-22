#!/opt/pytorch/bin/python
r"""DIR-59 · 正式轮驱动（rev 1.0 卡冻结；`DIRECTION_rev59_v10.md §2/§3/§6`）

5 格 × 4 臂（`AA/AB/BA/BB`）= 20 次训练；`AA` 臂额外做一次 `rand_content` 评估（不训练）。
唯一自变量 = **哪一张 $P_v$ 进写步、哪一张进任务目标**（§2.2）。`Pk1` 四臂逐位共享。
"""
import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir59")
import R1_task59 as T59        # noqa: E402
import R1_task33 as R          # noqa: E402
import R1_task40 as T40        # noqa: E402

OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir59/out")
CARD = "/home/ec2-user/sgmn-loop/revolve/DIR-59/DIRECTION_rev59_v10.md"
CELLS = [{"j": 1, "pk1": 20263331, "pvA": 20263401, "pvB": 20263411},
         {"j": 2, "pk1": 20263332, "pvA": 20263402, "pvB": 20263412},
         {"j": 3, "pk1": 20263333, "pvA": 20263403, "pvB": 20263413},
         {"j": 4, "pk1": 20263334, "pvA": 20263404, "pvB": 20263414},
         {"j": 5, "pk1": 20263335, "pvA": 20263405, "pvB": 20263415}]
ARMS = ("AA", "AB", "BA", "BB")
TRAIN = list(range(63000, 63096))     # 96
EVAL = list(range(63100, 63500))      # 400
CARD_MD5 = "339ddd7545d96f6398c9d17f9cd59270"


def pk1_for(seed):
    """与 `dir53/R1_task53.py::pk1_for` 逐字同式（`randn(D_H,D_K)*s_h*KEY_SCALE`）。"""
    g = torch.Generator().manual_seed(int(seed))
    s_h = 1.0 / math.sqrt(T59.D_H)
    return torch.randn(T59.D_H, T59.D_K, generator=g) * s_h * R.KEY_SCALE


def cell_spec(j):
    c = CELLS[j - 1]
    return pk1_for(c["pk1"]), T59._pv1_for(c["pvA"]), T59._pv1_for(c["pvB"]), c


def role_pv(arm, pvA, pvB):
    """返回 (写步用, 任务目标用)。"""
    return {"AA": (pvA, pvA), "AB": (pvA, pvB), "BA": (pvB, pvA), "BB": (pvB, pvB)}[arm]


def run(j, arm):
    t0 = time.time()
    pk, pvA, pvB, c = cell_spec(j)
    pv_w, pv_t = role_pv(arm, pvA, pvB)
    tag = f"c{j}_{arm}"
    print(f"[{tag}] 训练：写源={'A' if arm in ('AA','AB') else 'B'}，"
          f"目标={'A' if arm in ('AA','BA') else 'B'}，pk1={c['pk1']}，"
          f"pvA={c['pvA']}，pvB={c['pvB']}", flush=True)
    phi, meta = T59.train_arm59("learned", TRAIN, shared_pk=pk, shared_pv=pv_w, alpha=None,
                                train_ctrl=True, target_pv=pv_t)
    a = T59.eval_arm59("learned", phi, EVAL, shared_pk=pk, shared_pv=pv_w, target_pv=pv_t)
    print(f"[{tag}] L={a['mean_L']:.6f} (sd {a['sd_L']:.4f}) |q|={a['q_norm']:.4f} "
          f"|y|={a['y_norm']:.4f} Wmax={a['Wmax']:.4f}  {time.time()-t0:.0f}s", flush=True)
    rec = {"card_md5": CARD_MD5, "j": j, "arm": arm,
           "seeds": {"pk1": c["pk1"], "pvA": c["pvA"], "pvB": c["pvB"]},
           "role": {"write": "A" if arm in ("AA", "AB") else "B",
                    "target": "A" if arm in ("AA", "BA") else "B"},
           "protocol": {"qsrc": "learned", "tctrl": True, "alpha": 1.0,
                        "steps": T59.STEPS, "lr": T59.LR, "n_rep": T59.N_REP,
                        "n_train": len(TRAIN), "n_eval": len(EVAL),
                        "train_off": [TRAIN[0], TRAIN[-1]], "eval_off": [EVAL[0], EVAL[-1]]},
           "sha1_pk1": hashlib.sha1(pk.numpy().tobytes()).hexdigest(),
           "sha256_pvA": hashlib.sha256(pvA.numpy().tobytes()).hexdigest(),
           "sha256_pvB": hashlib.sha256(pvB.numpy().tobytes()).hexdigest(),
           "mean_L": a["mean_L"], "sd_L": a["sd_L"], "L_series": a["L_series"],
           "q_norm": a["q_norm"], "y_norm": a["y_norm"], "m_abs": a["m_abs"],
           "Wmax": a["Wmax"], "cos_q_k": a["cos_q_k"], "align_q_k": a["align_q_k"],
           "c_eq_h_max": a["c_eq_h_max"], "gradnorms": meta["gradnorms"],
           "dWq1_from_init": meta["dWq1_from_init"], "dbq1_from_init": meta["dbq1_from_init"],
           "secs": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    if arm == "AA":
        rc = T59.eval_arm59("learned", phi, EVAL, shared_pk=pk, shared_pv=pv_w,
                            rand_content=True)
        rec["rc"] = {"mean_L": rc["mean_L"], "sd_L": rc["sd_L"], "L_series": rc["L_series"]}
        print(f"[{tag}] rc（内容置换）L={rc['mean_L']:.6f} (sd {rc['sd_L']:.4f})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"DIR59r10_{tag}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(f"[{tag}] → DIR59r10_{tag}.json", flush=True)


def smoke():
    r"""Stage 4 · 只证明协议可执行（**不产出判据读数**）：小预算跑通（4 步 / 3 训 / 5 评估），
    产物写 `DIR59r10_smoke_*.json`（裁决器**不读**该文件名）。"""
    import statistics as st
    pk, pvA, pvB, c = cell_spec(1)
    tr, ev = TRAIN[:3], EVAL[:5]
    for arm in ARMS:
        w, t = role_pv(arm, pvA, pvB)
        phi, meta = T59.train_arm59("learned", tr, shared_pk=pk, shared_pv=w, alpha=None,
                                    train_ctrl=True, steps=4, target_pv=t)
        a = T59.eval_arm59("learned", phi, ev, shared_pk=pk, shared_pv=w, target_pv=t)
        assert math.isfinite(a["mean_L"]), f"smoke 非有限：{arm}"
        print(f"[smoke c1/{arm}] L={a['mean_L']:.4f} (n_eval={len(a['L_series'])})", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"DIR59r10_smoke_c1_{arm}.json").write_text(json.dumps(
            {"smoke": True, "arm": arm, "mean_L": a["mean_L"], "n_eval": len(a["L_series"]),
             "steps": 4, "n_train": 3, "card_md5": CARD_MD5}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    print("★ smoke 通过（协议可执行；不产出判据读数）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int); ap.add_argument("--arm")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        smoke()
    elif a.cell and a.arm:
        assert a.arm in ARMS
        run(a.cell, a.arm)
    else:
        ap.print_help()
