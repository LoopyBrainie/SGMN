#!/opt/pytorch/bin/python
r"""PROBE60B · $P_v$ 尺度剂量-响应（`PROBE60B.md` rev 1.0，md5 `539a84d9…`）。

**逐字派生**自 `dir60/probe60a.py`：装置、格、种子、协议、训练/评估函数全部不动；
唯一改动 = **测量层**（单剂量 `SCALE2` → 剂量集 `DOSES`）与**独立变量**（尺度 $s$）。
基线（$s{=}1$ 的 `RAND`、$s{=}2$ 的 `SCALE2`、`NONE`）直接复用 `PROBE60_*.json`，本脚本**不重跑**它们。
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir59")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir60")
import R1_task59 as T59        # noqa: E402
import R1_formal59 as F59      # noqa: E402
import probe60a as P60         # noqa: E402  （只为逐位对齐基线：RAND / SCALE2 / 装置 md5）

OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir60/out")
PROBE = "/home/ec2-user/sgmn-loop/dir60/PROBE60B.md"
BASE_PROBE = "/home/ec2-user/sgmn-loop/dir60/PROBE60.md"
BASE_PROBE_MD5 = "4ecdb04bca2f83c486ebac69dd9adbf5"
CELLS = P60.CELLS                      # 同两格（pk1 = 20263351 / 20263352）
PV_SEED = P60.PV_SEED                  # 同 20263431（共享映射）
TRAIN, EVAL = P60.TRAIN, P60.EVAL      # 同段：64000..64095 / 64100..64499
DOSES = (0.5, 1.5, 3.0, 4.0)           # ★ 冻结剂量集（s=1、s=2 复用 PROBE60）


def probe_md5():
    return hashlib.md5(pathlib.Path(PROBE).read_bytes()).hexdigest()


def base_pv():
    return T59._pv1_for(PV_SEED)       # 与 probe60a.pv_variants()["RAND"] 同一构造


def pv_dose(s):
    return base_pv() * float(s)


def run(j, s):
    t0 = time.time()
    pk = F59.pk1_for(CELLS[j - 1]["pk1"])
    pv = pv_dose(s)
    tag = f"c{j}_s{s:g}"
    print(f"[{tag}] P_v = {s:g}×base（写步与目标同源）pk1={CELLS[j-1]['pk1']} pv_seed={PV_SEED}",
          flush=True)
    phi, meta = T59.train_arm59("learned", TRAIN, shared_pk=pk, shared_pv=pv, alpha=None,
                                train_ctrl=True, target_pv=pv)
    a = T59.eval_arm59("learned", phi, EVAL, shared_pk=pk, shared_pv=pv, alpha=None,
                       target_pv=pv)
    print(f"[{tag}] L={a['mean_L']:.6f} (sd {a['sd_L']:.4f}) cos={a['cos_q_k']:.4f} "
          f"{time.time()-t0:.0f}s", flush=True)
    rec = {"probe_md5": probe_md5(), "probe": "PROBE60B", "j": j, "s": float(s),
           "arm": f"s{s:g}", "pv_seed": PV_SEED, "pk1_seed": CELLS[j - 1]["pk1"],
           "alpha": None,
           "protocol": {"qsrc": "learned", "tctrl": True, "steps": T59.STEPS, "lr": T59.LR,
                        "n_rep": T59.N_REP, "n_train": len(TRAIN), "n_eval": len(EVAL),
                        "train_off": [TRAIN[0], TRAIN[-1]], "eval_off": [EVAL[0], EVAL[-1]],
                        "pv_roles": "write=target（同一张）"},
           "sha256_pv": hashlib.sha256(pv.numpy().tobytes()).hexdigest(),
           "sha1_pk1": hashlib.sha1(pk.numpy().tobytes()).hexdigest(),
           "pv_fro": float(pv.norm()),
           "pv_gram_offdiag_max": float((pv.T @ pv
                                         - torch.diag(torch.diagonal(pv.T @ pv))).abs().max()),
           "pv_rank": int(torch.linalg.matrix_rank(pv)),
           "mean_L": a["mean_L"], "sd_L": a["sd_L"], "L_series": a["L_series"],
           "q_norm": a["q_norm"], "y_norm": a["y_norm"], "m_abs": a["m_abs"],
           "Wmax": a["Wmax"], "cos_q_k": a["cos_q_k"], "gradnorms": meta["gradnorms"],
           "dWq1_from_init": meta["dWq1_from_init"],
           "secs": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"PROBE60B_{tag}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(f"[{tag}] → PROBE60B_{tag}.json", flush=True)


def stageA():
    ok = {}
    base, P60V = P60.pv_variants()
    # A60B-1：s=1 / s=2 与 PROBE60 的 RAND / SCALE2 逐位相同
    d1 = float((pv_dose(1.0) - P60V["RAND"]).abs().max())
    d2 = float((pv_dose(2.0) - P60V["SCALE2"]).abs().max())
    ok["A60B-1"] = {"max_abs_s1": d1, "max_abs_s2": d2, "pass": d1 == 0.0 and d2 == 0.0}
    # A60B-2/3/4：尺度算例（fro、Gram、rank）
    fb = float(base.norm())
    rows = {}
    for s in DOSES:
        v = pv_dose(s)
        f = float(v.norm())
        g = float((v.T @ v - (s ** 2) * (base.T @ base)).abs().max())
        rows[f"s{s:g}"] = {"fro_ratio": f / (s * fb), "gram_absdiff": g,
                           "rank": int(torch.linalg.matrix_rank(v))}
    ok["A60B-2"] = {"rows": rows, "pass": all(abs(r["fro_ratio"] - 1) < 1e-6 for r in rows.values())}
    ok["A60B-3"] = {"pass": all(r["gram_absdiff"] < 1e-6 for r in rows.values())}
    ok["A60B-4"] = {"pass": all(r["rank"] == 4 for r in rows.values())}
    # A60B-5：基线不可被换
    got = hashlib.md5(pathlib.Path(BASE_PROBE).read_bytes()).hexdigest()
    ok["A60B-5"] = {"want": BASE_PROBE_MD5, "got": got, "pass": got == BASE_PROBE_MD5}
    # A60B-6：装置六件 md5（与 probe60a 同一份清单）
    root = pathlib.Path("/home/ec2-user/sgmn-loop")
    A6 = {f: {"want": w, "got": hashlib.md5((root / f).read_bytes()).hexdigest()}
          for f, w in P60.DEVICE.items()}
    ok["A60B-6"] = {"files": A6, "pass": all(v["want"] == v["got"] for v in A6.values())}
    # A60B-7：基线可复用（种子与协议必须逐项相同）
    need = {"pk1_seed": CELLS[0]["pk1"], "pv_seed": PV_SEED}
    proto = {"qsrc": "learned", "tctrl": True, "steps": T59.STEPS, "lr": T59.LR,
             "n_rep": T59.N_REP, "n_train": len(TRAIN), "n_eval": len(EVAL),
             "train_off": [TRAIN[0], TRAIN[-1]], "eval_off": [EVAL[0], EVAL[-1]]}
    A7 = {}
    for j in (1, 2):
        for arm in ("RAND", "NONE", "SCALE2"):
            p = OUT / f"PROBE60_c{j}_{arm}.json"
            if not p.exists():
                A7[f"c{j}_{arm}"] = {"exists": False}
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            sub = {k: d["protocol"][k] == v for k, v in proto.items()}
            A7[f"c{j}_{arm}"] = {"exists": True, "pv_seed": d["pv_seed"],
                                 "pk1_seed": d["pk1_seed"],
                                 "proto_ok": all(sub.values()),
                                 "pass": d["pv_seed"] == PV_SEED
                                         and d["pk1_seed"] == CELLS[j - 1]["pk1"]
                                         and all(sub.values())}
    ok["A60B-7"] = {"rows": A7, "pass": all(v.get("pass") for v in A7.values())}
    # A60B-8：种子/段零碰撞（与 PROBE60 逐项相同，且 train∩eval=∅）
    ok["A60B-8"] = {"pk1": [c["pk1"] for c in CELLS], "pv_seed": PV_SEED,
                    "train": [TRAIN[0], TRAIN[-1]], "eval": [EVAL[0], EVAL[-1]],
                    "overlap": len(set(TRAIN) & set(EVAL)),
                    "pass": [c["pk1"] for c in CELLS] == [20263351, 20263352]
                            and PV_SEED == 20263431
                            and [TRAIN[0], TRAIN[-1], EVAL[0], EVAL[-1]] == [64000, 64095, 64100, 64499]
                            and len(set(TRAIN) & set(EVAL)) == 0}
    ok["all_pass"] = all(v["pass"] for k, v in ok.items() if isinstance(v, dict) and k != "all_pass")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PROBE60B_stageA.json").write_text(json.dumps(ok, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    for k in sorted(x for x in ok if x.startswith("A60B")):
        print(f"  {k}: {'PASS' if ok[k]['pass'] else '✗ FAIL'}", flush=True)
    print(f"★ Stage A {'全过' if ok['all_pass'] else '有失败'}（→ PROBE60B_stageA.json）", flush=True)
    assert ok["all_pass"], "Stage A 未过，禁止正式轮"


def smoke():
    """协议可执行（4 步 / 3 训 / 5 评），产物 `PROBE60B_smoke_*`（裁决器不读）。"""
    import math
    pk = F59.pk1_for(CELLS[0]["pk1"])
    for s in (0.5, 4.0):
        pv = pv_dose(s)
        phi, _ = T59.train_arm59("learned", TRAIN[:3], shared_pk=pk, shared_pv=pv, alpha=None,
                                 train_ctrl=True, steps=4, target_pv=pv)
        a = T59.eval_arm59("learned", phi, EVAL[:5], shared_pk=pk, shared_pv=pv, alpha=None,
                           target_pv=pv)
        assert math.isfinite(a["mean_L"]), f"smoke 非有限：s={s}"
        print(f"[smoke c1/s{s:g}] L={a['mean_L']:.4f} (n_eval={len(a['L_series'])})", flush=True)
        (OUT / f"PROBE60B_smoke_c1_s{s:g}.json").write_text(json.dumps(
            {"smoke": True, "s": s, "mean_L": a["mean_L"], "n_eval": len(a["L_series"]),
             "steps": 4, "n_train": 3, "probe_md5": probe_md5()}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    print("★ smoke 通过（协议可执行；不产出判据读数）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int)
    ap.add_argument("--s", type=float)
    ap.add_argument("--stageA", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.stageA:
        stageA()
    elif a.smoke:
        smoke()
    elif a.cell and a.s is not None:
        assert a.cell in (1, 2) and a.s in DOSES, "格或剂量不在冻结集内"
        run(a.cell, a.s)
    else:
        ap.print_help()
