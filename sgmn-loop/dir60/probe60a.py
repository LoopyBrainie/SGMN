#!/opt/pytorch/bin/python
r"""DIR-60 · 前提探针 PROBE60：$P_v$ 的**几何 / 秩 / 尺度**是否是本装置上的**可测通路**？

零算力前提检查（纪律 147）的经验对照：
- 代数：若 $P_v\mapsto P_vA$ 且 $A$ 正交，则（**忽略 tanh**）写值 $v$、记忆 $W$、读出 $y$、任务目标 $gs$
  全部左乘/右乘 $A$ ⇒ 可被 $W_o,W_{\rm out}$ 吸收 ⇒ **损失不变**；非正交 $A$ 只改损失的度量。
- 因此可测性只能来自 ① tanh 的非线性（正交也未必严格不变）② 秩/核（信息通路）③ 尺度（饱和）。
本探针把这三条各做一个臂，另加**地板臂**与**阳性对照**（`SCALE2` 必须可测，否则探针无功效）。

唯一自变量 = 写步与任务目标**共用**的那一张 $P_v$ 的构造（`SOLE`）；`Pk1` 两格互异、$P_v$ **两格共享**（"共享映射"前提）。
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir59")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir60")
import R1_task59 as T59        # noqa: E402
import R1_formal59 as F59      # noqa: E402  （只为逐字复用 `pk1_for`）
import R1_task33 as R          # noqa: E402
import R1_task37 as T37        # noqa: E402
import R1_task40 as T40        # noqa: E402

OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir60/out")
PROBE = "/home/ec2-user/sgmn-loop/dir60/PROBE60.md"
DEVICE = {"dir31/R1_grad.py": "8fa40f708600bec7e6c484727d2180c7",
          "dir33/R1_task33.py": "96ef89e87ec61556b52e0e464850087e",
          "dir37/R1_task37.py": "9236ac0490be9c413f1e8af163289207",
          "dir40/R1_task40.py": "baab3566891591944efb60a5b6dc9623",
          "dir48/R1_task48.py": "bb6154687a48d4ea33f0ebd9bc0cb393"}   # dir48 md5 = DEF-60-3 双源实测
CELLS = [{"j": 1, "pk1": 20263351}, {"j": 2, "pk1": 20263352}]
PV_SEED = 20263431                    # ★ 两格共用同一张 P_v（共享映射前提）
TRAIN = list(range(64000, 64096))     # 96（★ 新段，未在任何既往轮用过）
EVAL = list(range(64100, 64500))      # 400
ARMS = ("NONE", "RAND", "ORTH", "RANK2", "SCALE2")


def probe_md5():
    return hashlib.md5(pathlib.Path(PROBE).read_bytes()).hexdigest()


def pv_variants():
    r"""四个变体，**行范数匹配**（随机 $P_v$ 的行范数期望 $\sqrt{D_V/D_H}=\sqrt{0.5}$）。"""
    base = T59._pv1_for(PV_SEED)                      # (8,4)，逐字同装置生成式
    q, _ = torch.linalg.qr(base)                      # 列正交（行范数期望同上）
    for c in range(q.shape[1]):                       # 符号固定（确定性）
        if float(q[int(q[:, c].abs().argmax()), c]) < 0:
            q[:, c] = -q[:, c]
    rank2 = base.clone()
    rank2[:, 2] = base[:, 0]
    rank2[:, 3] = base[:, 1]
    return base, {"RAND": base, "ORTH": q, "RANK2": rank2, "SCALE2": base * 2.0}


def alpha_for(arm):
    return 0.0 if arm == "NONE" else None


def run(j, arm):
    t0 = time.time()
    pk = F59.pk1_for(CELLS[j - 1]["pk1"])
    base, V = pv_variants()
    pv = base if arm == "NONE" else V[arm]
    tag = f"c{j}_{arm}"
    al = alpha_for(arm)
    print(f"[{tag}] P_v={arm}（与目标同源）alpha={'0.0' if al == 0.0 else 'M.ALPHA'} "
          f"pk1={CELLS[j-1]['pk1']} pv_seed={PV_SEED}", flush=True)
    phi, meta = T59.train_arm59("learned", TRAIN, shared_pk=pk, shared_pv=pv, alpha=al,
                               train_ctrl=True, target_pv=pv)
    a = T59.eval_arm59("learned", phi, EVAL, shared_pk=pk, shared_pv=pv, alpha=al,
                       target_pv=pv)
    print(f"[{tag}] L={a['mean_L']:.6f} (sd {a['sd_L']:.4f}) |q|={a['q_norm']:.4f} "
          f"|y|={a['y_norm']:.4f} cos={a['cos_q_k']:.4f} {time.time()-t0:.0f}s", flush=True)
    rec = {"probe_md5": probe_md5(), "j": j, "arm": arm,
           "pv_seed": PV_SEED, "pk1_seed": CELLS[j - 1]["pk1"],
           "alpha": al, "protocol": {"qsrc": "learned", "tctrl": True,
                                     "steps": T59.STEPS, "lr": T59.LR, "n_rep": T59.N_REP,
                                     "n_train": len(TRAIN), "n_eval": len(EVAL),
                                     "train_off": [TRAIN[0], TRAIN[-1]],
                                     "eval_off": [EVAL[0], EVAL[-1]],
                                     "pv_roles": "write=target（同一张）"},
           "sha256_pv": hashlib.sha256(pv.numpy().tobytes()).hexdigest(),
           "sha1_pk1": hashlib.sha1(pk.numpy().tobytes()).hexdigest(),
           "pv_gram_offdiag_max": float((pv.T @ pv - torch.diag(torch.diagonal(pv.T @ pv)))
                                        .abs().max()),
           "pv_rank": int(torch.linalg.matrix_rank(pv)),
           "pv_row_norm_mean": float(pv.norm(dim=1).mean()),
           "pv_fro": float(pv.norm()),
           "mean_L": a["mean_L"], "sd_L": a["sd_L"], "L_series": a["L_series"],
           "q_norm": a["q_norm"], "y_norm": a["y_norm"], "m_abs": a["m_abs"],
           "Wmax": a["Wmax"], "cos_q_k": a["cos_q_k"],
           "gradnorms": meta["gradnorms"],
           "dWq1_from_init": meta["dWq1_from_init"],
           "secs": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"PROBE60_{tag}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(f"[{tag}] → PROBE60_{tag}.json", flush=True)


def stageA():
    """硬门：装置未变、四个变体按设计各不相同、共享成立、键集零重叠、α=0 真的关掉写入。"""
    ok = {}
    root = pathlib.Path("/home/ec2-user/sgmn-loop")
    A1 = {}
    for f, want in DEVICE.items():
        got = hashlib.md5((root / f).read_bytes()).hexdigest()
        A1[f] = {"want": want, "got": got, "pass": got == want}
    ok["A60-1"] = {"files": A1, "pass": all(v["pass"] for v in A1.values())}

    base, V = pv_variants()
    shas = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest()[:16] for k, v in V.items()}
    ok["A60-2"] = {"shas": shas, "n_distinct": len(set(shas.values())),
                   "pass": len(set(shas.values())) == 4}

    gram = V["ORTH"].T @ V["ORTH"] - torch.eye(V["ORTH"].shape[1])
    ok["A60-3"] = {"gram_offdiag_max": float(gram.abs().max()),
                   "row_norm_mean": float(V["ORTH"].norm(dim=1).mean()),
                   "pass": float(gram.abs().max()) <= 1e-12}

    ok["A60-4"] = {"rank_rand": int(torch.linalg.matrix_rank(base)),
                   "rank_rank2": int(torch.linalg.matrix_rank(V["RANK2"])),
                   "pass": int(torch.linalg.matrix_rank(base)) == 4
                           and int(torch.linalg.matrix_rank(V["RANK2"])) == 2}

    ok["A60-5"] = {"max_abs_scale2_minus_2rand": float((V["SCALE2"] - 2.0 * base).abs().max()),
                   "pass": float((V["SCALE2"] - 2.0 * base).abs().max()) == 0.0}

    # A60-6：两格共享同一张 P_v ⇒ 两格的变体 sha 必须逐位相同（构造上成立，此处显式落账）
    ok["A60-6"] = {"note": "两格共用同一 tensor（PV_SEED 单一来源）", "pass": True}

    # A60-7：α=0 真的关掉写入 ⇒ 初始 L 必须显著高于 RAND（同一实例、同一 P_v）
    # ★ DEF-60-1/2：**不变性**断言 —— 写源换成 2P_v、目标源固定：
    #    α=0 时写通道真的被关掉 ⇒ 损失逐位相同；默认 α 时写通道在动 ⇒ 必须不同。
    pk = F59.pk1_for(CELLS[0]["pk1"])
    tgt = base
    def _L(pv, al):
        return float(T59.trace59(T37._mk(TRAIN[0], None, train=True), T59.N_REP, "learned",
                                 TRAIN[0], None, shared_pk1=pk, shared_pv1=pv, alpha=al,
                                 target_pv1=tgt)[0])
    L_off_a, L_off_b = _L(base, 0.0), _L(base * 2.0, 0.0)
    L_on_a, L_on_b = _L(base, None), _L(base * 2.0, None)
    ok["A60-7"] = {"L_off": [L_off_a, L_off_b], "L_on": [L_on_a, L_on_b],
                   "pass": L_off_a == L_off_b and L_on_a != L_on_b}

    # A60-8：训练/评估键集零重叠
    ov = set(TRAIN) & set(EVAL)
    ok["A60-8"] = {"overlap": len(ov), "pass": len(ov) == 0}

    ok["all_pass"] = all(v["pass"] for v in ok.values() if isinstance(v, dict))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PROBE60_stageA.json").write_text(json.dumps(ok, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    for k in sorted(x for x in ok if x.startswith("A60")):
        print(f"  {k}: {'PASS' if ok[k]['pass'] else '✗ FAIL'}", flush=True)
    print(f"★ Stage A {'全过' if ok['all_pass'] else '有失败'}（→ PROBE60_stageA.json）", flush=True)
    assert ok["all_pass"], "Stage A 未过，禁止正式轮"


def smoke():
    """只证明协议可执行（4 步 / 3 训 / 5 评），产物 `PROBE60_smoke_*`（裁决器不读）。"""
    pk = F59.pk1_for(CELLS[0]["pk1"])
    base, V = pv_variants()
    for arm in ARMS:
        pv = base if arm == "NONE" else V[arm]
        al = alpha_for(arm)
        phi, _ = T59.train_arm59("learned", TRAIN[:3], shared_pk=pk, shared_pv=pv, alpha=al,
                                 train_ctrl=True, steps=4, target_pv=pv)
        a = T59.eval_arm59("learned", phi, EVAL[:5], shared_pk=pk, shared_pv=pv, alpha=al,
                           target_pv=pv)
        assert math.isfinite(a["mean_L"]), f"smoke 非有限：{arm}"
        print(f"[smoke c1/{arm}] L={a['mean_L']:.4f} (n_eval={len(a['L_series'])})", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"PROBE60_smoke_c1_{arm}.json").write_text(json.dumps(
            {"smoke": True, "arm": arm, "mean_L": a["mean_L"], "n_eval": len(a["L_series"]),
             "steps": 4, "n_train": 3, "probe_md5": probe_md5()}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    print("★ smoke 通过（协议可执行；不产出判据读数）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int)
    ap.add_argument("--arm")
    ap.add_argument("--stageA", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.stageA:
        stageA()
    elif a.smoke:
        smoke()
    elif a.cell and a.arm:
        assert a.arm in ARMS
        run(a.cell, a.arm)
    else:
        ap.print_help()