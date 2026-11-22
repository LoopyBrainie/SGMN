#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
r"""DIR-92 · **开卡前测量**（`task.md` DIR-92 明文要求）。**不是判据、不产出裁决**。
   目的：把 `D-109` §4 的根因 $\eta\lVert k\rVert^2$ 从"两个点的观测"变成**可设计的量**。
   测三件事：
     (1) $\lVert k\rVert$ 在 **V0 / V3 两种语料 × MOUNT 三层**上的**实际分布**（32 facts 每个一条）；
     (2) 真实隐状态每维 rms 与 `pk_scale` 标定假设（rms=1）的**比值** —— 即失配倍数；
     (3) 顺带 $\lVert v\rVert$（注意：$v$ **没有**任何尺度校准，$k$ 有）。
   并给出 $\eta\lVert k\rVert^2$ 与 $1-(1-c)^{N_i}$ 的**预测值**，供开卡时冻结判据用。
"""
import json, pathlib, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir90")
import numpy as np, torch
import R2_task91 as R91

ETA = R91.ETA_WRITE
D_K = 896
N_I_EXPECT = 19          # 600 块 / 32 事实

def corpus_V3():
    tok, model = R91.model_and_tok()
    facts = R91.build_facts(tok)
    chunks, owner = R91.build_chunks(tok, facts)
    return model, facts, chunks, owner

def corpus_V0():
    # 只用 dir90 的**语料构造**；装置一律用 R91 的（装置固定、只换语料 = D-109 的设计）
    import R1_task90 as R90
    tok, model = R91.model_and_tok()
    facts = R90.build_facts(tok)
    chunks, owner = R90.build_chunks(tok, facts)
    return model, facts, chunks, owner

out = {"eta": ETA, "d_k": D_K, "n_i_expected": N_I_EXPECT, "corpora": {}}

for tag, build in (("V0", corpus_V0), ("V3", corpus_V3)):
    model, facts, chunks, owner = build()
    st = R91.make_stack(gamma=0.0, no_write=True)
    try:
        # 每事实一条 k（与 key_matrices 同构：该事实所在块的前视块均值）
        K, V = R91.key_matrices(model, st, facts, chunks, owner)
        rec = {"n_facts": len(facts)}
        # 真实隐状态每维 rms（取该层全部 token）
        for li in sorted(st.mems):
            m = st.mems[li]
            kn = K[li].norm(dim=1); vn = V[li].norm(dim=1)
            # 隐状态 rms：再前向一遍，读 _hs_in
            model(input_ids=chunks[0].to(R91.DEV))
            hs = st._hs_in[li].reshape(-1, st._hs_in[li].shape[-1])
            rms = float((hs ** 2).mean().sqrt())
            expected_kn = float(np.sqrt(5.0 * D_K))     # pk_scale 的标定假设（rms=1）
            rec[f"L{li}"] = {
                "k_norm_mean": float(kn.mean()), "k_norm_std": float(kn.std()),
                "k_norm_min": float(kn.min()), "k_norm_max": float(kn.max()),
                "v_norm_mean": float(vn.mean()), "v_norm_std": float(vn.std()),
                "pk_scale": float(m.pk_scale),
                "hs_rms_per_dim": rms,
                "hs_rms_ratio_to_calibration": rms,
                "k_norm_expected_if_rms1": expected_kn,
                "k_norm_ratio_to_expected": float(kn.mean()) / expected_kn,
                "eta_k2": float(ETA * (kn.mean() ** 2)),
                "converged_pred": float(1 - (1 - ETA * (kn.mean() ** 2)) ** N_I_EXPECT),
            }
        out["corpora"][tag] = rec
        print(f"\n=== {tag}  ({len(facts)} facts) ===", flush=True)
        print(f"{'层':>3} {'|k|均值':>9} {'|k|std':>8} {'|k|min':>8} {'|k|max':>8} "
              f"{'|v|均值':>9} {'pk_scale':>9} {'hs_rms':>7} {'|k|/期望':>9} "
              f"{'η|k|²':>8} {'收敛预测':>9}", flush=True)
        for li in sorted(st.mems):
            q = rec[f"L{li}"]
            print(f"{li:>3} {q['k_norm_mean']:>9.2f} {q['k_norm_std']:>8.3f} "
                  f"{q['k_norm_min']:>8.2f} {q['k_norm_max']:>8.2f} "
                  f"{q['v_norm_mean']:>9.2f} {q['pk_scale']:>9.4f} {q['hs_rms_per_dim']:>7.4f} "
                  f"{q['k_norm_ratio_to_expected']:>9.4f} {q['eta_k2']:>8.4f} "
                  f"{q['converged_pred']:>9.3f}", flush=True)
    finally:
        st.remove()

# 跨语料/跨层的关键比值
q0 = [v for k, v in out["corpora"]["V0"].items() if k.startswith("L")]
q3 = [v for k, v in out["corpora"]["V3"].items() if k.startswith("L")]
out["summary"] = {
    "V0_kn_mean": float(np.mean([q["k_norm_mean"] for q in q0])),
    "V3_kn_mean": float(np.mean([q["k_norm_mean"] for q in q3])),
    "kn_sq_ratio_V0_over_V3": float(np.mean([q["k_norm_mean"] for q in q0])
                                    ** 2 / np.mean([q["k_norm_mean"] for q in q3]) ** 2),
    "eta_k2_V0": float(np.mean([q["eta_k2"] for q in q0])),
    "eta_k2_V3": float(np.mean([q["eta_k2"] for q in q3])),
    "converged_V0": float(np.mean([q["converged_pred"] for q in q0])),
    "converged_V3": float(np.mean([q["converged_pred"] for q in q3])),
    "hs_rms_V0": float(np.mean([q["hs_rms_per_dim"] for q in q0])),
    "hs_rms_V3": float(np.mean([q["hs_rms_per_dim"] for q in q3])),
}
out["summary"]["eta_k2_ratio_V0_over_V3"] = (
    out["summary"]["eta_k2_V0"] / out["summary"]["eta_k2_V3"])

print("\n=== 汇总 ===", flush=True)
print(json.dumps(out["summary"], ensure_ascii=False, indent=1), flush=True)
pathlib.Path(pathlib.Path(__file__).resolve().parent / "dir92" / "out").mkdir(
    parents=True, exist_ok=True)
(pathlib.Path(__file__).resolve().parent / "dir92" / "out" / "premeasure.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
