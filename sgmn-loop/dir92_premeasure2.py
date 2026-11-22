#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
r"""DIR-92 · **开卡前测量 ②**（**不是判据**）。修 `D-109` §4 的两个错误后，重定真因。
   已证：$V0/V3$ 的 $\lVert k\rVert$ 几乎相同（26.8 / 27.6）⇒ $\eta\lVert k\rVert^2$ **不能**解释 0.148 vs 0.566。
   本探针用**同一条代码路径 + 同一个统计量**测两个语料，并检验新假设：
     (H-v) 写规则在键近共线时只能收敛到"平均 $v$"，故 $\mathrm{rel.err}$ 由 **$v$ 的相对离散度**决定：
           $\mathrm{spread}(v)=\mathrm{median}_i\lVert v_i-\bar v\rVert/\lVert v_i\rVert$。
           若 $\mathrm{spread}(v)_{V3}\gg\mathrm{spread}(v)_{V0}$ ⇒ 直接解释 0.566 vs 0.148。
   同时报 $k$ 的相对离散度与原始 $\max\lvert\cos\rvert$，确认两者键的可分性相同。
"""
import json, pathlib, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir90")
import numpy as np, torch
import R2_task91 as R91

def corpus_V3():
    tok, model = R91.model_and_tok()
    return model, R91.build_facts(tok), None

def corpus_V0():
    import R1_task90 as R90
    tok, model = R91.model_and_tok()
    return model, R90.build_facts(tok), R90.build_chunks

out = {"note": "pre-card measurement, NOT a criterion", "corpora": {}}

for tag, build in (("V0", corpus_V0), ("V3", corpus_V3)):
    model, facts, chunkfn = build()
    tok = R91.model_and_tok()[0]
    chunks, owner = (chunkfn(tok, facts) if chunkfn else R91.build_chunks(tok, facts))
    st = R91.make_stack(gamma=0.0, no_write=False)
    try:
        R91.write_phase(model, st, chunks)                 # 全 600 块，与 D-109 同
        K, V = R91.key_matrices(model, st, facts, chunks, owner)
        rec = {}
        for li in sorted(st.mems):
            Km, Vm = K[li], V[li]
            W = R91.W_of(st.mems[li]).float()
            rel = ((Km @ W.T) - Vm).norm(dim=1) / Vm.norm(dim=1).clamp_min(1e-30)
            # v / k 的相对离散度（相对各自的均值，用中位数抗离群）
            vb = Vm.mean(dim=0, keepdim=True)
            kb = Km.mean(dim=0, keepdim=True)
            vs = (Vm - vb).norm(dim=1) / Vm.norm(dim=1).clamp_min(1e-30)
            ks = (Km - kb).norm(dim=1) / Km.norm(dim=1).clamp_min(1e-30)
            Kn = torch.nn.functional.normalize(Km, dim=1)
            C = (Kn @ Kn.T).abs()
            off = C[~torch.eye(len(C), dtype=torch.bool, device=C.device)]
            rec[f"L{li}"] = {
                "rel_med": float(rel.median()),                    # 统一统计量：fact 中位数
                "rel_mean": float(rel.mean()),
                "v_spread_med": float(vs.median()),                # ★ H-v 的自变量
                "k_spread_med": float(ks.median()),
                "k_norm_mean": float(Km.norm(dim=1).mean()),
                "v_norm_mean": float(Vm.norm(dim=1).mean()),
                "raw_max_cos": float(off.max()),
                "eta_k2": float(R91.ETA_WRITE * (Km.norm(dim=1).mean() ** 2)),
                "converged_pred": float(1 - (1 - R91.ETA_WRITE * float(Km.norm(dim=1).mean() ** 2)) ** 19),
            }
        out["corpora"][tag] = rec
        print(f"\n=== {tag} ===", flush=True)
        print(f"{'层':>3} {'rel.err中位':>11} {'v离散度':>9} {'k离散度':>9} "
              f"{'raw max cos':>12} {'|k|':>7} {'|v|':>7} {'η|k|²':>7} {'收敛预测':>9}", flush=True)
        for li in sorted(st.mems):
            q = rec[f"L{li}"]
            print(f"{li:>3} {q['rel_med']:>11.4f} {q['v_spread_med']:>9.4f} {q['k_spread_med']:>9.4f} "
                  f"{q['raw_max_cos']:>12.5f} {q['k_norm_mean']:>7.2f} {q['v_norm_mean']:>7.2f} "
                  f"{q['eta_k2']:>7.4f} {q['converged_pred']:>9.3f}", flush=True)
    finally:
        st.remove()

def agg(tag, field):
    return float(np.mean([v[field] for v in out["corpora"][tag].values()]))

out["summary"] = {f"{tag}_{f}": agg(tag, f) for tag in ("V0", "V3")
                  for f in ("rel_med", "v_spread_med", "k_spread_med", "k_norm_mean", "eta_k2")}
out["summary"]["rel_ratio_V3_over_V0"] = (agg("V3", "rel_med") / agg("V0", "rel_med"))
out["summary"]["vspread_ratio_V3_over_V0"] = (agg("V3", "v_spread_med") / agg("V0", "v_spread_med"))
print("\n=== 汇总 ===", flush=True)
print(json.dumps(out["summary"], ensure_ascii=False, indent=1), flush=True)
(pathlib.Path(__file__).resolve().parent / "dir92" / "out" / "premeasure2.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
