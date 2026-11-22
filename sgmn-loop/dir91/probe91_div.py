#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
r"""DIR-91 · **开卡前诊断**（**不是判据、不产生裁决**）：分离 CEIL 臂在 $\gamma>0$ 下发散的两个候选致因。

$$rev1.3 \text{ 的 Stage A 实测：}\ \gamma=[0.0301,0.0441,0.2276]\ (\text{原 }0.05),\quad
\lVert\gamma m\rVert_{L16}=1.91\times10^{8},\quad \text{CEIL}=0.375$$

**三组设定（只变训练设定，不动装置、不动判据阈值）**

| 臂 | $\gamma$ | `READ_LR` | 检验什么 |
|---|---|---|---|
| **T**（本轮 Stage A 的实况，作基线） | 可训练 | $10^{-2}$ | 复现发散 |
| **F**（冻结 $\gamma$） | 冻结 $0.05$ | $10^{-2}$ | 发散是否来自 $\gamma$ 自由度 |
| **L**（降 LR） | 可训练 | $10^{-3}$ | 发散是否来自步长过大 |

**读数**：CEIL 按题 acc · $\gamma$ 轨迹 · $\max\lVert\gamma m\rVert$ · loss 轨迹 · `diverged` 旗标。
**本探针不下结论**，只产出致因分离的事实。
"""
import json, pathlib, random, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
import numpy as np, torch
import R2_task91 as R91

tok, model = R91.model_and_tok()
facts = R91.build_facts(tok)
chunks, owner = R91.build_chunks(tok, facts)
order2 = list(range(R91.N_FACTS))
random.Random(778).shuffle(order2)
trk2 = order2[:R91.N_TRAIN_KEYS]

def run_arm(label, freeze_gamma, lr):
    old_lr = R91.READ_LR
    R91.READ_LR = lr
    st = R91.make_stack(gamma=R91.GAMMA_RUN)
    try:
        R91.write_phase(model, st, chunks)
        if freeze_gamma:
            for m in st.mems.values():
                m.gamma.requires_grad_(False)
        losses, viol, diverged = R91.train_read_task(model, st, tok, facts, trk2, random.Random(99))
        gam_after = {li: float(m.gamma) for li, m in st.mems.items()}
        for m in st.mems.values():
            m.no_write = True
        model(input_ids=chunks[0].to(R91.DEV))
        inj = {f"L{li}": float((m.gamma * m.injection).norm()) for li, m in st.mems.items()}
        injmax = max(float((m.gamma * m.injection).abs().max()) for m in st.mems.values())
        itemsC = R91.make_eval_items(tok, facts, trk2, random.Random(556))
        scorerC = lambda p, cs: [R91.nll_of_candidate(model, st, p, c) for c in cs]  # noqa: E731
        acc, _, det = R91.score_and_accuracy(itemsC, scorerC)
        accq, nq = R91.acc_by_question(itemsC, det)
        out = {"label": label, "freeze_gamma": freeze_gamma, "read_lr": lr,
               "gamma_after": gam_after, "inj_norm": inj, "inj_max": injmax,
               "ceil_acc_by_item": float(acc), "ceil_acc_by_question": float(accq), "n_question": int(nq),
               "loss_first": float(np.mean(losses[:10])) if losses else None,
               "loss_last": float(np.mean(losses[-10:])) if losses else None,
               "loss_max": float(np.max(losses)) if losses else None,
               "n_steps_ran": len(losses), "diverged_W": bool(diverged), "write_violations": int(viol)}
        print(f"\n=== {label}：γ={'冻结 0.05' if freeze_gamma else '可训练'}，LR={lr} ===", flush=True)
        print(f"  γ 训后 = " + "/".join(f"{v:.4f}" for v in gam_after.values()), flush=True)
        print(f"  ‖γ·injection‖ = " + str({k: f"{v:.2e}" for k, v in inj.items()})
              + f"  max={injmax:.2e}", flush=True)
        print(f"  loss 首 10 步={out['loss_first']:.4f} 末 10 步={out['loss_last']:.4f} "
              f"峰值={out['loss_max']:.4f}  步数={out['n_steps_ran']}  W发散={diverged}", flush=True)
        print(f"  ★ CEIL 按题 acc={accq:.3f}（按项 {acc:.3f}，n题={nq}）", flush=True)
        return out
    finally:
        R91.READ_LR = old_lr
        st.remove()

res = {"note": "pre-card diagnostic, NOT a criterion",
       "stageA_rev13_reference": {"ceil_acc_by_question": 0.375,
                                  "gamma_after": {"L8": 0.030143, "L12": 0.044086, "L16": 0.227573},
                                  "inj_norm_L16": 1.91e8},
       "arms": []}
for label, fz, lr in (("T_baseline", False, 1e-2), ("F_freeze_gamma", True, 1e-2), ("L_lower_lr", False, 1e-3)):
    res["arms"].append(run_arm(label, fz, lr))

print("\n=== 汇总 ===", flush=True)
print(f"{'臂':<16}{'γ':<8}{'LR':<8}{'CEIL(题)':<10}{'max‖γm‖':<12}{'loss 末':<10}{'W发散':<7}", flush=True)
for a in res["arms"]:
    print(f"{a['label']:<16}{('冻结' if a['freeze_gamma'] else '训练'):<8}{a['read_lr']:<8}"
          f"{a['ceil_acc_by_question']:<10.3f}{a['inj_max']:<12.2e}{a['loss_last']:<10.4f}{str(a['diverged_W']):<7}", flush=True)
(pathlib.Path(__file__).resolve().parent / "out").mkdir(exist_ok=True)
(pathlib.Path(__file__).resolve().parent / "out" / "probe91_div.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
