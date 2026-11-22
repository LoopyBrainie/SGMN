#!/usr/bin/env python3
"""DIR-116 · 开卡**前**的机械前提核查（不是实验轮，不出判据）。

要回答的唯一问题（`P115-1` 的前提）：
  任务说「取 `lm_head` 对正确 token 的那一行当 `m`」，这与 `install_ceil` 已在用的
  `input_embeddings` 行**是不是同一张量**？
  若是（tie）⇒ `P115-1` 的修补是**空的**，且 `DEF-115-06` 的诊断（"输入嵌入 ≠ 输出方向"）
  **需更正**，真因另有其它（剩余层 / final RMSNorm / 幅度不足）。

附带（同时服务 `P115-2`）：‖h‖ 与 ‖γ·m‖ 的量级 ⇒ 为真 oracle 的 α 网格做**尺度匹配**标定。

只读：不训练、不写状态、不改装置。
"""
import json
import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _ROOT / "dir90", _ROOT / "dir104", _ROOT / "dir114", _ROOT / "dir115"):
    sys.path.insert(0, str(p))

import R1_task90 as D90          # noqa: E402
import R1_task104 as D104        # noqa: E402
import R1_task114 as P           # noqa: E402
import R1_task115 as P115        # noqa: E402

OUT = _HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
D90.OUT = OUT


@torch.no_grad()
def main():
    tok, model = D90.model_and_tok()
    cfg = model.config
    emb = model.get_input_embeddings().weight.detach()
    lmh = model.lm_head.weight.detach()
    rep = {
        "hidden_size": int(cfg.hidden_size),
        "n_layers": int(cfg.num_hidden_layers),
        "vocab": int(cfg.vocab_size),
        "mount": list(D90.MOUNT),
        "gamma_tr": float(P115.GAMMA_TR),
        "lm_head_has_bias": bool(getattr(model.lm_head, "bias", None) is not None),
    }

    # ---------------- ① tie 判定（决定性） ----------------
    rep["tie"] = {
        "config_tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
        "same_data_ptr": bool(lmh.data_ptr() == emb.data_ptr()),
        "torch_equal": bool(torch.equal(lmh, emb)),
        "shapes": [list(emb.shape), list(lmh.shape)],
    }
    tied = rep["tie"]["same_data_ptr"] or rep["tie"]["torch_equal"]

    # ---------------- ② 正确值 token 的方向/范数 ----------------
    facts = P.build_facts_pairs(tok)
    toks = []
    for f in facts:
        assert len(f["val_ids"]) == 1, f["val_ids"]
        toks.append(int(f["val_ids"][0]))
    toks_u = sorted(set(toks))
    e = emb[toks_u].float()
    w = lmh[toks_u].float()
    cos = torch.nn.functional.cosine_similarity(e, w, dim=-1)
    rep["value_tokens"] = {
        "n_distinct": len(toks_u),
        "emb_norm_mean": float(e.norm(dim=-1).mean()),
        "emb_norm_min": float(e.norm(dim=-1).min()),
        "emb_norm_max": float(e.norm(dim=-1).max()),
        "lmhead_norm_mean": float(w.norm(dim=-1).mean()),
        "lmhead_norm_min": float(w.norm(dim=-1).min()),
        "lmhead_norm_max": float(w.norm(dim=-1).max()),
        "cos_emb_lmhead_mean": float(cos.mean()),
        "cos_emb_lmhead_min": float(cos.min()),
        "cos_emb_lmhead_max": float(cos.max()),
    }

    # ---------------- ③ ‖h‖：注入点所在的残差流量级 ----------------
    # 原生前向（不建 stack，避免 SGMN 注入扰动读数）
    cap = {}

    def mk(li, tag):
        def _hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            cap[(li, tag)] = float(out.detach().float().norm(dim=-1)[0, -1])
        return _hook

    # ★ 不要为了拿层表而实例化 SGMNStack —— 它会注册 forward hook 并一直触发（调用
    #   read/write、改 prev_hs），污染 ③④ 的读数。直接走 HF 的层表。
    layers = None
    for attr in ("model", "transformer"):
        base = getattr(model, attr, None)
        if base is not None and hasattr(base, "layers"):
            layers = base.layers
            break
    assert layers is not None and len(layers) > max(D90.MOUNT), "取层表失败"
    handles = []
    for li in list(D90.MOUNT) + [max(D90.MOUNT) + 1]:
        handles.append(layers[li].register_forward_hook(mk(li, "post")))
    n_probe = 4
    for f in facts[:n_probe]:
        cap.clear()
        seq = torch.tensor([f["prompt_ids"]], dtype=torch.long, device=D90.DEV)
        lg = model(input_ids=seq).logits.float()
        for (li, tag), v in cap.items():
            rep.setdefault("h_norm_by_layer", {}).setdefault(f"L{li}", []).append(v)
        # 最终 logits 的量级（相对于 ‖h‖）
        rep.setdefault("logit_absmax", []).append(float(lg[0, -1].abs().max()))
    for h in handles:
        h.remove()
    rep["h_norm_by_layer"] = {
        k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
        for k, v in rep["h_norm_by_layer"].items()}
    rep["logit_absmax"] = {"mean": float(np.mean(rep["logit_absmax"])),
                           "max": float(np.max(rep["logit_absmax"]))}

    # ---------------- ④ 未训练 ‖m‖ 与 ‖γ·m‖/‖h‖ ----------------
    st = D90.make_stack(gamma=P115.GAMMA_TR, no_write=True, seed=P115.SEEDS[0])
    D104._PROBE.clear()
    D104.install_probe(st)
    try:
        for f in facts[:n_probe]:
            D90.nll_of_candidate(model, st, f["prompt_ids"], f["val_ids"])
            for li, d in D104._PROBE.items():
                rep.setdefault("m_norm_by_layer", {}).setdefault(f"L{li}", []).append(
                    {"m": float(d["m"].norm()), "hnorm": float(d["hnorm"]),
                     "gamma": float(d["gamma"])})
    finally:
        D104.uninstall_probe(st, "read")
    mm = {}
    for li, vs in rep["m_norm_by_layer"].items():
        mn = float(np.mean([v["m"] for v in vs]))
        hn = float(np.mean([v["hnorm"] for v in vs]))
        g = float(vs[0]["gamma"])
        mm[li] = {"m_norm_mean": mn, "h_norm_mean": hn, "gamma": g,
                  "gamma_m_over_h": g * mn / hn}
    rep["m_norm_untrained"] = mm
    del rep["m_norm_by_layer"]

    # ---------------- ⑤ 为真 oracle 定 α 网格（尺度匹配） ----------------
    # 目标：让 ‖α·w_c‖ / ‖h_out(注入层)‖ 覆盖给定比值
    RATIOS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    plan = {}
    for li in D90.MOUNT:
        hn = rep["h_norm_by_layer"][f"L{li}"]["mean"]
        wn = rep["value_tokens"]["lmhead_norm_mean"]
        plan[f"L{li}"] = {f"{r:g}": float(r * hn / wn) for r in RATIOS}
    rep["alpha_plan"] = {
        "ratios": list(RATIOS),
        "h_norm_used": {f"L{li}": rep["h_norm_by_layer"][f"L{li}"]["mean"] for li in D90.MOUNT},
        "w_norm_used": rep["value_tokens"]["lmhead_norm_mean"],
        "alpha_by_layer": plan,
    }

    # ---------------- 结论行 ----------------
    rep["VERDICT_OF_PREMISE"] = {
        "tied": tied,
        "P115_1_is_vacuous": bool(tied),
        "note": ("`lm_head.weight` 与 `input_embeddings.weight` 是**同一张量** ⇒ "
                 "「改用 lm_head 行」与 `install_ceil` 已做的**逐位相同** ⇒ `P115-1` 的修补是空的；"
                 "`DEF-115-06` 的诊断须更正为「方向对但被剩余层/RMSNorm 抹掉或幅度不足」。"
                 if tied else
                 "二者不同张量 ⇒ `P115-1` 的修补**有意义**；`DEF-115-06` 的诊断成立。"),
    }
    (OUT / "premise116.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in
                      ("hidden_size", "n_layers", "tie", "value_tokens",
                       "h_norm_by_layer", "logit_absmax", "m_norm_untrained",
                       "VERDICT_OF_PREMISE")}, ensure_ascii=False, indent=1))
    print("\n=== alpha_plan ===")
    print(json.dumps(rep["alpha_plan"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
