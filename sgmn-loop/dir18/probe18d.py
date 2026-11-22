#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-18 · probe18d —— 判定「训练带来的 +3 nats 到底是**内容**还是**先验**」。

【动机】probe18c 的 Q4 已经把 DIR-18 的收益归因打到写侧：
    主臂(hebb η=1e-7) 11.0017 | nowrite(地址照写、W 从不更新) 11.0239
    | shuffle(键值错配) 11.0951 | misaddr 12.3837 | verbatim 10.4638 | base 14.1687
  ⇒ Hebbian 写步的**贡献 ≈ 0.02 nats**（nowrite 与主臂逐位同分）。
  但主臂相对 base 仍有 **+3.17 nats** ⇒ 这 +3.17 来自别处。

【候选解释】DIR-13 已实测：写入键几乎共线（cos 0.987–0.995）⇒ Hebbian 臂里
    y = W q ≈ 常向量 ⇒ 注入信号 m = W_o y ≈ 常数偏置 ⇒ 模型只学到
    「看到 The vault code is ⇒ 输出一个**稀有词**」的**边际先验**，
    而不是「读出**这个**稀有词」。⇒ **稀有词先验**假说。

【判别设计】同一批评测文档、同一个模型，测四个目标的 NLL（NLL 口径，越小越好）：
    (a) `code`    —— 本文档真码（**已写入**记忆，且地址匹配指向它）
    (b) `dcode`   —— 干扰文档的码（**已写入**记忆，但地址不指向它）
    (c) `rare0/1` —— **从未写入任何记忆**的稀有词（与 (a) 同分布、同稀有度）
    (d) `mid`     —— 语料中频词（稀有词的阴性对照）
  判读表：
    · NLL(a) ≈ NLL(c) ≪ NLL(d) ⇒ **先验**（内容未被读出）⇒ DIR-18 主假说**不成立**
    · NLL(a) ≪ NLL(c)          ⇒ **内容**确实被读出 ⇒ 结合 nowrite 对照可知
                                  「内容来自地址表+读接口，**不是** Hebbian 写步」
    · NLL(a) ≪ NLL(b)          ⇒ 寻址**区分**了真码与同批干扰码（内容寻址成立）
  对照跑 `verbatim` 臂 ⇒ 与 DIR-17 的 D-23（内容余量 +1.85 nats）互相印证。

【顺带】统计模型在评测位置上的概率质量落在哪儿（rare / mid / 其他），
  以便量化「稀有词先验」本身能解释多少 nats。

用法：/opt/pytorch/bin/python -u probe18d.py
"""
import importlib.util
import json
import math
import pathlib
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

D = "/home/ec2-user/sgmn-loop/dir18"
sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location("r1w", D + "/R1_write.py")
r1w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1w)

OUT = pathlib.Path(D) / "probe18d_out.json"
rep = {"arms": {}, "notes": []}


def log(*a):
    print(*a, flush=True)


def make_nowrite_variant(st):
    """地址表照写（keys/sigs），但 **W 从不更新** ⇒ 地址正确、内容无关联。"""
    for m in st.mems.values():
        core = m.core

        def patched(k, v, sig=None, core=core):
            with torch.no_grad():
                i = core.count % core.N
                core.count += 1
                core.keys[i].copy_(k)
                core.vals[i].copy_(v)
                if sig is not None:
                    core.sigs[i].copy_(sig)
            return None
        core.write = patched


def pick_controls(tok, pool, used):
    """挑两个**从未被写入**的稀有词 + 一个中频词。

    稀有词：沿用装置自己的 `rare_word_tokens` 池（与真码**同分布**），
            排除所有文档用过的码。
    中频词：从语料 pool 里取出现次数中位数的**单 token 纯字母**词。
    """
    rare_pool = r1w.rare_word_tokens(tok, pool)
    ctrl_rare = [int(t) for t in rare_pool if int(t) not in used][:2]
    assert len(ctrl_rare) == 2, f"对照稀有词不足（池 {len(rare_pool)}）"
    cnt = Counter(pool)
    # 语料里出现过、且是单 token 纯字母词的，按频次排序取中段
    cands = []
    for t, c in cnt.items():
        if c < 20:
            continue
        try:
            s = tok.decode([t])
        except Exception:
            continue
        w = s.strip()
        if len(w) < 5 or not w.isalpha() or not s or not s[0].isspace():
            continue
        if tok(s).input_ids != [t]:
            continue
        cands.append((c, int(t)))
    cands.sort(reverse=True)
    mid = None
    for c, t in cands:
        if t not in used and t not in ctrl_rare:
            mid = t
            break
    if mid is None:
        mid = int(tok(" the").input_ids[0])
    return ctrl_rare, int(mid)


def eval_targets(st, ev, ctrl_rare, mid_id, rare_set):
    """在同一次读前向上测四个目标的 NLL + 概率质量分布。"""
    rows = []
    snap0 = st.snapshot_all()
    for d in ev:
        st.unfreeze_write()
        st.restore_all(snap0)
        r1w.write_both(st, d)
        for m in st.mems.values():
            m.vals_written.clear()
            m.keys_written.clear()
        st.freeze_write()
        pre = torch.tensor(d["prefix"], dtype=torch.long, device=r1w.DEV).unsqueeze(0)
        with torch.no_grad():
            lg = st.model(input_ids=pre).logits.float()
        lp_all = F.log_softmax(lg[0, -1], dim=-1)
        p = lp_all.exp()
        code = int(d["code_ids"][0])
        dcode = int(d["dist_code"])
        row = {"nll_code": -float(lp_all[code]),
               "nll_dcode": -float(lp_all[dcode]),
               "nll_rare0": -float(lp_all[ctrl_rare[0]]),
               "nll_rare1": -float(lp_all[ctrl_rare[1]]),
               "nll_mid": -float(lp_all[mid_id]),
               "p_code": float(p[code]),
               "p_rare_mean": float(0.5 * (p[ctrl_rare[0]] + p[ctrl_rare[1]])),
               "p_mid": float(p[mid_id])}
        row["nll_rare_mean"] = 0.5 * (row["nll_rare0"] + row["nll_rare1"])
        row["content_margin"] = row["nll_rare_mean"] - row["nll_code"]
        row["distractor_margin"] = row["nll_rare_mean"] - row["nll_dcode"]
        # 概率质量：评测位置上的总概率里，落在「装置稀有词池」上的份额
        row["p_mass_on_rare_pool"] = float(p[list(rare_set)].sum())
        rows.append(row)
    st.unfreeze_write()
    return rows


def summarize(name, rows):
    s = {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()}
    rep["arms"][name] = s
    log(f"[{name}] NLL: code={s['nll_code']:.4f} | dcode={s['nll_dcode']:.4f} | "
        f"rare(未写入)={s['nll_rare_mean']:.4f} | mid={s['nll_mid']:.4f}")
    log(f"          ⇒ 内容余量(rare−code)={s['content_margin']:+.4f} | "
        f"干扰余量(rare−dcode)={s['distractor_margin']:+.4f} | "
        f"P(code)={s['p_code']:.3e} | 稀有池总质量={s['p_mass_on_rare_pool']:.3e}")
    return s


def main():
    tok = AutoTokenizer.from_pretrained(r1w.MODEL)
    model = AutoModelForCausalLM.from_pretrained(r1w.MODEL, dtype=torch.float32).to(r1w.DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size
    tr, ev = r1w.build_docs(tok, 20261028, 40, 12)

    # 语料 pool（与 build_docs 同源）—— 用于挑中频词
    import re as _re
    text = r1w.load_filler()
    paras = [q.strip() for q in _re.split(r"\n\s*\n", text) if len(q.strip()) > 120]
    pool = []
    for q in paras:
        pool.extend(tok(q).input_ids)
        if len(pool) > 160000:
            break

    used = set()
    for x in list(tr) + list(ev):
        used.add(int(x["code_ids"][0]))
        used.add(int(x["dist_code"]))
    ctrl_rare, mid_id = pick_controls(tok, pool, used)
    rare_pool = set(int(t) for t in r1w.rare_word_tokens(tok, pool))
    log(f"对照稀有词 = {ctrl_rare}（未写入任何记忆） | 中频词对照 = {mid_id} "
        f"({tok.decode([mid_id])!r}) | 稀有池大小 {len(rare_pool)}")

    def run(name, qm, mkind, eta, train, variant=None):
        st = r1w.make_stack(model, d, qm, seed=20261028, mem_kind=mkind, eta=eta)
        if variant == "nowrite":
            make_nowrite_variant(st)
        if train:
            r1w.train_arm(st, tr, qm)
        rows = eval_targets(st, ev, ctrl_rare, mid_id, rare_pool)
        st.remove()
        torch.cuda.empty_cache()
        return summarize(name, rows)

    log("=== Q5：训练收益是「内容」还是「稀有词先验」 ===")
    run("hebb_eta7_cue_train", "cue", "hebbian_full", 1e-7, True)
    run("hebb_eta7_nowrite_train", "cue", "hebbian_full", 1e-7, True, variant="nowrite")
    run("verbatim_cue_train", "cue", "explicit_kv", None, True)
    run("verbatim_cue_notrain", "cue", "explicit_kv", None, False)

    # 基座（不挂装置）
    rows_b = []
    for x in ev:
        pre = torch.tensor(x["prefix"], dtype=torch.long, device=r1w.DEV).unsqueeze(0)
        with torch.no_grad():
            lg = model(input_ids=pre).logits.float()
        lp_all = F.log_softmax(lg[0, -1], dim=-1)
        code = int(x["code_ids"][0])
        dcode = int(x["dist_code"])
        rows_b.append({"nll_code": -float(lp_all[code]), "nll_dcode": -float(lp_all[dcode]),
                       "nll_rare0": -float(lp_all[ctrl_rare[0]]),
                       "nll_rare1": -float(lp_all[ctrl_rare[1]]),
                       "nll_mid": -float(lp_all[mid_id]),
                       "p_code": float(lp_all[code].exp()),
                       "p_rare_mean": float(0.5 * (lp_all[ctrl_rare[0]].exp() + lp_all[ctrl_rare[1]].exp())),
                       "p_mid": float(lp_all[mid_id].exp())})
    s = {k: float(np.mean([r[k] for r in rows_b])) for k in rows_b[0].keys()}
    s["nll_rare_mean"] = 0.5 * (s["nll_rare0"] + s["nll_rare1"])
    s["content_margin"] = s["nll_rare_mean"] - s["nll_code"]
    s["distractor_margin"] = s["nll_rare_mean"] - s["nll_dcode"]
    s["p_mass_on_rare_pool"] = float("nan")
    rep["arms"]["base"] = s
    log(f"[base] NLL: code={s['nll_code']:.4f} | dcode={s['nll_dcode']:.4f} | "
        f"rare(未写入)={s['nll_rare_mean']:.4f} | mid={s['nll_mid']:.4f} | "
        f"内容余量={s['content_margin']:+.4f}")

    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n写出 {OUT}")


if __name__ == "__main__":
    main()
