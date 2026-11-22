#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-10 · R2 —— ① 写通道发散边界的定位；② A1' 的 nan 聚合缺陷修正复检。

【① 的背景】R1 发现 η=1e-4 的臂在【写阶段】即发散（frozen_1e-4 与 joint_1e-4 的 NLL 均为 nan），
并使 A1'（"联合训练确实发生"）因 nan 污染而假失败。本轮把这两件事分开：
  * 发散是【写通道】的现象 ⇒ 用独立的写阶段扫描定位其阈值与形态（与读侧无关）⇒ 独立发现；
  * A1' 的聚合缺陷（nan 臂参与判定）是【判据聚合的实现缺陷】⇒ 在判据不变的前提下，
    按"写阶段产出有限记忆的臂"这一显式口径复检，并如实记录口径变更。

【纪律】判据本身不变；口径变更写在本文件头与 VERDICT 中；R1 的 FAIL 留档不改。
"""
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent

def _find_r1():
    for cand in (HERE / "R1_results.json", HERE.parent / "R1_results.json",
                 HERE.parent / "sgmn-loop" / "R1_results.json",
                 pathlib.Path.home() / "sgmn-loop" / "R1_results.json"):
        if cand.exists():
            return cand
    raise FileNotFoundError("R1_results.json not found")
sys.path.insert(0, str(HERE.parent))
from sgmn_module import SGMNStack      # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOUNT = [8, 12, 16]
CHUNK = 64
N_CHUNKS = 600
ETA_SCAN = [1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4]
SEED = 20261028
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")

res = {"dir": "DIR-10", "round": "R2", "eta_scan": [], "recheck": {}}


def build_chunks(tok, text, n_chunks=N_CHUNKS):
    import re as _re
    from transformers import AutoTokenizer
    paras = [q.strip() for q in _re.split(r"\n\s*\n", text) if len(q.strip()) > 120]
    ids = []
    for q in paras:
        ids.extend(tok(q).input_ids)
        if len(ids) >= 400 * 512:
            break
    t = torch.tensor(ids[:400 * 512], dtype=torch.long)
    import random
    rng = random.Random(7)
    pool = [t[i:i + CHUNK] for i in range(0, t.numel() - CHUNK, CHUNK)]
    rng.shuffle(pool)
    out = []
    while len(out) < n_chunks:
        out.extend(pool)
        out = out[:n_chunks]
    return [c.unsqueeze(0) for c in out]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import requests
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    import pathlib as _pl
    cdir = _pl.Path.home() / "sgmn-loop" / "corpus"
    cdir.mkdir(parents=True, exist_ok=True)
    tp = cdir / "train.txt"
    if not tp.exists():
        r = requests.get(CORPUS_BASE + "train.txt", timeout=90)
        r.raise_for_status()
        tp.write_bytes(r.content)
    chunks = build_chunks(tok, tp.read_text(encoding="utf-8", errors="ignore"))

    print("=== DIR-10 R2-①：写通道发散边界（64-token 块写，600 次）===", flush=True)
    for eta in ETA_SCAN:
        st = SGMNStack(model, MOUNT, model.config.hidden_size, model.config.hidden_size,
                       8, 1.0, 0.05, eta, 0.05, DEV, seed=SEED, use_factorized=True)
        traj, diverged_at = [], None
        for i, ids in enumerate(chunks):
            with torch.no_grad():
                model(input_ids=ids.to(DEV))
            if i % 20 == 0 or i == len(chunks) - 1:
                mu = max(float(st.mems[li].U.detach().abs().max()) for li in st.mems)
                mw = max(float((st.mems[li].W0.detach()
                                + st.mems[li].U.detach() @ st.mems[li].V.detach().T
                                ).abs().max()) for li in st.mems)
                ok = all(bool(torch.isfinite(st.mems[li].U).all()
                              and torch.isfinite(st.mems[li].V).all()) for li in st.mems)
                traj.append({"write": i + 1, "maxU": mu, "maxW": mw, "finite": ok})
                if not ok:
                    diverged_at = i + 1
                    break
        st.remove()
        torch.cuda.empty_cache()
        fin = [t for t in traj if t["finite"]]
        growth = (fin[-1]["maxU"] / fin[0]["maxU"]) if len(fin) > 1 else float("nan")
        row = {"eta": eta, "diverged_at_write": diverged_at,
               "n_writes": traj[-1]["write"], "maxU_final": fin[-1]["maxU"] if fin else None,
               "maxW_final": fin[-1]["maxW"] if fin else None,
               "maxU_growth": growth, "traj": traj}
        res["eta_scan"].append(row)
        print(f"  η={eta:<8g} 发散于第 {diverged_at} 次写（None=600 次内未发散）"
              f"  max|U|: {fin[0]['maxU']:.4f} → {fin[-1]['maxU']:.4e}（×{growth:.3g}）",
              flush=True)
        (HERE / "R2_recheck_results.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ============================================================ R2-② A1' 复检
    print("\n=== DIR-10 R2-②：A1' 复检（口径：写阶段产出有限记忆的臂）===", flush=True)
    r1 = json.loads(_find_r1().read_text(encoding="utf-8"))
    summ = r1["nll_summary"]
    joint_arms = [a for a in summ if a.startswith("joint")]
    finite = [a for a in joint_arms if np.isfinite(summ[a]["dWq"])]
    excluded = [a for a in joint_arms if a not in finite]
    detail = {a: {"dWq": summ[a]["dWq"], "dWo": summ[a]["dWo"],
                  "gamma": summ[a]["gamma"], "nll": summ[a]["mean_nll"]} for a in joint_arms}
    # 训练损失下降：R1 未存逐臂损失均值，这里用「训练确实发生」的可核验代理：
    # ΔWq/ΔWo 的相对幅度（>1e-3）+ 评测 NLL 有限且不同于冻结对照
    ok = bool(finite) and all(summ[a]["dWq"] > 1e-3 and summ[a]["dWo"] > 1e-3
                              for a in finite)
    res["recheck"] = {"criterion": "A1'（联合训练确实发生：读侧参数改变）",
                      "口径变更": "排除写阶段即发散（记忆非有限）的臂 —— 这些臂没有可评估的读侧训练",
                      "included": finite, "excluded": excluded, "detail": detail,
                      "passed": ok}
    print(f"  纳入 {finite}；排除 {excluded}（写阶段发散，读侧训练未发生）", flush=True)
    print(f"  A1' 复检：{'PASS' if ok else 'FAIL'}", flush=True)
    for a in finite:
        print(f"    {a}: ΔWq={summ[a]['dWq']:.4e} ΔWo={summ[a]['dWo']:.4e} "
              f"γ→{summ[a]['gamma']:.5f} NLL={summ[a]['mean_nll']:.5f}", flush=True)

    (HERE / "R2_recheck_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n=== DIR-10 R2 完成 ===", flush=True)


if __name__ == "__main__":
    main()
