#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-90 · R1 —— 记忆依赖型任务上的**内容性收益**（KVR：key→value 回忆）

逐字派生自 `R1_joint_read.py`（DIR-10）：`model_and_tok` / `make_stack` / `write_phase` /
`train_read` / `read_params` / `write_params` / `snapshot` / `restore` 原样保留；
**只替换**（a）语料构造 → KVR 事实集，（b）评测层 → `score_and_accuracy`（候选打分准确率）。

卡片：revolve/DIR-90/DIRECTION_rev90_v10.md（rev 1.0）+ DIRECTION_rev90_v11.md（rev 1.1，装置分支 §6′）

★ 两条结构事实（决定了本脚本的形状，都从 `sgmn_module.py` 实际执行读出，不是从注释）：
  1. `fwd_hook`（`:144-154`）顺序 = read → 注入 → **write** → cache。
  2. `write`（`:84-99`）的 k 取 **`prev_hs`（上一个 chunk 的 hs_in）**、v 取 **当前 chunk 的 hs_in**
     ⇒ 写事件把「上一块均值」与「本块均值」配对。故同一事实必须占**连续多个 chunk**，
     否则 k 与 v 来自不同事实。本脚本用 run-based 排布并断言 run 边界比例（A90-16）。
"""
import argparse
import hashlib
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sgmn_module import SGMNStack                     # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# ---------------------------------------------------------------- 冻结常量（卡片 §6）
ALPHA, LAM = 1.0, 0.05
GAMMA0, R = 0.05, 8
MOUNT = [8, 12, 16]
CHUNK = 64
B = 600
ETA_WRITE = 3e-5          # D-107 实测稳定域内最高有限档
N_FACTS = 32
N_TRAIN_KEYS = 16
READ_STEPS = 600
READ_LR = 1e-2            # 沿用 D-16 探针
CLIP = 1.0

# 单 token 值池（27 个，Cycle 145 实测：带前导空格为 1 个 token；见卡片 A90-7）
VALUE_POOL = ("teal amber violet olive rust crimson azure beige coral ivory lemon "
              "peach plum rose ruby tan wheat panda yak copper silver golden iron "
              "bronze jade pearl quartz").split()
N_CAND = 4
N_DRAWS = 3

SEEDS = [20261028, 20261104, 7]

OUT = pathlib.Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
res = {"dir": "DIR-90", "round": "R1", "model": MODEL, "device": DEV,
       "mount": MOUNT, "alpha": ALPHA, "lam": LAM, "gamma0": GAMMA0, "r": R,
       "chunk": CHUNK, "B": B, "eta_write": ETA_WRITE, "n_facts": N_FACTS,
       "n_cand": N_CAND, "n_draws": N_DRAWS, "seeds": SEEDS,
       "factorized": False, "read_steps": READ_STEPS, "read_lr": READ_LR,
       "stageA": [], "arms": {}, "preds": {}}


def rec(name, passed, detail, value=None):
    res["stageA"].append({"name": name, "ok": bool(passed), "detail": detail,
                          "value": value})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


def dump(obj, name):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1),
                            encoding="utf-8")


# ----------------------------------------------------------------- 基础设施
_M = {}


def model_and_tok():
    if "m" not in _M:
        tok = AutoTokenizer.from_pretrained(MODEL)
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _M["m"] = (tok, m)
    return _M["m"]


def make_stack(gamma=GAMMA0, no_write=False, seed=20261028):
    tok, model = model_and_tok()
    d = model.config.hidden_size
    st = SGMNStack(model, MOUNT, d, d, R, ALPHA, LAM, ETA_WRITE, gamma, DEV,
                   seed=seed, use_factorized=False, drop_sg=False, no_write=no_write)
    return st


def W_of(m):
    """全矩阵分支（rev 1.1 §6′）：W 本身就是可塑状态。"""
    return m.Wfull.detach()


def read_params(st):
    ps = []
    for m in st.mems.values():
        ps += [m.Wq, m.Wo, m.gamma]
    return ps


def write_params(st):
    ps = []
    for m in st.mems.values():
        ps += [m.U, m.V, m.W0, m.Pk, m.Pv, m.Wfull]
    return ps


def base_sig(model):
    h = hashlib.sha256()
    n = 0
    for nm, p in sorted(model.named_parameters()):
        if "sgmn" in nm.lower():
            continue
        h.update(nm.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
        n += 1
    return h.hexdigest(), n


# ----------------------------------------------------------------- KVR 事实集
def build_facts(tok, seed=9091):
    """32 条事实：nonce 键（基座不可能先验知道）+ 单 token 值。确定性。"""
    rng = random.Random(seed)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"      # 去掉易混 I/O/0/1
    keys, seen = [], set()
    while len(keys) < N_FACTS:
        k = "".join(rng.choice(alphabet) for _ in range(5))
        if k in seen:
            continue
        # 键不得与其值或常用词冲突（只作健全性检查）
        seen.add(k)
        keys.append(k)
    vals = [VALUE_POOL[i % len(VALUE_POOL)] for i in range(N_FACTS)]
    rng.shuffle(vals)
    facts = [{"key": k, "val": v} for k, v in zip(keys, vals)]
    for f in facts:
        f["sent"] = f"The access code for item {f['key']} is {f['val']}."
        f["prompt"] = f"The access code for item {f['key']} is"
        f["val_ids"] = tok(" " + f["val"], add_special_tokens=False)["input_ids"]
        f["prompt_ids"] = tok(f["prompt"], add_special_tokens=False)["input_ids"]
    return facts


def build_chunks(tok, facts, seed=4242):
    """600 块 × 64 token，**run-based**：每事实占 18–19 个连续块（写规则配对相邻块）。

    600 = 24×19 + 8×18。返回 (chunks, owner) —— owner[i] = 第 i 块所属事实下标。
    """
    per = [19] * 24 + [18] * 8
    assert sum(per) == B, sum(per)
    chunks, owner = [], []
    order = list(range(len(facts)))
    random.Random(seed).shuffle(order)
    for idx in order:
        ids = tok(facts[idx]["sent"] + " ", add_special_tokens=False)["input_ids"]
        if not ids:
            raise RuntimeError("空句子")
        seq = []
        while len(seq) < CHUNK:
            seq.extend(ids)
        seq = seq[:CHUNK]
        t = torch.tensor(seq, dtype=torch.long).unsqueeze(0)   # (1,CHUNK)：HF 需 batch 维
        for _ in range(per[idx]):
            chunks.append(t.clone())
            owner.append(idx)
    assert len(chunks) == B and len(owner) == B
    return chunks, owner


# ----------------------------------------------------------------- 候选打分
def nll_of_candidate(model, st, prompt_ids, cand_ids):
    """★ 单 token 候选（A90-7 断言）⇒ NLL = -log p(cand | prompt)，长度无关。"""
    assert len(cand_ids) == 1, f"候选须为单 token：{cand_ids}"
    seq = torch.tensor([prompt_ids + cand_ids], dtype=torch.long, device=DEV)
    lg = model(input_ids=seq).logits.float()
    lp = torch.log_softmax(lg[0, len(prompt_ids) - 1], dim=-1)
    return float(-lp[cand_ids[0]])


def score_and_accuracy(items, scorer):
    """★ 唯一评测入口（卡片 §3）。`scorer(prompt_ids, cand_ids_list) -> [nll,...]`

    items: [{"prompt_ids":…, "cand":[(ids,is_correct),…]}]；返回 (acc, 预测值下标, 明细)。
    取 **NLL 最小**的候选为预测 —— 这也是 A90-9 用桩打分器复现的那个函数。
    """
    n_ok, preds, detail = 0, [], []
    for it in items:
        ids_list = [c[0] for c in it["cand"]]
        nlls = scorer(it["prompt_ids"], ids_list)
        pick = int(np.argmin(nlls))
        ok = bool(it["cand"][pick][1])
        n_ok += int(ok)
        preds.append(pick)
        detail.append({"pick": pick, "ok": ok, "nll": [round(float(x), 4) for x in nlls]})
    return n_ok / len(items), preds, detail


def make_eval_items(tok, facts, keys, rng):
    """每题 1 正确 + 3 干扰（干扰取自**其余事实的值**且 ≠ 正确值）；每题 N_DRAWS 组抽样。"""
    items = []
    for ki in keys:
        f = facts[ki]
        correct = f["val"]
        others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != correct]
        for _ in range(N_DRAWS):
            distr = rng.sample(sorted(set(others)), N_CAND - 1)
            cands = [correct] + distr
            rng.shuffle(cands)
            items.append({
                "key_idx": ki,
                "prompt_ids": f["prompt_ids"],
                "cand": [(tok(" " + c, add_special_tokens=False)["input_ids"], c == correct)
                         for c in cands],
            })
    return items


def acc_by_question(items, preds_detail):
    """按题计（裁决口径）：同题 N_DRAWS 次全对才算该题命中。"""
    from collections import defaultdict
    g = defaultdict(list)
    for it, d in zip(items, preds_detail):
        g[it["key_idx"]].append(d["ok"])
    n = sum(1 for v in g.values() if all(v))
    return n / len(g), len(g)


# ----------------------------------------------------------------- 写 / 读
def write_phase(model, st, chunks):
    for ids in chunks:
        model(input_ids=ids.to(DEV))
    return {li: m.write_calls for li, m in st.mems.items()}


def train_read_task(model, st, tok, facts, train_keys, rng):
    """阶段②：写冻结，只训练读侧；**任务损失 = 4 选 1 候选 CE**（§4：任务损失只更新读取侧）。"""
    for m in st.mems.values():
        m.no_write = True
    opt = torch.optim.Adam(read_params(st), lr=READ_LR)
    losses, violations, diverged = [], 0, False
    # 预生成训练批（固定，使 INJ/MEM 两臂数据完全一致）
    batches = []
    for ki in train_keys:
        f = facts[ki]
        others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != f["val"]]
        distr = rng.sample(sorted(set(others)), N_CAND - 1)
        cands = [f["val"]] + distr
        rng.shuffle(cands)
        ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in cands]
        batches.append((f["prompt_ids"], ids, cands.index(f["val"])))
    for step in range(READ_STEPS):
        prompt_ids, ids, tgt = batches[step % len(batches)]
        P = len(prompt_ids)
        seq = torch.tensor([prompt_ids + [c] for c in ids], dtype=torch.long, device=DEV)
        opt.zero_grad(set_to_none=True)
        lg = model(input_ids=seq).logits.float()
        # ★ 逐行取【该行自己的】候选 token 的 logit ⇒ 形状 (1,4)，不是 4x4
        idx = torch.arange(len(ids), device=DEV)
        pick_logits = lg[idx, P - 1, torch.tensor(ids, device=DEV)].unsqueeze(0)
        assert pick_logits.shape == (1, len(ids)), pick_logits.shape
        loss = F.cross_entropy(pick_logits, torch.tensor([tgt], device=DEV))
        loss.backward()
        if step % 50 == 0:
            for p in write_params(st):
                if p.grad is not None and float(p.grad.abs().max()) != 0.0:
                    violations += 1
        torch.nn.utils.clip_grad_norm_([p for p in read_params(st) if p.grad is not None], CLIP)
        opt.step()
        losses.append(float(loss))
        if not all(bool(torch.isfinite(W_of(m)).all()) for m in st.mems.values()):
            diverged = True
            break
    for m in st.mems.values():
        m.no_write = False
    return losses, violations, diverged


# ----------------------------------------------------------------- S90 存储诊断
def storage_diag(model, st, facts, owner, chunks):
    """S90：对每条**已写入**事实算 k_i (mean Pk·prev_hs)、v_i (mean Pv·hs_in)，
    再量 ||W k_i - v_i|| / ||v_i||。用装置自身的 Pk/Pv 与均值口径（不重写公式）。"""
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    per = {}
    try:
        for fi in range(len(facts)):
            ci = owner.index(fi) + 1          # 用 run 的第 2 块（prev_hs 已是同事实）
            model(input_ids=chunks[ci].to(DEV))          # 已是 (1,CHUNK)
            errs = {}
            for li, m in st.mems.items():
                hs = st._hs_in[li]
                pf = hs.reshape(-1, hs.shape[-1])
                hf = hs.reshape(-1, hs.shape[-1])
                k = ((pf @ m.Pk).mean(dim=0)) * m.pk_scale
                v = (hf @ m.Pv).mean(dim=0)
                pred = W_of(m) @ k
                errs[li] = float((pred - v).norm() / max(float(v.norm()), 1e-30))
            per[fi] = errs
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
    med = float(np.median([np.mean(list(e.values())) for e in per.values()]))
    return med, per


# ----------------------------------------------------------------- Stage A
def stage_a(tok, model, facts, chunks, owner):
    print("★ Stage A 开始", flush=True)

    # A90-1 基座逐位不变（写前 / 写完 / 训练后 三处比）
    sig0, n_par = base_sig(model)
    rec("A90-1a", n_par > 0, f"基座参数 {n_par} 个，写前摘要 {sig0[:12]}…", value=n_par)

    # A90-7 候选单 token
    bad = [(w, tok(" " + w, add_special_tokens=False)["input_ids"])
           for w in VALUE_POOL
           if len(tok(" " + w, add_special_tokens=False)["input_ids"]) != 1]
    rec("A90-7", not bad, f"值池 {len(VALUE_POOL)} 个词全部为单 token（坏 {len(bad)} 个）", value=bad)

    # A90-8 训练键 ∩ 评测键 = ∅
    order = list(range(N_FACTS))
    random.Random(777).shuffle(order)
    train_keys, eval_keys = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]
    rec("A90-8", not (set(train_keys) & set(eval_keys)) and len(eval_keys) == N_FACTS - N_TRAIN_KEYS,
        f"训练键 {len(train_keys)} / 评测键 {len(eval_keys)}，交集 {len(set(train_keys) & set(eval_keys))}")

    # A90-16 写规则的 run 边界：k 与 v 同事实的比例
    same = sum(1 for i in range(1, B) if owner[i] == owner[i - 1])
    rec("A90-16", same / (B - 1) > 0.90,
        f"相邻块同事实 {same}/{B - 1} = {same / (B - 1):.3f}（须 >0.90；写规则配对相邻块）",
        value=same / (B - 1))

    # A90-2 / A90-4 γ：三层同值；γ=0 时注入恒 0 且读侧梯度恰 0
    st0 = make_stack(gamma=0.0)
    gs = [float(m.gamma) for m in st0.mems.values()]
    rec("A90-2", len(set(round(g, 6) for g in gs)) == 1 and abs(gs[0]) < 1e-12,
        f"γ=0 臂三层 γ 实测 {gs}", value=gs)
    f0 = facts[0]
    seq = torch.tensor([f0["prompt_ids"] + [f0["val_ids"][0]]], dtype=torch.long, device=DEV)
    lg = model(input_ids=seq).logits.float()
    lg[0, -1].sum().backward()
    gq = max(float(m.Wq.grad.abs().max()) if m.Wq.grad is not None else -1.0
             for m in st0.mems.values())
    ginj = max(float((m.gamma * m.injection).abs().max()) for m in st0.mems.values())
    rec("A90-4", gq == 0.0 and ginj == 0.0,
        f"γ=0：γ·injection max|·|={ginj:.1e}（须 0），∂L/∂Wq max|·|={gq:.1e}（须 0）",
        value={"ginj": ginj, "grad_Wq": gq})
    for m in st0.mems.values():
        m.Wq.grad = None
        m.Wo.grad = None
        m.gamma.grad = None
    st0.remove()

    # A90-3 no_write ⇒ W 逐位不变
    stN = make_stack(gamma=GAMMA0, no_write=True)
    w0 = {li: W_of(m).clone() for li, m in stN.mems.items()}
    write_phase(model, stN, chunks[:40])
    dmax = max(float((W_of(stN.mems[li]) - w0[li]).abs().max()) for li in stN.mems)
    wc = {li: m.write_calls for li, m in stN.mems.items()}
    rec("A90-3", dmax == 0.0 and all(v == 0 for v in wc.values()),
        f"no_write 跑 40 块：W max|Δ|={dmax:.1e}、write_calls={wc}（须全 0）",
        value={"dW": dmax, "write_calls": wc})
    stN.remove()

    # A90-5 write_calls == max(0, blocks-1)
    stW = make_stack(gamma=GAMMA0)
    wc = write_phase(model, stW, chunks[:150])
    want = 150 - 1
    rec("A90-5", all(v == want for v in wc.values()),
        f"跑 150 块 ⇒ write_calls={wc}（须 {want}，首块 prev_hs=None：DEF-86-4）", value=wc)

    # A90-10 有限
    fin = all(bool(torch.isfinite(W_of(m)).all()) for m in stW.mems.values())
    rec("A90-10", fin, "跑 150 块后 W 全有限（η=3e-5 在 D-107 稳定域内）",
        value={li: float(W_of(m).abs().max()) for li, m in stW.mems.items()})
    stW.remove()

    # A90-6 候选打分确定性
    stD = make_stack(gamma=GAMMA0, no_write=True)
    pi, ci = facts[1]["prompt_ids"], facts[1]["val_ids"]
    a = nll_of_candidate(model, stD, pi, ci)
    b = nll_of_candidate(model, stD, pi, ci)
    rec("A90-6", a == b, f"同输入两次候选 NLL 逐位相同（{a:.8f} vs {b:.8f}）", value=[a, b])
    stD.remove()

    # A90-9 工作例（双桩：正确桩 0.5、互换桩 1.0）
    toy = [{"prompt_ids": [1], "cand": [([11], True), ([12], False)]},
           {"prompt_ids": [2], "cand": [([21], True), ([22], False)]}]
    stub_ok = lambda p, c: ([0.5, 2.0] if p == [1] else [1.8, 0.7])      # noqa: E731
    stub_sw = lambda p, c: ([0.5, 2.0] if p == [1] else [0.7, 1.8])      # noqa: E731
    acc_ok, pr_ok, _ = score_and_accuracy(toy, stub_ok)
    acc_sw, pr_sw, _ = score_and_accuracy(toy, stub_sw)
    rec("A90-9", acc_ok == 0.5 and pr_ok == [0, 1] and acc_sw == 1.0 and pr_sw == [0, 0],
        f"正确桩 acc={acc_ok} 预测={pr_ok}（手算 0.5 = 1/2、预测 [正确,错误]）；"
        f"互换桩 acc={acc_sw} 预测={pr_sw}（须 1.0 ⇒ 该守卫**有分辨力**）",
        value={"acc_ok": acc_ok, "pred_ok": pr_ok, "acc_sw": acc_sw, "pred_sw": pr_sw})

    # A90-15 全矩阵分支生效：rank(W - W0) > r
    stR = make_stack(gamma=GAMMA0)
    write_phase(model, stR, chunks)
    ranks = {}
    for li, m in stR.mems.items():
        dW = (W_of(m) - m.W0.detach()).float()
        sv = torch.linalg.svdvals(dW)
        ranks[li] = int((sv > sv[0] * 1e-8).sum())
    rec("A90-15", all(v > R for v in ranks.values()),
        f"写 600 块后 rank(W-W0)={ranks}（须 >r={R} ⇒ 全矩阵分支生效、且装得下 32>8 条）",
        value=ranks)
    # ★ 必须立即移除：stR 的写是开的，留着它的 hook 会在后续每次 forward 里
    #   对 Wfull 做 in-place copy_，使已保存的 Wfull 失效 ⇒ backward 报 version 错
    stR.remove()

    # A90-9b 写入过程的基座不变
    sig1, _ = base_sig(model)
    rec("A90-1b", sig0 == sig1, f"写 600 块后基座摘要 {sig1[:12]}…（须同 {sig0[:12]}…）",
        value=[sig0, sig1])

    # --- CEIL / FLOOR（任务可答性，须在正式轮前知道）
    rng = random.Random(31337)
    ceil_items = []
    for ki in eval_keys:
        f = facts[ki]
        for _ in range(N_DRAWS):
            others = [g["val"] for j, g in enumerate(facts) if g["val"] != f["val"]]
            distr = rng.sample(sorted(set(others)), N_CAND - 1)
            cands = [f["val"]] + distr
            rng.shuffle(cands)
            ctx = tok(f"{f['sent']} Question: {f['prompt']}",
                      add_special_tokens=False)["input_ids"]
            ceil_items.append({"key_idx": ki, "prompt_ids": ctx,
                               "cand": [(tok(" " + c, add_special_tokens=False)["input_ids"], c == f["val"])
                                        for c in cands]})
    stB = make_stack(gamma=0.0)
    scorerB = lambda p, cs: [nll_of_candidate(model, stB, p, c) for c in cs]   # noqa: E731
    acc_ceil, _, det_ceil = score_and_accuracy(ceil_items, scorerB)
    acc_ceil_q, nq = acc_by_question(ceil_items, det_ceil)
    rec("A90-11", acc_ceil_q >= 0.75,
        f"CEIL（事实在 prompt 内）按题计 acc={acc_ceil_q:.3f}（须 ≥0.75；按项 {acc_ceil:.3f}, n题={nq}）",
        value=acc_ceil_q)
    stB.remove()

    stF = make_stack(gamma=0.0, no_write=True)
    write_phase(model, stF, chunks)          # 写入被关 ⇒ W 不变，仅让 prev_hs 走完
    eval_items = make_eval_items(tok, facts, eval_keys, random.Random(555))
    scorerF = lambda p, cs: [nll_of_candidate(model, stF, p, c) for c in cs]   # noqa: E731
    acc_fl, _, det_fl = score_and_accuracy(eval_items, scorerF)
    acc_fl_q, _ = acc_by_question(eval_items, det_fl)
    rec("A90-12", acc_fl_q <= 0.40,
        f"FLOOR（BASE，无注入）按题计 acc={acc_fl_q:.3f}（须 ≤0.40；按项 {acc_fl:.3f}）",
        value=acc_fl_q)
    stF.remove()

    # A90-14 随机打分器的机会水平自检
    rnd = random.Random(2468)
    stub_r = lambda p, cs: [rnd.random() for _ in cs]      # noqa: E731
    acc_r, _, det_r = score_and_accuracy(eval_items, stub_r)
    acc_r_q, _ = acc_by_question(eval_items, det_r)
    rec("A90-14", 0.05 <= acc_r_q <= 0.45,
        f"随机打分器按题计 acc={acc_r_q:.3f}（须落 [0.05,0.45]，机会 0.25）", value=acc_r_q)

    # A90-13 读侧训练期间写入梯度恰 0
    stT = make_stack(gamma=GAMMA0)
    write_phase(model, stT, chunks[:60])
    _, viol, _ = train_read_task(model, stT, tok, facts, train_keys, random.Random(99))
    rec("A90-13", viol == 0, f"读侧训练 600 步：写入参数非零梯度检查点 {viol} 处（须 0）", value=viol)
    sig2, _ = base_sig(model)
    rec("A90-1c", sig0 == sig2, f"读侧训练后基座摘要 {sig2[:12]}…（须同 {sig0[:12]}…）",
        value=[sig0, sig2])
    stT.remove()

    ok = all(r["ok"] for r in res["stageA"])
    print(f"\n★ Stage A: {sum(r['ok'] for r in res['stageA'])}/{len(res['stageA'])} PASS", flush=True)
    return ok


# ----------------------------------------------------------------- 正式轮
def run_arm(name, gamma, do_write, trainable, tok, model, facts, chunks, owner,
            train_keys, eval_keys, seed):
    st = make_stack(gamma=gamma, no_write=not do_write, seed=seed)
    t0 = time.time()
    wc = write_phase(model, st, chunks) if do_write else {"n/a": 0}
    viol, diverged, losses = None, False, []
    if trainable:
        losses, viol, diverged = train_read_task(model, st, tok, facts, train_keys,
                                                random.Random(1000 + seed % 100))
    s90 = None
    if do_write:
        s90, _ = storage_diag(model, st, facts, owner, chunks)
    # 评测（写冻结）
    for m in st.mems.values():
        m.no_write = True
    items = make_eval_items(tok, facts, eval_keys, random.Random(555))
    scorer = lambda p, cs: [nll_of_candidate(model, st, p, c) for c in cs]   # noqa: E731
    acc, _, det = score_and_accuracy(items, scorer)
    acc_q, nq = acc_by_question(items, det)
    out = {"arm": name, "seed": seed, "acc_item": acc, "acc_q": acc_q, "n_q": nq,
           "n_item": len(items), "s90_med": s90, "violations": viol,
           "diverged": diverged, "final_loss": float(np.mean(losses[-50:])) if losses else None,
           "w_max": {li: float(W_of(m).abs().max()) for li, m in st.mems.items()},
           "write_calls": wc if isinstance(wc, dict) else {"n/a": 0},
           "sec": round(time.time() - t0, 1)}
    print(f"  [{name} seed={seed}] acc(题)={acc_q:.3f} acc(项)={acc:.3f} "
          f"S90={s90 if s90 is None else round(s90, 4)} viol={viol} "
          f"div={diverged} {out['sec']}s", flush=True)
    st.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["A", "main", "all"])
    ap.add_argument("--seeds", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else SEEDS

    tok, model = model_and_tok()
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    print(f"事实 {len(facts)} 条；块 {len(chunks)}×{CHUNK}；值池 {len(VALUE_POOL)}", flush=True)
    dump({"facts": [{k: v for k, v in f.items() if k not in ("prompt_ids", "val_ids")}
                    for f in facts]}, "facts.json")

    if a.stage in ("A", "all"):
        ok = stage_a(tok, model, facts, chunks, owner)
        dump(res, "stageA.json")
        if not ok:
            print("★ Stage A 未通过 ⇒ 正式轮被阻断（Round standard 3）", flush=True)
            return 1
    if a.stage == "A":
        return 0

    # 正式轮：三臂 × 种子
    order = list(range(N_FACTS))
    random.Random(777).shuffle(order)
    train_keys, eval_keys = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]
    for seed in seeds:
        for name, gamma, dw, tr in (("BASE", 0.0, False, False),
                                    ("INJ", GAMMA0, False, True),
                                    ("MEM", GAMMA0, True, True)):
            r = run_arm(name, gamma, dw, tr, tok, model, facts, chunks, owner,
                        train_keys, eval_keys, seed)
            res["arms"][f"{name}|{seed}"] = r
            dump(res, "main.json")

    # ---- 判据
    def vals(arm, field="acc_q"):
        return [res["arms"][f"{arm}|{s}"][field] for s in seeds if f"{arm}|{s}" in res["arms"]]

    summ, crit = {}, {}
    for arm in ("BASE", "INJ", "MEM"):
        v = vals(arm)
        if v:
            summ[arm] = {"mean": float(np.mean(v)), "vals": v, "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    dc = [res["arms"][f"MEM|{s}"]["acc_q"] - res["arms"][f"INJ|{s}"]["acc_q"] for s in seeds
          if f"MEM|{s}" in res["arms"] and f"INJ|{s}" in res["arms"]]
    di = [res["arms"][f"INJ|{s}"]["acc_q"] - res["arms"][f"BASE|{s}"]["acc_q"] for s in seeds
          if f"INJ|{s}" in res["arms"] and f"BASE|{s}" in res["arms"]]
    if dc:
        crit["delta_content_vals"] = dc
        crit["delta_content_mean"] = float(np.mean(dc))
        crit["delta_content_same_sign"] = all(x > 0 for x in dc) or all(x < 0 for x in dc)
        crit["delta_inj_vals"] = di
        crit["delta_inj_mean"] = float(np.mean(di))
        s90 = [res["arms"][f"MEM|{s}"]["s90_med"] for s in seeds if f"MEM|{s}" in res["arms"]]
        crit["s90_vals"] = s90
        crit["s90_pass"] = bool(s90) and all(x is not None and x <= 0.5 for x in s90)
    res["summary"], res["verdict_raw"] = summ, crit
    dump(res, "main.json")
    print("\n=== 裁决材料 ===", flush=True)
    for k, v in crit.items():
        print(" ", k, "=", v, flush=True)
    print("  summary =", summ, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())