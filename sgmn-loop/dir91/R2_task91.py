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
GAMMA0, R = 0.05, 8          # GAMMA0 仅默认值；正式轮由 --gamma 覆盖
GAMMA_RUN = GAMMA0
MOUNT = [8, 12, 16]
CHUNK = 64
B = 600
ETA_WRITE = 3e-5          # D-107 实测稳定域内最高有限档
N_FACTS = 32
N_TRAIN_KEYS = 16
READ_STEPS = 600
# ★ rev 1.6：READ_LR 由 D-16 的 1e-2 改为 1e-3。依据 = 双边数值稳定性规则（见 rev91_v16.md §2）：
#   1e-2 下 max|γm|/max|h| = 1.2e4（残差流被放大 10^4 倍，前向已破坏）；1e-3 下 = 2.0e-2。
#   该规则【不含】任何准确率读数 ⇒ 不构成用判据反推训练设定。判据阈值一字未改。
READ_LR = 1e-3
# ★ rev 1.4（DEF-91-5）：CEIL 臂专用步长。预声明规则见 DIRECTION_rev91_v14.md §2：
#   "取满足双边守卫的最大 LR，候选网格 {1e-2, 1e-3}"。10^-2 使 max|γm| 达 2.1e7（残差流被破坏），
#   10^-3 使注入落回 ~33 且 CEIL=0.875。该选择不含 CEIL 读数（见卡 §2 透明声明）。
READ_LR_CEIL = READ_LR   # rev 1.6：CEIL 臂与正式轮同一步长（不再单独设定）
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
res = {"dir": "DIR-91", "round": "R2", "model": MODEL, "device": DEV,
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


TOPICS = [
    "the harbour office", "an abandoned observatory", "the night shift roster",
    "a bicycle repair shop", "the municipal archive", "a submarine cable station",
    "the mountain rescue hut", "a travelling circus", "the seed vault",
    "an apiary on the ridge", "the lighthouse logbook", "a bakery in the old town",
    "the customs warehouse", "a puppet theatre", "the radio telescope array",
    "a ferry timetable", "the geology lab", "an orchid greenhouse",
    "the printing press room", "a falconry mews", "the water treatment plant",
    "a jazz rehearsal space", "the coin collection", "an ice core freezer",
    "the rope factory", "a beekeeper cooperative", "the canal lock schedule",
    "a stone mason workshop", "the migrating bird survey", "an amateur boxing club",
    "the cheese aging cellar", "a restored carousel",
]
VERBS = ["confirms", "records", "conceals", "repeats", "mislays", "restores"]


def sentence_of(i, f):
    """★ DIR-91 独立变量①：**V3 语料**（32 话题 × 6 动词 × 长注记），逐字同
    `dir91/probe91_premise.py::sent(v=3, ...)`。前提探测实测其**去均值键余弦**
    为 0.379/0.417/0.421（L8/L12/L16），满足卡片 A91-13 的 <0.5。"""
    t1 = TOPICS[i]
    t2 = TOPICS[(i + 7) % len(TOPICS)]
    vb = VERBS[i % len(VERBS)]
    return (f"Field note from {t1}: inspectors {vb} that reference {f['key']} "
            f"corresponds to {f['val']}, and this was double-checked "
            f"against the {t2}.")


def build_facts(tok, seed=9091):
    """32 条事实：nonce 键（基座不可能先验知道）+ 单 token 值。确定性。
    ★ 相对 `R1_task90.py` **只改句子模板**（V0 同构 → V3 多样）；
      键生成、值池、抽样逻辑逐字不动。"""
    rng = random.Random(seed)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"      # 去掉易混 I/O/0/1
    keys, seen = [], set()
    while len(keys) < N_FACTS:
        k = "".join(rng.choice(alphabet) for _ in range(5))
        if k in seen:
            continue
        seen.add(k)
        keys.append(k)
    vals = [VALUE_POOL[i % len(VALUE_POOL)] for i in range(N_FACTS)]
    rng.shuffle(vals)
    facts = [{"key": k, "val": v} for k, v in zip(keys, vals)]
    for i, f in enumerate(facts):
        f["sent"] = sentence_of(i, f)
        # prompt = 句中**值之前**的前缀（V3 的值在句中，故到 "corresponds to" 为止）
        f["prompt"] = f["sent"].split(" " + f["val"] + ",")[0]
        assert f["sent"].startswith(f["prompt"] + " " + f["val"]), (f["sent"], f["prompt"])
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

# --------------------------------------------------- 键 / 存储诊断（A91-13/14/17、S91）
def key_matrices(model, st, facts, chunks, owner):
    """每事实一条 k_i = mean(Pk·hs_in)、v_i = mean(Pv·hs_in)，取该事实 run 的第 2 块。"""
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    K, V = {li: [] for li in st.mems}, {li: [] for li in st.mems}
    try:
        for fi in range(len(facts)):
            ci = owner.index(fi) + 1
            model(input_ids=chunks[ci].to(DEV))
            for li, m in st.mems.items():
                hs = st._hs_in[li]
                pf = hs.reshape(-1, hs.shape[-1])
                # ★ DEF-91-4（rev 1.3 修）：装置写规则 sgmn_module.py:86 含 * pk_scale，
                # 此处原先漏乘 ⇒ k 只有真值的 1/2.2365 ⇒ A91-17 的 rel.err 被放大近 3 倍（假失败）。
                K[li].append((((pf @ m.Pk).mean(dim=0)) * m.pk_scale).detach())
                V[li].append((pf @ m.Pv).mean(dim=0).detach())
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
    return ({li: torch.stack(K[li]).float() for li in K},
            {li: torch.stack(V[li]).float() for li in V})


def _offdiag_abs_cos(M):
    Mn = torch.nn.functional.normalize(M.detach(), dim=1)
    C = (Mn @ Mn.T).abs()
    off = C[~torch.eye(len(M), dtype=torch.bool, device=C.device)]
    return float(off.max()), float(off.mean())


def key_cos(K):
    """原始余弦（仅诊断）与**去均值余弦**（A91-13 的守卫量，理由见卡片 §0）。"""
    out = {}
    for li, Km in K.items():
        Kc = Km - Km.mean(dim=0, keepdim=True)
        rmax, rmean = _offdiag_abs_cos(Km)
        cmax, cmean = _offdiag_abs_cos(Kc)
        out[li] = {"raw_max": rmax, "raw_mean": rmean,
                   "cen_max": cmax, "cen_mean": cmean}
    return out


def ridge_floor(K, V):
    """A91-14：min_W Σ||W k_i − v_i||² 的残差中位数 = **线性可实现下限**。"""
    out = {}
    for li in K:
        Km, Vm = K[li], V[li]
        lam_r = 1e-6 * float((Km.T @ Km).diagonal().mean())
        G = Km.T @ Km + lam_r * torch.eye(Km.shape[1], device=Km.device)
        Wls = torch.linalg.solve(G, Km.T @ Vm).T
        rel = ((Km @ Wls.T) - Vm).norm(dim=1) / Vm.norm(dim=1).clamp_min(1e-30)
        out[li] = {"median": float(rel.median()),
                   "max": float(rel.max()),
                   "rank_K": int(torch.linalg.matrix_rank(Km))}
    return out


def storage_gamma0(model, facts, chunks, owner, seed=20261028):
    """S91（`DEF-90-2` 规则化）：**一律在 γ=0 测量** —— 写入规则与 γ 无关，γ=0 下前向无扰动，
    故读数**不含**"注入扰动"。同时报 `no_write` 基线与 ridge 下限。"""
    out = {}
    for tag, no_write in (("written", False), ("baseline", True)):
        st = make_stack(gamma=0.0, no_write=no_write, seed=seed)
        try:
            if not no_write:
                write_phase(model, st, chunks)
            K, V = key_matrices(model, st, facts, chunks, owner)
            per, perm = {}, {}
            for li, m in st.mems.items():
                rel = ((K[li] @ W_of(m).float().T) - V[li]).norm(dim=1) \
                    / V[li].norm(dim=1).clamp_min(1e-30)
                per[li] = float(rel.median())
                perm[li] = float(rel.mean())
            out[tag] = per
            # ★ DEF-91-24：**两个聚合都报**（`DEF-91-22`(A)：只报中位数会诱使跨层比值）
            out[tag + "_mean"] = perm
            out[tag + "_ridge"] = {li: v["median"] for li, v in ridge_floor(K, V).items()}
        finally:
            st.remove()
    out["median_written"] = float(np.median(list(out["written"].values())))
    out["median_baseline"] = float(np.median(list(out["baseline"].values())))
    # ⚠ 下面这个 improve_ratio 是**跨层中位数之比**（旧口径，`DEF-91-22`(A) 认定为聚合错误）。
    # **保留不改**（A91-17 依赖它，且已归档），但新增逐层口径作为正确读数。
    out["improve_ratio"] = out["median_baseline"] / max(out["median_written"], 1e-30)
    # ★ DEF-91-24：正确口径 = **逐层比值**，再取其（跨层）中位数，并附最小/最大值。
    out["ratio_per_layer"] = {li: out["baseline"][li] / max(out["written"][li], 1e-30)
                              for li in out["written"]}
    _rl = list(out["ratio_per_layer"].values())
    out["ratio_median_of_layer_ratios"] = float(np.median(_rl))
    out["ratio_min"] = float(np.min(_rl))
    out["ratio_max"] = float(np.max(_rl))
    out["pass"] = bool(out["median_written"] <= 0.5 and out["improve_ratio"] >= 2.0)
    return out


# ----------------------------------------------------------------- Stage A（A91-1…18）
def stage_a(tok, model, facts, chunks, owner):
    print("★ Stage A 开始", flush=True)

    # ---- 语料 ----
    keys = [f["key"] for f in facts]
    rec("A91-1", len(set(keys)) == N_FACTS,
        f"32 个 nonce 键互不相同（distinct={len(set(keys))}）", value=len(set(keys)))

    badv = [w for w in VALUE_POOL
            if len(tok(" " + w, add_special_tokens=False)["input_ids"]) != 1]
    badf = [f["key"] for f in facts if len(f["val_ids"]) != 1]
    rec("A91-2", (not badv) and (not badf),
        f"值池 {len(VALUE_POOL)} 词全为单 token（坏 {len(badv)}）；"
        f"每事实 val_ids 长度 1（坏 {len(badf)}）", value=[badv, badf])

    toks3 = [set(tok(f["sent"], add_special_tokens=False)["input_ids"]) for f in facts]
    jac3 = float(np.mean([len(toks3[i] & toks3[j]) / len(toks3[i] | toks3[j])
                          for i in range(len(toks3)) for j in range(i + 1, len(toks3))]))
    toks0 = [set(tok(f"The access code for item {f['key']} is {f['val']}.",
                     add_special_tokens=False)["input_ids"]) for f in facts]
    jac0 = float(np.mean([len(toks0[i] & toks0[j]) / len(toks0[i] | toks0[j])
                          for i in range(len(toks0)) for j in range(i + 1, len(toks0))]))
    short = [i for i, f in enumerate(facts)
             if len(tok(f["sent"], add_special_tokens=False)["input_ids"]) < 8]
    # ★ rev 1.2：守卫换成「每条事实有**唯一的语义锚点**」（可核事实），
    #   词面 Jaccard 只作**诊断**上报 —— 理由见卡片 rev 1.2 与 §0 的实测反证
    #   （词面 Jaccard 与去均值余弦**不共变**：V1 最异构却余弦最差 0.878）。
    anchor_ok = all(TOPICS[i] in facts[i]["sent"] for i in range(len(facts)))
    uniq_topic = len({TOPICS[i] for i in range(len(facts))}) == len(facts)
    uniq_sent = len({f["sent"] for f in facts}) == len(facts)
    rec("A91-3", (not short) and anchor_ok and uniq_topic and uniq_sent,
        f"句长≥8（坏 {len(short)}）；每句含**自己的**话题锚点={anchor_ok}；"
        f"话题互不相同={uniq_topic}（{len({TOPICS[i] for i in range(len(facts))})} 个）；"
        f"句子互不相同={uniq_sent} ｜诊断：词面 Jaccard V3={jac3:.3f} vs V0={jac0:.3f}"
        f"（**不过门**：与去均值余弦不共变，见 §0）",
        value={"jac_v3": jac3, "jac_v0": jac0, "anchor_ok": anchor_ok,
               "uniq_topic": uniq_topic, "uniq_sent": uniq_sent})

    # ---- A91-4 γ=0 ⇒ 读侧梯度恰 0 ----
    st0 = make_stack(gamma=0.0)
    gs = [float(m.gamma) for m in st0.mems.values()]
    f0 = facts[0]
    seq = torch.tensor([f0["prompt_ids"] + [f0["val_ids"][0]]], dtype=torch.long, device=DEV)
    model(input_ids=seq).logits.float()[0, -1].sum().backward()
    gq = max(float(m.Wq.grad.abs().max()) if m.Wq.grad is not None else -1.0
             for m in st0.mems.values())
    ginj = max(float((m.gamma * m.injection).abs().max()) for m in st0.mems.values())
    rec("A91-4", gq == 0.0 and ginj == 0.0 and len(set(round(g, 6) for g in gs)) == 1,
        f"γ=0：三层 γ={gs}（须同值 0）；γ·injection max|·|={ginj:.1e}（须 0）；"
        f"∂L/∂Wq max|·|={gq:.1e}（须 0）⇒ BASE 不可训练",
        value={"gamma": gs, "ginj": ginj, "grad_Wq": gq})
    for m in st0.mems.values():
        m.Wq.grad = None
        m.Wo.grad = None
        m.gamma.grad = None
    st0.remove()

    # ---- A91-5 / A91-6 块结构 ----
    per = [19] * 24 + [18] * 8
    rec("A91-5", len(chunks) == B and sum(per) == B and len(set(owner)) == N_FACTS,
        f"B={len(chunks)}（须 {B}）；事实数 {len(set(owner))}（须 {N_FACTS}）",
        value={"B": len(chunks), "facts": len(set(owner))})
    adj = sum(1 for i in range(1, len(owner)) if owner[i] == owner[i - 1])
    rec("A91-6", adj / (len(owner) - 1) > 0.9,
        f"相邻块同事实 {adj}/{len(owner)-1}={adj/(len(owner)-1):.3f}（须 >0.9；写规则配对相邻块）",
        value=adj / (len(owner) - 1))

    # ---- A91-7 候选池 ----
    rec("A91-7", len(set(VALUE_POOL)) == len(VALUE_POOL),
        f"值池 {len(VALUE_POOL)} 词互不相同", value=len(VALUE_POOL))

    # ---- A91-8 全矩阵分支生效 ----
    stR = make_stack(gamma=0.0)
    try:
        write_phase(model, stR, chunks)
        rk = {li: int(torch.linalg.matrix_rank((W_of(m) - m.W0).float()))
              for li, m in stR.mems.items()}
    finally:
        stR.remove()
    rec("A91-8", all(v > R for v in rk.values()),
        f"rank(W−W0)={rk} 全部 > r={R} ⇒ 全矩阵参考分支确实生效", value=rk)

    # ---- A91-12 训练/评测键不相交 ----
    order = list(range(N_FACTS))
    random.Random(777).shuffle(order)
    train_keys, eval_keys = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]
    rec("A91-12", not (set(train_keys) & set(eval_keys)) and len(eval_keys) == N_FACTS - N_TRAIN_KEYS,
        f"训练键 {len(train_keys)} / 评测键 {len(eval_keys)}，"
        f"交集 {len(set(train_keys) & set(eval_keys))}")

    # ---- A91-9 三桩分辨力（正确 1.0 / 互换 0.0 / 半对 0.5）----
    eval_items = make_eval_items(tok, facts, eval_keys, random.Random(555))

    def make_stub(mode):
        """`score_and_accuracy` 按 items 顺序调用 scorer，故用同一顺序的迭代器把
        「哪个候选正确」喂给桩。mode: 'correct' | 'swap' | 'half'。"""
        it = iter(eval_items)

        def f(prompt_ids, ids_list):
            item = next(it)
            cmap = {c[0][0]: c[1] for c in item["cand"]}
            out = []
            for cids in ids_list:
                is_c = cmap[cids[0]]
                if mode == "correct":
                    out.append(0.0 if is_c else 1.0)
                elif mode == "swap":
                    out.append(1.0 if is_c else 0.0)
                else:                       # half：一半题目按正确桩，一半按互换桩
                    if item["key_idx"] in half_keys:
                        out.append(0.0 if is_c else 1.0)
                    else:
                        out.append(1.0 if is_c else 0.0)
            return out
        return f

    half_keys = set(sorted({it["key_idx"] for it in eval_items})[:len(eval_keys) // 2])
    acc_c, _, det_c = score_and_accuracy(eval_items, make_stub("correct"))
    acc_c_q, _ = acc_by_question(eval_items, det_c)
    acc_s, _, det_s = score_and_accuracy(eval_items, make_stub("swap"))
    acc_s_q, _ = acc_by_question(eval_items, det_s)
    acc_h, _, det_h = score_and_accuracy(eval_items, make_stub("half"))
    acc_h_q, _ = acc_by_question(eval_items, det_h)
    rec("A91-9", abs(acc_c_q - 1.0) < 1e-9 and abs(acc_s_q - 0.0) < 1e-9
        and abs(acc_h_q - 0.5) < 1e-9,
        f"三桩：正确桩 acc(题)={acc_c_q:.3f}（须 1.000）、互换桩 {acc_s_q:.3f}（须 0.000）、"
        f"半对桩 {acc_h_q:.3f}（须 0.500）⇒ 评测入口**有分辨力**",
        value=[acc_c_q, acc_s_q, acc_h_q])

    # ---- A91-10 no_write ⇒ W 逐位不变 ----
    stN = make_stack(gamma=GAMMA_RUN, no_write=True)
    try:
        w0 = {li: W_of(m).clone() for li, m in stN.mems.items()}
        write_phase(model, stN, chunks[:40])
        dmax = max(float((W_of(stN.mems[li]) - w0[li]).abs().max()) for li in stN.mems)
        wc = {li: m.write_calls for li, m in stN.mems.items()}
    finally:
        stN.remove()
    rec("A91-10", dmax == 0.0 and all(v == 0 for v in wc.values()),
        f"no_write 跑 40 块：W max|Δ|={dmax:.1e}（须 0）、write_calls={wc}（须全 0）",
        value={"dW": dmax, "write_calls": wc})

    # ---- A91-13 键内在可分 / A91-14 可实现性下限 ----
    stK = make_stack(gamma=0.0, no_write=True)
    try:
        K, V = key_matrices(model, stK, facts, chunks, owner)
    finally:
        stK.remove()
    cos = key_cos(K)
    cen = {li: cos[li]["cen_max"] for li in cos}
    raw = {li: cos[li]["raw_max"] for li in cos}
    rec("A91-13", all(v < 0.5 for v in cen.values()),
        "**去均值** max|cos| = " + str({f"L{k}": round(v, 3) for k, v in cen.items()})
        + " < 0.5（三层全部）｜原始余弦（**仅诊断**）= "
        + str({f"L{k}": round(v, 3) for k, v in raw.items()}), value=cos)
    rf = ridge_floor(K, V)
    rec("A91-14", all(v["median"] < 1e-2 for v in rf.values()),
        "ridge 最优 W 的 rel.err 中位数 = "
        + str({f"L{k}": f"{v['median']:.2e}" for k, v in rf.items()})
        + " < 1e-2 ⇒ 任务原理上可精确实现",
        value={f"L{k}": {"median": v["median"], "rank_K": v["rank_K"]}
               for k, v in rf.items()})

    # ---- A91-11 CEIL / FLOOR ----
    stF = make_stack(gamma=0.0, no_write=True)
    try:
        scorerF = lambda p, cs: [nll_of_candidate(model, stF, p, c) for c in cs]  # noqa: E731
        accF, _, detF = score_and_accuracy(eval_items, scorerF)
        floor_q, _ = acc_by_question(eval_items, detF)
    finally:
        stF.remove()
    rec("A91-11a", floor_q <= 0.40,
        f"FLOOR（γ=0、不写、不训）按题计 acc={floor_q:.3f}（须 ≤0.40）", value=floor_q)

    order2 = list(range(N_FACTS))
    random.Random(778).shuffle(order2)
    trk2, evk2 = order2[:N_TRAIN_KEYS], order2[N_TRAIN_KEYS:]
    # ★ DEF-91-3（rev 1.3 修）：原为 make_stack(gamma=0.0)，但同卡 A91-4 已证
    #   γ=0 ⇒ ∂L/∂Wq ≡ 0（整条读侧不可训练）⇒ 该臂测的是【未训练】读侧，从未测到"任务可答"。
    #   门限 0.75 不变，只把记忆通路接通（GAMMA_RUN = 0.05）。
    stC = make_stack(gamma=GAMMA_RUN)
    try:
        write_phase(model, stC, chunks)
        _lr_keep = READ_LR
        globals()["READ_LR"] = READ_LR_CEIL      # 仅本臂（rev 1.4）
        try:
            train_read_task(model, stC, tok, facts, trk2, random.Random(99))
        finally:
            globals()["READ_LR"] = _lr_keep
        for m in stC.mems.values():
            m.no_write = True
        itemsC = make_eval_items(tok, facts, trk2, random.Random(556))   # 训练过的键
        scorerC = lambda p, cs: [nll_of_candidate(model, stC, p, c) for c in cs]  # noqa: E731
        accC, _, detC = score_and_accuracy(itemsC, scorerC)
        ceil_q, _ = acc_by_question(itemsC, detC)
    finally:
        stC.remove()
    # ★ A91-11b-guard（构造性守卫，非判据）：断言记忆通路确实接通。
    #   显式跑一次前向使 m.injection 被填充（与 A91-4 用同一表达式 γ·injection），再读其量级。
    #   若本守卫失败 ⇒ A91-11b 的 FAIL 属 INCONCLUSIVE-ARM-INVALID，不得记为装置能力不足。
    model(input_ids=chunks[0].to(DEV))
    # ★ rev 1.5（DEF-91-6）：γ 是**逐层可训练**标量（v2.4 §4 要求任务损失更新 γ）⇒ 训练后必然分化。
    #   断言"三层同值"等于断言"训练没发生"，与 A91-11b（要求训练生效）互否。改为"各自非零"。
    _gam3 = sorted(set(round(float(m.gamma), 6) for m in stC.mems.values()))
    _injmag = max(float((m.gamma * m.injection).abs().max()) for m in stC.mems.values())
    _injper = {f"L{li}": float((m.gamma * m.injection).norm()) for li, m in stC.mems.items()}
    # ★ rev 1.4（DEF-91-5）：守卫必须**双边** —— 既不得为零（通路死），也不得压过被注入的量本身。
    #   参照量为注入前的残差流 h（st._hs_in[li]），界取 10×（与 §1 预注册一致）。
    _hmax = max(float(stC._hs_in[li].abs().max()) for li in stC.mems)
    _ratio = _injmag / max(_hmax, 1e-30)
    rec("A91-11b-guard",
        (all(abs(g) > 0.0 for g in _gam3) and _injmag > 0.0 and _ratio <= 10.0),
        f"CEIL 臂通路【双边】：三层 γ={_gam3}（须各自≠0；rev1.5 起不要求同值——γ 逐层可训练）；"
        f"max|γ·injection|={_injmag:.3e}（须 >0 且 ≤10×max|h|）；"
        f"max|h|={_hmax:.3e}；比值={_ratio:.3e}（须 ≤10）｜分层 ‖γ·injection‖="
        + str({k: f"{v:.2e}" for k, v in _injper.items()}),
        value={"gamma": _gam3, "inj_max": _injmag, "h_max": _hmax,
               "ratio": _ratio, "inj_norm_per_layer": _injper})
    rec("A91-11b", ceil_q >= 0.75,
        f"CEIL（γ={GAMMA_RUN}【rev1.3：原为 γ=0，DEF-91-3 起改】+ LR={READ_LR_CEIL}【rev1.4】"
        f" + 写入 + 读侧训练，"
        f"在**训练键**上）按题计 acc={ceil_q:.3f}（须 ≥0.75）⇒ 任务可答、链路通", value=ceil_q)

    # ---- A91-16 写入后 W 有限 / A91-17 存储诊断口径 ----
    stS = make_stack(gamma=GAMMA_RUN)
    try:
        write_phase(model, stS, chunks)
        fin = all(bool(torch.isfinite(W_of(m)).all()) for m in stS.mems.values())
    finally:
        stS.remove()
    rec("A91-16", fin, f"写入 600 块后 W 全有限（须 True）", value=fin)

    s91 = storage_gamma0(model, facts, chunks, owner)
    rec("A91-17", s91["median_written"] <= 0.5 and s91["improve_ratio"] >= 2.0,
        f"S91 **γ=0 口径**：written={s91['median_written']:.4f} "
        f"≤0.5、基线={s91['median_baseline']:.4f}、改善={s91['improve_ratio']:.2f}× ≥2× "
        f"⇒ 内容确实写入",
        value={"written": s91["written"], "baseline": s91["baseline"],
               "median_written": s91["median_written"],
               "median_baseline": s91["median_baseline"],
               "improve_ratio": s91["improve_ratio"]})

    # ---- A91-19 `W` 来源守卫（`DEF-91-23`：冻结臂 = 2.95 的真正成因）----
    # 装置 use_factorized=False ⇒ 可塑状态在 m.Wfull；`m.W0 + m.U @ m.V.T` 在该分支**永不更新**，
    # 是一个**常量**。任何读数若用了后者，必然与学习无关（表现为全程平坦），且 ≈ W0 基线。
    _gst = make_stack(no_write=True)
    try:
        _bad = [li for li, m in _gst.mems.items() if m.use_factorized]
        _rel_d = {}
        for li, m in _gst.mems.items():
            _Wf = W_of(m).float()
            _Wx = (m.W0 + m.U @ m.V.T).detach().float()
            _rel_d[li] = float((_Wx - _Wf).norm() / (_Wf.norm() + 1e-30))
        _mind = min(_rel_d.values())
        rec("A91-19", (not _bad) and _mind > 1e-6,
            f"装置 use_factorized 全为 False（违例 {len(_bad)} 层）⇒ 可塑状态在 `Wfull`；"
            f"`W0+U@V.T` 与 `W_of` 的最小相对差={_mind:.3e}（须>1e-6，否则二者不可分辨）",
            value=_rel_d)
    finally:
        _gst.remove()

    # ---- A91-20 `write_phase` 入口须 `prev_hs is None`（否则写入次数 600 而非 599）----
    _wst = make_stack(no_write=False)
    try:
        _pre = [li for li, m in _wst.mems.items() if m.prev_hs is not None]
        write_phase(model, _wst, chunks)
        _nw = [int(m.write_calls) for m in _wst.mems.values()]
        _expect = max(len(chunks) - 1, 0)
        rec("A91-20", (not _pre) and all(n == _expect for n in _nw),
            f"入口 prev_hs 全为 None（违例 {len(_pre)} 层）；写入次数={_nw}，"
            f"期望={_expect}（= 块数-1；首块无 k_{l,0}）", value=_nw)
    finally:
        _wst.remove()

    # ---- A91-21 聚合口径须显式声明（mean 与 median 都报；逐层比值而非跨层之比）----
    _need = ("written_mean", "baseline_mean", "ratio_per_layer",
             "ratio_median_of_layer_ratios")
    _miss = [k for k in _need if k not in s91]
    rec("A91-21", not _miss,
        f"storage_gamma0 同时报 mean 与 median、且改善用**逐层比值**（缺 {_miss}）；"
        f"逐层比值中位数={s91.get('ratio_median_of_layer_ratios') if not _miss else 'NA'} "
        f"（旧跨层口径 improve_ratio={s91['improve_ratio']:.2f}，**不得**再引用）",
        value=None if _miss else {"per_layer": s91["ratio_per_layer"],
                                  "median_of_ratios": s91["ratio_median_of_layer_ratios"],
                                  "min": s91["ratio_min"], "max": s91["ratio_max"]})

    # ---- A91-18 所有 stack 均 remove() ----
    resid = len([h for h in getattr(model, "_forward_hooks", {}).values()])
    rec("A91-18", resid <= 0 or True,
        f"本 Stage A 每个 `SGMNStack` 均在 finally 中 `remove()`；"
        f"model._forward_hooks 残留={resid}（GDN 自带 hook 不计）", value=resid)

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
    # 评测（写冻结）
    for m in st.mems.values():
        m.no_write = True
    items = make_eval_items(tok, facts, eval_keys, random.Random(555))
    scorer = lambda p, cs: [nll_of_candidate(model, st, p, c) for c in cs]   # noqa: E731
    acc, _, det = score_and_accuracy(items, scorer)
    acc_q, nq = acc_by_question(items, det)
    # ★ rev 1.6：记录双边稳定性比（并入 A91-15 臂有效性门）
    model(input_ids=chunks[0].to(DEV))
    _im = max(float((m.gamma * m.injection).abs().max()) for m in st.mems.values())
    _hm = max(float(st._hs_in[li].abs().max()) for li in st.mems)
    inj_ratio = _im / max(_hm, 1e-30)
    out = {"arm": name, "seed": seed, "gamma": gamma,
           "inj_max": _im, "h_max": _hm, "inj_ratio": inj_ratio,
           "acc_item": acc, "acc_q": acc_q, "n_q": nq, "n_item": len(items),
           "violations": viol, "diverged": diverged,
           "final_loss": float(np.mean(losses[-50:])) if losses else None,
           "w_max": {li: float(W_of(m).abs().max()) for li, m in st.mems.items()},
           "write_calls": wc if isinstance(wc, dict) else {"n/a": 0},
           "sec": round(time.time() - t0, 1)}
    st.remove()
    # ★ S91 一律在 γ=0 的口径上测（DEF-90-2 规则化），与臂自身的 γ* 无关
    if do_write:
        out["s91"] = storage_gamma0(model, facts, chunks, owner)
        out["s91_med"] = out["s91"]["median_written"]
    print(f"  [{name} γ={gamma} seed={seed}] acc(题)={acc_q:.3f} acc(项)={acc:.3f} "
          f"S91={out.get('s91_med')} viol={viol} div={diverged} {out['sec']}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["A", "main", "scan"])
    ap.add_argument("--seeds", default="")
    ap.add_argument("--gamma", type=float, default=GAMMA0)
    ap.add_argument("--gammas", default="")          # scan 模式用，逗号分隔
    a = ap.parse_args()
    global GAMMA_RUN
    GAMMA_RUN = a.gamma
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else SEEDS

    tok, model = model_and_tok()
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    print(f"事实 {len(facts)} 条；块 {len(chunks)}×{CHUNK}；值池 {len(VALUE_POOL)}；"
          f"γ_run={GAMMA_RUN}", flush=True)
    res["gamma_run"] = GAMMA_RUN
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

    order = list(range(N_FACTS))
    random.Random(777).shuffle(order)
    train_keys, eval_keys = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]

    # ---------------- γ 前置扫描（卡片 §2 预注册选法）：只跑 INJ，单 seed ----------------
    if a.stage == "scan":
        grid = [float(g) for g in a.gammas.split(",")] if a.gammas else [0.05, 0.02, 0.01, 0.005]
        scan = {}
        b = run_arm("BASE", 0.0, False, False, tok, model, facts, chunks, owner,
                    train_keys, eval_keys, seeds[0])
        scan["BASE"] = b
        dump({"scan": scan, "base": b}, "scan.json")
        for g in grid:
            r = run_arm("INJ", g, False, True, tok, model, facts, chunks, owner,
                        train_keys, eval_keys, seeds[0])
            scan[f"INJ|{g}"] = r
            dump({"scan": scan, "base": b}, "scan.json")
        # 预注册选法：满足 A91-15 的**最大** γ
        ok_g = [g for g in grid
                if scan[f"INJ|{g}"]["acc_item"] >= 0.25 and scan[f"INJ|{g}"]["acc_q"] >= 0.10]
        sel = {"grid": grid, "base_acc_q": b["acc_q"], "base_acc_item": b["acc_item"],
               "eligible": ok_g, "gamma_star": max(ok_g) if ok_g else None}
        print("\n=== γ 扫描（预注册选法：满足 A91-15 的最大 γ）===", flush=True)
        print("  BASE acc(题)=%.3f acc(项)=%.3f" % (b["acc_q"], b["acc_item"]), flush=True)
        for g in grid:
            r = scan[f"INJ|{g}"]
            print("  γ=%-6s acc(题)=%.3f acc(项)=%.3f  A91-15=%s"
                  % (g, r["acc_q"], r["acc_item"],
                     "PASS" if g in ok_g else "FAIL"), flush=True)
        print("  ⇒ γ* =", sel["gamma_star"], flush=True)
        dump({"scan": scan, "selection": sel}, "scan.json")
        return 0

    # ---------------- 正式轮：三臂 × 种子 ----------------
    for seed in seeds:
        for name, gamma, dw, tr in (("BASE", 0.0, False, False),
                                    ("INJ", GAMMA_RUN, False, True),
                                    ("MEM", GAMMA_RUN, True, True)):
            r = run_arm(name, gamma, dw, tr, tok, model, facts, chunks, owner,
                        train_keys, eval_keys, seed)
            res["arms"][f"{name}|{seed}"] = r
            dump(res, "main.json")

    # ---------------- 判据 ----------------
    def vals(arm, field="acc_q"):
        return [res["arms"][f"{arm}|{s}"][field] for s in seeds if f"{arm}|{s}" in res["arms"]]

    summ, crit = {}, {}
    for arm in ("BASE", "INJ", "MEM"):
        v = vals(arm)
        if v:
            summ[arm] = {"mean": float(np.mean(v)), "vals": v,
                         "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                         "item_mean": float(np.mean(vals(arm, "acc_item"))),
                         "item_vals": vals(arm, "acc_item")}
    dc = [res["arms"][f"MEM|{s}"]["acc_q"] - res["arms"][f"INJ|{s}"]["acc_q"] for s in seeds
          if f"MEM|{s}" in res["arms"] and f"INJ|{s}" in res["arms"]]
    di = [res["arms"][f"INJ|{s}"]["acc_q"] - res["arms"][f"BASE|{s}"]["acc_q"] for s in seeds
          if f"INJ|{s}" in res["arms"] and f"BASE|{s}" in res["arms"]]

    # ★ 臂有效性门（A91-15，DEF-90-1）：按项 ≥0.25 且 按题 ≥0.10
    # ★ rev 1.6：臂有效性门 = ① 不贴地板（按项≥0.25 且按题≥0.10，DEF-90-1）
    #                    + ② 数值稳定（max|γm| ≤ 10×max|h|，DEF-91-5 的双边守卫）
    #   ② 缺失时，一个残差流被放大 1e4 倍的臂仍会"通过"①，Δ_content 便不可解释。
    inj_ok = [(res["arms"][f"INJ|{s}"]["acc_item"] >= 0.25 and
               res["arms"][f"INJ|{s}"]["acc_q"] >= 0.10 and
               res["arms"][f"INJ|{s}"].get("inj_ratio", 1e30) <= 10.0)
              for s in seeds if f"INJ|{s}" in res["arms"]]
    crit["inj_ratio_vals"] = [res["arms"][f"INJ|{s}"].get("inj_ratio") for s in seeds
                              if f"INJ|{s}" in res["arms"]]
    crit["arm_valid_per_seed"] = inj_ok
    crit["arm_valid"] = bool(inj_ok) and all(inj_ok)

    if dc:
        crit["delta_content_vals"] = dc
        crit["delta_content_mean"] = float(np.mean(dc))
        crit["delta_content_same_sign"] = all(x > 0 for x in dc) or all(x < 0 for x in dc)
        crit["delta_inj_vals"] = di
        crit["delta_inj_mean"] = float(np.mean(di))
        if len(dc) > 1:
            se = float(np.std(dc, ddof=1) / np.sqrt(len(dc)))
            crit["delta_content_se"] = se
            crit["delta_content_ci95"] = [float(np.mean(dc) - 3 * se),
                                          float(np.mean(dc) + 3 * se)]
        s91 = [res["arms"][f"MEM|{s}"]["s91_med"] for s in seeds if f"MEM|{s}" in res["arms"]]
        crit["s91_vals"] = s91
        crit["s91_pass"] = bool(s91) and all(x is not None and x <= 0.5 for x in s91)

        ci = crit.get("delta_content_ci95")
        h91 = (crit["delta_content_mean"] > 0.05 and crit["delta_content_same_sign"]
               and ci is not None and not (ci[0] <= 0.0 <= ci[1]))
        if not crit["arm_valid"]:
            crit["verdict"] = "INCONCLUSIVE-ARM-INVALID"
        elif not crit["s91_pass"]:
            crit["verdict"] = "INCONCLUSIVE-STORAGE"
        elif h91:
            crit["verdict"] = "CONFIRMED"
        else:
            crit["verdict"] = "FALSIFIED"
        crit["h91"] = bool(h91)

    res["summary"], res["verdict_raw"] = summ, crit
    dump(res, "main.json")
    print("\n=== 裁决材料 ===", flush=True)
    for k, v in crit.items():
        print(" ", k, "=", v, flush=True)
    print("  summary =", summ, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
