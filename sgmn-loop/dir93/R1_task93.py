#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-92 · R1 —— 取值算子（块均值 vs max-pool）下的**内容性收益**（KVR）

逐字派生自 `dir90/R1_task90.py`（DIR-90），后者逐字派生自 `R1_joint_read.py`（DIR-10）。
**只改三处**（round standard §2「只有测量层与独立变量可变」）：
  (a) `make_stack` 增加 `op` 参数（"mean" 默认 / "maxpool"），透传到 `SGMNStack`；
  (b) `run_arm` 增加 `op` 与 `facts_override`（后者用于 **MEM-RAND** 内容特异性对照）；
  (c) `main` 的臂表由 3 臂扩为 6 臂，并新增 Stage A 的 A92-2″/15″/16″/17″ 四条守卫。
`build_facts` / `build_chunks` / `write_phase` / `train_read_task` / `score_and_accuracy` /
`make_eval_items` / `storage_diag` / `base_sig` **全部原样保留，一字未改**。

卡片：revolve/DIR-92/DIRECTION_rev92_v10.md（rev 1.0，判据冻结）
      + DIRECTION_rev92_v11.md（rev 1.1，算子分支，已被 rev 1.2 取代）
      + DIRECTION_rev92_v12.md（rev 1.2，**max-pool**，本脚本实现的就是它）

★ 原始 DIR-90 的两条结构事实（决定了脚本形状，仍适用）：
  1. `fwd_hook` 顺序 = read → 注入 → **write** → cache。
  2. `write` 的 k 取 **`prev_hs`（上一块）**、v 取 **当前块** ⇒ 同一事实必须占**连续多个 chunk**。
"""
SRC_DERIVED_FROM = "dir90/R1_task90.py"
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
SPLIT_CLASSES = 16        # ★ DIR-93：类覆盖分割使用的值类数（= N_TRAIN_KEYS）

SEEDS = [20261028, 20261104, 7]

OUT = pathlib.Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
res = {"dir": "DIR-93", "round": "R1", "model": MODEL, "device": DEV,
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


def make_stack(gamma=GAMMA0, no_write=False, seed=20261028, op="mean", eta=None):
    tok, model = model_and_tok()
    d = model.config.hidden_size
    st = SGMNStack(model, MOUNT, d, d, R, ALPHA, LAM,
                   ETA_WRITE if eta is None else eta, gamma, DEV,
                   seed=seed, use_factorized=False, drop_sg=False, no_write=no_write,
                   op=op)          # ★ DIR-92：op 透传（默认 "mean" ⇒ 与 DIR-90 逐位一致）
    return st


# ---------------------------------------------------------------- η 的算子间标定
def calib_ksq(model, st, chunks, n=40):
    """★ DIR-92 rev 1.3：实测 E‖k‖²（写入路径**逐字**不变，只读 `last_k_sq` 诊断量）。

    `D-107` 的稳定边界是 η‖k‖² 的函数；换归约算子会改变 ‖k‖ 的量级
    （逐分量 max ≫ 均值），故**固定 η 比较算子不是受控比较**。本函数给出标定所需比值。

    ★★ 调用方**必须**用 `eta=0.0` 建栈（见 `DEF-92R-03`）：$k$ 只依赖 $h$ 与 $P_k$，
    **与 $W$ 无关** ⇒ `eta=0` 不改 $k$，却让 $W$ 恒为 $W_0$ ⇒ 不会发散。
    否则 maxpool 在未标定 η 下 40 块内即发散 ⇒ `‖k‖²=nan` ⇒ 标定自身失效（**本卡实测踩过**）。
    """
    ws = st.mems
    vals = []
    for ids in chunks[:n]:
        model(input_ids=ids.to(DEV))
        for m in ws.values():
            if m.last_k_sq is not None:
                vals.append(m.last_k_sq)
    return float(np.mean(vals)) if vals else float("nan")


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
    # ★★★ DIR-93 rev 1.0 唯一独立变量：值-类分配（类覆盖分割）
    #   旧（D-108/D-115/DIR-92）：`VALUE_POOL[i % 27]` + shuffle ⇒ 留出 16 条里 15 条的
    #     答案类**训练集从未出现** ⇒ acc_held 结构上界 = 1/16 = 0.0625 < 机会 0.25
    #   新：16 类 × 每类 2 条（fact j 与 fact j+16 同类）⇒ 训练覆盖全 16 类、留出也只用到这
    #      16 类 ⇒ acc_held 结构上界 = 16/16 = 1.0（首次高于机会）
    _C = VALUE_POOL[:SPLIT_CLASSES]
    vals = [_C[i % SPLIT_CLASSES] for i in range(N_FACTS)]
    assert len(set(vals)) == SPLIT_CLASSES, (len(set(vals)), SPLIT_CLASSES)
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


# --------------------------------------------------- MEM-RAND（内容特异性对照）
def build_rand_facts(facts, seed=13171):
    """★ DIR-92 §2.1″：**键不变、值错配**的事实集（32-循环置换，无不动点）。

    与 MEM 臂**写同样的键、同样的块布局、同样的写预算、同样的 eta**，唯一差别是
    「键对应的值」被轮换错开 ⇒ 若 MEM-max 的增益在 MEM-RAND 上消失，
    则增益来自**内容正确性**，而非"W 里有东西"这一通用效应。

    构造：值序列 `vals` 左移 1 位（`vals[(i+1) % 32]`）。循环置换长度 32 > 1 ⇒
    **不存在** i 使 vals_rand[i] == vals_true[i]（A92-17″ 逐条机械核）。
    """
    n = len(facts)
    vals = [f["val"] for f in facts]
    out = []
    for i, f in enumerate(facts):
        g = dict(f)                                   # 浅拷贝：键、句模板、ids 都不改
        g["val"] = vals[(i + 1) % n]                  # 值错配（≥1 位轮换）
        g["sent"] = f"The access code for item {g['key']} is {g['val']}."
        g["val_ids"] = None                           # 需重算，但 build_chunks 只用 sent
        out.append(g)
    return out


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
def _split_ceiling(vals):
    """给定值分配，返回 (训练覆盖类数, 留出类覆盖数, acc_held 结构上界)。纯算术。"""
    tr = list(range(N_TRAIN_KEYS)); he = list(range(N_TRAIN_KEYS, N_FACTS))
    trc = {vals[i] for i in tr}
    cov = sum(1 for i in he if vals[i] in trc)
    return len(trc), cov, cov / len(he)


def _old_vals():
    """复现**旧**分配（`VALUE_POOL[i % 27]` + shuffle），供 A93-S2 负向对照。"""
    rng = random.Random(9091)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    seen, n = set(), 0
    while n < N_FACTS:                      # 消耗与 build_facts 相同的 rng 序列
        k = "".join(rng.choice(alphabet) for _ in range(5))
        if k in seen:
            continue
        seen.add(k); n += 1
    v = [VALUE_POOL[i % len(VALUE_POOL)] for i in range(N_FACTS)]
    rng.shuffle(v)
    return v


def stage_a(tok, model, facts, chunks, owner):
    print("★ Stage A 开始", flush=True)

    # ================= DIR-93 类覆盖分割守卫（唯一独立变量，必须机械守） =================
    vals_new = [f["val"] for f in facts]
    trc, cov, ceil = _split_ceiling(vals_new)
    rec("A93-S1", cov == N_FACTS - N_TRAIN_KEYS and abs(ceil - 1.0) < 1e-12,
        f"新分割：训练覆盖 {trc} 类、留出类覆盖 {cov}/{N_FACTS-N_TRAIN_KEYS} "
        f"⇒ acc_held 结构上界 = {ceil:.4f}（须 1.0000）", value=[trc, cov, ceil])

    # ★ 负向对照：旧分配必须复现 1/16 < chance —— 若这条不成立，本卡动机不成立（§7 工作例）
    vals_old = _old_vals()
    trc_o, cov_o, ceil_o = _split_ceiling(vals_old)
    rec("A93-S2", cov_o == 1 and abs(ceil_o - 1 / 16) < 1e-12 and ceil_o < 1 / N_CAND,
        f"**负向对照**：旧分配（i%27+shuffle）留出类覆盖 = {cov_o}/{N_FACTS-N_TRAIN_KEYS} "
        f"⇒ 上界 = {ceil_o:.4f} < 机会 {1/N_CAND:.4f}（须恰好 1/16 且 < chance）",
        value=[trc_o, cov_o, ceil_o])

    # 双射：fact j 与 fact j+16 同类，且 16 类两两不同
    bij = all(vals_new[j] == vals_new[j + N_TRAIN_KEYS] for j in range(N_TRAIN_KEYS))
    rec("A93-S3", bij and trc == SPLIT_CLASSES,
        f"类映射双射：24 对全部同类={bij}；训练覆盖 {trc} 类（须 {SPLIT_CLASSES}）", value=bij)

    keys = [f["key"] for f in facts]
    tr_k = set(keys[:N_TRAIN_KEYS]); he_k = set(keys[N_TRAIN_KEYS:])
    rec("A93-S4", len(set(keys)) == N_FACTS and not (tr_k & he_k),
        f"键互不相同 {len(set(keys))}/{N_FACTS}；训练∩留出 = {len(tr_k & he_k)}（须 0）")

    from collections import Counter as _Ctr
    c = _Ctr(vals_new)
    rec("A93-S5", len(c) == SPLIT_CLASSES and all(v == 2 for v in c.values())
        and N_FACTS == 32 and N_CAND == 4,
        f"每类恰好 2 条：{sorted(set(c.values()))}（须 [2]）；类数 {len(c)}（须 {SPLIT_CLASSES}）；"
        f"N_FACTS={N_FACTS} N_CAND={N_CAND}", value=len(c))

    # A90-1 基座逐位不变（写前 / 写完 / 训练后 三处比）
    sig0, n_par = base_sig(model)
    rec("A90-1a", n_par > 0, f"基座参数 {n_par} 个，写前摘要 {sig0[:12]}…", value=n_par)

    # A90-7 候选单 token
    bad = [(w, tok(" " + w, add_special_tokens=False)["input_ids"])
           for w in VALUE_POOL
           if len(tok(" " + w, add_special_tokens=False)["input_ids"]) != 1]
    rec("A90-7", not bad, f"值池 {len(VALUE_POOL)} 个词全部为单 token（坏 {len(bad)} 个）", value=bad)

    # ★ A92-15″ 算子确实生效：同一批 z 上 amax != mean（三层两分支）
    _worst = 0.0
    for li in MOUNT:
        _g = torch.Generator(device="cpu"); _g.manual_seed(1000 + li)
        _z = torch.randn(CHUNK, 32, generator=_g)
        _worst = max(_worst, float((_z.amax(dim=0) - _z.mean(dim=0)).abs().max()))
    rec("A92-15pp", _worst > 1e-3,
        f"同一 z 上 |amax-mean|_inf = {_worst:.4f} > 1e-3 ⇒ 算子确实不同", value=_worst)

    # ★ A92-2″ 退化等价：64 个位置**逐位相同**时 amax == mean
    _d = torch.randn(1, 16, generator=torch.Generator(device="cpu").manual_seed(4))
    _z = _d.expand(CHUNK, -1).contiguous()
    _eq = float((_z.amax(dim=0) - _z.mean(dim=0)).abs().max())
    rec("A92-2pp", _eq <= 1e-6,
        f"位置逐位相同时 amax 与 mean 差 {_eq:.2e} ≤ 1e-6 ⇒ 退化等价成立", value=_eq)

    # ★ A92-17″ / A92-16″ MEM-RAND 构造正确（无不动点、键逐位不变、值多集相等）
    _rf = build_rand_facts(facts)
    _nofix = sum(1 for a, b in zip(facts, _rf) if a["val"] == b["val"])
    _keysame = all(a["key"] == b["key"] for a, b in zip(facts, _rf))
    _multiset = sorted(f["val"] for f in facts) == sorted(f["val"] for f in _rf)
    rec("A92-17pp", _nofix == 0,
        f"MEM-RAND 无不动点：错配后仍相等的项 {_nofix}/32（须 0）", value=_nofix)
    rec("A92-16pp", _keysame and _multiset,
        f"MEM-RAND 键逐位不变={_keysame}、值多集相等={_multiset}", value=[_keysame, _multiset])

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

    # ★★ A92-18″ 新算子的**稳定性**守卫（补 A90-10 的缺口：后者只跑了默认 op="mean"）
    #   两件事：① 标定后的 ETA_MAX 下 maxpool 必须全有限；② 不标定（用 mean 的 eta）会发散
    #   ⇒ 证明"η 标定是承重的"，不是装饰。
    _ks = {}
    for _op in ("mean", "maxpool"):
        _st = make_stack(gamma=0.0, no_write=False, seed=20261028, op=_op, eta=0.0)
        _ks[_op] = calib_ksq(model, _st, chunks, n=40)
        _st.remove()
    _ratio = _ks["mean"] / _ks["maxpool"] if _ks["maxpool"] else float("nan")
    _eta_max = ETA_WRITE * _ratio
    rec("A92-19pp", _ratio == _ratio and 0 < _ratio < 1,
        f"E‖k‖² mean={_ks['mean']:.2f} vs maxpool={_ks['maxpool']:.2f} ⇒ ratio={_ratio:.4f}"
        f"（maxpool 的 ‖k‖ 更大 ⇒ ratio<1 ⇒ η 须缩小）", value=[_ks["mean"], _ks["maxpool"], _ratio])

    _stO = make_stack(gamma=GAMMA0, op="maxpool", eta=_eta_max, seed=20261028)
    write_phase(model, _stO, chunks[:150])
    _finO = all(bool(torch.isfinite(W_of(m)).all()) for m in _stO.mems.values())
    _mxO = {li: float(W_of(m).abs().max()) for li, m in _stO.mems.items()}
    rec("A92-18pp", _finO,
        f"maxpool 在**标定后** η={_eta_max:.2e} 下跑 150 块 W 全有限（max|W|={_mxO}）", value=_mxO)
    _stO.remove()

    _stB = make_stack(gamma=GAMMA0, op="maxpool", eta=ETA_WRITE, seed=20261028)
    write_phase(model, _stB, chunks[:150])
    _finB = all(bool(torch.isfinite(W_of(m)).all()) for m in _stB.mems.values())
    rec("A92-20pp", not _finB,
        f"**负向对照**：maxpool 用**未标定** η={ETA_WRITE:.2e} 跑 150 块必然发散"
        f"（实测全有限={_finB}；须为 False）", value=_finB)
    _stB.remove()

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
            train_keys, eval_keys, seed, op="mean", write_chunks=None, diag_facts=None,
            eta=None):
    """★ DIR-92：`op` 传算子；`write_chunks`/`diag_facts` 只给 **MEM-RAND** 用。

    隔离原则（§2.1″）：**读侧训练、存储诊断、评测全部用真事实集 `facts`（与 MEM 臂完全相同）**，
    唯一差别是 `write_chunks` 里装的是**错配的内容** ⇒ 任何准确率差异只能归因于**写入内容**。
    """
    if write_chunks is None:
        write_chunks = chunks
    if diag_facts is None:
        diag_facts = facts
    st = make_stack(gamma=gamma, no_write=not do_write, seed=seed, op=op, eta=eta)
    t0 = time.time()
    wc = write_phase(model, st, write_chunks) if do_write else {"n/a": 0}
    viol, diverged, losses = None, False, []
    if trainable:
        losses, viol, diverged = train_read_task(model, st, tok, facts, train_keys,
                                                random.Random(1000 + seed % 100))
    s90 = None
    if do_write:
        # 诊断对**实际写入的内容**做（MEM-RAND 传 diag_facts=错配集）⇒ 检查"写进去了没有"
        s90, _ = storage_diag(model, st, diag_facts, owner, write_chunks)
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

    # ================= 正式轮：6 臂 × 种子（DIR-92 §2.1″） =================
    order = list(range(N_FACTS))
    random.Random(777).shuffle(order)
    train_keys, eval_keys = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]

    # MEM-RAND 的写入块：键不变、值错配（A92-16″/17″ 已在 Stage A 机械核过）
    rand_facts = build_rand_facts(facts)
    rand_chunks, rand_owner = build_chunks(tok, rand_facts)
    assert rand_owner == owner, "MEM-RAND 的块归属必须与 MEM 完全一致"
    dump({"rand_facts": [{k: v for k, v in f.items()
                          if k not in ("prompt_ids", "val_ids", "val_ids")}
                         for f in rand_facts]}, "rand_facts.json")

    # ---- ★ 标定（rev 1.3）：使两算子的 η‖k‖² 相同 ----
    ks = {}
    for op in ("mean", "maxpool"):
        st_c = make_stack(gamma=0.0, no_write=False, seed=seeds[0], op=op, eta=0.0)
        ks[op] = calib_ksq(model, st_c, chunks, n=40)
        st_c.remove()
    ratio = ks["mean"] / ks["maxpool"] if ks["maxpool"] else float("nan")
    ETA_MAX = ETA_WRITE * ratio
    print(f"★ η 标定：E‖k‖² mean={ks['mean']:.2f} maxpool={ks['maxpool']:.2f} "
          f"ratio={ratio:.4f} ⇒ ETA_MAX={ETA_MAX:.3e}", flush=True)
    dump({"ksq_mean": ks["mean"], "ksq_maxpool": ks["maxpool"],
          "ratio": ratio, "eta_write_ref": ETA_WRITE, "eta_maxpool": ETA_MAX}, "calib.json")

    # (name, gamma, do_write, trainable, op, write_chunks, diag_facts)
    ARMS = [
        ("BASE",     0.0,    False, False, "mean",    None,        None),
        ("INJ-mean", GAMMA0, False, True,  "mean",    None,        None),
        ("MEM-mean", GAMMA0, True,  True,  "mean",    None,        None),
        ("INJ-max",  GAMMA0, False, True,  "maxpool", None,        None),
        ("MEM-max",  GAMMA0, True,  True,  "maxpool", None,        None),
        ("MEMR-max", GAMMA0, True,  True,  "maxpool", rand_chunks, rand_facts),
    ]
    for seed in seeds:
        for name, gamma, dw, tr, op, wch, dfac in ARMS:
            r = run_arm(name, gamma, dw, tr, tok, model, facts, chunks, owner,
                        train_keys, eval_keys, seed, op=op,
                        write_chunks=wch, diag_facts=dfac,
                        eta=(ETA_MAX if op == "maxpool" else ETA_WRITE))
            res["arms"][f"{name}|{seed}"] = r
            dump(res, "main.json")

    # ================= 判据 =================
    def vals(arm, field="acc_q"):
        return [res["arms"][f"{arm}|{s}"][field] for s in seeds if f"{arm}|{s}" in res["arms"]]

    summ, crit = {}, {}
    for name, *_ in ARMS:
        v = vals(name)
        if v:
            summ[name] = {"mean": float(np.mean(v)), "vals": v,
                          "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    res["summary"] = summ

    def paired(a, b):
        return [res["arms"][f"{a}|{s}"]["acc_q"] - res["arms"][f"{b}|{s}"]["acc_q"]
                for s in seeds if f"{a}|{s}" in res["arms"] and f"{b}|{s}" in res["arms"]]

    # 主判据：maxpool 下的内容性收益
    dc = paired("MEM-max", "INJ-max")
    # 对照读数
    dc_mean = paired("MEM-mean", "INJ-mean")
    di = paired("INJ-max", "BASE")
    dspec = paired("MEM-max", "MEMR-max")

    from math import comb
    def binom_p(k, n):
        """单侧精确二项检验（H0: p=0.5），k = 正号数。"""
        return sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n) if n else None

    if dc:
        pos = sum(1 for x in dc if x > 0)
        crit["delta_content_max_vals"] = dc
        crit["delta_content_max_mean"] = float(np.mean(dc))
        crit["delta_content_max_same_sign_pos"] = all(x > 0 for x in dc)
        crit["delta_content_max_binom_p"] = binom_p(pos, len(dc))
        crit["delta_content_max_pass"] = bool(
            float(np.mean(dc)) >= 0.25 and all(x > 0 for x in dc)
            and (binom_p(pos, len(dc)) or 1.0) < 0.05)
        crit["delta_content_mean_vals"] = dc_mean
        crit["delta_content_mean_mean"] = float(np.mean(dc_mean)) if dc_mean else None
        crit["delta_inj_vals"] = di
        crit["delta_inj_mean"] = float(np.mean(di)) if di else None
        crit["delta_spec_vals"] = dspec
        crit["delta_spec_mean"] = float(np.mean(dspec)) if dspec else None
        # 第三态判定（F2/F3/F5）—— 零结果只有在对照臂有效时才有信息量
        floor = 0.25
        eff_arms = [res["arms"][f"{a}|{s}"]["acc_q"] for a in ("INJ-max", "MEM-max")
                    for s in seeds if f"{a}|{s}" in res["arms"]]
        crit["arm_invalid_both_at_floor"] = bool(eff_arms) and all(x <= floor for x in eff_arms)
        s90 = [res["arms"][f"MEM-max|{s}"]["s90_med"] for s in seeds
               if f"MEM-max|{s}" in res["arms"]]
        crit["s90_vals"] = s90
        crit["s90_pass"] = bool(s90) and all(x is not None and x <= 0.5 for x in s90)
        # 一页式判决
        if crit["delta_content_max_pass"]:
            verdict = "SUCCEEDED(H92)"
        elif crit["arm_invalid_both_at_floor"] or not crit["s90_pass"]:
            verdict = "INCONCLUSIVE-ARM-INVALID"
        elif abs(crit["delta_content_max_mean"]) < 0.25:
            verdict = "FALSIFIED(H92)"
        else:
            verdict = "INCONCLUSIVE-POWER"
        crit["verdict"] = verdict
    res["verdict_raw"] = crit
    dump(res, "main.json")

    print("\n=== 裁决材料 ===", flush=True)
    for k, v in crit.items():
        print(" ", k, "=", v, flush=True)
    print("  summary =", summ, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())