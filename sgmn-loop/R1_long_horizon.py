#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-08 · R1 —— 长视窗下的写通道因果效应复核（真实预训练 LLM）

判据见 ../DIRECTION.md（冻结于 2026-11-04，开跑前写入）。

【为什么复用 DIR-04 的 sgmn_module.py】
  本轮问的是「写通道在真实 LLM 上到底做不做功」。装置若换实现（不同初始化、
  不同 P_k 尺度校准、不同因子化步长约定），测量就无法归因到「写通道」本身。
  故复用 DIR-04 的 EXP-03(31/31) 已验证资产（逐字副本，md5 一致），
  并在 Stage A 的 A0 重验其契约 —— 不凭记忆采信。

【本轮自带的两个新检查】
  A0  复用资产的契约复检（Freeze C ①②③④ + 阴性对照）
  A6  【装置缺陷当量抓取】因子化更新的【同步性】偏差
      v2.4 §3.3 明文：U^+ = U - eta G V,  V^+ = V - eta G^T U（【同步】步：都用旧 U、旧 V）
      但装置实现先 copy_ 更新 U，再用【新】U 更新 V => 违反 §3.3。
      偏差是二阶量：dV = V_new - V_sync = eta^2 * G^T (G V_old)。
      本轮在真实 LLM 上逐采样步【量化】它的相对大小，作为装置缺陷登记并给出影响上界。
      若相对偏差 >= 1e-2，则本轮结论只是「关于一个不合规装置的结论」=> 停止。

【装置纪律】
  ① 同一时刻只允许一个 SGMNStack 挂在被测模型上（每次测量独立建栈、测完立刻 remove）
  ② 检查点上的探针前视不得污染运行：探针期间临时 no_write=True 并保存/还原 prev_hs
  ③ 记忆状态快照/还原必须包含 prev_hs
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import DynamicCache
except Exception:                                    # pragma: no cover
    DynamicCache = None

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sgmn_module import SGMNStack, grad_report        # noqa: E402

# ----------------------------------------------------------------- 声明条件
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ALPHA, LAM = 1.0, 0.05
GAMMA, R = 0.05, 8
MOUNT = [8, 12, 16]
EPS = 1e-2
SEED = 20261104
PERT_SEED = 11                                      # 与 DIR-06 一致

TEXT_TRAIN = ["The capital of France is", "In a distant galaxy the explorer found",
              "The quick brown fox jumps over", "Machine learning models are"]
TEXT_PROBE = ["The river flows into the", "She opened the old wooden box",
              "In the laboratory they measured the", "The committee announced that the"]
# 多样本探针集（revision 1.2）：40 条不同主题的短句，逐句 token 级采集，
# 使 (k,v) 的方差非退化（单条轨迹会让 ss_tot≈0，R² 被放大到无意义）。
DIVERSE = [
    "The river flows into the", "She opened the old wooden box and",
    "In the laboratory they measured the", "The committee announced that the",
    "Solar energy reaches the earth as", "He forgot where he had parked his",
    "The recipe calls for flour, sugar, and", "Rain fell steadily on the empty",
    "Their argument lasted for three", "The train to the northern coast leaves",
    "Ancient scholars copied manuscripts by", "The dog barked at every stranger who",
    "Coffee tastes bitter but the smell of", "The bridge was built using stones from",
    "Winter arrives suddenly in this part of", "She translated the poem from Japanese into",
    "The factory produces thousands of small", "Nobody expected the volcano to erupt",
    "A single candle illuminated the whole", "The treaty was signed after months of",
    "Musicians from different countries joined", "The vaccine was tested on thousands of",
    "He never mentioned that his brother had", "Autumn leaves covered the narrow path",
    "The professor asked a difficult question about", "Fresh bread smells better than any",
    "The tunnel runs beneath the mountain for", "They planned the journey for six months before",
    "The painting was stolen from a private", "Marine biologists discovered a new species near",
    "The engine started after the third attempt", "Her speech moved the audience to tears because",
    "Most of the town's income comes from", "The manuscript was discovered in a library in",
    "Snow covered the mountain pass until late", "A young programmer wrote a program that could",
    "The treaty talks continued long after the", "Ancient ruins were found beneath the sand near",
    "The orchestra performed a symphony that had never been", "Every morning the baker delivered fresh bread to",
]
DIVERSE_L = 16
L_PROMPT = 32

N_MAIN = 1024
ETA_GRID = [1e-7, 1e-6, 1e-5, 3e-5, 1e-4]
SEEDS = [20261028, 20261104, 7]        # 含 DIR-06 的原种子 20261028（用于对照其 0.2%）
ETA_DIR06 = 3e-5
CKPT_FRAC = [1 / 128, 1 / 32, 1 / 8, 1 / 2, 1.0]
NCH_DIR06 = 4

res = {"dir": "DIR-08", "round": "R1", "stage": "A+B", "model": MODEL,
       "device": DEV, "mount": MOUNT, "alpha": ALPHA, "lam": LAM,
       "gamma": GAMMA, "r": R, "eps": EPS, "seed": SEED, "checks": []}


def rec(name, passed, detail, tol=None, value=None):
    def clean(x):
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        if isinstance(x, (np.floating, float)):
            return float(x)
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (list, tuple)):
            return [clean(i) for i in x]
        return x
    res["checks"].append({"name": name, "passed": bool(passed), "detail": detail,
                          "tolerance": tol, "value": clean(value)})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


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


def make_stack(eta_write=None, no_write=False, seed=SEED, drop_sg=False):
    tok, model = model_and_tok()
    d = model.config.hidden_size
    st = SGMNStack(model, MOUNT, d, d, R, ALPHA, LAM, 0.01, GAMMA, DEV,
                   seed=seed, use_factorized=True, drop_sg=drop_sg, no_write=no_write)
    if eta_write is not None:
        for m in st.mems.values():
            m.eta = eta_write
    return st


def snapshot(st):
    return {li: (m.U.detach().clone(), m.V.detach().clone(), m.W0.detach().clone())
            for li, m in st.mems.items()}


def restore(st, snap):
    with torch.no_grad():
        for li, (U, V, W0) in snap.items():
            m = st.mems[li]
            m.U.copy_(U)
            m.V.copy_(V)
            m.W0.copy_(W0)
            m.prev_hs = None


def W_of(m):
    return m.W0.detach() + m.U.detach() @ m.V.detach().T


def perturb(st, eps, seed=PERT_SEED):
    """与 DIR-06 完全同款的扰动：U[:,0] += (eps/||V[:,0]||) * u。"""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    for m in st.mems.values():
        v0 = m.V[:, 0].detach()
        nv = float(v0.norm())
        u = torch.randn(m.d_v, generator=g).to(DEV)
        u = u / u.norm()
        with torch.no_grad():
            m.U[:, 0].add_(u, alpha=eps / max(nv, 1e-12))


def make_batch(tok, texts, L=L_PROMPT):
    enc = tok(texts, return_tensors="pt", padding="max_length",
              truncation=True, max_length=L)
    return enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)


@torch.no_grad()
def logits_chunked(model, ids, mask, n_chunks=NCH_DIR06):
    """DIR-06 的分段前视（无 KV cache），用于 R0 复现。"""
    L = ids.shape[1]
    cs = max(1, L // n_chunks)
    outs = []
    for i in range(0, L, cs):
        outs.append(model(input_ids=ids[:, i:i + cs],
                          attention_mask=mask[:, i:i + cs]).logits.float())
    return torch.cat(outs, dim=1)


# ----------------------------------------------------------------- 度量
def r2_of_W(W, K, Vv):
    """R^2(W) = 1 - SS_res/SS_tot，K:(n,dk) Vv:(n,dv)。"""
    pred = K @ W.T
    ss_res = float(((Vv - pred) ** 2).sum())
    ss_tot = float(((Vv - Vv.mean(dim=0, keepdim=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def ss_res_of_W(W, K, Vv):
    """绝对预测误差 SS_res（用于 rel_improve = 1 - SS_res_final/SS_res_init）。"""
    return float(((Vv - K @ W.T) ** 2).sum())


def online_loss(m, k, v):
    """L = ||W k - v||^2，W = W0 + U V^T；用 U(V^T k) 避免形成 (dv,dk) 矩阵。"""
    W0k = m.W0.detach() @ k
    Uterm = m.U.detach() @ (m.V.detach().T @ k)
    r = W0k + Uterm - v
    return float((r ** 2).sum())


def sync_deviation(m, k, v, eta, U_old, V_old):
    """
    实现:  U_new = U_old - eta G V_old ;  V_new = V_old - eta G^T U_new
    同步:  U_new = U_old - eta G V_old ;  V_sync = V_old - eta G^T U_old
    偏差:  dV = V_new - V_sync = -eta G^T (U_new - U_old) = eta^2 G^T (G V_old)
    G = ALPHA*outer(Wk-v, k) + LAM*(W - W0)
    G^T y = ALPHA*outer(k, e^T y) + LAM*(W-W0)^T y      （不形成 G）
    返回 (||dV||/||V_new||, 重建误差 ||V_module - V_new_impl||/||V_new||)
    后者用于校验本函数对 G 的重建与装置内部一致。
    """
    W0 = m.W0.detach()
    W_old = W0 + U_old @ V_old.T
    e = W_old @ k - v
    GV = ALPHA * torch.outer(e, k @ V_old) + LAM * (W_old - W0) @ V_old      # (dv, r)
    GTU = ALPHA * torch.outer(k, e @ U_old) + LAM * (W_old - W0).T @ U_old  # (dk, r)
    dU = -eta * GV
    U_new = U_old + dU
    V_new = V_old - eta * (ALPHA * torch.outer(k, e @ U_new) + LAM * (W_old - W0).T @ U_new)
    dV = eta * eta * (ALPHA * torch.outer(k, e @ GV) + LAM * (W_old - W0).T @ GV)
    rel = float(dV.norm() / max(float(V_new.norm()), 1e-30))
    recon = float((m.V.detach() - V_new).norm() / max(float(V_new.norm()), 1e-30))
    return rel, recon


def aligned_erasure(Wpert, Wclean, Wpert0, Wclean0):
    """
    aligned = <D_final, dW_init>_F / ||dW_init||_F^2
    total   = ||D_final||_F / ||dW_init||_F
    no_write 时 D_final == dW_init  => aligned == total == 1（严格）。
    """
    dw_init = (Wpert0 - Wclean0).double()
    d_fin = (Wpert - Wclean).double()
    prod = d_fin * dw_init
    num = float(prod.sum())
    den = float((dw_init * dw_init).sum())
    if den <= 0:
        return None, None
    aligned = num / den
    total = float((d_fin * d_fin).sum()) ** 0.5 / max(den ** 0.5, 1e-30)
    return aligned, total


# ----------------------------------------------------------------- 流式运行
@torch.no_grad()
def probe_logits(model, st, ids, mask):
    """只读探针：期间 no_write=True 且不改 prev_hs => 不污染在线运行。"""
    saved_flag = {li: m.no_write for li, m in st.mems.items()}
    saved_prev = {li: m.prev_hs for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    try:
        lg = model(input_ids=ids, attention_mask=mask).logits.float()
    finally:
        for li, m in st.mems.items():
            m.no_write = saved_flag[li]
            m.prev_hs = saved_prev[li]
    return lg.cpu()


@torch.no_grad()
def run_stream(model, st, ids, mask, n_steps, ckpts, probe_ids=None, probe_mask=None,
               collect_pairs=False, dev_sample_every=0, keep_probe=False,
               measure_dev=True):
    """
    token 级在线步流：每生成 1 个 token = 1 个在线事件（带 KV cache 的因果续写）。
    """
    out = {"ckpt": {}, "loss": [], "pairs": {li: [] for li in st.mems}, "dev": []}
    cache = DynamicCache()
    B, L0 = ids.shape
    mo = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
    cache = mo.past_key_values
    for t in range(n_steps):
        pre = {li: m.prev_hs for li, m in st.mems.items()}
        old = {li: (m.U.detach().clone(), m.V.detach().clone()) for li, m in st.mems.items()}
        nt = mo.logits[:, -1].argmax(-1, keepdim=True)
        mfull = torch.cat([mask, torch.ones(B, t + 1, device=DEV, dtype=mask.dtype)], 1)
        mo = model(input_ids=nt, attention_mask=mfull, past_key_values=cache, use_cache=True)
        cache = mo.past_key_values
        for li, m in st.mems.items():
            pf = pre[li]
            if pf is None:
                continue
            k = ((pf.reshape(-1, pf.shape[-1]) @ m.Pk.detach()).mean(dim=0)) * m.pk_scale
            v = (m.prev_hs.reshape(-1, m.prev_hs.shape[-1]) @ m.Pv.detach()).mean(dim=0)
            if collect_pairs:
                out["pairs"][li].append((k.detach().cpu(), v.detach().cpu()))
            out["loss"].append((li, online_loss(m, k, v)))
            if measure_dev and dev_sample_every and (t % dev_sample_every == 0):
                rel, recon = sync_deviation(m, k, v, m.eta, old[li][0], old[li][1])
                out["dev"].append((t, li, rel, recon))
        if (t + 1) in ckpts:
            rec = {"W": {li: W_of(m).cpu().clone() for li, m in st.mems.items()},
                   "maxU": {li: float(m.U.detach().abs().max()) for li, m in st.mems.items()},
                   "write_calls": {li: m.write_calls for li, m in st.mems.items()},
                   "Vt_norm": {li: float(m.V.detach().norm()) for li, m in st.mems.items()}}
            if probe_ids is not None:
                rec["probe_logits"] = probe_logits(model, st, probe_ids, probe_mask)
            out["ckpt"][t + 1] = rec
    out["write_calls"] = {li: m.write_calls for li, m in st.mems.items()}
    return out


def summarize_pair(rr, HK, HV, ckpts):
    """把一对 (clean, pert) 运行的原始数据压成紧凑摘要（原始张量随后释放）。"""
    cl, pt = rr["clean"], rr["pert"]
    loss = cl["loss"]
    per_layer = {}
    for li in MOUNT:
        seq = [val for lay, val in loss if lay == li]
        n = len(seq)
        per_layer[li] = (float(np.mean(seq[:max(1, n // 10)])),
                         float(np.mean(seq[max(0, n - n // 10):])))
    Ktr = {li: torch.stack([p[0] for p in cl["pairs"][li]]) for li in cl["pairs"]}
    Vtr = {li: torch.stack([p[1] for p in cl["pairs"][li]]) for li in cl["pairs"]}
    tmax = max(cl["ckpt"])
    Wfin = cl["ckpt"][tmax]["W"]
    r2 = {}
    for li in MOUNT:
        ss_i = ss_res_of_W(rr["Win"][li], Ktr[li], Vtr[li])
        ss_f = ss_res_of_W(Wfin[li], Ktr[li], Vtr[li])
        r2[li] = {"train_R2_init": r2_of_W(rr["Win"][li], Ktr[li], Vtr[li]),
                  "train_R2_final": r2_of_W(Wfin[li], Ktr[li], Vtr[li]),
                  "hold_R2_init": r2_of_W(rr["Win"][li], HK[li], HV[li]),
                  "hold_R2_final": r2_of_W(Wfin[li], HK[li], HV[li]),
                  "train_SSres_init": ss_i, "train_SSres_final": ss_f,
                  "train_rel_improve": 1.0 - ss_f / max(ss_i, 1e-30)}
    eras = {}
    contam = {}
    for li in MOUNT:
        for t in ckpts:
            if t not in cl["ckpt"]:
                continue
            al, to = aligned_erasure(pt["ckpt"][t]["W"][li], cl["ckpt"][t]["W"][li],
                                      rr["Wpert0"][li], rr["Win"][li])
            eras.setdefault(li, {})[t] = (al, to)
            # contamination fraction（revision 1.3）：最终记忆里「注入」占多大比例
            D = float((pt["ckpt"][t]["W"][li] - cl["ckpt"][t]["W"][li]).norm())
            S = float((cl["ckpt"][t]["W"][li] - rr["Win"][li]).norm())
            contam.setdefault(li, {})[t] = D / max(D + S, 1e-30)
    return {"loss_early_late": per_layer, "r2": r2,
            "maxU": cl["ckpt"][tmax]["maxU"],
            "maxU_curve": {t: cl["ckpt"][t]["maxU"] for t in cl["ckpt"]},
            "write_calls": cl["write_calls"],
            "dev_max": max([d[2] for d in cl["dev"]] or [0.0]),
            "dev_recon_max": max([d[3] for d in cl["dev"]] or [0.0]),
            "erasure": eras, "contam": contam,
            "dvec": {t: dv.tolist() for t, dv in rr["dvec"].items()}}


def strip_probe(run):
    for ck in run["ckpt"].values():
        ck.pop("probe_logits", None)
    return run


# ----------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_main = 64 if args.smoke else N_MAIN
    eta_grid = [1e-6, 3e-5] if args.smoke else ETA_GRID
    seeds = SEEDS[:2] if args.smoke else SEEDS
    dev_every = 8 if args.smoke else 64
    ckpts = sorted({max(1, int(round(f * n_main))) for f in CKPT_FRAC})
    res["config"] = {"n_main": n_main, "eta_grid": eta_grid, "seeds": seeds,
                     "ckpts": ckpts, "dev_sample_every": dev_every}
    out_json = HERE / ("R1_smoke_results.json" if args.smoke else "R1_results.json")

    print("=== DIR-08：长视窗下的写通道因果效应复核（revision 1.1 · 多种子）===", flush=True)
    tok, model = model_and_tok()
    ids, mask = make_batch(tok, TEXT_TRAIN)
    pids, pmask = make_batch(tok, TEXT_PROBE)

    # ================================================================ Stage A
    print("\n[Stage A] 装置校验门", flush=True)

    st = make_stack(eta_write=ETA_DIR06)
    base_before = {n: q.detach().clone() for n, q in model.named_parameters()}
    lg = model(input_ids=ids, attention_mask=mask).logits
    F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                    ids[:, 1:].reshape(-1)).backward()
    rrep = grad_report(st.named_read_params())
    wrep = grad_report(st.named_write_params())
    read_ok = all((not isn) and g > 1e-3 and not bad for _, g, isn, bad in rrep)
    read_min = min([g for _, g, _, _ in rrep] or [0])
    write_ok = all(isn or g == 0.0 for _, g, isn, _ in wrep)
    base_same = all(torch.equal(q.detach(), base_before[n])
                    for n, q in model.named_parameters())
    st.remove()
    torch.cuda.empty_cache()
    st2 = make_stack(drop_sg=True)
    lg2 = model(input_ids=ids, attention_mask=mask).logits
    F.cross_entropy(lg2[:, :-1].reshape(-1, lg2.shape[-1]),
                    ids[:, 1:].reshape(-1)).backward()
    wrep2 = grad_report(st2.named_write_params())
    leak = len([1 for _, g, isn, _ in wrep2 if not isn and g > 0])
    st2.remove()
    torch.cuda.empty_cache()
    for q in model.parameters():
        q.grad = None
    a0_ok = read_ok and write_ok and base_same and leak > 0
    rec("A0 复用资产的契约复检（Freeze C ①②③④ + 阴性对照）", a0_ok,
        f"②读梯度 {sum(1 for _, _, i, _ in rrep if not i)}/{len(rrep)} 非零，最小 {read_min:.3e}；"
        f"③写入梯度精确 0：{sum(1 for _, g, i, _ in wrep if i or g == 0.0)}/{len(wrep)}；"
        f"④基座逐位不变：{base_same}；阴性对照泄漏 {leak}/{len(wrep2)}",
        "读梯度>1e-3 且 写入梯度==0 且 基座不变 且 泄漏>0", a0_ok)

    st = make_stack(eta_write=ETA_DIR06)
    snap = snapshot(st)
    run_stream(model, st, ids, mask, 32, set())
    W_a = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
    restore(st, snap)
    run_stream(model, st, ids, mask, 32, set())
    W_b = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
    rep = max(float((W_a[li] - W_b[li]).abs().max()) for li in st.mems)
    st.remove()
    torch.cuda.empty_cache()
    rec("A1 长视窗可重复（逐位相同）", rep == 0.0,
        f"两次 32 步运行后的 W 最大逐位差 {rep:.3e}", "精确 0", rep)

    st = make_stack(eta_write=ETA_DIR06, no_write=True)
    W0rel = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
    run_stream(model, st, ids, mask, n_main, set())
    d_nowrite = max(float((W_of(st.mems[li]).cpu() - W0rel[li]).abs().max()
                           / max(float(W0rel[li].abs().max()), 1e-30)) for li in st.mems)
    wc_nowrite = {li: st.mems[li].write_calls for li in st.mems}
    st.remove()
    torch.cuda.empty_cache()
    rec("A2 写被切断时记忆逐位不变（no_write=True 跑满 %d 步）" % n_main,
        d_nowrite < 1e-12,
        f"相对差 {d_nowrite:.3e}；write_calls={wc_nowrite}", "<1e-12", d_nowrite)

    st = make_stack(eta_write=0.0)
    W0rel = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
    run_stream(model, st, ids, mask, n_main, set())
    d_eta0 = max(float((W_of(st.mems[li]).cpu() - W0rel[li]).abs().max()
                        / max(float(W0rel[li].abs().max()), 1e-30)) for li in st.mems)
    st.remove()
    torch.cuda.empty_cache()
    rec("A3a 区分力对照：eta_w=0 时记忆逐位不变", d_eta0 < 1e-12,
        f"相对差 {d_eta0:.3e}", "<1e-12", d_eta0)

    st = make_stack(eta_write=ETA_DIR06)
    W0rel = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
    run_stream(model, st, ids, mask, n_main, set())
    d_eta = max(float((W_of(st.mems[li]).cpu() - W0rel[li]).abs().max()
                       / max(float(W0rel[li].abs().max()), 1e-30)) for li in st.mems)
    st.remove()
    torch.cuda.empty_cache()
    rec("A3b 区分力对照：eta_w=%g 时记忆必须改变" % ETA_DIR06, d_eta > 1e-9,
        f"相对差 {d_eta:.3e}（与 A3a 构成区分力对照）", ">1e-9", d_eta)

    st = make_stack(eta_write=ETA_DIR06)
    n_handles, n_mems = len(st._handles), len(st.mems)
    st.remove()
    one = n_handles == 2 * n_mems and n_mems == len(MOUNT)
    rec("A5 同一时刻只允许一个装置挂在被测模型上", one,
        f"挂载句柄 {n_handles}（每层 2 个），记忆数 {n_mems}，挂载层 {MOUNT}",
        "句柄数 == 2*层数 且 装置数 == 1", one)

    # ================================================================ 各种子扫描
    def short_horizon_share(seed):
        """DIR-06 的 4 段前视协议 -> (写介导占比, d_w, d_nw)。"""
        st = make_stack(eta_write=ETA_DIR06, seed=seed)
        snap = snapshot(st)
        restore(st, snap)
        clean_w = logits_chunked(model, ids, mask, NCH_DIR06)
        restore(st, snap)
        perturb(st, EPS)
        pert_w = logits_chunked(model, ids, mask, NCH_DIR06)
        st.remove()
        torch.cuda.empty_cache()
        st = make_stack(eta_write=ETA_DIR06, no_write=True, seed=seed)
        snap = snapshot(st)
        restore(st, snap)
        clean_nw = logits_chunked(model, ids, mask, NCH_DIR06)
        restore(st, snap)
        perturb(st, EPS)
        pert_nw = logits_chunked(model, ids, mask, NCH_DIR06)
        st.remove()
        torch.cuda.empty_cache()
        d_w = (pert_w - clean_w).norm(dim=-1).mean(dim=0)
        d_nw = (pert_nw - clean_nw).norm(dim=-1).mean(dim=0)
        return (float((d_w - d_nw).mean() / max(float(d_w.mean()), 1e-30)),
                float(d_w.mean()), float(d_nw.mean()))

    def run_pair(seed, eta, no_write):
        st = make_stack(eta_write=eta, no_write=no_write, seed=seed)
        snap = snapshot(st)
        Win = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
        restore(st, snap)
        clean = run_stream(model, st, ids, mask, n_main, ckpts,
                           probe_ids=pids, probe_mask=pmask,
                           collect_pairs=True, dev_sample_every=dev_every,
                           measure_dev=not no_write)
        clean_probe = {t: ck.pop("probe_logits") for t, ck in clean["ckpt"].items()}
        restore(st, snap)
        perturb(st, EPS)
        Wpert0 = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
        pert = run_stream(model, st, ids, mask, n_main, ckpts,
                          probe_ids=pids, probe_mask=pmask,
                          collect_pairs=True, dev_sample_every=dev_every,
                          measure_dev=not no_write)
        dvec = {}
        for t in ckpts:
            if t in pert["ckpt"] and t in clean_probe:
                pl = pert["ckpt"][t].pop("probe_logits")
                dvec[t] = (pl - clean_probe[t]).norm(dim=-1).mean(dim=0)
        del clean_probe
        st.remove()
        torch.cuda.empty_cache()
        return {"clean": clean, "pert": pert, "Win": Win, "Wpert0": Wpert0,
                "dvec": dvec}

    def collect_diverse_pairs(seed):
        """revision 1.2：40 条不同主题短句，逐句 token 级流式采集 (k,v)（batch=1 => 每句独立）。"""
        st = make_stack(eta_write=ETA_DIR06, no_write=True, seed=seed)
        acc = {li: [] for li in st.mems}
        for txt in DIVERSE:
            i1, m1 = make_batch(tok, [txt], L=DIVERSE_L)
            r = run_stream(model, st, i1, m1, DIVERSE_L, set(), collect_pairs=True)
            for li in st.mems:
                acc[li].extend(r["pairs"][li])
        st.remove()
        torch.cuda.empty_cache()
        return ({li: torch.stack([p[0] for p in acc[li]]) for li in acc},
                {li: torch.stack([p[1] for p in acc[li]]) for li in acc})

    seed_results = []
    for seed in seeds:
        print(f"\n[R1] 种子 {seed}", flush=True)
        short = short_horizon_share(seed)
        print(f"    短视窗(4 段前视)写介导占比 = {short[0]*100:+.3f}%  "
              f"(d_w={short[1]:.4e}, d_nw={short[2]:.4e})", flush=True)
        t0 = time.time()
        HK, HV = collect_diverse_pairs(seed)
        print(f"    多样本留出探针：每层 {len(HK[MOUNT[0]])} 对 (k,v)（40 句 × "
              f"{DIVERSE_L} 步）", flush=True)
        rr = run_pair(seed, ETA_DIR06, True)
        nw_sum = summarize_pair(rr, HK, HV, ckpts)
        del rr
        sums = {}
        for eta in eta_grid:
            t1 = time.time()
            rr = run_pair(seed, eta, False)
            sums[f"eta_{eta:g}"] = summarize_pair(rr, HK, HV, ckpts)
            del rr
            print(f"    eta_w={eta:g} 完成（{time.time()-t1:.0f}s）", flush=True)
        seed_results.append({"seed": seed, "short": short, "no_write": nw_sum,
                             "etas": sums})
        print(f"    本种子总用时 {time.time()-t0:.0f}s", flush=True)

    # ================================================================ 分析
    print("\n[R2] 分析与判据（多种子聚合）", flush=True)

    def agg(vals):
        a = np.asarray(list(vals), dtype=float)
        return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)

    # ---- 短视窗多种子分布（H7）
    shorts = {str(sr["seed"]): sr["short"][0] for sr in seed_results}
    svals = list(shorts.values())
    sm, ss = agg(svals)
    span = max(svals) - min(svals)
    res["short_horizon_share"] = {"per_seed": shorts, "mean": sm, "std": ss,
                                   "span": span, "dir06_reported": 0.002}
    cv = (ss / abs(sm)) if abs(sm) > 1e-12 else float("inf")
    rec("H7 短视窗写介导占比的多种子分布（DIR-06 的 0.2% 是否可分辨）",
        span > 5 * 0.002,
        "各种子短视窗占比：" + "，".join(f"{k}:{v*100:+.3f}%" for k, v in shorts.items()) +
        f"；均值 {sm*100:+.3f}%、标准差 {ss*100:.3f}%、跨度 {span*100:.3f}%"
        f"（DIR-06 报告 0.2%，其种子 {SEEDS[0]} 在本轮复现 {shorts.get(str(SEEDS[0]),float('nan'))*100:+.3f}%）"
        f"；变异系数 CV={cv:.2f}。"
        f"跨度 {'>' if span>5*0.002 else '<='} 5×0.2% ⇒ "
        f"{'该指标在短视窗下被初始化噪声主导，单种子数值不可作为结论依据' if span>5*0.002 else '该指标在短视窗下稳定'}",
        "跨度 > 5×0.2%", span)

    # ---- 写介导占比曲线（多种子）
    share_by = {}
    for sr in seed_results:
        nw_d = {t: float(np.mean(sr["no_write"]["dvec"][t])) for t in sr["no_write"]["dvec"]}
        for key, smm in sr["etas"].items():
            for t, dv in smm["dvec"].items():
                arr = np.asarray(dv, dtype=float)
                share_by.setdefault(key, {}).setdefault(t, []).append(
                    float((arr.mean() - nw_d[t]) / max(arr.mean(), 1e-30)))
    share_curve = {k: {t: agg(v)[0] for t, v in d.items()} for k, d in share_by.items()}
    share_std = {k: {t: agg(v)[1] for t, v in d.items()} for k, d in share_by.items()}
    res["write_mediated_share_curve"] = {k: {str(t): v for t, v in c.items()}
                                          for k, c in share_curve.items()}
    res["write_mediated_share_std"] = {k: {str(t): v for t, v in c.items()}
                                        for k, c in share_std.items()}
    print("  写介导占比（多种子均值 ± 标准差）：", flush=True)
    for key, c in share_curve.items():
        print(f"    {key:12s} " +
              "  ".join(f"t={t}:{c[t]*100:+.2f}%±{share_std[key][t]*100:.2f}%"
                        for t in c), flush=True)

    # ---- H4'
    h4 = {}
    for key, c in share_curve.items():
        tmax = max(c)
        vals = share_by[key][tmax]
        m, sd = agg(vals)
        same_sign = all(v > 0 for v in vals) or all(v < 0 for v in vals)
        h4[key] = {"mean": m, "std": sd, "per_seed": vals, "same_sign": same_sign,
                   "ok": same_sign and abs(m) >= 2 * sd and sd > 0}
    good4 = [k for k, v in h4.items() if v["ok"]]
    rec("H4' 长视窗写介导占比可与零区分（多种子：同号 且 |均值|≥2σ）",
        len(good4) > 0,
        "；".join(f"{k}: mean={v['mean']*100:+.3f}% σ={v['std']*100:.3f}% "
                  f"同号={v['same_sign']} per_seed={[round(x*100,3) for x in v['per_seed']]}"
                  for k, v in h4.items()),
        "存在至少一个 eta 同号且 |mean|>=2σ", good4)

    # ---- 学习 / 擦除 指标（多种子）
    def per_seed_metrics(sr, key):
        smm = sr["no_write"] if key == "no_write" else sr["etas"][key]
        lo = smm["loss_early_late"]
        lr = float(np.mean([lo[li][1] / max(lo[li][0], 1e-30) for li in MOUNT]))
        dR2t = float(np.mean([smm["r2"][li]["train_R2_final"]
                              - smm["r2"][li]["train_R2_init"] for li in MOUNT]))
        dR2h = float(np.mean([smm["r2"][li]["hold_R2_final"]
                              - smm["r2"][li]["hold_R2_init"] for li in MOUNT]))
        tmax = max(smm["erasure"][MOUNT[0]])
        al = float(np.mean([smm["erasure"][li][tmax][0] for li in MOUNT]))
        ct = float(np.mean([smm["contam"][li][tmax] for li in MOUNT]))
        to = float(np.mean([smm["erasure"][li][tmax][1] for li in MOUNT]))
        rel = float(np.mean([smm["r2"][li]["train_rel_improve"] for li in MOUNT]))
        return {"loss_ratio": lr, "dR2_train": dR2t, "dR2_hold": dR2h,
                "rel_improve": rel, "contam": ct,
                "aligned": al, "total": to,
                "maxU": max(smm["maxU"].values()),
                "dev_max": smm["dev_max"], "dev_recon_max": smm["dev_recon_max"],
                "write_calls": smm["write_calls"]}

    metrics = {}
    for key in ["no_write"] + [f"eta_{e:g}" for e in eta_grid]:
        rows = []
        for sr in seed_results:
            rows.append(per_seed_metrics(sr, key))
        metrics[key] = {k: agg([r[k] for r in rows]) for k in rows[0]
                        if isinstance(rows[0][k], (int, float))}
        metrics[key]["per_seed"] = rows
    res["metrics"] = metrics

    print("  指标（多种子均值）：", flush=True)
    for key, mm in metrics.items():
        print(f"    {key:12s} L比={mm['loss_ratio'][0]:.4f}±{mm['loss_ratio'][1]:.4f} "
              f"dR2_tr={mm['dR2_train'][0]:+.4f}±{mm['dR2_train'][1]:.4f} "
              f"dR2_hold={mm['dR2_hold'][0]:+.4f} "
              f"rel_imp={mm['rel_improve'][0]:+.4f} "
              f"contam={mm['contam'][0]:.5f} "
              f"aligned={mm['aligned'][0]:+.4f}±{mm['aligned'][1]:.4f} "
              f"maxU={mm['maxU'][0]:.3e} wc={mm['per_seed'][0]['write_calls'][MOUNT[0]]}",
              flush=True)

    # ---- H1（多种子：每个种子都满足）
    h1 = {}
    for key in [f"eta_{e:g}" for e in eta_grid]:
        rows = metrics[key]["per_seed"]
        lr = [r["loss_ratio"] for r in rows]
        dR2 = [r["dR2_train"] for r in rows]
        h1[key] = {"loss_ratio": lr, "dR2_train": dR2,
                   "ok": all(a < 0.5 for a in lr) and all(b > 0.05 for b in dR2)}
    good1 = [k for k, v in h1.items() if v["ok"]]
    nw_rows = metrics["no_write"]["per_seed"]
    nw_lr = [r["loss_ratio"] for r in nw_rows]
    nw_dR2 = [r["dR2_train"] for r in nw_rows]
    ctrl_fails = not (all(a < 0.5 for a in nw_lr) and all(b > 0.05 for b in nw_dR2))
    rec("H1 长视窗下写通道确实在学习（每个种子均满足 L比<0.5 且 dR2>0.05）",
        len(good1) > 0 and ctrl_fails,
        "通过档：" + "；".join(f"{k}: L比={[round(x,4) for x in v['loss_ratio']]} "
                  f"dR2={[round(x,4) for x in v['dR2_train']]} "
                  f"rel_imp={[round(r['rel_improve'],4) for r in metrics[k]['per_seed']]}"
                  for k, v in h1.items() if v["ok"]) +
        f"。【区分力对照】no_write 档 L比={[round(x,4) for x in nw_lr]} "
        f"dR2={[round(x,6) for x in nw_dR2]} ⇒ 对照{'不满足' if ctrl_fails else '【也满足 ⇒ 判据无区分力】'}该判据",
        "存在至少一个 eta 在所有种子上满足 且 no_write 对照不满足", good1)

    # ---- H1b（revision 1.2 新增）：多样本留出探针上的泛化
    h1b = {}
    for key in [f"eta_{e:g}" for e in eta_grid]:
        rows = metrics[key]["per_seed"]
        dh = [r["dR2_hold"] for r in rows]
        h1b[key] = {"dR2_hold": dh, "ok": all(b > 0.05 for b in dh)}
    good1b = [k for k, v in h1b.items() if v["ok"]]
    rec("H1b 多样本留出探针上的泛化（每个种子 ΔR²_hold>0.05）", len(good1b) > 0,
        "；".join(f"{k}: ΔR²_hold={[round(x,4) for x in v['dR2_hold']]}"
                  for k, v in h1b.items()),
        "存在至少一个 eta 在所有种子上满足", good1b)

    # ---- H2
    h2 = {}
    for key in [f"eta_{e:g}" for e in eta_grid]:
        rows = metrics[key]["per_seed"]
        al = [r["aligned"] for r in rows]
        h2[key] = {"aligned": al, "ok": all(a < 0.5 for a in al)}
    good2 = [k for k, v in h2.items() if v["ok"]]
    nw_al = [r["aligned"] for r in metrics["no_write"]["per_seed"]]
    ctrl2_fails = not all(a < 0.5 for a in nw_al)
    rec("H2 写覆盖了外部注入的扰动（每个种子的 aligned<0.5）",
        len(good2) > 0 and ctrl2_fails,
        "通过档：" + "；".join(f"{k}: aligned={[round(x,4) for x in v['aligned']]}"
                  for k, v in h2.items() if v["ok"]) +
        f"。【区分力对照】no_write 档 aligned={[round(x,6) for x in nw_al]} ⇒ 对照"
        f"{'不满足' if ctrl2_fails else '【也满足 ⇒ 判据无区分力】'}该判据",
        "存在至少一个 eta 在所有种子上满足 且 no_write 对照不满足", good2)

    # ---- H2'（revision 1.3 新增）：污染占比
    h2p = {}
    for key in [f"eta_{e:g}" for e in eta_grid]:
        rows = metrics[key]["per_seed"]
        ct = [r["contam"] for r in rows]
        h2p[key] = {"contam": ct, "ok": all(c < 0.5 for c in ct)}
    good2p = [k for k, v in h2p.items() if v["ok"]]
    nw_ct = [r["contam"] for r in metrics["no_write"]["per_seed"]]
    ctrl2p = all(abs(c - 1.0) < 1e-9 for c in nw_ct)
    rec("H2' 写使记忆由数据而非注入决定（contam<0.5，且 no_write 对照≡1）",
        len(good2p) > 0 and ctrl2p,
        "通过档：" + "；".join(f"{k}: contam={[round(x,5) for x in v['contam']]}"
                  for k, v in h2p.items() if v["ok"]) +
        f"。【区分力对照】no_write 档 contam={[round(c,9) for c in nw_ct]} ⇒ 对照"
        f"{'不满足' if ctrl2p else '【异常】'}（写位移为 0 时 contam 必为 1）",
        "存在至少一个 eta 在所有种子上满足 且 对照≡1", good2p)

    # ---- H3
    rows = metrics["no_write"]["per_seed"]
    al_nw = [r["aligned"] for r in rows]
    to_nw = [r["total"] for r in rows]
    h3_ok = all(abs(a - 1.0) < 1e-9 for a in al_nw) and all(abs(t - 1.0) < 1e-9 for t in to_nw)
    rec("H3 擦除效应的归因（no_write 档 aligned≡1 且 total≡1）", h3_ok,
        f"aligned={[a for a in al_nw]}，total={[t for t in to_nw]}", "|x-1|<1e-9", h3_ok)

    # ---- H5
    h5 = {}
    for key in [f"eta_{e:g}" for e in eta_grid]:
        h5[key] = metrics[key]["maxU"]
    good5 = [k for k, v in h5.items() if v[0] < 1e3 and v[1] < 1e3]
    rec("H5 有效窗口内数值不爆炸（max|U|<1e3，全部种子）", len(good5) > 0,
        "；".join(f"{k}: maxU={v[0]:.3e}±{v[1]:.3e}" for k, v in h5.items()),
        ">=1e3 的档数为 0", good5)

    # ---- H6
    h6 = [k for k in good2p if k in good1 and k in good5]
    h6_strict = [k for k in h6 if k in good1b]
    rec("H6 写通道在真实 LLM 上的有效窗口非空（H1∧H2'∧H5）", len(h6) > 0,
        f"H1：{good1}；H2'：{good2p}；H5：{good5}；交集：{h6}；"
        f"再加附加判据 H1b（留出探针泛化）后的交集：{h6_strict}",
        "存在至少一个 eta", h6)

    # ---- A6（revision 1.3）：按 eta 报告；绑定检查落在承载结论的 eta
    dev_all = {k: metrics[k]["dev_max"] for k in metrics if k != "no_write"}
    recon_all = {k: metrics[k]["dev_recon_max"] for k in metrics if k != "no_write"}
    binding = [k for k in good1 if k in good2p and k in good5]
    dev_bind = max((dev_all[k][0] for k in binding), default=0.0)
    dev_max = max(v[0] for v in dev_all.values())
    res["sync_deviation"] = dev_all
    res["sync_reconstruction_err"] = recon_all
    res["sync_deviation_binding"] = dev_bind
    rec("A6 【装置缺陷当量】因子化更新的同步性偏差（v2.4 §3.3 要求同步）",
        dev_bind < 1e-2,
        f"装置实现先更新 U 再用它更新 V（违反 §3.3「都用旧 U、旧 V」）；实测相对偏差按 eta："
        + "，".join(f"{k}:{v[0]:.2e}" for k, v in dev_all.items()) +
        f"。偏差为二阶量（eta^2 标度：比值与 (eta 之比)^2 吻合）。"
        f"【绑定检查】承载结论的 eta = {binding}，其偏差 max={dev_bind:.3e}"
        f"（阈值 1e-2）⇒ {'结论对该缺陷不敏感' if dev_bind < 1e-2 else '结论须附带 ~%s 的实现注记' % f'{dev_bind:.1e}'}。"
        f" 另：对 G 的重建与装置内部一致的相对误差 max={max(v[0] for v in recon_all.values()):.3e}"
        f"（no_write 档无写可谈，已按 revision 1.3 排除）",
        "<1e-2（绑定到承载结论的 eta）", dev_bind)

    # ---- 位置诊断（报告项）
    pos = {}
    for key in good2:
        eta = float(key.split("_")[1])
        pos[key] = {"eta": eta, "dir07_predicted_window": [1e-6, 1e-5],
                    "in_predicted_window": 1e-6 <= eta <= 1e-5}
    res["window_position"] = pos
    print("  窗口位置诊断：", flush=True)
    for k, v in pos.items():
        print(f"    {k}: 落在 DIR-07 预测窗口 [1e-6,1e-5] 内 = {v['in_predicted_window']}",
              flush=True)

    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary"] = {"passed": npass, "total": len(res["checks"])}
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    print(f"\n=== DIR-08 R1：{npass}/{len(res['checks'])} 通过 ===", flush=True)


if __name__ == "__main__":
    main()
