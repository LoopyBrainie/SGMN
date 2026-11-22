#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-13 · R1 —— 键的去相关能否让检索发生？正交键能否分离「写规则」？

判据见 ../DIRECTION.md（H1–H6，冻结于 rev 1.0）。

装置：`sgmn_module_keys.py` = DIR-12 装置的逐字派生（md5 25bc6e90a3f2020fba558caa62b8a32d），
新增 `key_mode ∈ {linear, sparse, ortho}`；**写规则 $G=\\alpha(Wk-v)k^\\top+\\lambda(W-W^0)$
与读规则 $y=\\mathrm{sg}(W)q$ 一字未改**。

臂（任务层，λ=1 竞争读 + 查询对齐）：
  K0 linear（基线，= v2.4 §2.3 原样）｜K1 whitened（离线学习的白化 $P_k$）｜
  K2 sparse（top-32 门控，非线性键）｜K3 linear + 随机 $W_q$（对齐性对照）｜K5 sparse + 写关（严格对照）
代数探针：K4 ortho（人为严格正交的键）⇒ 只报写保真度 $\\cos(Wk_i,v_i)$（H5）
"""
import argparse
import json
import math
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sgmn_module_keys import SGMNStack as SGK   # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOUNT = [8, 12, 16]
ALPHA, LAM, GAMMA0, R = 1.0, 0.05, 0.05, 8
ETA = 1e-5      # 位置级写把有效步长放大 ~63x（卡片 rev 1.1 修订 12）；全矩阵分支的稳定值
TOPK = 32
SEEDS = [20261028, 20261104, 7]
N_DOCS, FILLER_TOK = 60, 30
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")

res = {"dir": "DIR-13", "round": "R1", "checks": [], "arms": {}}
REF_KEY_NORM = [None]   # 线性臂的平均键范数（用作各臂重标定 η 的参考）


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


# ------------------------------------------------------------------ 语料
def load_filler():
    import requests
    p = pathlib.Path.home() / "sgmn-loop" / "corpus" / "train.txt"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(CORPUS_BASE + "train.txt", timeout=90)
        r.raise_status()
        p.write_bytes(r.content)
    return p.read_text(encoding="utf-8", errors="ignore")


def build_facts(tok, seed, n=N_DOCS):
    import re as _re
    text = load_filler()
    paras = [q.strip() for q in _re.split(r"\n\s*\n", text) if len(q.strip()) > 120]
    rng = random.Random(seed)
    rng.shuffle(paras)
    pool = []
    for q in paras:
        pool.extend(tok(q).input_ids)
        if len(pool) > 40000:
            break
    docs, used = [], set()
    while len(docs) < n:
        i = rng.randrange(0, len(pool) - 200)
        fill = pool[i:i + FILLER_TOK]
        code = f"{rng.randrange(1000, 9999)}"
        if code in used:
            continue
        used.add(code)
        pre = " The vault code is"
        pre_ids = tok(pre).input_ids
        full_ids = tok(pre + " " + code + ".").input_ids
        if full_ids[:len(pre_ids)] != pre_ids:
            continue
        docs.append({"ids": fill + full_ids, "code_ids": full_ids[len(pre_ids):],
                     "prefix_len": len(fill) + len(pre_ids)})
    return docs


def sign_test_p(d):
    d = [x for x in d if abs(x) > 1e-15]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0
    k = sum(1 for x in d if x > 0)
    pk = math.comb(n, k) * 0.5 ** n
    p = sum(math.comb(n, i) * 0.5 ** n for i in range(n + 1)
            if math.comb(n, i) * 0.5 ** n <= pk + 1e-12)
    return min(1.0, p), k, n


def paired(A, B):
    D = np.asarray(A, dtype=float) - np.asarray(B, dtype=float)
    D = D[np.isfinite(D)]
    if len(D) == 0:
        return {"mean": float("nan"), "t": float("nan"), "p": 1.0, "pos": 0, "n": 0}
    m = float(D.mean())
    sd = float(D.std(ddof=1))
    t = m / (sd / math.sqrt(len(D))) if sd > 0 else 0.0
    ps, k, n = sign_test_p(list(D))
    return {"mean": m, "sd": sd, "t": t, "p": ps, "pos": k, "n": n}


def mem_state(st):
    out = {}
    for li, m in st.mems.items():
        Wf = (m.Wfull.detach().clone() if getattr(m, "Wfull", None) is not None else None)
        out[li] = (m.U.detach().clone(), m.V.detach().clone(),
                   m.W0.detach().clone(), Wf)
    return out


def reset_mem(st, snap):
    """【D12·协议修复】每篇文档前把记忆重置回初始状态（与 DIR-11/DIR-12 的 R1 同款）。
    漏掉这一步 ⇒ 记忆跨文档累积 ⇒ 位置级写在 η=1e-5 下爆炸（实测 max|W| 1.67→173→8.7e27）。"""
    with torch.no_grad():
        for li, (U, V, W0, Wf) in snap.items():
            m = st.mems[li]
            m.U.copy_(U)
            m.V.copy_(V)
            m.W0.copy_(W0)
            if Wf is not None and getattr(m, "Wfull", None) is not None:
                m.Wfull.copy_(Wf)
            m.prev_hs = None
            m.write_calls = 0
            m.keys_written.clear()
            m.vals_written.clear()


def state_of(m):
    """记忆状态：全矩阵分支用 Wfull（§3.2），因子化分支用 W0+UV^T（§3.3）。"""
    if getattr(m, "Wfull", None) is not None:
        return m.Wfull.detach()
    return (m.W0.detach() + m.U.detach() @ m.V.detach().T)


def pr_eff(lam):
    lam = np.asarray(lam, dtype=float)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 0.0
    return float(lam.sum() ** 2 / (lam ** 2).sum())


# ------------------------------------------------------------------ 装置
def make_stack(model, d, key_mode="linear", seed=20261028, no_write=False, random_q=False):
    # ★ 与 DIR-12 R1 完全相同的写入配置（位置级 + 全矩阵，§3.2）⇒ 本轮唯一变量是【键的构造】
    st = SGK(model, MOUNT, d, d, R, ALPHA, LAM, ETA, GAMMA0, DEV,
             seed=seed, use_factorized=False, no_write=no_write, perpos=True)
    st._key_tag = f"{key_mode}{'_randq' if random_q else ''}{'_nowrite' if no_write else ''}"
    for m in st.mems.values():
        m.key_mode = key_mode
        m.key_topk = TOPK
        m.read_beta = 16.0
        m.read_lambda = 1.0
    if not random_q:
        with torch.no_grad():
            for m in st.mems.values():
                m.Wq.copy_(m.Pk)
    return st


def run_arm(st, docs):
    """写阶段一次整篇前视 → 读阶段（写冻结）截断到码之前；返回逐文档量 + 键诊断。"""
    keys = {li: [] for li in st.mems}
    vals = {li: [] for li in st.mems}
    rows = []
    snap = mem_state(st)
    # ---- 【D14·步长重标定】各臂的键范数不同 ⇒ 写步长 ∝ η‖k‖² ⇒ 按各臂的平均 ‖k‖² 反比缩放 η
    #      （卡片 §4-2 预登记）⇒ 使"写进去的量"在各臂间可比，避免把"写弱了"误判成"键不好"
    with torch.no_grad():
        st.model(input_ids=torch.tensor(docs[0]["ids"], device=DEV).unsqueeze(0))
    kn = []
    for li, m in st.mems.items():
        if m.keys_written:
            kn.append(float(torch.stack(m.keys_written).norm(dim=-1).mean()))
    reset_mem(st, snap)
    if kn and REF_KEY_NORM[0] is not None:
        scale = (REF_KEY_NORM[0] ** 2) / (float(np.mean(kn)) ** 2)
        for m in st.mems.values():
            m.eta = ETA * scale
        res.setdefault("eta_rescale", {})[st._key_tag] = float(ETA * scale)
    else:
        REF_KEY_NORM[0] = float(np.mean(kn)) if kn else None
        res.setdefault("eta_rescale", {})[st._key_tag] = float(ETA)
    for d in docs:
        reset_mem(st, snap)                               # ★ 每篇文档前重置记忆（见 D12）
        ids = torch.tensor(d["ids"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            st.model(input_ids=ids)
        tp = d["prefix_len"] - 1
        for li, m in st.mems.items():
            # 【D13】用装置【实际写入】的键与值（而不是脚本里另算的线性投影）
            if m.keys_written:
                keys[li].append(torch.stack(m.keys_written))   # (T,d_k) 本篇实际写入的键
                vals[li].append(torch.stack(m.vals_written))   # (T,d_v) 本篇实际写入的值
            m.keys_written.clear()
            m.vals_written.clear()
        pre = ids[:, :d["prefix_len"]]
        with torch.no_grad():
            lg = st.model(input_ids=pre).logits.float()
        lp = float(F.log_softmax(lg[0, -1], dim=-1)[d["code_ids"][0]])
        hits, ctrue, amax = [], [], []
        for li, m in st.mems.items():
            y = m.last_y[0] if m.last_y is not None else None
            if y is None:
                continue
            if not vals[li]:
                hits.append(0.0)
                ctrue.append(0.0)
                amax.append(float(m.last_attn.max()) if m.last_attn is not None else 0.0)
                continue
            V = torch.cat(vals[li], 0)                     # 本篇文档内实际写入的值（位置 1..L-1）
            cs = F.cosine_similarity(V, y.unsqueeze(0), dim=1)
            # 位置 t 的值写在索引 t-1（写入位置 1..L-1 ⇒ 索引 0..L-2）
            ti = tp - 1
            hits.append(1.0 if int(torch.argmax(cs)) == ti else 0.0)
            ctrue.append(float(cs[ti]) if 0 <= ti < cs.shape[0] else 0.0)
            amax.append(float(m.last_attn.max()) if m.last_attn is not None else 0.0)
        rows.append({"lp_first": lp, "hit": float(np.mean(hits)) if hits else 0.0,
                     "ctrue": float(np.mean(ctrue)) if ctrue else 0.0,
                     "attn_max": float(np.mean(amax)) if amax else 0.0})
    # 键诊断（用写阶段实际记录的键与值）
    diag = {}
    for li, m in st.mems.items():
        if not keys[li]:
            diag[str(li)] = {"key_cos_mean": float("nan"), "key_pr": float("nan"),
                             "write_fid_mean": float("nan"), "key_norm": float("nan"),
                             "eta": float(getattr(m, "eta", ETA))}
            continue
        K = torch.cat(keys[li], 0).double()              # (N*L,d_k)
        V = torch.cat(vals[li], 0).double()              # (N*L,d_v)
        Kn = K / K.norm(dim=1, keepdim=True).clamp_min(1e-30)
        # 成对余弦（抽样上限 800 对，避免 O(n^2) 爆内存）
        n = Kn.shape[0]
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(7))[:800]
        S = Kn[idx]
        G = (S @ S.T).cpu().numpy()
        off = G[~np.eye(G.shape[0], dtype=bool)]
        sv = torch.linalg.svdvals(S).cpu().numpy()      # 奇异值（比 Gram 的 eigh 稳定）
        ev = sv ** 2
        W = m.Wfull.detach().double() if getattr(m, "Wfull", None) is not None else (
            m.W0.detach() + m.U.detach() @ m.V.detach().T).double()
        fid = F.cosine_similarity((W @ K.T).T, V, dim=1)   # 逐位置：写后的 W 作用在该位置的键上
        diag[str(li)] = {"key_cos_mean": float(off.mean()),
                         "key_pr": pr_eff(ev),
                         "write_fid_mean": float(fid.mean()),
                         "key_norm": float(K.norm(dim=1).mean())}
    return rows, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_docs = 8 if args.smoke else N_DOCS
    seeds = SEEDS[:2] if args.smoke else SEEDS

    print("=== DIR-13 R1：键的去相关 vs 检索 ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size

    # ---------------- 白化校准（用【基座】的隐状态 ⇒ 干净的校准分布）
    print("\n[校准] 采集基座隐状态 → 白化 $P_k$", flush=True)
    cal = {li: [] for li in MOUNT}
    handles = []
    layers = model.model.layers
    for li in MOUNT:
        def pre_hook(mod, inp, li=li):
            cal[li].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu())
        handles.append(layers[li].register_forward_pre_hook(pre_hook))
    with torch.no_grad():
        for _ in range(4):
            ids = torch.randint(100, 100000, (1, 512), device=DEV)
            model(input_ids=ids)
    for h in handles:
        h.remove()
    whiten_Pk = {}
    for li in MOUNT:
        X = torch.cat(cal[li], 0).double()
        X = X - X.mean(0, keepdim=True)
        Sig = (X.T @ X) / max(1, X.shape[0] - 1)
        g = torch.Generator().manual_seed(1000 + li)
        Pk = torch.randn(d, d, generator=g).double() / math.sqrt(d)
        Sk = Pk.T @ Sig @ Pk
        ev, V = torch.linalg.eigh(Sk)
        ev = ev.clamp_min(1e-12)
        eps = 1e-6 * float(ev.max())
        inv = (ev + eps).rsqrt()
        Pkw = (Pk @ V * inv) @ V.T
        whiten_Pk[li] = Pkw.float()
        print(f"  层 {li}: 校准样本 {X.shape[0]}，特征值 [{float(ev.min()):.3e}, {float(ev.max()):.3e}]",
              flush=True)

    # ---------------- Stage A
    print("\n[Stage A]", flush=True)
    from sgmn_module_keys import grad_report
    st = make_stack(model, d, "linear")
    base_before = {n: q.detach().clone() for n, q in model.named_parameters()}
    ids = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
    lg = model(input_ids=ids).logits.float()
    F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)).backward()
    rrep = grad_report(st.named_read_params())
    wrep = grad_report(st.named_write_params())
    gam = [g for n, g, i, b in rrep if n.endswith("gamma")]
    read_ok = (all((not b) for _, _, _, b in rrep) and all(g > 0 for g in gam)
               and max(g for _, g, _, _ in rrep) > 1e-6)
    write_ok = all(isn or g == 0.0 for _, g, isn, _ in wrep)
    base_same = all(torch.equal(q.detach(), base_before[n]) for n, q in model.named_parameters())
    nh, nm = len(st._handles), len(st.mems)
    st.remove()
    torch.cuda.empty_cache()
    st2 = SGK(model, MOUNT, d, d, R, ALPHA, LAM, ETA, GAMMA0, DEV, seed=20261028,
              use_factorized=False, perpos=True, drop_sg=True)
    for m in st2.mems.values():
        m.key_mode = "linear"
        m.read_beta = 16.0
        m.read_lambda = 1.0
    lg2 = model(input_ids=ids).logits.float()
    F.cross_entropy(lg2[:, :-1].reshape(-1, lg2.shape[-1]), ids[:, 1:].reshape(-1)).backward()
    wrep2 = grad_report(st2.named_write_params())
    leak = len([1 for _, g, isn, _ in wrep2 if not isn and g > 0])
    st2.remove()
    torch.cuda.empty_cache()
    for q in model.parameters():
        q.grad = None
    rec("A1' Freeze C 逐臂复检（写入梯度精确0 / 读梯度有限非零 / 基座逐位不变 / 阴性对照）",
        read_ok and write_ok and base_same and leak > 0,
        f"读梯度 {sum(1 for _,_,i,_ in rrep if not i)}/{len(rrep)} 非零且有限；"
        f"写入梯度精确0 {sum(1 for _,g,i,_ in wrep if i or g==0.0)}/{len(wrep)}；"
        f"基座逐位不变 {base_same}；阴性对照泄漏 {leak}/{len(wrep2)}",
        "读梯度>0 且 写入梯度==0 且 基座不变 且 泄漏>0",
        {"read": read_ok, "write": write_ok, "base": base_same, "leak": leak})

    # A3'：单一装置 + 读阶段记忆冻结（两侧）
    st = make_stack(model, d, "linear")
    nh, nm = len(st._handles), len(st.mems)
    with torch.no_grad():
        model(input_ids=ids)
    Wb = {li: state_of(st.mems[li]).cpu().clone() for li in st.mems}
    with torch.no_grad():
        model(input_ids=ids)
    dm_on = max(float((state_of(st.mems[li]).cpu() - Wb[li]).abs().max()
                       / max(float(Wb[li].abs().max()), 1e-30)) for li in st.mems)
    st.remove()
    torch.cuda.empty_cache()
    st = make_stack(model, d, "linear", no_write=True)
    with torch.no_grad():
        model(input_ids=ids)
    Wb2 = {li: state_of(st.mems[li]).cpu().clone() for li in st.mems}
    with torch.no_grad():
        model(input_ids=ids)
    dm_off = max(float((state_of(st.mems[li]).cpu() - Wb2[li]).abs().max()
                        / max(float(Wb2[li].abs().max()), 1e-30)) for li in st.mems)
    st.remove()
    torch.cuda.empty_cache()
    rec("A3' 单一装置（句柄 2×层数）+ 写关记忆冻结 / 写开必须变化",
        nh == 2 * nm and dm_off < 1e-12 and dm_on > 1e-9,
        f"句柄 {nh}；写关相对差 {dm_off:.3e}（须 <1e-12）；写开相对差 {dm_on:.3e}（须 >1e-9）",
        "句柄==2*层数 且 写关<1e-12 且 写开>1e-9", {"off": dm_off, "on": dm_on})

    # ---------------- 主测量
    arms = {}
    for seed in seeds:
        print(f"\n  [种子 {seed}]", flush=True)
        docs = build_facts(tok, seed, n_docs)
        sa = {}
        for name, key_mode, random_q, nowrite in (
                ("K0_linear", "linear", False, False),
                ("K1_whitened", "whitened", False, False),
                ("K2_sparse", "sparse", False, False),
                ("K3_linear_randomq", "linear", True, False),
                ("K5_sparse_nowrite", "sparse", False, True)):
            st = make_stack(model, d, key_mode, seed=seed, no_write=nowrite, random_q=random_q)
            if key_mode == "whitened":
                with torch.no_grad():
                    for li, m in st.mems.items():
                        if li in whiten_Pk:
                            m.Pk.copy_(whiten_Pk[li])
                            m.Wq.copy_(m.Pk)
            rows, diag = run_arm(st, docs)
            st.remove()
            torch.cuda.empty_cache()
            sa[name] = {k: [r[k] for r in rows] for k in ("lp_first", "hit", "ctrue", "attn_max")}
            sa[name]["diag"] = diag
            print(f"    {name:20s} 码logp {np.mean(sa[name]['lp_first']):+.4f} "
                  f"命中 {np.mean(sa[name]['hit']):.4f} "
                  f"键cos {np.mean([v['key_cos_mean'] for v in diag.values()]):+.4f} "
                  f"键PR {np.mean([v['key_pr'] for v in diag.values()]):.3f} "
                  f"写保真 {np.mean([v['write_fid_mean'] for v in diag.values()]):+.4f} "
                  f"注意力max {np.mean(sa[name]['attn_max']):.4f}", flush=True)
        arms[seed] = sa

    # ---------------- K4：正交键探针（决定性分离）
    print("\n[K4] 正交键探针（代数层）", flush=True)
    g = torch.Generator().manual_seed(4242)
    A = torch.randn(4096, d, generator=g).double()
    Q, _ = torch.linalg.qr(A)          # 4096 行严格正交
    Q = Q.float()
    probe = {}
    docs = build_facts(tok, 20261028, 8)
    for tag, mode in (("K0_linear_probe", "linear"), ("K4_ortho_probe", "ortho")):
        if mode == "ortho":
            # 【D10】正交键库必须在任何前视之前装配好（否则第一次写就用到 None 的键库）
            # 先用 linear 栈量出平均键范数 ⇒ 正交键标定到同一尺度 ⇒ 写步长可比
            st_tmp = make_stack(model, d, "linear", seed=20261028)
            with torch.no_grad():
                st_tmp.model(input_ids=torch.tensor(docs[0]["ids"], device=DEV).unsqueeze(0))
            m0 = st_tmp.mems[list(st_tmp.mems)[0]]
            scale = float((st_tmp._hs_in[list(st_tmp.mems)[0]][0] @ m0.Pk.detach().T).norm(dim=-1).mean())
            st_tmp.remove()
            torch.cuda.empty_cache()
            st = make_stack(model, d, mode, seed=20261028)
            for m in st.mems.values():
                m.ortho_bank = (Q * scale).to(DEV)      # 【D11】键库必须在模型设备上
        else:
            st = make_stack(model, d, mode, seed=20261028)
        rows, diag = run_arm(st, build_facts(tok, 20261028, 8))
        st.remove()
        torch.cuda.empty_cache()
        probe[tag] = {"fid": float(np.mean([v["write_fid_mean"] for v in diag.values()])),
                      "key_cos": float(np.mean([v["key_cos_mean"] for v in diag.values()])),
                      "key_pr": float(np.mean([v["key_pr"] for v in diag.values()]))}
        print(f"    {tag:16s} 写保真度 {probe[tag]['fid']:+.4f}  键cos {probe[tag]['key_cos']:+.4f} "
              f"键PR {probe[tag]['key_pr']:.3f}", flush=True)
    res["probe"] = probe

    # ---------------- 分析
    print("\n[R1] 分析", flush=True)

    def agg(seed, arm, key):
        return np.asarray(arms[seed][arm][key], dtype=float)

    def cmp(a, b, key="hit"):
        A = np.concatenate([agg(s, a, key) for s in seeds])
        B = np.concatenate([agg(s, b, key) for s in seeds])
        c = paired(A, B)
        c["seed_means"] = [float(np.mean(arms[s][a][key]) - np.mean(arms[s][b][key])) for s in seeds]
        c["consistent"] = bool(c["pos"] == c["n"] or c["pos"] == 0)
        return c

    summ = {}
    for arm in arms[seeds[0]]:
        summ[arm] = {k: float(np.mean([np.mean(arms[s][arm][k]) for s in seeds]))
                     for k in ("lp_first", "hit")}
    res["summary"] = summ
    print("  各臂：", json.dumps(summ, ensure_ascii=False), flush=True)

    # A2'：键构造的生效性
    k0_cos = np.mean([v["key_cos_mean"] for v in arms[seeds[0]]["K0_linear"]["diag"].values()])
    k0_pr = np.mean([v["key_pr"] for v in arms[seeds[0]]["K0_linear"]["diag"].values()])
    eff = {}
    for arm in ("K1_whitened", "K2_sparse"):
        c_ = np.mean([v["key_cos_mean"] for v in arms[seeds[0]][arm]["diag"].values()])
        p_ = np.mean([v["key_pr"] for v in arms[seeds[0]][arm]["diag"].values()])
        eff[arm] = {"key_cos": float(c_), "key_pr": float(p_),
                    "improved": bool((c_ < 0.8 * k0_cos) or (p_ > 1.2 * k0_pr))}
    res["key_effectiveness"] = eff
    rec("A2' 键构造的生效性（去相关臂的键间余弦 ↓20% 或 有效槽位 PR ↑20%）",
        all(v["improved"] for v in eff.values()),
        f"K0 基线：键cos {k0_cos:.4f}，键PR {k0_pr:.3f}；" +
        "；".join(f"{a}: cos {v['key_cos']:.4f}, PR {v['key_pr']:.3f}, 改善 {v['improved']}"
                 for a, v in eff.items()),
        "至少一项指标相对 K0 改善 ≥20%", eff)

    h1 = cmp("K1_whitened", "K0_linear", "hit")
    h1b = cmp("K2_sparse", "K0_linear", "hit")
    best = max(("K1_whitened", "K2_sparse"), key=lambda a: summ[a]["hit"])
    h2 = {"best": best, "hit": summ[best]["hit"],
          "per_seed": [float(np.mean(arms[s][best]["hit"])) for s in seeds]}
    h3 = cmp("K2_sparse", "K5_sparse_nowrite", "hit")
    h4 = cmp("K2_sparse", "K0_linear", "lp_first")
    h5 = {"mean": probe["K4_ortho_probe"]["fid"] - probe["K0_linear_probe"]["fid"],
          "k0": probe["K0_linear_probe"]["fid"], "k4": probe["K4_ortho_probe"]["fid"],
          "k4_key_cos": probe["K4_ortho_probe"]["key_cos"],
          "k4_key_pr": probe["K4_ortho_probe"]["key_pr"]}

    rec("H1 〔键去相关是必要条件〕白化臂 K1 的命中显著 > 基线 K0",
        h1["mean"] > 0 and h1["p"] < 0.05 and h1["consistent"],
        f"K1-K0 配对差 {h1['mean']:+.4f}（t={h1['t']:+.2f}, p={h1['p']:.4g}, 正向 {h1['pos']}/{h1['n']}）"
        f"跨种子 " + " ".join(f"{x:+.4f}" for x in h1["seed_means"]),
        "均值>0 且 p<0.05 且 跨种子同向", h1)
    rec("H1' 〔稀疏门控〕稀疏臂 K2 的命中显著 > 基线 K0",
        h1b["mean"] > 0 and h1b["p"] < 0.05 and h1b["consistent"],
        f"K2-K0 配对差 {h1b['mean']:+.4f}（t={h1b['t']:+.2f}, p={h1b['p']:.4g}, 正向 {h1b['pos']}/{h1b['n']}）"
        f"跨种子 " + " ".join(f"{x:+.4f}" for x in h1b["seed_means"]),
        "均值>0 且 p<0.05 且 跨种子同向", h1b)
    rec("H2 〔存在性〕最佳臂命中 ≥ 0.9（随机基线 0.0244）",
        h2["hit"] >= 0.9 and all(x >= 0.9 for x in h2["per_seed"]),
        f"最佳臂 {h2['best']}，命中 {h2['hit']:.4f}（逐种子 "
        + " ".join(f"{x:.4f}" for x in h2["per_seed"]) + "），随机基线 0.0244",
        "≥0.9 且 3 种子均满足", h2)
    rec("H3 〔严格对照〕K2 命中显著 > 其写关对照 K5",
        h3["mean"] > 0 and h3["p"] < 0.05 and h3["consistent"],
        f"K2-K5 配对差 {h3['mean']:+.4f}（t={h3['t']:+.2f}, p={h3['p']:.4g}, 正向 {h3['pos']}/{h3['n']}）",
        "均值>0 且 p<0.05 且 跨种子同向", h3)
    rec("H4 〔任务层〕K2 的正确码 log-prob 显著 > K0",
        h4["mean"] > 0 and h4["p"] < 0.05 and h4["consistent"],
        f"K2-K0 码logp 配对差 {h4['mean']:+.4f}（t={h4['t']:+.2f}, p={h4['p']:.4g}）",
        "均值>0 且 p<0.05 且 跨种子同向", h4)
    rec("H5 〔决定性分离〕正交键探针 K4 的写保真度显著 > K0（否则障碍在写规则）",
        h5["mean"] > 0,
        f"K4 写保真度 {h5['k4']:+.4f}（键cos {h5['k4_key_cos']:+.4f}、键PR {h5['k4_key_pr']:.3f}）"
        f"vs K0 {h5['k0']:+.4f}，差 {h5['mean']:+.4f}",
        "K4 - K0 > 0", h5)

    res["comparisons"] = {"H1": h1, "H1b": h1b, "H2": h2, "H3": h3, "H4": h4, "H5": h5}
    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary_pass"] = {"passed": npass, "total": len(res["checks"])}
    (HERE / ("R1_smoke_results.json" if args.smoke else "R1_results.json")).write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== DIR-13 R1：{npass}/{len(res['checks'])} 通过 ===", flush=True)


if __name__ == "__main__":
    main()
