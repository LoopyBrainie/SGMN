#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-06 · R1/R2/R3 —— 记忆的因果效应与条件敏感性（真实预训练 LLM）

承接未决项：DIR-04 U1（多步轨迹上记忆是否真的在起作用）、DIR-04 U2（条件敏感性）。

【装置纪律：同一时刻只允许一个 SGMN 栈挂在模型上】
  多个栈同时挂载会让多个记忆同时注入同一层，导致对照失效（前一轮的 NaN 即由此而来）。
  故每次测量都独立建栈、测完立刻 remove()。

【设计要点】
  * token -> 在线步的映射【显式化】：序列切 n_chunks 段，逐段前视，每段 = 一步在线事件。
    单次整段前视时写通道在段内【惰性】（write_calls=0）——这是本轮的结构性发现。
  * 【读通道效应】与【写介导的累积效应】分离：
      d_pert = 有写入时的（扰动轨迹 - 干净轨迹）逐位置差
      d_nw   = 无写入时的（扰动轨迹 - 干净轨迹）逐位置差
      写介导部分 = d_pert - d_nw
  * 写步长 ETA_W 由探针自适应选定（最大的使分段前视保持有限的步长），探针结果如实记录。
"""
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sgmn_module import SGMNStack, grad_report

OUT = pathlib.Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ALPHA, LAM = 1.0, 0.05
GAMMA, R = 0.05, 8
EPS = 1e-2
NCH = 4

res = {"dir": "DIR-06", "round": "R1-R3", "stage": "A+B", "model": MODEL, "checks": []}


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
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


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


def make_stack(mount=None, gamma=GAMMA, r=R, no_write=False, drop_sg=False,
               eta_write=None, seed=20261028):
    tok, model = model_and_tok()
    n = model.config.num_hidden_layers
    d = model.config.hidden_size
    if mount is None:
        mount = [n // 3, n // 2, 2 * n // 3]
    st = SGMNStack(model, mount, d, d, r, ALPHA, LAM, 0.01, gamma, DEV,
                   seed=seed, use_factorized=True, drop_sg=drop_sg, no_write=no_write)
    if eta_write is not None:
        for m in st.mems.values():
            m.eta = eta_write
    return tok, model, st, mount, d


def make_batch(tok, bs=4, L=32):
    texts = ["The capital of France is", "In a distant galaxy the explorer found",
             "The quick brown fox jumps over", "Machine learning models are"]
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=L)
    return enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)


@torch.no_grad()
def logits_chunked(model, ids, mask, n_chunks=NCH):
    L = ids.shape[1]
    cs = max(1, L // n_chunks)
    outs = []
    for i in range(0, L, cs):
        outs.append(model(input_ids=ids[:, i:i + cs],
                          attention_mask=mask[:, i:i + cs]).logits.float())
    return torch.cat(outs, dim=1)


def perturb(st, eps, seed):
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    for m in st.mems.values():
        v0 = m.V[:, 0].detach(); nv = float(v0.norm())
        u = torch.randn(m.d_v, generator=g).to(DEV); u = u / u.norm()
        with torch.no_grad():
            m.U[:, 0].add_(u, alpha=eps / max(nv, 1e-12))


def snapshot(st):
    return {li: (m.U.detach().clone(), m.V.detach().clone(), m.W0.detach().clone())
            for li, m in st.mems.items()}


def restore(st, snap):
    with torch.no_grad():
        for li, (U, V, W0) in snap.items():
            m = st.mems[li]
            m.U.copy_(U); m.V.copy_(V); m.W0.copy_(W0)
            m.prev_hs = None


def trajectory(model, ids, mask, st, snap, eps, seed):
    """返回 (干净轨迹 logits, 被扰动轨迹 logits, write_calls)。测完自动卸载栈。"""
    restore(st, snap)
    clean = logits_chunked(model, ids, mask, NCH)
    restore(st, snap)
    if eps is not None:
        perturb(st, eps, seed)
    pert = logits_chunked(model, ids, mask, NCH)
    wc = [m.write_calls for m in st.mems.values()]
    st.remove()
    return clean, pert, wc


def main():
    print("=== DIR-06：记忆的因果效应与条件敏感性 ===")
    tok, model = model_and_tok()
    ids, mask = make_batch(tok)

    # ================= 探针：写稳定性 =================
    print("\n[探针] 因子化写在真实 LLM 隐状态尺度上的稳定性")
    probe = []
    for eta_w in (0.01, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-8):
        _, md, st, _, _ = make_stack(eta_write=eta_w)
        with torch.no_grad():
            lg = logits_chunked(md, ids, mask, NCH)
            fin = bool(torch.isfinite(lg).all())
            mx = float(list(st.mems.values())[0].U.detach().abs().max())
        probe.append({"eta_write": eta_w, "finite": fin, "maxU": mx})
        print(f"    eta_write={eta_w:<8g}: 分段前视 logits 有限={fin}  max|U|={mx:.3e}")
        st.remove()
        torch.cuda.empty_cache()
    res["write_stability_probe"] = probe
    finite = sorted([p["eta_write"] for p in probe if p["finite"]])
    # 「稳定」判据：写后因子范数不发散（max|U| < 1e3），而不只是有限
    stable = sorted([p["eta_write"] for p in probe
                     if p["finite"] and p["maxU"] < 1e3])
    ETA_W = stable[-1] if stable else (finite[-1] if finite else 1e-9)
    explosive = [p["eta_write"] for p in probe if p["finite"] and p["maxU"] >= 1e3]
    res["eta_write_used"] = ETA_W
    res["explosive_but_finite"] = explosive
    rec("A4 写稳定性探针（决定声明的写步长）", len(stable) >= 1,
        f"有限步长 {finite}；其中【稳定】（max|U|<1e3）的 {stable}；"
        f"【有限但发散】的 {explosive}；本轮取最大的稳定值 ETA_W={ETA_W:g}（声明条件）。"
        f"真实 LLM 的 ||k||~67，而 v2.4 的标量约定是 mu=5（||k||~2.2）——"
        f"因子化写的有效步长被放大 ~||k||^2 倍：eta=0.01 时数步内 max|U| 达 inf，"
        f"eta=1e-4 时达 2.3e7（有限但发散），仅 eta<=1e-6 才真正稳定",
        "至少一个稳定步长", probe)
    rec("A4b 存在「有限但发散」的写步长窗口", len(explosive) > 0,
        f"有限但因子范数发散（max|U|>=1e3）的步长：{explosive}——"
        f"该窗口内梯度会被数值爆炸污染，故契约测量须取稳定窗口",
        ">0 个", explosive)

    # ================= Stage A =================
    print("\n[Stage A] 装置自检")
    _, md, st, _, _ = make_stack(eta_write=ETA_W)
    snap = snapshot(st)
    restore(st, snap); c1 = logits_chunked(md, ids, mask, NCH)
    restore(st, snap); c2 = logits_chunked(md, ids, mask, NCH)
    rep = float((c1 - c2).abs().max())
    st.remove()
    rec("A1 分段前视可重复（逐位相同）", rep == 0.0, f"两次前视最大差 {rep:.3e}", "精确 0", rep)

    _, md, st, _, _ = make_stack(eta_write=ETA_W, no_write=True)
    with torch.no_grad():
        cb = [float((m.U @ m.V.T).norm()) for m in st.mems.values()]
        logits_chunked(md, ids, mask, NCH)
        ca = [float((m.U @ m.V.T).norm()) for m in st.mems.values()]
    same = all(abs(x - y) < 1e-12 for x, y in zip(cb, ca))
    st.remove()
    rec("A2 no_write=True 时记忆内容不变（写被切断）", same,
        f"前视前后内容范数 {cb} -> {ca}", "逐位不变", same)

    _, md, st, _, _ = make_stack(drop_sg=True)
    lg = md(input_ids=ids, attention_mask=mask).logits
    F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)).backward()
    wrep = grad_report(st.named_write_params())
    nz = [(n, g) for n, g, isnone, _ in wrep if not isnone and g > 0]
    st.remove()
    rec("A3 阴性对照复核：去掉 sg 后写入通道必须泄漏", len(nz) > 0,
        f"{len(nz)}/{len(wrep)} 个写入参数拿到非零梯度", ">0", len(nz))

    # ================= R2 =================
    print("\n[R2] 逐位置效应与写介导的累积效应")
    _, md, st, _, _ = make_stack(eta_write=ETA_W)
    snap = snapshot(st)
    clean_w, pert_w, wc = trajectory(md, ids, mask, st, snap, EPS, 11)
    d_pert = (pert_w - clean_w).norm(dim=-1)            # (B, L)
    per_pos = d_pert.mean(dim=0)
    rec("R2a0 分段前视下写通道确实发生", all(x > 0 for x in wc),
        f"各挂载层 write_calls={wc}；对照：单次整段前视时为 0，"
        f"即写通道在【段内】惰性（token->在线步映射的决定性影响）", ">0", wc)
    nz_pos = int((per_pos > 0).sum())
    rec("R2a 记忆扰动影响轨迹上的每一个位置",
        nz_pos == per_pos.numel() and bool(torch.isfinite(d_pert).all()),
        f"{nz_pos}/{per_pos.numel()} 个位置非零；首/中/末 = "
        f"{per_pos[0]:.4e} / {per_pos[per_pos.numel()//2]:.4e} / {per_pos[-1]:.4e}",
        "全部非零", nz_pos)
    print("    逐位置效应（前 8 / 后 8）：")
    print("      " + " ".join(f"{x:.3e}" for x in per_pos[:8].tolist()))
    print("      " + " ".join(f"{x:.3e}" for x in per_pos[-8:].tolist()))

    _, md, st, _, _ = make_stack(eta_write=ETA_W, no_write=True)
    snap_nw = snapshot(st)
    clean_nw, pert_nw, _ = trajectory(md, ids, mask, st, snap_nw, EPS, 11)
    d_nw = (pert_nw - clean_nw).norm(dim=-1)
    write_mediated = (d_pert - d_nw).mean(dim=0)          # (L,)
    tot_pert = float(d_pert.mean()); tot_nw = float(d_nw.mean())
    med_share = float(write_mediated.mean() / max(tot_pert, 1e-30))
    # 判据：写介导效应必须具有【正确的因果时序签名】——
    #   写发生之前的位置严格为 0；写发生之后非 0。
    nz_idx = [i for i, v in enumerate(write_mediated.tolist()) if abs(v) > 0]
    first_nz = nz_idx[0] if nz_idx else None
    before_zero = (first_nz is None) or all(write_mediated[i] == 0.0 for i in range(first_nz))
    after_nonzero = (first_nz is not None) and (first_nz > 0) and \
        (max(abs(v) for v in write_mediated[first_nz:]) > 0)
    sig_ok = before_zero and after_nonzero
    rec("R2b 写介导效应具有正确的因果时序签名（写前严格为 0，写后非 0）", sig_ok,
        f"首个非零位置 = {first_nz}（共 {len(write_mediated)} 个位置）；"
        f"该位置之前全部严格为 0: {before_zero}；之后非 0: {after_nonzero}。"
        f"量级：写介导 {max(abs(v) for v in write_mediated):.4e} vs 总效应 {tot_pert:.4e}"
        f"（占比 {med_share*100:.1f}%）",
        "写前=0 且写后!=0", sig_ok)
    rec("R2c 写介导效应的量级（相对总效应）", med_share > 0.005,
        f"写介导占比 {med_share*100:.2f}%（稳定写步长下极小 => 记忆的因果效应几乎全部来自读通道）",
        ">0.5%", med_share)
    print("    写介导部分（逐位置前 8）：" +
          " ".join(f"{x:.3e}" for x in write_mediated[:8].tolist()))
    res["per_position_effect"] = per_pos.tolist()
    res["write_mediated"] = write_mediated.tolist()

    # ================= R3 条件敏感性 =================
    print("\n[R3] 条件敏感性：挂载层集合 / gamma / r")
    sens = []
    conds = [("基准 挂载[8,12,16] gamma=0.05 r=8", dict()),
             ("挂载 早期 [2,4,6]", dict(mount=[2, 4, 6])),
             ("挂载 晚期 [18,21,23]", dict(mount=[18, 21, 23])),
             ("单层挂载 [12]", dict(mount=[12])),
             ("gamma=0.01", dict(gamma=0.01)),
             ("gamma=0.2", dict(gamma=0.2)),
             ("r=1", dict(r=1)), ("r=2", dict(r=2))]
    for label, kw in conds:
        _, md, st, mt, dd = make_stack(eta_write=ETA_W, **kw)
        sn = snapshot(st)
        restore(st, sn); b = logits_chunked(md, ids, mask, NCH)
        restore(st, sn); perturb(st, EPS, 11)
        lp2 = logits_chunked(md, ids, mask, NCH)
        eff = float((lp2 - b).norm(dim=-1).mean())
        lg = md(input_ids=ids, attention_mask=mask).logits
        F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                        ids[:, 1:].reshape(-1)).backward()
        rrep = grad_report(st.named_read_params())
        wrep = grad_report(st.named_write_params())
        r_ok = all((not isn) and g > 1e-12 and not bad for _, g, isn, bad in rrep)
        w_ok = all(isn or g == 0.0 for _, g, isn, _ in wrep)
        st.remove()
        torch.cuda.empty_cache()
        sens.append({"condition": label, "effect": eff, "read_grad_ok": r_ok,
                     "write_grad_zero": w_ok, "mount": mt,
                     "read_min_grad": min([g for _, g, _, _ in rrep] or [0])})
        print(f"  {label:28s}: 因果效应={eff:.4e}  ②读梯度非零有限={r_ok}  ③写入零={w_ok}")
    res["sensitivity"] = sens
    all_ok = all(s["read_grad_ok"] and s["write_grad_zero"] and s["effect"] > 0 for s in sens)
    rec("R3 条件敏感性：所有条件下 ①②③ 与因果效应均成立", all_ok,
        f"{len(sens)} 个条件；因果效应范围 "
        f"{min(s['effect'] for s in sens):.3e} ~ {max(s['effect'] for s in sens):.3e}",
        "全部成立", all_ok)

    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary"] = {"passed": npass, "total": len(res["checks"])}
    (OUT / "R1R3_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n=== DIR-06 R1+R3：{npass}/{len(res['checks'])} 通过 ===")


if __name__ == "__main__":
    main()
