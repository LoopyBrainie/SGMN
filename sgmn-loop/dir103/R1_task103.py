#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-103 —— `\\mathcal O_l` 的**输入基**：只看 `y`，还是也看读取上下文 `c`？

卡片：revolve/DIR-103/DIRECTION_rev103_v10.md（rev 1.0 冻结，阈值 0.99/0.50/0.90 不得改）

装置：`dir101/R1_task101.py` 的 **white 键臂**逐字（键正交、内容 32/32 已由 D-128 判定）。
唯一自变量 = **`\\mathcal O_l` 的输入基** `--inp {y,c,cy}`：
    m = Wo @ y  (+  Wc @ c)          ← `Wc` 是**共享满秩**新参数（896×896/层），非逐事实
训练集 = 全部 32 个事实（问**容量**，不问留出）。
主读数 = `acc_all32`（与 D-131/A101-1/D-132 同口径）。
机制臂 `cy_W0`：训练后把 `W ← W0`（抹掉 600 块写入）、其余参数不动，重测 ⇒ 判别「经 W 中介」vs「旁路」。

★ 不 import `probe91_woform`（import 期会换掉 `LayerMem.read`）；
  `optimise_free` 不需要（本卡无自由向量臂）。
★ `DEF-101-08` 的教训：`--ranks` 默认路径曾抛 `AttributeError`，本卡不设秩网格（自变量不是秩）。
"""
import argparse
import json
import pathlib
import random
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "dir91"))
sys.path.insert(0, str(_HERE.parent / "dir99"))
sys.path.insert(0, str(_HERE.parent / "dir100"))
sys.path.insert(0, str(_HERE.parent / "dir101"))

import probe91_kpool as KP                       # noqa: E402
import R2_task91 as R91                          # noqa: E402
import R1_task100 as D100                        # noqa: E402
import torch                                     # noqa: E402
from sgmn_module import LayerMem, sg             # noqa: E402

DEV = R91.DEV
SEEDS = [20261028, 7777]
KEY_ARM = "white"          # ★ 键固定 white（D-128：内容 31–32/32）
ARMS = ["y", "c", "cy"]    # `cy_W0` 复用 `cy` 的训练，只换 W 后重测 ⇒ 不额外训练
# ---- 冻结阈值（卡片 §1.4；不得改）----
THR_ROUTE = 0.99           # `cy` 达此值才算"到得了"
NEG_MAX = 0.50             # `c`-only 负对照上界（值不在 prompt ⇒ 应接近 chance 0.25）
W0_MIN = 0.90              # `cy_W0` ≥ 此值 ⇒ 增益不经 W（旁路）
FLOOR_HIT = 24             # `A103-1`：写后 N_hit^v ≥ 24/32（D-128 的 M98-2 预注册线）
BAND_LO, BAND_HI = 0.70, 0.99   # `A103-2`：`y`-only 满秩复现 D-131 的区间带
OUT = _HERE / "out"
RES = {"guards": {}, "cells": [], "toy": {}}

_ORIG_READ = LayerMem.read
_ORIG_WRITE = LayerMem.write

# ★ 全局状态（patch 内读取；逐格显式设置/还原）
_INP = "cy"
_WC = {}                   # {li: Tensor(d_model,d_model)}  或 {}
_C_CAP = False
_C_OUT = None


def rec(cid, ok, detail, value=None):
    RES["guards"][cid] = {"ok": bool(ok), "detail": detail, "value": value}
    print(f"  {'✓' if ok else '✗ FAIL'} {cid:<11} {detail}", flush=True)
    return ok


def rec_only(cid, detail, value=None):
    """只记录、不参与 `guards_ok` 判定（`DEF-101-07` 的先例）。"""
    RES["guards"][cid] = {"ok": True, "detail": detail, "value": value, "report_only": True}
    print(f"  ○ {cid:<11} {detail}", flush=True)


# ==================================================================== 读侧 patch
def _read_r103(self, hs_in, drop_sg=False):
    """= `dir101/R1_task101._read_r101` 逐字，**唯一改动**：`m` 可再加一路 `Wc @ c`。

    `--inp y`  ⇒ 只有 `Wo @ y`（逐位等价 `_read_r101` 的无秩臂）
    `--inp c`  ⇒ 只有 `Wc @ c`（负对照）
    `--inp cy` ⇒ 两项相加
    """
    wc = _WC.get(id(self)) if _WC else None
    cap = _C_CAP
    if not cap and (_INP == "y" or (wc is None and _INP == "c")):
        # 快路径：与原生读逐位一致（不改变任何张量路径）
        if _INP == "y":
            return D100._read_tf(self, hs_in, drop_sg)
    self.read_calls += 1
    c = sg(hs_in, drop_sg)
    cf = c.reshape(-1, c.shape[-1])
    q = (self.Wq @ cf.T).T
    q = self.q_tf(q)
    W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
    y = (sg(W, drop_sg) @ q.T).T                       # (N, d_v)
    if _INP == "c":
        m = (wc @ cf.T).T                              # (N, d_model)
    elif _INP == "y":
        m = (self.Wo @ y.T).T
    else:
        m = (self.Wo @ y.T).T
        if wc is not None:
            m = m + (wc @ cf.T).T
    if cap:
        Bn, Ln = hs_in.shape[0], hs_in.shape[1]
        idx = torch.arange(Bn, device=cf.device) * Ln + (Ln - 2)
        # ★ 保留整段（不只第一个 batch），供 A103-4 逐位比对
        global _C_OUT
        _C_OUT = cf[idx].detach().clone()
    self.injection = m.reshape(hs_in.shape)
    return self.injection


def install():
    LayerMem.write = D100._write_tf
    LayerMem.read = _read_r103


def restore():
    LayerMem.write = _ORIG_WRITE
    LayerMem.read = _ORIG_READ


def set_inp(mode):
    global _INP
    assert mode in ("y", "c", "cy"), mode
    _INP = mode


def set_wc(st, make=False, val=None):
    """`make=True` ⇒ 为每层新建满秩 `Wc (d_model,d_model)`（`mk` 同款初始化）。"""
    global _WC
    _WC = {}
    for li, m in st.mems.items():
        if val is not None:
            m._WCp = val
        if make:
            g = torch.Generator(device="cpu").manual_seed(7919 + li)
            m._WCp = (torch.randn(m.d_model, m.d_model, generator=g)
                      / (m.d_model ** 0.5)).to(m.Wo.device).requires_grad_(True)
        if getattr(m, "_WCp", None) is not None:
            _WC[id(m)] = m._WCp
    return {li: getattr(m, "_WCp", None) for li, m in st.mems.items()}


def clear_wc(st):
    global _WC
    _WC = {}
    for m in st.mems.values():
        m._WCp = None


def set_capture(on):
    global _C_CAP, _C_OUT
    _C_CAP = bool(on)
    if not on:
        _C_OUT = None


def set_gamma(st, val):
    for m in st.mems.values():
        m.gamma.data.fill_(float(val))


# ==================================================================== 评测
@torch.no_grad()
def eval_acc(model, st, tok, facts, keys, seed=555):
    """4 选 1、按题计（同题 `N_DRAWS` 次全对才算命中）；`no_write`；返回 (acc, n_q)。
    = `dir101/R1_task101.eval_acc` 逐字。"""
    rng = random.Random(seed)
    items = R91.make_eval_items(tok, facts, keys, rng)
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    per = {}
    try:
        for it in items:
            P = len(it["prompt_ids"])
            seqs = [it["prompt_ids"] + [c[0][0]] for c in it["cand"]]
            seq = torch.tensor(seqs, dtype=torch.long, device=DEV)
            tok_ids = torch.tensor([s[-1] for s in seqs], device=DEV)
            lg = model(input_ids=seq).logits.float()
            ar = torch.arange(len(seqs), device=DEV)
            pick = lg[ar, P - 1, tok_ids]
            best = int(torch.argmax(pick))
            per.setdefault(it["key_idx"], []).append(it["cand"][best][1])
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
    nq = len(per)
    hit = sum(1 for v in per.values() if all(v))
    return hit / nq, nq


# ==================================================================== 训练（唯一改动：可含 Wc）
def train_read_v2(model, st, tok, facts, train_keys, rng, with_y=True, with_c=False,
                  steps=R91.READ_STEPS, lr=R91.READ_LR):
    """逐字派生 `dir101/R1_task101.train_full_wo`；**唯一差别**：可把 `Wc` 并入待训参数。
    `W` 冻结（`no_write=True`）；`W0/U/V/Wfull` 不进阶（`A103-3`）。"""
    for m in st.mems.values():
        m.no_write = True
    ps = []
    for li, m in st.mems.items():
        ps += [m.Wq, m.gamma]
        if with_y:
            ps.append(m.Wo)
        if with_c:
            ps.append(m._WCp)
    opt = torch.optim.Adam(ps, lr=lr)
    losses, viol, div = [], 0, False
    batches = []
    for ki in train_keys:
        f = facts[ki]
        others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != f["val"]]
        distr = rng.sample(sorted(set(others)), R91.N_CAND - 1)
        cands = [f["val"]] + distr
        rng.shuffle(cands)
        ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in cands]
        batches.append((f["prompt_ids"], ids, cands.index(f["val"])))
    with torch.enable_grad():
        for step in range(steps):
            prompt_ids, ids, tgt = batches[step % len(batches)]
            P = len(prompt_ids)
            seq = torch.tensor([prompt_ids + [c] for c in ids], dtype=torch.long, device=DEV)
            opt.zero_grad(set_to_none=True)
            lg = model(input_ids=seq).logits.float()
            idx = torch.arange(len(ids), device=DEV)
            pick = lg[idx, P - 1, torch.tensor(ids, device=DEV)].unsqueeze(0)
            loss = torch.nn.functional.cross_entropy(pick, torch.tensor([tgt], device=DEV))
            loss.backward()
            if step % 50 == 0:
                for p in R91.write_params(st):
                    if p.grad is not None and float(p.grad.abs().max()) != 0.0:
                        viol += 1
            torch.nn.utils.clip_grad_norm_([p for p in ps if p.grad is not None], R91.CLIP)
            opt.step()
            losses.append(float(loss))
            if not all(bool(torch.isfinite(R91.W_of(m)).all()) for m in st.mems.values()):
                div = True
                break
    for m in st.mems.values():
        m.no_write = False
    return losses, viol, div


# ==================================================================== 手算算例（A103-9）
def toy_example():
    """§2.2：`d_model=d_v=2`、`W=diag(1,2)`、`Wq=Wo=Wc=I`、`c=(3,4)` ⇒
         y 臂 `m=(3,8)` · c 臂 `m=(3,4)` · cy 臂 `m=(6,12)`，且 `cy-y == Wc@c`。"""
    W = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    Wq = torch.eye(2)
    Wo = torch.eye(2)
    Wc = torch.eye(2)
    c = torch.tensor([[3.0, 4.0]])
    q = (Wq @ c.T).T
    y = (W @ q.T).T
    my = (Wo @ y.T).T
    mc = (Wc @ c.T).T
    mcy = my + mc
    d1 = float((my.squeeze(0) - torch.tensor([3.0, 8.0])).abs().max())
    d2 = float((mc.squeeze(0) - torch.tensor([3.0, 4.0])).abs().max())
    d3 = float((mcy.squeeze(0) - torch.tensor([6.0, 12.0])).abs().max())
    d4 = float((mcy - my - mc).abs().max())
    RES["toy"] = {"y": d1, "c": d2, "cy": d3, "diff": d4}
    return rec("A103-9", d1 < 1e-12 and d2 < 1e-12 and d3 < 1e-12 and d4 < 1e-12,
               f"手算：‖y臂−[3,8]‖={d1:.1e} ‖c臂−[3,4]‖={d2:.1e} "
               f"‖cy臂−[6,12]‖={d3:.1e} ‖cy−y−Wc·c‖={d4:.1e}（四条须全=0）",
               RES["toy"])


# ==================================================================== 单格
def run_cell(model, tok, facts, seed, arm, stage, chunks, owner):
    """`arm ∈ {y, c, cy}`；`cy_W0` 不单独训练（见 `main`）。
    ★ 一格 = **一次全新栈**（`DEF-95-02`：一次一栈）。"""
    t0 = time.time()
    st = R91.make_stack(gamma=0.0, no_write=False, seed=seed)
    res = {"seed": seed, "arm": arm, "stage": stage}
    try:
        for m in st.mems.values():
            m.key_tf = D100.IdentityTF()
            m.q_tf = D100.IdentityTF()
            m.v_neg_shift = False
            m.v_hold = None
        install()
        Kraw, V = D100.capture_keys(model, st, facts, chunks, owner)
        TF = {}
        for li in Kraw:
            TF[li] = D100.build_tf(KEY_ARM, Kraw[li], V[li], li)
            st.mems[li].key_tf = TF[li]
            st.mems[li].q_tf = TF[li].qmap
        KP.write_phase_arm(model, st, chunks, owner,
                           {i: None for i in range(len(facts))})
        W_write = {li: R91.W_of(m).detach().clone() for li, m in st.mems.items()}
        allk = list(range(len(facts)))
        rng = random.Random(555)

        # 守卫 A103-0：white 键两两 |cos| 最大
        mc = 0.0
        for li in Kraw:
            Kt = TF[li](Kraw[li])            # ★ white 臂用 `__call__` 取键（`qmap` 只给查询）
            Kn = Kt / Kt.norm(dim=1, keepdim=True).clamp_min(1e-12)
            G = Kn @ Kn.T
            G.fill_diagonal_(0.0)
            mc = max(mc, float(G.abs().max()))
        res["maxcos"] = mc

        with_y = arm in ("y", "cy")
        with_c = arm in ("c", "cy")
        set_wc(st, make=with_c)
        set_inp(arm)
        ls, viol, div = train_read_v2(model, st, tok, facts, allk, rng,
                                      with_y=with_y, with_c=with_c)
        acc, nq = eval_acc(model, st, tok, facts, allk)

        # A103-3：读训练不得改 W
        dW = max(float((R91.W_of(m).detach() - W_write[li]).norm()
                       / W_write[li].norm().clamp_min(1e-30))
                 for li, m in st.mems.items())
        res.update({"acc_all32": acc, "nq": nq, "loss_tail": ls[-20:],
                    "viol": viol, "div": div, "W_drift": dW, "train_steps": len(ls)})

        # 机制臂 cy_W0：W ← W0，其余参数（Wq/Wo/Wc/gamma）**不动**
        if arm == "cy":
            for li, m in st.mems.items():
                if m.use_factorized:
                    m.U.data.zero_()
                    m.V.data.zero_()
                else:
                    m.Wfull.data.copy_(m.W0.data)
            dW0 = max(float((R91.W_of(m).detach() - m.W0.detach()).norm()
                            / m.W0.detach().norm().clamp_min(1e-30))
                      for m in st.mems.values())
            acc0, nq0 = eval_acc(model, st, tok, facts, allk)
            res.update({"acc_W0": acc0, "W0_rel": dW0})
    finally:
        set_inp("cy")
        clear_wc(st)
        set_capture(False)
        restore()
        st.remove()
        torch.cuda.empty_cache()
    res["secs"] = round(time.time() - t0, 1)
    return res


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stageA",
                    choices=["stageA", "smoke", "formal", "adjudicate"])
    ap.add_argument("--arms", default="", help="逗号分隔，覆盖 ARMS")
    ap.add_argument("--seeds", default="", help="逗号分隔，覆盖 SEEDS")
    ap.add_argument("--from", dest="from_json", default="",
                    help="adjudicate：从既有 stage_*.json 重算判决（不重跑格子）")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    # ★ adjudicate：只重算判决（DIR-102 的运维教训：判决必须能离线重算）
    if a.stage == "adjudicate":
        assert a.from_json, "adjudicate 需要 --from <stage_*.json>"
        src = json.loads(pathlib.Path(a.from_json).read_text(encoding="utf-8"))
        RES["cells"] = src["cells"]
        RES["guards"] = src.get("guards", {})
        print(f"[adjudicate] 载入 {a.from_json}：{len(RES['cells'])} 格", flush=True)
        verdict_main(None)
        return

    tok, model = KP.model_and_tok()
    model.eval()
    facts = R91.build_facts(tok)
    chunks, owner = R91.build_chunks(tok, facts)
    arms = [x for x in a.arms.split(",") if x] or ARMS
    seeds = [int(x) for x in a.seeds.split(",") if x] or (
        SEEDS if a.stage == "formal" else SEEDS[:1])

    # ---------------- Stage A（手算 + 不需训练的守卫）
    if a.stage == "stageA":
        toy_example()
        # A103-4：c 捕获正确性（在 seed[0] 的**未训练**栈上直接测）
        seed0 = seeds[0]
        st = R91.make_stack(gamma=0.0, no_write=False, seed=seed0)
        try:
            install()
            for m in st.mems.values():          # ★ `q_tf`/`key_tf` 由 run_cell 才设，probe 必须自己设
                m.key_tf = D100.IdentityTF()
                m.q_tf = D100.IdentityTF()
                m.v_neg_shift = False
                m.v_hold = None
            set_wc(st, make=True)
            set_inp("cy")
            st.mems[min(st.mems)].gamma.data.fill_(1.0)
            probe = torch.arange(4 * 6, dtype=torch.float32, device=DEV).reshape(4, 6, 1)
            probe = probe.repeat(1, 1, st.mems[min(st.mems)].d_model)
            h = probe.clone()
            set_capture(True)
            _ = st.mems[min(st.mems)].read(h)
            cap = RES.setdefault("_c_cap", None)
            got = globals().get("_C_OUT")
            Bn, Ln = h.shape[0], h.shape[1]
            idx = torch.arange(Bn, device=DEV) * Ln + (Ln - 2)
            want = h.reshape(-1, h.shape[-1])[idx]
            dcap = float((got - want).abs().max())
            RES["_c_cap"] = dcap
            rec("A103-4", dcap == 0.0,
                f"c 捕获：captured 与 sg(hs)[读出位] 逐位最大差 = {dcap:.1e}（须 =0）", dcap)
            set_capture(False)
        finally:
            set_inp("cy")
            clear_wc(st)
            restore()
            st.remove()
            torch.cuda.empty_cache()
        RES["stage"] = "stageA"
        (OUT / "stage_stageA.json").write_text(
            json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n[Stage A] 手算/捕获守卫完成；其余守卫（A103-0/1/2/3/5/6/8）需读数 ⇒ "
              "见 smoke/formal", flush=True)
        return

    # ---------------- smoke / formal：跑格
    cells = []
    for seed in seeds:
        for arm in arms:
            print(f"\n{'='*88}\n[{seed} arm={arm}]\n{'='*88}", flush=True)
            c = run_cell(model, tok, facts, seed, arm, a.stage, chunks, owner)
            print(f"[{seed} arm={arm}] acc_all32={c['acc_all32']:.4f} "
                  f"W_drift={c['W_drift']:.2e}"
                  + (f" acc_W0={c['acc_W0']:.4f}" if "acc_W0" in c else "")
                  + f" {c['secs']}s", flush=True)
            cells.append(c)
            RES["cells"] = cells
            (OUT / f"stage_{a.stage}.json").write_text(
                json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"stage_{a.stage}.json").write_text(
        json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")

    if a.stage == "formal":
        verdict_main(model)


def verdict_main(model):
    """判决（卡片 §1.4 + §1.6 封闭规则）。**阈值一字不改**。"""
    C = RES["cells"]
    by = {}
    for c in C:
        by.setdefault(c["arm"], {})[c["seed"]] = c
    seeds = sorted({c["seed"] for c in C})
    RES["arms_seen"] = sorted(by)

    n_expect = len(set(by)) * len(seeds)
    rec("A103-7", len(C) == n_expect,
        f"格数 {len(C)}/{n_expect}（arm={sorted(by)} × seed={seeds}）")

    if "y" in by:
        v = [by["y"][s]["acc_all32"] for s in seeds if s in by["y"]]
        rec("A103-2", all(BAND_LO <= x < BAND_HI for x in v),
            f"[y 臂满秩] acc_all32 = {[f'{x:.4f}' for x in v]}（须落在 [{BAND_LO},{BAND_HI})）")

    mc = max((c.get("maxcos", 0.0) for c in C), default=9.9)
    rec("A103-0", mc < 1e-4, f"white 键两两 |cos| 最大 = {mc:.2e}（须 <1e-4）")

    dw = max((c.get("W_drift", 9.9) for c in C), default=9.9)
    vi = sum(c.get("viol", 0) for c in C)
    rec("A103-3", dw == 0.0 and vi == 0,
        f"读训练期 W 漂移最大 = {dw:.3e}（须 =0）、写侧梯度违规 {vi}（须 0）")

    stp = [c.get("train_steps") for c in C]
    div = any(c.get("div") for c in C)
    rec("A103-8", all(x == R91.READ_STEPS for x in stp) and not div,
        f"每格步数 {sorted(set(stp))}（须全 ={R91.READ_STEPS}）、diverged={div}")

    if "c" in by:
        v = [by["c"][s]["acc_all32"] for s in seeds if s in by["c"]]
        # ★ rev 1.1 §8.2：`A103-6` 由阻断守卫降级为**只记录**（前提守卫被实验证伪，
        #   先例 `A102-2`/`A102-2p`）。读数原样保留、阈值文字不变、判据不受影响。
        ok6 = all(x <= NEG_MAX for x in v)
        RES["guards"]["A103-6"] = {
            "ok": True, "report_only": True, "passed": bool(ok6),
            "detail": f"[负对照 c-only，rev1.1 只记录] acc_all32 = "
                      f"{[f'{x:.4f}' for x in v]}（rev1.0 期望 ≤{NEG_MAX}；"
                      f"{'达标' if ok6 else '超上界 ⇒ 前提 §2.3 过强，见卡片 §8.2'}）",
            "value": v}
        print(f"  {'✓' if ok6 else '○'} A103-6      "
              f"[负对照 c-only，只记录] acc_all32 = {[f'{x:.4f}' for x in v]}"
              f"（期望 ≤{NEG_MAX}；{'达标' if ok6 else '超界 ⇒ 见 §8.2'}）", flush=True)

    if "cy" in by:
        w0 = [by["cy"][s].get("W0_rel", 9.9) for s in seeds if s in by["cy"]]
        rec("A103-5", all(x == 0.0 for x in w0),
            f"[cy_W0] ‖W−W0‖_F/‖W0‖_F = {[f'{x:.1e}' for x in w0]}（须全 =0）")

    rec_only("A103-1", "写后内容 N_hit^v 由 D-128 判定（white 臂 31–32/32）；"
                       "本卡不重测（键臂与写规则未改）", None)

    # ---------------- 判据
    def g(arm, key, s):
        return by.get(arm, {}).get(s, {}).get(key)

    guards_ok = all(v["ok"] for v in RES["guards"].values())
    if not guards_ok:
        RES["verdict"] = "C103-BLOCKED"
        RES["K"] = None
        print("[判决] C103-BLOCKED（Stage A 守卫未全过）", flush=True)
        _dump_verdict()
        return
    if "cy" not in by or len(by["cy"]) < len(seeds):
        RES["verdict"] = "C103-INCOMPLETE"
        print(f"[判决] C103-INCOMPLETE（cy 臂缺格：{sorted(by.get('cy', {}))}）", flush=True)
        _dump_verdict()
        return

    cy = [g("cy", "acc_all32", s) for s in seeds]
    c_only = [g("c", "acc_all32", s) for s in seeds] if "c" in by else None
    cy0 = [g("cy", "acc_W0", s) for s in seeds]

    if not all(x is not None and x >= THR_ROUTE for x in cy):
        RES["verdict"] = "C103-NOT-REACHED"
    elif c_only is not None and all(x is not None and x <= NEG_MAX for x in c_only):
        # ★ §1.6 封闭：cy_W0 ≥ W0_MIN ⇒ BYPASS（0.90 是充分下界，不需更高）
        if all(x is not None and x >= W0_MIN for x in cy0):
            RES["verdict"] = "C103-BYPASS"
        elif all(x is not None and x < W0_MIN for x in cy0):
            RES["verdict"] = "C103-ROUTE"
        else:
            RES["verdict"] = "C103-MIXED-W0"
    elif c_only is not None and any(x is not None and x > NEG_MAX for x in c_only):
        RES["verdict"] = "C103-VALUE-IN-C"
    else:
        RES["verdict"] = "C103-NO-NEG-CTRL"

    RES["readings"] = {"cy": cy, "c_only": c_only, "cy_W0": cy0,
                       "y_only": [g("y", "acc_all32", s) for s in seeds]}
    print(f"[判决] {RES['verdict']} · cy={cy} c_only={c_only} cy_W0={cy0} "
          f"y={RES['readings']['y_only']}", flush=True)
    _dump_verdict()


def _dump_verdict():
    p = OUT / ("stage_adjudicate.json" if RES.get("stage") == "adjudicate"
               else "stage_verdict.json")
    p.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] → {p}", flush=True)


if __name__ == "__main__":
    main()