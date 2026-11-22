"""DIR-102 · R1_task102 —— 读出参数**压缩下限** `K_min`（`§7.2-2` 的「$\\mathcal O_l$ 参数共享」）

**逐字派生**自 `dir101/R1_task101.py`（`D-131`）。只改**注入参数化的这一层**：

    D-131：`Wo = U Vᵀ`（`rank ≤ r`），注入 `m = Wo·y` —— **线性于 `y`**
    本卡  ：`table[code_i]`（`K` 行任意向量），注入 `m_i = table[code_i]`，`code_i = i mod K`
            —— **对 `y` 完全无约束**（不是 `y` 的函数）

其余（白键臂 `KEY_ARM="white"`、`capture_keys`/`build_tf`/`write_phase_arm` 的写侧序列、
`make_eval_items`/`Random(555)` 题集、`acc_all32` 定义、`W_drift` 口径）**全部照抄**。

★ **装置零改动**：完全复用 `dir101` 的 `_read_r101`/`_write_tf`/`_m_override`。
  码本约束**只**体现在参数共享上 —— 同一码的事实在优化时取**同一个** `tab[ci]`
  （`D101.set_m` 本来就支持"整批一个向量"，而 `optimise_free` 是**逐事实**一次前向 ⇒ 天然够用）。
  ⇒ 无需逐行覆盖（首版写过 `_m_rows`，在该结构下用不上，已删除）。

★ 为什么 `K=32` 必须逐位复现 `A101-1` 的 `1.0000`（本卡 `A102-1` 守卫）：
  `code_i = i mod 32 = i` ⇒ 每事实独享一行、参数互不共享；
  第 `i` 行的梯度**只**来自事实 `i`（各事实 prompt 不同 ⇒ 前向互不影响），
  且 Adam 是**逐参数**状态 ⇒「32 行联合优化」与「逐事实分别优化」**数学等价**
  （仅浮点求和次序可能不同）⇒ 读数必须相同，不等即实现错。
"""
import argparse
import json
import pathlib
import random
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_SIB = _HERE.parent
for p in (_SIB / "dir91", _SIB / "dir100", _SIB / "dir101", _SIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import probe91_kpool as KP                       # noqa: E402
import R2_task91 as R91                          # noqa: E402
import R1_task100 as D100                        # noqa: E402
import R1_task101 as D101                        # noqa: E402
import torch                                     # noqa: E402
from sgmn_module import sg                       # noqa: E402

# ---------------------------------------------------------------- §4 冻结网格
KGRID = [1, 2, 4, 8, 16, 32]
SEEDS = [20261028, 7777]
DEV = R91.DEV
OUT = _HERE / "out"
KEY_ARM = "white"          # ★ 与 D-131 同一键臂（D-128：32/32 内容）

# ★★ rev 1.1（§10.4）：自变量由「码本行数 K」改为「无跨层再入条件下的独立向量数 K_eff」。
#    `[8, 12, 16]` = rev 1.0 的原栈（三层，低层注入会被后两层经 c=sg(h) 再读，DEF-102-02）；
#    `[8]`        = 单层栈，m 无法被任何后续层再读 ⇒ 此时 K 才真正等于自由度数目。
MOUNT_TRIPLE = [8, 12, 16]
MOUNT_SINGLE = [8]

# 判据（冻结，卡片 §1.3）
THR_HI = 0.99
THR_MID = 0.90
K_SHARE_MAX = 8
# ★ rev 1.2 §11.5-2：通道守卫阈值（**新增**，不改任何既有阈值）
CH_SOLO_MAX = 0.50        # 任一单通道 ≤0.50（不足以解释 all）

RES = {"card": "DIRECTION_rev102_v10.md", "guards": {}, "reports": {},
       "cells": [], "channel": None}


def _grad(n):
    return "\u2500" * n


def rec(cid, ok, detail, value=None):
    RES["guards"][cid] = {"ok": bool(ok), "detail": detail, "value": value}
    print(f"  {'\u2713' if ok else '\u2717'} {cid:<10} {detail}", flush=True)


# ==================================================================== 码本装置
def code_of(i, K):
    """★ 冻结规则（卡片 §2.1）：`code_i = i mod K`。
    **与答案值无关**：`build_facts` 先造 key、再 `rng.shuffle(vals)` ⇒ 下标与值独立。"""
    return i % K


def optimise_codes(model, st, tok, facts, keys, rng, K, steps=D101.STEPS_FREE,
                   lr=D101.LR_FREE):
    """参数 = `table ∈ R^{K×d_model}`（每层一份）；注入 `m_fi = table[code_of(fi)]`。

    ★ **成本与 `dir101.optimise_free` 完全相同**：`epochs × 每事实一次前向`
      （`STEPS_FREE=120` epoch × 32 事实 = 3840 次 batch-4 前向）。
      首版曾"每步在全部 96 题上求损失和"⇒ 600 步 × 6 个长度组 × 每批 ~64 行
      ≈ `D-131` 的 10⁴ 倍 token 量（实测 15 min 仍未完成一格，13.2 GB 显存）。**已弃**。

    ★ 为何 `K=32` 与 `optimise_free` **数学等价**（`A102-1` 守卫的依据）：
      `code_i = i` ⇒ 每事实独享一行；模型冻结 ⇒ 第 `i` 行的梯度只由事实 `i` 决定；
      Adam 逐参数状态互不影响 ⇒ 每行被更新恰好 120 次，与逐事实顺序无关
      （唯一差别是更新次序：epoch-major vs fact-major）。
    """
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    items = R91.make_eval_items(tok, facts, keys, rng)
    by_fi = {}
    for it in items:
        by_fi.setdefault(it["key_idx"], []).append(it)
    try:
        D101.set_m(st, None)
        # 预先备好每个事实的 batch（与 optimise_free 逐字同构）
        batch = {}
        for fi in keys:
            its = by_fi[fi]
            P = len(its[0]["prompt_ids"])
            nc = len(its[0]["cand"])
            seqs = [it["prompt_ids"] + [c[0][0]] for it in its for c in it["cand"]]
            batch[fi] = (
                torch.tensor(seqs, dtype=torch.long, device=DEV),
                torch.tensor([s[-1] for s in seqs], device=DEV),
                torch.tensor([next(j for j, c in enumerate(it["cand"]) if c[1])
                              for it in its], dtype=torch.long, device=DEV),
                P - 1, nc, len(its))
        tab = {li: torch.zeros(K, st.mems[li].d_model, device=DEV, requires_grad=True)
               for li in st.mems}
        opt = torch.optim.Adam(tab.values(), lr=lr)
        loss_hist = []
        # ★ `main()` 里 `torch.set_grad_enabled(False)`（评测口径）⇒ 必须显式重开
        for _ in range(steps):
          with torch.enable_grad():
            tot = 0.0
            for fi in keys:
                seq, tok_ids, tgt, Pl, nc, nq = batch[fi]
                ci = code_of(fi, K)
                opt.zero_grad(set_to_none=True)
                D101.set_m(st, {li: tab[li][ci] for li in st.mems})
                lg = model(input_ids=seq).logits.float()
                pick = lg[torch.arange(len(seq), device=DEV), Pl, tok_ids].reshape(nq, nc)
                loss = torch.nn.functional.cross_entropy(pick, tgt)
                loss.backward()
                opt.step()
                tot += float(loss)
            loss_hist.append(tot / len(keys))
        Mv = {li: {} for li in st.mems}
        for fi in keys:
            ci = code_of(fi, K)
            for li in st.mems:
                Mv[li][fi] = tab[li][ci].detach().clone()
        return Mv, {"K": K, "head": float(sum(loss_hist[:20]) / 20),
                    "tail": float(sum(loss_hist[-20:]) / 20), "nq": len(items),
                    "epochs": steps}
    finally:
        D101.set_m(st, None)
        for li, m in st.mems.items():
            m.no_write = saved[li]


def build_stack_white(model, tok, facts, chunks, owner, seed, mount=None):
    """= `dir101.run_cell` 的**写侧序列**，逐字（白键臂 + 600 块真写）。

    `mount` 为 None ⇒ 用 `R91.make_stack`（= `R91.MOUNT = [8,12,16]`，rev 1.0 原栈）；
    给了 `mount` ⇒ 直接 `SGMNStack(model, mount, ...)`，**不改** `R2_task91.MOUNT` 常量。
    """
    if mount is None:
        st = R91.make_stack(gamma=0.0, no_write=False, seed=seed)
    else:
        from sgmn_module import SGMNStack
        d = model.config.hidden_size
        st = SGMNStack(model, list(mount), d, d, R91.R, R91.ALPHA, R91.LAM,
                       R91.ETA_WRITE, 0.0, R91.DEV, seed=seed, use_factorized=False,
                       drop_sg=False, no_write=False)
    for m in st.mems.values():
        m.key_tf = D100.IdentityTF()
        m.q_tf = D100.IdentityTF()
        m.v_neg_shift = False
        m.v_hold = None
    D101.install()
    Kraw, V = D100.capture_keys(model, st, facts, chunks, owner)
    for li in Kraw:
        TF = D100.build_tf(KEY_ARM, Kraw[li], V[li], li)
        st.mems[li].key_tf = TF
        st.mems[li].q_tf = TF.qmap
    KP.write_phase_arm(model, st, chunks, owner, {i: None for i in range(len(facts))})
    return st


def run_cell(model, tok, facts, seed, K, chunks, owner, mount=None):
    t0 = time.time()
    st_mount = MOUNT_SINGLE if mount == "single" else (
        MOUNT_TRIPLE if mount == "triple" else list(R91.MOUNT))
    torch.manual_seed(seed)
    st = build_stack_white(model, tok, facts, chunks, owner, seed, st_mount)
    try:
        W_write = {li: R91.W_of(m).detach().clone() for li, m in st.mems.items()}
        allk = list(range(len(facts)))
        D101.set_gamma(st, 1.0)          # ★ DEF-101-05：gamma=0 会把注入整体乘零
        Mv, ls = optimise_codes(model, st, tok, facts, allk, random.Random(555), K)
        acc = D101.eval_free(model, st, tok, facts, allk, Mv)
        dW = max(float((R91.W_of(m).detach() - W_write[li]).norm()
                       / W_write[li].norm().clamp_min(1e-30))
                 for li, m in st.mems.items())
    finally:
        D101.restore()
        st.remove()
        torch.cuda.empty_cache()
    mt = list(st_mount)
    RES["cells"].append({"seed": seed, "K": K, "mount": mt, "acc_all32": acc,
                         "W_drift": dW, "loss": ls, "secs": round(time.time() - t0, 1)})
    print(f"[{seed} K={K:>2} mount={mt}] acc_all32={acc:.4f} W_drift={dW:.2e} "
          f"{time.time() - t0:.1f}s", flush=True)
    return acc


# ==================================================================== 守卫
def toy_example(n_facts):
    """`A102-3`/`A102-6`：码分配手算（卡片 §2.3），直接调 `code_of`。"""
    K = 2
    ok1 = (code_of(0, K) == 0 and code_of(2, K) == 0 and code_of(1, K) == 1)
    ok2 = all(code_of(i, 1) == 0 for i in range(n_facts))
    ok3 = all(code_of(i, 32) == i for i in range(n_facts))
    bal = {}
    for Kx in KGRID:
        if n_facts % Kx:
            continue
        cnt = {}
        for i in range(n_facts):
            cnt[code_of(i, Kx)] = cnt.get(code_of(i, Kx), 0) + 1
        bal[Kx] = sorted(set(cnt.values()))
    return ok1, ok2, ok3, bal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stageA",
                    choices=["stageA", "formal", "smoke", "adjudicate"])
    ap.add_argument("--ks", default="", help="逗号分隔，覆盖 KGRID")
    ap.add_argument("--seeds", default="", help="逗号分隔，覆盖 SEEDS")
    ap.add_argument("--from", dest="from_json", default="",
                    help="adjudicate 阶段：从既有 stage_*.json 重算判决（不重跑格子）")
    ap.add_argument("--channel", default="",
                    help="注入通道读数 JSON（来自 diag102_channel.py）供 A102-7 判定")
    ap.add_argument("--mount", default="triple",
                    choices=["single", "triple", "both"],
                    help="single=[8]（无跨层再入，rev1.1 的 A102-2'）/ triple=[8,12,16]（rev1.0 原栈）/ both")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.channel:
        RES["channel"] = json.loads(pathlib.Path(a.channel).read_text(encoding="utf-8"))
    if a.stage == "adjudicate":
        # ★ 只重算判决：读数、守卫、通道全部沿用既有 JSON；阈值一字不动。
        assert a.from_json, "adjudicate 需要 --from <stage_*.json>"
        src = json.loads(pathlib.Path(a.from_json).read_text(encoding="utf-8"))
        for k in ("cells", "channel", "guards"):
            if src.get(k):
                RES[k] = src[k]
        print(f"[adjudicate] 载入 {a.from_json}：{len(RES['cells'])} 格、"
              f"{len(RES['guards'])} 守卫", flush=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    tok, model = KP.model_and_tok()
    model.eval()
    facts = R91.build_facts(tok)
    chunks, owner = R91.build_chunks(tok, facts)
    ks = [x for x in a.ks.split(",") if x] or KGRID
    ks = [int(x) if isinstance(x, str) and x.isdigit() else x for x in ks]  # DEF-101-08 教训
    mounts = ["triple", "single"] if a.mount == "both" else [a.mount]
    seeds = [int(x) for x in a.seeds.split(",") if x] or (
        SEEDS if a.stage == "formal" else SEEDS[:1])

    print(_grad(88))
    print(f"DIR-102 · stage={a.stage} · K={ks} · seeds={seeds} · mount={mounts}")
    print(_grad(88))

    ok1, ok2, ok3, bal = toy_example(len(facts))
    rec("A102-3", ok1 and ok2 and ok3,
        f"码分配手算：code_0==code_2==0、code_1==1 ✓；K=1 全 0 ✓；K=32 code_i==i ✓；均衡 {bal}")
    rec("A102-6", ok1 and ok2 and ok3, "手算算例（§2.3 四条断言）直接调 code_of 全过")

    if a.stage != "adjudicate":          # ★ adjudicate 只重算判决，不重跑格子
        for seed in seeds:
            for mt in mounts:
                for K in ks:
                    run_cell(model, tok, facts, seed, K, chunks, owner, mt)
    else:
        # 从既有读数反推 ks/seeds/mounts，保证打印表格与判据一致
        ks = sorted({c["K"] for c in RES["cells"]})
        seeds = sorted({c["seed"] for c in RES["cells"]})
        _mt = {tuple(c["mount"]) for c in RES["cells"]}
        mounts = ["single" if m == tuple(MOUNT_SINGLE) else "triple" for m in sorted(_mt)]

    by = {}
    for c in RES["cells"]:
        by.setdefault((tuple(c["mount"]), c["K"]), {})[c["seed"]] = c["acc_all32"]
    byk = {}                              # 兼容旧口径：仅 triple 臂
    for c in RES["cells"]:
        if tuple(c["mount"]) == tuple(MOUNT_TRIPLE):
            byk.setdefault(c["K"], {})[c["seed"]] = c["acc_all32"]
    print(_grad(88))
    for mt in mounts:
        key = tuple(MOUNT_SINGLE if mt == "single" else MOUNT_TRIPLE)
        print(f"-- mount={list(key)} --")
        print(f"{'K':>4} " + " ".join(f"{s:>10}" for s in seeds))
        for K in ks:
            print(f"{K:>4} " + " ".join(
                f"{by.get((key, K), {}).get(s, float('nan')):>10.4f}" for s in seeds))

    n_expect = len(ks) * len(seeds) * len(mounts)   # ★ 修正：加 `--mount both` 后未同步计数
    rec("A102-5", len(RES["cells"]) == n_expect,
        f"{len(RES['cells'])}/{n_expect} 格有读数（K={len(ks)} × seed={len(seeds)} × mount={len(mounts)}）")
    dmax = max((c["W_drift"] for c in RES["cells"]), default=float("nan"))
    rec("A102-4", dmax == 0.0, f"读训练期 W 漂移最大 = {dmax:.3e}（须 =0，A97-4）")

    # ★ rev 1.1 §10.4：原 `A102-1`/`A102-2` 的冻结文字只针对 rev 1.0 的 `[8,12,16]` 栈，
    #   故只在 triple 臂上断言（文字与阈值一字未改）；single 臂上新开 `A102-2'`/`A102-1'`。
    a1 = byk.get(1, {})
    if a1:
        # ★ rev 1.2 §11.5-1：`A102-2` **作废**（同 `A102-2p`），改为只记录。
        RES["reports"]["A102-2(作废)"] = (
            f"[triple] K=1 的 acc_all32 = {['%.4f' % v for v in a1.values()]}"
            f"（rev1.2 已作废：常数注入有效，用途是引导，非装置错误）")
        print(f"  ○ A102-2(作废) {RES['reports']['A102-2(作废)']}", flush=True)
    a32 = byk.get(32, {})
    if a32:
        ok = all(abs(v - 1.0) < 1e-12 for v in a32.values())
        rec("A102-1", ok,
            f"[triple] K=32 的 acc_all32 = {['%.4f' % v for v in a32.values()]}"
            f"（须逐位 1.0000，复现 A101-1 的 free 臂）")
    s1 = by.get((tuple(MOUNT_SINGLE), 1), {})
    if s1:
        # ★ rev 1.2 §11.5-1：`A102-2p` **作废**（其前提"常数注入应当无用"已被 §11.2/11.3 证伪）。
        #   改为**只记录**，不计入 `guards_ok`。
        RES["reports"]["A102-2p(作废)"] = (
            f"[single] K=1 的 acc_all32 = {['%.4f' % v for v in s1.values()]}"
            f"（rev1.2 已作废此守卫：常数确实有用，用途是引导）")
        print(f"  ○ A102-2p(作废) {RES['reports']['A102-2p(作废)']}", flush=True)

    # ★ rev 1.2 §11.5-2：`A102-7` 通道守卫（读数由 `diag102_channel.py` 提供）
    ch = RES.get("channel")
    if ch:
        rall, rd, nr, nn = (ch.get(k) for k in
                            ("all", "readout_only", "nonreadout_only", "none"))
        ok = (rd is not None and nn is not None and rd > nn
              and nr is not None and rall is not None and nr < rall
              and min(rd, nr) <= CH_SOLO_MAX)
        rec("A102-7", ok,
            f"通道：all={rall:.4f} readout_only={rd:.4f} nonreadout_only={nr:.4f} "
            f"none={nn:.4f}（须 readout_only>none、nonreadout_only<all、单通道≤{CH_SOLO_MAX}）")

    guards_ok = all(g["ok"] for g in RES["guards"].values())
    if a.stage in ("formal", "adjudicate") and guards_ok:
        # ★ rev1.2 修 bug（不改任何阈值）：`by` 的键是 `(tuple(mount), K)`（第 270 行），
        #   原代码用 `by.get(32, {})` / `by[K]` ⇒ 恒取不到 ⇒ v32=[None,…] ⇒ 误判 C102-BLOCKED。
        #   判决只在**单一 mount** 下有定义（K_min 依赖挂载集合）；formal 轮只跑 triple。
        mt = sorted({tuple(c["mount"]) for c in RES["cells"]})
        RES["verdict_mount"] = [list(m) for m in mt]
        if len(mt) != 1:
            RES["verdict"] = "C102-MULTIMOUNT"
            RES["K_min"] = None
            print(f"[判决] C102-MULTIMOUNT（formal 轮出现 {len(mt)} 个挂载，"
                  f"K_min 只在单挂载下有定义）· mounts={[list(m) for m in mt]}", flush=True)
        else:
            m0 = mt[0]
            g = lambda K: by.get((m0, K), {})
            v32 = [g(32).get(s) for s in seeds]
            v16 = [g(16).get(s) for s in seeds]
            share = [K for K in ks if K <= K_SHARE_MAX and g(K)
                     and all(g(K).get(s, 0.0) >= THR_HI for s in seeds)]
            if any(v is None or v < THR_HI for v in v32):
                RES["verdict"] = "C102-BLOCKED"
            elif share:
                RES["verdict"] = "C102-SHARED"
                RES["share_ks"] = share
            elif all(v >= THR_HI for v in v32) and all(v < THR_MID for v in v16):
                RES["verdict"] = "C102-NOSHARE"
            else:
                RES["verdict"] = "C102-PARTIAL"
            RES["K_min"] = next((K for K in sorted({K for (_, K) in by})
                                 if all(g(K).get(s, 0.0) >= THR_HI for s in seeds)), None)
            print(f"[判决] {RES['verdict']} · K_min=({list(m0)}, {RES['K_min']})"
                  f" · share_ks={RES.get('share_ks')}", flush=True)
    elif a.stage == "formal":
        RES["verdict"] = "BLOCKED"
        print("[判决] BLOCKED（Stage A 守卫未全过）", flush=True)
    else:
        RES["verdict"] = "STAGE_A_ONLY"

    p = OUT / f"stage_{a.stage}.json"
    p.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[完成] → {p}")


if __name__ == "__main__":
    main()