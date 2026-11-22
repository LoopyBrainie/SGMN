#!/opt/pytorch/bin/python
r"""DIR-50 · **Stage A 硬门**：任何一条不过 ⇒ **正式轮不得启动**。

断言（卡 rev 1.0 §4/§5）：
- `A50-1` 写入数：每实例每层 12 次、合计 **24**（**数**出来，不是抄常数）
- `A50-2` **工作例（独立第二链）** 3 实例 × 5 条件：`max|Δ| ≤ 1e-9`
- `A50-3` **结构不变量**：臂 `hit2` 的**层 1 轨迹**与 `none` **逐位相同**（层 2 的记忆不反馈回层 1）
- `A50-4` **级联的一步延迟**：臂 `hit1` 的**层 2 在 `d=1`** 与 `none` **逐位相同**
- `A50-5` **剂量精确**：`dose 0 ⇒ |cos(k_int,k_tgt)| ≤ 1e-12`；`dose 1 ⇒ ≥ 0.9999`；且 `‖k_int‖ = ‖k_nat‖` 逐位
- `A50-6` **五臂写入流逐位相同**，**只有**被命中的那一层在干扰步的**键**不同
- `A50-7` 卡 md5 + 装置常数
- `A50-8` 正式段 vs 既往**已落盘** seed **交集 0**
- `A50-9` `rel` 与 `c/c0` 一致（`ATOL=1e-12`）
- `A50-10` Stage A 段的 `n_bad ≤ N_BAD_MAX`

用法：`stageA50.py --out dir50/out/DIR50r10_stageA.json`
"""
import argparse, glob, hashlib, json, math, os, sys

import torch
torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir50"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import dev50 as D            # noqa: E402

CARD = os.path.join(ROOT, "revolve/DIR-50/DIRECTION_rev50_v11.md")
CARD_MD5 = "0be3668d2a6755adf0361b770b6d5005"
STAGE_SEEDS = [31000 + i for i in range(20)]          # 探针段（**不与正式段重合**）
FORMAL = list(range(33000, 37000))
ATOL = 1e-12
WORK_ATOL = 1e-9
N_BAD_MAX = 0.10
CONDS = [("none", None), ("hit1", 0.0), ("hit1", 1.0), ("hit2", 0.0), ("hit2", 1.0)]
REPORT = []


def teq(x, y):
    """NaN-aware 的**张量**逐位相同（`torch.equal` 在 NaN 上为 False）。"""
    if x.shape != y.shape:
        return False
    xx, yy = x.reshape(-1), y.reshape(-1)
    return bool(torch.all((xx == yy) | (torch.isnan(xx) & torch.isnan(yy))))


def same(x, y):
    """NaN-aware 的逐位相同（`NaN == NaN` 在 Python 里为 False ⇒ 必须显式处理）。"""
    return (x == y) or (isinstance(x, float) and isinstance(y, float)
                        and math.isnan(x) and math.isnan(y))


def rep(name, ok, detail):
    REPORT.append({"name": name, "ok": bool(ok), "detail": detail})
    print(("  ✓ " if ok else "  ✗ ") + name + " " + str(detail))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print("=== DIR-50 Stage A ===")
    # ── 跑 Stage A 段
    rows = {}
    for arm, dose in CONDS:
        for sd in STAGE_SEEDS:
            P = D.build(sd)
            cont = D.content(sd)
            rr = D.rollout(P, cont, arm, dose, record_writes=True)
            rows[(sd, arm, dose)] = rr
    # A50-1 写入数
    nw = set()
    for (sd, arm, dose), rr in rows.items():
        nw.add(len(rr["writes"]))
    rep("A50-1", nw == {2 * D.T_STEPS},
        f"每实例写入数集合 = {sorted(nw)}（应为 [{2*D.T_STEPS}]；层数 2 × 步数 {D.T_STEPS}）")
    # A50-2 工作例（独立第二链）
    sys.path.insert(0, os.path.join(ROOT, "dir50"))
    import workex50 as WX
    import subprocess
    wx_out = os.path.join(ROOT, "dir50/out/DIR50_workex.json")
    subprocess.run([sys.executable, os.path.join(ROOT, "dir50/workex50.py"),
                    "--n", "3", "--out", wx_out], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wx = json.load(open(wx_out, encoding="utf-8"))
    rep("A50-2", wx["ok"], f"独立第二链 {wx['n_inst']} 实例 × {wx['n_cmp']} 条件；max|Δ| = {wx['max_abs_diff']:.3e}"
        f"（容差 {WORK_ATOL:.0e}）")
    # A50-3 `hit2` 的层 1 ≡ `none`
    bad3 = 0
    for sd in STAGE_SEEDS:
        b = rows[(sd, "none", None)]
        for dose in (0.0, 1.0):
            r = rows[(sd, "hit2", dose)]
            if not (same(r["c0"][0], b["c0"][0]) and all(same(x, y) for x, y in zip(r["curve"][0], b["curve"][0]))):
                bad3 += 1
    rep("A50-3", bad3 == 0, f"`hit2` 的层 1 轨迹 vs `none` 逐位相同违例 {bad3}/{2*len(STAGE_SEEDS)}"
        f"（层 2 的记忆**不**反馈回层 1 ⇒ 耦合单向）")
    # A50-4 `hit1` 的层 2 在 d=1 ≡ `none`
    bad4 = 0
    for sd in STAGE_SEEDS:
        b = rows[(sd, "none", None)]
        for dose in (0.0, 1.0):
            r = rows[(sd, "hit1", dose)]
            if not (same(r["c0"][1], b["c0"][1]) and same(r["curve"][1][0], b["curve"][1][0])):
                bad4 += 1
    rep("A50-4", bad4 == 0, f"`hit1` 的层 2 在 d=1 vs `none` 逐位相同违例 {bad4}/{2*len(STAGE_SEEDS)}"
        f"（级联的一步延迟：层 2 的键取 h_pre_2[t−1]，早于命中）")
    # A50-5 剂量精确 + 范数匹配
    ov0, ov1, norm_ok = [], [], 0
    for sd in STAGE_SEEDS:
        for arm in ("hit1", "hit2"):
            for dose in (0.0, 1.0):
                rr = rows[(sd, arm, dose)]
                (ov0 if dose == 0.0 else ov1).append(rr["ov_k"])
                # 写入流里该层的干扰步：键的范数应与自然键相同
                w = [x for x in rr["writes"] if x[0] == D.T_HIT and x[1] == (0 if arm == "hit1" else 1)]
                if len(w) == 1:
                    nb = rows[(sd, "none", None)]
                    wn = [x for x in nb["writes"] if x[0] == D.T_HIT and x[1] == w[0][1]]
                    if wn:
                        a_, b_ = float(w[0][2].norm()), float(wn[0][2].norm())
                        if abs(a_ - b_) <= 1e-12 * max(1.0, b_):
                            norm_ok += 1
    rep("A50-5", max(ov0) <= 1e-12 and min(ov1) >= 0.9999 and norm_ok == 4 * len(STAGE_SEEDS),
        f"dose 0 的 |cos| 最大 {max(ov0):.3e}（≤1e-12）；dose 1 的最小 {min(ov1):.9f}（≥0.9999）；"
        f"‖k_int‖ 与 ‖k_nat‖ 相差 ≤1e-12 相对容差的计数 {norm_ok}/{4*len(STAGE_SEEDS)}")
    # A50-6 ★ 写入流的**正确**不变量集合（Stage A run1 证明原断言是错的 ⇒ `DEF-50-3`）：
    #   (a) `hit2` 臂：**两层**的写入流与 `none` 逐位相同，**只有**（T_HIT，层 2）的**键**不同；
    #   (b) `hit1` 臂：**层 1** 的写入流与 `none` 逐位相同，**只有**（T_HIT，层 1）的**键**不同；
    #       而 **层 2** 的写入流从 **step T_HIT+1** 起不同（★ **级联的确切足迹**），之前逐位相同。
    bad6 = 0
    foot6 = 0
    for sd in STAGE_SEEDS:
        b = rows[(sd, "none", None)]
        for arm, dose in CONDS[1:]:
            r = rows[(sd, arm, dose)]
            li_hit = 0 if arm == "hit1" else 1
            diffs = 0
            first_diff_2 = None
            for (t1, l1, k1, v1), (t2, l2, k2, v2) in zip(r["writes"], b["writes"]):
                assert (t1, l1) == (t2, l2)
                same_k = teq(k1, k2)
                same_v = teq(v1, v2)
                if l1 == li_hit:
                    if t1 == D.T_HIT:
                        if not same_v:
                            diffs += 1
                        n1_, n2_ = float(k1.norm()), float(k2.norm())
                        if abs(n1_ - n2_) > 1e-12 * max(1.0, n2_):
                            diffs += 1
                        if same_k:
                            diffs += 1
                    else:
                        if not (same_k and same_v):
                            diffs += 1
                else:
                    # ★ 非被命中层：只允许从 `T_HIT+1` 起不同（级联足迹）
                    if not (same_k and same_v):
                        if t1 > D.T_HIT:
                            if first_diff_2 is None:
                                first_diff_2 = t1
                        else:
                            diffs += 1
            if first_diff_2 is not None and first_diff_2 == D.T_HIT + 1:
                foot6 += 1
            if diffs:
                bad6 += 1
    rep("A50-6", bad6 == 0 and foot6 == len(STAGE_SEEDS) * 2,
        f"写入流不变量违例 {bad6}/{len(STAGE_SEEDS)*4}；★ 级联足迹首个不同步恰为 T_HIT+1 的计数 {foot6}/{len(STAGE_SEEDS)*2}"
        f"（只有 `hit1` 臂有级联足迹 ⇒ 2 剂量 × {len(STAGE_SEEDS)} 实例）"
        f"（★ 级联**只**从 step {D.T_HIT+1} 起进入下游模态的写入流）")
    # A50-7 卡 md5 + 装置常数
    md5 = hashlib.md5(open(CARD, "rb").read()).hexdigest()
    consts_ok = (D.N_REP == 16 and abs(M.ALPHA - 1.0) < 1e-15 and abs(M.LAM - 0.05) < 1e-15
                 and abs(M.ETA - 0.2) < 1e-15 and R.D_H == 8 and R.D_K == 16 and R.D_V == 4
                 and D.CTX == 3 and D.T_HIT == 4 and D.W_AFTER == 7 and D.T_STEPS == 12
                 and D.NMARK == 36 and R.KEY_SCALE == 0.5)
    rep("A50-7", md5 == CARD_MD5 and consts_ok,
        f"卡 md5={md5}；N_REP/ALPHA/LAM/ETA/D_H/D_K/D_V/CTX/T_HIT/W_AFTER/T_STEPS/NMARK/KEY_SCALE="
        f"{D.N_REP}/{M.ALPHA}/{M.LAM}/{M.ETA}/{R.D_H}/{R.D_K}/{R.D_V}/{D.CTX}/{D.T_HIT}/{D.W_AFTER}/{D.T_STEPS}/{D.NMARK}/{R.KEY_SCALE}")
    # A50-8 种子段隔离
    used = set()
    for f in glob.glob(os.path.join(ROOT, "dir*/out/*.jsonl")):
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(o, dict) and isinstance(o.get("seed"), int):
                        used.add(o["seed"])
        except OSError:
            pass
    inter = sorted(set(FORMAL) & used)
    rep("A50-8", not inter, f"正式段 {FORMAL[0]}..{FORMAL[-1]}（{len(FORMAL)}）vs 既往已落盘 seed {len(used)} 个；"
        f"交集 {len(inter)} 个 {inter[:5]}")
    # A50-9 `rel` 与 `c/c0` 一致
    bad9 = 0
    nrel = 0
    for (sd, arm, dose), rr in rows.items():
        for li in (0, 1):
            c0 = rr["c0"][li]
            manual = []
            for c in rr["curve"][li]:
                if math.isfinite(c0) and c0 > 0 and math.isfinite(c):
                    manual.append(c / c0)
                    # ★ 余弦的取值范围检查必须**逐元素**做（`c` 在循环外检查会泄漏成最后一个元素）
                    if not (-1.0000001 <= c <= 1.0000001):
                        bad9 += 1
                else:
                    manual.append(None)
            dev = D.rel_curve(rr, li)
            for x, y in zip(manual, dev):
                if (x is None) != (y is None):
                    bad9 += 1
                elif x is not None:
                    nrel += 1
                    if abs(x - y) > ATOL:
                        bad9 += 1
    rep("A50-9", bad9 == 0, f"`rel`（`dev50.rel_curve` vs 手算 `c/c0`）不一致 {bad9} 处 / 共 {nrel} 个可算读数"
        f"（ATOL={ATOL:.0e}；裁决器将独立重算）")
    # A50-10 n_bad
    nbad = 0
    tot = 0
    nbad_full = 0
    for (sd, arm, dose), rr in rows.items():
        tot += 1
        # ★ 判据量口径（rev 1.1）：判据量集合 = {rel_1(1), rel_2(1), rel_2(2)}
        ok = True
        for li, d in ((0, 1), (1, 1), (1, 2)):
            c0 = rr["c0"][li]
            c = rr["curve"][li][d - 1]
            if not (math.isfinite(c0) and c0 > 0 and math.isfinite(c)):
                ok = False
        if not ok:
            nbad += 1
        # 报告口径：全部 d 的可算率（**不入**判据）
        if any(not (math.isfinite(rr["c0"][li]) and rr["c0"][li] > 0 and math.isfinite(c))
               for li in (0, 1) for c in rr["curve"][li]):
            nbad_full += 1
    rep("A50-10", nbad / tot <= N_BAD_MAX,
        f"Stage A 段 `n_bad`（**判据量**口径）= {nbad}/{tot} = {nbad/tot:.4f} ≤ {N_BAD_MAX}；"
        f"（报告口径：全部 d 任一不可算 {nbad_full}/{tot} = {nbad_full/tot:.4f}，**不入**判据）")
    ok = all(x["ok"] for x in REPORT)
    out = {"card": "DIR-50 rev 1.0", "card_md5": md5, "ok": ok,
           "dev50_md5": hashlib.md5(open(os.path.join(ROOT, "dir50/dev50.py"), "rb").read()).hexdigest(),
           "stage_seeds": [STAGE_SEEDS[0], STAGE_SEEDS[-1]],
           "formal": [FORMAL[0], FORMAL[-1]], "assertions": REPORT}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(("★ Stage A 全部通过" if ok else "✗ Stage A 失败 ⇒ 正式轮不得启动"))


if __name__ == "__main__":
    main()
