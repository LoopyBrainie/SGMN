"""`precheck121.py` —— `DIR-121` 的**开卡前结构核**（★ 在冻结阈值**之前**跑）。

★ 目的（`DEF-106-10` / `DEF-106-13` 的教训）：**先证"待测量在结构上存在方差"**，
  再冻结判据。否则会造出「不可能失败」或「不可能成立」的死臂。

★ 本脚本**刻意只打印结构事实**：
  - 表形状、行/列边缘是否**非退化**（有方差、无全 0 列等）；
  - **已发表常数**的复现（`D-151` 的 `42` / `414` / 逐 seed 逃逸数）；
  - **不打印**任何将作为判据的统计量取值（离散指数、permutation p）⇒ **阈值不可能被结果倒推**。

★ 与正式轮的**唯一**差别：本脚本只读、只打印，不写 `out/`。
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
for _d in (str(_HERE), str(_HERE.parent), str(_HERE.parent / "dir120")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import R1_task120 as P120  # noqa: E402
import mech120 as M  # noqa: E402


def main():
    D = P120.load()
    seeds = D["seeds"]
    n_items = D["n_items"]
    n_seed = len(seeds)
    print(f"[precheck121] seeds={n_seed}  items/seed={n_items}  seeds={seeds}")

    # ---- 逐 (seed, item) 的逃逸布尔 + n_seen ----
    rows = []
    for k, s in enumerate(seeds):
        lo, hi = k * n_items, (k + 1) * n_items
        pick = D["pick"][lo:hi]
        meta = D["meta"][lo:hi]
        tgt = D["tgt"][lo:hi]
        tv = D["tv"]
        esc, ns = [], []
        for i in range(n_items):
            cv = meta[i]["cand_val"]
            ti = int(tgt[i])
            seen = M._n_seen_of(cv, ti, tv)
            ns.append(seen)
            picked_seen = cv[int(pick[i])] in tv
            esc.append(bool(seen >= 1 and not picked_seen))
        rows.append({"seed": s, "esc": esc, "ns": ns})

    # ---- 结构事实 1：n_seen 组成是否逐 seed 相同（应为真 ⇒ 组成不可解释分裂）----
    comps = {tuple(r["ns"]) for r in rows}
    print(f"\n[结构 1] n_seen 向量逐 seed 不同取值个数 = {len(comps)}"
          f"（=1 ⇒ 组成**不可**解释 seed 分裂；这是结构事实，不是判据）")

    # ---- 结构事实 2：保存哪些列（n_seen ≥ 1），与已发表常数对照 ----
    has = [i for i in range(n_items) if rows[0]["ns"][i] >= 1]
    print(f"[结构 2] n_seen>=1 的题号个数 = {len(has)}  × {n_seed} seed = {len(has)*n_seed}"
          f"（`D-151` 已发表 414 ⇒ {'一致' if len(has)*n_seed == 414 else '不一致！'}）")

    # ---- 结构事实 3：行边缘（逐 seed 逃逸数）—— 已发表常数，复现即护栏 ----
    rt = [sum(r["esc"][i] for i in has) for r in rows]
    print(f"[结构 3] 逐 seed 逃逸数 = {rt}  (合计 {sum(rt)}；`D-151` 已发表 "
          f"[0, 3, 18, 0, 2, 9, 5, 0, 5]、合计 42 ⇒ {'一致' if sum(rt) == 42 else '不一致！'})")

    # ---- 结构事实 4：列边缘是否退化（决定两向分解**是否可算**）----
    ct = [sum(rows[k]["esc"][i] for k in range(n_seed)) for i in has]
    hist = {v: ct.count(v) for v in sorted(set(ct))}
    print(f"[结构 4] 列边缘（每个题在 {n_seed} 个 seed 上的逃逸次数）直方图 = {hist}")
    print(f"         → 全 0 列 = {ct.count(0)}，全 {n_seed} 列 = {ct.count(n_seed)}，"
          f"严格中间列 = {sum(1 for v in ct if 0 < v < n_seed)}")
    print(f"         → 行边缘方差 = {'有' if len(set(rt)) > 1 else '无（退化！）'}；"
          f"列边缘方差 = {'有' if len(set(ct)) > 1 else '无（退化！）'}")

    # ---- 结构事实 5：逐 seed 的 n_seen 分层题数（供卡片 §3 手算例对齐）----
    lv = {n: sum(1 for i in has if rows[0]["ns"][i] == n) for n in (1, 2, 3)}
    print(f"[结构 5] n_seen 分层题数（n_seen>=1 内）= {lv}  合计 {sum(lv.values())}")

    # ---- 结构事实 6：二项参照（齐次零假设下的期望离散度尺度，仅打印尺度不打印观测）----
    n_tot, N = sum(rt), len(has) * n_seed
    p = n_tot / N
    print(f"\n[结构 6] 合并逃逸率 p = {n_tot}/{N} = {p:.6f}；"
          f"行边缘在齐次零假设下的**期望**离散指数 ≈ 1.0（Pearson χ²/df，df={n_seed-1}）"
          f"；列边缘 df={len(has)-1}")
    print("[precheck121] 结构核结束（★ 未计算、未打印任何判据统计量的观测值）")


if __name__ == "__main__":
    main()