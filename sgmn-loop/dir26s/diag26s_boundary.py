#!/opt/pytorch/bin/python
r"""DIR-26-S′ · **崩溃行复现 + 边界扫描**（诊断，**不产生判据、不写行**）。

★ **背景**：rev 2.0 正式轮在第 90 行为 `measure_hit` 的断言所杀：
```
assert (m["mu"] > 0) == (m["ret1"] > 0.5), f"符号与命中不一致：{m['mu']} / {m['ret1']}"
AssertionError: 符号与命中不一致：0.005133817973546684 / 0.49375
```
★ **本文件要回答三问**（只读、不改任何落盘数据）：
  **Q1** 这条断言是否**恒等式**？—— 否的话它就是**设备缺陷**（DEF-26S-20），不是数据缺陷。
  **Q2** 真正的**逐样本恒等式**是什么？—— 应是 `mean(m_{i,ρ} > 0) == ret1`（逐样本：$m>0\iff\arg\max$ 命中）。
  **Q3** 剩余 15 个种子里有**几个**会撞上这条边界？（决定修好之后要不要逐个重跑）

★ **手法**：monkeypatch 模块全局 `measure_hit` 为**宽容版**（只记录 `mu`/`ret1`/`pos_frac`，不 assert），
其余**全部走原代码路径**（`run_unit_dose` 一字不改）⇒ 复现是**忠实**的。
"""
import importlib
import json
import math
import sys

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26s")

import numpy as np

OUT = "/home/ec2-user/sgmn-loop/dir26s/out"
M = importlib.import_module("R1_dose_ar1_v2")

# ── ① 装宽容版 measure_hit：只记录，不 assert ──
_REAL = M.K12.ret_jitter_margin
CAP = []          # 每次调用落一条，便于事后看是哪条撞了边界


def tolerant_measure_hit(P, Kq, vq, n_read, r_q):
    m = _REAL(P, Kq, vq, n_read, r_q)
    # ★ 真正的逐样本恒等式：m>0 ⇔ argmax 命中 ⇒ mean(m>0) 应**逐位等于** ret1
    import torch

    vbU = M.unit(vq)
    pos = tol = 0
    for rho in range(r_q):
        Y = P @ Kq[rho].T
        S = M.unit(Y.T) @ vbU.T
        for i in range(n_read):
            row = S[i]
            other = torch.cat([row[:i], row[i + 1:]]) if i + 1 < n_read else row[:i]
            mm = float(row[i] - other.max())
            pos += 1 if mm > 0 else 0
            tol += 1 if mm == 0 else 0
    n = n_read * r_q
    pos_frac = pos / float(n)
    rec = {"mu": m["mu"], "ret1": m["ret1"], "pos_frac": pos_frac,
           "n": int(m["n_read"]), "n_pos": pos, "n_ties": tol,
           "id_gap": abs(pos_frac - m["ret1"]),          # 恒等式的残差（应为 0）
           "old_assert_holds": bool((m["mu"] > 0) == (m["ret1"] > 0.5))}
    CAP.append(rec)
    m["pos_frac"] = pos_frac
    return m


M.measure_hit = tolerant_measure_hit


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔的种子；空=全部剩余")
    ap.add_argument("--tag", default="DIR26S_boundary_scan")
    a = ap.parse_args()
    only = [int(x) for x in a.only.split(",") if x.strip()]

    by = M.load_rows()
    go = json.load(open(f"{OUT}/DIR26S_power.json"))
    n_use = int(go["n_use"])
    pool = (list(M.SEEDS) + list(M.EXTRA_SEEDS))[:n_use]
    cnt0_of = {int(s): by[(0.0, int(s))]["counts"] for s in pool if (0.0, int(s)) in by}
    tau_d = M.cal_at(M.N_KV)                    # ★ 与 main() 一字一致
    tau = float(tau_d["c_tau"]) if isinstance(tau_d, dict) else float(tau_d)

    done = {s for s in pool if (0.9, int(s)) in by}
    todo = [s for s in pool if s not in done]
    if only:
        todo = [s for s in todo if int(s) in only]
        done = {s for s in done if int(s) in only} or set()
    print(f"已完 φ=0.9：{len(done)} / {n_use}；剩余 {len(todo)} ⇒ {todo}")
    print(f"τ = {tau}")

    # ── ② 先复现：已完的行里有没有已经处于边界、只是恰好没撞上的？──
    print("\n=== ② 已落盘 37 行的「老断言裕度」扫描（不重算，只读落盘值）===")
    near = []
    for s in sorted(done):
        r = by[(0.9, int(s))]
        for tag in ("A", "B", "P"):
            a = r["arms"].get(tag)
            if not a:
                continue
            mu, ret1 = a["mu"], a["ret1"]
            holds = bool((mu > 0) == (ret1 > 0.5))
            margin = abs(ret1 - 0.5)
            if not holds or margin < 0.02:
                near.append({"seed": int(s), "arm": tag, "mu": mu, "ret1": ret1,
                             "old_assert_holds": holds, "ret1_margin_to_half": margin})
    print(f"  处于边界（`|ret1−0.5|<0.02`）或已违反的单元：{len(near)} 个")
    for e in near[:20]:
        print(f"    seed {e['seed']} arm {e['arm']}: mu {e['mu']:+.6f} ret1 {e['ret1']:.5f} "
              f"holds={e['old_assert_holds']} margin={e['ret1_margin_to_half']:.5f}")

    # ── ③ 对剩余种子逐个跑（宽容版），找出撞边界的 ──
    print("\n=== ③ 剩余种子的边界扫描（宽容版 measure_hit，只诊断不落行）===")
    hits = []
    for s in todo:
        CAP.clear()
        try:
            M.run_unit_dose(int(s), 0.9, tau, cnt0_of.get(int(s)))
        except Exception as e:  # noqa: BLE001
            print(f"  seed {s}: 其它异常 {type(e).__name__}: {e}")
            continue
        bad = [c for c in CAP if not c["old_assert_holds"]]
        maxgap = max((c["id_gap"] for c in CAP), default=0.0)
        worst = min(CAP, key=lambda c: abs(c["ret1"] - 0.5))
        print(f"  seed {s}: 调用 {len(CAP)} 次，老断言失败 {len(bad)} 次；"
              f"逐样本恒等式最大残差 {maxgap:.3e}；最靠近 0.5 的 ret1 = {worst['ret1']:.5f}（mu {worst['mu']:+.6f}）")
        for c in bad:
            hits.append({"seed": int(s), **c})
    print(f"\n  ⇒ 剩余 {len(todo)} 个种子里，老断言会在 {len({h['seed'] for h in hits})} 个种子处失败")
    for h in hits:
        print(f"    seed {h['seed']}: mu {h['mu']:+.6f} ret1 {h['ret1']:.5f} "
              f"pos_frac {h['pos_frac']:.5f}（n_pos {h['n_pos']}/{h['n']}）恒等式残差 {h['id_gap']:.3e}")

    json.dump({"near_in_disk": near, "boundary_hits": hits,
               "n_done": len(done), "todo": todo, "tau": tau},
              open(f"{OUT}/{a.tag}.json", "w"), indent=1)
    print(f"\n已写 {OUT}/{a.tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
