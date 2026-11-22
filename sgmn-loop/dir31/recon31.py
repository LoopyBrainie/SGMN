#!/opt/pytorch/bin/python
r"""**重建** DIR-31 正式轮**首跑**（rev 1.1 时的 `formal()`）的结果文件。

背景（DEF-31-3）：首跑因 P3-② 的**副本不同步**（用 $(\mathcal W^o_1,T{=}1)$ 而非 $(\gamma_2,T{=}1)$）
判 `MEASUREMENT-NOT-CREDIBLE`，其 JSON 在 re-run 时被**同名覆盖**（流程缺陷：失败轮未留档）。
本轮次为**确定性**的（A31-4 已断言逐位可复现），故重建文件与首跑读数一致；
`secs` 与文件字节不保证相同 —— 本脚本的输出**显式标注为重建**，不得当作原始产物引用。
"""
import json
import R1_grad as M

Mv, worst = M.matrix()
rel_bad = max(M.fd_grad_autograd(t_steps=1, entry=e, key="Wo1")[2]
              for e in ((0, 0), (7, 2)))          # ★ 首跑的错误口径
p1 = worst["V0"]["alive_min"] >= M.ALIVE_MIN
bad = [(v, c, Mv[v][c], M.PREDICT[v][c]) for v in M.VARIANTS for c in M.CLAUSES
       if Mv[v][c] != M.PREDICT[v][c]]
p2 = not bad
p3 = rel_bad < M.FD_TOL
out = {
    "_note": "★ 重建文件（原首跑 JSON 被 re-run 同名覆盖，见 VERDICT §7 DEF-31-3）。"
             "读数为确定性复现，secs/字节与原run不同，不得作为原始产物引用。",
    "_reconstructed": True,
    "P1_alive": p1, "P2_matrix": p2, "P3_rel": rel_bad, "P3_credible": p3,
    "C": {c: Mv["V0"][c] for c in M.CLAUSES},
    "worst_V0": worst["V0"],
    "verdict": "MEASUREMENT-NOT-CREDIBLE" if not (p1 and p2 and p3) else "?!",
}
json.dump(out, open(f"{M.OUT}/DIR31r10_formal_run1_RECONSTRUCTED.json", "w"),
          ensure_ascii=False, indent=1)
print(f"重建完成：P1={p1} P2={p2} P3={p3}（rel={rel_bad:.4f}）⇒ {out['verdict']}")
print("C1..C4a/C4b =", out["C"])
