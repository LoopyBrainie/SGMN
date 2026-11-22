#!/opt/pytorch/bin/python
r"""DIR-53 · **定论前演练**（`rehearse53.py`；CPU-free，不读真实 `out/`、不产判据）。

★ 动机（DIR-52 的教训）：真正的 15 个 JSON **只读一次**，裁决器若在定论当天才第一次跑，
  任何键缺失 / 形状不符 / 分支顺序错都会把"判据从未被计算"洗成结论。
★ 做法：在**临时根目录**合成 8 组同构 15 件套（`DIR53r12_<cell>_<arm>.json`），
  逐组用 `DIR53_OUT=<tmp>` 跑 `adjudicate53.py`，断言 `outcome / counted / pass_count`，
  并把**解析算出来的数**与**手算真值**对照（差分恒定 ⇒ `se=0`、`Δ=2.0`、`R=5.0`、地板 `1.0`）。

用例（期望）：PASS · FIXED-SUFFICES · OVERFIT-ONLY · INDETERMINATE · DIVERGED ·
             INCOMPLETE-缺臂 · INCOMPLETE-惰臂 · INCOMPLETE-同源错 · AGREE-FALSE

用法：`/opt/pytorch/bin/python dir53/rehearse53.py`
"""
import json
import os
import random
import subprocess
import sys
import tempfile

ADJ = "/home/ec2-user/sgmn-loop/dir53/adjudicate53.py"
CELLS = ("P1", "P2", "P3", "P4", "P5")
ARMS = ("FIX", "OFF", "ORACLE")
N_TRAIN, N_EVAL, N_REP, STEPS = 96, 400, 16, 250


def series(n, base, noise, rng):
    return [base + rng.gauss(0.0, noise) for _ in range(n)]


def make_cell(root, cell, *, L_fix, delta, R, C, noise=0.05,
              off_train=(10.0, 10.0, 0.1), rel_dPk=0.9, row_dev=1e-16,
              sha_over=None, sha_by_arm=None, skip_arm=None, nan_in=None):
    r"""写一格三臂。`delta = L_FIX − L_OFF`、`R = L_FIX − L_ORACLE`、`C = L_OFF_rc − L_OFF`。

    ★ 用**同一份噪声**构造配对量 ⇒ 逐实例差恒定 ⇒ `se = 0`（手算真值可验）。
    """
    rng = random.Random(hash(cell) & 0xFFFF)
    n = rng.gauss  # 噪声底：三臂共享同一 noise 序列时差恒定
    base_noise = [n(0.0, noise) for _ in range(N_EVAL)]
    s_fix = [L_fix + e for e in base_noise]
    s_off = [L_fix - delta + e for e in base_noise]
    s_orc = [L_fix - R + e for e in base_noise]
    s_rc = [L_fix - delta + C + e for e in base_noise]
    if nan_in == "FIX":
        s_fix[7] = float("nan")

    L = {"FIX": s_fix, "OFF": s_off, "ORACLE": s_orc, "OFF_rc": s_rc}
    sha = sha_over or f"{cell}-sha"
    for arm in ARMS:
        if arm == skip_arm:
            continue
        if arm == "FIX":
            lv, q = ("FIX", "learned")
        elif arm == "OFF":
            lv, q = ("OFF", "learned")
        else:
            lv, q = ("ORACLE", "oracle")
        arms = {lv: {"mean_L": sum(L[lv]) / len(L[lv]), "sd_L": 0.0,
                     "cos_q_k": 0.5, "align_q_k": 0.5, "q_norm": 1.0,
                     "y_norm": 1.0, "m_abs": 1.0, "Wmax": 2.0, "c_eq_h_max": 0.0,
                     "L_series": L[lv]}}
        if arm == "OFF":
            arms["OFF_rc"] = {"mean_L": sum(L["OFF_rc"]) / len(L["OFF_rc"]),
                              "sd_L": 0.0, "cos_q_k": 0.5, "align_q_k": 0.5,
                              "q_norm": 1.0, "y_norm": 1.0, "m_abs": 1.0,
                              "Wmax": 2.0, "c_eq_h_max": 0.0, "L_series": L["OFF_rc"]}
        d = {"cell": cell, "arm": arm, "n_train": N_TRAIN, "n_eval": N_EVAL,
             "n_rep": N_REP, "steps": STEPS,
             "pk1_sha1": (sha_by_arm or {}).get(arm, sha),   # ★ 按臂覆写（用例 8）
             "rel_dPk_arm": (0.0 if arm != "OFF" else rel_dPk),
             "off_meta": (None if arm != "OFF" else
                          {"L_off_first20": off_train[0], "L_off_last20": off_train[1],
                           "L_off_se_window": off_train[2], "rel_dPk": rel_dPk,
                           "row_norm_dev_max": row_dev, "pk0_sha1": sha}),
             "arms": arms,
             "L_series": {k: v["L_series"] for k, v in arms.items()}}
        json.dump(d, open(os.path.join(root, f"DIR53r12_{cell}_{arm}.json"), "w"))


def run(scn, **kw):
    root = tempfile.mkdtemp(prefix=f"reh53_{scn}_")
    common = dict(L_fix=20.0, delta=2.0, R=5.0, C=3.0)
    common.update(kw.pop("common", {}))
    for cell in CELLS:
        per = dict(common)
        per.update(kw.get(cell, {}))
        make_cell(root, cell, **per)
    if "formal" in kw:
        json.dump(kw["formal"], open(os.path.join(root, "DIR53r12_formal.json"), "w"))
    env = dict(os.environ, DIR53_OUT=root)
    p = subprocess.run([sys.executable, ADJ], capture_output=True, text=True, env=env)
    outp = os.path.join(root, "DIR53r12_adjudication.json")
    if p.returncode != 0 or not os.path.exists(outp):
        return {"crash": True, "stderr": p.stderr[-800:], "root": root}
    d = json.load(open(outp, encoding="utf-8"))
    d["_root"] = root
    d["_stdout_tail"] = p.stdout.strip().splitlines()[-1:] if p.stdout else []
    return d


def main():
    ok = True
    results = {}

    def check(name, cond, detail=""):
        nonlocal ok
        results[name] = {"ok": bool(cond), "detail": detail}
        ok = ok and bool(cond)
        print(f"  [{'✔' if cond else '✘'}] {name}  {detail}")

    print("── 用例 1：PASS（Δ=2.0、R=5.0 ⇒ 地板 1.0、C=3.0、se=0）")
    d = run("PASS")
    check("PASS.outcome", d.get("outcome") == "PASS", d.get("outcome"))
    check("PASS.counted", d.get("counted") is True, d.get("counted"))
    check("PASS.pass_count", d.get("pass_count") == 5, d.get("pass_count"))
    c1 = d.get("cells", {}).get("P1", {})
    check("PASS.Δ 手算 2.0", abs(c1.get("delta", 0) - 2.0) < 1e-12, c1.get("delta"))
    check("PASS.R 手算 5.0", abs(c1.get("R", 0) - 5.0) < 1e-12, c1.get("R"))
    check("PASS.地板 0.20R = 1.0", abs(c1.get("floor", 0) - 1.0) < 1e-12, c1.get("floor"))
    check("PASS.se = 0（配对差恒定）", abs((c1.get("pd") or {}).get("se", 1) - 0.0) < 1e-12,
          (c1.get("pd") or {}).get("se"))
    check("PASS.C 手算 3.0", abs((c1.get("pc") or {}).get("mean", 0) - 3.0) < 1e-12,
          (c1.get("pc") or {}).get("mean"))

    print("── 用例 2：FIXED-SUFFICES（Δ=0、C=3.0、离线目标未显著改善）")
    d = run("FIX", common={"delta": 0.0, "off_train": (10.0, 10.0, 0.1)})
    check("FIXED.outcome", d.get("outcome") == "FIXED-SUFFICES", d.get("outcome"))
    check("FIXED.counted 必须 False", d.get("counted") is False, d.get("counted"))
    check("FIXED.pass_count = 0", d.get("pass_count") == 0, d.get("pass_count"))

    print("── 用例 3：OVERFIT-ONLY（Δ=0，但离线目标 10→1 ≫ 3se）")
    d = run("OVER", common={"delta": 0.0, "off_train": (10.0, 1.0, 0.1)})
    check("OVER.outcome", d.get("outcome") == "OVERFIT-ONLY", d.get("outcome"))
    check("OVER.counted False", d.get("counted") is False, d.get("counted"))

    print("── 用例 4：INDETERMINATE（P3 格 R<0）")
    d = run("IND", P3={"R": -1.0})
    check("IND.outcome", d.get("outcome") == "INDETERMINATE", d.get("outcome"))
    check("IND.counted False", d.get("counted") is False, d.get("counted"))

    print("── 用例 5：DIVERGED（P1/FIX 含 NaN）")
    d = run("DIV", P1={"nan_in": "FIX"})
    check("DIV.outcome", d.get("outcome") == "DIVERGED", d.get("outcome"))
    check("DIV.counted False", d.get("counted") is False, d.get("counted"))

    print("── 用例 6：INCOMPLETE-缺臂（P2/ORACLE 未写）")
    d = run("MISS", P2={"skip_arm": "ORACLE"})
    check("MISS.outcome", d.get("outcome") == "INCOMPLETE", d.get("outcome"))
    check("MISS.counted False", d.get("counted") is False, d.get("counted"))
    check("MISS.missing 记到 P2/ORACLE", "P2/ORACLE" in (d.get("missing") or []),
          d.get("missing"))

    print("── 用例 7：INCOMPLETE-惰臂（P4/OFF 的 rel_dPk=0 ⇒ 判据从未被计算）")
    d = run("INERT", P4={"rel_dPk": 0.0})
    check("INERT.outcome", d.get("outcome") == "INCOMPLETE", d.get("outcome"))
    check("INERT.inert_arms 非空", bool(d.get("inert_arms")), d.get("inert_arms"))

    print("── 用例 8：INCOMPLETE-同源错（P5 的 FIX 与 ORACLE 的 pk1_sha1 不同）")
    d = run("SHA", P5={"sha_by_arm": {"ORACLE": "P5-DIFFERENT"}})
    check("SHA.outcome", d.get("outcome") == "INCOMPLETE", d.get("outcome"))
    check("SHA.sha_mismatch 非空", bool(d.get("sha_mismatch")), d.get("sha_mismatch"))

    print("── 用例 9：AGREE-FALSE（数据 = PASS，但 merge 文件写 FIXED-SUFFICES）")
    fake_formal = {"outcome": "FIXED-SUFFICES", "counted": False, "cells": {}}
    d = run("AGREE", formal=fake_formal)
    ma = d.get("merge_agreement") or {}
    check("AGREE 被检出 False", ma.get("agree") is False, ma)

    print(f"\n★ 演练结论：{'全部 ' + str(len(results)) + ' 项通过' if ok else '有失败项'}")
    print("★★ 提醒：真实 15 件套**只读一次**；先跑本演练、再跑 `adjudicate53.py`。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())