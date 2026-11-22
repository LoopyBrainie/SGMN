#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-108 判决器单元测试（trainer 无 pytest ⇒ 走 `__main__`，见 `DEF-107-12`）。

目的：证明**分支逻辑本身**正确 —— 不是"它这次恰好输出了 C108-PREMISE-INVALID"。
做法：喂**构造的**格读数，检查 §5 的三个分支各自能否被触发，以及
      「判据从未被计算」是否**不**被记成 PASS/FAIL（`loop.md` 硬规则）。

用法：/opt/pytorch/bin/python test_verdict108.py
"""
import json
import pathlib
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import R1_task108 as T                                    # noqa: E402

FAILED = []
PASSED = []


def chk(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  {'✓' if cond else '✗'} {name}" + (f"   {detail}" if detail else ""), flush=True)


def make_cell(rho, acc=0.90625, grad=8.699483089458226e-08, wd=0.0,
              base="26f966c3b98ccc30", wg_sha="b746217fa1d7f943", finite=True):
    return {"seed": 7777, "family": "gate", "steps": 3000, "stage": "cell108",
            "acc_all32": acc, "rho": rho, "grad_max": grad, "grad_dist": None,
            "W_drift": wd, "Wg_norm_init": 51.8717, "Wg_norm_delta": rho * 51.8717,
            "Wg_init_sha1": wg_sha, "Wg_finite": finite,
            "base_sha1_before": base, "base_sha1_after": base, "secs": 1.0}


class Args:
    def __init__(self, cd):
        self.cells_dir = str(cd); self.force = False
        self.families = self.steps = self.seeds = ""


def run_case(tag, cells, synth_guards=None, expect=None):
    """在临时目录里摆好 cells(+synth)，跑 verdict，返回 branch。"""
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "out").mkdir(); (td / "out" / "cells").mkdir()
        for name, c in cells.items():
            (td / "out" / "cells" / f"{name}.json").write_text(
                json.dumps(c), encoding="utf-8")
        sg = synth_guards if synth_guards is not None else {
            "A108-1": True, "A108-3": True, "A108-4": True, "A108-7": True, "A108-10": True}
        (td / "out" / "synth.json").write_text(
            json.dumps({"guards": sg, "adam": {}}), encoding="utf-8")
        old_out = T.OUT
        T.OUT = td / "out"
        try:
            got = T.verdict108_main(Args(td / "out" / "cells"))
        finally:
            T.OUT = old_out
        print(f"  → [{tag}] branch = {got['branch']}", flush=True)
        if expect is not None:
            chk(f"{tag}: branch == {expect}", got["branch"] == expect,
                f"实得 {got['branch']}")
        return got


print("[1] §5 三支各自能否被触发", flush=True)
run_case("rho=0.589（本轮实测）", {"gate_3000_7777": make_cell(0.589229),
                                "gate_600_7777": make_cell(0.587304)}, expect="C108-PREMISE-INVALID")
run_case("rho=0.0099（略低于阈值）", {"gate_3000_7777": make_cell(0.0099),
                                  "gate_600_7777": make_cell(0.005)}, expect="C108-GATE-UNLEARNED")
run_case("rho=0.0101（略高于阈值）", {"gate_3000_7777": make_cell(0.0101),
                                  "gate_600_7777": make_cell(0.005)}, expect="C108-PREMISE-INVALID")
run_case("rho=0.0（完全没动）", {"gate_3000_7777": make_cell(0.0),
                              "gate_600_7777": make_cell(0.0)}, expect="C108-GATE-UNLEARNED")

print("\n[2] 边界：阈值是 >= 还是 >（0.01 本身）", flush=True)
run_case("rho=0.01 恰在阈值", {"gate_3000_7777": make_cell(0.01),
                             "gate_600_7777": make_cell(0.01)}, expect="C108-PREMISE-INVALID")

print("\n[3] 守卫失败 ⇒ 必须是 INCOMPLETE，不得记 PASS/FAIL", flush=True)
run_case("W_drift != 0（A108-5 失败）", {"gate_3000_7777": make_cell(0.589, wd=1e-9),
                                     "gate_600_7777": make_cell(0.587)}, expect="C108-INCOMPLETE")
run_case("rho 非有限（A108-6 失败）", {"gate_3000_7777": make_cell(float("nan")),
                                  "gate_600_7777": make_cell(0.587)}, expect="C108-INCOMPLETE")
run_case("W_g 非有限", {"gate_3000_7777": make_cell(0.589, finite=False),
                      "gate_600_7777": make_cell(0.587)}, expect="C108-INCOMPLETE")
run_case("基座 sha1 变了（A108-4 失败）", {"gate_3000_7777": make_cell(0.589, base="deadbeefdeadbeef"),
                                     "gate_600_7777": make_cell(0.587)}, expect="C108-INCOMPLETE")
run_case("synth 守卫 A108-3 失败", {"gate_3000_7777": make_cell(0.589),
                                "gate_600_7777": make_cell(0.587)},
         synth_guards={"A108-1": True, "A108-3": False, "A108-4": True,
                       "A108-7": True, "A108-10": True}, expect="C108-INCOMPLETE")
run_case("缺 gate_3000 格", {"gate_600_7777": make_cell(0.587)}, expect="C108-INCOMPLETE")
run_case("两格 Wg_init_sha1 不一致（A108-1 失败）",
         {"gate_3000_7777": make_cell(0.589),
          "gate_600_7777": make_cell(0.587, wg_sha="0000000000000000")},
         expect="C108-INCOMPLETE")

print("\n[4] A108-2：不逐位复现基线 ⇒ INCOMPLETE（防「装置被动了却当结论」）", flush=True)
run_case("acc 变了", {"gate_3000_7777": make_cell(0.589, acc=0.875),
                    "gate_600_7777": make_cell(0.587)}, expect="C108-INCOMPLETE")
run_case("grad_max 偏 1%", {"gate_3000_7777": make_cell(0.589, grad=8.699483089458226e-08 * 1.01),
                         "gate_600_7777": make_cell(0.587)}, expect="C108-INCOMPLETE")

print("\n[5] 手算 ρ（A108-7）几何/算术正确性", flush=True)
h, ok = T.check_hand_rho()
chk("A108-7 两个手算例", ok)
chk("例1 ρ=√0.25/√2=0.353553", abs(h["diag(1,1)"]["rho"] - 0.35355339059327373) < 1e-12)
chk("例2 ρ=0.001/0.0424264=0.0235702",
    abs(h["diag(.03,.03)"]["rho"] - 0.001 / (0.03 * 2 ** 0.5)) < 1e-12)

print("\n[6] ρ 的定义：只看增幅（平移不变的部分不得被误算）", flush=True)
import torch                                              # noqa: E402
w0 = torch.ones(2, 2); w1 = w0.clone()
chk("ΔW=0 ⇒ ρ=0", float((w1 - w0).norm() / w0.norm()) == 0.0)
w2 = w0 * 2.0
chk("W 翻倍 ⇒ ρ=1", abs(float((w2 - w0).norm() / w0.norm()) - 1.0) < 1e-12)
w3 = -w0
chk("W 取反 ⇒ ρ=2（不是 0）", abs(float((w3 - w0).norm() / w0.norm()) - 2.0) < 1e-12)

print("\n" + "=" * 70, flush=True)
print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项", flush=True)
if FAILED:
    print("★ 失败项：" + "; ".join(FAILED), flush=True)
    sys.exit(1)
print("★ 全部通过", flush=True)