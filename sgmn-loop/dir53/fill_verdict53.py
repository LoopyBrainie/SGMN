#!/opt/pytorch/bin/python
r"""DIR-53 · **裁决书填数器**（`fill_verdict53.py`；只写**草案**，永不写 `VERDICT.md`）。

★ 定位：定论当天**只做"抄数 + 套固定措辞"**（DIR-52 的教训）。本工具把机械部分自动化：
  读 `DIR53r12_adjudication.json`（独立裁决）+ 逐格 `_OFF.json` 的 `off_meta` + `_FIX/_ORACLE` 的机制量，
  产出 `revolve/DIR-53/VERDICT.draft.md`：§0 结局/计数、§1 **逐字**固定措辞（`VERDICT-SCAFFOLD.md §1`）、
  §2 五格读数表、§3 `OFF` 臂被训到的证据。§4 机制与 §5 缺陷留 `〔人工〕` 占位。
★ **拒绝条件**（不写草案、退出码 2）：`merge_agreement.agree != true`、`n_cells_read != 5`、
  或 `outcome` 不在六支之内 —— 一致性没成立就不许开始写结论。
★ `--write` 才会真写文件；默认只打印。`DIR53_OUT` 可指向临时树（自测用）。

用法：
  `/opt/pytorch/bin/python dir53/fill_verdict53.py            # 打印草案`
  `/opt/pytorch/bin/python dir53/fill_verdict53.py --write    # 落盘 VERDICT.draft.md`
  `/opt/pytorch/bin/python dir53/fill_verdict53.py --selftest # 两场景端到端自测`
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/ec2-user/sgmn-loop"
OUT = os.environ.get("DIR53_OUT", os.path.join(ROOT, "dir53", "out"))
DRAFT = os.path.join(ROOT, "revolve", "DIR-53", "VERDICT.draft.md")
CELLS = ("P1", "P2", "P3", "P4", "P5")

# ★ 六支固定措辞：**逐字**取自 `VERDICT-SCAFFOLD.md §1`（改字即改判据表述，须新开 revision）
WORDING = {
    "PASS": "在本玩具/本协议/本训练预算下，**离线学到的 $P_k$ 比同初值的固定随机 $P_k$ 在未见实例上有独立的内容性收益**"
            "—— 5 张预注册抽取中 ≥4 张同时满足 P1（`Δ ≥ 3se`）、P2（`Δ ≥ 0.20·R_j`）、P3（内容置换使收益消失）。",
    "FIXED-SUFFICES": "在本玩具/本协议/本训练预算下，**固定随机 $P_k$ 已足够**：把 $P_k$ 学出来换不到超过 20% 可寻址区间的收益"
                      "⇒ v2.4 不必为 $P_k$ 规定离线学习程序。★ **不得**写成「学习无用」「$P_k$ 不可学」。",
    "OVERFIT-ONLY": "离线目标在训练段显著下降（≥3se），但评估段无对应收益 ⇒ 收益**不迁移**。",
    "INDETERMINATE": "某格 `R_j ≤ 0` ⇒ 『oracle 是上限』的前提不成立 ⇒ 判据**从未被计算**（**不是失败**）。",
    "DIVERGED": "存在非有限读数 ⇒ 整轮作废。",
    "INCOMPLETE": "缺臂 / n 不符 / `OFF` 臂未真的被训到 / 同格 `FIX`≠`ORACLE` 的 `Pk1` ⇒ 判据从未被计算。",
}


# ★ §5：本轮缺陷（`PROBE53.md`/`decisions` 已登记；草案**照抄**，不新造）
DEFECTS = """\
- `DEF-53-1`（rev 1.0）：Freeze C ⇒ 任务损失对 $P_k$ 梯度**恒零** ⇒ 原 `OFF` 设计不可实现；`OFF` 改为**写侧自检索目标**。
- `DEF-53-2`（rev 1.1）：判据公式维度不合法，已修正。
- `DEF-53-3`：投影广播 `(8,)` vs `(8,1)` 造成误投影，已修（`probe53b` 全绿）。
- `DEF-53-4`：关卡把**层**当**臂**索引 ⇒ 每格被跳过 ⇒ `cells` 空 ⇒ 结局被**静默洗成** `FIXED-SUFFICES`。
- `DEF-53-5`：`statistics.stdev` 遇 `NaN` 抛 `ValueError` ⇒ 裁决器崩溃、报不出 `DIVERGED`。
- 纪律候选 **123**：补丁必带 `assert`；守卫必须**真读**被守卫对象（一次静默失效补丁会让测试变成空测试）。
- 纪律：正式轮读数**只算一次**；`out/` 只出不进；推送只走 `push52.sh`。"""

# ★ §6：卡 `DIRECTION_rev53_v12.md §8` 的**原样**文本（不得改写；改字须新开 revision）
BOUNDARY = """\
1. **条件于本玩具**：`D_H=8, D_K=16, D_V=4, R=2, T=16, n_rep=16, η=0.2, λ=0.05, KEY_SCALE=0.5` ⇒ **不得**外推为架构结论，**不得**写「记忆改善 LLM」。
2. **「离线学习」= 本卡冻结的定义**（同初值 + 250 步 Adam + 行范数投影 + 与头联合训练）。**其他学习程序**（不同损失、不同预算、不同初始化）**未测**，不得声称已排除。
3. **`OFF` 比 `FIX` 多了 `D_H·D_K = 128` 个可训参数** ⇒ 本卡回答的是**设计问题**（"值不值得学"），**不是**"学习优于同等参数量的其他用法"。此点不得模糊。
4. **`P_v` 未测**：本卡只取 $P_k$ 一支；$P_v$ 与内容保真度纠缠，需另设控制（未开卡）。
5. **`FIXED-SUFFICES` 的合法表述**：「在**本玩具、本协议、本训练预算**下，把 $P_k$ 学出来换不到超过 `20%` 可寻址区间的收益 ⇒ **固定随机 $P_k$ 已足够**；v2.4 不必为 $P_k$ 规定学习程序」。**不得**写「学习无用」「$P_k$ 不可学」。"""

def f4(x):
    return "—" if x is None else f"{x:.4f}"


SIG = 3.0          # ★ 冻结判据用的 SIG_K；guard 会**实测**核对它（P1/P3 必须能被它复现）


def s3(d):
    """`3se`：裁决器落盘的是 `{mean,se}`（判据 `SIG_K*se`）；老 schema 的 `3se` 键若在则直用。"""
    d = d or {}
    if d.get("3se") is not None:
        return d["3se"]
    return None if d.get("se") is None else SIG * d["se"]


def jload(path):
    try:
        return load(path)
    except Exception:
        return {}


def arm(path_tpl, cell, arm):
    """读某格某臂的落盘文件（§4 机制量的来源；缺文件返回 {}）。"""
    return jload(os.path.join(OUT, path_tpl.format(cell=cell, arm=arm)))


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def guard(adj):
    """一致性守卫：不成立就**不许**开始写结论。"""
    problems = []
    if adj.get("outcome") not in WORDING:
        problems.append(f"outcome 不在六支之内：{adj.get('outcome')!r}")
    if adj.get("n_cells_read") != len(CELLS):
        problems.append(f"n_cells_read={adj.get('n_cells_read')} ≠ 5")
    ma = adj.get("merge_agreement") or {}
    if ma.get("agree") is not True:
        problems.append(f"merge_agreement.agree ≠ true（{ma.get('numeric_diffs') or ma}）")
    # ★ 谓词复算：草案要打印 `3se`，必须证明 `SIG*se` 就是裁决器用的那个量
    for c, x in (adj.get("cells") or {}).items():
        try:
            p1 = x["delta"] >= SIG * x["pd"]["se"]
            p2 = x["delta"] >= x["floor"]
            p3 = x["pc"]["mean"] >= SIG * x["pc"]["se"]
        except (KeyError, TypeError) as e:
            problems.append(f"{c}: 谓词字段缺失（{e}）")
            continue
        if (p1, p2, p3) != (x.get("P1"), x.get("P2"), x.get("P3")):
            problems.append(f"{c}: 谓词复算不符（重算 {p1}/{p2}/{p3} vs 落盘 "
                            f"{x.get('P1')}/{x.get('P2')}/{x.get('P3')}）")
    return problems


def table(adj):
    head = ("| 格 | `L_FIX` | `L_OFF` | `L_OFF_rc` | `L_ORACLE` | `R` | 地板 `0.20R` | "
            "`Δ` | `3se(Δ)` | `C` | `3se(C)` | P1 | P2 | P3 | 该格 |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [head, sep]
    for c in CELLS:
        x = (adj.get("cells") or {}).get(c)
        if not x:                       # 缺格 ⇒ 明确写"未计算"，绝不填 0
            rows.append(f"| {c} | 〔未计算〕 | | | | | | | | | | | | | |")
            continue
        L = x.get("L") or {}
        pd, pc = x.get("pd") or {}, x.get("pc") or {}
        yn = lambda b: "✔" if b else "✘"
        rows.append(
            f"| {c} | {f4(L.get('FIX'))} | {f4(L.get('OFF'))} | {f4(L.get('OFF_rc'))} | {f4(L.get('ORACLE'))} | "
            f"{f4(x.get('R'))} | {f4(x.get('floor'))} | {f4(x.get('delta'))} | {f4(s3(pd))} | "
            f"{f4(pc.get('mean'))} | {f4(s3(pc))} | {yn(x.get('P1'))} | {yn(x.get('P2'))} | {yn(x.get('P3'))} | "
            f"{'过' if x.get('pass') else '未过'} |")
    return "\n".join(rows)


def off_evidence(adj):
    """§3：`OFF` 臂真的被训到的证据（优先用裁决器内联的 `off_*`，缺失才读 OFF 文件）。"""
    lines = ["| 格 | `L_off_first20` | `L_off_last20` | `se_window` | `rel_dPk` | `row_norm_dev_max` | "
             "`‖Pk0‖→‖Pk1‖` | `grad_pk_last` |",
             "|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        x = (adj.get("cells") or {}).get(c)
        if not x:
            lines.append(f"| {c} | 〔未计算〕 | | | | | | |")
            continue
        off_file = arm("DIR53r12_{cell}_OFF.json", c, "OFF")   # ★ `pk1_fro` 在**顶层**，不在 off_meta
        om = off_file.get("off_meta") or {}
        first = x.get("off_L_first20", om.get("L_off_first20"))
        last = x.get("off_L_last20", om.get("L_off_last20"))
        rel = x.get("off_rel_dPk", om.get("rel_dPk"))
        dev = x.get("off_row_norm_dev_max", om.get("row_norm_dev_max"))
        dev_s = "—" if dev is None else f"{dev:.2e}"
        lines.append(f"| {c} | {f4(first)} | {f4(last)} | {f4(om.get('L_off_se_window'))} | "
                     f"{f4(rel)} | {dev_s} | {f4(om.get('pk0_fro'))}→{f4(off_file.get('pk1_fro'))} | "
                     f"{f4(om.get('grad_pk_last'))} |")
    return "\n".join(lines)


def mechanism_numbers(adj):
    """§4 的**数字**部分（机制解释文字仍由人写）。逐格读三臂落盘文件，不臆造字段。"""
    lines = []
    for c in CELLS:
        x = (adj.get("cells") or {}).get(c) or {}
        if not x:
            continue
        fix = arm("DIR53r12_{cell}_FIX.json", c, "FIX")
        off = arm("DIR53r12_{cell}_OFF.json", c, "OFF")
        f_fix = (fix.get("arms") or {}).get("FIX") or {}
        f_off = (off.get("arms") or {}).get("OFF") or {}
        R, d = x.get("R"), x.get("delta")
        ratio = (d / R) if (R not in (None, 0) and d is not None) else None
        lines.append(
            f"- `{c}`：`cos_q_k` FIX={f4(f_fix.get('cos_q_k'))} vs OFF={f4(f_off.get('cos_q_k'))} · "
            f"`Wmax` {f4(f_fix.get('Wmax'))}/{f4(f_off.get('Wmax'))} · "
            f"`align_q_k` {f4(f_fix.get('align_q_k'))}/{f4(f_off.get('align_q_k'))} · "
            f"`c_eq_h_max` {f4(f_fix.get('c_eq_h_max'))}/{f4(f_off.get('c_eq_h_max'))} · "
            f"`dWq1_init` {f4(fix.get('dWq1_from_init'))}/{f4(off.get('dWq1_from_init'))} · "
            f"`rel_dPk_arm` {f4(fix.get('rel_dPk_arm'))}/{f4(off.get('rel_dPk_arm'))} · "
            f"`Δ/R`={f4(ratio)} · `C`={f4((x.get('pc') or {}).get('mean'))}"
            f"（`{SIG:.0f}se`={f4(s3(x.get('pc')))}）")
    return "\n".join(lines) if lines else "〔无格读数〕"


def render(adj):
    oc, counted = adj.get("outcome"), adj.get("counted")
    n = 39 if oc == "PASS" else 38
    got = adj.get("pass_count")
    head = [
        "# DIR-53 · 裁决书**草案**（`VERDICT.draft.md`；由 `fill_verdict53.py` 生成，须人工复核后改名 `VERDICT.md`）",
        "",
        "> ★ 本文件是**草案**，不是证据。定论当天：复核 §4 机制文字与 §5 缺陷 → 改名 `VERDICT.md` → 追加证据行 `D-76`。",
        "",
        "## §0 结论",
        "",
        # ★ 审计纪律：`SUCCEEDED` 附近**只**许出现挂计数本身；故这里不写任何 `k/k` 形式。
        f"- 结局 = **`{oc}`** · 计入 `SUCCEEDED` = **{'是' if counted else '否'}**"
        f"（`pass_count = {got}`，共 5 格）",
        f"- 挂计数：`SUCCEEDED = {n} / {n}`",
        f"- 一句话结论：{WORDING.get(oc, '〔未知结局〕')}",
        "",
        "## §1 主假说与判定（固定措辞，逐字取自 `VERDICT-SCAFFOLD.md §1`）",
        "",
        f"- **`{oc}`**：{WORDING.get(oc, '〔未知结局〕')}",
        "",
        "## §2 逐格读数（5 格，每格 4 个 `L`）",
        "",
        table(adj),
        "",
        "〔来源：`DIR53r12_adjudication.json: cells.<cell>.{L.*,R,floor,delta,pd.3se,pc.mean,pc.3se,P1,P2,P3,pass}`〕",
        "",
        "## §3 `OFF` 臂真的被训到的证据（必披露）",
        "",
        off_evidence(adj),
        "",
        "〔来源：`DIR53r12_<cell>_OFF.json: off_meta`；`DIR53r12_stageA.json: guards`〕",
        "",
        "## §4 机制（数字部分；**解释文字人工补**）",
        "",
        mechanism_numbers(adj),
        "",
        "〔人工〕三问：①`P_k` 变成什么了（`rel_dPk`/行范数/键行间余弦）②收益走哪条路（`cos_q_k`/`Wmax`/`C_j` 是否使收益消失）③`R_j` 的量级",
        "",
        "## §5 缺陷与纪律",
        "",
        DEFECTS,
        "",
        "## §6 证据边界（**逐字**取自卡 `DIRECTION_rev53_v12.md §8`）",
        "",
        BOUNDARY,
        "",
    ]
    return "\n".join(head)


def main(argv):
    if "--selftest" in argv:
        return selftest()
    if not os.path.exists(os.path.join(OUT, "DIR53r12_adjudication.json")):
        print(f"★ 拒绝：{OUT}/DIR53r12_adjudication.json 不存在 —— 先跑 `adjudicate53.py`")
        return 2
    adj = load(os.path.join(OUT, "DIR53r12_adjudication.json"))
    bad = guard(adj)
    if bad:
        print("★ 拒绝写草案（一致性守卫未成立）：")
        for b in bad:
            print("   -", b)
        return 2
    text = render(adj)
    if "--write" in argv:
        with open(DRAFT, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"✔ 草案已写：{DRAFT}（★ 不是裁决书；复核后改名 VERDICT.md）")
    else:
        print(text)
    return 0


def selftest():
    """两场景端到端：合成 15 件套 → adjudicate → fill，断言措辞/计数/表格行数。"""
    import subprocess
    import tempfile

    sys.path.insert(0, HERE)
    import rehearse53 as R

    def _adj(env):
        p = subprocess.run([sys.executable, os.path.join(HERE, "adjudicate53.py")],
                           capture_output=True, text=True, env=env)
        assert p.returncode == 0, p.stderr[-600:]
        return load(os.path.join(env["DIR53_OUT"], "DIR53r12_adjudication.json"))

    def build(**common):
        """造 15 件套 → 首跑裁决 → 由裁决值造**同构** `formal.json`（模拟 merge 层）→ 复跑。"""
        root = tempfile.mkdtemp(prefix="self53_")
        for c in CELLS:
            R.make_cell(root, c, **dict(dict(L_fix=20.0, delta=2.0, R=5.0, C=3.0), **common))
        env = dict(os.environ, DIR53_OUT=root)
        a0 = _adj(env)
        formal = {"outcome": a0["outcome"], "counted": a0["counted"],
                  "cells": {c: {"R": v["R"], "delta": v["delta"],
                                "floor_0.20R": v["floor"], "pass": v["pass"]}
                            for c, v in a0["cells"].items()}}
        json.dump(formal, open(os.path.join(root, "DIR53r12_formal.json"), "w"))
        return root, env

    ok = True
    for name, kw, want_out, want_n, want_phrase, want_refusal in (
            ("PASS", {}, "PASS", 39, "有独立的内容性收益", False),
            ("FIXED", {"delta": 0.0, "off_train": (10.0, 10.0, 0.1)}, "FIXED-SUFFICES", 38, "已足够", False),
            ("IND", {"R": -1.0}, "INDETERMINATE", 38, "前提不成立", False),
            ("NOMERGE", {}, None, None, None, True),
            ("MISMATCH", {}, None, None, None, True)):
        root, env = build(**kw)
        adjp = os.path.join(root, "DIR53r12_adjudication.json")
        if name == "NOMERGE":           # 删掉 merge 交叉核对 ⇒ 守卫必须拒绝
            adj = load(adjp)
            adj.pop("merge_agreement", None)
            json.dump(adj, open(adjp, "w"))
        elif name == "MISMATCH":        # merge 层与独立裁决差 1e-3 ⇒ 守卫必须拒绝
            fp = os.path.join(root, "DIR53r12_formal.json")
            f = load(fp)
            f["cells"]["P3"]["delta"] += 1e-3
            json.dump(f, open(fp, "w"))
            adj = _adj(env)
        else:
            adj = _adj(env)
        good = True
        if want_refusal:
            good = bool(guard(adj))
        else:
            good = (adj.get("outcome") == want_out and not guard(adj))
            if good:
                txt = render(adj)
                good = (want_phrase in txt) and (f"{want_n} / {want_n}" in txt)
                good = good and all(f"| {c} |" in txt for c in CELLS)
                good = good and txt.count("\n| P") >= 5
        print(f"  [{'✔' if good else '✘'}] {name}: outcome={adj.get('outcome')} 守卫拒绝={bool(guard(adj))}")
        ok = ok and good
    print(f"★ 自测：{'全过' if ok else '有失败项'}（PASS/FIXED/IND 走渲染+措辞+计数+5 行；NOMERGE 走拒绝）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))