#!/opt/pytorch/bin/python
r"""DIR-52 · **路径表可追溯性自测**（`selftest_paths52.py`）—— 纯结构检查，**不读任何真实数据**。

动机（Cycle 79）：`VERDICT-SCAFFOLD.md §3` 的「JSON 路径」列是我在定论当日要**逐条引用**的地址。
若其中任何一条在真实裁决输出里**不存在**，VERDICT 就会引用一个不存在的字段（或写错路径），
而那种错误在定论当日**没有第二次机会**修正（冻结口径不许追认）。

做法：用 `selftest_adjudicate52.py` 同款的**合成 200 行**跑一次裁决器，得到**完整**输出 JSON，
再把 SCAFFOLD §3 表格第二列的每一个反引号路径**逐条解析**，断言其可被解析（含嵌套 `a.b`）。

★ 本脚本**绝不**触碰 `dir52/out/DIR52r12_shard*.jsonl`，也**不**打印任何真实读数。
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ADJ = os.path.join(HERE, "adjudicate52.py")
SCAFFOLD = os.path.join(HERE, "..", "revolve", "DIR-52", "VERDICT-SCAFFOLD.md")
N = 200
SEED0 = 900000  # ★ 合成种子段，与 62000..62199 不相交，避免与真实数据混淆


def synth_rows(n=N, seed0=SEED0):
    """合成 n 行：`SHAPE` 带微小负偏，使 P4 通过、`PASS` 支成立（只为让所有字段被填充）。"""
    rs = []
    for i in range(n):
        early = -0.0004 + 0.0002 * ((i % 7) - 3)
        late = early - (-0.0002)
        total = early + late
        rs.append({
            "seed": seed0 + i,
            "phi": {"0": {"F": 1.0, "K": 1.0}, "0.9": {"F": 1.0, "K": 1.0}},
            "EARLY": early, "LATE": late, "TOTAL": total,
            "SHAPE": early - late,
            "mu": {"0": [0.0, 0.0, 0.0, 0.0, total], "0.9": [0.0, 0.0, 0.0, 0.0, total]},
            "finite": True, "identity_err": 0.0,
            "n_keep": 4, "keep_n": [1, 2, 3, 4], "wnorm": 1.0,
        })
    return rs


def resolve(obj, path):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False
    return True


def main():
    txt = open(SCAFFOLD, encoding="utf-8").read()

    # 只取 §3 表格段（到下一个 `## ` 为止）
    m = re.search(r"^## §3(.*?)(?=^## )", txt, re.S | re.M)
    if not m:
        print("FAIL 找不到 §3 段"); return 1
    sec = m.group(1)

    paths = []
    for line in sec.splitlines():
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) < 3:
            continue
        for p in re.findall(r"`([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)`", cells[2]):
            if p not in paths:
                paths.append(p)

    print(f"=== DIR-52 SCAFFOLD §3 路径表自测（{len(paths)} 条路径）===")
    if not paths:
        print("FAIL 未从 §3 提取到任何路径"); return 1

    with tempfile.TemporaryDirectory() as d:
        sh = os.path.join(d, "rows.jsonl")
        with open(sh, "w", encoding="utf-8") as f:
            for r in synth_rows():
                f.write(json.dumps(r) + "\n")
        out = os.path.join(d, "adj.json")
        cmd = [sys.executable, ADJ, "--shards", sh,
               "--delta-star", "0.000430", "--power-gate", "0.000861",
               "--n-expected", str(N), "--max-indet-frac", "0.20",
               "--min-n-eff", "160", "--seed-segment", str(SEED0), str(SEED0 + N - 1),
               "--out", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("FAIL 裁决器在合成数据上非零退出："); print(r.stdout[-2000:]); print(r.stderr[-2000:])
            return 1
        adj = json.load(open(out, encoding="utf-8"))

    ok = 0
    for p in paths:
        good = resolve(adj, p)
        ok += good
        print(f"  {'OK  ' if good else 'FAIL'}  {p}")
    print(f"\n路径可解析：{ok}/{len(paths)}")

    # ★ 附加：分支名与 §6 固定措辞的键是否**逐字**一致
    m6 = re.search(r"^## §6(.*?)(?=^## )", txt, re.S | re.M)
    # §6 用小节书写：`**① \`INVALID\`**：` 形式
    heads = re.findall(r"^\*\*[①②③④⑤⑥⑦⑧⑨]\s*`([A-Z][A-Z\-]+)`\*\*", m6.group(1), re.M) if m6 else []
    code_branches = ["INVALID", "INFEASIBLE-POWER", "PASS",
                     "EARLY-LOCALIZED", "NOT-LOCALIZED", "INCONCLUSIVE-BELOW-RESOLUTION"]
    same = heads == code_branches
    print(f"§6 固定措辞小节顺序：{heads}")
    print(f"与六支代码名逐字一致且同序：{'OK' if same else 'FAIL'}")

    allok = ok == len(paths) and same
    print("\n" + ("★ 全部通过" if allok else "★ 有失败"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())