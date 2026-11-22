#!/opt/pytorch/bin/python
r"""DIR-43 · **离线自检 ①**：判定表**七支**逐支可达 + 与卡 §5 的工作例一致。

★ **只读源码 / 不训练 / 不占 trainer 算力**（纪律：分支存在 ≠ 分支可达，纪律 104）。
用法：python3 dir43/verify_merge43.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path("/home/ec2-user/sgmn-loop")
sys.path.insert(0, str(ROOT / "dir43"))
import R1_task43 as T43


def main():
    fails = []

    # ── ① `decide()` 的七支逐支可达
    st = T43.selftest_decide()
    print(f"[1] decide() 七支合成自检：{st['n_cases']} 支 ⇒ ok={st['ok']}")
    for c in st["cases"]:
        print(f"    want={c['want']:18s} got={c['got']:18s} counted={c['counted']} "
              f"{'OK' if c['ok'] else '★ FAIL'}")
        if not c["ok"]:
            fails.append(f"decide 分支 {c['want']} 未被触发")
    if not st["secondary_ok"]:
        fails.append("次判据标签三分叉失败")

    # ── ② 计数只发生在 `PASS`
    for c in st["cases"]:
        if c["want"] == "PASS" and not c["counted"]:
            fails.append("PASS 分支未计数")
        if c["want"] != "PASS" and c["counted"]:
            fails.append(f"{c['want']} 分支**错误地**计数")

    # ── ③ 卡 §5 的手算工作例：逐例复算
    card = (ROOT / "revolve/DIR-43/DIRECTION_rev43_v10.md").read_text(encoding="utf-8")
    base = {"n_ok": True, "finite": True,
            "rho_mean": 0.09, "rho_se": 0.03, "rho0_mean": 0.09, "rho0_se": 0.03,
            "rho_diff_mean": 0.10, "rho_diff_se": 0.032,
            "direct_mean": 0.995}
    examples = [
        ("例 A", "PASS", dict(base)),
        ("例 B", "NOT-ADDRESSING", dict(base, rho_mean=0.020)),
        ("例 C", "ANTI-ADDRESSING", dict(base, rho_mean=-0.090)),
        ("例 D", "NOT-SPECIFIC", dict(base, rho_diff_mean=0.020)),
        ("例 E", "EMPTY-CRITERION", dict(base, direct_mean=0.850)),
    ]
    for tag, want, vv in examples:
        got = T43.decide(vv)["outcome"]
        ok = (got == want)
        print(f"[3] {tag}：want={want:18s} got={got:18s} {'OK' if ok else '★ FAIL'}")
        if not ok:
            fails.append(f"{tag} 与卡 §5 不一致")
        if tag not in card:
            fails.append(f"卡 §5 缺 {tag}")

    # ── ④ 卡 §4 的判定表：七个分支名都在卡里，且与 `decide()` 的取值集合一致
    card_branches = set(re.findall(r"`(INCOMPLETE|DIVERGED|ANTI-ADDRESSING|NOT-ADDRESSING|"
                                   r"NOT-SPECIFIC|EMPTY-CRITERION|PASS)`", card))
    code_branches = {"INCOMPLETE", "DIVERGED", "ANTI-ADDRESSING", "NOT-ADDRESSING",
                     "NOT-SPECIFIC", "EMPTY-CRITERION", "PASS"}
    missing = code_branches - card_branches
    print(f"[4] 卡 §4 分支集合：缺 {sorted(missing) if missing else '无'}")
    if missing:
        fails.append(f"卡 §4 缺分支 {sorted(missing)}")

    # ── ⑤ `merge` 不得读探针文件（卡 §4 第 1 支的语义）
    src = (ROOT / "dir43/R1_task43.py").read_text(encoding="utf-8")
    m = re.search(r"def merge\(\):(.*?)\nif __name__", src, re.S)
    body = m.group(1) if m else ""
    reads_probe = bool(re.search(r"probe", body, re.I))
    has_pos = ("read_files_all_shards" in body)
    print(f"[5] merge() 内出现 'probe' 字样：{reads_probe}（应为 False）；"
          f"含**正向断言** read_files_all_shards：{has_pos}（应为 True）")
    if not has_pos:
        fails.append("merge 缺「只读分片产物」的正向断言")
    if reads_probe:
        fails.append("merge() 疑似读取探针文件")

    print("=" * 60)
    if fails:
        print("★ verify_merge43 FAIL：")
        for f in fails:
            print("   -", f)
        return 1
    print("★ verify_merge43 PASS（七支全覆盖 + 工作例一致 + 计数只在 PASS + merge 不读探针）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
