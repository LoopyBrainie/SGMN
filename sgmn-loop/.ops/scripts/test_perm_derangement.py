#!/usr/bin/env python3
"""★ DIR-18 修订 68 · 「内容置换」钩子的最小回归测试（自包含单文件）。

**背景**：派生装置 `sgmn_module_hebb_v2.py` 新增 `content_wrong` 开关
⇒ 写入记忆的**值**取自文档内另一个位置 ⇒ 「记忆内 · 内容错」对照。

**本测试覆盖**（每项都有「应当成立」与「应当不成立」两侧）：
  ① 置换是**排列**（不丢值、不重复）；
  ② 置换**无不动点**（⇒ 每个写入的值都来自**别的** token ⇒ 内容**系统性**错）；
  ③ **同种子 ⇒ 逐位可复现**（确定性）；
  ④ **跨臂固定**（两个独立实例 ⇒ 同一置换序列 ⇒ 配对比较合法）；
  ⑤ ★ **别名陷阱**：`p[i], p[j] = p[j], p[i]` 在 torch 上会**写坏排列**
     （`p[i]` 是**视图** ⇒ 别名）⇒ 必须先取**标量副本**再交换。

用法：PYTHONPATH=/workspace/pylibs python3 test_perm_derangement.py
"""
import torch


def perm_fixed(L1, gen):
    """**修正版**：先取标量副本再交换（见测试 ⑤）。"""
    p = torch.randperm(L1, generator=gen)
    for i in range(L1):
        if int(p[i]) == i:
            j = (i + 1) % L1
            a, b = int(p[i]), int(p[j])
            p[i] = b
            p[j] = a
    return p


def perm_buggy(L1, gen):
    """**缺陷版**（保留以复现缺陷）：直接元组交换 ⇒ torch 视图别名 ⇒ 排列被写坏。"""
    p = torch.randperm(L1, generator=gen)
    for i in range(L1):
        if int(p[i]) == i:
            j = (i + 1) % L1
            p[i], p[j] = p[j], p[i]
    return p


def main():
    fails = []

    def check(ok, msg):
        print(("  ✓ " if ok else "  ✗ ") + msg)
        if not ok:
            fails.append(msg)

    L1S = (2, 3, 5, 17, 40, 67)

    print("① 置换是排列")
    ok_perm = True
    for L1 in L1S:
        g = torch.Generator(device="cpu")
        g.manual_seed(20261106)
        for _ in range(500):
            p = perm_fixed(L1, g)
            if sorted(int(x) for x in p) != list(range(L1)):
                ok_perm = False
                break
    check(ok_perm, f"L1 ∈ {L1S}：500 次抽样全部是排列")

    print("② 无不动点（内容系统性错）")
    resid = 0
    for L1 in L1S:
        g = torch.Generator(device="cpu")
        g.manual_seed(20261106)
        for _ in range(500):
            p = perm_fixed(L1, g)
            resid += sum(1 for i in range(L1) if int(p[i]) == i)
    check(resid == 0, f"不动点残留 = {resid}（应为 0）")

    print("③ 同种子 ⇒ 逐位可复现")
    g1 = torch.Generator(device="cpu")
    g1.manual_seed(20261106)
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(20261106)
    same = all(torch.equal(perm_fixed(40, g1), perm_fixed(40, g2)) for _ in range(5))
    check(same, "同种子两次抽样逐位相同")

    print("④ 跨臂固定（独立实例 ⇒ 同一序列）")
    gA = torch.Generator(device="cpu")
    gA.manual_seed(20261106)
    gB = torch.Generator(device="cpu")
    gB.manual_seed(20261106)
    cross = all(torch.equal(a, b) for a, b in
                zip([perm_fixed(40, gA) for _ in range(6)],
                    [perm_fixed(40, gB) for _ in range(6)]))
    check(cross, "两个独立生成器的置换序列一致 ⇒ 配对比较合法")

    print("⑤ ★ 别名陷阱：缺陷版会被写坏")
    gbug = torch.Generator(device="cpu")
    gbug.manual_seed(20261106)
    broken = 0
    for L1 in (17, 40, 67):
        g2 = torch.Generator(device="cpu")
        g2.manual_seed(20261106)
        for _ in range(200):
            p = perm_buggy(L1, g2)
            if sorted(int(x) for x in p) != list(range(L1)):
                broken += 1
    check(broken > 0, f"缺陷版确实写坏排列（{broken} / 600 次）⇒ 证明必须用修正版")

    print("\n" + ("★ 全部通过" if not fails else f"★ {len(fails)} 项未通过"))
    for m in fails:
        print("   - " + m)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
