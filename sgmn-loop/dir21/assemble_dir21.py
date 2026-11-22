#!/usr/bin/env python3
"""DIR-21 · 拼装器：DIR-20 的**正式脚本** ⇒ DIR-21 的 `R1_granularity.py`。

★ 派生纪律（沿用本 loop 的「无代码复用」条款）：**逐字派生 + md5 记录 + Stage A 全面复验**。
★ 每一处替换都**断言唯一命中**（命中数 ≠ 1 ⇒ 直接失败 ⇒ **绝不静默跳过**）。
★ 判据的阈值与文字**不在本文件里**（在卡片 rev 1.1 §3，已冻结）。
"""
import hashlib
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "R1_xsession.py"          # ★ DIR-20 的正式脚本（权威副本）
OUT = HERE / "R1_granularity.py"


def read(p):
    return p.read_text(encoding="utf-8")


def md5b(b):
    return hashlib.md5(b).hexdigest()


def must_replace(text, old, new, tag, count=1):
    n = text.count(old)
    if n != count:
        raise SystemExit(f"[拼装失败] {tag}: 命中 {n} 次（期望 {count}）")
    return text.replace(old, new, count)


def block(name):
    return read(HERE / name).rstrip("\n")


def main():
    t = read(SRC)
    log = []

    # ── ① 装置 v6 ⇒ v7 ────────────────────────────────────────────────────
    t = must_replace(
        t,
        "from sgmn_module_hebb_v6 import SGMNStack as SGS, grad_report   # noqa: E402\n"
        "import sgmn_module_hebb_v6 as _MOD   # noqa: E402\n",
        "from sgmn_module_hebb_v7 import SGMNStack as SGS, grad_report   # noqa: E402\n"
        "import sgmn_module_hebb_v7 as _MOD   # noqa: E402\n",
        "device", 1)
    log.append("① 装置 ⇒ v7（只加写步粒度 + 键索引按 token 位置）")

    # ── ② 常量：种子 / 规格 / 步轴 / 臂表 ───────────────────────────────────
    t = must_replace(
        t,
        "SEEDS = [20261028, 20261104, 7]\nN_EVAL, N_TRAIN, FILLER_TOK = 60, 200, 30\n",
        "SEEDS = [20261028, 20261104, 7, 20261111, 20261118]   # ★ DIR-21：5 个种子\n"
        "N_EVAL, N_TRAIN, FILLER_TOK = 520, 200, 30\n"
        "ETA_BASE = 1e-1   # ★ DIR-21：写侧速率基准（`ortho` 可用配置，沿用 DIR-18/19/20）\n"
        "# ★★ 曲线横轴 = **在线步数**（跨粒度可比：文档数随每篇步数 m 变 ⇒ 不用文档轴）\n"
        "STEP_GRID_FULL = [0, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]\n"
        "STEP_GRID_SMOKE = [0, 16, 32, 48, 64, 96, 128, 192, 256]\n"
        "STEP_GRID = STEP_GRID_FULL\n"
        "# ★ DIR-21 的臂表：(name, eta, write_stride, stride_pick)\n"
        "#   ★ 唯一自变量 = **写步调度**（每隔几个 token 触发一次写规则）\n"
        "#   `G_seg4_alt` 用 stride=18 ⇒ 与 `G_seg4` **步数相同、调度不同** ⇒ 隔离「衰减只认步数」\n"
        "#   `G_tok_rate` 把 η 缩到 1/17 ⇒ 与 `G_seg4` 的**每篇累积 (ηλ) 相同** ⇒ 测「η_S ≡ m·η_W」\n"
        "ARMS21 = [\n"
        "    (\"G_tok\",      ETA_BASE,        1, \"last\"),\n"
        "    (\"G_seg4\",     ETA_BASE,       17, \"last\"),\n"
        "    (\"G_seg4_alt\", ETA_BASE,       18, \"last\"),\n"
        "    (\"G_tok_rate\", ETA_BASE / 17.0,  1, \"last\"),\n"
        "    (\"G_doc\",      ETA_BASE,       68, \"last\"),\n"
        "]\n",
        "constants", 1)
    log.append("② 常量 ⇒ 5 种子 / 步轴 / 臂表")

    # ── ③ 命名空间：DIR-21 ─────────────────────────────────────────────────
    t = must_replace(t, '_DIR_TAG = "DIR20r16"', '_DIR_TAG = "DIR21r12"', "dir-tag", 1)
    t = must_replace(t, 'R20_smoke_results.json" if args.smoke else "R20_results.json"',
                     'R21_smoke_results.json" if args.smoke else "R21_results.json"', "result-name", 2)
    log.append("③ 命名空间 ⇒ DIR21r12 / R21_*.json")

    # ── ④ 臂表：DIR-20 的 11 臂不再使用（换成 ARMS21）──────────────────────
    m = re.search(r"ARMS = \[\n(?:.*\n)*?\]\n", t)
    if not m:
        raise SystemExit("[拼装失败] 找不到 DIR-20 的 ARMS 块")
    t = t[:m.start()] + ("# ★ DIR-21 的臂表见上面的 `ARMS21`（DIR-20 的 11 臂在本方向不使用）\n"
                         "ARMS = []\n") + t[m.end():]
    log.append("④ DIR-20 的 ARMS ⇒ ARMS21")

    # ── ⑤ Stage A：在 DIR-20 的闸之前插入 DIR-21 的 9 项 ────────────────────
    t = must_replace(
        t,
        "    print(\"[Stage A · DIR-20] 13 项断言完成\", flush=True)\n",
        "    print(\"[Stage A · DIR-20] 13 项断言完成\", flush=True)\n"
        + block("_blk_stageA21.py") + "\n",
        "stageA21", 1)
    log.append("⑤ Stage A += A21-1…A21-9（A21-10 = 继承 DIR-20 的 44 项）")

    # —— 判据块 ⇒ DIR-21 的判据（+ 报告项 + 汇总打印）——
    #   ★ 用**行号定位**（不用正则：中文注释里的符号会让正则难写且脆弱）
    _lines = t.splitlines(keepends=True)
    _s = None
    for _i, _ln in enumerate(_lines):
        if "★★【DIR-20】判据" in _ln:
            _s = _i
            break
    if _s is None:
        raise SystemExit("[拼装失败] 找不到 DIR-20 的判据块起点")
    _e = None
    for _i in range(_s, len(_lines)):
        if "缺陷 D76：原写" in _lines[_i]:
            _e = _i
            break
    if _e is None:
        raise SystemExit("[拼装失败] 找不到 DIR-20 判据块的终点")
    t = "".join(_lines[:_s]) + block("_blk_crit21.py") + "\n" + "".join(_lines[_e:])
    # ★ 防再入（曾因此把汇总插进了判据函数体 ⇒ `mean_curve` 未绑定就崩溃）：
    #   ① 记录拼入后**剩余**多少个「# ---------------- 汇总」锚（必须恰好 1，给⑦′用）；
    #   ② 判据块必须**包含** `def mean_curve`（防拼错位置）。
    if t.count("# ---------------- 汇总") != 1:
        raise SystemExit(f"[拼装失败] 汇总锚数量异常（{t.count('# ---------------- 汇总')}）")
    if "def mean_curve" not in t:
        raise SystemExit("[拼装失败] 判据块未包含 mean_curve（拼错位置）")
    log.append("⑥ 判据 ⇒ H21-1/1b/2/3/4 + Rp-1…Rp-4")

    # ── ⑥′ DIR-20 的汇总打印（ben/agg）⇒ DIR-21 的步轴汇总 ──────────────
    _lines = t.splitlines(keepends=True)
    _s = None
    for _i, _ln in enumerate(_lines):
        if _ln.lstrip().startswith('print("\\n【DIR-20 汇总】", flush=True)'):
            _s = _i
            break
    if _s is None:
        raise SystemExit("[拼装失败] 找不到 DIR-20 的汇总打印块")
    _e = None
    for _i in range(_s, len(_lines)):
        if "缺陷 D76：原写" in _lines[_i]:
            _e = _i
            break
    if _e is None:
        raise SystemExit("[拼装失败] 找不到 DIR-20 汇总打印块的终点")
    _sum = [
        '    # ══════════════════ ★★【DIR-21】汇总（替换 DIR-20 的 agg 汇总）══════',
        '    #   ★ 各臂写下的条目数不同（136/8/136/8/2 每篇）⇒ **绝对**数不可跨臂比较 ⇒',
        '    #   只汇总「比例 / 归一化 / 步轴」读数（rev 1.2 §0.4 证据边界）。',
        '    agg = {}',
        '    for name in [a for a, _, _, _ in ARMS21]:',
        "        _hs = mean_curve(name, 'hit_mean')",
        "        _fs = mean_curve(name, 'fid_mean')",
        "        _ns = mean_curve(name, 'nll_tgt')",
        '        _kk, _vv = capacity_knee(name)',
        '        _hlv, _cenv = half_life(name)',
        '        _mval = float(_m_of(name))',
        "        agg[name] = {'m': _mval, 'hit_step0': _hs.get(0, float('nan')),",
        "                     'fid_step0': _fs.get(0, float('nan')), 'nll_step0': _ns.get(0, float('nan')),",
        "                     'half_life_step': _hlv, 'capacity_knee_docs': _kk}",
        "        print(('  %-14s m=%4.0f nll(step0)=%.3f hit(0/512)=%.3f/%.3f fid(0/512)=%.3f/%.3f'",
        "               ' K*=%.0f 半衰期=%.0f步') % (name, _mval, _ns.get(0, float('nan')),",
        "               _hs.get(0, float('nan')), _hs.get(512, float('nan')),",
        "               _fs.get(0, float('nan')), _fs.get(512, float('nan')), _kk, _hlv), flush=True)",
        '    res["arms"] = agg',
        '',
    ]
    t = "".join(_lines[:_s]) + "\n".join(_sum) + "".join(_lines[_e:])
    log.append("⑥′ DIR-20 汇总打印 ⇒ DIR-21 的步轴汇总")

    # ── ⑦ 主循环 ⇒ DIR-21（训练一次/种子 + 步轴扫描）──────────────────────
    _lines = t.splitlines(keepends=True)
    _s = None
    for _i, _ln in enumerate(_lines):
        if _ln.startswith("    # ---------------- 主循环"):
            _s = _i
            break
    if _s is None:
        raise SystemExit("[拼装失败] 找不到 DIR-20 的主循环起点")
    _e = None
    for _i in range(_s, len(_lines)):
        if _lines[_i].startswith("    # ---------------- 汇总"):
            _e = _i
            break
    if _e is None:
        raise SystemExit("[拼装失败] 找不到 DIR-20 主循环的终点")
    t = "".join(_lines[:_s]) + block("_blk_main21.py") + "\n" + "".join(_lines[_e:])
    # ── ⑦′ 汇总：删除 DIR-20 的 agg 汇总块（⑥′ 已把步轴汇总放进判据函数体）──
    _lines = t.splitlines(keepends=True)
    _s = None
    for _i, _ln in enumerate(_lines):
        if _ln.startswith("    # ---------------- 汇总"):
            _s = _i
            break
    if _s is None:
        raise SystemExit("[拼装失败] 找不到 DIR-20 的汇总块")
    _e = None
    for _i in range(_s, len(_lines)):
        if "res[\"arms\"] = agg" in _lines[_i]:
            _e = _i + 1
            break
    if _e is None:
        raise SystemExit("[拼装失败] 找不到 DIR-20 汇总块的终点")
    t = "".join(_lines[:_s]) + "".join(_lines[_e:])
    log.append("⑦′ 汇总 ⇒ 删除（步轴汇总已并入判据函数体）")

    # ── ⑧ main()：smoke 的规格 + 步轴 ─────────────────────────────────────
    t = must_replace(
        t,
        "    n_eval = 12 if args.smoke else N_EVAL\n"
        "    n_train = 40 if args.smoke else N_TRAIN\n"
        "    seeds = SEEDS[:2] if args.smoke else SEEDS\n",
        "    global STEP_GRID\n"
        "    STEP_GRID = STEP_GRID_SMOKE if args.smoke else STEP_GRID_FULL\n"
        "    n_eval = 140 if args.smoke else N_EVAL\n"
        "    n_train = 40 if args.smoke else N_TRAIN\n"
        "    seeds = SEEDS[:2] if args.smoke else SEEDS\n",
        "main-spec", 1)
    log.append("⑧ main() ⇒ smoke 规格 + 步轴")

    # ── ⑨ 头部说明（DIR-21）───────────────────────────────────────────────
    t = must_replace(
        t,
        "装置：`../sgmn_module_hebb_v6.py`（v5 的逐字派生；★ 缺陷 D91）",
        "装置：`../sgmn_module_hebb_v7.py`（v6 的逐字派生；★ **只加写步粒度 + 键索引按 token 位置**）",
        "header-device", 1)

    OUT.write_text(t, encoding="utf-8")
    print("拼装完成 ⇒", OUT)
    print("行数 =", len(t.splitlines()), "；md5 =", md5b(t.encode("utf-8")))
    for line in log:
        print("  ·", line)


if __name__ == "__main__":
    main()
