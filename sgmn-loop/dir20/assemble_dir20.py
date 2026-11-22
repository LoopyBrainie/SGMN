#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-20 脚本拼装器：把 _blk_*.py 按卡片 rev 1.2 §5 写死的顺序拼进 R1_xsession.py。

纪律：
  · **逐字派生** —— 除本脚本显式列出的五处改动外，R1_xsession.py 的内容一字未改；
  · **判据的阈值与文字一字未改**（H1″…H7″ / R-1…R-5 全部来自 _blk_crit.py，原样嵌入）；
  · 每处替换都必须**精确命中唯一标记**，否则直接抛错（不做静默跳过）。
"""
import hashlib
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "R1_xsession_base_rev11.py"   # ★ 交接包的中间态（md5 96f4f680126d6831fad8fec9b605bbe7）
OUT = HERE / "R1_xsession.assembled.py"


def read(p):
    return pathlib.Path(p).read_text(encoding="utf-8")


def must_replace(text, old, new, tag, count=1):
    n = text.count(old)
    if n != count:
        raise SystemExit(f"[拼装失败] 标记 {tag!r} 命中 {n} 次（期望 {count}）")
    return text.replace(old, new, count)


def block(name):
    return read(HERE / name).rstrip("\n")


def main():
    t = read(SRC)
    log = []

    # ── ① imports：补 gc / hashlib / os / shutil（_blk_main 用到）────────────
    t = must_replace(
        t,
        "import argparse\nimport json\nimport math\nimport pathlib\nimport random\nimport sys\n",
        "import argparse\nimport gc\nimport hashlib\nimport json\nimport math\nimport os\n"
        "import pathlib\nimport random\nimport shutil\nimport sys\n",
        "imports", 1)
    log.append("① imports += gc/hashlib/os/shutil")

    # ── ② N_ENTRIES：128 → 256（卡片 rev 1.1 §0.2 / rev 1.2 §4）──────────
    t = must_replace(
        t,
        "N_ENTRIES = 128\n",
        "# ★★【DIR-20 · 卡片 rev 1.1 §0.2 / rev 1.2 §4】`N_ENTRIES` 必须 = **256**：\n"
        "#   本协议的写阶段在**训练残留**之上再写 68 条 ⇒ 68 + 68 = **136**\n"
        "#   ⇒ 128 会**回绕**（真值槽不可预测 ⇒ 指标口径错位）⇒ 256 ⇒ 136 ≤ 256 不回绕。\n"
        "#   （实测依据：DIR-19 正式轮 `E5_base_mem` 的 addr=1.0 而 `N1_frozen_base` 的 addr=0.0\n"
        "#     ⇒ 偏移确实来自训练残留 ⇒ 与 DIR-18/19 同构。）\n"
        "N_ENTRIES = 256\n",
        "N_ENTRIES", 1)
    log.append("② N_ENTRIES 128 → 256")

    # ── ③ 文件头 docstring：加 DIR-20 段 ────────────────────────────────
    head_old = '"""\nDIR-19 · R1（**受保护联合训练**）'
    head_new = (
        '"""\n'
        'DIR-20 · R1（**跨会话状态语义**）—— ★ 本脚本是 `R1_protect.py`（DIR-19）'
        '与 `R4_formal.py`（DIR-18）的**逐字派生**。\n'
        '\n'
        '**判据见 ../revolve/DIR-20/DIRECTION_rev12.md（卡片 rev 1.2，**正式轮开跑前冻结**）**：\n'
        'H1″…H7″（7 条）与 R-1…R-5（5 个报告项），**阈值与文字一字未改**（逐字取自 rev 1.1）。\n'
        '装置：`../sgmn_module_hebb_v4.py`（v3 的逐字派生 + `export_state` / `import_state` /\n'
        '`state_fingerprint`），md5 `ad7c1eb621397a37c590b003576e6169`。\n'
        '\n'
        '**相对 DIR-19 的改动（全部留档；卡片 rev 1.2 §5）**：\n'
        '  ① 臂表 ⇒ 11 臂（`X0…X11`），字段含 `export_mode` 与 `session_mode`（卡片 rev 1.1 §2）；\n'
        '  ② 主循环 ⇒ **两段式会话协议**（会话 A 写并导出状态 ⇒ 会话 B 重建实例并导入后只读读出）；\n'
        '  ③ 新增 `eval_arm_readonly()`（只读评测；★ **读期间必须冻结写** ⇒ 缺陷 D84 的修法）；\n'
        '  ④ 判据 ⇒ H1″…H7″ + R-1…R-5（`_blk_crit.py`，逐字嵌入）；\n'
        '  ⑤ Stage A ⇒ **+A20-1…A20-9 / A20-11…A20-14**（`_blk_stageA.py`；A20-10 = 继承 DIR-19 的 30 项）；\n'
        '  ⑥ `N_ENTRIES` 128 ⇒ **256**（卡片 rev 1.1 §0.2 / rev 1.2 §4：写阶段 = 训练残留 68 + 本篇 68 = 136 > 128）；\n'
        '  ⑦ `res["dir"]`/`_DIR_TAG`/`R20_*.json` ⇒ DIR-20 命名空间（缺陷 D78/D79 的修法，逐字沿用）；\n'
        '  ⑧ `import gc/hashlib/os`（会话 B 的显存回收与状态文件指纹）。\n'
        '\n'
        '**未改动**：判据阈值与文字、任务构造、指标口径、训练口径（与 DIR-19 的 `N1_frozen_base`\n'
        '逐字同构）、写配置（`ortho` + `η=1e-1`，`X8_linear` 除外，按修订 20-C 用 `η=1e-5`）。\n'
        '"""\n'
        '"""\nDIR-19 · R1（**受保护联合训练**）')
    t = must_replace(t, head_old, head_new, "docstring", 1)
    log.append("③ 文件头 docstring += DIR-20 段")

    # ── ③b 装置 v4 → v5（缺陷 D89：`explicit_kv` 核没有 W）────────────────
    t = must_replace(t,
                     "from sgmn_module_hebb_v4 import SGMNStack as SGS, grad_report   # noqa: E402",
                     "from sgmn_module_hebb_v5 import SGMNStack as SGS, grad_report   # noqa: E402",
                     "import-v5", 1)
    t = must_replace(t,
                     "import sgmn_module_hebb_v4 as _MOD   # noqa: E402",
                     "import sgmn_module_hebb_v5 as _MOD   # noqa: E402",
                     "import-v5-mod", 1)
    t = must_replace(t,
                     "装置：`../sgmn_module_hebb_v4.py`",
                     "装置：`../sgmn_module_hebb_v5.py`（v4 的逐字派生；★ 缺陷 D89）",
                     "doc-device", 1)
    log.append("③b 装置 v4 ⇒ v5（★ D89：状态搬运对「无 W 的核」可用）")

    # ── ③c 装置 v5 → v6（缺陷 D91：`W=None` 时的形状校验归属）──────────────
    t = must_replace(t,
                     "from sgmn_module_hebb_v5 import SGMNStack as SGS, grad_report   # noqa: E402",
                     "from sgmn_module_hebb_v6 import SGMNStack as SGS, grad_report   # noqa: E402",
                     "import-v6", 1)
    t = must_replace(t,
                     "import sgmn_module_hebb_v5 as _MOD   # noqa: E402",
                     "import sgmn_module_hebb_v6 as _MOD   # noqa: E402",
                     "import-v6-mod", 1)
    t = must_replace(t,
                     "装置：`../sgmn_module_hebb_v5.py`（v4 的逐字派生；★ 缺陷 D89）",
                     "装置：`../sgmn_module_hebb_v6.py`（v5 的逐字派生；★ 缺陷 D91）",
                     "doc-device-v6", 1)
    log.append("③c 装置 v5 ⇒ v6（★ D91）")


    # ── ④ 臂表 ⇒ DIR-20 的 11 臂 ────────────────────────────────────────
    m = re.search(r"# ★ DIR-19 臂表[^\n]*\n(?:.*?\n)*?\]\n", t)
    if not m:
        raise SystemExit("[拼装失败] 找不到 DIR-19 臂表")
    t = t[:m.start()] + block("_blk_arms.py") + "\n" + t[m.end():]
    log.append("④ 臂表 ⇒ DIR-20 11 臂")

    # ── ④b X8_linear 的 η：1e-5 ⇒ 1e-7（★ 缺陷 D92 / 修订 20-C′）─────────
    #   实测依据（Stage A 的 A20-13）：`linear` 的实测 E‖k‖² ≈ **3.5e5–4.0e5**
    #   ⇒ 稳定上界 η < 2/(λ + α‖k‖²) ≈ **5.6e-6** ⇒ ⇒ η=1e-5 **超出上界 ⇒ 写规则发散**
    #   ⇒ smoke#5 实测：`X8_linear` 的 nll = **NaN**（W 发散 ⇒ 读出 NaN ⇒ 判据无法计算）。
    #   ⇒ 修法：采用 **η = 1e-7** = DIR-19 的 `N1_linear_ctrl` 的**既有已验证配置**
    #   （「稳定但惰性」：fid_true ≈ 0 ⇒ 内容取不回）⇒ ⇒ 与卡片 §0.4 的读法完全一致
    #   ⇒ ⇒ 且**跨方向可比**（同一配置）。
    t = must_replace(t,
                     "    (\"X8_linear\",            \"hebbian_full\", 1e-5, \"linear\", \"normal\",        \"full\",       \"cross\"),",
                     "    # ★★【DIR-20 · 卡片 rev 1.5 · 修订 20-C′】η 由 1e-5 ⇒ **1e-7**（DIR-19 的 `N1_linear_ctrl` 配置）\n"
                     "    #   ⇒ 实测 A20-13：linear 的 E‖k‖² ≈ 3.5e5 ⇒ 稳定上界 η < 2/(λ+α‖k‖²) ≈ 5.6e-6\n"
                     "    #   ⇒ 1e-5 超出上界 ⇒ 写规则发散 ⇒ smoke#5 实测 nll = NaN ⇒ 判据无法计算。\n"
                     "    (\"X8_linear\",            \"hebbian_full\", 1e-7, \"linear\", \"normal\",        \"full\",       \"cross\"),",
                     "X8-eta", 1)
    log.append("④b X8_linear 的 η：1e-5 ⇒ 1e-7（★ D92 / 修订 20-C′）")

    # ── ⑤ 只读评测 ⇒ 插入 eval_arm_readonly（★ D84：读期间冻结写）────────
    ro = block("_blk_readonly.py")
    # ★ 只做**改名**（函数体逐字不动）⇒ 读出逻辑与 DIR-19 的 `eval_arm` 同口径。
    #   冻结写（D84 的修法）放在**外层包装函数**里 ⇒ 不污染原函数体。
    if "def eval_arm_readonly(st, d, ctrl_ids=None):" not in ro:
        raise SystemExit("[拼装失败] readonly 块缺少 eval_arm_readonly")
    ro = ro.replace(
        "def eval_arm_readonly(st, d, ctrl_ids=None):",
        "def _eval_arm_readonly_body(st, d, ctrl_ids=None):", 1)
    ro += (
        "\n\n"
        "def eval_arm_readonly(st, d, ctrl_ids=None):\n"
        "    \"\"\"【DIR-20】只读评测的**包装层**：读期间冻结写，读后恢复调用方的冻结状态。\n"
        "\n"
        "    ★ 缺陷 D84 的修法（卡片 rev 1.2 §5.1）：真正的读出逻辑在\n"
        "      `_eval_arm_readonly_body`（逐字沿用 DIR-19 的 `eval_arm` 口径）。\n"
        "    \"\"\"\n"
        "    _flags = [m.no_write for m in st.mems.values()]\n"
        "    st.freeze_write()\n"
        "    try:\n"
        "        return _eval_arm_readonly_body(st, d, ctrl_ids)\n"
        "    finally:\n"
        "        for m, f in zip(st.mems.values(), _flags):\n"
        "            m.no_write = f\n")
    t = must_replace(t, "def eval_base(model, eval_docs, ctrl_ids=None):",
                     ro + "\n\ndef eval_base(model, eval_docs, ctrl_ids=None):", "readonly", 1)
    log.append("⑤ 插入 eval_arm_readonly（★ D84：冻结写）")

    # ── ⑥ 主循环 ⇒ DIR-20 会话协议（★ D85/D88/D90/D95/D96）──────────────────
    #   ★ 本块（`_blk_main.py`）**已含**全部主循环修复：
    #     D85（真值槽跨会话传递）· D88（会话 A 打印不假定有 md5）· D90（`_keepmodel` 按 (臂,种子)）
    #     D95（每轮开始清空遗留状态文件）· **D96（★ 按臂推进 ⇒ 磁盘峰值 18 GB ⇒ 1.7 GB）**
    mn = block("_blk_main.py")
    for tag in ("meta[i] = {\"true_slot\"", "_mm = _smeta.get", "_m0 = next(",
                "_keepmodel[(name, seed)] = modelA", "_n_stale",
                "_free_mb", "shutil.disk_usage"):
        if tag not in mn:
            raise SystemExit(f"[拼装失败] 新主循环块缺少标记：{tag}")
    # D96 的结构断言：**阶段 2（会话 B）必须嵌套在臂循环内** ⇒ 否则磁盘峰值 ≈ 18 GB
    if "\n    # ══════════ 阶段 2：会话 B" in mn:
        raise SystemExit("[拼装失败] 阶段 2 未嵌套在臂循环内（★ D96）")
    if "\n        # ══════════ 阶段 2：会话 B" not in mn:
        raise SystemExit("[拼装失败] 找不到嵌套的阶段 2（★ D96）")
    m = re.search(r"    # ---------------- 主循环\n(?:.*\n)*?(?=    # ---------------- 汇总\n)", t)
    if not m:
        raise SystemExit("[拼装失败] 找不到 DIR-19 主循环区间")
    t = t[:m.start()] + mn + "\n" + t[m.end():]
    log.append("⑥ 主循环 ⇒ 两段式会话协议（★ D85）")

    # ── ⑦ 判据 ⇒ H1″…H7″ + R-1…R-5（并删掉 DIR-19 的未被使用的辅助函数）──
    m_help = re.search(r"    def flat\(name, field\):\n(?:.*\n)*?(?=    # ══════════════════ ★★【DIR-19】判据)", t)
    if not m_help:
        raise SystemExit("[拼装失败] 找不到 DIR-19 的 flat/seed_delta/check_pair 辅助函数块")
    t = t[:m_help.start()] + t[m_help.end():]
    log.append("⑦a 删除 DIR-19 的 flat/seed_delta/check_pair（DIR-20 不使用；"
               "★ 防「同名遮蔽」类缺陷）")
    m = re.search(r"    # ══════════════════ ★★【DIR-19】判据[^\n]*\n(?:.*\n)*?(?=    # —— 汇总报告 ——\n)", t)
    if not m:
        raise SystemExit("[拼装失败] 找不到 DIR-19 判据区间")
    t = t[:m.start()] + block("_blk_crit.py") + "\n\n" + t[m.end():]
    log.append("⑦b 判据 ⇒ H1″…H7″ + R-1…R-5")

    # ── ⑧ Stage A：插入 A20-1…A20-9 + A20-11…A20-14 ────────────────────
    t = must_replace(t,
                     "    # ---------------- ★ Stage A 硬闸",
                     block("_blk_stageA.py") + "\n    # ---------------- ★ Stage A 硬闸",
                     "stageA", 1)
    log.append("⑧ Stage A += A20-1…A20-9 + A20-11…A20-14")

    # ── ⑨ 文案：DIR-17 / DIR-19 ⇒ DIR-20 ─────────────────────────────────
    t = must_replace(t,
                     'print("=== DIR-17 R1（脚本 v1.1 · 卡片 rev 1.3 任务构造）：可寻址读取 + 读侧训练 ===", flush=True)',
                     'print("=== DIR-20 R1（脚本 v1.0 · 卡片 rev 1.2 跨会话状态语义：两段式会话协议）===", flush=True)',
                     "banner", 1)
    t = must_replace(t,
                     'print("\\\\n【DIR-19 汇总】", flush=True)',
                     'print("\\\\n【DIR-20 汇总】", flush=True)',
                     "summary-banner", 1)
    log.append("⑨ 文案 DIR-17/DIR-19 ⇒ DIR-20")

    # ── ⑩ 缺陷 D93：增量文件必须**按 revision 隔离**（纪律 #37 的延伸）──────
    #   现象（smoke#6 实测）：`X8_linear` 的 η 已改（1e-5 ⇒ 1e-7），但**增量复用**了
    #   上一版（η=1e-5）的行 ⇒ nll 仍是 NaN ⇒ H4″ 读到 NaN ⇒ **「判据从未被计算」**。
    #   ⇒ 根因：增量文件名只带「方向 + 规格」，**不带 revision** ⇒ ⇒ 跨 revision 混用
    #     （LOOP-MECHANISM §1.2：新 revision 内的结果**不得**与旧 revision 并列比较）。
    #   ⇒ 修法：`_DIR_TAG` 加 revision ⇒ `formal_incremental_DIR20r16_{smoke,full}.json`。
    t = must_replace(t, '_DIR_TAG = "DIR20"', '_DIR_TAG = "DIR20r16"', "dir-tag", 1)
    log.append("⑩ 增量文件按 revision 隔离（★ D93：DIR20r15）")

    # ── ⑪ 缺陷 D94：收尾汇总打印仍调用 DIR-19 的 `dppl` / `ben_ret` ⇒ NameError ──
    t = must_replace(
        t,
        "        print(f\"  {nm:20s} nll={agg[nm]['nll']:.4f} benefit={ben(nm):+.4f} \"\n"
        "              f\"PPL={agg[nm].get('ppl')} ΔPPL={dppl(nm):+.4f} dkv={agg[nm].get('dkv')} \"\n"
        "              f\"base_drift={agg[nm].get('base_drift')} 保留率={ben_ret(nm):.3f}\", flush=True)\n",
        "        print(f\"  {nm:20s} nll={agg[nm]['nll']:.4f} benefit={ben(nm):+.4f} \"\n"
        "              f\"hit={agg[nm].get('hit')} addr_slot={agg[nm].get('addr_slot')} \"\n"
        "              f\"fid_true={agg[nm].get('fid_true')} cos_y_vtrue={agg[nm].get('cos_y_vtrue')}\",\n"
        "              flush=True)\n",
        "summary-print", 1)
    log.append("⑪ 收尾汇总打印 ⇒ DIR-20 口径（★ D94）")

    OUT.write_text(t, encoding="utf-8")
    md5 = hashlib.md5(t.encode("utf-8")).hexdigest()
    print("\n".join("  " + x for x in log))
    print(f"\n拼装完成 ⇒ {OUT}")
    print(f"行数 = {t.count(chr(10)) + 1}；md5 = {md5}")


if __name__ == "__main__":
    main()
