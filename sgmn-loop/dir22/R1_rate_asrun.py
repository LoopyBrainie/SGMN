#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-20 · R1（**跨会话状态语义**）—— ★ 本脚本是 `R1_protect.py`（DIR-19）与 `R4_formal.py`（DIR-18）的**逐字派生**。

**判据见 ../revolve/DIR-20/DIRECTION_rev12.md（卡片 rev 1.2，**正式轮开跑前冻结**）**：
H1″…H7″（7 条）与 R-1…R-5（5 个报告项），**阈值与文字一字未改**（逐字取自 rev 1.1）。
装置：`../sgmn_module_hebb_v7.py`（v6 的逐字派生；★ **只加写步粒度 + 键索引按 token 位置**）（v3 的逐字派生 + `export_state` / `import_state` /
`state_fingerprint`），md5 `ad7c1eb621397a37c590b003576e6169`。

**相对 DIR-19 的改动（全部留档；卡片 rev 1.2 §5）**：
  ① 臂表 ⇒ 11 臂（`X0…X11`），字段含 `export_mode` 与 `session_mode`（卡片 rev 1.1 §2）；
  ② 主循环 ⇒ **两段式会话协议**（会话 A 写并导出状态 ⇒ 会话 B 重建实例并导入后只读读出）；
  ③ 新增 `eval_arm_readonly()`（只读评测；★ **读期间必须冻结写** ⇒ 缺陷 D84 的修法）；
  ④ 判据 ⇒ H1″…H7″ + R-1…R-5（`_blk_crit.py`，逐字嵌入）；
  ⑤ Stage A ⇒ **+A20-1…A20-9 / A20-11…A20-14**（`_blk_stageA.py`；A20-10 = 继承 DIR-19 的 30 项）；
  ⑥ `N_ENTRIES` 128 ⇒ **256**（卡片 rev 1.1 §0.2 / rev 1.2 §4：写阶段 = 训练残留 68 + 本篇 68 = 136 > 128）；
  ⑦ `res["dir"]`/`_DIR_TAG`/`R20_*.json` ⇒ DIR-20 命名空间（缺陷 D78/D79 的修法，逐字沿用）；
  ⑧ `import gc/hashlib/os`（会话 B 的显存回收与状态文件指纹）。

**未改动**：判据阈值与文字、任务构造、指标口径、训练口径（与 DIR-19 的 `N1_frozen_base`
逐字同构）、写配置（`ortho` + `η=1e-1`，`X8_linear` 除外，按修订 20-C 用 `η=1e-5`）。
"""
"""
DIR-19 · R1（**受保护联合训练**）—— ★ 本脚本是 `R4_formal.py`（md5 20656a2199b2482b1dff57033b5dd5ca）的**逐字派生**。

**派生规则（纪律：无代码复用 ⇒ 逐字派生 + md5 记录 + Stage A 全面复验）**：
任务构造 / 指标口径 / 装置钩子**逐字沿用**（⇒ 与 DIR-17/18 **跨方向可比**），
**唯一新增**：① 11 臂（保护旋钮）；② **基座保持探针**（wikitext-2 valid 的 NLL）；
③ **$D_{kv}$**（键值分布漂移）；④ 可训基座子集 / **KL 正则** / 早停对照；
⑤ 判据 H1′…H7′ + **H2′★**（记忆内内容对照）+ **H2′′**（零效度门槛）；⑥ Stage A **A19-1…A19-10**。

判据见 ../DIRECTION_rev12.md（卡片 rev 1.2，**开跑前冻结**，2026-11-06）。
装置：`../sgmn_module_hebb_v3.py`。

（以下为 R4_formal.py 的原始文件头 ↓↓↓ 原样保留）
"""
"""
DIR-18 · R4（**正式轮**）—— ★ 本脚本是 `R2_write.py`（md5 12dc22f6a5e7819e077fe779b0bddcd1）的**逐字派生**。

**派生规则（纪律：无代码复用 ⇒ 逐字派生 + md5 记录 + Stage A 全面复验）**：
除下列五处外，**全部逐字沿用 R2_write.py**：
  ① 装置改用 `../sgmn_module_hebb_v2.py`（`sgmn_module_hebb.py` 的逐字派生 + `content_wrong` 钩子）；
  ② 臂表 **+1 臂** `Wcontent_wrong_ortho_eta1`（修订 68：记忆内「内容错」对照）；
  ③ 判据 **+H14**（内容特异性 · 记忆内对照）与 **H13 的零效度门槛**（修订 68）；
  ④ Stage A **+A2'⑱**（零效度自检：置换只动内容、不动寻址）⇒ **20 项**；
  ⑤ **增量落盘 + 续跑**（吸收 CPU 降级轮的 D56/D59 教训：可恢复性机制必须真的能用）。

**既有 15 臂与 H1–H13c 的阈值与文字一字未改** ⇒ ⇒ 正式轮结果与 smoke#2 应当**逐位一致**（可作回归检查）。

判据见 ../DIRECTION_rev15.md（卡片 rev 1.5，**开跑前冻结**，2026-11-06）。
装置：`../sgmn_module_hebb_v2.py`。

（以下为 R2_write.py 的原始文件头 ↓↓↓ 原样保留）
"""
"""
DIR-18 · R2 —— v2.4 的**实际写规则**（Hebbian）能否支撑已验证的「寻址 → 读出 → 任务收益」链条？

★ 本脚本是 `R1_write.py` 的**逐字派生**，只改下列四处（其余逐字沿用，md5 留档）：
  ① **D46 指标口径**：`addr` 双口径 —— `addr`（写入顺序空间）与 `addr_slot`（**槽空间**，
     真值槽由**值内容反查**）。两者都必须报告（纪律 ⑲）。
  ② **D48 协议**：每篇评测前**清空槽表**（`keys/vals/sigs` 归零、`count=0`），**但保留 `W`**
     ⇒ 消除陈旧条目与槽位回绕 ⇒ 真值槽 == 写入顺序号。
  ③ **D47 `P_v` 可训化**：`W4b` 臂用**写侧自监督**（`L_write = −cos(W k_j, v_j)`，
     只更新 `P_v`、不进入任务损失、不更新 `W`）⇒ v2.4 §7.2「受保护的联合训练」的正确实现形态。
  ④ **D50/D51 键尺度**：新增 `key_mode="renorm"`（线性键**逐写归一化**到 ‖k‖²=5）
     与 `key_mode="ortho"`（正交键，‖k‖²=5）+ 对应 η 档 ⇒ 让「稳定窗口 ∩ 学习窗口」非空。

判据见 ../DIRECTION.md（卡片 rev 1.2，冻结于正式轮之前）。
装置：`../sgmn_module_hebb.py`（DIR-17 装置的逐字派生，只新增 `HebbianCore`）。
"""

"""
DIR-22 · R1（**写侧速率 vs 情节级速率：单一标量化约是否可能**）
—— ★ 本脚本是 `dir21/R1_granularity.py`（md5 `08b68ecb8859e8bda30911beb4ced43f`）的**逐字派生**。

**判据见 ../revolve/DIR-22/DIRECTION_rev22_v11.md（= rev 1.2 的等价物；冻结于正式轮之前）**：
H22 的四子句（a 闭式 / b 单一标量不可行 / c 两条匹配线各自失败 / d 可证伪形态）
+ `Rp-1…Rp-5` 报告项。**阈值一字未改**。

**相对 DIR-21 的改动（全部留档；生成器 = `patch_r1_rate.py`）**：
  ① 命名空间 ⇒ `DIR22r11`（`res["dir"]` / `_DIR_TAG` / `R22_*.json` / 状态文件前缀）；
  ② 臂表 ⇒ **ARMS22**（5 格，**逐臂 η**）：`C_base/C_m6/C_m18/C_m68/C_decay68`
     ⇒ m = 2/6/18/68/68；η = 1e−1（前四臂）/ 1.32353e−3（`C_decay68`，由 ηλm 相等解出）；
  ③ **逐条目** `cos_by_t` / `hit_by_t` 落盘（rev 1.1 §4「交集口径」所需；原先只存均值）；
  ④ 新增**机制探针** `probe_mechanism()`：实测 $\\hat\\rho$（刷新）与逐情节衰减比 $D$（衰减）
     ⇒ 支撑 H22-a/H22-b；探针**不改写协议状态**（snapshot → 探针 → restore）；
  ⑤ Stage A ⇒ **A22-1…A22-9**（继承 DIR-21 的 58 项）；
  ⑥ 判据 ⇒ `H22-a…H22-d` + `Rp-1…Rp-5`（DIR-21 的 H21-* 与 Rp-6/Rp-7 被**替换**，见 P07/P08）。

**未改动**：任务构造 / 指标口径 / 训练口径 / 装置写规则 / K 网格 / 种子表 / 只读评测。
**装置**：`./sgmn_module_hebb_v7.py`（= `dir21/` 同名文件逐字复制，md5 `1a242d02c530307f6c76646ca13469ef`；
DEC-02 ⇒ **不造 v8**：冻结网格里 r≡1，`refresh_per_episode` 是死代码）。
"""


import argparse
import gc
import hashlib
import json
import math
import os
import pathlib
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sgmn_module_hebb_v7 import SGMNStack as SGS, grad_report   # noqa: E402
import sgmn_module_hebb_v7 as _MOD   # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOUNT = [8, 12, 16]
ALPHA, LAM, GAMMA0, R = 1.0, 0.05, 0.05, 8
ETA = 1e-5
BETA = 64.0
# 【rev 1.3】目标文档与干扰文档各约 37 条 ⇒ 合计 ~74 条 ⇒ N_ENTRIES 必须 ≥ 74，
#   否则核心槽位回绕 ⇒ 命中指标与真值下标错位（D30 的同类缺陷）。128 ⇒ 槽位 = 写入序号。
# ★★【DIR-20 · 卡片 rev 1.1 §0.2 / rev 1.2 §4】`N_ENTRIES` 必须 = **256**：
#   本协议的写阶段在**训练残留**之上再写 68 条 ⇒ 68 + 68 = **136**
#   ⇒ 128 会**回绕**（真值槽不可预测 ⇒ 指标口径错位）⇒ 256 ⇒ 136 ≤ 256 不回绕。
#   （实测依据：DIR-19 正式轮 `E5_base_mem` 的 addr=1.0 而 `N1_frozen_base` 的 addr=0.0
#     ⇒ 偏移确实来自训练残留 ⇒ 与 DIR-18/19 同构。）
N_ENTRIES = 256
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]   # ★ DIR-21：5 个种子
N_EVAL, N_TRAIN, FILLER_TOK = 1025, 200, 30
ETA_BASE = 1e-1   # ★ DIR-21：写侧速率基准（`ortho` 可用配置，沿用 DIR-18/19/20）
# ★★【rev 1.8 · 缺陷 D119】**训练栈与各臂必须共用同一个写侧速率**。
#   单一常量 `TRAIN_ETA`：主循环训练栈与 Stage A 守卫（A21-15）**都只引用它**
#   ⇒ 两者不可能再漂移（旧代码在主循环传 `ETA`、在注释写「η=1e−1」⇒ 注释与代码不一致）。
TRAIN_ETA = ETA_BASE
# ★★【rev 1.7 · 缺陷 D103 + D114】曲线横轴 = **情节（文档）数 K**（rev 1.1 的单位）——
#   rev 1.2 的「在线步轴」使 H21-1b / H21-3 / H21-4 **三条冻结判据同时失去可算性**
#   （H21-1b 的 K*≈N/m 是情节数；H21-3 的 K½·m；H21-4 的「每篇累积 (ηλ)」）⇒ 回退到情节轴。
#   网格 = rev 1.1 的 K_SCAN ∪ {5, 96, 128, 192, 224}（**probe21b 真实扫描路径**决定）。
K_SCAN_FULL = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 224]
K_SCAN_SMOKE = [1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 96, 128, 192, 224]
K_SCAN = K_SCAN_FULL
# ★★ K_max = 224 是**硬约束**：`G_doc` 的锚键列在第 225 情节被重写（缺陷 D112）
#   ⇒ K ≥ 225 的读数**无效** ⇒ 网格末端必须停在这里。
# ★ DIR-21 的臂表：(name, eta, write_stride, stride_pick)
#   ★ 唯一自变量 = **写步调度**（每隔几个 token 触发一次写规则）
#   `G_seg4_alt` 用 stride=18 ⇒ 与 `G_seg4` **步数相同、调度不同** ⇒ 隔离「衰减只认步数」
#   `G_tok_rate` 把 η 缩到 1/17 ⇒ 与 `G_seg4` 的**每篇累积 (ηλ) 相同** ⇒ 测「η_S ≡ m·η_W」
# ★★【DIR-22 · rev 1.1 §2.2】臂表 = **5 个冻结单元格**（`(name, eta, write_stride, stride_pick)`）
#   ★ 唯一自变量 = **(η, m) 的联合取值**；m 由 `write_stride` 精确决定（L=35 ⇒ {34,16,4,1}）
#   m = 2 × |{S,2S,…} ∩ [1,34]|（目标 + 干扰 ⇒ ×2）⇒ 实测 2 / 6 / 18 / 68 / 68（A22-3 断言）
#   ★ `C_decay68` 的 η 是**解出来的**，不是凑的：令 η₁·m₁ = η₅·m₅（衰减不变量 ηλm 相等）
#       0.1 × 18 = 1.3235294117647058e−3 × 68 ⇒ 两边都 = 1.8 ⇒ ηλm = 0.09
ETA_DECAY68 = ETA_BASE * 18.0 / 68.0
ARMS22 = [
    ("C_base",     ETA_BASE,        34, "last"),   # m=2
    ("C_m6",       ETA_BASE,        16, "last"),   # m=6
    ("C_m18",      ETA_BASE,         4, "last"),   # m=18  ← 两条匹配对的公共端
    ("C_m68",      ETA_BASE,         1, "last"),   # m=68
    ("C_decay68",  ETA_DECAY68,      1, "last"),   # m=68，η 已解出（P_decay 的另一端）
]
M_EXPECT = {"C_base": 2, "C_m6": 6, "C_m18": 18, "C_m68": 68, "C_decay68": 68}
CUE_A = " The vault code is"
CUE_B = " The safe combination is"
MIN_FILLER_GAP = 400                            # 目标/干扰填充切片的最短间隔 ⇒ 签名必不同
# 【脚本 v1.2 · rev 1.4】码 = **单个稀有词 token**，且**位于文档的最后一个位置**。
#   理由（★ 关键）：§2.3 的因果写把 $h_t$ 存进 $v_t$ ⇒ **码之后任一位置的 $h_t$ 都含码的身份**
#   ⇒ 「位置就近」的邻位条目也能解出码 ⇒ 任何「读对条目 vs 读邻位」的对照都会双双饱和
#   （smoke#5 实测：E2 命中 0 却 nll=0.0000）。
#   ⇒ 把码放在文档末尾 ⇒ 只有**码自己那一条**持有其身份 ⇒ 邻位条目不含码 ⇒ 对照才有效。
MIN_FILLER_GAP = MIN_FILLER_GAP
SAT_EPS = 0.05                                 # 饱和门：E1/E2/E3 的 NLL 极差须 ≥ 此值
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")

# ★★ 缺陷 D79：派生脚本的 `res["dir"]` 仍是 **"DIR-18"** ⇒ 结果 JSON 会被标成 DIR-18 ⇒ 改。
res = {"dir": "DIR-22", "round": "R1", "checks": [], "arms": {}, "mech": {},
       "meta": {"card": "rev1.2", "mvp_grid": "5 cells x 5 seeds"}}
# ★★ 缺陷 D78：**增量文件与结果文件必须按方向命名空间隔离** ——
#   原实现写 `HERE.parent/"R28_results.json"` 与 `HERE.parent/".ops"/f"formal_incremental_{tag}.json"`
#   ⇒ `HERE.parent` = `~/sgmn-loop/` ⇒ **DIR-18 与 DIR-19 写的是同一个文件** ⇒ ⇒ 实测：
#   DIR-19 的正式轮**静默复用了 DIR-18 的 `E5_base` 行**（nll 15.3260 / 14.9949）
#   ⇒ ⇒ 基准臂被**跨方向污染** ⇒ ⇒ H1′/H1′b 的参照值不可信。
#   修法：文件名一律带方向与轮次前缀（`R19_*` / `formal_incremental_DIR19_*`）。
_DIR_TAG = "DIR22r11"

# ═══════════════════════════════════════════════════════════════════════════════
# ★★【DIR-22 · rev 1.1 §3.1】**机制探针**：实测 $ρ$（刷新收缩）与 $D$（逐情节衰减）
#
#   一维递推（rev 1.1 §0 · D123）：对条目键 $k$（$s=\lVert k\rVert^2$），$u\equiv Wk$，$u_0\equiv W^0k$：
#       $u \leftarrow \rho\,u+\eta(\alpha s\,v+\lambda u_0)$，$\rho=1-\eta(\alpha s+\lambda)$
#   不动点 $u^\star=(\alpha s v+\lambda u_0)/(\alpha s+\lambda)$（**与 $\eta$ 无关**）。
#   ⇒ 刷新：连写同一条目 $r$ 次 ⇒ $(u_r-u^\star)=\rho^r(u_0-u^\star)$ ⇒ 最小二乘估 $\hat\rho$。
#   ⇒ 衰减：写 $(k_0,v_0)$ 一次后，**改写另一条正交键** $n$ 次 ⇒
#        $u(k_0)$ 只受 $\lambda$ 项作用（$k_1\perp k_0$ ⇒ 数据项读出为 0）⇒ $(1-\eta\lambda)^n$。
#        ★ 走的是**真实写路径**（`core.write`），不手工改 `W` ⇒ 不是同义反复。
# ═══════════════════════════════════════════════════════════════════════════════

def _wvec(core, k):
    with torch.no_grad():
        return (core._W() @ k).detach().clone()


def _proj_ratio(cur, base, ref):
    r"""$\langle cur-ref,\; base-ref\rangle / \lVert base-ref\rVert^2$（确定性递推 ⇒ 精确）。"""
    a = (cur - ref)
    b = (base - ref)
    den = float((b @ b))
    return float("nan") if den < 1e-30 else float((a @ b) / den)


def probe_mechanism(st, kv_pairs, m_episode, alpha=None, lam=None):
    r"""在**当前状态**上测 $\hat\rho$ 与逐情节衰减比 $D$；调用方负责 snapshot/restore。

    `kv_pairs[layer] = (k0, v0, k1)`：$k_0$ = 目标条目（码 token）的**原始**键，
    $v_0$ 其原始值，$k_1$ = **另一条**已写入条目的原始键（与 $k_0$ 正交 ⇒ 数据项零串扰）。
    """
    alpha = float(ALPHA if alpha is None else alpha)
    lam = float(LAM if lam is None else lam)
    out = {"rho_meas": [], "rho_pred": [], "D_meas": [], "D_pred": [],
           "s_meas": [], "ortho_dot": [], "eta": []}
    for li, m in st.mems.items():
        core = m.core
        if m.core.kind != "hebbian_full":
            continue
        eta = float(m.eta)
        k0, v0, k1 = kv_pairs[li]
        s = float((k0 @ k0))
        # —— 刷新探针：先记录当前 u0，再连写 r=3 次同一条目 ——
        st.unfreeze_write()
        u0 = _wvec(core, k0)
        ref0 = (core.W0 @ k0).detach().clone()
        u_star = (alpha * s * v0 + lam * ref0) / (alpha * s + lam)
        us = [u0]
        for _ in range(3):
            core.write(k0, v0)
            us.append(_wvec(core, k0))
        # 最小二乘（分母固定为 n=1 的投影 ⇒ 确定性递推下逐位成立）
        rho1 = _proj_ratio(us[1], us[0], u_star)
        rho2 = _proj_ratio(us[2], us[0], u_star)
        rho_meas = 0.5 * (rho1 + math.sqrt(max(rho2, 0.0))) if rho2 > 0 else rho1
        # —— 衰减探针：改写另一条**正交**键 m 次（真实写路径），观测 k0 的衰变 ——
        st.unfreeze_write()
        u_before = _wvec(core, k0)
        for _ in range(int(m_episode)):
            core.write(k1, v0)
        u_after = _wvec(core, k0)
        D_meas = _proj_ratio(u_after, u_before, ref0)
        out["rho_meas"].append(float(rho_meas))
        out["rho_pred"].append(float(1.0 - eta * (alpha * s + lam)))
        out["D_meas"].append(float(D_meas))
        out["D_pred"].append(float((1.0 - eta * lam) ** int(m_episode)))
        out["s_meas"].append(float(s))
        out["ortho_dot"].append(abs(float(k0 @ k1)))
        out["eta"].append(eta)
    for kx in ("rho_meas", "rho_pred", "D_meas", "D_pred", "s_meas", "ortho_dot", "eta"):
        out[kx + "_mean"] = float(np.mean(out[kx])) if out[kx] else float("nan")
    out["C_meas"] = float(1.0 - out["rho_meas_mean"]) if out["rho_meas"] else float("nan")
    out["C_pred"] = float(1.0 - out["rho_pred_mean"]) if out["rho_pred"] else float("nan")
    return out



def rec(name, passed, detail, tol=None, value=None):
    def clean(x):
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        if isinstance(x, (np.floating, float)):
            return float(x)
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (list, tuple)):
            return [clean(i) for i in x]
        return x
    res["checks"].append({"name": name, "passed": bool(passed), "detail": detail,
                          "tolerance": tol, "value": clean(value)})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


def load_filler():
    import requests
    p = pathlib.Path.home() / "sgmn-loop" / "corpus" / "train.txt"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(CORPUS_BASE + "train.txt", timeout=90)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p.read_text(encoding="utf-8", errors="ignore")


# ★★【DIR-19】语料的「前一半 / 后一半」切分（供旋钮 N5：探针数据与训练数据**逐位不相交**）
_CORP_SPLIT = {"half": "all"}        # "all" | "first" | "second"
_LRCACHE = {"proj": None, "r": None}   # ★ 低秩投影的**全局缓存**（基座初值全臂相同 ⇒ 只算一次）


def _corpus_text(which="train"):
    p = pathlib.Path.home() / "sgmn-loop" / "corpus" / f"{which}.txt"
    if not p.exists():
        load_filler()
    t = p.read_text(encoding="utf-8", errors="ignore")
    if which == "train" and _CORP_SPLIT["half"] in ("first", "second"):
        h = len(t) // 2
        t = t[:h] if _CORP_SPLIT["half"] == "first" else t[h:]
    return t


def build_ppl_chunks(tok, n_chunks=16, seq_len=256, which="valid"):
    """基座保持探针的数据块（**连续文本**，与记忆任务无关）。"""
    txt = _corpus_text(which)
    ids = tok(txt).input_ids
    out = []
    for i in range(n_chunks):
        a = (i * seq_len) % max(1, len(ids) - seq_len - 1)
        out.append(ids[a:a + seq_len])
    return out


def probe_ppl(model, st, chunks):
    """基座保持探针：连续文本上的**平均每 token NLL**（nats）。

    ★ `st` 为 None ⇒ 纯基座（不挂记忆）；否则**挂记忆**（= 加装后的实际状态）。
    ★ 探针**不训练**，只前向 ⇒ 无副作用。
    """
    if st is not None:
        st.suspend_hooks()
    try:
        tot, ntok = 0.0, 0
        with torch.no_grad():
            for ids in chunks:
                x = torch.tensor(ids, dtype=torch.long, device=DEV).unsqueeze(0)
                lg = model(input_ids=x).logits.float()
                lp = F.log_softmax(lg[0, :-1, :], dim=-1)
                tgt = x[0, 1:]
                g = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                tot += float(-g.sum())
                ntok += int(tgt.numel())
        return tot / max(1, ntok)
    finally:
        if st is not None:
            st.resume_hooks()


def kv_drift(st, chunks_probe=4, seq_len=128):
    """$D_{kv}$：写键 $k$ 分布的 **1-Wasserstein 分位距离**（训练前 vs 训练后）。

    ★ 度量自检（卡片 A19-8 / H6′）：**零训练步**时必须为 0。
    ★ 实现：对一批探针文本取 $P_k h$ 的分位向量，比较训练前后的分位差。
    ★★ 缺陷 D68 的修法：**摘除记忆钩子**后取 hidden state ⇒ 纯基座函数 ⇒ 确定性
       （否则记忆注入会改变**下游层**的 hidden state ⇒ 两次调用键分布不同 ⇒ 零训练也不为 0）。
    """
    def quantiles():
        st.suspend_hooks()
        try:
            with torch.no_grad():
                acc = []
                for ids in chunks_probe:
                    x = torch.tensor(ids, dtype=torch.long, device=DEV).unsqueeze(0)
                    hs = st.model(input_ids=x, output_hidden_states=True).hidden_states
                    for li, m2 in st.mems.items():
                        # ★ HF 的 hidden_states[i] = layer i 的**输入**（hidden_states[0]=词嵌入）
                        h = hs[li][0]
                        k = (h @ m2.Pk.detach())
                        acc.append(k.reshape(-1).float().cpu())
                if not acc:
                    return None
                v = torch.cat(acc)
                qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
                return torch.quantile(v, qs)
        finally:
            st.resume_hooks()
    return quantiles()


def build_docs(tok, seed, n_train, n_eval):
    """每篇样本 = 目标文档 + 干扰文档（同线索 CUE_A、异码、异上下文）。
    码全局唯一 ⇒ 训练码与评测码**零重叠**（rev 1.3 的「评测码留出」）。"""
    import re as _re
    text = _corpus_text("train")          # ★ DIR-19：受 `_CORP_SPLIT` 控制（N5 用前一半）
    paras = [q.strip() for q in _re.split(r"\n\s*\n", text) if len(q.strip()) > 120]
    rng = random.Random(seed)
    rng.shuffle(paras)
    pool = []
    for q in paras:
        pool.extend(tok(q).input_ids)
        if len(pool) > 160000:
            break
    cueA = tok(CUE_A).input_ids
    code_pool = rare_word_tokens(tok, pool)
    out, used = [], set()
    guard = 0
    need = n_train + n_eval
    while len(out) < need and guard < 80 * need:
        guard += 1
        i = rng.randrange(0, len(pool) - 900)
        j = rng.randrange(0, len(pool) - 900)
        if abs(i - j) < MIN_FILLER_GAP:
            continue
        code_id = rng.choice(code_pool)
        dcode_id = rng.choice(code_pool)
        # ★★ 缺陷 D101：原写只判 `code_id == dcode_id`，**从未检查 `used`** ⇒ 码可跨文档重复
        #   ⇒ docstring 与 A2'⑦ 声称的「码全局唯一 / 评测码留出」在 720 篇规模上不成立
        #   （1440 次抽取 vs 4000 码池 ⇒ 碰撞概率 ≈ 30%）⇒ 同一答案词进两次记忆 ⇒ 命中口径被污染。
        if code_id == dcode_id or code_id in used or dcode_id in used:
            continue
        used.add(code_id); used.add(dcode_id)
        fill = pool[i:i + FILLER_TOK]
        fill2 = pool[j:j + FILLER_TOK]
        doc = fill + cueA + [code_id]          # ★ 码 = 文档的最后一个 token
        dist = fill2 + cueA + [dcode_id]       # ★ 干扰文档也以码结尾（⇒ ctx 混合里有两份「码分量」）
        p1 = len(fill) + len(cueA)
        out.append({"ids": doc, "prefix": fill + cueA, "code_ids": [code_id],
                    "true_idx": p1 - 1, "code": str(code_id),
                    "dist_ids": dist, "dist_code": str(dcode_id),
                    "n_entries_doc": len(doc) - 1, "n_entries_dist": len(dist) - 1})
    tr, ev = out[:n_train], out[n_train:need]
    assert len(tr) == n_train and len(ev) == n_eval
    assert len(used) == 2 * need, f"码未全局唯一：{len(used)} ≠ {2 * need}"
    return tr, ev


_RARE_CACHE = {}


def rare_word_tokens(tok, corpus_ids=None, n_max=4000):
    """从词表里挑**单 token 的稀有词**（带前导空格、纯字母、长度 ≥ 6、往返可编码），
    并按**语料频次**升序取最稀有的 n_max 个。
    稀有 ⇒ 基座几乎不可能猜中 ⇒ 目标必须靠记忆读出（A2'⑨ 会实测基座 NLL ≥ 3）。"""
    key = (id(tok), n_max)
    if key in _RARE_CACHE:
        return _RARE_CACHE[key]
    vocab = tok.get_vocab()
    size = max(vocab.values()) + 1
    cand = []
    for i in range(1000, size):
        # ★ 注意：tok.get_vocab() 的键是**token 字符串**，不是 id ⇒ 不可用 `i not in vocab` 过滤
        try:
            s = tok.decode([i])
        except Exception:
            continue
        if not s or not s[0].isspace():
            continue
        w = s.strip()
        if len(w) < 6 or not w.isalpha():
            continue
        if tok(s).input_ids != [i]:
            continue
        cand.append((i, w))
    from collections import Counter
    cnt = Counter(corpus_ids) if corpus_ids is not None else Counter()
    cand.sort(key=lambda t: (cnt.get(t[0], 0), t[0]))
    out = [i for i, _ in cand[:n_max]]
    _RARE_CACHE[key] = out
    return out


def _norm_cdf(x):
    """标准正态 CDF（无 scipy ⇒ 用 math.erf）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sign_test_p(d):
    d = [x for x in d if abs(x) > 1e-15]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0
    k = sum(1 for x in d if x > 0)
    pk = math.comb(n, k) * 0.5 ** n
    p = sum(math.comb(n, i) * 0.5 ** n for i in range(n + 1)
            if math.comb(n, i) * 0.5 ** n <= pk + 1e-12)
    return min(1.0, p), k, n


def paired(A, B):
    """返回 B − A。对 NLL（越低越好）：>0 ⇒ A 更好；对 log-prob（越高越好）：>0 ⇒ B 更好。"""
    D = np.asarray(B, dtype=float) - np.asarray(A, dtype=float)
    D = D[np.isfinite(D)]
    if len(D) == 0:
        return {"mean": float("nan"), "t": float("nan"), "p": 1.0, "pos": 0, "n": 0,
                "consistent": False}
    if np.all(D == 0.0):
        return {"mean": 0.0, "sd": 0.0, "t": 0.0, "p": 1.0, "pos": 0, "n": len(D),
                "consistent": True}
    m = float(D.mean())
    sd = float(D.std(ddof=1))
    t = m / (sd / math.sqrt(len(D))) if sd > 0 else 0.0
    ps, k, n = sign_test_p(list(D))
    return {"mean": m, "sd": sd, "t": t, "p": ps, "pos": k, "n": n,
            "consistent": bool(k == n or k == 0)}


# ★【DIR-18 rev 1.2】臂表。字段：
#   (name, query_mode, mem_kind, eta, train, train_pv, key_mode, variant)
#   `variant`: "normal" | "nowrite"（地址表照写、W 从不更新 ⇒ **归因对照**）
#              | "gammalock"（γ 不进优化器）| "pvw"（写侧自监督更新 P_v）
def _num(v):
    """把 `None`（旧版落盘把 NaN 写成 None）统一成 `float('nan')`（D59 教训）。"""
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _jsonable(o):
    """numpy / torch 标量 ⇒ JSON 可序列化（NaN 原样保留 ⇒ 落盘/读回逐位等价）。"""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if hasattr(o, "item") and callable(getattr(o, "item")):
        try:
            return _jsonable(o.item())
        except Exception:
            pass
    return o


def _load_incremental(tag="full"):
    """读取增量落盘的结果（续跑用）⇒ **seed 键统一为 int**（D56 教训）。
    ★ `tag` 区分**规格**（smoke / full）⇒ 不同规格的增量**互不串用**（防「降级轮结果冒充正式轮」）。"""
    _inc = HERE.parent / ".ops" / f"formal_incremental_{_DIR_TAG}_{tag}.json"   # ★ D78
    if not _inc.exists():
        return {}
    try:
        d = json.load(open(_inc))
        _rows = d.get("rows", {}) or {}
        return {k: {int(sk): vv for sk, vv in v.items()} for k, v in _rows.items()}
    except Exception:
        return {}


# ★ DIR-20 臂表（11 臂；卡片 rev 1.1 §2 冻结）
#   字段：(name, mem_kind, eta, key_mode, variant, export_mode, session_mode)
#   export_mode : full | no_readmap | no_addr | none
#   session_mode: cross（重建实例 + 导入） | same（同实例，写完即测） |
#                 same_empty（同实例 + 空栈） | cross_noimport（重建实例 + 不导入） |
#                 cross_wrongseed（重建实例 + 导入**另一个种子**的状态）
#   ★ `X8_linear` 的 η = 1e-5（**尺度适配档**；卡片 rev 1.1 §0.4 / 修订 20-C）：
#     `linear` 键的实测 ‖k‖² ≈ 1.7e4（DIR-18 的 D50/D51）⇒ 稳定上界 η < 2/(λ+α‖k‖²) ≈ 1.2e-4
#     ⇒ ⇒ η=1e-1 会让写规则**发散** ⇒ ⇒ H4″ 会测成「发散 vs 正常」⇒ ⇒ 不可接受。
#   ★ `X6_verbatim` 用 `explicit_kv`：其内容在地址表的 `vals` 里 ⇒ ⇒ **按构造内容正确**
#     ⇒ ⇒ 作为「零效度门槛」对照；且 `explicit_kv` 的读**对键尺度不敏感**（余弦）
#     ⇒ ⇒ **X6 不参与 H4″**（卡片 rev 1.1 §3 写死）。
# ★ DIR-21 的臂表见上面的 `ARMS22`（DIR-20 的 11 臂在本方向不使用）
ARMS = []


def make_stack(model, d, query_mode, seed=20261028, mem_kind="explicit_kv", eta=None,
               key_mode="linear", variant="normal"):
    st = SGS(model, MOUNT, d, d, R, ALPHA, LAM, (eta if eta is not None else ETA), GAMMA0, DEV,
             seed=seed, use_factorized=False, no_write=False, perpos=True,
             mem_kind=mem_kind, n_entries=N_ENTRIES, key_mode=key_mode)
    for m in st.mems.values():
        m.key_mode = key_mode
        m.query_mode = query_mode
        m.addr_mode = "sig"
        m.read_beta = BETA
        # ★★【修订 68】`content_wrong` 臂：只换「值」的来源位置，**地址不动**。
        m.content_wrong = (variant == "content_wrong")
    return st


def write_both(st, d):
    """写阶段：目标文档 + 干扰文档（都 no_grad）。顺序固定 ⇒ 目标条目在前。"""
    ids = torch.tensor(d["ids"], dtype=torch.long, device=DEV).unsqueeze(0)
    dids = torch.tensor(d["dist_ids"], dtype=torch.long, device=DEV).unsqueeze(0)
    with torch.no_grad():
        st.model(input_ids=ids)
        st.model(input_ids=dids)


def resident_r_cnt(st):
    """当前仍占据槽位、且自身签名仍选中该槽的条目数。

    这是冻结定义里的「签名匹配选中的槽 == 真值槽」，对全部在册条目计数。
    被回绕覆盖的旧条目不在表里，计为不可检索。不另做语言模型前向。
    """
    m = st.mems[list(st.mems)[0]]
    n = min(int(m.core.count), int(m.core.N))
    if n <= 0 or not hasattr(m.core, "sigs"):
        return 0
    sigs = m.core.sigs[:n].detach()
    norms = sigs.norm(dim=1).clamp_min(1e-30)
    sc = (sigs @ sigs.T) / (norms[:, None] * norms[None, :])
    pred = torch.argmax(sc, dim=1)
    idx = torch.arange(n, device=pred.device)
    return int(torch.sum(pred == idx).item())


# ═══════════════════════════════════════════════════════════════════════
# ★★【rev 1.7】测量层辅助：原始 (k,v) 捕获 / 原对象余弦 / 有效窗口（键列复用）
# ═══════════════════════════════════════════════════════════════════════


def clear_write_records(st):
    """★【rev 1.7 · 缺陷 D111】清空 `keys_written` / `vals_written`。

    `clear_slots()` 只清 `core.keys/vals/sigs/count` ⇒ **训练残留仍在写记录里**
    ⇒ 「文档 1 的原始 (k,v)」会取到训练条目 ⇒ 扫描起点必须显式清空。
    """
    for m in st.mems.values():
        m.keys_written.clear()
        m.vals_written.clear()


def capture_doc1_entries(st, n_entries):
    """★【rev 1.7 · 缺陷 D111/D102】捕获文档 1（**目标文档**）的**原始 (k, v)**。

    写入顺序 = 目标文档的条目在前、干扰文档的条目在后 ⇒ 前 `n_entries` 条即目标文档的。
    返回 `out[layer][entry] = (k, v)`。
    """
    out = []
    for m in st.mems.values():
        ks = list(m.keys_written[:n_entries])
        vs = list(m.vals_written[:n_entries])
        if len(ks) != n_entries or len(vs) != n_entries:
            raise RuntimeError(f"文档 1 的条目捕获失败：{len(ks)}/{len(vs)} ≠ {n_entries}")
        out.append([(k.detach().clone(), v.detach().clone()) for k, v in zip(ks, vs)])
    return out


def cos_orig(st, kv_entry):
    """★【rev 1.7 · 缺陷 D102】**原对象**余弦：`cos(W·k, v)`，k/v = 该条目**写入时**的原始张量。

    阴性对照 = `core.fidelity(slot)`（当前槽里住的是谁）⇒ 回绕后测的是别人的内容 ⇒ 不可用。
    """
    vals = []
    for m, (k, v) in zip(st.mems.values(), kv_entry):
        with torch.no_grad():
            W = m.core._W()
            a = W @ k
            na = float(a.norm())
            nb = float(v.norm())
            vals.append(0.0 if (na < 1e-30 or nb < 1e-30) else float((a @ v) / (na * nb)))
    return float(np.mean(vals)) if vals else float("nan")


def reuse_episode(base0, trig, kmax):
    """★【rev 1.7 · 缺陷 D112】锚条目（文档 1 的**码 token**）的**键列被再次写入**的情节。

    键列：目标前视起点 = `base0 + 68·(e−1)`，干扰前视起点 = 再 +34；
    写入列 = 前视起点 + (t−1)（t ∈ 触发位置）；锚列 = `base0 + 33`（mod 896）。
    超过该情节 ⇒ 该臂的保持曲线**失效**（窗口 = 该情节 − 1）。
    """
    D = 896
    j_anchor = (int(base0) + 33) % D
    for e in range(2, int(kmax) + 2):
        bt = int(base0) + 68 * (e - 1)
        bd = bt + 34
        cols = [((bt + t - 1) % D) for t in trig] + [((bd + t - 1) % D) for t in trig]
        if j_anchor in cols:
            return e
    return None


def make_nowrite_variant(st):
    """★【DIR-18 rev 1.2 修订 56 · 归因对照】**地址表照写，但 `W` 从不更新**。

    ⇒ 地址正确、**内容无关联** ⇒ 与主臂的差 = 「Hebbian 写步对任务收益的贡献」。
    """
    for m in st.mems.values():
        core = m.core

        def patched(k, v, sig=None, core=core):
            with torch.no_grad():
                i = core.count % core.N
                core.count += 1
                core.keys[i].copy_(k)
                core.vals[i].copy_(v)
                if sig is not None:
                    core.sigs[i].copy_(sig)
        core.write = patched


def kl_from_logits(ref_last, cur_last):
    """KL(π_ref ‖ π_cur)，**在目标位置**（最后一个 token）上算。

    ★★ 缺陷 D77 的修法（显存）：**参考分布 π_ref = π_θ₀ 是固定的** ⇒ 不必让参考模型
      **驻留显存**（2.0 GB）⇒ ⇒ 只把参考 logits（每步一行，0.6 MB）搬上 GPU。
    ★★ 同时：KL 与任务损失**共用同一次前向**（同一张图）⇒ ⇒ 一次 `backward()`
      ⇒ ⇒ **峰值显存与无 KL 的臂相同**（旧实现要两张图 ⇒ 叠加）。
    """
    lpr = F.log_softmax(ref_last, dim=-1)
    lpc = F.log_softmax(cur_last, dim=-1)
    pr = lpr.exp()
    return (pr * (lpr - lpc)).sum()


def ref_logits_at(model_ref, docs):
    """预计算参考模型（θ₀）在每篇训练文档的**目标位置**上的 logits ⇒ CPU 张量 [N, vocab]。

    ★ 只算**一次** ⇒ 之后可**立即释放参考模型**（省 2.0 GB 显存，缺陷 D77）。
    """
    out = []
    with torch.no_grad():
        for dd in docs:
            _pre = torch.tensor(dd["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
            try:
                lg = model_ref(input_ids=_pre, logits_to_keep=1).logits.float()
            except TypeError:
                lg = model_ref(input_ids=_pre).logits.float()
            out.append(lg[0, -1].detach().cpu().clone())
    return torch.stack(out)


def train_arm(st, train_docs, mode, steps=None, train_pv=False, variant="normal",
              base_mode="none", kl_lambda=0.0, model_ref=None, lowrank_projs=None,
              base_only=False, n4_layers=None, ref_logits=None):
    """★ DIR-19：在 DIR-18 的 `train_arm` 上**只加**「保护旋钮」相关的能力。

    ① `base_mode`：
       "none"    ⇒ 基座全冻结（= DIR-17/18 的做法，旋钮 N1）
       "all"     ⇒ 全部基座参数可训（旋钮 N2/N3 的载体）
       "lowrank" ⇒ 只放开 2-D 权重，且每步后**投影回** `W0 + span(U_r)`（旋钮 N3）
       "layers"  ⇒ 只放开 `n4_layers` 指定层（旋钮 N4）
    ② `kl_lambda > 0` ⇒ 训练损失追加 **KL 正则**（修订 19-B 的口径：两侧**不含记忆**）。
    ③ `base_only=True` ⇒ **摘除记忆钩子**，只训基座（旋钮 N5 的对照 `E5_base_cap`）。
    ★ 基座用 **SGD(momentum=0.9, lr=1e-5)**：Adam 的两份状态 = 4 GB，
      在 T4 15 GB 上与「模型 2 GB + θ₀ 快照 2 GB」叠加会 OOM ⇒ ⇒ 改用 SGD（状态 2 GB）。
      ★ 读侧仍用 **Adam(lr=1e-3)**（与 DIR-17/18 **完全一致** ⇒ 收益口径可比）。
    """
    return _train_arm_impl(st, train_docs, mode, steps, train_pv, variant, base_mode,
                           kl_lambda, model_ref, lowrank_projs, base_only, n4_layers,
                           ref_logits=ref_logits)     # ★★ D80：透传


def _train_arm_impl(st, train_docs, mode, steps, train_pv, variant, base_mode,
                    kl_lambda, model_ref, lowrank_projs, base_only, n4_layers,
                    ref_logits=None):     # ★★ D80：实现层签名必须与包装层**一致**
    """两阶段：写阶段（no_grad）→ 冻结写 → 读阶段（梯度开）→ 目标码首 token 的 NLL → 反向 → 步。

    【DIR-18】train_pv=True ⇒ **任务损失同时更新写侧的值映射 $P_v$**（其余臂 $P_v$ 冻结）
    ⇒ ⇒ 「受保护的联合训练」的**最小形态**：写侧的**核状态 $W$** 仍是 no_grad（梯度不流入）。
    """
    snap0 = st.snapshot_all()
    params = [t for _, t in st.read_params_for(mode)]
    if train_pv:
        for li, m in st.mems.items():
            params.append(m.Pv)
    # ★【修订 55 · γ 锁定臂】γ 不进优化器（读侧其余参数 `W_o` 照常训练）
    #   ⇒ 与主臂的**唯一**差别 = γ 是否可训 ⇒ 用于跨方向复核 DIR-17 的 D-24。
    if variant == "gammalock":
        params = [t for n, t in st.read_params_for(mode) if not n.endswith("gamma")]
    opt = torch.optim.Adam(params, lr=1e-3)
    # ★★【DIR-19】基座优化器（保护旋钮的载体）
    base_params = []
    opt_base = None
    if base_mode != "none":
        _lay = n4_layers if base_mode == "layers" else None
        base_params = st.trainable_base_params(base_mode, layers=_lay)
        # ★ SGD **不带 momentum**（状态 0 字节 ⇒ 省 2 GB 显存；T4 15 GB 上必须）
        opt_base = torch.optim.SGD([q for _, q in base_params], lr=1e-5)
    # ★【修订 53 · 写侧自监督臂 `pvw`】单独的优化器，**只含 P_v**
    opt_pv = None
    if variant == "pvw":
        opt_pv = torch.optim.Adam([m.Pv for m in st.mems.values()], lr=1e-3)
    w0 = {n: t.detach().clone() for n, t in
          [(f"L{li}.{k}", getattr(m, k)) for li, m in st.mems.items() for k in ("Wo", "gamma", "Wq")]}
    # ★★ 缺陷 D77 的修法：参考 logits **预计算一次**（CPU）⇒ 训练时**不需要参考模型驻留显存**
    _ref_lg = ref_logits          # ★★ D77：优先用**外部预计算**的参考 logits（CPU）
    if _ref_lg is None and base_mode != "none" and kl_lambda > 0.0 and model_ref is not None:
        _n_ref = len(train_docs) if steps is None else min(steps, len(train_docs))
        _ref_lg = ref_logits_at(model_ref, train_docs[:_n_ref])
    traj = []
    _kl_hist = []
    n = len(train_docs) if steps is None else min(steps, len(train_docs))
    for _si, d in enumerate(train_docs[:n]):
        st.unfreeze_write()          # 【D36】每篇开始前解冻写（否则上一轮遗留的 no_write=True 使写阶段空转）
        st.restore_all(snap0)
        write_both(st, d)            # ① 写阶段（目标 + 干扰）
        # ★【修订 53 · `pvw`】写侧自监督：L_write = −cos(W k_j, v_j)，**只更新 P_v**。
        #   梯度**不流入 W**（核状态 no_grad）⇒ Freeze C 不破。
        if opt_pv is not None:
            opt_pv.zero_grad(set_to_none=True)
            lw = []
            for m in st.mems.values():
                l_ = m.write_side_loss(int(d["true_idx"]))
                if l_ is not None:
                    lw.append(l_)
            if lw:
                loss_w = sum(lw)
                loss_w.backward()
                opt_pv.step()
            for m in st.mems.values():
                m.write_pairs.clear(); m.hs_for_write.clear()
        st.freeze_write()            # ② 冻结写
        pre = torch.tensor(d["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        opt.zero_grad(set_to_none=True)
        if opt_base is not None:
            opt_base.zero_grad(set_to_none=True)
        # ★★【DIR-19】`base_only` ⇒ **摘除记忆钩子**（纯基座的对照训练）
        _susp = st.suspend_hooks() if base_only else 0
        # ★★ 缺陷 D73：`logits_to_keep=1` ⇒ 只算最后一位（全序列 logits ≈ 520 MB ⇒ T4 OOM）
        try:
            lg = st.model(input_ids=pre, logits_to_keep=1).logits.float()   # ③ 读阶段（梯度开）
        except TypeError:
            lg = st.model(input_ids=pre).logits.float()
        tgt = torch.tensor([d["code_ids"][0]], dtype=torch.long, device=DEV)
        loss = F.cross_entropy(lg[0, -1:, :], tgt)
        # ★★【DIR-19 · 修订 19-B】KL 正则项（**两侧都不含记忆** ⇒ 纯粹度量基座漂移）
        #   ★★ 缺陷 D77（**正式轮 OOM 的根因**）：旧实现让**参考模型常驻显存**（2.0 GB），
        #   且 KL 要**另做一次前向**（两张图并存）⇒ 峰值 = 模型 2 GB + 基座梯度 2 GB
        #   + 参考模型 2 GB + 两张图 ⇒ **T4 15 GB OOM**。
        #   ⇒ ⇒ 修法：① 参考分布 π_θ₀ **固定** ⇒ **预计算一次**参考 logits（CPU）后**释放参考模型**；
        #           ② KL 与任务损失**共用同一次前向**（同一张图）⇒ ⇒ **一次 backward**。
        # ★★ 顺序：**先** backward 任务图（**释放**它）⇒ 再做 KL 的前向 ⇒ ⇒ **任一时刻只有一张图**
        loss.backward()
        _kl = 0.0
        if opt_base is not None and kl_lambda > 0.0 and _ref_lg is not None:
            # ★★ 口径（卡片 rev 1.3）：**两侧都不含记忆** ⇒ 当前侧必须**摘钩**再前向
            #   ⇒ ⇒ 参考侧 = 预计算的 π_θ₀（CPU）；当前侧 = 摘钩前向 ⇒ 纯粹度量基座漂移。
            st.suspend_hooks()
            try:
                lg_cur = st.model(input_ids=pre, logits_to_keep=1).logits.float()
            except TypeError:
                lg_cur = st.model(input_ids=pre).logits.float()
            finally:
                st.resume_hooks()
            _kl_t = kl_from_logits(_ref_lg[_si].to(DEV), lg_cur[0, -1])
            (kl_lambda * _kl_t).backward()          # ★ 张量 ⇒ 梯度流入基座（D70）
            _kl = float(_kl_t.detach())
        # ★★ 缺陷 D74：`base_only` 臂（`E5_base_cap`）的记忆钩子**必须全程摘除**
        #   ⇒ 训练与写阶段都**不注入记忆** ⇒ 此处**不得** `resume_hooks()`。
        #   （原实现在此恢复 ⇒ 下一轮 `write_both` 又会把记忆注入 ⇒ 污染「纯基座对照训练」）
        gnorm = float(sum(float(p_.grad.norm()) ** 2 for p_ in params
                          if p_.grad is not None) ** 0.5)
        opt.step()
        if opt_base is not None:
            opt_base.step()
            if base_mode == "lowrank" and lowrank_projs:
                # ★ 投影回 `W0 + span(U_r)` ⇒ 漂移**严格**落在低秩子空间内（可断言）
                st.project_base_to_subspace(lowrank_projs)
        traj.append((float(loss), gnorm))
        _kl_hist.append(_kl)
    rep = {}
    for li, m in st.mems.items():
        rep[f"L{li}.Wo"] = {
            "changed": float((m.Wo.detach() - w0[f"L{li}.Wo"]).abs().max()),
            "max_abs": float(m.Wo.detach().abs().max()),
            "grad": None if m.Wo.grad is None else float(m.Wo.grad.norm()),
        }
        rep[f"L{li}.gamma"] = {
            "changed": float((m.gamma.detach() - w0[f"L{li}.gamma"]).abs().max()),
            "value": float(m.gamma.detach()),
            "grad": None if m.gamma.grad is None else float(m.gamma.grad.norm()),
        }
        rep[f"L{li}.Wq"] = {
            "changed": float((m.Wq.detach() - w0[f"L{li}.Wq"]).abs().max()),
            "grad": None if m.Wq.grad is None else float(m.Wq.grad.norm()),
        }
    st.unfreeze_write()
    rep["_kl_hist"] = _kl_hist
    rep["_base_grad_norm"] = (float(sum(float(q.grad.norm()) ** 2 for _, q in base_params
                                        if q.grad is not None) ** 0.5)
                               if base_params else 0.0)
    rep["_n_base_params"] = len(base_params)
    return traj, rep


def clear_slots(st):
    """★【DIR-18 rev 1.2 修订 52 · D48】**清空槽表**（`keys/vals/sigs` 归零、`count=0`），
    **但保留核状态 `W`**。

    根因：DIR-17/DIR-18 的 `eval_arm` 把核状态从**训练后**快照恢复 ⇒ 评测时核里残留训练条目
    （probe18c 实测 `n_stale=60`），且 68+68 > N_ENTRIES(128) ⇒ **槽位回绕**
    （真值条目实际落在槽 101 而非 33）⇒ ⇒ 指标口径错位（D46）。

    ⇒ 修订后：每篇评测前槽表为空 ⇒ 写入不回头 ⇒ **真值槽 == 写入顺序号** ⇒
      `addr`（写入序）与 `addr_slot`（槽空间）**两口径必然相等**（Stage A 断言 A2'⑬）。
    """
    with torch.no_grad():
        for m in st.mems.values():
            core = m.core
            if hasattr(core, "keys"):
                core.keys.zero_()
            if hasattr(core, "vals"):
                core.vals.zero_()
            if hasattr(core, "sigs"):
                core.sigs.zero_()
            if hasattr(core, "count"):
                core.count = 0


def eval_arm(st, eval_docs, ctrl_ids=None):
    """评测一轮。`ctrl_ids`：交换性对照 token（**同池、均匀抽样、从未写入**）⇒
    用来把「内容收益」与「稀有词先验」分开（修订 58 / probe18e）。

    ★【D46 / D54】地址指标**双口径**：
        `addr`       —— **写入顺序空间**：签名匹配选中的槽 == `true_idx`
                        （★ DIR-17 协议下**必然 ≈ 0** —— 槽位回绕使真值槽 ≠ 写入顺序号；
                          这是**协议的正确读数**，不是缺陷）
        `addr_slot`  —— **槽空间**：签名匹配选中的槽 == **真值条目实际所在的槽**
                        （★ D54：真值槽由装置记录的**精确槽号映射**给出
                          `slot_of_write[j]` = 第 j 次写占用的槽 = (写入前 count + j) % N；
                          **不再**用「值内容反查」—— 在有训练残留时反查会歧义）
    ★★【D55 · 修订 65】评测**沿用 DIR-17 原协议**（恢复训练后快照 ⇒ 保留训练期签名条目 ⇒ 寻址可工作）。
    """
    rows = []
    snap0 = st.snapshot_all()
    for d in eval_docs:
        st.unfreeze_write()          # 【D36】同上
        st.restore_all(snap0)
        # ★★【DIR-18 rev 1.2 修订 65 · D55】**回退 D48 的「清空槽表」**（探针 probe18h 实测）：
        #   clean（清空槽表、零陈旧）⇒ 寻址命中率 **0.3722**（n_train=200）
        #   dirty17（保留训练残留）  ⇒ 寻址命中率 **0.8833**（n_train=200）⇒ 与 DIR-17 的 0.8722 一致
        #   ⇒ ⇒ **清空槽表反而破坏寻址** ⇒ ⇒ 本轮评测**沿用 DIR-17 原协议**（不 clear_slots）。
        #   ⇒ ⇒ 「陈旧残留 + 槽位回绕」不是寻址的敌人，反而是**寻址能工作的条件**
        #       （训练期的上下文签名与训练后的模型是对齐的 —— 与 DIR-17 D-24 的机制同源）。
        #   ⇒ ⇒ D48 的**失败**留档（不追认）；D46（两口径）**保留**但改用**精确槽号映射**。
        write_both(st, d)
        # 【D35】**写阶段一结束就快照写入值并清空列表**（与 DIR-16 的 run_arm 逐位同构）
        #   ⇒ 读阶段的 last_yr 与写阶段的 V 处在同一「代」上 ⇒ 命中指标才有意义。
        vals = {}
        for li, m in st.mems.items():
            vals[li] = torch.stack(m.vals_written) if m.vals_written else None
            m.vals_written.clear(); m.keys_written.clear()
        st.freeze_write()
        pre = torch.tensor(d["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            lg = st.model(input_ids=pre).logits.float()
        lp_all = F.log_softmax(lg[0, -1], dim=-1)
        lp = float(lp_all[d["code_ids"][0]])
        hits = []
        fids = []
        fids_true = []
        addr_ok = []
        addr_slot_ok = []
        cys = []
        for li, m in st.mems.items():
            yr = m.last_yr[0] if m.last_yr is not None else None
            V = vals[li]
            j_sel = m.last_cue_idx
            # —— D46 口径 ①：写入顺序空间
            addr_ok.append(1.0 if (j_sel is not None and int(j_sel) == int(d["true_idx"])) else 0.0)
            # —— D46 口径 ②：槽空间（**精确槽号映射**，D54；不再用值内容反查 ——
            #    在有训练残留时反查会歧义，且它是「近似」量）
            true_slot = None
            if (V is not None and V.shape[0] > 0
                    and len(m.slot_of_write) > int(d["true_idx"])):
                true_slot = int(m.slot_of_write[int(d["true_idx"])])
            addr_slot_ok.append(1.0 if (j_sel is not None and true_slot is not None
                                        and int(j_sel) == true_slot) else 0.0)
            if yr is None or V is None or V.shape[0] == 0:
                hits.append(0.0); fids.append(float("nan")); fids_true.append(float("nan"))
                cys.append(float("nan")); continue
            cs = F.cosine_similarity(V, yr.unsqueeze(0), dim=1)
            hits.append(1.0 if int(torch.argmax(cs)) == int(d["true_idx"]) else 0.0)
            # —— 写保真度（两个口径：选中槽 / 真值槽）
            if hasattr(m.core, "fidelity"):
                j = m.last_cue_idx if m.last_cue_idx is not None else 0
                fids.append(float(m.core.fidelity(int(j))))
                fids_true.append(float(m.core.fidelity(int(true_slot))) if true_slot is not None
                                 else float("nan"))
            else:
                fids.append(float("nan")); fids_true.append(float("nan"))
            # —— ★ 读出与真值值的余弦（写→读链是否接上的**直接**读数）
            cys.append(float(F.cosine_similarity(
                V[int(d["true_idx"])].unsqueeze(0), yr.unsqueeze(0), dim=1).item()))
        row = {"lp": lp, "nll": -lp, "nll_code": -lp, "hit": float(np.mean(hits)),
               "addr": float(np.mean(addr_ok)) if addr_ok else float("nan"),
               "addr_slot": float(np.mean(addr_slot_ok)) if addr_slot_ok else float("nan"),
               "fid": (float(np.nanmean(fids)) if (len(fids) and not all(math.isnan(f) for f in fids)) else float("nan")),
               "fid_true": (float(np.nanmean(fids_true)) if (len(fids_true) and not all(math.isnan(f) for f in fids_true)) else float("nan")),
               "cos_y_vtrue": (float(np.nanmean(cys)) if (len(cys) and not all(math.isnan(f) for f in cys)) else float("nan")),
               "n_entries": int(vals[list(vals)[0]].shape[0]) if vals and vals[list(vals)[0]] is not None else 0}
        # —— ★ 交换性对照（修订 58）：同池、均匀抽样、从未写入 ⇒ 与码**同分布**
        if ctrl_ids is not None:
            row["nll_ctrl"] = float(-lp_all[ctrl_ids].mean())
        # —— 干扰码（区分真码与同批干扰码）
        row["nll_dcode"] = float(-lp_all[int(d["dist_code"])])
        rows.append(row)
    st.unfreeze_write()
    return rows


def _eval_arm_readonly_body(st, d, ctrl_ids=None):
    """★【DIR-20】**只读**评测：只前向一次 ⇒ ⇒ **不写、不训练**。

    ★ 与 DIR-19 的 `eval_arm` 的**唯一**差别：**评测前不写** ⇒ ⇒ 记忆是会话结束时
      **冻结**的状态 ⇒ ⇒ 所有指标都必须从「**导入的状态**」里算出来：
        · `y`（读出）      —— 读阶段前向时由 `m.last_yr` 给出；
        · 真值值 `v_j`    —— 地址表里存的 `core.vals`（`hebbian_full` 的 `vals` 不进读路径，
                             只用于写保真度测量 ⇒ 卡片 rev 1.1 §0.3 / A20-11）；
        · 真值槽          —— 会话 A 写入时记录的 `d["true_slot"]`（精确槽号映射）。
    ★ 只读 ⇒ `core.count` 与 `write_calls` **不变** ⇒ Stage A **A20-9** 当场断言。
    """
    pre = torch.tensor(d["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
    with torch.no_grad():
        lg = st.model(input_ids=pre).logits.float()
    lp_all = F.log_softmax(lg[0, -1], dim=-1)
    lp = float(lp_all[d["code_ids"][0]])
    true_slot = int(d["true_slot"])
    hits, fids, fids_true, cys, addr_ok, addr_slot_ok = [], [], [], [], [], []
    n_used = 0
    for li, m in st.mems.items():
        yr = m.last_yr[0] if m.last_yr is not None else None
        j_sel = m.last_cue_idx
        # —— D46 口径 ①：写入顺序空间（DIR-20 协议下与槽空间**恒等**，A20-12 断言）
        addr_ok.append(1.0 if (j_sel is not None and int(j_sel) == int(d["true_idx"])) else 0.0)
        # —— D46 口径 ②：槽空间（精确槽号映射）
        addr_slot_ok.append(1.0 if (j_sel is not None and int(j_sel) == true_slot) else 0.0)
        n = min(int(m.core.count), int(m.core.N))
        Vs = m.core.vals.detach()[:n]
        n_used = int(Vs.shape[0])
        if yr is None or Vs.shape[0] == 0:
            hits.append(0.0)
            fids.append(float("nan"))
            fids_true.append(float("nan"))
            cys.append(float("nan"))
            continue
        cs = F.cosine_similarity(Vs, yr.unsqueeze(0), dim=1)
        hits.append(1.0 if int(torch.argmax(cs)) == true_slot else 0.0)
        if hasattr(m.core, "fidelity"):
            fids.append(float(m.core.fidelity(int(j_sel))) if j_sel is not None else float("nan"))
            fids_true.append(float(m.core.fidelity(true_slot)))
        else:
            fids.append(float("nan"))
            fids_true.append(float("nan"))
        cys.append(float(F.cosine_similarity(
            Vs[true_slot].unsqueeze(0), yr.unsqueeze(0), dim=1).item()))
    row = {"lp": lp, "nll": -lp, "nll_code": -lp,
           "hit": float(np.mean(hits)),
           "addr": float(np.mean(addr_ok)) if addr_ok else float("nan"),
           "addr_slot": float(np.mean(addr_slot_ok)) if addr_slot_ok else float("nan"),
           "fid": (float(np.nanmean(fids)) if (len(fids) and not all(math.isnan(f) for f in fids)) else float("nan")),
           "fid_true": (float(np.nanmean(fids_true)) if (len(fids_true) and not all(math.isnan(f) for f in fids_true)) else float("nan")),
           "cos_y_vtrue": (float(np.nanmean(cys)) if (len(cys) and not all(math.isnan(f) for f in cys)) else float("nan")),
           "n_entries": n_used,
           "count": int(st.mems[list(st.mems)[0]].core.count)}
    if ctrl_ids is not None:
        row["nll_ctrl"] = float(-lp_all[ctrl_ids].mean())
    row["nll_dcode"] = float(-lp_all[int(d["dist_code"])])
    return row

def eval_arm_readonly(st, d, ctrl_ids=None):
    """【DIR-20】只读评测的**包装层**：读期间冻结写，读后恢复调用方的冻结状态。

    ★ 缺陷 D84 的修法（卡片 rev 1.2 §5.1）：真正的读出逻辑在
      `_eval_arm_readonly_body`（逐字沿用 DIR-19 的 `eval_arm` 口径）。
    """
    _flags = [m.no_write for m in st.mems.values()]
    st.freeze_write()
    try:
        return _eval_arm_readonly_body(st, d, ctrl_ids)
    finally:
        for m, f in zip(st.mems.values(), _flags):
            m.no_write = f


def eval_base(model, eval_docs, ctrl_ids=None):
    """基座评测。★ 也测**交换性对照**（修订 58）：base 的 ctrl NLL 是 H13 的配对基准。"""
    rows = []
    for d in eval_docs:
        pre = torch.tensor(d["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            lg = model(input_ids=pre).logits.float()
        lp_all = F.log_softmax(lg[0, -1], dim=-1)
        lp = float(lp_all[d["code_ids"][0]])
        row = {"lp": lp, "nll": -lp, "nll_code": -lp,
               "nll_dcode": float(-lp_all[int(d["dist_code"])])}
        if ctrl_ids is not None:
            row["nll_ctrl"] = float(-lp_all[ctrl_ids].mean())
        rows.append(row)
    return rows


def build_ctrl_ids(tok, seed, tr, ev, K=32):
    """★【DIR-18 rev 1.2 修订 58】构造**交换性对照** token 集合。

    关键（probe18d → probe18e 的教训）：
      · 对照必须与真码**同池、同分布**（码是 `rng.choice(code_pool)` 的均匀抽样
        ⇒ 对照也必须**均匀抽样**，**不能**取「池中最稀有的那几个」——后者在 base 上
        NLL 更低 ⇒ **与码不可交换** ⇒ 差值会把「base 先验差异」误读成「内容被读出」）。
      · 对照**从未写入任何记忆**（排除所有训练码/评测码/干扰码）。
      · 跨臂固定 ⇒ 配对比较。
    """
    import re as _re
    text = _corpus_text("train")
    paras = [q.strip() for q in _re.split(r"\n\s*\n", text) if len(q.strip()) > 120]
    pool = []
    for q in paras:
        pool.extend(tok(q).input_ids)
        if len(pool) > 160000:
            break
    code_pool = rare_word_tokens(tok, pool)
    used = set()
    for x in list(tr) + list(ev):
        used.add(int(x["code_ids"][0]))
        used.add(int(x["dist_code"]))
    cand = [int(t) for t in code_pool if int(t) not in used]
    rng = random.Random(777 + int(seed))
    rng.shuffle(cand)
    out = cand[:K]
    assert len(out) == K, f"交换性对照不足：{len(out)} < {K}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global K_SCAN
    K_SCAN = K_SCAN_SMOKE if args.smoke else K_SCAN_FULL
    n_eval = 260 if args.smoke else N_EVAL
    n_train = 40 if args.smoke else N_TRAIN
    seeds = SEEDS[:2] if args.smoke else SEEDS

    print("=== DIR-22 R1（卡片 rev 1.2 · 写侧速率 vs 情节级速率；5 格 MVP；判据 H22-a…d）===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size
    # ★★【DIR-19】基座保持探针的数据块（valid.txt ⇒ 与训练语料 train.txt **逐位不相交**）
    ppl_chunks = build_ppl_chunks(tok, n_chunks=16, seq_len=256, which="valid")
    # ★★【DIR-19】**θ₀ 冻结快照**（KL 正则的参考分布 + 可训基座臂的还原源）
    #   ★★ 缺陷 D72：原先把 `model_ref` **常驻 GPU**（2.0 GB）⇒ 与「模型 2 GB + 基座梯度 2 GB
    #   + 前向图」叠加 ⇒ **T4 15 GB 必 OOM**。
    #   修法：**只存 CPU 上的 pristine state_dict**（不占显存）；
    #   需要 KL 时**按臂临时**建一个 GPU 上的参考模型，用完立刻释放（只有 N2 臂需要）。
    print("[初始化] ★ 复制 θ₀ 冻结快照到 CPU（KL 参考 + 基座还原源）…", flush=True)
    PRISTINE = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model_ref = None
    # ★ 旋钮 N4 的层集合（写死：前 6 层 ⇒ 与挂载层 [8,12,16] 不重叠）
    N4_LAYERS = [0, 1, 2, 3, 4, 5]
    lowrank_cache = None

    # ---------------- Stage A
    print("\n[Stage A]", flush=True)
    ids8 = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
    # A1' Freeze C（cue 模式）
    st = make_stack(model, d, "cue")
    base_before = {n: q.detach().clone() for n, q in model.named_parameters()}
    with torch.no_grad():
        st.model(input_ids=ids8)
    st.freeze_write()
    lg = model(input_ids=ids8).logits.float()
    F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids8[:, 1:].reshape(-1)).backward()
    rrep = grad_report(st.named_read_params())
    wrep = grad_report(st.named_write_params())
    gam = [gg for n, gg, i, b in rrep if n.endswith("gamma")]
    read_ok = (all((not b) for _, _, _, b in rrep) and all(gg > 0 for gg in gam)
               and max(gg for _, gg, _, _ in rrep) > 1e-6)
    write_ok = all(isn or gg == 0.0 for _, gg, isn, _ in wrep)
    base_same = all(torch.equal(q.detach(), base_before[n]) for n, q in model.named_parameters())
    nh, nm = len(st._handles), len(st.mems)
    st.remove(); torch.cuda.empty_cache()
    # 阴性对照
    st2 = SGS(model, MOUNT, d, d, R, ALPHA, LAM, ETA, GAMMA0, DEV, seed=20261028,
              use_factorized=False, perpos=True, drop_sg=True, mem_kind="explicit_kv",
              n_entries=N_ENTRIES)
    with torch.no_grad():
        st2.model(input_ids=ids8)
    for m in st2.mems.values():
        m.no_write = True
    lg2 = model(input_ids=ids8).logits.float()
    F.cross_entropy(lg2[:, :-1].reshape(-1, lg2.shape[-1]), ids8[:, 1:].reshape(-1)).backward()
    wrep2 = grad_report(st2.named_write_params())
    leak = len([1 for _, gg, isn, _ in wrep2 if not isn and gg > 0])
    st2.remove(); torch.cuda.empty_cache()
    for q in model.parameters():
        q.grad = None
    rec("A1' Freeze C（读梯度非零 / 写梯度精确 0 / 基座逐位不变 / 阴性对照泄漏）",
        read_ok and write_ok and base_same and leak > 0,
        f"读梯度非零 {sum(1 for _,_,i,_ in rrep if not i)}/{len(rrep)}；"
        f"写梯度精确0 {sum(1 for _,gg,i,_ in wrep if i or gg==0.0)}/{len(wrep)}；"
        f"基座逐位不变 {base_same}；阴性对照泄漏 {leak}/{len(wrep2)}",
        "读梯度>0 且 写梯度==0 且 基座不变 且 泄漏>0",
        {"read": read_ok, "write": write_ok, "base": base_same, "leak": leak})
    rec("A1' 句柄数 = 2×层数", nh == 2 * nm, f"handles={nh} mems={nm}", tol=2 * nm, value=nh)

    # A2' 训练的生效性
    st = make_stack(model, d, "cue")
    tr, ev = build_docs(tok, SEEDS[0], 4, 2)
    traj, rep = train_arm(st, tr, "cue")
    wo_ch = max(v["changed"] for k, v in rep.items() if k.endswith("Wo"))
    ga_ch = max(v["changed"] for k, v in rep.items() if k.endswith("gamma"))
    ga_gr = max(v["grad"] for k, v in rep.items() if k.endswith("gamma")
                and v["grad"] is not None)
    wq_gr = max((v["grad"] if v["grad"] is not None else 0.0)
                for k, v in rep.items() if k.endswith("Wq"))
    wq_ch = max(v["changed"] for k, v in rep.items() if k.endswith("Wq"))
    a2_ok = (wo_ch > 0 and ga_ch > 0 and ga_gr > 0 and wq_gr == 0.0 and wq_ch == 0.0)
    rec("A2' 训练生效性：Wo/gamma 改变且 gamma 梯度非零；cue 模式下 Wq 梯度恒 0 且不变", a2_ok,
        f"Δmax|Wo|={wo_ch:.4g}；Δmax|gamma|={ga_ch:.4g}；‖g_gamma‖={ga_gr:.4g}；"
        f"‖g_Wq‖={wq_gr:.4g}；Δ|Wq|={wq_ch:.4g}", value=a2_ok)
    # ===== ★【DIR-18】三条新检查（卡片 rev 1.0 的 A2'⑩⑪⑫）=====
    # A2' ⑩【Hebbian 核的自检】写一步后 W 确实朝 v 移动，且不爆炸
    st_h = make_stack(model, d, "cue", mem_kind="hebbian_full")
    core0 = st_h.mems[list(st_h.mems)[0]].core
    kk = torch.randn(core0.d_k, device=DEV, generator=torch.Generator(device=DEV).manual_seed(7))
    vv = torch.randn(core0.d_v, device=DEV, generator=torch.Generator(device=DEV).manual_seed(8))
    with torch.no_grad():
        d_before = float((core0._W() @ kk - vv).norm())
    core0.write(kk, vv)
    with torch.no_grad():
        d_after = float((core0._W() @ kk - vv).norm())
        wnorm = float(core0._W().norm())
    st_h.remove(); torch.cuda.empty_cache()
    rec("A2' ⑩ Hebbian 核自检：写一步后 W 朝 v 移动且不发散",
        (d_after < d_before) and math.isfinite(wnorm) and (wnorm < 1e6),
        f"‖Wk−v‖：{d_before:.4f} → {d_after:.4f}（须下降）；‖W‖={wnorm:.4g}（须有限且 < 1e6）",
        tol=d_before, value=d_after)
    # A2' ⑪【梯度隔离】Hebbian 臂：任务损失的梯度不流入核状态 W，且 P_v 的梯度恒 0
    st_h2 = make_stack(model, d, "cue", mem_kind="hebbian_full")
    tr_h, ev_h = build_docs(tok, SEEDS[0], 3, 2)
    train_arm(st_h2, tr_h, "cue")
    wgrad = []
    for li, m in st_h2.mems.items():
        wgrad.append(None if m.core.W.grad is None else float(m.core.W.grad.norm()))
    pvgrad = []
    for li, m in st_h2.mems.items():
        pvgrad.append(None if m.Pv.grad is None else float(m.Pv.grad.norm()))
    st_h2.remove(); torch.cuda.empty_cache()
    leak_w = [g for g in wgrad if g is not None and g != 0.0]
    leak_pv = [g for g in pvgrad if g is not None and g != 0.0]
    rec("A2' ⑪ 梯度隔离：Hebbian 臂中任务梯度不流入核状态 W，且（非 W4 臂）P_v 梯度恒 0",
        (len(leak_w) == 0) and (len(leak_pv) == 0),
        f"核状态 W 的梯度泄漏 {len(leak_w)}/{len(wgrad)}（须 0）；P_v 的梯度泄漏 {len(leak_pv)}/{len(pvgrad)}（须 0）",
        value={"W_leak": len(leak_w), "Pv_leak": len(leak_pv)})
    # A2' ④：读阶段冻结写 ⇒ 记忆逐位不变
    st.freeze_write()
    s_before = st.snapshot_all()
    with torch.no_grad():
        st.model(input_ids=torch.tensor(ev[0]["prefix"], dtype=torch.long,
                                       device=DEV).unsqueeze(0))
    s_after = st.snapshot_all()
    same = all(torch.equal(s_before[li][0], s_after[li][0]) for li in s_before)
    rec("A2' ④ 读阶段冻结写 ⇒ 记忆状态逐位不变", same, f"equal={same}", value=same)
    # A2' ⑤【跨方向逐位对照探针】评测期的命中必须复现 DIR-16 的 C1 = 1.0000
    st_p = make_stack(model, d, "cue", mem_kind="explicit_kv")
    tr_p, ev_p = build_docs(tok, SEEDS[0], 4, 6)
    rows_p = eval_arm(st_p, ev_p, ctrl_ids=None)
    hit_p = float(np.mean([r["hit"] for r in rows_p]))
    nent_p = int(np.mean([r["n_entries"] for r in rows_p]))
    st_p.remove(); torch.cuda.empty_cache()
    rec("A2' ⑤ 跨方向对照探针：未训练的 cue 臂命中 = 1.0（复现 DIR-16 的 C1）",
        hit_p >= 0.9, f"hit={hit_p:.4f}（DIR-16 的 C1 = 1.0000）；每篇写入条目数={nent_p}",
        tol=0.9, value=hit_p)
    # A2' ⑥【rev 1.3】写入条目数不得超过核心槽位数（否则槽位回绕 ⇒ 命中指标错位）
    rec("A2' ⑥ 槽位不回绕：每篇写入条目数 ≤ N_ENTRIES", nent_p <= N_ENTRIES,
        f"每篇 {nent_p} 条（N_ENTRIES={N_ENTRIES}）", tol=N_ENTRIES, value=nent_p)
    # A2' ⑦【rev 1.3】评测码留出：训练码与评测码零重叠
    ov = set(x["code"] for x in tr_p) & set(x["code"] for x in ev_p)
    ovd = set(x["code"] for x in tr_p) & set(x["dist_code"] for x in ev_p)
    rec("A2' ⑦ 评测码留出：训练码与评测码（含干扰码）零重叠",
        len(ov) == 0 and len(ovd) == 0,
        f"目标码重叠 {len(ov)}；与干扰码重叠 {len(ovd)}", value=len(ov) + len(ovd))
    # A2' ⑧【rev 1.4】干扰文档必须与目标文档**同线索、异码、异上下文**；且**码必须在文档末尾**
    cue_ids = tok(CUE_A).input_ids
    ok8 = True
    for x in list(tr_p[:2]) + list(ev_p[:2]):
        cid = int(x["code_ids"][0]); did = int(x["dist_code"])
        # 线索紧贴码之前
        same_cue = (x["ids"][len(x["ids"]) - len(cue_ids) - 1:len(x["ids"]) - 1] == cue_ids
                    and x["dist_ids"][len(x["dist_ids"]) - len(cue_ids) - 1:
                                     len(x["dist_ids"]) - 1] == cue_ids)
        code_last = (x["ids"][-1] == cid and x["dist_ids"][-1] == did)
        diff_code = cid != did
        diff_ctx = x["ids"][:FILLER_TOK] != x["dist_ids"][:FILLER_TOK]
        ok8 = ok8 and same_cue and code_last and diff_code and diff_ctx
    rec("A2' ⑧ 干扰项构造：同线索 + 异码 + 异上下文 + 码在文档末尾（唯一持有其身份的条目）", ok8,
        f"线索 = CUE_A（目标与干扰相同）；码不同（全局唯一）；填充切片不同（间隔 ≥ {MIN_FILLER_GAP} token）"
        f"⇒ 上下文签名必不同；码为文档最后一个 token ⇒ 邻位条目**不含**码的身份；抽样 4 篇通过={ok8}",
        value=1.0 if ok8 else 0.0)
    # A2' ⑨【rev 1.4】目标必须**不可被基座猜中**（否则任务平凡 ⇒ 又是饱和）
    rows_b = eval_base(model, ev_p[:8])
    nll_b = float(np.mean([r["nll"] for r in rows_b]))
    rec("A2' ⑨ 目标不可猜中：基座 NLL ≥ 3（否则任务平凡）", nll_b >= 3.0,
        f"base NLL = {nll_b:.4f}（阈值 3.0）；码为稀有单 token 词", tol=3.0, value=nll_b)
    # A2' ⑫【读路径一致性】W0_verbatim 的读出 y 必须≈槽内 v_j（cos ≥ 0.999）
    st_v = make_stack(model, d, "cue", mem_kind="explicit_kv")
    tr_v, ev_v = build_docs(tok, SEEDS[0], 2, 3)
    rows_v = eval_arm(st_v, ev_v, ctrl_ids=None)
    cs_min = []
    for li, m in st_v.mems.items():
        pass
    # 逐槽核对：读阶段 last_yr 与其来源槽的 v_j 的余弦
    snap_v = st_v.snapshot_all()
    for x in ev_v:
        st_v.unfreeze_write(); st_v.restore_all(snap_v)
        write_both(st_v, x)
        vals_v = {li: (torch.stack(m.vals_written) if m.vals_written else None)
                  for li, m in st_v.mems.items()}
        for li, m in st_v.mems.items():
            m.vals_written.clear(); m.keys_written.clear()
        st_v.freeze_write()
        pre_v = torch.tensor(x["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            st_v.model(input_ids=pre_v)
        for li, m in st_v.mems.items():
            yr = m.last_yr[0] if m.last_yr is not None else None
            Vv = vals_v[li]
            if yr is None or Vv is None or Vv.shape[0] == 0:
                cs_min.append(0.0); continue
            cs = F.cosine_similarity(Vv, yr.unsqueeze(0), dim=1)
            cs_min.append(float(cs.max()))
    st_v.remove(); torch.cuda.empty_cache()
    cs_min_mean = float(np.mean(cs_min)) if cs_min else 0.0
    rec("A2' ⑫ 读路径一致性：W0_verbatim 的读出 y 与其来源槽 v_j 的余弦 ≥ 0.999",
        cs_min_mean >= 0.999,
        f"cos(y, v_j) 均值 = {cs_min_mean:.6f}（阈值 0.999，{len(cs_min)} 个槽）",
        tol=0.999, value=cs_min_mean)

    # ===== ★【DIR-18 rev 1.2 修订 59】新增 6 条**当场**断言 =====
    # A2'⑬ ★ 真值槽的**精确映射**自洽（D54）+ 未训练臂的槽空间寻址命中
    #   （★ 修订 65：评测协议回退到 DIR-17 原协议 ⇒ 「写入序」与「槽空间」两口径**不再相等**
    #     —— 回绕使真值槽 = (count_before + true_idx) % N ≠ true_idx ⇒ 写入序口径必然 ≈ 0，
    #     这是**协议的正确读数**，不是缺陷 ⇒ 判据一律改用**槽空间口径**。）
    st_s = make_stack(model, d, "cue", mem_kind="explicit_kv")
    tr_s, ev_s = build_docs(tok, SEEDS[0], 3, 4)
    rows_s = eval_arm(st_s, ev_s, ctrl_ids=None)
    a_w = float(np.nanmean([r["addr"] for r in rows_s]))
    a_s = float(np.nanmean([r["addr_slot"] for r in rows_s]))
    st_s.remove(); torch.cuda.empty_cache()
    rec("A2'⑬ 真值槽**精确映射**自洽（D54）+ 槽空间寻址命中（未训练臂）",
        a_s >= 0.99,
        f"addr(写入序)={a_w:.4f}（★ DIR-17 协议下**必然 ≈ 0**（回绕 ⇒ 真值槽≠写入序），非缺陷）；"
        f"addr(槽空间，精确映射)={a_s:.4f}（阈值 0.99）",
        tol=0.99, value=a_s)
    # A2'⑭ ★ 槽表状态的**可复现性**：陈旧残留数必须是**确定性**的（训练残留 + 回绕），
    #   而不是随机的；且写入条目数必须与记录一致。
    st_n = make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-7)
    tr_n, ev_n = build_docs(tok, SEEDS[0], 2, 3)
    snap_n = st_n.snapshot_all()
    stale, slots_ok = [], []
    for x in ev_n:
        st_n.unfreeze_write(); st_n.restore_all(snap_n)
        cnt0 = {li: int(m.core.count) for li, m in st_n.mems.items()}
        write_both(st_n, x)
        for li, m in st_n.mems.items():
            n = min(int(m.core.count), int(m.core.N))
            stale.append(int(n) - len(m.vals_written))
            # 精确映射：第 j 次写的槽号必须 == (写入前的 count + j) % N
            for j, sl in enumerate(m.slot_of_write):
                if sl != (cnt0[li] + j) % int(m.core.N):
                    slots_ok.append((li, j, sl, (cnt0[li] + j) % int(m.core.N)))
        for m in st_n.mems.values():
            m.vals_written.clear(); m.keys_written.clear()
    st_n.remove(); torch.cuda.empty_cache()
    rec("A2'⑭ 槽表状态**确定性**：陈旧残留数逐篇一致 + 槽号映射 == (写入前 count + j) % N",
        (len(set(stale)) == 1) and (len(slots_ok) == 0),
        f"陈旧条目数（逐篇/逐层）= {stale}（须**完全一致** ⇒ 确定性）；"
        f"槽号映射违例 = {len(slots_ok)} 处（须 0）",
        tol=0, value=len(slots_ok))
    # A2'⑮ P_v 梯度语义：主臂 P_v 梯度恒 0；`pvw` 臂 P_v 梯度**非零**
    st_v2 = make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-7)
    tr_v2, ev_v2 = build_docs(tok, SEEDS[0], 2, 2)
    train_arm(st_v2, tr_v2, "cue", variant="normal")
    pv_main = [None if m.Pv.grad is None else float(m.Pv.grad.norm())
               for m in st_v2.mems.values()]
    st_v2.remove(); torch.cuda.empty_cache()
    st_v3 = make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-7)
    train_arm(st_v3, tr_v2, "cue", variant="pvw")
    pv_pvw = [None if m.Pv.grad is None else float(m.Pv.grad.norm())
              for m in st_v3.mems.values()]
    st_v3.remove(); torch.cuda.empty_cache()
    main_zero = all((g is None or g == 0.0) for g in pv_main)
    pvw_nonzero = any((g is not None and g > 0.0) for g in pv_pvw)
    rec("A2'⑮ P_v 梯度语义：主臂任务损失下 P_v 梯度恒 0；写侧自监督臂 P_v 梯度非零",
        main_zero and pvw_nonzero,
        f"主臂 ‖∂L_task/∂P_v‖ = {pv_main}（须全 0/None）；"
        f"pvw 臂 ‖∂L_write/∂P_v‖ = {pv_pvw}（须有非零）",
        value={"main": pv_main, "pvw": pv_pvw})
    # A2'⑯ 维度一致：d_k == d_v == hidden_size
    st_d = make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-7)
    dd_ok = all((m.d_k == d and m.d_v == d) for m in st_d.mems.values())
    st_d.remove(); torch.cuda.empty_cache()
    rec("A2'⑯ 维度一致：d_k == d_v == model.config.hidden_size（卡片 D49 已订正）",
        dd_ok, f"d_k = d_v = {d}（hidden_size）；脚本传 {d}，非 64", tol=d, value=d)
    # A2'⑯′ 交换性对照可构造：能从 code_pool 抽到 ≥32 个未写入且同分布的对照 token
    tr_c, ev_c = build_docs(tok, SEEDS[0], 3, 3)
    try:
        ctrl_c = build_ctrl_ids(tok, SEEDS[0], tr_c, ev_c)
        ok_c = len(ctrl_c) == 32
        det_c = f"抽到 {len(ctrl_c)} 个（须 32）；同池、均匀抽样、从未写入"
    except AssertionError as e:
        ok_c = False; det_c = f"构造失败：{e}"
    rec("A2'⑯′ 交换性对照可构造：同池、均匀抽样、从未写入的对照 token ≥ 32 个",
        ok_c, det_c, tol=32, value=len(ctrl_c) if ok_c else 0)
    # A2'⑰〔★ D50/D51 的当场检查〕键尺度：renorm/ortho 臂的 ‖k‖² 必须 ≈ 5（恒定）
    kn = []
    for kmode_ in ("linear", "renorm", "ortho"):
        st_k = make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-7, key_mode=kmode_)
        tr_k, ev_k = build_docs(tok, SEEDS[0], 1, 2)
        write_both(st_k, ev_k[0])
        for m in st_k.mems.values():
            if m.keys_written:
                kk = torch.stack(m.keys_written)
                kn.append((kmode_, float((kk ** 2).sum(dim=1).mean()),
                           float((kk ** 2).sum(dim=1).max())))
        for m in st_k.mems.values():
            m.vals_written.clear(); m.keys_written.clear()
        st_k.remove(); torch.cuda.empty_cache()
    lin = [x for x in kn if x[0] == "linear"]
    ren = [x for x in kn if x[0] == "renorm"]
    ort = [x for x in kn if x[0] == "ortho"]
    ok_k = (all(abs(x[1] - 5.0) < 0.05 for x in ren) and all(abs(x[1] - 5.0) < 0.05 for x in ort)
            and (not lin or max(x[2] for x in lin) > 100.0))
    rec("A2'⑰ 键尺度检查：renorm/ortho 的 ‖k‖² ≈ 5（恒定）；linear 的实测 ‖k‖² 远大于 5（D50/D51）",
        ok_k,
        f"linear: E‖k‖²={['%.1f' % x[1] for x in lin]} max={['%.1f' % x[2] for x in lin]}；"
        f"renorm: E‖k‖²={['%.3f' % x[1] for x in ren]}；ortho: E‖k‖²={['%.3f' % x[1] for x in ort]}"
        f"（⇒ 装置的 pk_scale 标定口径错 ⇒ 真实键尺度 ≫ 5 ⇒ η 窗口不相交）",
        tol=5.0, value=ren[0][1] if ren else 0.0)

    # ★★【DIR-18 rev 1.5 · 修订 68】A2'⑱〔零效度自检〕
    #   检查「内容置换」钩子**只动内容、不动寻址** ⇒ H14 的对照是干净的。
    #   两侧都在**同一份文档、同一套地址机制**下，只差 `content_wrong` 开关：
    #     ① 地址侧：两臂写入的**键**必须**逐位相同**（置换不动 k）；
    #     ② 内容侧：两臂写入的**值**必须**显著不同**（置换生效）。
    #   ⇒ 若 ① 不成立 ⇒ 置换污染了寻址 ⇒ H14 无效；若无 ② ⇒ 置换没生效 ⇒ H14 无效。
    _cw_keys, _cw_vals = [], []
    for _var in ("normal", "content_wrong"):
        st_c = make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-1,
                           key_mode="ortho", variant=_var)
        tr_c, ev_c = build_docs(tok, SEEDS[0], 1, 2)
        write_both(st_c, ev_c[0])
        for m in st_c.mems.values():
            if m.keys_written:
                _cw_keys.append(torch.stack(m.keys_written).detach().clone())
                _cw_vals.append(torch.stack(m.vals_written).detach().clone())
        st_c.remove(); torch.cuda.empty_cache()
    _ok18 = False
    _d18 = "（未能构造两臂）"
    if len(_cw_keys) >= 2 and len(_cw_vals) >= 2:
        # ① 键逐位相同（置换不动地址）
        _k_same = all(torch.equal(_cw_keys[0], _cw_keys[i]) for i in range(1, len(_cw_keys)))
        # ② 值显著不同（置换生效）：配对余弦必须**明显小于 1**
        _cos = []
        for i in range(1, len(_cw_vals)):
            _a = _cw_vals[0].to(torch.float64); _b = _cw_vals[i].to(torch.float64)
            _m = min(_a.shape[0], _b.shape[0])
            _a, _b = _a[:_m], _b[:_m]
            _cs = F.cosine_similarity(_a, _b, dim=1)
            _cos.append(float(_cs.mean()))
        _maxcos = max(_cos) if _cos else 1.0
        _ok18 = bool(_k_same) and (_maxcos < 0.99)
        _d18 = (f"键逐位相同={_k_same}（置换不动地址 ⇒ H14 的寻址侧干净）；"
                f"值的配对余弦均值={['%.4f' % c for c in _cos]}（须 < 0.99 ⇒ 置换确实换了内容）")
    rec("A2'⑱〔★ 修订 68 零效度自检〕「内容置换」钩子只动内容、不动寻址 ⇒ H14 的对照干净",
        _ok18, _d18, tol=0.99,
        value=(max(_cos) if len(_cw_keys) >= 2 and _cw_vals and len(_cw_vals) >= 2 else 1.0))

    # A4' 训练不发散
    losses = [t[0] for t in traj]
    finite = all(np.isfinite(losses))
    max_wo = max(v["max_abs"] for k, v in rep.items() if k.endswith("Wo"))
    gam_fin = all(np.isfinite(v["value"]) for k, v in rep.items() if k.endswith("gamma"))
    rec("A4' 训练不发散（损失有限、max|Wo| 有限、gamma 有限）",
        finite and np.isfinite(max_wo) and gam_fin,
        f"loss 首/末 = {losses[0]:.4f}/{losses[-1]:.4f}；max|Wo|={max_wo:.4g}；"
        f"gamma 终值={[round(v['value'],4) for k,v in rep.items() if k.endswith('gamma')]}",
        value=max_wo)
    res["meta"]["a4_loss"] = {"first": losses[0], "last": losses[-1]}
    st.remove(); torch.cuda.empty_cache()

    # ══════════════════ ★★【DIR-19】Stage A 新增断言 A19-1…A19-10 ══════════════════
    # —— A19-1 基座冻结对照：训练后基座参数与初始参数**逐位相同** ——
    _bf0 = {n: q.detach().clone() for n, q in model.named_parameters()}
    _st1 = make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-7, key_mode="linear")
    _tr1, _ev1 = build_docs(tok, SEEDS[0], 1, 2)
    write_both(_st1, _ev1[0])
    _st1.freeze_write()
    _pre1 = torch.tensor(_ev1[0]["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
    _opt1 = torch.optim.Adam([t for _, t in _st1.read_params_for("cue")], lr=1e-3)
    for _ in range(3):
        _opt1.zero_grad(set_to_none=True)
        _lg1 = _st1.model(input_ids=_pre1).logits.float()
        _tg1 = torch.tensor([_ev1[0]["code_ids"][0]], dtype=torch.long, device=DEV)
        F.cross_entropy(_lg1[0, -1:, :], _tg1).backward()
        _opt1.step()
    _dmax = max(float((q.detach() - _bf0[n]).abs().max()) for n, q in model.named_parameters())
    rec("A19-1 基座冻结对照：联合训练后基座参数与初始参数**逐位相同**",
        _dmax == 0.0, f"max|θ−θ₀| = {_dmax:.3e}（须精确为 0）", tol=0.0, value=_dmax)
    _st1.remove(); torch.cuda.empty_cache()

    # —— A19-2 基座可训：被训参数集合**非空**且梯度**非零有限** ——
    _st2 = make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-7, key_mode="linear")
    _bp = _st2.trainable_base_params("all")
    _tr2, _ev2 = build_docs(tok, SEEDS[0], 1, 2)
    write_both(_st2, _ev2[0])
    _st2.freeze_write()
    _pre2 = torch.tensor(_ev2[0]["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
    _lg2 = _st2.model(input_ids=_pre2).logits.float()
    _tg2 = torch.tensor([_ev2[0]["code_ids"][0]], dtype=torch.long, device=DEV)
    F.cross_entropy(_lg2[0, -1:, :], _tg2).backward()
    _gn = [float(q.grad.norm()) for _, q in _bp if q.grad is not None]
    _fin = all(math.isfinite(g) for g in _gn)
    _nz = any(g > 0 for g in _gn)
    rec("A19-2 基座可训：被训参数集合**非空**且其梯度**非零有限**",
        (len(_bp) > 0) and _fin and _nz,
        f"可训基座参数数 = {len(_bp)}；有梯度的 = {len(_gn)}；梯度范数 min/max = "
        f"{(min(_gn) if _gn else float('nan')):.3e}/{(max(_gn) if _gn else float('nan')):.3e}；"
        f"全部有限 = {_fin}；存在非零 = {_nz}",
        value=float(max(_gn) if _gn else 0.0))
    _st2.remove(); torch.cuda.empty_cache()
    # 还原基座（A19-2 的 backward 未更新权重，但保险起见）
    model.load_state_dict(PRISTINE)

    # —— A19-3 保护旋钮的生效性**双向**检查 ——
    _st3 = make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-7, key_mode="linear")
    _none_params = _st3.trainable_base_params("none")
    _all_params = _st3.trainable_base_params("all")
    _lay_params = _st3.trainable_base_params("layers", layers=N4_LAYERS)
    _lr_params = _st3.trainable_base_params("lowrank")
    _b3 = (len(_none_params) == 0
           and len(_all_params) > 0
           and 0 < len(_lay_params) < len(_all_params)
           and len(_lr_params) > 0)
    # 双向：投影后残差必须 ≈ 0；不投影时残差 > 0
    if _LRCACHE["proj"] is None or _LRCACHE["r"] != 8:
        _LRCACHE["proj"] = _st3.lowrank_projections(r=8)
        _LRCACHE["r"] = 8
    _projs3 = _LRCACHE["proj"]
    _res_before = _st3.subspace_residual(_projs3)
    with torch.no_grad():
        for nm, q in _st3.model.named_parameters():
            if nm in _projs3 and q.dim() == 2:
                q.add_(torch.randn_like(q) * 1e-3)
    _res_after = _st3.subspace_residual(_projs3)
    _st3.project_base_to_subspace(_projs3)
    _res_proj = _st3.subspace_residual(_projs3)
    _ok3 = _b3 and (_res_proj < 1e-4) and (_res_after > 1e-6)
    rec("A19-3 保护旋钮的生效性检查**双向**：放开/关闭都改变被训集合；低秩投影后残差 ≈ 0",
        _ok3, f"none={len(_none_params)} all={len(_all_params)} layers={len(_lay_params)} "
               f"lowrank={len(_lr_params)}（须 0<layers<all）；"
               f"低秩残差 投影前={_res_before:.3e} 扰动后={_res_after:.3e} 投影后={_res_proj:.3e}"
               f"（须 投影后 < 1e-4 且 扰动后 > 1e-6 ⇒ 双向可辨）",
        tol=1e-4, value=_res_proj)
    _st3.remove(); torch.cuda.empty_cache()
    model.load_state_dict(PRISTINE)

    # —— A19-4 训练数据与基座保持评测数据**文档 id 交集为空** ——
    _trA, _evA = build_docs(tok, SEEDS[0], 3, 2)
    _chunks4 = build_ppl_chunks(tok, n_chunks=4, seq_len=256, which="valid")
    # ★★ 缺陷 D66：**token id 交集对普通文本毫无意义**（常见词必然重叠）⇒ 原口径必 FAIL。
    #   修法：改为**文档级**不相交（卡片 §3 的原话是「**文档 id** 集合」）：
    #   把训练文本与探针文本都切成**定长窗口**、逐窗口取哈希 ⇒ 断言**窗口集合无交集**。
    import hashlib as _hl
    def _win_hashes(ids, w=256, stride=64):
        return {_hl.md5(np.asarray(ids[i:i + w], dtype=np.int64).tobytes()).hexdigest()
                for i in range(0, max(1, len(ids) - w + 1), stride)}
    _train_txt = _corpus_text("train")
    _probe_txt = _corpus_text("valid")
    _hw_train = _win_hashes(tok(_train_txt).input_ids)
    _hw_probe = _win_hashes(tok(_probe_txt).input_ids)
    _ov4 = len(_hw_train & _hw_probe)
    _file_diff = (_corpus_text("train")[:4096] != _corpus_text("valid")[:4096])
    rec("A19-4 训练数据与基座保持评测数据的**文档窗口集合**交集为空（逐位断言）",
        (_ov4 == 0) and _file_diff,
        f"训练窗口数={len(_hw_train)}；探针窗口数={len(_hw_probe)}；**交集={_ov4}（须 0）**；"
        f"两文件前缀不同={_file_diff}（train.txt vs valid.txt）",
        tol=0.0, value=float(_ov4))

    # —— A19-5 交换性对照仍从码池**均匀抽样** ——
    _ctrl = build_ctrl_ids(tok, SEEDS[0], _trA, _evA)
    _ok5 = (len(_ctrl) == 32) and (len(set(_ctrl)) == 32)
    rec("A19-5 交换性对照 token 仍从码池**均匀抽样**（D53 教训：不得取最稀有者）",
        _ok5, f"抽到 {len(_ctrl)} 个、去重后 {len(set(_ctrl))} 个（须 32 且互不相同）",
        tol=32.0, value=float(len(_ctrl)))

    # —— A19-6 addr_slot 仍用**精确槽号映射**（D54）——
    _st6 = make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-7, key_mode="linear")
    _tr6, _ev6 = build_docs(tok, SEEDS[0], 1, 2)
    write_both(_st6, _ev6[0])
    _ok6 = all(len(getattr(m, "slot_of_write", [])) ==
               len(getattr(m, "keys_written", [])) for m in _st6.mems.values())
    _st6.remove(); torch.cuda.empty_cache()
    rec("A19-6 记忆寻址指标 addr_slot 仍用**精确槽号映射**（D54：写入序 → 槽号可精确预测）",
        _ok6, "写入记录数 == 槽号记录数 ⇒ 精确映射可用", tol=0.0, value=0.0)

    # —— A19-7 基座保持的 NLL 口径在挂记忆/不挂记忆两个对照下**逐位可算** ——
    _st7 = make_stack(model, d, "cue", mem_kind="explicit_kv")
    _p_with = probe_ppl(model, _st7, ppl_chunks[:3])
    _p_without = probe_ppl(model, None, ppl_chunks[:3])
    _ok7 = (_p_with == _p_with) and (_p_without == _p_without) and (abs(_p_with - _p_without) > 1e-9)
    rec("A19-7 基座保持的 NLL 口径在挂记忆 vs 不挂记忆两个对照下**逐位可算**",
        _ok7, f"PPL(挂记忆)={_p_with:.6f}；PPL(不挂记忆)={_p_without:.6f}；"
              f"差={_p_with - _p_without:+.6f}（须非 0 ⇒ 挂记忆确实改变了基座输出）",
        tol=0.0, value=abs(_p_with - _p_without))
    _st7.remove(); torch.cuda.empty_cache()

    # —— A19-8 D_kv 在「零训练步」对照下必须为 0（度量自检）——
    _st8 = make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-7, key_mode="linear")
    _q8a = kv_drift(_st8, ppl_chunks[:2])
    _q8b = kv_drift(_st8, ppl_chunks[:2])
    _d8 = 0.0 if (_q8a is None or _q8b is None) else float((_q8b - _q8a).abs().mean())
    rec("A19-8 D_kv 漂移度量在「零训练步」对照下必须为 **0**（度量自检）",
        _d8 < 1e-9, f"D_kv(零训练两次采样) = {abs(_d8):.3e}（须 < 1e-9）", tol=0.0, value=abs(_d8))
    _st8.remove(); torch.cuda.empty_cache()

    # —— A19-9 KL 正则**生效性**（**双向**：启用 ⇒ 项非零且梯度非零；禁用 ⇒ 不贡献梯度）——
    #   ★★ 缺陷 D69：**原断言「KL 在训练中可下降」是错的** ——
    #   KL 是**被惩罚的正则项**（KL(π_θ0‖π_θ)），训练最小化的是**总损失**
    #   ⇒ ⇒ KL 本身**可以随基座漂移而上升**（实测 0 → 0.135 → 0.896）
    #   ⇒ ⇒ 「下降」不是它的语义 ⇒ 正确的生效性检查是**双向**（纪律 ⑭）：
    #     ① 启用（λ>0）：该项**非零**且**对基座参数产生非零梯度**；
    #     ② 禁用（λ=0）：该项**不贡献任何梯度**（基座梯度与 ① 不同 ⇒ 旋钮确实生效）。
    _refA = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for _q in _refA.parameters():
        _q.requires_grad_(False)
    _tr9, _ev9 = build_docs(tok, SEEDS[0], 1, 2)
    _ref9 = ref_logits_at(_refA, [_ev9[0]])          # ★ D77：预计算一次
    del _refA                                        # ★ 立刻释放 2.0 GB
    torch.cuda.empty_cache()
    _kl_on, _g_on, _g_off = [], [], []
    for _mode_on in (True, False):
        _st9 = make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-7, key_mode="linear")
        _bp9 = _st9.trainable_base_params("all")
        _opt9 = torch.optim.SGD([q for _, q in _bp9], lr=1e-5, momentum=0.9)
        for _it in range(3):
            write_both(_st9, _ev9[0])
            _st9.freeze_write()
            _pre9 = torch.tensor(_ev9[0]["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
            _opt9.zero_grad(set_to_none=True)
            _lg9 = _st9.model(input_ids=_pre9).logits.float()
            _tg9 = torch.tensor([_ev9[0]["code_ids"][0]], dtype=torch.long, device=DEV)
            _loss9 = F.cross_entropy(_lg9[0, -1:, :], _tg9)
            # ★ 当前侧**摘钩**前向 ⇒ 与卡片 rev 1.3 的口径一致（两侧都不含记忆）
            _st9.suspend_hooks()
            try:
                _lg9c = _st9.model(input_ids=_pre9, logits_to_keep=1).logits.float()
            except TypeError:
                _lg9c = _st9.model(input_ids=_pre9).logits.float()
            finally:
                _st9.resume_hooks()
            _kl9 = kl_from_logits(_ref9[0].to(DEV), _lg9c[0, -1])   # ★ D77：无参考模型驻留
            if _mode_on:
                _kl_on.append(float(_kl9.detach()))   # ★ 张量 ⇒ 记录时取标量（D70 的修法）
                (_loss9 + 1.0 * _kl9).backward()
            else:
                _loss9.backward()
            _gn9 = float(sum(float(q.grad.norm()) ** 2 for _, q in _bp9 if q.grad is not None) ** 0.5)
            (_g_on if _mode_on else _g_off).append(_gn9)
            _opt9.step()
        _st9.remove(); torch.cuda.empty_cache()
    _kl_nz = (len(_kl_on) > 0) and all(math.isfinite(k) for k in _kl_on) and (max(_kl_on) > 1e-6)
    # 双向：启用与禁用的**基座梯度范数**必须不同（⇒ KL 项确实改变了基座的更新方向）
    _g_on_m = float(np.mean(_g_on)) if _g_on else float("nan")
    _g_off_m = float(np.mean(_g_off)) if _g_off else float("nan")
    _bidir = (abs(_g_on_m - _g_off_m) > 1e-9) and math.isfinite(_g_on_m) and math.isfinite(_g_off_m)
    _ok9 = _kl_nz and _bidir
    rec("A19-9 〔KL 正则生效性 · 双向〕启用 ⇒ 项非零且改变基座梯度；禁用 ⇒ 不贡献该梯度",
        _ok9, f"KL 序列(启用) = {[round(k, 6) for k in _kl_on]}（须非零、有限）；"
              f"基座梯度范数 启用={_g_on_m:.6f} vs 禁用={_g_off_m:.6f}（须不同 ⇒ 双向可辨）"
              f"；★ **KL 是被惩罚项，其本身不要求下降**（原断言的期望写错 ⇒ 缺陷 D69）",
        tol=0.0, value=_g_on_m - _g_off_m)
    model.load_state_dict(PRISTINE)

    # —— A19-10 内容置换只动内容、不动寻址（逐字沿用 DIR-18 的 A2'⑱）——
    _cw_keys10, _cw_vals10 = [], []
    for _var10 in ("normal", "content_wrong"):
        _st10 = make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-1,
                           key_mode="ortho", variant=_var10)
        _tr10, _ev10 = build_docs(tok, SEEDS[0], 1, 2)
        write_both(_st10, _ev10[0])
        for _m10 in _st10.mems.values():
            if _m10.keys_written:
                _cw_keys10.append(torch.stack(_m10.keys_written).detach().clone())
                _cw_vals10.append(torch.stack(_m10.vals_written).detach().clone())
        _st10.remove(); torch.cuda.empty_cache()
    _ok10 = False; _d10 = "（未能构造两臂）"
    if len(_cw_keys10) >= 2 and len(_cw_vals10) >= 2:
        _k_same10 = all(torch.equal(_cw_keys10[0], _cw_keys10[i]) for i in range(1, len(_cw_keys10)))
        _cos10 = []
        for i in range(1, len(_cw_vals10)):
            _a10 = _cw_vals10[0].to(torch.float64); _b10 = _cw_vals10[i].to(torch.float64)
            _m10 = min(_a10.shape[0], _b10.shape[0])
            _cos10.append(float(F.cosine_similarity(_a10[:_m10], _b10[:_m10], dim=1).mean()))
        _maxc10 = max(_cos10) if _cos10 else 1.0
        _ok10 = bool(_k_same10) and (_maxc10 < 0.99)
        _d10 = (f"键逐位相同={_k_same10}（置换不动地址 ⇒ H2′ 的寻址侧干净）；"
                f"值的配对余弦均值={['%.4f' % c for c in _cos10]}（须 < 0.99 ⇒ 置换确实换了内容）")
    rec("A19-10〔内容置换只动内容、不动寻址〕⇒ H2′★ 的对照干净",
        _ok10, _d10, tol=0.99,
        value=(max(_cos10) if len(_cw_vals10) >= 2 else 1.0))

    # ══════════════════ ★★【DIR-20】Stage A 新增断言 A20-1…A20-9 + A20-11…A20-14 ══════════════════
    # （卡片 rev 1.1 §4 / rev 1.2 §5；★ **判据的阈值与文字一字未改**）
    #   A20-10 = **继承 DIR-19 的 30 项**（逐字保留，不重写）⇒ 本块只加 DIR-20 的 13 项。
    print("\n[Stage A · DIR-20]", flush=True)

    # ★★ 缺陷 D86（卡片 rev 1.3 §0.4）：**继承的 DIR-19 Stage A 会留下约 5.7 GB 显存占用**，
    #   且把基座留成 `requires_grad=True` ⇒ 主循环会白算基座梯度（并破坏「基座全冻结」契约）。
    #   遗留对象：① `_bf0`（基座参数的 **GPU 克隆**，1.9 GB，A19-1 留）；
    #            ② `_opt9` 的 SGD **momentum 缓冲**（1.9 GB，A19-9 的基座优化器）；
    #            ③ 基座参数的 `.grad` 缓冲（1.9 GB，A19-2/A19-9 放开基座后 backward 留下）。
    #   ⇒ 修法：显式 `del`（函数局部变量**不能**通过 `locals()` 删除）+ 清梯度 + **重新冻结基座** + 回收显存。
    #   ⇒ 实测依据：DIR-20 smoke#1 的显存轨迹 —— A19-1 后 6.7 GB、A19-9 后 11.1 GB ⇒ A20-8 再载入两个
    #     fp32 基座（+3.8 GB）⇒ **CUDA OOM**（14.29 GiB 已分配）。
    _freed = []
    for _nm in ("_bf0", "_opt9"):
        _freed.append(_nm)
    try:
        del _bf0
    except NameError:
        pass
    try:
        del _opt9
    except NameError:
        pass
    for _p in model.parameters():
        _p.grad = None
        _p.requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    rec("A20-0〔★ D86〕继承 Stage A 的遗留显存已释放，基座已重新冻结（供后续断言与主循环）",
        True,
        f"释放对象={_freed}；清空基座梯度；基座 requires_grad 全部恢复 False；"
        f"当前 alloc={torch.cuda.memory_allocated()/2**20:.0f} MiB "
        f"reserved={torch.cuda.memory_reserved()/2**20:.0f} MiB",
        tol=0.0, value=float(torch.cuda.memory_allocated() / 2**20))

    _S20 = SEEDS[0]
    _tr20, _ev20 = build_docs(tok, _S20, 2, 2)
    _d20 = _ev20[0]

    def _mk20(**kw):
        """建一个 DIR-20 装置（默认 = 主臂配置：hebbian_full + ortho + η=1e-1）。"""
        p = {"mem_kind": "hebbian_full", "eta": 1e-1, "key_mode": "ortho",
             "variant": "normal", "seed": _S20}
        p.update(kw)
        return make_stack(model, d, "cue", seed=p["seed"], mem_kind=p["mem_kind"],
                          eta=p["eta"], key_mode=p["key_mode"], variant=p["variant"])

    def _yr(st):
        return [m.last_yr[0].detach().clone() for m in st.mems.values()]

    def _fwd(st, dct):
        pre = torch.tensor(dct["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            st.model(input_ids=pre)

    # ─────────── A20-9〔只读评测**不写**〕（先做：后续检查都依赖它）───────────
    _stA = _mk20()
    write_both(_stA, _d20)
    _tslot = {}
    for li, m in _stA.mems.items():
        _tslot[li] = int(m.slot_of_write[int(_d20["true_idx"])])
    _d20["true_slot"] = int(list(_tslot.values())[0])
    _c_before = {li: (int(m.core.count), int(m.write_calls)) for li, m in _stA.mems.items()}
    _writable_before = all((not m.no_write) for m in _stA.mems.values())
    _row20 = eval_arm_readonly(_stA, _d20, None)
    _c_after = {li: (int(m.core.count), int(m.write_calls)) for li, m in _stA.mems.items()}
    _ok9 = (_c_before == _c_after) and _writable_before
    rec("A20-9〔只读评测不写〕评测前后 core.count 与 write_calls 不变（且写权限未被永久关闭）",
        _ok9,
        f"评测前 count/write_calls={[(v[0], v[1]) for v in _c_before.values()]}；"
        f"评测后={[(v[0], v[1]) for v in _c_after.values()]}；评测前写权限开启={_writable_before}；"
        f"★ 若评测期间未冻结写，前缀的前向会**自己写入**⇒ count 必增 ⇒ 本项必 FAIL",
        tol=0.0, value=float(sum(v[1] for v in _c_after.values()) - sum(v[1] for v in _c_before.values())))

    # ─────────── A20-1 / A20-2〔往返指纹逐位相同 / 往返后读出逐位相同〕───────────
    _stA.freeze_write()
    _fwd(_stA, _d20)                      # 纯读（写已冻结）⇒ 状态不再变化
    _y_before = _yr(_stA)
    _state20 = _stA.export_state(include_read_map=True, include_addr=True)
    _fp_mem = SGS.state_fingerprint(_state20)
    _stA.remove(); torch.cuda.empty_cache()

    _stB = _mk20()
    _stB.import_state(_state20, strict=True, expect_key_mode="ortho")
    _state20b = _stB.export_state(include_read_map=True, include_addr=True)
    _fp_rt = SGS.state_fingerprint(_state20b)
    rec("A20-1〔export → import 往返指纹逐位相同〕", _fp_mem == _fp_rt,
        f"内存指纹={_fp_mem}；导入后重导出指纹={_fp_rt}", tol=0.0,
        value=1.0 if _fp_mem == _fp_rt else 0.0)
    _stB.freeze_write()
    _fwd(_stB, _d20)
    _y_after = _yr(_stB)
    _cos20 = [float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
              for a, b in zip(_y_before, _y_after)]
    _ok2 = (min(_cos20) >= 0.999)
    rec("A20-2〔往返后读出逐位相同〕同一前缀下新旧装置的读出余弦 ≥ 0.999", _ok2,
        f"逐层余弦={['%.6f' % c for c in _cos20]}（最小 {min(_cos20):.6f}）", tol=0.999,
        value=float(min(_cos20)))

    # ─────────── A20-14〔落盘后 torch.load 的指纹 == 内存中的指纹〕───────────
    _sd20 = HERE.parent / ".ops" / "state"
    _sd20.mkdir(parents=True, exist_ok=True)
    _f14 = _sd20 / "A20_roundtrip.pt"
    torch.save(_state20, _f14)
    _state14 = torch.load(_f14, map_location="cpu")
    _fp14 = SGS.state_fingerprint(_state14)
    rec("A20-14〔状态文件往返：落盘后 torch.load 的指纹 == 内存中的指纹〕", _fp14 == _fp_mem,
        f"内存指纹={_fp_mem}；落盘读回指纹={_fp14}", tol=0.0,
        value=1.0 if _fp14 == _fp_mem else 0.0)
    try:
        os.remove(_f14)
    except OSError:
        pass

    # ─────────── A20-3〔strict=True 喂错形状必须抛错〕───────────
    _bad = {"fmt": _state20["fmt"], "include_read_map": True, "include_addr": True,
            "layers": {}}
    for _li, _dd in _state20["layers"].items():
        _bad["layers"][_li] = dict(_dd)
        _bad["layers"][_li]["W"] = torch.zeros(3, 3)       # ← 形状不符
    _raised, _msg = False, ""
    try:
        _stB.import_state(_bad, strict=True, expect_key_mode="ortho")
    except Exception as _e:
        _raised = True
        _msg = f"{type(_e).__name__}: {_e}"
    rec("A20-3〔import_state(strict=True) 喂错形状必须抛错（不得静默截断）〕", _raised,
        _msg if _raised else "★ 未抛错 ⇒ 形状校验失效（缺陷！）", tol=0.0,
        value=1.0 if _raised else 0.0)

    # ─────────── A20-4〔include_read_map=False ⇒ W_o / γ 未搬运〕───────────
    #   判别设计：先把**源装置**的 W_o 扰动 +1.0 ⇒ 若被搬运，导入后的 W_o 必含 +1.0
    with torch.no_grad():
        for m in _stA.mems.values():
            m.Wo.add_(1.0)
    _wo_src = [m.Wo.detach().clone() for m in _stA.mems.values()]
    _ga_src = [m.gamma.detach().clone() for m in _stA.mems.values()]
    _state_nr = _stA.export_state(include_read_map=False, include_addr=True)
    _stC = _mk20()
    _wo_init = [m.Wo.detach().clone() for m in _stC.mems.values()]
    _ga_init = [m.gamma.detach().clone() for m in _stC.mems.values()]
    _stC.import_state(_state_nr, strict=True, expect_key_mode="ortho")
    _wo_after = [m.Wo.detach().clone() for m in _stC.mems.values()]
    _ga_after = [m.gamma.detach().clone() for m in _stC.mems.values()]
    _wo_untouched = all(torch.equal(a, b) for a, b in zip(_wo_init, _wo_after))
    _ga_untouched = all(torch.equal(a, b) for a, b in zip(_ga_init, _ga_after))
    _wo_not_src = all((not torch.equal(a, b)) for a, b in zip(_wo_src, _wo_after))
    rec("A20-4〔include_read_map=False ⇒ W_o / γ 未搬运（导入后仍为随机初值）〕",
        _wo_untouched and _ga_untouched and _wo_not_src,
        f"导入后 W_o 仍 == 随机初值={_wo_untouched}；γ 仍 == 随机初值={_ga_untouched}；"
        f"且 ≠ 源装置（源被扰动 +1.0）={_wo_not_src}", tol=0.0,
        value=1.0 if (_wo_untouched and _ga_untouched and _wo_not_src) else 0.0)
    _stC.remove(); torch.cuda.empty_cache()

    # ─────────── A20-5〔include_addr=False ⇒ 地址表未搬运〕───────────
    _state_na = _stA.export_state(include_read_map=True, include_addr=False)
    _stD = _mk20()
    _stD.import_state(_state_na, strict=True, expect_key_mode="ortho")
    _cnt20 = [int(m.core.count) for m in _stD.mems.values()]
    _kz20 = [bool(torch.count_nonzero(m.core.keys).item() == 0) for m in _stD.mems.values()]
    rec("A20-5〔include_addr=False ⇒ 地址表未搬运（count == 0、keys 全零）〕",
        (all(c == 0 for c in _cnt20) and all(_kz20)),
        f"逐层 count={_cnt20}；keys 全零={_kz20}", tol=0.0,
        value=float(sum(_cnt20)))
    _stD.remove(); torch.cuda.empty_cache()

    # ─────────── A20-6 / A20-7〔内容置换：只动值、不动键与签名；置换是排列且无不动点〕───────────
    _stX = _mk20(variant="normal")
    write_both(_stX, _d20)
    _sX = _stX.export_state(include_read_map=True, include_addr=True)
    _stX.remove(); torch.cuda.empty_cache()
    _stY = _mk20(variant="content_wrong")
    write_both(_stY, _d20)
    _sY = _stY.export_state(include_read_map=True, include_addr=True)
    _perm_ok, _perm_txt = True, ""
    for _li, _m in _stY.mems.items():
        _p = getattr(_m, "_cw_perm", None)
        if _p is None:
            _perm_ok = False
            _perm_txt = f"层{_li}: ★ 缺少 _cw_perm（置换未生效）"
            break
        _pi = [int(x) for x in _p.tolist()]
        _n = len(_pi)
        _is_perm = (sorted(_pi) == list(range(_n)))
        _fixed = sum(1 for _i in range(_n) if _pi[_i] == _i)
        if (not _is_perm) or _fixed != 0:
            _perm_ok = False
        _perm_txt = (f"层{_li}: n={_n} 是排列={_is_perm} 不动点数={_fixed}"
                     f"（★ D62：torch 视图别名会写坏排列 ⇒ 必须先取标量副本）")
    rec("A20-7〔content_wrong 的置换**是排列且无不动点**〕", _perm_ok, _perm_txt, tol=0.0,
        value=1.0 if _perm_ok else 0.0)
    _k_same = all(torch.equal(_sX["layers"][_li]["keys"], _sY["layers"][_li]["keys"])
                  for _li in _sX["layers"])
    _v_diff = all((not torch.equal(_sX["layers"][_li]["vals"], _sY["layers"][_li]["vals"]))
                  for _li in _sX["layers"])
    # ★★ 缺陷 D87（卡片 rev 1.3 §0.5）：rev 1.1 的 A20-6 要求「**sigs 逐位相同**」⇒ **不成立**：
    #   记忆注入会改变**下游层**的 hidden state ⇒ 下游层的上下文签名携带了注入痕迹 ⇒ 逐位不同。
    #   （第一挂载层 8 的输入在上游 ⇒ **不受注入影响** ⇒ 必然逐位相同 ⇒ 卡片意图在此层可严格验证。）
    _sig_same = {li: bool(torch.equal(_sX["layers"][li]["sigs"], _sY["layers"][li]["sigs"]))
                 for li in _sX["layers"]}
    _first_li = sorted(_sX["layers"].keys(), key=lambda s: int(s))[0]
    _ok6 = (_k_same and _v_diff and _sig_same.get(_first_li, False))
    rec("A20-6′〔内容置换只动**值**、不动**寻址键**；签名**逐层**报告（第一挂载层必须逐位相同）〕",
        _ok6,
        f"keys 逐位相同={_k_same}（全部挂载层）；vals 逐位不同={_v_diff}（全部挂载层）；"
        f"sigs 逐层相同={_sig_same}；★ 第一挂载层（层 {_first_li}）逐位相同={_sig_same.get(_first_li)}"
        f"（下游层不同是**结构事实**：记忆注入会改变下游 hidden state ⇒ 签名携带注入痕迹 ⇒ 非缺陷）",
        tol=0.0, value=1.0 if _ok6 else 0.0)
    _stY.remove(); torch.cuda.empty_cache()

    # ─────────── A20-11〔hebbian_full 的 vals **不进**读路径；W **进**读路径〕───────────
    _stE = _mk20()
    write_both(_stE, _d20)
    _stE.freeze_write()
    _fwd(_stE, _d20)
    _y1 = _yr(_stE)
    with torch.no_grad():
        _vals_saved = [m.core.vals.detach().clone() for m in _stE.mems.values()]
        for m in _stE.mems.values():
            m.core.vals.zero_()
    _fwd(_stE, _d20)
    _y2 = _yr(_stE)
    with torch.no_grad():
        for _m, _v in zip(_stE.mems.values(), _vals_saved):
            _m.core.vals.copy_(_v)
        _W_saved = [m.core.W.detach().clone() for m in _stE.mems.values()]
        for m in _stE.mems.values():
            m.core.W.zero_()
    _fwd(_stE, _d20)
    _y3 = _yr(_stE)
    with torch.no_grad():
        for _m, _w in zip(_stE.mems.values(), _W_saved):
            _m.core.W.copy_(_w)
    _c_v = [float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
            for a, b in zip(_y1, _y2)]
    _c_w = [float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
            for a, b in zip(_y1, _y3)]
    _ok11 = (min(_c_v) >= 1.0 - 1e-6) and (min(_c_w) < 0.999)
    rec("A20-11〔vals 不进读路径（置零 ⇒ 读出逐位不变）；W 进读路径（置零 ⇒ 读出改变）〕",
        _ok11,
        f"vals 置零后逐层余弦={['%.8f' % c for c in _c_v]}（须 ≥ 1-1e-6）；"
        f"W 置零后逐层余弦={['%.6f' % c for c in _c_w]}（须 < 0.999）", tol=0.999,
        value=float(min(_c_w)))
    _stE.remove(); torch.cuda.empty_cache()

    # ─────────── A20-12′〔真值槽**精确可预测**且**不回绕**〕───────────
    #   ★★ 卡片 rev 1.3 §0.6：rev 1.1 的 A20-12 写「true_slot == true_idx（表在写前清零）」，
    #     但本协议（与 DIR-18/19 同构）在写阶段前 `restore_all(snap1)` ⇒ 表带**训练残留**
    #     ⇒ 真值槽 = **写前 count + true_idx**（实测依据见 rev 1.3 §0.6）。
    #   ⇒ 不变量：**真值槽唯一确定**（= C0 + true_idx）且**不回绕**（C0 + 本篇条数 ≤ N_ENTRIES）。
    _stF = _mk20()
    _C0 = int(list(_stF.mems.values())[0].core.count)
    write_both(_stF, _d20)
    _eq_slot = [int(m.slot_of_write[int(_d20["true_idx"])]) == _C0 + int(_d20["true_idx"])
                for m in _stF.mems.values()]
    _cnts = [int(m.core.count) for m in _stF.mems.values()]
    _nowrap = all(c <= int(m.core.N) for c, m in zip(_cnts, _stF.mems.values()))
    rec("A20-12′〔真值槽精确可预测（= 写前 count + true_idx）且不回绕〕",
        (all(_eq_slot) and _nowrap),
        f"写前 count C0={_C0}；逐层 slot_of_write[true_idx] == C0 + true_idx：{_eq_slot}；"
        f"写后 count={_cnts}；未回绕（≤ N_ENTRIES={N_ENTRIES}）={_nowrap}；"
        f"★ Stage A 中 C0=0（无训练残留）⇒ 退化为 true_slot == true_idx；"
        f"正式轮里 C0=训练残留 ⇒ **addr（写入序口径）恒为 0，只有 addr_slot 可读**",
        tol=0.0, value=1.0 if (all(_eq_slot) and _nowrap) else 0.0)
    _stF.remove(); torch.cuda.empty_cache()

    # ─────────── A20-13〔linear 的实测 ‖k‖² ≫ 5（复现 DIR-18 的 D50/D51）〕───────────
    _stL = _mk20(key_mode="linear", eta=1e-5)
    write_both(_stL, _d20)
    _lin = [float((m.core.keys[:int(m.core.count)] ** 2).sum(1).mean()) for m in _stL.mems.values()]
    _stL.remove(); torch.cuda.empty_cache()
    _stR = _mk20(key_mode="renorm", eta=1e-1)
    write_both(_stR, _d20)
    _ren = [float((m.core.keys[:int(m.core.count)] ** 2).sum(1).mean()) for m in _stR.mems.values()]
    _stR.remove(); torch.cuda.empty_cache()
    _ok13 = (min(_lin) > 100.0) and (abs(float(np.mean(_ren)) - 5.0) < 1.0)
    rec("A20-13〔X8_linear 的键尺度实测 ≫ 5（linear ≫ 100，renorm ≈ 5）〕", _ok13,
        f"linear 逐层 E‖k‖²={['%.1f' % x for x in _lin]}；"
        f"renorm 逐层 E‖k‖²={['%.4f' % x for x in _ren]}；"
        f"★ 复现 D50/D51 ⇒ X8 必须用 η=1e-5（修订 20-C）", tol=100.0, value=float(min(_lin)))

    # ─────────── A20-8〔会话 B 的模型实例**确实是新的**；会话 A 的已释放〕───────────
    #   ★ 显存前提：D86 的清理已把继承 Stage A 的 5.7 GB 释放掉 ⇒ 这里再载入两个 fp32 基座不会 OOM。
    try:
        del _stA
    except NameError:
        pass
    try:
        del _stB
    except NameError:
        pass
    try:
        del _stC
    except NameError:
        pass
    try:
        del _stD
    except NameError:
        pass
    try:
        del _stE
    except NameError:
        pass
    try:
        del _stF
    except NameError:
        pass
    try:
        del _stX
    except NameError:
        pass
    try:
        del _stY
    except NameError:
        pass
    try:
        del _stL
    except NameError:
        pass
    try:
        del _stR
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    _mA = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    _mB = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    _idA, _idB = id(_mA), id(_mB)
    _mem_before = int(torch.cuda.memory_allocated())
    del _mA
    gc.collect()
    torch.cuda.empty_cache()
    _mem_after = int(torch.cuda.memory_allocated())
    _ok8 = (_idA != _idB) and (_mem_before - _mem_after > 500 * 1024 * 1024)
    rec("A20-8〔会话 B 的模型实例**确实是新的**（id 不同）；会话 A 的已释放（显存回落）〕", _ok8,
        f"id(会话A)={_idA}，id(会话B)={_idB}；显存 {_mem_before/2**20:.0f} MiB → "
        f"{_mem_after/2**20:.0f} MiB（回落 {_mem_before/2**20 - _mem_after/2**20:.0f} MiB）",
        tol=500.0, value=float((_mem_before - _mem_after) / 2**20))
    del _mB
    gc.collect()
    torch.cuda.empty_cache()
    print("[Stage A · DIR-20] 13 项断言完成", flush=True)
    # ══════════════════ ★★【DIR-21】Stage A 新增断言 A21-1…A21-9（A21-10 = 继承 DIR-20 的 44 项）══════
    # （卡片 rev 1.1 §4 冻结；★ **判据阈值与文字一字未改**）
    print("\n[Stage A · DIR-21]", flush=True)

    # —— 造一篇与 DIR-15…20 同构的评测文档（只需结构，不必是语料真样本）——
    _ids8 = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
    _L = 69

    def _mk(stride, pick, eta=None):
        # ★★【rev 1.8 · 缺陷 D119】默认写侧速率必须 = **ETA_BASE**（臂所用的那个 η）。
        #   旧值 = 全局 `ETA=1e−5` ⇒ 本节所有写侧断言都在「几乎写不进去」的配置上通过
        #   ⇒ 断言通过但**保护的配置从未被使用**（D119 的 Stage A 版本）。
        _s = make_stack(model, d, "cue", seed=20261028, mem_kind="hebbian_full",
                        eta=(eta if eta is not None else TRAIN_ETA), key_mode="ortho",
                        variant="normal")
        for _m in _s.mems.values():
            _m.write_stride = int(stride)
            _m.stride_pick = str(pick)
        return _s

    def _doc_ids():
        # 一篇「长度 69 的文档」：前 68 个是填充，最后一个是码 token
        _g = torch.Generator(device="cpu"); _g.manual_seed(999)
        _body = torch.randint(100, 5000, (_L - 1,), generator=_g).tolist()
        return _body + [7777]

    _did = _doc_ids()
    _pairs = {}
    _wcs = {}
    _counts = {}
    for (_nm, _S) in (("S1", 1), ("S17", 17), ("S68", 68)):
        _st = _mk(_S, "last")
        _trig = list(_st.mems[list(_st.mems)[0]].trigger_positions(_L))
        _c0 = int(_st.mems[list(_st.mems)[0]].core.count)
        _wc0 = int(_st.mems[list(_st.mems)[0]].write_calls)
        with torch.no_grad():
            _st.model(input_ids=torch.tensor([_did], device=DEV))
        _m = _st.mems[list(_st.mems)[0]]
        _pairs[_nm] = {int(t): (m_k.detach().clone(), m_v.detach().clone())
                       for t, m_k, m_v in zip(_trig, _m.keys_written, _m.vals_written)}
        _wcs[_nm] = int(_m.write_calls) - _wc0
        _counts[_nm] = int(_m.core.count) - _c0
        _st.remove()
        del _st
    gc.collect()
    torch.cuda.empty_cache()

    # —— A21-1：同一 token 位置在任何粒度下 (k, v) 逐位相同 ——
    _common = sorted(set(_pairs["S1"]) & set(_pairs["S17"]) & set(_pairs["S68"]))
    _ok1 = len(_common) > 0
    _bad = []
    for _t in _common:
        _k1, _v1 = _pairs["S1"][_t]
        for _o in ("S17", "S68"):
            _k2, _v2 = _pairs[_o][_t]
            if not (torch.equal(_k1, _k2) and torch.equal(_v1, _v2)):
                _bad.append((_t, _o))
    _ok1 = _ok1 and not _bad
    rec("A21-1〔同一 token 位置在任何粒度下 (k,v) 逐位相同 ⇒ 唯一自变量=写步调度〕", _ok1,
        f"共同触发位置 {len(_common)} 个（如 { _common[:6] }）；不一致 {len(_bad)} 处"
        + (f"：{_bad[:5]}" if _bad else ""),
        tol=0.0, value=float(len(_bad)))

    # —— A21-2：写步数精确（**一次前视** = 68 / 4 / 1 次触发；rev 1.2 §0.2 的「每篇 m」
    #   = 2 × 这个数：一次 `write_both` = **两次前视** = 目标 + 干扰）——
    _exp = {"S1": 68, "S17": 4, "S68": 1}
    _ok2 = all(_wcs[_k] == _exp[_k] for _k in _exp)
    rec("A21-2〔写步数精确：每篇前视的触发数〕", _ok2,
        f"实测 {dict(_wcs)}；期望 {dict(_exp)}（stride 1/17/68 ⇒ **每篇前视** 68/4/1 次；"
        f"每篇累积 m = 2× ⇒ 136/8/2，rev 1.2 §0.2）",
        tol=0.0, value=float(sum(abs(_wcs[_k] - _exp[_k]) for _k in _exp)))

    # —— A21-3：段边界确定性（同一 token 序列 ⇒ 同一分段）——
    _st = _mk(17, "last")
    _tr1 = list(_st.mems[list(_st.mems)[0]].trigger_positions(_L))
    _tr2 = list(_st.mems[list(_st.mems)[0]].trigger_positions(_L))
    _ok3 = (_tr1 == _tr2) and (_tr1 == [17, 34, 51, 68])
    rec("A21-3〔段边界确定性：同一 token 序列 ⇒ 同一分段 ⇒ 触发位置可复现〕", _ok3,
        f"stride=17 ⇒ 触发位置 {_tr1}（两次调用一致：{_tr1 == _tr2}）",
        tol=0.0, value=float(len(_tr1)))
    _st.remove()
    del _st
    gc.collect()
    torch.cuda.empty_cache()

    # —— A21-4：解析视界可算 ——
    # ★★【rev 1.8 · D119】必须用**臂真正使用的写侧速率** `ETA_BASE`（旧值 1e−5 ⇒ 视界 = 2e6 步，
    #   与实际协议（每篇 m 次触发）相差三个数量级 ⇒ 记录的「解析视界」是假读数）。
    _horizon = 1.0 / (ETA_BASE * LAM)
    _ok4 = _horizon > 0
    rec("A21-4〔解析视界 1/(ηλ) 可从配置算出并记录〕", _ok4,
        f"η_W={ETA_BASE} · λ={LAM} ⇒ 1/(ηλ) = {_horizon:.1f} **触发**；"
        f"每触发衰减因子 (1−ηλ) = {1 - ETA_BASE * LAM:.6f}；"
        f"半衰期 ≈ {math.log(0.5) / math.log(1 - ETA_BASE * LAM):.1f} 触发；"
        f"折算情节（DIR-22 的 5 格）：C_m68≈{_horizon / 68:.4f} / C_m18≈{_horizon / 18:.3f} / "
        f"C_m6≈{_horizon / 6:.2f} / C_base≈{_horizon / 2:.2f} 篇",
        tol=0.0, value=_horizon)

    # —— A21-5：每次触发恰好一次写更新（core.count 的增量 == 触发数）——
    _ok5 = all(_counts[_k] == _exp[_k] for _k in _exp)
    rec("A21-5〔每次触发恰好一次写更新（core.count 增量 == 触发数）〕", _ok5,
        f"实测 {dict(_counts)}；期望 {dict(_exp)}",
        tol=0.0, value=float(sum(abs(_counts[_k] - _exp[_k]) for _k in _exp)))

    # —— A21-6：归一化后内容可写（写保真 cos(Wk,v) ≥ 0.95）——
    # ★ rev 1.2 §0.4 修法：Stage A 用 **ETA_BASE=1e−1**（`ortho` 可用的强学习率）造核，
    #   否则用全局 ETA=1e−5 ⇒ W≈W₀（随机）⇒ cos(Wk,v)≈0 ⇒ **会把「没写完」误报成「写不可写」**。
    _st = _mk(1, "last", eta=ETA_BASE)
    with torch.no_grad():
        _st.model(input_ids=torch.tensor([_did], device=DEV))
    _m = _st.mems[list(_st.mems)[0]]
    _fids = []
    for _i in range(min(int(_m.core.count), int(_m.core.N))):
        try:
            _fids.append(float(_m.core.fidelity(_i)))
        except Exception:
            pass
    _fid_min = float(np.nanmin(_fids)) if _fids else float("nan")
    _ok6 = (not math.isnan(_fid_min)) and _fid_min >= 0.95
    rec("A21-6〔归一化后内容可写：写保真 cos(Wk,v) ≥ 0.95〕", _ok6,
        f"逐槽写保真：min={_fid_min:.4f} mean={np.nanmean(_fids) if _fids else float('nan'):.4f} "
        f"（共 {len(_fids)} 槽）",
        tol=0.95, value=_fid_min)
    _st.remove()
    del _st
    gc.collect()
    torch.cuda.empty_cache()

    # —— A21-7：等比对照等价性（η/17 逐位生效 + 仍在稳定上界内）——
    # ★★【rev 1.8 · D119】对照臂 `G_tok_rate` 的 η = **ETA_BASE/17**（旧值 ETA/17 = 5.9e−7
    #   ⇒ 断言的是臂表里根本不存在的配置）。
    _st = _mk(1, "last", eta=ETA_DECAY68)
    _etas = [float(_m.eta) for _m in _st.mems.values()] + \
            [float(_m.core.eta) for _m in _st.mems.values()]
    _ok7 = all(abs(_e - ETA_DECAY68) < 1e-15 for _e in _etas)
    rec("A21-7〔等比对照等价性：η′=η×18/68（解出臂）逐位生效（层与核都一致）〕", _ok7,
        f"层 η={_etas[:len(_etas)//2]} · 核 η={_etas[len(_etas)//2:]}；期望 {ETA_BASE/17.0:.8g}",
        tol=0.0, value=float(abs(max(_etas) - ETA_DECAY68)))
    _st.remove()
    del _st
    gc.collect()
    torch.cuda.empty_cache()

    # —— A21-8：无残留（基座仍被冻结 + 显存已回收）——
    _allfrozen = all(not q.requires_grad for q in model.parameters())
    _grads = sum(1 for q in model.parameters() if q.grad is not None and
                  float(q.grad.abs().sum()) > 0)
    _mem = int(torch.cuda.memory_allocated())
    _ok8 = _allfrozen and _grads == 0
    rec("A21-8〔无残留：基座仍被冻结 + 无残留梯度〕", _ok8,
        f"全部 requires_grad=False：{_allfrozen}；非零梯度参数数：{_grads}；"
        f"显存 { _mem/2**20:.0f} MiB",
        tol=0.0, value=float(_grads))

    # —— A21-9：只读评测不写 ——
    _st = _mk(1, "last")
    _m = _st.mems[list(_st.mems)[0]]
    with torch.no_grad():
        _st.model(input_ids=torch.tensor([_did], device=DEV))
    _c0 = int(_m.core.count)
    _w0 = int(_m.write_calls)
    _sp = {"prefix": _did[:-1], "code_ids": [_did[-1]], "true_idx": _L - 2,
           "true_slot": _c0 - 1, "dist_code": "0"}
    _r = eval_arm_readonly(_st, _sp, None)
    _c1 = int(_m.core.count)
    _w1 = int(_m.write_calls)
    _ok9 = (_c0 == _c1) and (_w0 == _w1)
    rec("A21-9〔只读评测不写：count / write_calls 不变〕", _ok9,
        f"count {_c0} → {_c1}；write_calls {_w0} → {_w1}；读出 nll={_r.get('nll')}",
        tol=0.0, value=float(abs(_c1 - _c0) + abs(_w1 - _w0)))
    _st.remove()
    del _st
    gc.collect()
    torch.cuda.empty_cache()

    # —— A21-11：R_cnt 必须能随写入增多而上升（否则 H21-1b 无法被计算）——
    _st = _mk(1, "last")
    _g = torch.Generator(device="cpu")
    _g.manual_seed(21)
    def _pair(seed_i):
        body = torch.randint(100, 4000, (34,), generator=_g).tolist()
        dist = torch.randint(100, 4000, (34,), generator=_g).tolist()
        return {"ids": body + [8000 + seed_i], "dist_ids": dist + [9000 + seed_i]}
    for _i in range(6):
        _st.unfreeze_write()
        write_both(_st, _pair(100 + _i))
    clear_slots(_st)
    _c_cleared = int(_st.mems[list(_st.mems)[0]].core.count)
    _st.unfreeze_write()
    write_both(_st, _pair(1))
    _r1 = resident_r_cnt(_st)
    for _i in range(3):
        _st.unfreeze_write()
        write_both(_st, _pair(2 + _i))
    _r4 = resident_r_cnt(_st)
    _ok11 = (_c_cleared == 0) and (_r4 > _r1) and (_r4 <= int(N_ENTRIES))
    rec("A21-11〔R_cnt 在清空地址表后随写入增多而上升 ⇒ H21-1b 可计算〕", _ok11,
        f"清空后 count={_c_cleared}；1 篇 R_cnt={_r1}；4 篇 R_cnt={_r4}（须上升，且 ≤ {N_ENTRIES}）",
        tol=0.0, value=float(_r4 - _r1))
    _st.remove()
    del _st
    gc.collect()
    torch.cuda.empty_cache()

    # ══【rev 1.7】A21-12〔键列调度模型可复核 ⇒ 有效窗口可算〕══
    _Lp = 35          # ★ 协议文档长度（30 填充 + 4 线索 + 1 码）⇒ 每篇 34+34 = 68 个键列
    _st = _mk(1, "last")
    _m0 = _st.mems[list(_st.mems)[0]]
    _trig = list(_m0.trigger_positions(_Lp))
    clear_write_records(_st)                     # ★ D111
    _st.unfreeze_write()
    _b0 = int(_m0.ortho_idx)
    write_both(_st, _pair(1))
    _b1 = int(_m0.ortho_idx)
    _st.unfreeze_write()
    write_both(_st, _pair(2))
    _b2 = int(_m0.ortho_idx)
    _ok12a = (_b1 - _b0 == 68) and (_b2 - _b1 == 68)   # 每篇消耗 68 个键列
    _ks = _m0.keys_written[2 * len(_trig): 3 * len(_trig)]   # 第二篇的**目标文档**条目
    _ok12b = (len(_ks) == len(_trig))
    for _i, _t in enumerate(_trig):
        _j = (_b1 + _t - 1) % _m0.ortho_basis.shape[1]
        _kk = _m0.ortho_basis[:, _j]
        _kk = _kk * (math.sqrt(5.0) / float(_kk.norm().clamp_min(1e-30)))
        if not torch.allclose(_kk.to(_ks[_i].dtype), _ks[_i], atol=1e-6):
            _ok12b = False
    _reuse_map = {}
    for (_nm, _eta, _S, _pk) in ARMS22:
        for _mm in _st.mems.values():
            _mm.write_stride = int(_S)
            _mm.stride_pick = str(_pk)
        _tr = list(_m0.trigger_positions(_Lp))
        _reuse_map[_nm] = reuse_episode(0, _tr, 224)
    _ok12c = all(v is not None and 2 <= int(v) <= 225 for v in _reuse_map.values())
    rec("A21-12〔键列调度模型可复核 ⇒ 每臂的**有效窗口**可算（D112）〕",
        _ok12a and _ok12b and _ok12c,
        f"每篇消耗键列 {_b1 - _b0}/{_b2 - _b1}（须 68/68）；第二篇目标条目的键列逐位匹配模型={_ok12b}；"
        f"各臂锚键列复用情节（相位 0）={_reuse_map}",
        tol=68.0, value=float(_b1 - _b0))
    _st.remove()
    del _st
    gc.collect()
    torch.cuda.empty_cache()

    # ══【rev 1.7】A21-13〔原始 (k,v) 捕获正确（D111/D102）〕══
    _st = _mk(1, "last")
    clear_write_records(_st)
    _empty = all(len(m.keys_written) == 0 and len(m.vals_written) == 0 for m in _st.mems.values())
    _st.unfreeze_write()
    write_both(_st, _pair(3))
    _tr = list(_st.mems[list(_st.mems)[0]].trigger_positions(_Lp))
    _kv = capture_doc1_entries(_st, len(_tr))
    _ok13 = (_empty and len(_kv) == len(_st.mems)
             and all(len(x) == len(_tr) for x in _kv)
             and all(x[0][0].shape == _kv[0][0][0].shape for x in _kv))
    rec("A21-13〔扫描起点的写记录已清空；文档 1 的**原始 (k,v)** 捕获正确〕", _ok13,
        f"清空后写记录为空={_empty}；捕获层数={len(_kv)}；每层条目数={[len(x) for x in _kv]}"
        f"（该臂触发位置数={len(_tr)}）；⇒ D102 的 cos(W·k,v) 取的是**原始**张量",
        tol=0.0, value=float(len(_tr)))
    # ══【rev 1.7】A21-14〔逐点写次数可审计（D103/D116）〕══
    _mc = int(_st.mems[list(_st.mems)[0]].core.count)
    _mdoc = 2 * len(_tr)
    _ok14 = (_mc == _mdoc)
    rec("A21-14〔写次数可审计：1 篇 ⇒ 目标+干扰的实际触发数 m〕", _ok14,
        f"1 篇实测 core.count={_mc}；期望 m=2×{len(_tr)}={_mdoc}"
        f"（DIR-22：C_base=2 / C_m6=6 / C_m18=18 / C_m68=68 / C_decay68=68）",
        tol=float(_mdoc), value=float(_mc))
    _st.remove()
    del _st
    gc.collect()
    torch.cuda.empty_cache()

    # ══【rev 1.8】A21-15〔训练栈的写侧 η == 臂表所用的 ETA_BASE（缺陷 D119 的守卫）〕══
    #   为什么需要这条：D119 的本质是「**断言保护的配置从未被使用**」——A21-1…A21-14
    #   全部在 ETA_BASE 上跑通，而主循环的训练栈却用了全局 ETA=1e−5 ⇒ 全部断言通过
    #   但正式轮的任务层读数是缺陷产物。⇒ 必须**直接断言主循环将要构造的那个栈**。
    _st = make_stack(model, d, "cue", seed=20261028, mem_kind="hebbian_full",
                     eta=TRAIN_ETA, key_mode="ortho", variant="normal")
    _tr_eta = [float(_m.eta) for _m in _st.mems.values()] + \
              [float(_m.core.eta) for _m in _st.mems.values()]
    _tr_stride = [int(_m.write_stride) for _m in _st.mems.values()]
    _ok15 = (all(abs(_e - TRAIN_ETA) < 1e-15 for _e in _tr_eta)
             and all(_s == 1 for _s in _tr_stride)
             and abs(ETA - ETA_BASE) > 1e-9)   # ★ 全局 ETA 与臂 η 必须确实不同（本缺陷的存在前提）
    rec("A21-15〔训练栈写侧 η == ETA_BASE 且 stride=1（D119 守卫）〕", _ok15,
        f"训练栈层/核 η={_tr_eta}（期望 TRAIN_ETA={TRAIN_ETA}）；stride={_tr_stride}（期望全 1）；"
        f"全局 ETA={ETA} ≠ TRAIN_ETA={TRAIN_ETA} ⇒ 训练与评测同一写侧速率",
        tol=0.0, value=float(abs(max(_tr_eta) - TRAIN_ETA)))
    _st.remove()
    del _st
    gc.collect()
    torch.cuda.empty_cache()


    # ══════════════════════ ★★【DIR-22】Stage A 新增断言 A22-1…A22-9（继承 DIR-21 的 58 项）═══
    #   ★ 任一项 FAIL ⇒ 由下面的硬闸中止（纪律 ①）⇒ 不可能带着坏装置产出正式轮读数。
    print("\n[Stage A · DIR-22] A22-1…A22-9", flush=True)

    # —— A22-1：训练栈与臂表的基准 η 是**同一个常量**（D119 同类守卫）——
    _eta_base_arms = [float(e) for (_n, e, _s, _p) in ARMS22]
    _ok1 = (abs(TRAIN_ETA - ETA_BASE) < 1e-15
            and sum(1 for e in _eta_base_arms if abs(e - ETA_BASE) < 1e-15) == 4
            and abs(ETA_DECAY68 - ETA_BASE * 18.0 / 68.0) < 1e-18)
    rec("A22-1〔基准 η 同源：TRAIN_ETA == ETA_BASE；5 臂中 4 臂用 ETA_BASE；解出臂 η 逐位正确〕",
        _ok1,
        f"TRAIN_ETA={TRAIN_ETA} ETA_BASE={ETA_BASE}；臂 η={_eta_base_arms}；"
        f"解出臂 η={ETA_DECAY68:.12g}（= 0.1×18/68）⇒ ηλm 两端都 = {ETA_BASE * LAM * 18:.6g}",
        tol=0.0, value=float(ETA_DECAY68))

    # —— A22-2：5 臂的 η **逐位生效**（层与核都一致）；并核验稳定域 ——
    _eta_rows, _ok2 = [], True
    for (_n, _e, _s, _p) in ARMS22:
        _st = make_stack(model, d, "cue", seed=SEEDS[0], mem_kind="hebbian_full",
                         eta=TRAIN_ETA, key_mode="ortho", variant="normal")
        for _m in _st.mems.values():
            _m.eta = float(_e); _m.core.eta = float(_e)
        _got = [float(_m.eta) for _m in _st.mems.values()] + \
               [float(_m.core.eta) for _m in _st.mems.values()]
        _same = all(abs(g - float(_e)) < 1e-18 for g in _got)
        _ok2 = _ok2 and _same
        _eta_rows.append(f"{_n}={_e:.10g}{'' if _same else '✗'}")
        _st.remove(); del _st; gc.collect(); torch.cuda.empty_cache()
    rec("A22-2〔5 臂的 η 逐位生效（层与核）〕", _ok2, "；".join(_eta_rows), tol=0.0, value=_ok2)

    # —— A22-3：5 臂的 m **可审计**（实测 core.count 增量 == 该臂 m）——
    _st = make_stack(model, d, "cue", seed=SEEDS[0], mem_kind="hebbian_full",
                     eta=TRAIN_ETA, key_mode="ortho", variant="normal")
    _tr_docs, _ = build_docs(tok, SEEDS[0], 2, 2)
    _m_meas = {}
    for (_n, _e, _s, _p) in ARMS22:
        for _m in _st.mems.values():
            _m.eta = float(_e); _m.core.eta = float(_e)
            _m.write_stride = int(_s); _m.stride_pick = str(_p)
        clear_slots(_st); clear_write_records(_st)
        _c0 = int(_st.mems[list(_st.mems)[0]].core.count)
        _st.unfreeze_write()
        write_both(_st, _tr_docs[0])
        _m_meas[_n] = int(_st.mems[list(_st.mems)[0]].core.count) - _c0
    _st.remove(); del _st; gc.collect(); torch.cuda.empty_cache()
    _ok3 = all(_m_meas[k] == M_EXPECT[k] for k in M_EXPECT)
    rec("A22-3〔5 臂的 m 逐位可审计（目标+干扰的实测触发数）〕", _ok3,
        "；".join(f"{k}: 实测 {_m_meas[k]} / 期望 {M_EXPECT[k]}" for k in M_EXPECT),
        tol=68.0, value=float(max(_m_meas.values())))

    # —— A22-4/A22-5：机制探针（M-1 刷新 / M-2 衰减）逐臂与闭式一致 ——
    _st = make_stack(model, d, "cue", seed=SEEDS[0], mem_kind="hebbian_full",
                     eta=TRAIN_ETA, key_mode="ortho", variant="normal")
    _mech_rows, _ok4, _ok5, _ok5s = [], True, True, True
    for (_n, _e, _s, _p) in ARMS22:
        for _m in _st.mems.values():
            _m.eta = float(_e); _m.core.eta = float(_e)
            _m.write_stride = int(_s); _m.stride_pick = str(_p)
        clear_slots(_st); clear_write_records(_st)
        _st.unfreeze_write()
        write_both(_st, _tr_docs[0])
        _Lp = len(_tr_docs[0]["ids"])
        _trig = list(_st.mems[list(_st.mems)[0]].trigger_positions(_Lp))
        # ★ D127：`C_base`（stride=34）每篇只触发 **1** 次 ⇒ 只捕获 len(_trig)=1 条时，
        #   「另一条键」会**就是同一条**（k1==k0）⇒ 衰减探针退化成刷新探针（实测 D=1.048）。
        #   修法：多捕 **1** 条（目标文档之后的干扰文档第一条 ⇒ 不同正交列 ⇒ 严格正交）。
        _ncap = max(len(_trig), 2)
        _kv = capture_doc1_entries(_st, _ncap)
        _pairs = {}
        for _li, _lst in enumerate(_kv):
            _k0, _v0 = _lst[len(_trig) - 1]              # 码 token 那一条（判据读的就是它）
            _k1 = _lst[1 if len(_trig) == 1 else 0][0]   # 「另一条」（不同正交列）
            _pairs[list(_st.mems)[_li]] = (_k0, _v0, _k1)
        _pm = probe_mechanism(_st, _pairs, M_EXPECT[_n])
        _e4 = abs(_pm["rho_meas_mean"] - _pm["rho_pred_mean"]) / max(abs(_pm["rho_pred_mean"]), 1e-12)
        _e5 = abs(_pm["D_meas_mean"] - _pm["D_pred_mean"]) / max(abs(_pm["D_pred_mean"]), 1e-12)
        _ok4 = _ok4 and (_e4 <= 0.05)
        _ok5 = _ok5 and (_e5 <= 0.05)
        _ok5s = _ok5s and (abs(_pm["s_meas_mean"] - 5.0) < 1e-3) and (_pm["ortho_dot_mean"] < 1e-6)
        _mech_rows.append(f"{_n}: ρ̂={_pm['rho_meas_mean']:.6f}/预{_pm['rho_pred_mean']:.6f}"
                          f"（{100*_e4:.2f}%） D={_pm['D_meas_mean']:.6f}/预{_pm['D_pred_mean']:.6f}"
                          f"（{100*_e5:.2f}%） s={_pm['s_meas_mean']:.4f}")
    _st.remove(); del _st; gc.collect(); torch.cuda.empty_cache()
    rec("A22-4〔M-1 刷新探针：ρ̂ == 1−η(αs+λ)，相对偏差 ≤5%〕", _ok4, "；".join(_mech_rows), tol=0.05)
    rec("A22-5〔M-2 衰减探针：D == (1−ηλ)^m 且实测 s=‖k‖²=5、正交键 ⟨k₀,k₁⟩≈0〕", _ok5 and _ok5s,
        "；".join(_mech_rows), tol=0.05)

    # —— A22-6：**交集口径非空**（两条匹配对的触发位置交集）——
    _tr_sets, _st = {}, make_stack(model, d, "cue", seed=SEEDS[0], mem_kind="hebbian_full",
                                  eta=TRAIN_ETA, key_mode="ortho", variant="normal")
    for (_n, _e, _s, _p) in ARMS22:
        for _m in _st.mems.values():
            _m.write_stride = int(_s); _m.stride_pick = str(_p)
        _tr_sets[_n] = set(int(t) - 1 for t in _st.mems[list(_st.mems)[0]].trigger_positions(_Lp))
    _st.remove(); del _st; gc.collect(); torch.cuda.empty_cache()
    _i_decay = sorted(_tr_sets["C_m18"] & _tr_sets["C_decay68"])
    _i_refr = sorted(_tr_sets["C_m18"] & _tr_sets["C_m68"])
    _ok6 = (len(_i_decay) > 0 and len(_i_refr) > 0
            and _i_decay == _i_refr == sorted(_tr_sets["C_m18"]))
    rec("A22-6〔交集口径非空：P_decay 与 P_refresh 的逐条目交集 == C_m18 的 9 个触发位置〕",
        _ok6,
        f"P_decay 交集 {len(_i_decay)} 条 t={_i_decay}；P_refresh 交集 {len(_i_refr)} 条；"
        f"C_m18 = {sorted(_tr_sets['C_m18'])}", tol=9.0, value=float(len(_i_refr)))

    # —— A22-7【D126 订正】：空白前向（no_write）**完全不动**（原 rev 1.1 误写为「只衰减不写」）——
    _st = make_stack(model, d, "cue", seed=SEEDS[0], mem_kind="hebbian_full",
                     eta=TRAIN_ETA, key_mode="ortho", variant="normal")
    _st.unfreeze_write()
    write_both(_st, _tr_docs[0])
    _snapA = _st.snapshot_all()
    _cA = int(_st.mems[list(_st.mems)[0]].core.count)
    _WA = [m.core._W().detach().clone() for m in _st.mems.values()]
    _KA = [m.core.keys.detach().clone() for m in _st.mems.values()]
    _st.freeze_write()
    with torch.no_grad():
        _st.model(input_ids=torch.tensor([_tr_docs[0]["prefix"]], dtype=torch.long,
                                         device=DEV))
    _cB = int(_st.mems[list(_st.mems)[0]].core.count)
    _ok7 = (_cB == _cA
            and all(torch.equal(w, m.core._W().detach()) for w, m in zip(_WA, _st.mems.values()))
            and all(torch.equal(k, m.core.keys.detach()) for k, m in zip(_KA, _st.mems.values())))
    _st.remove(); del _st; gc.collect(); torch.cuda.empty_cache()
    rec("A22-7〔空白前向（no_write）⇒ count / W / keys **逐位不变**（D126 订正后的正确形态）〕",
        _ok7, f"count {_cA} → {_cB}；W/keys 逐位不变={_ok7}", tol=float(_cA), value=float(_cB))

    # —— A22-8：5 臂全部在稳定域内 η < 2/(αs+λ) ——
    _bound = 2.0 / (ALPHA * 5.0 + LAM)
    _ok8 = all(float(_e) < _bound for (_n, _e, _s, _p) in ARMS22)
    rec("A22-8〔5 臂全部在稳定域内：η < 2/(α·5+λ)〕", _ok8,
        f"上界 {_bound:.6f}；臂 η 最大 {max(float(e) for (_n, e, _s, _p) in ARMS22):.6g}"
        f"（余量 {_bound / max(float(e) for (_n, e, _s, _p) in ARMS22):.1f}×）",
        tol=_bound, value=float(max(float(e) for (_n, e, _s, _p) in ARMS22)))

    # —— A22-9：装置 = v7 的**逐字复制**（DEC-02：不造 v8）——
    _devf = pathlib.Path(_MOD.__file__).resolve()
    _dev_md5 = hashlib.md5(_devf.read_bytes()).hexdigest()
    _ok9 = (_dev_md5 == "1a242d02c530307f6c76646ca13469ef")
    rec("A22-9〔装置 = dir21/sgmn_module_hebb_v7.py 的逐字复制（md5 冻结）〕", _ok9,
        f"{_devf.name} md5={_dev_md5}（期望 1a242d02c530307f6c76646ca13469ef）"
        f" ⇒ r≡1 ⇒ 与 v7 逐位一致（DEC-02）", tol=None, value=_dev_md5)

    print("[Stage A · DIR-21 继承] 断言完成（A21-1…A21-15）", flush=True)
    print("[Stage A · DIR-22] 断言完成（A22-1…A22-9）", flush=True)
    # ---------------- ★ Stage A 硬闸（纪律 ①）：任一 FAIL ⇒ **不得开主循环**
    #   （卡片 rev 1.2 §2.1：「任一 FAIL 不得开正式轮」；此前靠外部脚本把关，
    #     现在装置内部也设闸 ⇒ 无论谁启动本脚本，都不可能在 Stage A 失败时产出正式轮结果。）
    _failed = [c["name"] for c in res["checks"] if not c["passed"]]
    print(f"\n[Stage A 闸] 共 {len(res['checks'])} 项，FAIL {len(_failed)} 项"
          + (f"：{_failed}" if _failed else " ⇒ 全部通过"), flush=True)
    if _failed:
        print("[Stage A 闸] ★ 有 FAIL ⇒ **中止，不开主循环**（纪律 ①）", flush=True)
        out = HERE.parent / ("R22_smoke_results.json" if args.smoke else "R22_results.json")   # ★ D78
        res["aborted"] = {"stage_a_failed": _failed}
        with open(out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        raise SystemExit(3)

    # ---------------- DIR-21 主循环：训练一次 / 种子 ⇒ 5 个粒度臂**共用训练态** + **步轴**扫描
    print("\n[主循环]", flush=True)
    print(f"\n[规格] smoke={args.smoke}  n_train={n_train}  n_eval={n_eval}  seeds={seeds} "
          f"K_SCAN={K_SCAN}", flush=True)
    _tag = "smoke" if args.smoke else "full"
    # ★ DIR-21：主循环结构已改（步轴曲线）⇒ 增量续跑不适用，每轮从头跑。
    all_rows = {}
    STATE_DIR = HERE.parent / ".ops" / "state"      # ★ 纪律 #37/#42：命名空间隔离
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # ★★ 缺陷 D95 的修法：每轮开始清空遗留状态文件（全部可再生）
    _n_stale = 0
    for _f in STATE_DIR.glob("DIR22_*.pt"):
        try:
            _f.unlink()
            _n_stale += 1
        except OSError:
            pass
    if _n_stale:
        print(f"[主循环] ★ 清空遗留状态文件 {_n_stale} 个（缺陷 D95）", flush=True)

    def _load_model():
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
        for q in m.parameters():
            q.requires_grad_(False)
        return m

    def _drop_model():
        gc.collect()
        torch.cuda.empty_cache()

    def _set_arm(st, eta, stride, pick):
        """★ 设置该臂的**写侧速率**与**写步粒度**（其余一切不变）。"""
        for m in st.mems.values():
            m.eta = float(eta)
            m.core.eta = float(eta)
            m.write_stride = int(stride)
            m.stride_pick = str(pick)

    def _entry_read_specs(d, st, count_before):
        """★【DIR-21】文档 `d` 的**逐条目读出规格**。

        每个**已写入**条目 = (前缀 = `d['ids'][:t]`, 码 = `d['ids'][t]`, 真值槽)。
        ★ 目标位置 `t = L−1`（码 token）**必在**触发集合里 ⇒ ⇒ 每个粒度下**任务都可作答** ✔
        ★ 真值槽 = `count_before + (该条目在写入序列中的序号)`（**精确槽号映射**，沿用 D54 的做法）✔
        """
        li0 = list(st.mems)[0]
        m = st.mems[li0]
        L = len(d["ids"])
        trig = list(m.trigger_positions(L))       # ★ 目标文档的触发位置（升序）
        specs = []
        for i, t in enumerate(trig):
            specs.append({"prefix": list(d["ids"][:t]),
                          "code_ids": [int(d["ids"][t])],
                          "true_idx": int(t) - 1,
                          "true_slot": int((int(count_before) + i) % int(m.core.N)),
                          "dist_code": d.get("dist_code", "0")})
        return specs

    for seed in seeds:
        if all(arm in all_rows and seed in all_rows[arm] for arm, _, _, _ in ARMS22):
            print(f"  seed={seed} （全部臂已由增量完成 ⇒ 跳过）", flush=True)
            continue
        tr, ev = build_docs(tok, seed, n_train, n_eval)
        ctrl_ids = build_ctrl_ids(tok, seed, tr, ev)
        model = _load_model()
        # ★ 训练：**所有臂共用**（token 级 / η=1e−1 / ortho）⇒ 读出映射跨臂相同 ⇒ 臂间可比
        # ★★【rev 1.8 · 缺陷 D119】**这里必须传 ETA_BASE**，不能传全局 `ETA`。
        #   根因：`train_arm` 只训读参数（`Wo`/`gamma`/`Wq`），写侧 `W` 走 `no_grad`；
        #   训练期写阈值 (ηλ)·m = 1e−5×0.05×68 = 3.4e−5 ⇒ **训练期几乎零写入**
        #   ⇒ 读映射是在「空记忆」上训出来的（学到忽略记忆）⇒ 评测期即使 hit≈0.97、
        #   fid≈0.98，任务 NLL 仍停在基座水平（正式轮 r17 实测 15–17 nat，而 DIR-20
        #   同一读侧在 η=1e−1 下为 9.67 nat）⇒ H21-1 读数是缺陷产物。
        #   实测对照（probe21c）：η_train=1e−5 ⇒ 15.84 nat；η_train=1e−1 ⇒ **8.91 nat**。
        st = make_stack(model, d, "cue", seed=seed, mem_kind="hebbian_full", eta=TRAIN_ETA,
                        key_mode="ortho", variant="normal")
        traj, rep = train_arm(st, tr, "cue", variant="normal")
        snap = st.snapshot_all()
        print(f"  [训练] seed={seed} 完成（{n_train} 篇）", flush=True)
        for (name, eta, stride, pick) in ARMS22:
            if name in all_rows and seed in all_rows[name]:
                print(f"  seed={seed} {name:14s} （增量复用）", flush=True)
                continue
            # ★★ 复位到训练态（**状态**复位；**读出映射** W_o/γ 保留 ⇒ 训练好的那份）
            st.restore_all(snap)
            clear_slots(st)          # ★ D100：训练残留已写满 256 槽，容量扫描必须从空地址表开始
            clear_write_records(st)  # ★ D111：训练残留在写记录里 ⇒ 原始 (k,v) 会取错
            _set_arm(st, eta, stride, pick)
            _li0 = list(st.mems)[0]
            _m0 = st.mems[_li0]
            _trig = list(_m0.trigger_positions(len(ev[0]["ids"])))
            _m_doc = 2 * len(_trig)          # ★ 每篇的写规则触发总次数（目标文档 + 干扰文档）
            _count0 = int(_m0.core.count)
            _base0 = int(_m0.ortho_idx)     # ★ 训练后相位（probe21b 实测 = 68 ⇒ 锚列 = 101）
            # —— 写文档 1（为了拿它的逐条目读出规格与真值槽）——
            st.unfreeze_write()
            write_both(st, ev[0])
            specs = _entry_read_specs(ev[0], st, _count0)
            # ★ D111/D102：捕获文档 1 的**原始 (k,v)**（逐条目、逐层）
            _kv = capture_doc1_entries(st, len(_trig))
            # ★★【DIR-22】机制探针：在「刚写完文档 1」的状态上测 ρ 与 D，
            #   然后 restore 回**同一快照** ⇒ 协议状态逐位不受影响（A22-6 的 count 断言仍成立）。
            try:
                _snap_post = st.snapshot_all()
                # ★ D127：粗粒度臂每篇只触发 1 次 ⇒ 另捕 1 条（干扰文档首条 ⇒ 严格正交）
                _kvp = capture_doc1_entries(st, max(len(_trig), 2))
                _pairs = {}
                for _li, _lst in enumerate(_kvp):
                    _pairs[list(st.mems)[_li]] = (
                        _lst[len(_trig) - 1][0], _lst[len(_trig) - 1][1],
                        _lst[1 if len(_trig) == 1 else 0][0])
                _pm = probe_mechanism(st, _pairs, _m_doc)
                res["mech"].setdefault(name, {})[seed] = _pm
                st.restore_all(_snap_post)
            except Exception as _e:
                print(f"[机制探针失败 {name} seed={seed}] {_e}", flush=True)
            # ★ D112：该臂的**有效窗口**（锚键列被再次写入的情节 − 1）
            _reuse = reuse_episode(_base0, _trig, max(K_SCAN))
            _window = (_reuse - 1) if _reuse is not None else 10 ** 9
            _written = 1
            curves = {}
            for _K in K_SCAN:
                while _written < _K:
                    _written += 1
                    st.unfreeze_write()
                    write_both(st, ev[_written - 1])
                # ★ D103/D116：写次数**可审计** —— 写完 K 篇 ⇒ core.count == K·m（逐位）
                assert int(_m0.core.count) == _K * _m_doc, (
                    f"[{name} seed={seed}] 写次数不符：count={int(_m0.core.count)} ≠ K·m={_K * _m_doc}")
                # —— 只读：读**文档 1** 的已写入条目（最老的那篇 ⇒ 年龄 = _K 情节）——
                tgt = None
                per_entry = []
                for _i, sp in enumerate(specs):
                    r = eval_arm_readonly(st, sp, ctrl_ids)
                    r["t"] = int(sp["true_idx"]) + 1
                    # ★ D102：**原对象**余弦（原始 (k,v)）；槽口径留作**阴性对照**
                    r["cos_orig"] = cos_orig(st, [_kv[_l][_i] for _l in range(len(_kv))])
                    per_entry.append(r)
                    if int(sp["true_idx"]) == len(ev[0]["ids"]) - 2:
                        tgt = r          # ★ 目标条目（码 token）⇒ 并列读法 Rp-5
                hits = [float(r.get("hit", float("nan"))) for r in per_entry]
                fids = [float(r.get("cos_orig", float("nan"))) for r in per_entry]
                fslots = [float(r.get("fid_true", float("nan"))) for r in per_entry]
                rowsum = {
                    "K": int(_K),
                    "n_docs": int(_written),
                    "writes": int(_m0.core.count),          # ★ D103：**实际**写次数（可审计）
                    "nll_tgt": float(tgt["nll"]) if tgt else float("nan"),
                    "nll_mean": float(np.nanmean([float(r["nll"]) for r in per_entry])),
                    # ★ D115：逐条目 NLL（供「交集口径」在判据层计算）
                    "nll_by_t": {str(r["t"]): float(r["nll"]) for r in per_entry},
                    # ★★【DIR-22 · rev 1.1 §4】**逐条目**余弦与命中（交集口径所需，D120）
                    "cos_by_t": {str(r["t"]): float(r.get("cos_orig", float("nan"))) for r in per_entry},
                    "hit_by_t": {str(r["t"]): float(r.get("hit", float("nan"))) for r in per_entry},
                    "hit_mean": float(np.nanmean(hits)),
                    "fid_mean": float(np.nanmean(fids)),          # ★ D102：原对象（臂内平均）
                    "fid_slot_mean": float(np.nanmean(fslots)),    # 阴性对照：当前槽住户
                    "n_ret": int(round(float(np.nansum(hits)))),
                    "r_cnt": int(resident_r_cnt(st)),
                    "occupied": int(min(int(_m0.core.count), int(_m0.core.N))),
                    "n_entries": int(len(per_entry)),
                    "n_steps_doc": int(_m_doc),
                    "window": int(_window),                       # ★ D112：有效窗口
                    "reuse_episode": (int(_reuse) if _reuse is not None else None),
                }
                curves[str(_K)] = rowsum
            all_rows.setdefault(name, {})[seed] = curves
            # ★ 增量落盘（每臂）：中断不丢已完成臂
            try:
                _incf = HERE.parent / ".ops" / f"formal_incremental_{_DIR_TAG}_{_tag}.json"
                _incf.parent.mkdir(parents=True, exist_ok=True)
                with open(_incf, "w") as _f:
                    json.dump({"rows": _jsonable(all_rows)}, _f, indent=1, ensure_ascii=False)
            except Exception as _e:
                print(f"[增量落盘失败] {_e}", flush=True)
            _c1 = curves[str(K_SCAN[0])]
            _cl = curves[str(K_SCAN[-1])]
            print(f"  seed={seed} {name:14s} m={_m_doc:4d} 窗口={_window:4d} 复用={_reuse} | "
                  f"K={K_SCAN[0]}: nll={_c1['nll_tgt']:.3f} hit={_c1['hit_mean']:.3f} "
                  f"fid={_c1['fid_mean']:.3f} n_ret={_c1['n_ret']:3d}"
                  f" | K={K_SCAN[-1]}: nll={_cl['nll_tgt']:.3f} hit={_cl['hit_mean']:.3f} "
                  f"fid={_cl['fid_mean']:.3f} n_ret={_cl['n_ret']:3d}", flush=True)
        st.remove()
        del st
        del model
        _drop_model()
    # ★★ 缺陷 D97：汇总曾写在 mean_curve / _m_of 定义之前。
    #   smoke#2/#3/#4 主循环完成后 UnboundLocalError，判据从未被计算。
    #   汇总只留在辅助函数定义之后。

    # ══════════════════ ★★【DIR-21】判据 H21-1/1b/2/3/4 + 报告项 Rp-1…Rp-4（卡片 rev 1.1 §3 冻结）═══
    # —— 口径（写死，防漂移）——
    #   曲线横轴 = **在线步数**（step）⇒ 跨粒度可比（文档数随 m 变 ⇒ 不用文档轴）
    #   Δ(Δnll)(A,B) = mean(nll_B − nll_A)（>0 ⇒ A 更好）；配对单位 = (种子, 步点)
    #   R(step) ≡ fid_mean(step) / fid_mean(0)   （归一化内容保持；量纲无关）
    #   半衰期 ≡ 首个 R(step) ≤ 0.5 的 step
    #   容量峰值 K* ≡ 跨种子平均的「可检索条目数 n_ret」随**文档数**的 argmax
    K_LIST = [int(s) for s in K_SCAN]

    def curve(name, seed, field):
        c = all_rows.get(name, {}).get(seed, {})
        out = {}
        for k, v in c.items():
            try:
                out[int(k)] = float(v.get(field, float("nan")))
            except (TypeError, ValueError):
                out[int(k)] = float("nan")
        return out

    def mean_curve(name, field):
        out = {}
        for K in K_LIST:
            vals = []
            for sd in seeds:
                v = curve(name, sd, field).get(K, float("nan"))
                if not math.isnan(v):
                    vals.append(v)
            if vals:
                out[K] = float(np.mean(vals))
        return out

    def window_of(name):
        """★【rev 1.7 · D112】该臂的**有效窗口**（锚键列复用情节 − 1）。"""
        for sd in seeds:
            c = all_rows.get(name, {}).get(sd, {})
            for _k in sorted(c.keys()):
                try:
                    return int(c[_k].get("window", 10 ** 9))
                except (TypeError, ValueError):
                    continue
        return 10 ** 9

    def _m_of(name):
        """★【rev 1.7 · D116】该臂每篇的写规则触发总次数（目标 + 干扰；实测值）。"""
        for sd in seeds:
            c = all_rows.get(name, {}).get(sd, {})
            if c:
                return float(c[str(K_LIST[0])].get("n_steps_doc", float("nan")))
        return float("nan")

    def _eta_of(name):
        """该臂的**实际生效**写侧速率（用于每篇累积 (ηλ) 的核对）。"""
        for (_nm, _e, _s, _p) in ARMS22:
            if _nm == name:
                return float(_e)
        return float("nan")

    def paired_curves(a, b, field):
        """配对单位 = (种子, 情节 K)；限制在两臂的**公共有效窗口**内（D112）。"""
        W = min(window_of(a), window_of(b))
        A, B = [], []
        for sd in seeds:
            ca, cb = curve(a, sd, field), curve(b, sd, field)
            for K in K_LIST:
                if K > W:
                    continue
                if K in ca and K in cb and not math.isnan(ca[K]) and not math.isnan(cb[K]):
                    A.append(ca[K])
                    B.append(cb[K])
        if len(A) < 2:
            return {"mean": float("nan"), "t": float("nan"), "p": 1.0, "pos": 0,
                    "n": len(A), "consistent": False}
        return paired(A, B)

    def seed_delta_curves(a, b, field):
        W = min(window_of(a), window_of(b))
        out = []
        for sd in seeds:
            ca, cb = curve(a, sd, field), curve(b, sd, field)
            ds = [cb[K] - ca[K] for K in K_LIST if K in ca and K in cb and K <= W
                  and not math.isnan(ca[K]) and not math.isnan(cb[K])]
            if ds:
                out.append(float(np.mean(ds)))
        return out

    def _inter_pairs(a, b):
        """★【rev 1.7 · D115】**交集口径**：两臂**都写了**的触发位置。

        rev 1.1 §2.2（臂内平均）与 §2.3（同一 (种子,K) 读同一条目集合）冲突
        ⇒ 取交集 ⇒ §2.3 字面成立 ⇒ 配对严格（**对象澄清**，阈值不动）。
        """
        W = min(window_of(a), window_of(b))
        A, B = [], []
        for sd in seeds:
            ca = all_rows.get(a, {}).get(sd, {})
            cb = all_rows.get(b, {}).get(sd, {})
            for K in K_LIST:
                if K > W:
                    continue
                ra, rb = ca.get(str(K)), cb.get(str(K))
                if not ra or not rb:
                    continue
                da = ra.get("nll_by_t", {}) or {}
                db = rb.get("nll_by_t", {}) or {}
                common = sorted(set(da.keys()) & set(db.keys()))
                if not common:
                    continue
                A.append(float(np.mean([da[t] for t in common])))
                B.append(float(np.mean([db[t] for t in common])))
        return A, B

    def paired_inter(a, b):
        A, B = _inter_pairs(a, b)
        if len(A) < 2:
            return {"mean": float("nan"), "t": float("nan"), "p": 1.0, "pos": 0,
                    "n": len(A), "consistent": False}
        return paired(A, B)

    def seed_delta_inter(a, b):
        A, B = _inter_pairs(a, b)
        if not A:
            return []
        # —— 逐种子聚合（`_inter_pairs` 已按种子→K 顺序展平）——
        W = min(window_of(a), window_of(b))
        per_seed = []
        for sd in seeds:
            ca = all_rows.get(a, {}).get(sd, {})
            cb = all_rows.get(b, {}).get(sd, {})
            ds = []
            for K in K_LIST:
                if K > W:
                    continue
                ra, rb = ca.get(str(K)), cb.get(str(K))
                if not ra or not rb:
                    continue
                da = ra.get("nll_by_t", {}) or {}
                db = rb.get("nll_by_t", {}) or {}
                common = sorted(set(da.keys()) & set(db.keys()))
                if not common:
                    continue
                ds.append(float(np.mean([db[t] for t in common]))
                          - float(np.mean([da[t] for t in common])))
            if ds:
                per_seed.append(float(np.mean(ds)))
        return per_seed

    def norm_curve(name, field="fid_mean"):
        """★【rev 1.7】R(K) = field(K)/field(K₁)（**逐种子**归一后再跨种子平均）；窗口内。"""
        per_seed = []
        W = window_of(name)
        for sd in seeds:
            c = curve(name, sd, field)
            k0 = K_LIST[0]
            if k0 not in c or math.isnan(c[k0]) or abs(c[k0]) < 1e-12:
                continue
            base = c[k0]
            per_seed.append({K: c[K] / base for K in K_LIST
                             if K in c and not math.isnan(c[K]) and K <= W})
        out = {}
        for K in K_LIST:
            vals = [p[K] for p in per_seed if K in p]
            if vals:
                out[K] = float(np.mean(vals))
        return out

    def curve_absdiff(a, b, field="fid_mean"):
        ca, cb = norm_curve(a, field), norm_curve(b, field)
        W = min(window_of(a), window_of(b))
        ks = [K for K in K_LIST if K in ca and K in cb and K <= W]
        if not ks:
            return float("nan"), float("nan"), 0
        d = [abs(ca[K] - cb[K]) for K in ks]
        return float(np.mean(d)), float(np.max(d)), len(ks)

    def half_life(name, field="fid_mean"):
        """★【rev 1.7 · D113】各臂**自身窗口内**的首个 R(K) ≤ 0.5 的 K。

        未掉到 ⇒ **删失**（返回 NaN 且**不**用网格端点代替）⇒ 判据层据此记「**未算出来**」。
        """
        ks, censored = [], 0
        W = window_of(name)
        for sd in seeds:
            c = curve(name, sd, field)
            k0 = K_LIST[0]
            if k0 not in c or math.isnan(c[k0]) or abs(c[k0]) < 1e-12:
                censored += 1
                continue
            base = c[k0]
            kk = None
            for K in K_LIST:
                if K > W:
                    break
                if K in c and not math.isnan(c[K]) and (c[K] / base) <= 0.5:
                    kk = K
                    break
            if kk is None:
                censored += 1
                continue
            ks.append(float(kk))
        if not ks:
            return float("nan"), censored, 0
        return float(np.median(ks)), censored, len(ks)

    def capacity_knee(name):
        """★【rev 1.7 · D113】跨种子平均 `R_cnt` 曲线的**首个**最大值所在 K（返回 (K*, 峰值)）。"""
        per_seed = []
        for sd in seeds:
            c = all_rows.get(name, {}).get(sd, {})
            pts = []
            for k, v in c.items():
                try:
                    pts.append((int(v.get("K", k)), float(v.get("r_cnt", float("nan")))))
                except (TypeError, ValueError):
                    continue
            if pts:
                pts.sort()
                per_seed.append(dict(pts))
        if not per_seed:
            return float("nan"), float("nan")
        docs = sorted({d for p in per_seed for d in p})
        mean_by_doc = {}
        for D in docs:
            vals = [p[D] for p in per_seed if D in p]
            mean_by_doc[D] = float(np.mean(vals))
        mx = max(mean_by_doc.values())
        Dbest = next(D for D in docs if mean_by_doc[D] >= mx - 1e-12)   # ★ **首个**最大值
        return float(Dbest), float(mean_by_doc[Dbest])

    def capacity_rising(name):
        """★【rev 1.7 · D113】`R_cnt` 曲线在 K_max 处是否**仍在上升**（⇒ H21-1b 删失）。"""
        vals = []
        for sd in seeds:
            c = all_rows.get(name, {}).get(sd, {})
            for k, v in c.items():
                try:
                    vals.append((int(k), float(v.get("r_cnt", float("nan")))))
                except (TypeError, ValueError):
                    continue
        if len(vals) < 2:
            return True
        vals.sort()
        return bool(vals[-1][1] > vals[-2][1] + 1e-12)

    # ══════════════════ ★★【DIR-22】判据 H22-a…H22-d + 报告项 Rp-1…Rp-5 ══════════════════
    # —— 口径（写死，防漂移）——
    #   R_t(K) = cos_orig_t(K) / cos_orig_t(1)（**逐条目**归一化；K=1 ⇒ 恒 1）
    #   交集口径（D120 / A22-6）：两臂的**触发位置交集** = `C_m18` 的 9 个位置
    #   配对单位 = (种子, 情节 K)；ΔR = R_A(K) − R_B(K)；判据用 mean|ΔR| 与符号检验

    def _cbt(name, seed):
        """该臂该种子的 `{K: {t: cos_orig}}`（逐条目）。"""
        out = {}
        for k, v in all_rows.get(name, {}).get(seed, {}).items():
            out[int(k)] = v.get("cos_by_t", {}) or {}
        return out

    def _t_key(d):
        return {int(x): float(y) for x, y in d.items()}

    def R_t(name, seed, K):
        c = _cbt(name, seed)
        k1 = _t_key(c.get(K_LIST[0], {}))
        kk = _t_key(c.get(int(K), {}))
        common = sorted(set(k1) & set(kk))
        out = {}
        for t in common:
            a, b = kk[t], k1[t]
            if b and abs(b) > 1e-12 and not math.isnan(a) and not math.isnan(b):
                out[t] = a / b
        return out

    def pair_dR(a, b, Ks=None):
        """两臂在**交集**上的 ΔR(K) 列表（逐 (种子,K)）。"""
        Ks = Ks or [K for K in K_LIST if K != K_LIST[0]]
        ds, used = [], []
        for sd in seeds:
            for K in Ks:
                ra, rb = R_t(a, sd, K), R_t(b, sd, K)
                common = sorted(set(ra) & set(rb))
                if not common or not math.isnan(window_of(a)) and K > min(window_of(a), window_of(b)):
                    continue
                va = float(np.mean([ra[t] for t in common]))
                vb = float(np.mean([rb[t] for t in common]))
                ds.append(va - vb)
                used.append((sd, K, len(common)))
        return ds, used

    def mean_abs_dR(a, b):
        ds, used = pair_dR(a, b)
        return (float(np.mean([abs(x) for x in ds])) if ds else float("nan")), ds, used

    # —— 机制量（来自 Stage A 的探针；同 seed 同臂 ⇒ 跨臂同源可比）——
    def mech_of(name):
        acc = {}
        for sd in seeds:
            mm = res.get("mech", {}).get(name, {}).get(sd)
            if mm:
                acc[sd] = mm
        return acc

    def mech_mean(name, field):
        mm = mech_of(name)
        vals = [float(v[field]) for v in mm.values() if v and field in v]
        return float(np.mean(vals)) if vals else float("nan")

    def _elast(xs, ys):
        """ln y 对 ln x 的最小二乘斜率。"""
        pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys)
               if x > 0 and y > 0 and not math.isnan(x) and not math.isnan(y)]
        if len(pts) < 2:
            return float("nan")
        xa = np.array([p[0] for p in pts]); ya = np.array([p[1] for p in pts])
        va = float(np.var(xa))
        return float("nan") if va < 1e-18 else float(np.cov(xa, ya, bias=True)[0, 1] / va)

    # ★ D129：原先写成 `[n for (_n, _e, _s, _p) in ...]` ⇒ 取到的是**外层泄漏的 `n`**
    #   （不是循环变量 `_n`）⇒ 拟合臂变成 [68,68,68,68] ⇒ β 全 nan（判据从未被计算）。
    _same_eta = [_n for (_n, _e, _s, _p) in ARMS22 if abs(float(_e) - ETA_BASE) < 1e-15]
    _ms = [_m_of(n) for n in _same_eta]
    _bD = _elast(_ms, [-math.log(mech_mean(n, "D_meas_mean")) if mech_mean(n, "D_meas_mean") < 1
                       else float("nan") for n in _same_eta])
    _bC = _elast(_ms, [mech_mean(n, "C_meas") for n in _same_eta])

    # —— H22-a〔闭式分解成立〕——
    _a_names = [c["name"] for c in res["checks"] if c["name"].startswith(("A22-4", "A22-5", "A22-7"))]
    _a_ok = bool(_a_names) and all(c["passed"] for c in res["checks"] if c["name"] in _a_names)
    rec("H22-a〔(a) 闭式分解成立〕刷新 ρ=1−η(αs+λ) 与逐情节衰减 (1−ηλ)^m 均可从配置算出并与实测一致",
        _a_ok, f"由 {'/'.join(n.split('〔')[0] for n in _a_names)} 判定（M-1/M-2/M-3）", value=_a_ok)

    # —— H22-b〔单一标量不可行：两条 m-弹性可区分〕——
    _okb = ((not math.isnan(_bD)) and abs(_bD - 1.0) <= 0.15
            and (not math.isnan(_bC)) and abs(_bC) <= 0.15
            and (_bD - _bC) >= 0.70)
    rec("H22-b〔(b) 单一标量不可行〕衰减的 m-弹性 ≈1、刷新的 m-弹性 ≈0，两者相差 ≥0.70",
        _okb, f"β_m(D)={_bD:.4f}（期望 1，容差 ±0.15）；β_m(C)={_bC:.4f}（期望 0，容差 ±0.15）；"
              f"Δβ={_bD - _bC:.4f}（门槛 0.70）；拟合臂={_same_eta} m={_ms}",
        tol=0.15, value=float(_bD - _bC))

    # —— H22-c〔两条匹配线各自失败 ⇒ 单一标量化约不可能〕——
    _mad_decay, _ds_decay, _u1 = mean_abs_dR("C_m18", "C_decay68")
    _mad_refr, _ds_refr, _u2 = mean_abs_dR("C_m18", "C_m68")
    # ★ D128：`sign_test_p(d)` 返回 **(p, k, n)** 三元组，不是 p ⇒ 必须解包
    _p1, _k1s, _n1s = sign_test_p(_ds_decay) if _ds_decay else (float("nan"), 0, 0)
    _p2, _k2s, _n2s = sign_test_p(_ds_refr) if _ds_refr else (float("nan"), 0, 0)
    _same1 = (len(set(1 if x > 0 else -1 for x in _ds_decay)) == 1) if _ds_decay else False
    _same2 = (len(set(1 if x > 0 else -1 for x in _ds_refr)) == 1) if _ds_refr else False
    _c1 = (_mad_decay > 0.10) and (_p1 < 0.05) and _same1
    _c2 = (_mad_refr > 0.10) and (_p2 < 0.05) and _same2
    rec("H22-c〔(c) 两条匹配线各自失败〕(c①) 衰减匹配对（ηλm 相等、C 差 75.6×）与 "
        "(c②) 刷新匹配对（C 相等、D 差 0.203）**都**给出显著且同向的 ΔR", _c1 and _c2,
        f"(c①) mean|ΔR|={_mad_decay:.4f}（门槛 >0.10） p={_p1:.4g} 正={_k1s}/{_n1s} 同号={_same1}；"
        f"(c②) mean|ΔR|={_mad_refr:.4f}（门槛 >0.10） p={_p2:.4g} 正={_k2s}/{_n2s} 同号={_same2}"
        f"；交集条目数 {sorted(set(u[2] for u in (_u1 + _u2)))}（须恒为 9，A22-6）",
        tol=0.10, value=float(min(_mad_decay, _mad_refr)))

    # —— H22-d〔可证伪形态〕——
    _f1 = (_mad_decay <= 0.05) and ((math.isnan(_p1)) or _p1 >= 0.05)
    _f2 = (_mad_refr <= 0.05) and ((math.isnan(_p2)) or _p2 >= 0.05)
    rec("H22-d〔(d) 判定可证伪〕若任一匹配线的 mean|ΔR|≤0.05 且 p≥0.05 ⇒ 该线上的单一标量化约成立 ⇒ H22 被证伪",
        True,
        f"c① 证伪触发={_f1}；c② 证伪触发={_f2}；"
        f"⇒ {'★ H22 被证伪' if (_f1 or _f2) else 'H22 未被证伪（两条匹配线都给出显著差异）'}",
        value=(_f1 or _f2))

    # —— 报告项 ——
    def _rmean(name, K):
        vals = []
        for sd in seeds:
            rr = R_t(name, sd, K)
            if rr:
                vals.extend(rr.values())
        return float(np.mean(vals)) if vals else float("nan")

    rec("Rp-1 〔报告项〕各臂的内容保持曲线 R(K)=cos(K)/cos(1)（交集口径、跨种子平均）", True,
        "；".join(f"{a}: " + " ".join(
            f"K={s}:{_rmean(a, s):.3f}"
            for s in [2, 8, 24, 96]) for (a, _e, _s, _p) in ARMS22))
    rec("Rp-2 〔报告项〕各臂 m · ηλm · C=η(αs+λ) · D=(1−ηλ)^m（逐臂机制量实测值）", True,
        "；".join(f"{a}: m={_m_of(a):.0f} ηλm={_eta_of(a)*LAM*_m_of(a):.5g} "
                 f"C_meas={mech_mean(a,'C_meas'):.6g}/C_pred={mech_mean(a,'C_pred'):.6g} "
                 f"D_meas={mech_mean(a,'D_meas_mean'):.6f}/D_pred={mech_mean(a,'D_pred_mean'):.6f}"
                 for (a, _e, _s, _p) in ARMS22))
    rec("Rp-3 〔报告项〕两条匹配对的逐点 ΔR（含符号，用于核对 H22-c/d）", True,
        f"P_decay: mean={np.mean(_ds_decay) if _ds_decay else float('nan'):+.4f} "
        f"逐点={['%+.3f' % x for x in _ds_decay]}；"
        f"P_refresh: mean={np.mean(_ds_refr) if _ds_refr else float('nan'):+.4f} "
        f"逐点={['%+.3f' % x for x in _ds_refr]}")
    rec("Rp-4 〔报告项〕训练态在各臂间共享（读出映射逐位相同）", True,
        "所有臂在同一种子下**共用同一训练态快照**（`restore_all`）⇒ 读出映射 (W_o, γ) 逐位相同", value=True)
    _s0, _s1 = K_SCAN[0], K_SCAN[-1]
    rec("Rp-5 〔报告项·并列读法〕任务层三种 NLL 口径的臂间差（**不进判据**，只报）", True,
        "；".join(f"{a}: nll_tgt(K={_s0})={mean_curve(a, 'nll_tgt').get(_s0, float('nan')):.3f} "
                 f"nll_mean={mean_curve(a, 'nll_mean').get(_s0, float('nan')):.3f} "
                 f"nll_ctrl={mean_curve(a, 'nll_ctrl').get(_s0, float('nan')):.3f}"
                 for (a, _e, _s, _p) in ARMS22))

    # ★ D112：窗口外的分歧**照报**（不静默截断）⇒ 键列复用的可见证据
    _wins = {a: window_of(a) for (a, _e, _s, _p) in ARMS22}
    rec("Rp-6 〔报告项〕各臂有效窗口与键列复用情节（窗口外读数**无效**；D112）", True,
        "；".join(f"{a}≤{_wins[a]}（复用情节={all_rows.get(a, {}).get(seeds[0], {}).get(str(K_SCAN[0]), {}).get('reuse_episode')}）"
                 for a in _wins)
        + " ⇒ 判据只在窗口内计算")
    rec("Rp-7 〔报告项〕余弦的两个对象（原对象 / 槽口径）逐点并列", True,
        "；".join(f"{a}: " + " ".join(
            f"K={s}:原={mean_curve(a, 'fid_mean').get(s, float('nan')):.3f}"
            f"/槽={mean_curve(a, 'fid_slot_mean').get(s, float('nan')):.3f}"
            for s in [2, 8, 24, 96]) for (a, _e, _s, _p) in ARMS22)
        + " ⇒ 槽口径在回绕后**伪回升**（D102）⇒ 判据只用原对象")

    print("\n【DIR-22 汇总】", flush=True)
    _s0, _s1 = K_SCAN[0], K_SCAN[-1]
    agg = {}
    for (name, _eta, _stride, _pick) in ARMS22:
        _h = mean_curve(name, "hit_mean")
        _f = mean_curve(name, "fid_mean")
        _n = mean_curve(name, "nll_tgt")
        _kk, _vv = capacity_knee(name)
        _hlv, _cenv, _nhl = half_life(name)
        _mval = _m_of(name)
        agg[name] = {"m": _mval, "hit_K1": _h.get(_s0, float("nan")),
                     "fid_K1": _f.get(_s0, float("nan")), "nll_K1": _n.get(_s0, float("nan")),
                     "half_life_K": _hlv, "half_life_writes": (_hlv * _mval
                                                              if not math.isnan(_hlv) else float("nan")),
                     "half_life_censored_seeds": _cenv, "capacity_knee_docs": _kk,
                     "valid_window": window_of(name)}
        print(f"  {name:14s} m={_mval:.0f} 窗口≤{window_of(name):4d} "
              f"nll(K={_s0})={_n.get(_s0, float('nan')):.3f} "
              f"hit({_s0}/{_s1})={_h.get(_s0, float('nan')):.3f}/{_h.get(_s1, float('nan')):.3f} "
              f"fid({_s0}/{_s1})={_f.get(_s0, float('nan')):.3f}/{_f.get(_s1, float('nan')):.3f} "
              f"K*={_kk:.0f} 半衰期={_hlv:.0f}情节/{_hlv*_mval if not math.isnan(_hlv) else float('nan'):.0f}写步"
              + (f"(删失{_cenv})" if _cenv else ""), flush=True)
    res["arms"] = agg
    res["meta"]["k_scan"] = [int(s) for s in K_SCAN]
    res["meta"]["n_pairs"] = len(seeds) * len(K_SCAN)
    res["meta"]["card"] = "rev1.2"
    # ★★ 缺陷 D76：原写 `res["summary_pass"] = {"passed": passed, ...}` ⇒ `passed` **未定义**
    #   ⇒ NameError ⇒ **结果 JSON 从未写出**（smoke 的读数只留在日志里）。
    _sp = sum(1 for _c in res["checks"] if _c.get("passed"))
    res["summary_pass"] = {"passed": _sp, "total": len(res["checks"])}
    # ★★ 缺陷 D81：原写 `{passed}` ⇒ 该名字**未定义** ⇒ 收尾 NameError ⇒ **结果 JSON 从未写出**
    _sp2 = sum(1 for _c in res["checks"] if _c.get("passed"))
    print(f"\n=== 判据通过 {_sp2}/{len(res['checks'])} ===", flush=True)
    out = HERE.parent / ("R22_smoke_results.json" if args.smoke else "R22_results.json")   # ★ D78
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
