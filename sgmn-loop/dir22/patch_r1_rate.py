#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-22 · `dir22/R1_rate.py` 的**唯一生成器**（逐字派生 + 受断言保护的定点替换）。

派生源：`dir21/R1_granularity.py`（md5 `08b68ecb8859e8bda30911beb4ced43f`）。
纪律：**不用代码复用 / import 上游脚本** ⇒ 逐字复制后打定点补丁，每处替换都断言「锚点唯一」。
装置：`dir22/sgmn_module_hebb_v7.py` = `dir21/` 同名文件的**逐字复制**
      （md5 `1a242d02c530307f6c76646ca13469ef`，DEC-02：不造 v8）。

本文件即 DIR-22 的**改动清单**（可审计、可重放）：
  P01 头部说明 · P02 命名空间 `DIR22r11` · P03 臂表 ARMS22（5 格，逐臂 η）·
  P04 逐条目 `cos_by_t` / `hit_by_t`（交集口径所需）· P05 机制探针 `probe_mechanism` ·
  P06 Stage A `A22-1…A22-9` · P07 判据 `H22-a…d` + `Rp-1…Rp-5` ·
  P08 报告项修正（替换 DIR-21 的 Rp-6/Rp-7）· P09 输出文件名 / 汇总横幅。
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "dir21" / "R1_granularity.py"
DEV_SRC = HERE.parent / "dir21" / "sgmn_module_hebb_v7.py"
DST = HERE / "R1_rate.py"
DEV_DST = HERE / "sgmn_module_hebb_v7.py"

MD5_SRC = "08b68ecb8859e8bda30911beb4ced43f"
MD5_DEV = "1a242d02c530307f6c76646ca13469ef"


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def sub1(text, old, new, tag):
    """replace 并要求 old 在整个文件里**恰好出现一次**（唯一匹配断言）。"""
    n = text.count(old)
    assert n == 1, f"[{tag}] 锚点出现 {n} 次（要求恰好 1 次）⇒ 拒绝打补丁"
    return text.replace(old, new)


def splice(text, start, end, new, tag):
    """把 [start, end) 之间的内容替换成 new（两锚点各须唯一）。"""
    assert text.count(start) == 1, f"[{tag}] 起点锚点不唯一"
    assert text.count(end) == 1, f"[{tag}] 终点锚点不唯一"
    i, j = text.index(start), text.index(end)
    assert i < j, f"[{tag}] 锚点顺序错"
    return text[:i] + new + text[j:]


# ══════════════════════════════════════════════════════════ P01 头部
HEAD = r'''
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

'''

# ══════════════════════════════════════════════════════════ P03 臂表
ARMS_NEW = '''# ★★【DIR-22 · rev 1.1 §2.2】臂表 = **5 个冻结单元格**（`(name, eta, write_stride, stride_pick)`）
#   ★ 唯一自变量 = **(η, m) 的联合取值**；m 由 `write_stride` 精确决定（L=35 ⇒ {34,16,4,1}）
#   m = 2 × |{S,2S,…} ∩ [1,34]|（目标 + 干扰 ⇒ ×2）⇒ 实测 2 / 6 / 18 / 68 / 68（A22-3 断言）
#   ★ `C_decay68` 的 η 是**解出来的**，不是凑的：令 η₁λm₁ = η₅λm₅（衰减不变量相等）
#       0.1 × 0.05 × 18 = 0.026470588235294117 × 0.05 × 68 ⇒ 两边都 = 0.09
#   ★★【D130】rev 1.1 的单元格表把这个**公式**算错了一个数：写成 1.32353e−3
#      （= 0.09/68，漏了 ÷λ）⇒ 若真用那个值，ηλm = 0.0045 ≠ 0.09 ⇒ **不满足冻结不变量**。
#      本轮**用的是公式** 0.1×18/68 = 0.0264706（A22-1 打印逐位核对）⇒ 不变量成立。
#      卡片的「C 差 75.6×」也随之订正为实测 3.78×（0.505 / 0.13368）。
ETA_DECAY68 = ETA_BASE * 18.0 / 68.0
ARMS22 = [
    ("C_base",     ETA_BASE,        34, "last"),   # m=2
    ("C_m6",       ETA_BASE,        16, "last"),   # m=6
    ("C_m18",      ETA_BASE,         4, "last"),   # m=18  ← 两条匹配对的公共端
    ("C_m68",      ETA_BASE,         1, "last"),   # m=68
    ("C_decay68",  ETA_DECAY68,      1, "last"),   # m=68，η 已解出（P_decay 的另一端）
]
M_EXPECT = {"C_base": 2, "C_m6": 6, "C_m18": 18, "C_m68": 68, "C_decay68": 68}
'''

# ══════════════════════════════════════════════════════════ P05 机制探针
PROBE = '''# ═══════════════════════════════════════════════════════════════════════════════
# ★★【DIR-22 · rev 1.1 §3.1】**机制探针**：实测 $\u03c1$（刷新收缩）与 $D$（逐情节衰减）
#
#   一维递推（rev 1.1 §0 · D123）：对条目键 $k$（$s=\\lVert k\\rVert^2$），$u\\equiv Wk$，$u_0\\equiv W^0k$：
#       $u \\leftarrow \\rho\\,u+\\eta(\\alpha s\\,v+\\lambda u_0)$，$\\rho=1-\\eta(\\alpha s+\\lambda)$
#   不动点 $u^\\star=(\\alpha s v+\\lambda u_0)/(\\alpha s+\\lambda)$（**与 $\\eta$ 无关**）。
#   ⇒ 刷新：连写同一条目 $r$ 次 ⇒ $(u_r-u^\\star)=\\rho^r(u_0-u^\\star)$ ⇒ 最小二乘估 $\\hat\\rho$。
#   ⇒ 衰减：写 $(k_0,v_0)$ 一次后，**改写另一条正交键** $n$ 次 ⇒
#        $u(k_0)$ 只受 $\\lambda$ 项作用（$k_1\\perp k_0$ ⇒ 数据项读出为 0）⇒ $(1-\\eta\\lambda)^n$。
#        ★ 走的是**真实写路径**（`core.write`），不手工改 `W` ⇒ 不是同义反复。
# ═══════════════════════════════════════════════════════════════════════════════

def _wvec(core, k):
    with torch.no_grad():
        return (core._W() @ k).detach().clone()


def _proj_ratio(cur, base, ref):
    r"""$\\langle cur-ref,\\; base-ref\\rangle / \\lVert base-ref\\rVert^2$（确定性递推 ⇒ 精确）。"""
    a = (cur - ref)
    b = (base - ref)
    den = float((b @ b))
    return float("nan") if den < 1e-30 else float((a @ b) / den)


def probe_mechanism(st, kv_pairs, m_episode, alpha=None, lam=None):
    r"""在**当前状态**上测 $\\hat\\rho$ 与逐情节衰减比 $D$；调用方负责 snapshot/restore。

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


'''

# ══════════════════════════════════════════════════════════ P06 Stage A A22-*
A22 = '''
    # ══════════════════════ ★★【DIR-22】Stage A 新增断言 A22-1…A22-9（继承 DIR-21 的 58 项）═══
    #   ★ 任一项 FAIL ⇒ 由下面的硬闸中止（纪律 ①）⇒ 不可能带着坏装置产出正式轮读数。
    print("\\n[Stage A · DIR-22] A22-1…A22-9", flush=True)

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
        _got = [float(_m.eta) for _m in _st.mems.values()] + \\
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

'''

# ══════════════════════════════════════════════════════════ P07 判据
CRIT = '''    # ══════════════════ ★★【DIR-22】判据 H22-a…H22-d + 报告项 Rp-1…Rp-5 ══════════════════
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

'''

# ══════════════════════════════════════════════════════════ P08 Rp-6/Rp-7 替换
RP67 = '''    # ★ D112：窗口外的分歧**照报**（不静默截断）⇒ 键列复用的可见证据
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

'''


def main():
    if md5(SRC) != MD5_SRC:
        sys.exit(f"源脚本 md5 不符：{md5(SRC)} ≠ {MD5_SRC}")
    if md5(DEV_SRC) != MD5_DEV:
        sys.exit(f"源装置 md5 不符：{md5(DEV_SRC)} ≠ {MD5_DEV}")
    # —— 装置：逐字复制（DEC-02）——
    DEV_DST.write_bytes(DEV_SRC.read_bytes())
    assert md5(DEV_DST) == MD5_DEV

    t = SRC.read_text(encoding="utf-8")
    n0 = len(t)

    # P01 头部：插在 `import argparse` 之前（原本的旧 docstring 原样保留在后面）
    t = sub1(t, "\nimport argparse\n", HEAD + "\nimport argparse\n", "P01")

    # P02 命名空间
    t = sub1(t, 'res = {"dir": "DIR-21", "round": "R1", "checks": [], "arms": {}, "meta": {"card": "rev1.8"}}',
             'res = {"dir": "DIR-22", "round": "R1", "checks": [], "arms": {}, "mech": {},\n'
             '       "meta": {"card": "rev1.2", "mvp_grid": "5 cells x 5 seeds"}}', "P02a")
    t = sub1(t, '_DIR_TAG = "DIR21r18"', '_DIR_TAG = "DIR22r11"', "P02b")
    t = sub1(t, '    for _f in STATE_DIR.glob("DIR21_*.pt"):',
             '    for _f in STATE_DIR.glob("DIR22_*.pt"):', "P02c")

    # P03 臂表
    old_arms = '''ARMS21 = [
    ("G_tok",      ETA_BASE,        1, "last"),
    ("G_seg4",     ETA_BASE,       17, "last"),
    ("G_seg4_alt", ETA_BASE,       18, "last"),
    ("G_tok_rate", ETA_BASE / 17.0,  1, "last"),
    ("G_doc",      ETA_BASE,       68, "last"),
]
'''
    t = sub1(t, old_arms, ARMS_NEW, "P03")
    # 全量重命名（臂表名本身已改；其余引用点）
    assert t.count("ARMS21") >= 8, f"ARMS21 引用数异常：{t.count('ARMS21')}"
    t = t.replace("ARMS21", "ARMS22")

    # P05 机制探针（插在 `def rec(` 之前 ⇒ 模块级，`main()` 可用）
    t = sub1(t, "\ndef rec(name, passed, detail, tol=None, value=None):",
             PROBE + "\ndef rec(name, passed, detail, tol=None, value=None):", "P05")
    # probe_mechanism 用到 ALPHA/LAM/np/torch ⇒ 均已在模块级定义 ✔（需在其之后：rec 在常量之后）

    # P04 逐条目 cos/hit 落盘
    t = sub1(t, '''                    "nll_by_t": {str(r["t"]): float(r["nll"]) for r in per_entry},''',
             '''                    "nll_by_t": {str(r["t"]): float(r["nll"]) for r in per_entry},
                    # ★★【DIR-22 · rev 1.1 §4】**逐条目**余弦与命中（交集口径所需，D120）
                    "cos_by_t": {str(r["t"]): float(r.get("cos_orig", float("nan"))) for r in per_entry},
                    "hit_by_t": {str(r["t"]): float(r.get("hit", float("nan"))) for r in per_entry},''',
             "P04")

    # P04b 机制探针接入臂循环（写完文档 1、捕获 (k,v) 之后；探针后**精确复原**）
    t = sub1(t, '''            # ★ D112：该臂的**有效窗口**（锚键列被再次写入的情节 − 1）''',
             '''            # ★★【DIR-22】机制探针：在「刚写完文档 1」的状态上测 ρ 与 D，
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
            # ★ D112：该臂的**有效窗口**（锚键列被再次写入的情节 − 1）''', "P04b")

    # P06 Stage A A22-*（插在 DIR-21 的断言收尾之前）
    t = sub1(t, '    print("[Stage A · DIR-21] 断言完成（A21-1…A21-15 + 继承项）", flush=True)',
             A22 + '    print("[Stage A · DIR-21 继承] 断言完成（A21-1…A21-15）", flush=True)\n'
             + '    print("[Stage A · DIR-22] 断言完成（A22-1…A22-9）", flush=True)', "P06")

    # P07 判据块：替换 DIR-21 的 H21-1…Rp-5
    t = splice(t,
               "    # —— H21-1〔粒度不可互换：任务层〕——",
               '    # ★ D112：窗口外的分歧**照报**（不静默截断）⇒ 键列复用的可见证据',
               CRIT, "P07")
    # P08 Rp-6/Rp-7：替换成 DIR-22 的版本（原块以 `rec("Rp-7` 结尾）
    t = splice(t,
               '    # ★ D112：窗口外的分歧**照报**（不静默截断）⇒ 键列复用的可见证据',
               '    print("\\n【DIR-21 汇总】", flush=True)',
               RP67, "P08")

    # P09 横幅 / 输出文件名 / card
    t = sub1(t, '    print("=== DIR-21 R1（卡片 rev 1.8 · 写步粒度 / 情节轴扫描；判据阈值仍为 rev 1.1 §3）===", flush=True)',
             '    print("=== DIR-22 R1（卡片 rev 1.2 · 写侧速率 vs 情节级速率；5 格 MVP；判据 H22-a…d）===", flush=True)',
             "P09a")
    t = t.replace('print("\\n【DIR-21 汇总】", flush=True)',
                  'print("\\n【DIR-22 汇总】", flush=True)')
    t = t.replace('"R21_smoke_results.json" if args.smoke else "R21_results.json"',
                  '"R22_smoke_results.json" if args.smoke else "R22_results.json"')
    t = sub1(t, '    res["meta"]["card"] = "rev1.8"', '    res["meta"]["card"] = "rev1.2"', "P09b")

    # A21 继承项的**文字**校正（数值/逻辑不动）：
    t = sub1(t, '''f"折算情节：G_tok≈{_horizon / 68:.3f} / G_seg4≈{_horizon / 4:.2f} / G_doc≈{_horizon / 2:.2f} 篇",''',
             '''f"折算情节（DIR-22 的 5 格）：C_m68≈{_horizon / 68:.4f} / C_m18≈{_horizon / 18:.3f} / "
        f"C_m6≈{_horizon / 6:.2f} / C_base≈{_horizon / 2:.2f} 篇",''', "P09c")
    t = sub1(t, '''f"1 篇实测 core.count={_mc}；期望 m=2×{len(_tr)}={_mdoc}（D116：G_tok=68 / G_seg4=4 / "
        f"G_seg4_alt=4 / G_tok_rate=68 / G_doc=2）",''',
             '''f"1 篇实测 core.count={_mc}；期望 m=2×{len(_tr)}={_mdoc}"
        f"（DIR-22：C_base=2 / C_m6=6 / C_m18=18 / C_m68=68 / C_decay68=68）",''', "P09d")
    # A21-7 的等比对照臂 ⇒ DIR-22 的「解出臂」（η = ETA_BASE×18/68）
    t = t.replace("ETA_BASE / 17.0", "ETA_DECAY68")
    t = t.replace("η′=η/17 逐位生效", "η′=η×18/68（解出臂）逐位生效")

    DST.write_text(t, encoding="utf-8")
    print(f"写出 {DST}（{len(t)} B；源 {n0} B）")
    print(f"装置 {DEV_DST}（md5 {md5(DEV_DST)}）")


if __name__ == "__main__":
    main()
