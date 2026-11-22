#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import argparse
import ctypes
import gc
import hashlib
import os
import json
import math
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sgmn_module_hebb import SGMNStack as SGS, grad_report   # noqa: E402

MODEL = os.environ.get("SGMN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOUNT = [8, 12, 16]
ALPHA, LAM, GAMMA0, R = 1.0, 0.05, 0.05, 8
ETA = 1e-5
BETA = 64.0
# 【rev 1.3】目标文档与干扰文档各约 37 条 ⇒ 合计 ~74 条 ⇒ N_ENTRIES 必须 ≥ 74，
#   否则核心槽位回绕 ⇒ 命中指标与真值下标错位（D30 的同类缺陷）。128 ⇒ 槽位 = 写入序号。
N_ENTRIES = 128
SEEDS = [20261028, 20261104, 7]
N_EVAL, N_TRAIN, FILLER_TOK = 60, 200, 30
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

res = {"dir": "DIR-18", "round": os.environ.get("SGMN_ROUND_TAG", "R2"), "checks": [], "arms": {}, "meta": {}}



def _ck(t):
    """参数逐位校验和（避免 clone 整份基座 ⇒ 沙箱 3 GB 内存不够）。"""
    t = t.detach().flatten()
    h = hashlib.md5()
    step = 8_000_000
    for i in range(0, t.numel(), step):
        h.update(t[i:i + step].numpy().tobytes())
    return h.hexdigest()



_LIBC = ctypes.CDLL("libc.so.6")



def _jsonable(o):
    """把 numpy / torch 标量转成 JSON 可序列化（用于增量落盘）。"""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        # ★ 缺陷修复 D59：原先把 NaN 写成 None ⇒ **读回后**聚合时 `np.nanmean([...None...])`
        #   抛 `TypeError: unsupported operand type(s) for +: 'NoneType' and 'NoneType'`
        #   ⇒ 续跑路径**必然崩溃**。修法：NaN **原样保留**（json.dump 默认写 `NaN`，
        #   Python 的 json.load 能读回 float('nan')）⇒ 落盘/读回**逐位等价**。
        #   （读回端另有 `_num()` 兜底 ⇒ 兼容旧版写出的 None。）
        return f
    if isinstance(o, (np.integer, int)):
        return int(o)
    if hasattr(o, "item") and callable(getattr(o, "item")):
        try:
            return _jsonable(o.item())
        except Exception:
            pass
    return o


def _num(v):
    """★ 缺陷修复 D59：把 `None`（旧版落盘把 NaN 写成 None）统一成 `float('nan')`。
    ⇒ 聚合端与「增量复用」打印端都走这个函数 ⇒ 旧文件与新文件都能读。"""
    if v is None:
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _load_incremental():
    """读取增量落盘的结果（续跑用）。"""
    _inc = HERE.parent / ".ops" / "cpu" / "R3_cpu_incremental.json"
    if not _inc.exists():
        return {}
    try:
        d = json.load(open(_inc))
        _rows = d.get("rows", {}) or {}
        # ★ 改动 14：seed 键转回 int（与首次运行一致 ⇒ `per_seed` 的迭代顺序稳定）
        return {k: {int(sk): vv for sk, vv in v.items()} for k, v in _rows.items()}
    except Exception:
        return {}


def _pick_seed(per_seed, seed):
    """★ 缺陷修复 D56：按 **int 键**取一个 (臂,种子) 的行；str 键兜底。
    （`_load_incremental` 返回 int 键；首次运行写入的也是 int 键 ⇒ 统一 int。）
    返回 None 表示该组合尚未完成。"""
    if not per_seed:
        return None
    if seed in per_seed:
        return per_seed[seed]
    sk = str(seed)
    if sk in per_seed:
        return per_seed[sk]
    return None


def _free_stack(st):
    """★ CPU 降级轮改动 7：释放装置栈占用的张量（`st.remove()` 只摘钩子，不释放参数）。
    ⇒ 语义不变（钩子已摘、参数不再被引用），只是把内存还给分配器。
    ★ 改动 9：追加 `malloc_trim(0)` —— glibc 默认把释放的小块留在 arena 里不还给 OS
      （实测 RSS 只降 2379 vs 2614 ⇒ 有 ~150 MB 被 glibc 扣住）⇒ 在 3 GB 上限下这是关键。"""
    st.remove()
    try:
        st.mems.clear()
    except Exception:
        pass
    gc.collect()
    try:
        _LIBC.malloc_trim(0)
    except Exception:
        pass


# ★ CPU 降级轮改动 8：两阶段分离（沙箱 3 GB 内存下，Stage A 19 项与主循环同进程会 OOM）
_SGMN_STAGEA_CACHE = HERE.parent / ".ops" / "cpu" / "stageA_checks.json"


_SGMN_STAGEA_PART1 = HERE.parent / ".ops" / "cpu" / "stageA_part1.json"


def _stagea_cache_ok():
    """已缓存且全部 PASS 的 Stage A 结果（跨进程复用，避免重复计算占用内存）。"""
    if not _SGMN_STAGEA_CACHE.exists():
        return None
    try:
        d = json.load(open(_SGMN_STAGEA_CACHE))
    except Exception:
        return None
    if not d.get("all_passed"):
        return None
    return d.get("checks", [])


def _stagea_part1_load():
    """前半段（①–⑭）的缓存结果（改动 11）。"""
    if not _SGMN_STAGEA_PART1.exists():
        return None
    try:
        d = json.load(open(_SGMN_STAGEA_PART1))
    except Exception:
        return None
    return d.get("checks", [])


def _stagea_part1_save(checks):
    """前半段（①–⑭）落盘（改动 11）。"""
    _SGMN_STAGEA_PART1.parent.mkdir(parents=True, exist_ok=True)
    with open(_SGMN_STAGEA_PART1, "w") as f:
        json.dump({"all_passed": all(c["passed"] for c in checks),
                   "checks": checks}, f, indent=2, ensure_ascii=False)

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


def build_docs(tok, seed, n_train, n_eval):
    """每篇样本 = 目标文档 + 干扰文档（同线索 CUE_A、异码、异上下文）。
    码全局唯一 ⇒ 训练码与评测码**零重叠**（rev 1.3 的「评测码留出」）。"""
    import re as _re
    text = load_filler()
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
        if code_id == dcode_id:
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
ARMS = [
    ("W0_verbatim",          "cue", "explicit_kv", None,  True,  False, "linear",  "normal"),
    # —— 主臂（卡片 rev 1.2 修订 55 冻结：η=1e-7、线性键、装置原始尺度）
    ("W3_hebb_full_eta7",    "cue", "hebbian_full", 1e-7, True,  False, "linear",  "normal"),
    ("W3_hebb_full_eta8",    "cue", "hebbian_full", 1e-8, True,  False, "linear",  "normal"),
    # —— ★ 键尺度修正臂（D50/D51：逐写归一化到 ‖k‖²=5 ⇒ 稳定窗口 ∩ 学习窗口非空）
    ("W6_hebb_full_renorm_eta2", "cue", "hebbian_full", 1e-2, True, False, "renorm", "normal"),
    ("W6b_hebb_full_renorm_eta1", "cue", "hebbian_full", 1e-1, True, False, "renorm", "normal"),
    # —— ★ 正交键臂（修订 53/54：‖k‖²=5 且键两两正交 ⇒ 键共线障碍也被隔离）
    ("W5_hebb_full_ortho_eta2",  "cue", "hebbian_full", 1e-2, True, False, "ortho", "normal"),
    ("W5b_hebb_full_ortho_eta1", "cue", "hebbian_full", 1e-1, True, False, "ortho", "normal"),
    # —— ★ 归因对照臂（修订 56）：地址表照写，W 从不更新 ⇒ 收益是否来自写规则？
    ("Wnowrite_hebb_cue_train", "cue", "hebbian_full", 1e-7, True, False, "linear", "nowrite"),
    # ★【卡片 rev 1.2 修订 64】**可用配置下的写规则归因对照**：
    #   ortho（键尺度已修）+ W 从不更新 ⇒ 与 `W5b_hebb_full_ortho_eta1` 唯一差别 = 写步
    #   ⇒ ⇒ H2'' 测「在写规则真正能工作���配置下，写步本身是否贡献任务收益」
    #   （linear 配置的 nowrite 对照不够：那里 W 本来就没动 ⇒ 差值必为 0）
    ("Wnowrite_ortho_eta1", "cue", "hebbian_full", 1e-1, True, False, "ortho", "nowrite"),
    ("Wmisaddr_hebb_cue_train", "misaddress", "hebbian_full", 1e-7, True, False, "linear", "normal"),
    # —— ★ γ 锁定臂（修订 55）：跨方向复核 DIR-17 的 D-24
    ("W7_hebb_full_gammalock", "cue", "hebbian_full", 1e-7, True, False, "linear", "gammalock"),
    # —— ★ 写侧自监督臂（修订 53 · D47）：P_v 由写侧目标训练（Freeze C 下唯一可实现形态）
    ("W4b_hebb_full_pvw",     "cue", "hebbian_full", 1e-7, True,  False, "linear",  "pvw"),
    # —— 缺陷留档样本（D47：任务损失更新 P_v ⇒ 梯度恒 None ⇒ 与主臂逐位相同）
    ("W4_hebb_full_trainPv",  "cue", "hebbian_full", 1e-7, True,  True,  "linear",  "normal"),
    ("E4_cue_notrain",        "cue", "explicit_kv", None,  False, False, "linear",  "normal"),
    ("E5_base",               None, "explicit_kv", None,  False, False, "linear",  "normal"),
]


def make_stack(model, d, query_mode, seed=20261028, mem_kind="explicit_kv", eta=None,
               key_mode="linear"):
    st = SGS(model, MOUNT, d, d, R, ALPHA, LAM, (eta if eta is not None else ETA), GAMMA0, DEV,
             seed=seed, use_factorized=False, no_write=False, perpos=True,
             mem_kind=mem_kind, n_entries=N_ENTRIES, key_mode=key_mode)
    for m in st.mems.values():
        m.key_mode = key_mode
        m.query_mode = query_mode
        m.addr_mode = "sig"
        m.read_beta = BETA
    return st


def write_both(st, d):
    """写阶段：目标文档 + 干扰文档（都 no_grad）。顺序固定 ⇒ 目标条目在前。"""
    ids = torch.tensor(d["ids"], dtype=torch.long, device=DEV).unsqueeze(0)
    dids = torch.tensor(d["dist_ids"], dtype=torch.long, device=DEV).unsqueeze(0)
    with torch.no_grad():
        st.model(input_ids=ids)
        st.model(input_ids=dids)


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


def train_arm(st, train_docs, mode, steps=None, train_pv=False, variant="normal"):
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
    # ★【修订 53 · 写侧自监督臂 `pvw`】单独的优化器，**只含 P_v**
    opt_pv = None
    if variant == "pvw":
        opt_pv = torch.optim.Adam([m.Pv for m in st.mems.values()], lr=1e-3)
    w0 = {n: t.detach().clone() for n, t in
          [(f"L{li}.{k}", getattr(m, k)) for li, m in st.mems.items() for k in ("Wo", "gamma", "Wq")]}
    traj = []
    n = len(train_docs) if steps is None else min(steps, len(train_docs))
    for d in train_docs[:n]:
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
        lg = st.model(input_ids=pre).logits.float()   # ③ 读阶段（梯度开）
        tgt = torch.tensor([d["code_ids"][0]], dtype=torch.long, device=DEV)
        loss = F.cross_entropy(lg[0, -1:, :], tgt)
        loss.backward()
        gnorm = float(sum(float(p_.grad.norm()) ** 2 for p_ in params
                          if p_.grad is not None) ** 0.5)
        opt.step()
        traj.append((float(loss), gnorm))
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
    text = load_filler()
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
    n_eval = int(os.environ.get('SGMN_N_EVAL', 12 if args.smoke else N_EVAL))
    n_train = int(os.environ.get('SGMN_N_TRAIN', 40 if args.smoke else N_TRAIN))
    _ns = int(os.environ.get('SGMN_N_SEED', 2 if args.smoke else len(SEEDS)))
    seeds = SEEDS[:_ns]

    print("=== DIR-17 R1（脚本 v1.1 · 卡片 rev 1.3 任务构造）：可寻址读取 + 读侧训练 ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # ★ CPU 降级轮改动 10：避开 `from_pretrained` 的加载期双份拷贝（实测峰值 2920 MB ⇒ 3 GB 上限下 OOM）
    #   ⇒ 先用 safetensors **mmap** 读权重，再 `load_state_dict` 灌入空壳模型 ⇒ 峰值显著降低。
    #   ★ 语义等价：**同一份权重、同一 dtype（float32）**、同一模型结构 ⇒ 数值完全一致。
    try:
        import safetensors.torch as _st_
        _sd = _st_.load_file(os.path.join(MODEL, "model.safetensors"), device="cpu")
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                     low_cpu_mem_usage=True)
        _miss, _unexp = model.load_state_dict(_sd, strict=False)
        del _sd
        gc.collect()
        _LIBC.malloc_trim(0)
        assert not _unexp, f"unexpected keys: {_unexp}"
    except Exception as _e:
        print(f"[CPU 降级轮] 改动 10 不可用（{_e}）⇒ 回退标准加载", flush=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size

    # ---------------- Stage A
    _cached = None
    if os.environ.get("SGMN_PHASE") == "main":
        _cached = _stagea_cache_ok()
    if _cached is not None:
        res["checks"] = _cached
        print(f"\n[Stage A] ★ 复用跨进程缓存的 Stage A 结果（{len(_cached)} 项，全部 PASS）"
              f" ⇒ 跳过重算（改动 8：两阶段分离以规避 3 GB 内存上限）", flush=True)
        for _c in _cached:
            print(f"[PASS] {_c['name']}: {_c['detail']}", flush=True)
    else:
        print("\n[Stage A]", flush=True)
        # ★ 改动 11：前半段（①–⑭）若已有缓存 ⇒ 直接复用（跨进程拼接）
        ids8 = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
        _p1 = _stagea_part1_load()
        if _p1 is not None:
            res["checks"] = _p1
            print(f"[Stage A] ★ 复用前半段（①–⑭）缓存：{len(_p1)} 项", flush=True)
            for _c in _p1:
                print(f"[PASS] {_c['name']}: {_c['detail']}", flush=True)

        else:
            # A1' Freeze C（cue 模式）
            st = make_stack(model, d, "cue")
            base_before = {n: _ck(q) for n, q in model.named_parameters()}   # ★ CPU 降级轮：md5 代替 clone
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
            base_same = all(_ck(q) == base_before[n] for n, q in model.named_parameters())  # ★ CPU 降级轮：md5 比较
            nh, nm = len(st._handles), len(st.mems)
            _free_stack(st)
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
            _free_stack(st2)
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
            _free_stack(st_h)
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
            _free_stack(st_h2)
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
            _free_stack(st_p)
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
            _free_stack(st_v)
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
            _free_stack(st_n)
            rec("A2'⑭ 槽表状态**确定性**：陈旧残留数逐篇一致 + 槽号映射 == (写入前 count + j) % N",
                (len(set(stale)) == 1) and (len(slots_ok) == 0),
                f"陈旧条目数（逐篇/逐层）= {stale}（须**完全一致** ⇒ 确定性）；"
                f"槽号映射违例 = {len(slots_ok)} 处（须 0）",
                tol=0, value=len(slots_ok))
        # ★ 改动 11：前半段（①–⑭）落盘；若是 part1 阶段 ⇒ 落盘后退出（释放内存）
        try:
            _stagea_part1_save(res['checks'])
            print(f"[Stage A] ★ 前半段（①–⑭）已缓存：{len(res['checks'])} 项", flush=True)
        except Exception as _e:
            print(f"[Stage A] ★ 前半段缓存失败：{_e}", flush=True)
        if os.environ.get('SGMN_PHASE') == 'stagea1':
            raise SystemExit(0 if all(c['passed'] for c in res['checks']) else 3)
        else:
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

            # A4' 训练不发散
            #   ★ 改动 12：第二阶段（⑮–⑰）独立运行时，前半段的 `traj`/`rep` 不可见
            #     ⇒ ⇒ 在本阶段**自己跑一遍小规模训练**来取这两个量（判据本身不变：
            #     「损失有限 + max|Wo| 有限 + gamma 有限」）。
            _st_a4 = make_stack(model, d, "cue")
            _tr_a4, _ev_a4 = build_docs(tok, SEEDS[0], 4, 2)
            _traj_a4, _rep_a4 = train_arm(_st_a4, _tr_a4, "cue")
            _free_stack(_st_a4)
            losses = [t[0] for t in _traj_a4]
            finite = all(np.isfinite(losses))
            max_wo = max(v["max_abs"] for k, v in _rep_a4.items() if k.endswith("Wo"))
            gam_fin = all(np.isfinite(v["value"]) for k, v in _rep_a4.items() if k.endswith("gamma"))
            rec("A4' 训练不发散（损失有限、max|Wo| 有限、gamma 有限）",
                finite and np.isfinite(max_wo) and gam_fin,
                f"loss 首/末 = {losses[0]:.4f}/{losses[-1]:.4f}；max|Wo|={max_wo:.4g}；"
                f"gamma 终值={[round(v['value'],4) for k,v in _rep_a4.items() if k.endswith('gamma')]}",
                value=max_wo)
            res["meta"]["a4_loss"] = {"first": losses[0], "last": losses[-1]}


    # ★ CPU 降级轮改动 8：Stage A 结果落盘（供主循环进程复用 ⇒ 规避 3 GB 内存峰值）
    try:
        _SGMN_STAGEA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(_SGMN_STAGEA_CACHE, "w") as _f:
            json.dump({"all_passed": all(c["passed"] for c in res["checks"]),
                       "checks": res["checks"]}, _f, indent=2, ensure_ascii=False)
        print(f"[Stage A] ★ 已缓存到 {_SGMN_STAGEA_CACHE}", flush=True)
    except Exception as _e:
        print(f"[Stage A] ★ 缓存写入失败（不影响本轮）：{_e}", flush=True)

    # ★ CPU 降级轮改动 8：Stage A 阶段跑完即退出（把峰值内存还给 OS）
    if os.environ.get("SGMN_PHASE") == "stagea":
        _f0 = [c["name"] for c in res["checks"] if not c["passed"]]
        print(f"\n[Stage A 阶段结束] 共 {len(res['checks'])} 项，FAIL {len(_f0)} 项"
              + (f"：{_f0}" if _f0 else " ⇒ 全部通过"), flush=True)
        out = HERE.parent / ".ops" / "cpu" / ("stageA_only_results.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"checks": res["checks"], "failed": _f0}, f, indent=2, ensure_ascii=False)
        raise SystemExit(0 if not _f0 else 3)

    # ---------------- ★ Stage A 硬闸（纪律 ①）：任一 FAIL ⇒ **不得开主循环**
    #   （卡片 rev 1.2 §2.1：「任一 FAIL 不得开正式轮」；此前靠外部脚本把关，
    #     现在装置内部也设闸 ⇒ 无论谁启动本脚本，都不可能在 Stage A 失败时产出正式轮结果。）
    _failed = [c["name"] for c in res["checks"] if not c["passed"]]
    print(f"\n[Stage A 闸] 共 {len(res['checks'])} 项，FAIL {len(_failed)} 项"
          + (f"：{_failed}" if _failed else " ⇒ 全部通过"), flush=True)
    if _failed:
        print("[Stage A 闸] ★ 有 FAIL ⇒ **中止，不开主循环**（纪律 ①）", flush=True)
        out = HERE.parent / ("R28_smoke_results.json" if args.smoke else "R28_results.json")
        res["aborted"] = {"stage_a_failed": _failed}
        with open(out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        raise SystemExit(3)

    # ---------------- 主循环
    print("\n[主循环]", flush=True)
    all_rows = _load_incremental()      # ★ 改动 14：续跑（跳过已完成的臂）
    _reuse_hits = 0
    if all_rows:
        _done = sum(len(v) for v in all_rows.values())
        print(f"\n[主循环] ★ 复用增量结果：{len(all_rows)} 臂 / {_done} 个 (臂,种子) 组合已完成 ⇒ 跳过",
              flush=True)
        # ★★ 缺陷修复 D56 的**自检**：跳过判据必须真的能命中（键类型一致）
        _probe = None
        for _n, _v in all_rows.items():
            for _k in _v:
                _probe = (_n, _k); break
            if _probe: break
        if _probe is not None:
            _n0, _k0 = _probe
            _hit = _pick_seed(all_rows[_n0], _k0 if isinstance(_k0, int) else int(_k0)) is not None
            print(f"[主循环] ★★ 续跑跳过判据自检：臂 {_n0} 键 {_k0!r}({type(_k0).__name__}) ⇒ 命中={_hit}",
                  flush=True)
            assert _hit, "★ 缺陷修复 D56 的自检失败：续跑跳过判据仍不命中 ⇒ 续跑会重算所有臂"
        else:
            print("[主循环] ★★ 续跑跳过判据自检：无已存臂可探 ⇒ 跳过", flush=True)
    for seed in seeds:
        tr, ev = build_docs(tok, seed, n_train, n_eval)
        # ★【修订 58】**交换性对照**：从 code_pool **均匀抽样** K=32 个**从未写入**的 token
        #   （码本身也是 `rng.choice(code_pool)` 的均匀抽样 ⇒ 对照与码**同分布、同稀有度**）
        #   ⇒ 跨臂固定 ⇒ 配对比较。
        ctrl_ids = build_ctrl_ids(tok, seed, tr, ev)
        for name, qm, mkind, aeta, train, train_pv, kmode, variant in ARMS:
            # ★★ 缺陷修复 D56（2026-11-06）：**续跑跳过判据的键类型不匹配**
            #   `_load_incremental()` 把 seed 键**转成 int**，而此处用 `str(seed)` 去查
            #   ⇒ **永远不命中** ⇒ 续跑会**重算所有已完成臂** ⇒ 增量落盘形同虚设。
            #   修法：统一用 **int 键**；同时保留 str 键兜底（防止外部写入的 JSON 用 str 键）。
            _rr = _pick_seed(all_rows.get(name, {}), seed)
            if _rr is not None:                                    # ★ 改动 14：跳过已完成
                _reuse_hits += 1
                print(f"  seed={seed} {name:26s} (增量复用) nll={np.mean([_num(r['nll']) for r in _rr]):.4f} "
                      f"lp={np.mean([_num(r['lp']) for r in _rr]):+.4f} "
                      f"hit={np.nanmean([_num(r.get('hit')) for r in _rr]):.4f} "
                      f"addr={np.nanmean([_num(r.get('addr')) for r in _rr]):.4f} "
                      f"addr_slot={np.nanmean([_num(r.get('addr_slot')) for r in _rr]):.4f} "
                      f"fid={np.nanmean([_num(r.get('fid')) for r in _rr]):.4f} "
                      f"fid_true={np.nanmean([_num(r.get('fid_true')) for r in _rr]):.4f} "
                      f"cos_y_vtrue={np.nanmean([_num(r.get('cos_y_vtrue')) for r in _rr]):.4f}", flush=True)
                continue
            if qm is None:
                rows = eval_base(model, ev, ctrl_ids=ctrl_ids)
            else:
                st = make_stack(model, d, qm, seed=seed, mem_kind=mkind, eta=aeta,
                                key_mode=kmode)
                # ★【修订 56】归因对照臂：地址表照写、W 从不更新
                if variant == "nowrite":
                    make_nowrite_variant(st)
                if train:
                    traj, rep = train_arm(st, tr, qm, train_pv=train_pv, variant=variant)
                else:
                    traj, rep = [], {}
                rows = eval_arm(st, ev, ctrl_ids=ctrl_ids)
                _free_stack(st)
            all_rows.setdefault(name, {})[seed] = rows
            # ★ 改动 13/14：**每臂完整结果落盘**（沙箱后台任务可能被回收 ⇒ 不让已完成臂的算力白费）
            #   ★ 改动 14：落盘的是**完整的行**（不只是计数）⇒ 下次启动可**跳过已完成的臂**（续跑）
            try:
                _inc = HERE.parent / ".ops" / "cpu" / "R3_cpu_incremental.json"
                _inc.parent.mkdir(parents=True, exist_ok=True)
                with open(_inc, "w") as _f:
                    json.dump({"rows": _jsonable(all_rows)}, _f, indent=2, ensure_ascii=False)
            except Exception as _e:
                print(f"[增量落盘失败] {_e}", flush=True)
            print(f"  seed={seed} {name:26s} nll={np.mean([r['nll'] for r in rows]):.4f} "
                  f"lp={np.mean([r['lp'] for r in rows]):+.4f} "
                  f"hit={np.nanmean([r.get('hit', np.nan) for r in rows]):.4f} "
                  f"addr={np.nanmean([r.get('addr', np.nan) for r in rows]):.4f} "
                  f"addr_slot={np.nanmean([r.get('addr_slot', np.nan) for r in rows]):.4f} "
                  f"fid={np.nanmean([r.get('fid', np.nan) for r in rows]):.4f} "
                  f"fid_true={np.nanmean([r.get('fid_true', np.nan) for r in rows]):.4f} "
                  f"cos_y_vtrue={np.nanmean([r.get('cos_y_vtrue', np.nan) for r in rows]):.4f}", flush=True)

    # ---------------- 汇总
    print("\n[汇总与判据]", flush=True)
    agg = {}
    for name, per_seed in all_rows.items():
        # ★ 改动 14：按**规范种子顺序**排序 ⇒ `*_by_seed` 与 `seed_delta` 在续跑后仍然对齐
        _ordered = sorted(per_seed.items(), key=lambda kv: seeds.index(kv[0]) if kv[0] in seeds else 99)
        per_seed = dict(_ordered)
        lps = [_num(r["lp"]) for rs in per_seed.values() for r in rs]
        nlls = [_num(r["nll"]) for rs in per_seed.values() for r in rs]
        hits = [_num(r["hit"]) for rs in per_seed.values() for r in rs if "hit" in r]
        fids = [_num(r["fid"]) for rs in per_seed.values() for r in rs if "fid" in r]
        addrs = [_num(r["addr"]) for rs in per_seed.values() for r in rs if "addr" in r]
        adrss = [_num(r["addr_slot"]) for rs in per_seed.values() for r in rs if "addr_slot" in r]
        fidt = [_num(r["fid_true"]) for rs in per_seed.values() for r in rs if "fid_true" in r]
        cyv = [_num(r["cos_y_vtrue"]) for rs in per_seed.values() for r in rs if "cos_y_vtrue" in r]
        nllc = [_num(r["nll_ctrl"]) for rs in per_seed.values() for r in rs if "nll_ctrl" in r]
        nllD = [_num(r["nll_dcode"]) for rs in per_seed.values() for r in rs if "nll_dcode" in r]
        agg[name] = {"lp": float(np.mean(lps)), "nll": float(np.mean(nlls)),
                     "fid": (float(np.nanmean(fids)) if fids else None),
                     "fid_true": (float(np.nanmean(fidt)) if fidt else None),
                     "cos_y_vtrue": (float(np.nanmean(cyv)) if cyv else None),
                     "nll_ctrl": (float(np.nanmean(nllc)) if nllc else None),
                     "nll_dcode": (float(np.nanmean(nllD)) if nllD else None),
                     "addr": (float(np.nanmean(addrs)) if addrs else None),
                     "addr_slot": (float(np.nanmean(adrss)) if adrss else None),
                     "lp_by_seed": [float(np.mean([r["lp"] for r in rs])) for rs in per_seed.values()],
                     "nll_by_seed": [float(np.mean([r["nll"] for r in rs])) for rs in per_seed.values()],
                     "hit": (float(np.mean(hits)) if hits else None), "n": len(lps)}
        print(f"  {name:26s} nll={agg[name]['nll']:.4f} lp={agg[name]['lp']:+.4f} "
              f"hit={agg[name]['hit']} addr={agg[name]['addr']} addr_slot={agg[name]['addr_slot']} "
              f"fid={agg[name]['fid']} fid_true={agg[name]['fid_true']} "
              f"cos_y_vtrue={agg[name]['cos_y_vtrue']} ctrl={agg[name]['nll_ctrl']}", flush=True)
    res["arms"] = agg

    def flat(name, field):
        return [_num(r[field]) for rs in all_rows[name].values() for r in rs if field in r]

    def seed_delta(name_a, name_b, field="nll"):
        return [b - a for a, b in zip(agg[name_a][f"{field}_by_seed"], agg[name_b][f"{field}_by_seed"])]

    def check_pair(hname, label, a, b):
        """★ rev 1.3：统一 **NLL 口径**（越低越好）。paired(a,b) = b − a ⇒ >0 ⇒ a 更好。
        （此前用 log-prob 口径时符号方向写反 ⇒ D33。）"""
        dd = paired(flat(a, "nll"), flat(b, "nll"))
        sd = seed_delta(a, b, "nll")
        ok = (dd["mean"] > 0 and dd["p"] < 0.05 and all(x > 0 for x in sd))
        rec(f"{hname} {label}", ok,
            f"Δ(Δnll)={dd['mean']:+.6f} t={dd['t']:.2f} p={dd['p']:.3g} "
            f"pos={dd['pos']}/{dd['n']} 逐种子={['%+.4f'%x for x in sd]} "
            f"（nll: {a}={agg[a]['nll']:.4f} vs {b}={agg[b]['nll']:.4f}）",
            tol=0.0, value=dd["mean"])
        return dd, sd

    # ===== ★【DIR-18】判据 H1–H13（卡片 rev 1.2 冻结，2026-11-05）=====
    MAIN_H = "W3_hebb_full_eta7"          # ★ 主 Hebbian 臂（rev 1.1 修订 55：η=1e-7 稳定）

    def check_pair_th(hname, label, a, b, thresh=0.0, pmax=0.05):
        """同 DIR-17 的 NLL 口径（paired(a,b) = b − a ⇒ >0 ⇒ a 更好），另加**幅度阈值**。"""
        dd = paired(flat(a, "nll"), flat(b, "nll"))
        sd = seed_delta(a, b, "nll")
        ok = (dd["mean"] > thresh and dd["p"] < pmax and all(x > 0 for x in sd))
        rec(f"{hname} {label}", ok,
            f"Δ(Δnll)={dd['mean']:+.6f}（阈值 > {thresh}） t={dd['t']:.2f} p={dd['p']:.3g} "
            f"pos={dd['pos']}/{dd['n']} 逐种子={['%+.4f' % x for x in sd]} "
            f"（nll: {a}={agg[a]['nll']:.4f} vs {b}={agg[b]['nll']:.4f}）",
            tol=thresh, value=dd["mean"])
        return dd, sd

    def mean_field(name, field):
        v = [r[field] for rs in all_rows[name].values() for r in rs
             if field in r and not (isinstance(r[field], float) and math.isnan(r[field]))]
        return (float(np.mean(v)) if v else float("nan")), len(v)

    def one_sample_test(name, field, thresh=0.05):
        """单样本 t 检验：均值 > thresh 且 p < 0.05。"""
        v = [r[field] for rs in all_rows[name].values() for r in rs
             if field in r and not (isinstance(r[field], float) and math.isnan(r[field]))]
        if not v:
            rec(f"{field} 单样本检验", False, "无有效样本", value=0.0)
            return False, float("nan"), 0
        m = float(np.mean(v)); sd = float(np.std(v, ddof=1)); n = len(v)
        t = m / (sd / math.sqrt(n)) if sd > 0 else 0.0
        pos = sum(1 for x in v if x > 0)
        p = 0.0 if pos == n else 2.0 * (1.0 - _norm_cdf(abs(t)))
        ok = (m > thresh) and (p < 0.05)
        return ok, m, p

    # —— H1〔天花板可复现〕W0_verbatim 相对 base
    d1, s1 = check_pair_th("H1", "〔★ 天花板可复现〕W0_verbatim（逐字存储）显著优于 base",
                            "W0_verbatim", "E5_base", thresh=4.0, pmax=1e-6)
    # —— H2′〔★ 写规则的任务贡献〕主臂相对 nowrite 对照（修订 56）
    d2, s2 = check_pair_th("H2'", "〔★ 写规则的任务贡献〕主臂显著优于 nowrite 对照（W 从不更新）",
                            MAIN_H, "Wnowrite_hebb_cue_train", thresh=0.5, pmax=0.05)
    # —— H2''〔★ 可用配置下的写规则贡献〕ortho 主臂显著优于 ortho nowrite 对照（修订 64）
    d2b, s2b = check_pair_th("H2''", "〔★ 可用配置下的写规则贡献〕ortho 主臂显著优于 ortho nowrite 对照",
                             "W5b_hebb_full_ortho_eta1", "Wnowrite_ortho_eta1", thresh=0.5, pmax=0.05)
    # —— H3〔★ 写规则是瓶颈〕W0 − 主臂 ≥ 1.0 nats
    d3, s3 = check_pair_th("H3", "〔★ 写规则是瓶颈〕W0_verbatim 显著优于主 Hebbian 臂",
                            "W0_verbatim", MAIN_H, thresh=1.0, pmax=0.05)
    # —— H4′〔★ 值映射是瓶颈〕W4b（写侧自监督）− 主臂 ≥ 0.5 nats（修订 53）
    d4, s4 = check_pair_th("H4'", "〔★ 值映射 P_v 是瓶颈〕W4b（写侧自监督更新 P_v）显著优于主臂",
                            "W4b_hebb_full_pvw", MAIN_H, thresh=0.5, pmax=0.05)
    # —— H5〔地址与写规则正交〕`addr_slot` ≥ 0.95（★ 修订 51/52：槽空间口径）
    heb_arms = [k for k in ("W3_hebb_full_eta7", "W3_hebb_full_eta8",
                             "W6_hebb_full_renorm_eta2", "W6b_hebb_full_renorm_eta1",
                             "W5_hebb_full_ortho_eta2", "W5b_hebb_full_ortho_eta1",
                             "Wnowrite_hebb_cue_train", "W4b_hebb_full_pvw",
                             "Wnowrite_ortho_eta1",
                             "W7_hebb_full_gammalock") if k in agg]
    adr_s = {k: agg[k]["addr_slot"] for k in heb_arms}
    adr_w = {k: agg[k]["addr"] for k in heb_arms}
    ok5 = all((v is not None and v >= 0.95) for v in adr_s.values())
    rec("H5 〔地址与写规则正交〕Hebbian 臂的**地址准确率（槽空间口径）** ≥ 0.95", ok5,
        "槽空间：" + "、".join(f"{k.split('_hebb')[0]}={adr_s[k]}" for k in heb_arms) +
        "；写入序口径（★ DIR-17 协议下**必然 ≈ 0**，回绕所致 ⇒ 非缺陷）：" +
        "、".join(f"{k.split('_hebb')[0]}={adr_w[k]}" for k in heb_arms),
        tol=0.95, value=min(v for v in adr_s.values() if v is not None))
    # —— H6〔写保真度〕主臂真值槽 fid > 0.05
    ok6, m6, p6 = one_sample_test(MAIN_H, "fid_true", thresh=0.05)
    rec("H6 〔★ 写保真度〕主臂**真值槽**的 cos(W k_j, v_j) 显著 > 0（写有做功）", ok6,
        f"均值={m6:.6f}（阈值 0.05） p≈{p6:.3g}；选中槽口径 fid={agg[MAIN_H]['fid']}",
        tol=0.05, value=m6)
    # —— H8〔★ 键几何归一 ⇒ 写规则可行〕W6（renorm）fid > 0 且任务收益 > 0（修订 54）
    ok8a, m8a, p8a = one_sample_test("W6b_hebb_full_renorm_eta1", "fid_true", thresh=0.05)
    d8, s8 = check_pair_th("H8a", "〔★ 键尺度归一 ⇒ 写规则有内容收益〕W6b（renorm, η=1e-1）优于 base",
                            "W6b_hebb_full_renorm_eta1", "E5_base", thresh=0.0, pmax=0.05)
    ok8 = ok8a and (d8["mean"] > 0.5) and (d8["p"] < 0.05)
    rec("H8 〔★ 键几何归一 ⇒ 写规则可用〕W6b（renorm，‖k‖²=5）的 fid_true > 0.05 且任务收益 > 0.5 nats",
        ok8,
        f"fid_true(renorm η=1e-1)={m8a:.6f}（阈值 0.05） p≈{p8a:.3g}；"
        f"Δ(Δnll) vs base={d8['mean']:+.6f}（阈值 > 0.5） p={d8['p']:.3g} "
        f"逐种子={['%+.4f' % x for x in s8]}",
        tol=0.05, value=m8a)
    # —— H8b〔★ 正交键臂（η 档 2）〕：与 H8 同构，键几何 = ortho
    ok8b, m8b, p8b = one_sample_test("W5b_hebb_full_ortho_eta1", "fid_true", thresh=0.05)
    d8b, s8b = check_pair_th("H8b", "〔★ 正交键 ⇒ 写规则有内容收益〕W5b（ortho, η=1e-1）优于 base",
                             "W5b_hebb_full_ortho_eta1", "E5_base", thresh=0.0, pmax=0.05)
    rec("H8b 〔★ 正交键 ⇒ 写规则可用〕W5b（ortho，‖k‖²=5）的 fid_true > 0.05 且任务收益 > 0.5 nats",
        (ok8b and d8b["mean"] > 0.5 and d8b["p"] < 0.05),
        f"fid_true(ortho η=1e-1)={m8b:.6f}（阈值 0.05） p≈{p8b:.3g}；"
        f"Δ(Δnll) vs base={d8b['mean']:+.6f}（阈值 > 0.5） p={d8b['p']:.3g} "
        f"逐种子={['%+.4f' % x for x in s8b]}",
        tol=0.05, value=m8b)
    # —— H8c〔★ 归一化 vs 原尺度：键尺度修正是必要条件？〕W6b（renorm）优于 W3（linear 原尺度）
    d8c, s8c = check_pair_th("H8c", "〔★ 键尺度归一 ⇒ 优于装置原尺度〕W6b（renorm）优于 W3（linear η=1e-7）",
                             "W6b_hebb_full_renorm_eta1", "W3_hebb_full_eta7", thresh=0.0, pmax=0.05)
    rec("H8c 〔★ 键尺度归一 ⇒ 优于装置原尺度〕W6b（renorm）显著优于 W3（linear η=1e-7）",
        (d8c["mean"] > 0 and d8c["p"] < 0.05),
        f"Δ(Δnll)={d8c['mean']:+.6f}（阈值 > 0） p={d8c['p']:.3g} "
        f"逐种子={['%+.4f' % x for x in s8c]}",
        tol=0.0, value=d8c["mean"])
    # —— H9〔★ 跨方向复核 D-24〕主臂 addr_slot < γ 锁定臂（修订 55）
    a_main = mean_field(MAIN_H, "addr_slot")[0]
    a_gam = mean_field("W7_hebb_full_gammalock", "addr_slot")[0]
    ok9 = (a_main < a_gam)
    rec("H9 〔★ 跨方向复核 D-24〕训练 γ 破坏寻址：主臂 addr_slot < γ 锁定臂", ok9,
        f"addr_slot(主臂，γ 可训)={a_main:.4f} vs addr_slot(γ 锁定)={a_gam:.4f} "
        f"（差 {a_main - a_gam:+.4f}；正=复现 D-24；★ 口径 = **槽空间**（精确映射，D54））",
        tol=0.0, value=a_main - a_gam)
    # —— H10〔内容关联存在〕主臂 cos(y_read, v_true) > 0.05
    ok10, m10, p10 = one_sample_test(MAIN_H, "cos_y_vtrue", thresh=0.05)
    rec("H10 〔★ 内容关联存在〕主臂的 cos(y_read, v_true) 显著 > 0（读出含目标值）", ok10,
        f"均值={m10:.6f}（阈值 0.05） p≈{p10:.3g}", tol=0.05, value=m10)
    # —— H11〔写保真度（真值槽）〕同 H6 的复述（保留原判据编号以对齐卡片）
    ok11, m11, p11 = one_sample_test(MAIN_H, "fid_true", thresh=0.05)
    rec("H11 〔写保真度（真值槽）〕主臂的 cos(W k_j, v_j) 显著 > 0", ok11,
        f"均值={m11:.6f} p≈{p11:.3g}", tol=0.05, value=m11)
    # —— H12〔地址独立于写规则〕即使 H10/H11 FAIL，主臂 addr_slot 仍 ≥ 0.95
    ok12 = (a_main == a_main) and (a_main >= 0.95)
    rec("H12 〔地址独立于写规则〕主臂 addr_slot ≥ 0.95（即使内容判据 FAIL）", ok12,
        f"addr_slot(主臂)={a_main:.4f}（阈值 0.95）", tol=0.95, value=a_main)
    # —— H13〔★ 收益的内容性〕主臂 Δcode 显著大于 Δctrl（**交换性对照**）
    #   实现：逐文档配对。Δcode = NLL_base(code) − NLL_arm(code)；
    #         Δctrl = NLL_base(ctrl) − NLL_arm(ctrl)；
    #         超额 = Δcode − Δctrl（**正** = 该臂比「稀有词先验」额外地更懂真码）。
    def excess_over_prior(arm):
        """★ 缺陷修复 D57/D58（修订 67）：
        返回 **超额 = Δcode − Δctrl**（正 = 该臂比「稀有词先验」额外地更懂真码）。

        卡片（§2.2「H13 的定义」）的**逐字**口径：
            Δcode = NLL_base(code) − NLL_arm(code)
            Δctrl = NLL_base(ctrl) − NLL_arm(ctrl)
            超额  = Δcode − Δctrl

        ★ D57：原实现把全部计算放进**内层函数 `pr(name)`** 却从未调用它，
          外层函数**直接落空返回 None** ⇒ 解包 `ex, p, n = excess_over_prior(...)`
          抛 `TypeError: cannot unpack non-iterable NoneType object`
          ⇒ ⇒ **H13 / H13b / H13c 三条内容性判据从未被计算**（本轮首次执行到此即崩溃）。
        ★ D58：原 `dk = rb["nll"] − ra["nll_ctrl"]` 把 **base 在「码」上的 NLL**
          与 **arm 在「对照码」上的 NLL** 相减 ⇒ 口径**与卡片不一致**（纪律 ⑳）。
          修法：`dk = rb["nll_ctrl"] − ra["nll_ctrl"]`（base 与 arm 都取「对照码」的 NLL）。

        ★ **判据本身（阈值 +0.5 nats / p<0.05 / 逐种子同向）一字未改**；
          本次只修**实现口径**，使其逐条对齐卡片文字。
        """
        out = []
        for seed in seeds:
            rs_a = all_rows.get(arm, {}).get(seed, [])
            rs_b = all_rows.get("E5_base", {}).get(seed, [])
            n = min(len(rs_a), len(rs_b))
            for ra, rb in zip(rs_a[:n], rs_b[:n]):
                if ("nll_ctrl" in ra and "nll" in ra
                        and "nll" in rb and "nll_ctrl" in rb):
                    dc = rb["nll"] - ra["nll"]              # Δcode = NLL_base(code) − NLL_arm(code)
                    dk = rb["nll_ctrl"] - ra["nll_ctrl"]   # Δctrl = NLL_base(ctrl) − NLL_arm(ctrl)  ★D58
                    out.append(dc - dk)                    # 超额 = Δcode − Δctrl
        out = [x for x in out if math.isfinite(x)]
        if not out:
            return float("nan"), 1.0, 0
        m = float(np.mean(out))
        sd = float(np.std(out, ddof=1)) if len(out) > 1 else 0.0
        t = m / (sd / math.sqrt(len(out))) if sd > 0 else 0.0
        p = 2.0 * (1.0 - _norm_cdf(abs(t)))
        return m, p, len(out)
    ex_main, ex_p, ex_n = excess_over_prior(MAIN_H)
    ex_verb, ex_pv, ex_nv = excess_over_prior("W0_verbatim")
    ex_ren, ex_pr, ex_nr = excess_over_prior("W6b_hebb_full_renorm_eta1")
    ex_ort, ex_po, ex_no = excess_over_prior("W5b_hebb_full_ortho_eta1")
    ex_now, ex_pn, ex_nn = excess_over_prior("Wnowrite_hebb_cue_train")
    ex_nowO, ex_pnO, ex_nnO = excess_over_prior("Wnowrite_ortho_eta1")
    ok13 = (ex_main == ex_main) and (ex_main > 0.5) and (ex_p < 0.05)
    rec("H13 〔★ 收益的内容性〕主臂的 Δcode 显著大于 Δctrl（交换性对照：同池、均匀抽样、从未写入）",
        ok13,
        f"超额(Δcode−Δctrl)={ex_main:+.4f} nats（阈值 > 0.5） p≈{ex_p:.3g} n={ex_n}；"
        f"对照组 verbatim={ex_verb:+.4f}（p≈{ex_pv:.3g} n={ex_nv}）；"
        f"renorm η=1e-1={ex_ren:+.4f}（p≈{ex_pr:.3g} n={ex_nr}）；"
        f"ortho η=1e-1={ex_ort:+.4f}（p≈{ex_po:.3g} n={ex_no}）；"
        f"nowrite(linear)={ex_now:+.4f}（p≈{ex_pn:.3g} n={ex_nn}）；"
        f"nowrite(ortho)={ex_nowO:+.4f}（p≈{ex_pnO:.3g} n={ex_nnO}）",
        tol=0.5, value=ex_main)
    # —— H13b〔★ 归一化写规则的内容性〕W6b（renorm）的超额 > 0.5
    ok13b = (ex_ren == ex_ren) and (ex_ren > 0.5) and (ex_pr < 0.05)
    rec("H13b 〔★ 归一化写规则的内容性〕W6b（renorm, η=1e-1）的 Δcode 显著大于 Δctrl",
        ok13b,
        f"超额(Δcode−Δctrl)={ex_ren:+.4f} nats（阈值 > 0.5） p≈{ex_pr:.3g} n={ex_nr}；"
        f"对照 nowrite(linear)={ex_now:+.4f}（p≈{ex_pn:.3g} n={ex_nn}）；"
        f"nowrite(ortho)={ex_nowO:+.4f}（p≈{ex_pnO:.3g} n={ex_nnO}）；"
        f"ortho η=1e-1={ex_ort:+.4f}（p≈{ex_po:.3g}）",
        tol=0.5, value=ex_ren)
    # —— H13c〔★ 正交键写规则的内容性〕W5b（ortho）的超额 > 0.5
    ok13c = (ex_ort == ex_ort) and (ex_ort > 0.5) and (ex_po < 0.05)
    rec("H13c 〔★ 正交键写规则的内容性〕W5b（ortho, η=1e-1）的 Δcode 显著大于 Δctrl",
        ok13c,
        f"超额(Δcode−Δctrl)={ex_ort:+.4f} nats（阈值 > 0.5） p≈{ex_po:.3g} n={ex_no}；"
        f"对照 nowrite(ortho)={ex_nowO:+.4f}（p≈{ex_pnO:.3g} n={ex_nnO}）；"
        f"nowrite(linear)={ex_now:+.4f}（p≈{ex_pn:.3g} n={ex_nn}）",
        tol=0.5, value=ex_ort)

    # —— 报告项：归因对照臂的读数（修订 56）
    for arm in ("Wnowrite_hebb_cue_train", "Wmisaddr_hebb_cue_train",
                "W4_hebb_full_trainPv", "W7_hebb_full_gammalock",
                "W4b_hebb_full_pvw", "W6_hebb_full_renorm_eta2",
                "W6b_hebb_full_renorm_eta1", "W5_hebb_full_ortho_eta2",
                "W5b_hebb_full_ortho_eta1"):
        if arm in agg:
            ex, p_, n_ = excess_over_prior(arm)
            print(f"  [归因报告] {arm:26s} nll={agg[arm]['nll']:.4f} addr_slot={agg[arm]['addr_slot']} "
                  f"fid={agg[arm]['fid']} fid_true={agg[arm]['fid_true']} "
                  f"cos_y_vtrue={agg[arm]['cos_y_vtrue']} 超额(Δcode−Δctrl)={ex:+.4f} (p={p_:.3g})",
                  flush=True)

    # 发散对照报告项（D43 的正面证据）
    for arm in ("W3_hebb_full_eta7", "W3_hebb_full_eta8",
                "W6_hebb_full_renorm_eta2", "W6b_hebb_full_renorm_eta1",
                "W5_hebb_full_ortho_eta2", "W5b_hebb_full_ortho_eta1"):
        if arm in agg:
            nf = int(np.sum([1 for rs in all_rows[arm].values() for r in rs
                             if math.isfinite(r["nll"])]))
            tot = int(sum(len(rs) for rs in all_rows[arm].values()))
            print(f"  [发散报告] {arm:26s} 有限样本 {nf}/{tot}；nll={agg[arm]['nll']:.4f}；"
                  f"addr_slot={agg[arm]['addr_slot']}；fid={agg[arm]['fid']}", flush=True)

    # A3' 饱和预检（纪律 ㉔）：W0 / 主臂 / W5b 三臂的 NLL 极差 ≥ 阈值
    trio = [agg[k]["nll"] for k in ("W0_verbatim", MAIN_H, "W5b_hebb_full_ortho_eta1")
            if k in agg]
    trio_ok = len(trio) == 3 and all(math.isfinite(x) for x in trio)
    spread = (float(max(trio) - min(trio)) if trio_ok else 0.0)
    res["meta"]["saturation"] = {"trio_nll": trio, "spread": spread, "eps": SAT_EPS}
    rec("A3' 饱和预检：W0/主臂/W5b 的 NLL 极差 ≥ eps ⇒ 任务具备分辨率（否则判据无信息量）",
        trio_ok and spread >= SAT_EPS,
        f"nll(W0)={trio[0]:.4f} nll(主臂)={trio[1]:.4f} nll(W5b)={trio[2]:.4f}；"
        f"极差={spread:.4f}（阈值 {SAT_EPS}）", tol=SAT_EPS, value=spread)

    passed = sum(1 for c in res["checks"] if c["passed"])
    res["summary_pass"] = {"passed": passed, "total": len(res["checks"])}
    print(f"\n=== 判据通过 {passed}/{len(res['checks'])} ===", flush=True)
    out = HERE.parent / ("R28_smoke_results.json" if args.smoke else "R28_results.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
