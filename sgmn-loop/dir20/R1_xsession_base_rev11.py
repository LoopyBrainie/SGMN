#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import argparse
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
from sgmn_module_hebb_v4 import SGMNStack as SGS, grad_report   # noqa: E402
import sgmn_module_hebb_v4 as _MOD   # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
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

# ★★ 缺陷 D79：派生脚本的 `res["dir"]` 仍是 **"DIR-18"** ⇒ 结果 JSON 会被标成 DIR-18 ⇒ 改。
res = {"dir": "DIR-20", "round": "R1", "checks": [], "arms": {}, "meta": {}}
# ★★ 缺陷 D78：**增量文件与结果文件必须按方向命名空间隔离** ——
#   原实现写 `HERE.parent/"R28_results.json"` 与 `HERE.parent/".ops"/f"formal_incremental_{tag}.json"`
#   ⇒ `HERE.parent` = `~/sgmn-loop/` ⇒ **DIR-18 与 DIR-19 写的是同一个文件** ⇒ ⇒ 实测：
#   DIR-19 的正式轮**静默复用了 DIR-18 的 `E5_base` 行**（nll 15.3260 / 14.9949）
#   ⇒ ⇒ 基准臂被**跨方向污染** ⇒ ⇒ H1′/H1′b 的参照值不可信。
#   修法：文件名一律带方向与轮次前缀（`R19_*` / `formal_incremental_DIR19_*`）。
_DIR_TAG = "DIR20"


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


# ★ DIR-19 臂表（11 臂；卡片 rev 1.2 §9.3）
#   字段：(name, query_mode, mem_kind, eta, train, train_pv, key_mode, variant,
#          base_mode, kl_lambda, steps_div, extra)
#   base_mode: none | all | lowrank | layers
ARMS = [
    # ① 基线
    ("E5_base",          None,          "explicit_kv", None,  False, False, "ortho", "normal",
     "none", 0.0, 1, "base_only"),          # 无记忆（基座基线）
    ("E5_base_mem",      "cue",         "explicit_kv", None,  False, False, "ortho", "normal",
     "none", 0.0, 1, "zero_train_mem"),     # 挂记忆但零训练（退化下界）
    # ② 收益对照族（N1：基座全冻结 + 读取侧训练）
    # ★★★ 修订 19-I（rev 1.5）：**记忆臂全部切到 DIR-18 已确立的「可用写配置」**
    #   —— `key_mode="ortho"` + `η=1e-1`（DIR-18 的 `W5b_hebb_full_ortho_eta1`：
    #   `fid_true`=0.9822 / `hit`=0.9278 / 内容判据 H14=**+1.7189** PASS）。
    #   ★ 理由：原臂表用 `linear` + `η=1e-7` ⇒ **写规则惰性**（smoke 实测 `fid_true`≈−0.0004）
    #   ⇒ ⇒ 记忆**取不回内容** ⇒ ⇒ 内容判据 H2′★ 测的是**空记忆**（smoke 实测 Δ(Δnll)=**−0.000028** ≈ 0）
    #   ⇒ ⇒ 若不修，**整个方向测不到任何东西**（这正是 DIR-18 的 D51 在新方向里的复现）。
    ("N1_frozen_base",   "cue",         "hebbian_full", 1e-1, True, False, "ortho", "normal",
     "none", 0.0, 1, ""),
    ("N1_nowrite",       "cue",         "hebbian_full", 1e-1, True, False, "ortho", "nowrite",
     "none", 0.0, 1, ""),
    ("N1_content_wrong", "cue",         "hebbian_full", 1e-1, True, False, "ortho", "content_wrong",
     "none", 0.0, 1, ""),
    ("N1_earlystop",     "cue",         "hebbian_full", 1e-1, True, False, "ortho", "normal",
     "none", 0.0, 4, ""),
    # ★★ 修订 19-I 新增：`N1_linear_ctrl`（**报告项**）—— 原惰性写配置（linear + η=1e-7）
    #   ⇒ 与 `N1_frozen_base`（ortho + η=1e-1）构成**写配置对照**
    #   ⇒ ⇒ 对 DIR-18 的 D51（键尺度标定口径错 ⇒ 写规则惰性）做**跨方向复现**。
    ("N1_linear_ctrl",   "cue",         "hebbian_full", 1e-7, True, False, "linear", "normal",
     "none", 0.0, 1, "linear_ctrl"),
    # ③ 保护旋钮族
    ("N2_kl",            "cue",         "hebbian_full", 1e-1, True, False, "ortho", "normal",
     "all", 1.0, 1, "kl"),
    ("N3_lowrank",       "cue",         "hebbian_full", 1e-1, True, False, "ortho", "normal",
     "lowrank", 0.0, 1, ""),
    ("N4_low_layers",    "cue",         "hebbian_full", 1e-1, True, False, "ortho", "normal",
     "layers", 0.0, 1, ""),
    ("N5_disjoint",      "cue",         "hebbian_full", 1e-1, True, False, "ortho", "normal",
     "none", 0.0, 1, "disjoint_train"),
    # ★★ 缺陷 D74a：原 `train=False` ⇒ 该臂与 `E5_base` **完全相同**
    #   ⇒ 「排除『任何联合训练都会让基座退化』」这条判据**落空** ⇒ 改为 `True`
    ("E5_base_cap",      None,          "explicit_kv", None,  True,  False, "ortho", "normal",
     "all", 0.0, 1, "base_only"),
]


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
    n_eval = 12 if args.smoke else N_EVAL
    n_train = 40 if args.smoke else N_TRAIN
    seeds = SEEDS[:2] if args.smoke else SEEDS

    print("=== DIR-17 R1（脚本 v1.1 · 卡片 rev 1.3 任务构造）：可寻址读取 + 读侧训练 ===", flush=True)
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

    # ---------------- ★ Stage A 硬闸（纪律 ①）：任一 FAIL ⇒ **不得开主循环**
    #   （卡片 rev 1.2 §2.1：「任一 FAIL 不得开正式轮」；此前靠外部脚本把关，
    #     现在装置内部也设闸 ⇒ 无论谁启动本脚本，都不可能在 Stage A 失败时产出正式轮结果。）
    _failed = [c["name"] for c in res["checks"] if not c["passed"]]
    print(f"\n[Stage A 闸] 共 {len(res['checks'])} 项，FAIL {len(_failed)} 项"
          + (f"：{_failed}" if _failed else " ⇒ 全部通过"), flush=True)
    if _failed:
        print("[Stage A 闸] ★ 有 FAIL ⇒ **中止，不开主循环**（纪律 ①）", flush=True)
        out = HERE.parent / ("R20_smoke_results.json" if args.smoke else "R20_results.json")   # ★ D78
        res["aborted"] = {"stage_a_failed": _failed}
        with open(out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        raise SystemExit(3)

    # ---------------- 主循环
    print("\n[主循环]", flush=True)
    print(f"\n[规格] smoke={args.smoke}  n_train={n_train}  n_eval={n_eval}  seeds={seeds}  臂数={len(ARMS)}",
          flush=True)
    _tag = "smoke" if args.smoke else "full"
    all_rows = _load_incremental(_tag)  # ★ 续跑：跳过已完成的 (臂, 种子) 组合
    if all_rows:
        _done = sum(len(v) for v in all_rows.values())
        print(f"\n[主循环] ★ 复用增量结果：{len(all_rows)} 臂 / {_done} 个 (臂,种子) 组合已完成 ⇒ 跳过",
              flush=True)
    for seed in seeds:
        # ★★【DIR-19】`tr/ev/ctrl_ids` 在**臂内**按语料切分重建（旋钮 N5 要用前一半语料）
        for (name, qm, mkind, aeta, train, train_pv, kmode, variant,
             base_mode, kl_lam, steps_div, extra) in ARMS:
            # ★★【增量落盘 + 续跑】跳过已完成的 (臂, 种子) 组合
            if name in all_rows and seed in all_rows[name]:
                _rr = all_rows[name][seed]
                print(f"  seed={seed} {name:26s} (增量复用) nll={np.mean([_num(r['nll']) for r in _rr]):.4f} "
                      f"ppl={np.mean([_num(r.get('ppl')) for r in _rr]):.4f}", flush=True)
                continue
            # ★★【DIR-19】语料切分：N5 用**前一半**训练 ⇒ 探针（后一半）与训练**逐位不相交**
            _CORP_SPLIT["half"] = "first" if extra == "disjoint_train" else "all"
            tr, ev = build_docs(tok, seed, n_train, n_eval)
            ctrl_ids = build_ctrl_ids(tok, seed, tr, ev)
            # ★★【DIR-19】**基座还原**：可训基座的臂必须从**原始基座**开始
            #   （否则前一个臂的漂移会污染下一个臂 ⇒ 混淆「保护」的效果）
            if base_mode != "none":
                model.load_state_dict(PRISTINE)
            _probe_st = None
            if qm is None:
                # ★★ 缺陷 D74：无记忆臂（`E5_base` / `E5_base_cap`）**必须**走这一支。
                #   原判据 `qm is None and base_mode == "none"` ⇒ `E5_base_cap`（qm=None, base_mode="all"）
                #   落进「挂记忆」分支 ⇒ `make_stack(qm=None)` **仍然挂钩** ⇒ 读阶段 `mem.read` 抛
                #   `ValueError: None` ⇒ 第 11 臂崩溃。
                if train:
                    # ★ `E5_base_cap`：**只训基座**（记忆钩子**全程摘除**）⇒ 纯基座的同分布对照训练
                    _st0 = make_stack(model, d, "cue", seed=seed, mem_kind=mkind, eta=aeta,
                                      key_mode=kmode, variant=variant)
                    _st0.suspend_hooks()          # ★ 全程摘除 ⇒ 训练时**不注入记忆**
                    _ref0 = None
                    if kl_lam > 0.0:
                        _ref0 = AutoModelForCausalLM.from_pretrained(
                            MODEL, dtype=torch.float32).to(DEV).eval()
                        for _q in _ref0.parameters():
                            _q.requires_grad_(False)
                    traj, rep = train_arm(_st0, tr, "cue", train_pv=train_pv, variant=variant,
                                          base_mode=base_mode, kl_lambda=kl_lam, model_ref=_ref0,
                                          lowrank_projs=None, base_only=True,
                                          n4_layers=N4_LAYERS)
                    if _ref0 is not None:
                        del _ref0
                    _st0.remove(); del _st0
                    for _q in model.parameters():
                        _q.grad = None
                    torch.cuda.empty_cache()
                # 纯基座评测（**不挂记忆**）
                rows = eval_base(model, ev, ctrl_ids=ctrl_ids)
                _ppl = probe_ppl(model, None, ppl_chunks)
                rows = [dict(r, ppl=_ppl, dkv=0.0) for r in rows]
            else:
                st = make_stack(model, d, qm, seed=seed, mem_kind=mkind, eta=aeta,
                                key_mode=kmode, variant=variant)
                # ★【修订 56】归因对照臂：地址表照写、W 从不更新
                if variant == "nowrite":
                    make_nowrite_variant(st)
                # ★★【DIR-19】低秩子空间投影（旋钮 N3）⇒ 只算一次、缓存
                _projs = None
                if base_mode == "lowrank":
                    _projs = lowrank_cache
                    if _projs is None:
                        if _LRCACHE["proj"] is None or _LRCACHE["r"] != 8:
                            print("  [N3] ★ 计算低秩投影（SVD，一次性，较慢）…", flush=True)
                            _LRCACHE["proj"] = st.lowrank_projections(r=8)
                            _LRCACHE["r"] = 8
                        _projs = _LRCACHE["proj"]
                        lowrank_cache = _projs
                # ★★【DIR-19】$D_{kv}$ 的**训练前**分位
                _q_before = kv_drift(st, ppl_chunks[:4]) if train else None
                if train:
                    # ★★ 缺陷 D72 的修法：KL 需要的参考模型**按臂临时**建、用完释放
                    # ★★ 缺陷 D77 的修法：**更早**释放 —— 训练前就把参考 logits 预计算到 CPU，
                    #   然后**立刻** `del` 参考模型 ⇒ ⇒ 训练全程显存里**没有**参考模型（省 2.0 GB）。
                    _ref_lg = None
                    if base_mode != "none" and kl_lam > 0.0:
                        _ref = AutoModelForCausalLM.from_pretrained(
                            MODEL, dtype=torch.float32).to(DEV).eval()
                        for _q in _ref.parameters():
                            _q.requires_grad_(False)
                        _n_ref = (len(tr) // steps_div) if steps_div > 1 else len(tr)
                        _ref_lg = ref_logits_at(_ref, tr[:_n_ref])   # ★ 预计算（CPU）
                        del _ref                                      # ★ 立即释放 2.0 GB
                        torch.cuda.empty_cache()
                    _steps = (len(tr) // steps_div) if steps_div > 1 else None
                    traj, rep = train_arm(st, tr, qm, train_pv=train_pv, variant=variant,
                                          base_mode=base_mode, kl_lambda=kl_lam,
                                          model_ref=None, lowrank_projs=_projs,
                                          base_only=(extra == "base_only"),
                                          n4_layers=N4_LAYERS, ref_logits=_ref_lg)
                    del _ref_lg
                    torch.cuda.empty_cache()
                else:
                    traj, rep = [], {}
                rows = eval_arm(st, ev, ctrl_ids=ctrl_ids)
                # ★★【DIR-19】$D_{kv}$ 的**训练后**分位 ⇒ 漂移
                _q_after = kv_drift(st, ppl_chunks[:4]) if _q_before is not None else None
                _dkv = 0.0
                if _q_before is not None and _q_after is not None:
                    _dkv = float((_q_after - _q_before).abs().mean())
                # ★★【DIR-19】基座保持探针（挂记忆 = 加装后的实际状态）
                _probe_st = None if extra == "base_only" else st
                _ppl = probe_ppl(model, _probe_st, ppl_chunks)
                rows = [dict(r, ppl=_ppl, dkv=_dkv) for r in rows]
                # ★★【DIR-19】基座漂移量（‖θ−θ₀‖）与低秩子空间残差
                _drift = 0.0
                if base_mode != "none":
                    _sd = PRISTINE
                    for _n, _q in model.named_parameters():
                        if _n in _sd:
                            _drift = max(_drift, float((_q.detach() - _sd[_n].to(_q.device)).norm()))
                _res_sub = st.subspace_residual(_projs) if (base_mode == "lowrank" and _projs) else 0.0
                rows = [dict(r, base_drift=_drift, sub_res=_res_sub,
                             kl_last=(rep.get("_kl_hist") or [0.0])[-1]) for r in rows]
                st.remove()
                # ★ 缺陷 D72：基座的**梯度缓冲**（2 GB）会跨臂累积 ⇒ 每臂结束必须清
                for _q in model.parameters():
                    _q.grad = None
                torch.cuda.empty_cache()
            all_rows.setdefault(name, {})[seed] = rows
            # ★ 增量落盘（每臂）：中断不丢已完成臂
            try:
                _incf = HERE.parent / ".ops" / f"formal_incremental_{_DIR_TAG}_{_tag}.json"   # ★ D78
                _incf.parent.mkdir(parents=True, exist_ok=True)
                with open(_incf, "w") as _f:
                    json.dump({"rows": _jsonable(all_rows)}, _f, indent=1, ensure_ascii=False)
            except Exception as _e:
                print(f"[增量落盘失败] {_e}", flush=True)
            print(f"  seed={seed} {name:26s} nll={np.mean([r['nll'] for r in rows]):.4f} "
                  f"lp={np.mean([r['lp'] for r in rows]):+.4f} "
                  f"hit={np.nanmean([r.get('hit', np.nan) for r in rows]):.4f} "
                  f"addr={np.nanmean([r.get('addr', np.nan) for r in rows]):.4f} "
                  f"addr_slot={np.nanmean([r.get('addr_slot', np.nan) for r in rows]):.4f} "
                  f"fid={np.nanmean([r.get('fid', np.nan) for r in rows]):.4f} "
                  f"fid_true={np.nanmean([r.get('fid_true', np.nan) for r in rows]):.4f} "
                  f"cos_y_vtrue={np.nanmean([r.get('cos_y_vtrue', np.nan) for r in rows]):.4f} "
                  f"ppl={np.nanmean([_num(r.get('ppl')) for r in rows]):.4f} "
                  f"dkv={np.nanmean([_num(r.get('dkv')) for r in rows]):.4f} "
                  f"base_drift={np.nanmean([_num(r.get('base_drift')) for r in rows]):.4f} "
                  f"kl_last={np.nanmean([_num(r.get('kl_last')) for r in rows]):.4f}", flush=True)

    # ---------------- 汇总
    print("\n[汇总与判据]", flush=True)
    agg = {}
    for name, per_seed in all_rows.items():
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
        ppls = [_num(r["ppl"]) for rs in per_seed.values() for r in rs if "ppl" in r]
        dkvs = [_num(r["dkv"]) for rs in per_seed.values() for r in rs if "dkv" in r]
        bdrs = [_num(r["base_drift"]) for rs in per_seed.values() for r in rs if "base_drift" in r]
        kls = [_num(r["kl_last"]) for rs in per_seed.values() for r in rs if "kl_last" in r]
        agg[name] = {"lp": float(np.mean(lps)), "nll": float(np.mean(nlls)),
                     "fid": (float(np.nanmean(fids)) if fids else None),
                     "fid_true": (float(np.nanmean(fidt)) if fidt else None),
                     "cos_y_vtrue": (float(np.nanmean(cyv)) if cyv else None),
                     "nll_ctrl": (float(np.nanmean(nllc)) if nllc else None),
                     "nll_dcode": (float(np.nanmean(nllD)) if nllD else None),
                     "addr": (float(np.nanmean(addrs)) if addrs else None),
                     "addr_slot": (float(np.nanmean(adrss)) if adrss else None),
                     # ★★【DIR-19】基座保持 / 漂移
                     "ppl": (float(np.nanmean(ppls)) if ppls else None),
                     "dkv": (float(np.nanmean(dkvs)) if dkvs else None),
                     "base_drift": (float(np.nanmean(bdrs)) if bdrs else None),
                     "kl_last": (float(np.nanmean(kls)) if kls else None),
                     "ppl_by_seed": [float(np.mean([_num(r.get("ppl")) for r in rs])) for rs in per_seed.values()],
                     "dkv_by_seed": [float(np.mean([_num(r.get("dkv")) for r in rs])) for rs in per_seed.values()],
                     "lp_by_seed": [float(np.mean([r["lp"] for r in rs])) for rs in per_seed.values()],
                     "nll_by_seed": [float(np.mean([r["nll"] for r in rs])) for rs in per_seed.values()],
                     "hit": (float(np.mean(hits)) if hits else None), "n": len(lps)}
        print(f"  {name:26s} nll={agg[name]['nll']:.4f} lp={agg[name]['lp']:+.4f} "
              f"hit={agg[name]['hit']} addr={agg[name]['addr']} addr_slot={agg[name]['addr_slot']} "
              f"fid={agg[name]['fid']} fid_true={agg[name]['fid_true']} "
              f"cos_y_vtrue={agg[name]['cos_y_vtrue']} ctrl={agg[name]['nll_ctrl']}", flush=True)
    res["arms"] = agg

    def flat(name, field):
        return [r[field] for rs in all_rows[name].values() for r in rs if field in r]

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

    # ══════════════════ ★★【DIR-19】判据 H1′…H7′ + H2′★ + H2′″（卡片 rev 1.2/1.3 冻结）══════════════════
    # —— 口径（写死，防漂移）——
    #   benefit(arm) = NLL(E5_base) − NLL(arm)  （NLL 口径：越低越好 ⇒ 正 = 有收益）
    #   ΔPPL(arm)   = PPL(arm) − PPL(E5_base)  （基座保持探针：valid.txt 的平均每 token NLL）
    #   retention(arm) = benefit(arm) / benefit(N1_frozen_base)
    MAIN = "N1_frozen_base"
    BASE = "E5_base"
    MEM0 = "E5_base_mem"

    def ben(name):
        if name not in agg or BASE not in agg:
            return float("nan")
        return agg[BASE]["nll"] - agg[name]["nll"]

    def dppl(name):
        if name not in agg or BASE not in agg:
            return float("nan")
        if agg[name].get("ppl") is None or agg[BASE].get("ppl") is None:
            return float("nan")
        return agg[name]["ppl"] - agg[BASE]["ppl"]

    def ben_ret(name):
        b0 = ben(MAIN)
        if not (b0 == b0) or abs(b0) < 1e-9:
            return float("nan")
        return ben(name) / b0

    def seed_delta_gen(a, b, field="nll"):
        out = []
        for sd in seeds:
            ra = all_rows.get(a, {}).get(sd, []); rb = all_rows.get(b, {}).get(sd, [])
            n = min(len(ra), len(rb))
            if n:
                out.append(float(np.mean([rb[i]["nll"] - ra[i]["nll"] for i in range(n)])))
        return out

    # —— H1′〔记忆收益〕——
    d1 = paired(flat(MAIN, "nll"), flat(MEM0, "nll"))
    sd1 = seed_delta_gen(MAIN, MEM0)
    ok1 = (d1["mean"] > 0.5) and (d1["p"] < 0.05) and all(x > 0 for x in sd1)
    rec("H1′ 〔记忆收益〕N1_frozen_base 相对挂记忆零训练(E5_base_mem)有显著收益",
        ok1, f"Δ(Δnll)={d1['mean']:+.6f}（阈值 > 0.5） t={d1['t']:.2f} p={d1['p']:.3g} "
              f"pos={d1['pos']}/{d1['n']} 逐种子={['%+.4f' % x for x in sd1]}",
        tol=0.5, value=d1["mean"])
    # —— H1′b〔记忆收益 vs 纯基座〕（报告项 + 收益保留率的基准）——
    d1b = paired(flat(MAIN, "nll"), flat(BASE, "nll"))
    sd1b = seed_delta_gen(MAIN, BASE)
    ok1b = (d1b["mean"] > 0.5) and (d1b["p"] < 0.05) and all(x > 0 for x in sd1b)
    rec("H1′b 〔记忆收益 vs 纯基座〕N1_frozen_base 相对 E5_base 有显著收益",
        ok1b, f"Δ(Δnll)={d1b['mean']:+.6f}（阈值 > 0.5） t={d1b['t']:.2f} p={d1b['p']:.3g} "
               f"逐种子={['%+.4f' % x for x in sd1b]}；benefit={ben(MAIN):+.4f} nats",
        tol=0.5, value=d1b["mean"])

    # —— H2′★〔★ 内容性 · 记忆内对照〕——
    CW = "N1_content_wrong"
    if CW in agg and MAIN in agg:
        _a = [r for rs in all_rows[MAIN].values() for r in rs if "nll" in r]
        _b = [r for rs in all_rows[CW].values() for r in rs if "nll" in r]
        _n = min(len(_a), len(_b))
        _d = [_b[i]["nll"] - _a[i]["nll"] for i in range(_n)]
        _m = float(np.mean(_d)); _sd = float(np.std(_d, ddof=1)) if len(_d) > 1 else 0.0
        _t = _m / (_sd / math.sqrt(len(_d))) if _sd > 0 else 0.0
        _pp = 2.0 * (1.0 - _norm_cdf(abs(_t)))
        _sdl = []
        for sd in seeds:
            ra = all_rows.get(MAIN, {}).get(sd, []); rb = all_rows.get(CW, {}).get(sd, [])
            _k = min(len(ra), len(rb))
            if _k:
                _sdl.append(float(np.mean([rb[i]["nll"] - ra[i]["nll"] for i in range(_k)])))
        ok2 = (_m > 0.5) and (_pp < 0.05) and all(x > 0 for x in _sdl)
        rec("H2′★ 〔★ 内容性 · 记忆内对照〕内容正确(N1_frozen_base) 显著优于 内容错误(N1_content_wrong)",
            ok2, f"Δ(Δnll)={_m:+.6f}（阈值 > 0.5） t={_t:.2f} p≈{_pp:.3g} n={len(_d)} "
                 f"逐种子={['%+.4f' % x for x in _sdl]}；"
                 f"★ **本判据无零效度问题**（两臂记忆基础设施相同 ⇒ 内容被换掉 ⇒ 差异只可能来自内容）",
            tol=0.5, value=_m)
    else:
        rec("H2′★ 〔★ 内容性 · 记忆内对照〕", False, f"缺少臂 {CW}", tol=0.5, value=float("nan"))

    # —— H2′″〔交换性对照口径的报告项〕——
    def excess_over_prior(arm):
        out = []
        for sd in seeds:
            ra = all_rows.get(arm, {}).get(sd, []); rb = all_rows.get(BASE, {}).get(sd, [])
            n = min(len(ra), len(rb))
            for i in range(n):
                if ("nll_ctrl" in ra[i] and "nll" in ra[i]
                        and "nll" in rb[i] and "nll_ctrl" in rb[i]):
                    dc = rb[i]["nll"] - ra[i]["nll"]
                    dk = rb[i]["nll_ctrl"] - ra[i]["nll_ctrl"]
                    out.append(dc - dk)
        out = [x for x in out if math.isfinite(x)]
        if not out:
            return float("nan"), 1.0, 0
        m = float(np.mean(out)); sd_ = float(np.std(out, ddof=1)) if len(out) > 1 else 0.0
        t = m / (sd_ / math.sqrt(len(out))) if sd_ > 0 else 0.0
        return m, 2.0 * (1.0 - _norm_cdf(abs(t))), len(out)
    ex_main, ex_p, ex_n = excess_over_prior(MAIN)
    rec("H2′″ 〔★ 报告项〕交换性对照口径的超额（Δcode−Δctrl）—— 本轮内容判据**不用**该口径",
        True, f"N1_frozen_base 的超额(Δcode−Δctrl)={ex_main:+.4f} nats p≈{ex_p:.3g} n={ex_n}；"
              f"★ DIR-19 的内容判据用 **H2′★（记忆内对照）** ⇒ 天然无零效度问题 ⇒ 本项仅供跨方向比较",
        tol=0.0, value=ex_main)

    # —— H3′〔保护有效：挂记忆+联合训练 相对 挂记忆零训练〕——
    _dp_main = dppl(MAIN); _dp_mem0 = dppl(MEM0)
    _d3 = _dp_main - _dp_mem0
    _sdl3 = []
    for sd in seeds:
        pa = all_rows.get(MAIN, {}).get(sd, []); pb = all_rows.get(MEM0, {}).get(sd, [])
        n = min(len(pa), len(pb))
        if n:
            _sdl3.append(float(np.mean([_num(pa[i].get("ppl")) - _num(pb[i].get("ppl")) for i in range(n)])))
    _m3 = float(np.mean(_sdl3)) if _sdl3 else float("nan")
    _sd3 = float(np.std(_sdl3, ddof=1)) if len(_sdl3) > 1 else 0.0
    _t3 = _m3 / (_sd3 / math.sqrt(len(_sdl3))) if (_sd3 > 0 and len(_sdl3) > 1) else 0.0
    _p3 = 2.0 * (1.0 - _norm_cdf(abs(_t3)))
    ok3 = (abs(_m3) < 0.05) and (_p3 >= 0.05)
    rec("H3′ 〔保护有效〕N1_frozen_base 的基座退化 相对 挂记忆零训练 无显著退化",
        ok3, f"ΔPPL(N1)={_dp_main:+.4f} nats；ΔPPL(挂记忆零训练)={_dp_mem0:+.4f} nats；"
             f"差={_m3:+.4f}（|差| < 0.05 且 p≥0.05）t={_t3:.2f} p≈{_p3:.3g} n={len(_sdl3)} "
             f"逐种子={['%+.4f' % x for x in _sdl3]}",
        tol=0.05, value=_m3)

    # —— H4′〔保护非平凡：存在旋钮 R 使退化显著更低 且 收益保留 ≥ 90%〕——
    KNOBS = ["N2_kl", "N3_lowrank", "N4_low_layers"]
    _best = None
    _detail = []
    for _Rk in KNOBS:
        # ★★ 缺陷 D82：原写 `if R not in agg or MAIN not in agg` —— 这里的 `R` 是**全局读秩**（一个数字）
        #   ⇒ `R not in agg` **恒为 True** ⇒ ⇒ **每个旋钮都被 `continue` 跳过**
        #   ⇒ ⇒ 「（无可用旋钮读数）」⇒ ⇒ **H4′ 从未被真正计算**（与 DIR-18 的 D57 同类）。
        if _Rk not in agg or MAIN not in agg:
            continue
        # ★★ 缺陷 D82（第二处）：配对统计的**口径**——
        #   原实现对**逐种子均值**做 t 检验 ⇒ 三个种子的差值几乎相同时 `sd≈0`
        #   ⇒ 代码把 `t` 置 0 ⇒ p=1.0 ⇒ ⇒ **该判据结构上永不可能 PASS**。
        #   修法：改用**逐文档配对**（与 H1′/H2′★ 同口径，n=180）⇒ `paired()` 内部用符号检验给 p。
        # ★★ 缺陷 D83：**符号约定反了** —— `paired(A, B)` 返回 **B − A**（见 `paired()` 的 docstring）
        #   ⇒ 这里 `_dR = MAIN.ppl − knob.ppl` ⇒ ⇒ **正值**才表示「旋钮的基座困惑度**低于**主臂」
        #   ⇒ ⇒ 判据要求「旋钮退化显著**低于**主臂」⇒ ⇒ 正确的判据条件是 **`_mR > 0`**；
        #   原实现写的是 `_mR < 0` ⇒ ⇒ 即便 D82 修好，该判据**仍然结构上永不可能 PASS**。
        #   ⇒ 本次只改**实现的符号**，判据的**阈值与文字一字未改**（卡片 rev 1.3 §8.5）。
        _dR = paired(flat(_Rk, "ppl"), flat(MAIN, "ppl"))
        _mR = _dR["mean"]; _tR = _dR["t"]; _pR = _dR["p"]
        _sdR = []
        for sd in seeds:
            pa = all_rows.get(_Rk, {}).get(sd, []); pb = all_rows.get(MAIN, {}).get(sd, [])
            n = min(len(pa), len(pb))
            if n:
                _sdR.append(float(np.mean([_num(pa[i].get("ppl")) - _num(pb[i].get("ppl")) for i in range(n)])))
        _ret = ben_ret(_Rk)
        _ok = (_mR > 0) and (_pR < 0.05) and (_ret == _ret) and (_ret >= 0.90)   # ★ D83：符号修正
        _detail.append(f"{_Rk}: ΔPPL差={_mR:+.4f}(t={_tR:.2f}, p≈{_pR:.3g}, n={_dR['n']}) "
                       f"收益保留率={_ret:.3f} 逐种子ΔPPL={['%+.4f' % x for x in _sdR]} "
                       f"⇒ {'PASS' if _ok else 'no'}")
        if _ok and (_best is None or _mR < _best[1]):
            _best = (_Rk, _mR, _ret)
    ok4 = (_best is not None)
    rec("H4′ 〔保护非平凡〕存在旋钮 R∈{N2,N3,N4} 使基座退化显著低于 N1_frozen_base 且收益保留 ≥ 90%",
        ok4, ("；".join(_detail) if _detail else "（无可用旋钮读数）")
             + (f" ⇒ ★ 最优旋钮 = {_best[0]}" if _best else " ⇒ ★ 无旋钮满足"),
        tol=0.0, value=(_best[1] if _best else float("nan")))

    # —— H5′〔非早停可解〕——
    _ret_es = ben_ret("N1_earlystop")
    _ret_best = max([ben_ret(_Rk) for _Rk in KNOBS if _Rk in agg
                     and ben_ret(_Rk) == ben_ret(_Rk)] or [float("nan")])
    _cands = [(_Rk, ben_ret(_Rk)) for _Rk in KNOBS if _Rk in agg and ben_ret(_Rk) == ben_ret(_Rk)]
    _bestknob = ("?" if not _cands else max(_cands, key=lambda kv: kv[1])[0])
    ok5 = (_ret_best == _ret_best) and (_ret_es == _ret_es) and (_ret_best > _ret_es)
    rec("H5′ 〔非早停可解〕保护旋钮的收益保留率 > 早停对照(N1_earlystop)的收益保留率",
        ok5, f"旋钮最优保留率={_ret_best:.4f}（{_bestknob}）"
             f" vs 早停对照保留率={_ret_es:.4f} ⇒ {'保护不只是少训练' if ok5 else '保护 ≈ 早停'}",
        tol=0.0, value=(_ret_best - _ret_es if (_ret_best == _ret_best and _ret_es == _ret_es) else float("nan")))

    # —— H6′〔D_kv 度量有效〕——
    _dk_train = agg.get("N3_lowrank", {}).get("dkv")
    _dk_zero = agg.get(MEM0, {}).get("dkv")
    _dk_frozen = agg.get(MAIN, {}).get("dkv")
    ok6 = (_dk_train is not None and _dk_train > 0) and (_dk_zero is not None and abs(_dk_zero) < 1e-9)
    rec("H6′ 〔D_kv 度量有效〕可训基座臂的 D_kv > 0，零训练臂的 D_kv = 0",
        ok6, f"D_kv(可训基座 N3)={_dk_train}；D_kv(零训练 {MEM0})={_dk_zero}；"
             f"D_kv(基座冻结 {MAIN})={_dk_frozen}（基座冻结 ⇒ 键分布不动 ⇒ 应为 0）",
        tol=0.0, value=(_dk_train if _dk_train is not None else float("nan")))

    # —— H7′〔不伤基座：所有臂的退化 ≤ 事先声明的界 0.10 nats〕——
    _viol = [(nm, dppl(nm)) for nm in agg
             if dppl(nm) == dppl(nm) and dppl(nm) > 0.10]
    ok7 = (len(_viol) == 0)
    rec("H7′ 〔不伤基座〕所有臂的基座困惑度退化 ≤ 事先声明的界 0.10 nats",
        ok7, (f"越界臂：{_viol}" if _viol else "无越界")
             + "；各臂 ΔPPL=" + ", ".join(f"{nm}:{dppl(nm):+.4f}" for nm in sorted(agg)),
        tol=0.10, value=max([dppl(nm) for nm in agg if dppl(nm) == dppl(nm)] or [float("nan")]))

    # —— 汇总报告 ——
    print("\\n【DIR-19 汇总】", flush=True)
    for nm in sorted(agg):
        print(f"  {nm:20s} nll={agg[nm]['nll']:.4f} benefit={ben(nm):+.4f} "
              f"PPL={agg[nm].get('ppl')} ΔPPL={dppl(nm):+.4f} dkv={agg[nm].get('dkv')} "
              f"base_drift={agg[nm].get('base_drift')} 保留率={ben_ret(nm):.3f}", flush=True)

    # ★★ 缺陷 D76：原写 `res["summary_pass"] = {"passed": passed, ...}` ⇒ `passed` **未定义**
    #   ⇒ NameError ⇒ **结果 JSON 从未写出**（smoke 的读数只留在日志里）。
    _sp = sum(1 for _c in res["checks"] if _c.get("passed"))
    res["summary_pass"] = {"passed": _sp, "total": len(res["checks"])}
    # ★★ 缺陷 D81：原写 `{passed}` ⇒ 该名字**未定义** ⇒ 收尾 NameError ⇒ **结果 JSON 从未写出**
    _sp2 = sum(1 for _c in res["checks"] if _c.get("passed"))
    print(f"\n=== 判据通过 {_sp2}/{len(res['checks'])} ===", flush=True)
    out = HERE.parent / ("R20_smoke_results.json" if args.smoke else "R20_results.json")   # ★ D78
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
