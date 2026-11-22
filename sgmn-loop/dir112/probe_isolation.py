#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-112 · 前置检查（**不是**方向卡）—— 单格隔离复测，检验 `DEF-105-02` 是否污染 `DIR-111`。

## 为什么需要这个探针（本 cycle 发现的两个**具体**隐患）

`DEF-105-02`（`ENV-NOTES`）：同一进程内连跑多格时，cuBLAS workspace 的分配受**前序格**
影响 ⇒ GEMM 归约顺序变化 ⇒ 长步 Adam 混沌放大 ⇒ **同一配置在不同「前序历史」下读数不同**
（实测 `acc_all32`：隔离 0.8750 vs 6 格之后 0.7812，基座 sha1 全程不变）。

`DIR-105/106/107/108` 都有 `run_formal*.sh`（**一格一进程**）—— 但 `DIR-109/110/111`
**一个都没有**（`ls dir{109,110,111}/run_formal*.sh` 全空）⇒ 三轮的正式轮**都是整批同进程**。

### ★★ 隐患一：`DIR-111` 的 γ 轴**混了两种进程历史**

`--stage formal` 会 `RES["cells"].setdefault(...)` **继承** `stage_stageA.json` 的格。
实测：Stage A 进程产出 **6 格，全部在 γ=0.0010**；formal 进程新跑 30 格。于是

| γ | Stage A 进程 | formal 进程 |
|---|---:|---:|
| 0.0010 | **6** | 6 |
| 0.0025 | 0 | 12 |
| 0.0100 | 0 | 12 |

⇒ **γ=0.0010 的格有 6 个来自另一个进程**，而 `δ` 是**跨 seed 取平均**的
（`_X` 对 `seeds` 求均值）⇒ γ=0.0010 的 `δ` 是**两个进程历史的混合**，
而 γ=0.0025/0.0100 是**纯 formal 进程**。
⇒ 而 `DIR-111` ③ 的假说（「小 γ 拮抗」）**正好压在最混合的那一端**。
⇒ 若 `DEF-105-02` 对 `rank_eff`/`cos_mean` 也有量级，这个 γ-趋势就**可能是进程历史的产物**。

### ★ 隐患二：`δ_Q`/`δ_O` 的「γ-不变性」（`DEF-111-01`）同样是跨格比较

`D-142` ② 用「5 个 γ 上 δ 几乎不动」下结论。若格间不可比，这条也需重述。

## 本探针做什么
对 `(γ, arm, seed)` **单格**重跑，**一格一进程**（每格一次独立 `python` 调用），
与 `DIR-111` 的**同格**读数逐项对比。**只读不判**，不改任何 `DIR-111` 产物。

用法（**每格一次进程**，由 `run_iso112.sh` 驱动）：
    python probe_isolation.py --gamma 0.001 --arms T-NONE,T-Q,T-O,T-BOTH --seed 20261118 \
        --out out/iso/seed20261118_g0.001.json
（同一进程里跑 4 个臂也**不行** —— 4 个臂就是 4 格。**本探针一次进程只跑 1 格**：
  `--arms` 若给多个，会**拒绝**并提示改用驱动脚本。）
"""
import argparse
import json
import pathlib
import random
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE.parent / "dir111"))

import R1_task111 as T                                      # noqa: E402


def build_inputs():
    """**逐字照抄** `R1_task111.main()` 的输入构造（seed 777 / 555 与键划分）。

    ★ 必须逐字照抄：否则隔离格与同进程格的**输入不同**，A/B 就没有意义。
    """
    tok, model = T.D90.model_and_tok()
    facts = T.D90.build_facts(tok)
    chunks, owner = T.D90.build_chunks(tok, facts)
    order = list(range(T.D90.N_FACTS))
    random.Random(777).shuffle(order)
    train_keys = order[:T.D90.N_TRAIN_KEYS]
    eval_keys = order[T.D90.N_TRAIN_KEYS:]
    items = T.D90.make_eval_items(tok, facts, eval_keys, random.Random(555))
    T.eval_items_cache["items"] = items        # run_cell 内部读这个全局
    return tok, model, facts, chunks, owner, train_keys, eval_keys, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", required=True, type=float)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.arm not in T.ARMS:
        print(f"★ arm 必须是 {T.ARMS} 之一", flush=True)
        return 2

    tok, model, facts, chunks, owner, train_keys, eval_keys, items = build_inputs()
    print(f"[iso] γ={a.gamma} arm={a.arm} seed={a.seed} · eval items={len(items)}", flush=True)
    cell = T.run_cell(a.arm, a.gamma, a.seed, tok, model, facts, chunks, owner,
                      train_keys, eval_keys)

    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"gamma": a.gamma, "arm": a.arm, "seed": a.seed, "cell": cell},
                            ensure_ascii=False), encoding="utf-8")
    print(f"  → {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())