"""DIR-102 诊断：`A102-2` FAIL（K=1 也 1.0000）—— 注入到底有没有决定答案？

零成本判别（**不训练**，只前向）：
  ① `m ≡ 0`（不注入）
  ② `m = 随机常数`（32 事实同一个向量，未拟合）×  3 个随机种子
  ③ `m = 逐事实随机`（不同事实不同向量，也未拟合）
若 ② 也给 ≈1.0 ⇒ 注入与答案无关（装置泄漏）；
若 ② ≈0.25 且 ③ ≈0.25 ⇒ 注入**确实**决定读数，那 `K=1=1.0000` 只能来自优化确实找到了解
   ⇒ 与"一个向量装不下 32 个答案"矛盾，需查优化是否泄到别的张量。
"""
import pathlib, sys, random
_HERE = pathlib.Path(__file__).resolve().parent
_SIB = _HERE.parent
for p in (_SIB / "dir91", _SIB / "dir100", _SIB / "dir101", _SIB):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import probe91_kpool as KP, R2_task91 as R91, R1_task100 as D100, R1_task101 as D101
import R1_task102 as D102
import torch

tok, model = KP.model_and_tok(); model.eval()
facts = R91.build_facts(tok); chunks, owner = R91.build_chunks(tok, facts)
allk = list(range(len(facts)))
st = D102.build_stack_white(model, tok, facts, chunks, owner, 20261028)
try:
    D101.set_gamma(st, 1.0)
    print("[① m≡0]        acc_all32 =", D101.eval_acc(model, st, tok, facts, allk)[0])
    for sd in (1, 2, 3):
        torch.manual_seed(sd)
        const = {li: torch.randn(m.d_model, device=D101.DEV).detach() for li, m in st.mems.items()}
        D101.set_m(st, const)
        a_const = D101.eval_acc(model, st, tok, facts, allk)[0]
        D101.set_m(st, None)
        torch.manual_seed(100 + sd)
        per = {li: {i: torch.randn(m.d_model, device=D101.DEV).detach() for i in allk}
               for li, m in st.mems.items()}
        a_per = D101.eval_free(model, st, tok, facts, allk, per)
        print(f"[② const sd={sd}] acc_all32 = {a_const:.4f}   [③ per-fact sd={sd}] = {a_per:.4f}")
finally:
    D101.set_m(st, None); D101.restore(); st.remove(); torch.cuda.empty_cache()
