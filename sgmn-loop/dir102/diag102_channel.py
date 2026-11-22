"""DIR-102 诊断二：`K=1` 与 `K=32` 都是 1.0000，**真正承载答案的是哪个位置的注入？**

★ 线索（`dir101/R1_task101._read_r101:88-94`）：
    `read()` 的注入 `m` 是 **(B,L,d_model)，在【所有位置】** 返回并加进残差；
    而 `_m_override` 的 `m[idx] = mo` **只改 `idx = arange(B)*Ln + (Ln-2)`，即每题【一个】位置**
    ⇒ 其余位置的 `m = Wo·y`（**真实检索值**）**原样保留**。

于是「`K` 个注入向量」这个自变量只作用在**一个位置**上，而检索值在其余位置照样流。

本诊断把注入**按位置过滤**（`_read_r101` 的 `m` 乘一个位置掩码），在**同一训练好的 `tab`** 下比四种模式：

    all            = 现状（全位置注入 + 读出位覆盖为 `tab[·]`）
    readout_only   = 只在 `Ln-2` 保留注入，其余位置**清零**
    nonreadout_only= 除 `Ln-2` 外全部保留，读出位**清零**（= 去掉覆盖）
    none           = 全位置清零

若 `nonreadout_only` 仍 ≈1.0 而 `readout_only` 掉到 chance ⇒ **答案由键位检索承载**，
`tab`/`K`（读出位覆盖）只是"修复/引导"，`K_min` 这个自变量测量的不是记忆通道 ⇒ `DEF-102-03`。
"""
import pathlib, sys, random
_HERE = pathlib.Path(__file__).resolve().parent
_SIB = _HERE.parent
for p in (_SIB / "dir91", _SIB / "dir100", _SIB / "dir101", _SIB):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import probe91_kpool as KP, R2_task91 as R91, R1_task100 as D100, R1_task101 as D101
import R1_task102 as D102
import torch

MODES = ("all", "readout_only", "nonreadout_only", "none")


def _patch_mode(mode):
    """把 `_read_r101` 的 `self.injection` 按位置掩码过滤。"""
    def _read(self, hs_in, drop_sg=False):
        return _patch_mode._orig(self, hs_in, drop_sg)
    return _read


def run(mount, K, seed=20261028):
    tok, model = KP.model_and_tok(); model.eval()
    facts = R91.build_facts(tok); chunks, owner = R91.build_chunks(tok, facts)
    allk = list(range(len(facts)))
    st = D102.build_stack_white(model, tok, facts, chunks, owner, seed,
                               D102.MOUNT_SINGLE if mount == "single" else D102.MOUNT_TRIPLE)
    try:
        D101.set_gamma(st, 1.0)
        Mv, ls = D102.optimise_codes(model, st, tok, facts, allk, random.Random(555), K)
        out = {}
        for mode in MODES:
            # 逐位置掩码：在 `_read_r101` 之后乘上去
            orig = D101._read_r101
            def make(orig, mode):
                def _rd(self, hs_in, drop_sg=False):
                    m = orig(self, hs_in, drop_sg)
                    Bn, Ln = hs_in.shape[0], hs_in.shape[1]
                    keep = torch.zeros(Ln, device=m.device)
                    if mode == "all":
                        keep[:] = 1.0
                    elif mode == "readout_only":
                        keep[Ln - 2] = 1.0
                    elif mode == "nonreadout_only":
                        keep[:] = 1.0; keep[Ln - 2] = 0.0
                    return m * keep.view(1, Ln, 1)
                return _rd
            from sgmn_module import LayerMem
            LayerMem.read = make(orig, mode)
            out[mode] = D101.eval_free(model, st, tok, facts, allk, Mv)
            LayerMem.read = orig
        return out, ls
    finally:
        D101.set_m(st, None); D101.restore(); st.remove(); torch.cuda.empty_cache()


if __name__ == "__main__":
    argv = sys.argv[1:]
    for mount in (argv[0].split(",") if argv else ["triple"]):
        for K in (int(x) for x in (argv[1].split(",") if len(argv) > 1 else ["1"])):
            acc, ls = run(mount, K)
            print(f"[mount={mount} K={K:>2}] " +
                  "  ".join(f"{m}={acc[m]:.4f}" for m in MODES), flush=True)
            import json
            (_HERE / "out").mkdir(exist_ok=True)
            fp = _HERE / "out" / f"channel_{mount}_K{K}.json"
            fp.write_text(json.dumps(acc, indent=1), encoding="utf-8")
            print(f"  → {fp}", flush=True)
