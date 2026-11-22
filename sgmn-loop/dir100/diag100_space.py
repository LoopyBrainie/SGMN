#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-100 诊断：`qmap(M c)` 与 `K_new` 的对应为何是 0？（`d = d_k = 896` ⇒ 转置错**形状不可见**）

逐步核对，**不训练**：
  ① `cos(TF(K_raw_i), K_new_i)` —— 两次采集是否只差一个 `TF`（应 = 1）
  ② `cos(C_tr A_tr, K_raw_he)` —— raw 空间泛化（前提核，应 ≈ 0.998）
  ③ `cos(qmap(C_he A_tr), K_new_he)` —— ★ 争议量（正式轮读到 0.0000）
  ④ `qmap` 与 `s*(A@x)` 逐位置是否同一（应 0 差）
  ⑤ `A` 的秩与奇异值（`A = Un·Kp` 理论 rank = 32）
"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir100")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir99")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
import probe91_kpool as KP                       # noqa: E402
import R1_task100 as R100                        # noqa: E402
import R2_task91 as R91                          # noqa: E402
import torch                                     # noqa: E402

SEED = 20261028


def main():
    torch.set_grad_enabled(False)
    tok, model = KP.model_and_tok()
    model.eval()
    facts = KP.build_facts(tok)
    chunks, owner = KP.build_chunks(tok, facts)
    tr, he = list(range(16)), list(range(16, 32))
    st = R91.make_stack(gamma=0.0, no_write=False, seed=SEED)
    for m in st.mems.values():
        m.key_tf = R100.IdentityTF()
        m.q_tf = R100.IdentityTF()
        m.v_neg_shift = False
        m.v_hold = None
    R100.install()
    try:
        Kraw, V = R100.capture_keys(model, st, facts, chunks, owner)
        TF = {}
        for li in Kraw:
            TF[li] = R100.build_tf(R100.KEY_ARM, Kraw[li], V[li], li)
            st.mems[li].key_tf = TF[li]
            st.mems[li].q_tf = TF[li].qmap
        KP.write_phase_arm(model, st, chunks, owner, {i: None for i in range(len(facts))})
        Knew, Vnew = R100.capture_keys(model, st, facts, chunks, owner)
        C = R100.prompt_reads(model, st, facts)
        allk = list(range(len(facts)))
        for li in Kraw:
            Kr, Kn, Cl = Kraw[li].float(), Knew[li].float(), C[li]
            tf = TF[li]
            A_tr = R100._ridge_A(Cl[tr], Kr[tr])
            # ① 两次采集只差 TF？
            tf_raw = torch.stack([tf(Kr[i]) for i in range(Kr.shape[0])])     # 逐行 key_tf
            c1 = R100._cos_rows(tf_raw, Kn)
            # ② raw 空间泛化
            pred_raw = Cl[he] @ A_tr
            c2 = R100._cos_rows(pred_raw, Kr[he])
            # ③ 争议量
            q = tf.qmap(pred_raw)
            c3 = R100._cos_rows(q, Kn[he])
            # ④ qmap vs 显式 s(A@x)
            expl = torch.stack([tf.s * (tf.A @ pred_raw[i]) for i in range(pred_raw.shape[0])])
            d4 = float((q - expl).abs().max())
            # ⑤ A 的秩
            sv = torch.linalg.svdvals(tf.A)
            rk = int((sv > sv[0] * 1e-6).sum())
            print(f"[L{li}] ①cos(TF(Kraw),Knew)={c1.mean():.6f}(min {c1.min():.6f}) "
                  f"②raw cos_he={c2.mean():.6f} ③cos_eff={c3.mean():+.6f}(min {c3.min():+.4f}) "
                  f"④|qmap−s(A x)|={d4:.2e} ⑤rank(A)={rk} sv1={float(sv[0]):.4f} sd={float(sv[31]):.4f}",
                  flush=True)
            # ★ 关键分离：把 pred_raw 换成**真 k_raw**，TF 后应与 K_new 完全一致
            c5 = R100._cos_rows(tf.qmap(Kr[he]), Kn[he])
            print(f"      ↳ 用**真 k_raw**过 qmap：cos(qmap(k_raw_he), K_new_he)={c5.mean():.6f}"
                  f"(min {c5.min():.6f})", flush=True)
            # ★ 若不通过 TF 而直接用 identity q_tf，看 M 在 K_new 空间能到多少
            A_new = R100._ridge_A(Cl[tr], Kn[tr])
            c6 = R100._cos_rows(Cl[he] @ A_new, Kn[he])
            print(f"      ↳ 直接拟合 K_new（q_tf=恒等）：cos_he={c6.mean():.6f}", flush=True)
    finally:
        R100.restore()
        st.remove()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())