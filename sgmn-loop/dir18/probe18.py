import sys, pathlib, math, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir18")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir17")
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util
spec = importlib.util.spec_from_file_location("r1w", "/home/ec2-user/sgmn-loop/dir18/R1_write.py")
r1w = importlib.util.module_from_spec(spec); spec.loader.exec_module(r1w)
from sgmn_module_hebb import SGMNStack as SGS

tok = AutoTokenizer.from_pretrained(r1w.MODEL)
model = AutoModelForCausalLM.from_pretrained(r1w.MODEL, torch_dtype=torch.float32).to(r1w.DEV)
model.eval()
d = model.config.hidden_size
tr, ev = r1w.build_docs(tok, 20261028, 6, 2)
print("doc len", len(tr[0]["ids"]), "true_idx", tr[0]["true_idx"], flush=True)

for mk, eta in (("hebbian_full", 1e-5), ("hebbian_full", 1e-7), ("hebbian_full", 1e-8)):
    st = r1w.make_stack(model, d, "cue", mem_kind=mk, eta=eta)
    # --- 单次写后的状态 ---
    st.unfreeze_write()
    with torch.no_grad():
        st.model(input_ids=torch.tensor(tr[0]["ids"], device=r1w.DEV).unsqueeze(0))
    m = st.mems[list(st.mems)[0]]
    print(f"[{mk} eta={eta:g}] 写一次后 ‖W‖={float(m.core._W().norm()):.4f} "
          f"‖W0‖={float(m.core.W0.norm()):.4f} ‖W-W0‖={float((m.core._W()-m.core.W0).norm()):.4f} "
          f"fid(槽0)={m.core.fidelity(0):.4f}", flush=True)
    # --- 读 ---
    st.freeze_write()
    pre = torch.tensor(tr[0]["prefix"], device=r1w.DEV).unsqueeze(0)
    with torch.no_grad():
        lg = st.model(input_ids=pre).logits.float()
    yr = m.last_yr[0]
    print(f"   读出 ‖y‖={float(yr.norm()):.4f}  max|logits|={float(lg.abs().max()):.4g} "
          f"cue_idx={m.last_cue_idx} (真值槽 {tr[0]['true_idx']})", flush=True)
    # --- 训练轨迹（看 NaN 从哪来）---
    st.unfreeze_write()
    params = [t for _, t in st.read_params_for("cue")]
    opt = torch.optim.Adam(params, lr=1e-3)
    snap = st.snapshot_all()
    for i, x in enumerate(tr[:6]):
        st.unfreeze_write(); st.restore_all(snap)
        r1w.write_both(st, x); st.freeze_write()
        pre = torch.tensor(x["prefix"], device=r1w.DEV).unsqueeze(0)
        opt.zero_grad(set_to_none=True)
        lg = st.model(input_ids=pre).logits.float()
        tgt = torch.tensor([x["code_ids"][0]], device=r1w.DEV)
        loss = F.cross_entropy(lg[0, -1:, :], tgt)
        loss.backward()
        gn = float(sum(float(p_.grad.norm())**2 for p_ in params if p_.grad is not None)**0.5)
        opt.step()
        mm = st.mems[list(st.mems)[0]]
        print(f"   step{i} loss={float(loss):.4f} ‖g‖={gn:.4g} ‖W‖={float(mm.core._W().norm()):.4f} "
              f"max|Wo|={float(mm.Wo.abs().max()):.4g} gamma={float(mm.gamma):.4f}", flush=True)
        if not math.isfinite(float(loss)):
            print("   ⇒ NaN 出现", flush=True); break
    st.remove(); torch.cuda.empty_cache()
