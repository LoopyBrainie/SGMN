import sys, torch, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir18")
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util
spec = importlib.util.spec_from_file_location("r1w", "/home/ec2-user/sgmn-loop/dir18/R1_write.py")
r1w = importlib.util.module_from_spec(spec); spec.loader.exec_module(r1w)
tok = AutoTokenizer.from_pretrained(r1w.MODEL)
model = AutoModelForCausalLM.from_pretrained(r1w.MODEL, torch_dtype=torch.float32).to(r1w.DEV); model.eval()
d = model.config.hidden_size
tr, ev = r1w.build_docs(tok, 20261028, 1, 1)

st = r1w.make_stack(model, d, "cue", mem_kind="hebbian_full", eta=1e-5)
m = st.mems[list(st.mems)[0]]
core = m.core
# 手动跑一遍写阶段，记录每一步的量级
orig = core.write
recs = []
def patched(k, v, sig=None):
    with torch.no_grad():
        W = core._W()
        G = core.alpha * torch.outer(W @ k - v, k) + core.lam * (W - core.W0)
        recs.append((float(k.norm()), float(v.norm()), float((W@k - v).norm()), float(G.norm())))
    return orig(k, v, sig)
core.write = patched
with torch.no_grad():
    st.model(input_ids=torch.tensor(tr[0]["ids"], device=r1w.DEV).unsqueeze(0))
core.write = orig
kn = np.array([r[0] for r in recs]); vn = np.array([r[1] for r in recs]); rn = np.array([r[2] for r in recs]); gn = np.array([r[3] for r in recs])
print(f"写步数 = {len(recs)}")
print(f"‖k‖   : mean={kn.mean():.4g} max={kn.max():.4g}")
print(f"‖v‖   : mean={vn.mean():.4g} max={vn.max():.4g}")
print(f"‖Wk−v‖: mean={rn.mean():.4g} max={rn.max():.4g}")
print(f"‖G‖   : mean={gn.mean():.4g} max={gn.max():.4g}")
print(f"⇒ 稳定步长上界 η < 2/(λ+α‖k‖²) = {2.0/(r1w.LAM + kn.mean()**2):.4g}")
print(f"⇒ 「一步把 W 移动 ‖v‖ 的比例」= η·‖G‖/‖v‖ ⇒ η* ≈ {np.median(vn/gn):.4g}")
print(f"前 5 步: " + "; ".join(f"‖G‖={gn[i]:.3g}" for i in range(min(5,len(recs)))))
st.remove(); torch.cuda.empty_cache()
