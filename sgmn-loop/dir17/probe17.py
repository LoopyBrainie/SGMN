import sys, pathlib, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path("/home/ec2-user/sgmn-loop/dir17").resolve()))
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir17")
from sgmn_module_train import SGMNStack as SGS
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util
spec = importlib.util.spec_from_file_location("r1", "/home/ec2-user/sgmn-loop/dir17/R1_train.py")
r1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r1)
DEV="cuda"; MODEL="Qwen/Qwen2.5-0.5B-Instruct"
tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
d=model.config.hidden_size
tr, ev = r1.build_docs(tok, 20261028, 4, 3)
doc = ev[0]
print("doc len", len(doc["ids"]), "prefix len", len(doc["prefix"]), "true_idx", doc["true_idx"])
st = r1.make_stack(model, d, "cue")
snap = st.snapshot_all()
st.restore_all(snap)
ids = torch.tensor(doc["ids"], dtype=torch.long, device=DEV).unsqueeze(0)
with torch.no_grad(): st.model(input_ids=ids)
for li, m in st.mems.items():
    V = torch.stack(m.vals_written) if m.vals_written else None
    print(f"  L{li}: n_vals={0 if V is None else V.shape[0]}  count={m.core.count}")
    m.vals_written.clear(); m.keys_written.clear()
st.freeze_write()
pre = torch.tensor(doc["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
with torch.no_grad(): lg = st.model(input_ids=pre).logits.float()
for li, m in st.mems.items():
    yr = m.last_yr[0]
    print(f"  L{li}: last_yr norm={float(yr.norm()):.4f}  cue_idx={m.last_cue_idx}")
    # 重建写入值
    m.vals_written.clear()
st2 = r1.make_stack(model, d, "cue"); st2.restore_all(st2.snapshot_all())
st2.model(input_ids=ids) if False else None
with torch.no_grad(): st2.model(input_ids=ids)
V = torch.stack(st2.mems[list(st2.mems)[0]].vals_written)
yr = None
st2.freeze_write()
with torch.no_grad(): st2.model(input_ids=pre)
m0 = st2.mems[list(st2.mems)[0]]
yr = m0.last_yr[0]
cs = F.cosine_similarity(V, yr.unsqueeze(0), dim=1)
print("argmax", int(torch.argmax(cs)), "true_idx", doc["true_idx"], "cos[true]", float(cs[doc["true_idx"]]), "max cos", float(cs.max()))
