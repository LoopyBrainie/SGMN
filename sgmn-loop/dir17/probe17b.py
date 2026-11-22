import sys, pathlib, torch, numpy as np, torch.nn.functional as F, importlib.util
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir17")
from transformers import AutoModelForCausalLM, AutoTokenizer
spec = importlib.util.spec_from_file_location("r1", "/home/ec2-user/sgmn-loop/dir17/R1_train.py")
r1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r1)
DEV="cuda"; MODEL="Qwen/Qwen2.5-0.5B-Instruct"
tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
d=model.config.hidden_size
tr, ev = r1.build_docs(tok, 20261028, 4, 6)
st = r1.make_stack(model, d, "cue")
snap0 = st.snapshot_all()
for i, doc in enumerate(ev):
    st.restore_all(snap0)
    ids = torch.tensor(doc["ids"], dtype=torch.long, device=DEV).unsqueeze(0)
    with torch.no_grad(): st.model(input_ids=ids)
    vals = {}
    for li, m in st.mems.items():
        vals[li] = torch.stack(m.vals_written) if m.vals_written else None
        m.vals_written.clear(); m.keys_written.clear()
    st.freeze_write()
    pre = torch.tensor(doc["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
    with torch.no_grad(): lg = st.model(input_ids=pre).logits.float()
    det = []
    for li, m in st.mems.items():
        yr = m.last_yr[0]; V = vals[li]
        cs = F.cosine_similarity(V, yr.unsqueeze(0), dim=1)
        det.append((int(torch.argmax(cs)), int(m.last_cue_idx), doc["true_idx"], float(cs.max())))
    print(f"doc{i}: len={len(doc['ids'])} pref={len(doc['prefix'])} true={doc['true_idx']} -> {det}")
