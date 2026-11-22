import os,sys,torch
h=os.path.join(os.path.expanduser("~"),"sgmn-loop","dir91")
sys.path.insert(0,h); sys.path.insert(0,os.path.dirname(h))
import probe91_womap as W91, R2_task91 as R2
from R2_task91 import DEV,MOUNT,N_FACTS,build_chunks,build_facts,make_stack,model_and_tok
from probe91_woform import count_mounted
tok,model=model_and_tok(); model.eval(); torch.set_grad_enabled(True)
facts=build_facts(tok); chunks,owner=build_chunks(tok,facts)
print("count_mounted fresh:",count_mounted(model,MOUNT))
st=make_stack(gamma=0.0)
print("count_mounted after make_stack:",count_mounted(model,MOUNT))
W0={li:W91.W_of(m).detach().clone() for li,m in st.mems.items()}
print("no_write init:",{li:m.no_write for li,m in st.mems.items()})
print("W_of shape:",W0[MOUNT[0]].shape, "gamma:",{li:float(m.gamma) for li,m in st.mems.items()})
# phase 1: key_matrices
K,V=R2.key_matrices(model,st,facts,chunks,owner)
d1={li:float((W91.W_of(st.mems[li])-W0[li]).abs().max()) for li in st.mems}
print("dW after key_matrices:",d1, "write_calls:",{li:m.write_calls for li,m in st.mems.items()})
# phase 2: plain forward with no_write=True
saved={li:m.no_write for li,m in st.mems.items()}
for m in st.mems.values(): m.no_write=True
for fi in range(3):
    ci=owner.index(fi)+1
    model(input_ids=chunks[ci].to(DEV))
d2={li:float((W91.W_of(st.mems[li])-W0[li]).abs().max()) for li in st.mems}
print("dW after 3 fwd (no_write=True):",d2,"write_calls:",{li:m.write_calls for li,m in st.mems.items()})
for li,m in st.mems.items(): m.no_write=saved[li]
# order-of-op check
li0=MOUNT[0]; model(input_ids=chunks[owner.index(0)+1].to(DEV))
hs=st._hs_in[li0]; pf=hs.reshape(-1,hs.shape[-1])
m=st.mems[li0]
k_mm_first=((pf@m.Pk).mean(dim=0))*m.pk_scale
H=pf.mean(dim=0)
k_mean_first=(H@m.Pk)*m.pk_scale
print("op-order: |mm_first-mean_first|max =",float((k_mm_first-k_mean_first).abs().max()),
      " scale||k||=",float(k_mm_first.norm()))
print("f32 eps check: relative =",float((k_mm_first-k_mean_first).abs().max()/k_mm_first.norm()))
