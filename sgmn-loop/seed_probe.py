"""R0 的种子敏感性探针：同一协议下换初始化种子，看写介导占比怎么变。
同时对比 DIR-06 原种子 20261028 与本轮种子 20261104。"""
import sys, pathlib, torch
sys.path.insert(0, "/home/ec2-user/sgmn-loop")
from sgmn_module import SGMNStack
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL="Qwen/Qwen2.5-0.5B-Instruct"; DEV="cuda"
ALPHA,LAM,GAMMA,R = 1.0,0.05,0.05,8
MOUNT=[8,12,16]; EPS=1e-2; NCH=4
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32).to(DEV).eval()
for p in m.parameters(): p.requires_grad_(False)
texts=["The capital of France is","In a distant galaxy the explorer found",
       "The quick brown fox jumps over","Machine learning models are"]
enc=tok(texts,return_tensors="pt",padding="max_length",truncation=True,max_length=32)
ids=enc["input_ids"].to(DEV); mask=enc["attention_mask"].to(DEV)

def mk(seed, eta, no_write=False):
    g=torch.Generator(device="cpu"); g.manual_seed(seed)
    st=SGMNStack(m,MOUNT,896,896,R,ALPHA,LAM,0.01,GAMMA,DEV,seed=seed,use_factorized=True,no_write=no_write)
    for mm in st.mems.values(): mm.eta=eta
    return st
def snap(st): return {li:(x.U.detach().clone(),x.V.detach().clone(),x.W0.detach().clone()) for li,x in st.mems.items()}
def restore(st,s):
    with torch.no_grad():
        for li,(U,V,W0) in s.items():
            x=st.mems[li]; x.U.copy_(U); x.V.copy_(V); x.W0.copy_(W0); x.prev_hs=None
def pert(st,eps,seed=11):
    g=torch.Generator(device="cpu"); g.manual_seed(seed)
    for x in st.mems.values():
        v0=x.V[:,0].detach(); nv=float(v0.norm())
        u=torch.randn(x.d_v,generator=g).to(DEV); u=u/u.norm()
        with torch.no_grad(): x.U[:,0].add_(u,alpha=eps/max(nv,1e-12))
@torch.no_grad()
def lc(ids,mask,n=4):
    L=ids.shape[1]; cs=max(1,L//n); outs=[]
    for i in range(0,L,cs): outs.append(m(input_ids=ids[:,i:i+cs],attention_mask=mask[:,i:i+cs]).logits.float())
    return torch.cat(outs,1)

for seed in [20261028, 20261104, 1, 2, 3, 4]:
    st=mk(seed,3e-5); s=snap(st)
    restore(st,s); cw=lc(ids,mask)
    restore(st,s); pert(st,EPS); pw=lc(ids,mask)
    st.remove()
    st=mk(seed,3e-5,no_write=True); s=snap(st)
    restore(st,s); cn=lc(ids,mask)
    restore(st,s); pert(st,EPS); pn=lc(ids,mask)
    st.remove()
    dw=(pw-cw).norm(dim=-1).mean(0); dn=(pn-cn).norm(dim=-1).mean(0)
    share=float((dw-dn).mean()/dw.mean())
    print(f"seed={seed:<9} d_w={float(dw.mean()):.4e} d_nw={float(dn.mean()):.4e} 写介导占比={share*100:+.3f}%", flush=True)
