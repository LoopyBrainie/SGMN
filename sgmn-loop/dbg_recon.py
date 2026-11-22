"""逐步对比：装置内部的 U,V 更新 vs 我在 sync_deviation 里的重建。"""
import sys, torch
sys.path.insert(0,"/home/ec2-user/sgmn-loop")
from sgmn_module import SGMNStack
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV="cuda"; ALPHA,LAM,GAMMA,R=1.0,0.05,0.05,8; MOUNT=[8,12,16]; ETA=3e-5
tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
mdl=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",dtype=torch.float32).to(DEV).eval()
for p in mdl.parameters(): p.requires_grad_(False)
g=torch.Generator(device="cpu"); g.manual_seed(20261028)
st=SGMNStack(mdl,MOUNT,896,896,R,ALPHA,LAM,0.01,GAMMA,DEV,seed=20261028,use_factorized=True)
for m in st.mems.values(): m.eta=ETA
enc=tok(["The capital of France is"],return_tensors="pt",padding="max_length",truncation=True,max_length=32)
ids=enc["input_ids"].to(DEV); mask=enc["attention_mask"].to(DEV)
from transformers import DynamicCache
li=12; m=st.mems[li]
cache=DynamicCache()
with torch.no_grad():
    out=mdl(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True)
    for t in range(3):
        pre=m.prev_hs
        U_old=m.U.detach().clone(); V_old=m.V.detach().clone()
        W0=m.W0.detach()
        nt=out.logits[:,-1].argmax(-1,keepdim=True)
        out=mdl(input_ids=nt,attention_mask=torch.cat([mask,torch.ones(1,t+1,device=DEV,dtype=mask.dtype)],1),
                past_key_values=out.past_key_values,use_cache=True)
        # 装置内部实际量
        pf=pre.reshape(-1,pre.shape[-1]); hf=m.prev_hs.reshape(-1,m.prev_hs.shape[-1])
        k_int=(pf@m.Pk.detach()).mean(dim=0)*m.pk_scale
        v_int=(hf@m.Pv.detach()).mean(dim=0)
        # 我的量
        k_me=((pre.reshape(-1,pre.shape[-1])@m.Pk.detach()).mean(dim=0))*m.pk_scale
        v_me=(m.prev_hs.reshape(-1,m.prev_hs.shape[-1])@m.Pv.detach()).mean(dim=0)
        W_old=W0+U_old@V_old.T
        e=W_old@k_me-v_me
        GV=ALPHA*torch.outer(e,k_me@V_old)+LAM*(W_old-W0)@V_old
        dU=-ETA*GV
        U_new=U_old+dU
        V_pred=V_old-ETA*(ALPHA*torch.outer(k_me,e@U_new)+LAM*(W_old-W0).T@U_new)
        print(f"--- step {t} ---")
        print(f"  k: |k_me-k_int|={float((k_me-k_int).norm()):.3e}  ||k||={float(k_me.norm()):.3f}")
        print(f"  v: |v_me-v_int|={float((v_me-v_int).norm()):.3e}  ||v||={float(v_me.norm()):.3f}")
        print(f"  U: |U_act-U_pred|={float((m.U.detach()-U_new).norm()):.3e} rel={float((m.U.detach()-U_new).norm()/m.U.detach().norm()):.3e}")
        print(f"  V: |V_act-V_pred|={float((m.V.detach()-V_pred).norm()):.3e} rel={float((m.V.detach()-V_pred).norm()/m.V.detach().norm()):.3e}")
        print(f"  ||dU||={float(dU.norm()):.4f} ||U_old||={float(U_old.norm()):.4f} ||V_old||={float(V_old.norm()):.4f}")
