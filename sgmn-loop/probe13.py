import sys, torch, math
sys.path.insert(0,"/home/ec2-user/sgmn-loop")
from sgmn_module_retrieval import SGMNStack as A
from sgmn_module_keys import SGMNStack as B
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).cuda().eval()
for p in m.parameters(): p.requires_grad_(False)
ids = torch.tensor([[100,200,300,400,500,600,700,800]], device="cuda")
for name, cls, lam in (("DIR12", A, 0.0), ("DIR12_lam1", A, 1.0), ("DIR13", B, 0.0), ("DIR13_lam1", B, 1.0)):
    st = cls(m,[8,12,16],896,896,8,1.0,0.05,1e-5,0.05,"cuda",seed=20261028,use_factorized=False,perpos=True)
    for mm in st.mems.values():
        mm.read_beta = 16.0; mm.read_lambda = lam
    tr = []
    with torch.no_grad():
        for i in range(6):
            m(input_ids=ids)
            tr.append(max(float(st.mems[li].Wfull.abs().max()) for li in st.mems))
    print(f"{name:12s} lam={lam} max|W| 轨迹:", [f"{x:.3g}" for x in tr], flush=True)
    st.remove(); torch.cuda.empty_cache()
