import torch, importlib.util, sys
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m; spec.loader.exec_module(m); return m
old=load("dir92_old_module.py","oldmod")
new=load("sgmn_module.py","newmod")
d_model,d_k,d_v,r=16,8,8,2
def build(mod):
    gen=torch.Generator(device="cpu"); gen.manual_seed(123)
    m=mod.LayerMem(d_model,d_k,d_v,r,0.5,0.05,1e-3,0.05,"cpu",gen,use_factorized=True,no_write=False)
    return m
a,b=build(old),build(new)
# 逐位核参数
for n in ("Pk","Pv","W0","U","V","Wq","Wo"):
    assert torch.equal(getattr(a,n),getattr(b,n)), n
print("  参数逐位一致 ✓")
# 同一输入下 write 后的 U,V 是否逐位相同
torch.manual_seed(7)
for trial in range(3):
    prev=torch.randn(1,64,d_model); cur=torch.randn(1,64,d_model)
    A,B=build(old),build(new)
    for m in (A,B):
        m.prev_hs=prev.detach(); m.write(cur)
    same=torch.equal(A.U,B.U) and torch.equal(A.V,B.V) and torch.equal(A.Wfull if A.Wfull is not None else A.U, B.Wfull if B.Wfull is not None else B.U)
    print(f"  trial{trial}: 写后 U/V 逐位一致 = {torch.equal(A.U,B.U) and torch.equal(A.V,B.V)}")
# maxpool 分支确实不同
C=build(new); C.op="maxpool"
C.prev_hs=prev.detach(); C.write(cur)
print("  maxpool 与 mean 的写后 U 是否不同 =", not torch.equal(A.U,C.U))
