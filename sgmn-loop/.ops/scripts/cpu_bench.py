"""CPU 吞吐基准：bf16 模型 + 装置参数转 bf16，测 write_both / eval_arm / train_arm 的单篇耗时。

用途：判断「沙箱 CPU 能否承担 DIR-18 正式轮（或降级复现轮）」。
不是实验脚本，不产出判据结论（纪律 ⑧：探索性测量明确标注「不进入判据」）。
"""
import torch, time, os, sys, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(int(os.environ.get("NT", "2")))
sys.path.insert(0, "/workspace/sgmn-loop/revolve/DIR-18/rounds")
sys.path.insert(0, "/workspace/sgmn-loop/revolve/DIR-18")
import R2_write as R
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/workspace/models/qwen05b"
DT = torch.bfloat16
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=DT)
model.eval()


def rss():
    return int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0]) // 1024


def cast_inplace(t):
    """**就地**转 bf16（保持 leaf 性与 requires_grad，不破坏优化器）"""
    if torch.is_tensor(t) and t.is_floating_point():
        with torch.no_grad():
            t.data = t.data.to(DT)


tr, ev = R.build_docs(tok, 20261028, 6, 3)
d = tr[0]
print("doc len:", len(d["ids"]), "| rss after load MB:", rss(), flush=True)

st = R.make_stack(model, 896, "cue", seed=20261028, mem_kind="hebbian_full", eta=1e-7, key_mode="linear")
# 把装置内所有浮点张量转 bf16
for m in st.mems.values():
    for name in ("Pk", "Pv", "Wq", "Wo", "gamma"):
        if hasattr(m, name) and torch.is_tensor(getattr(m, name)):
            cast_inplace(getattr(m, name))
    if hasattr(m, "core") and hasattr(m.core, "keys"):
        for nm in ("keys", "vals", "W", "W0", "sigs", "ortho_basis"):
            if hasattr(m.core, nm) and torch.is_tensor(getattr(m.core, nm)):
                cast_inplace(getattr(m.core, nm))
print("rss after stack MB:", rss(), flush=True)

t0 = time.time()
R.write_both(st, d)
t_wb = time.time() - t0
print(f"write_both: {t_wb:.2f}s", flush=True)

t0 = time.time()
rows = R.eval_arm(st, ev[:3], ctrl_ids=None)
t_ev = time.time() - t0
print(f"eval_arm(3): {t_ev:.2f}s  => {t_ev/3:.2f}s/doc", flush=True)

t0 = time.time()
traj, rep = R.train_arm(st, tr[:3], "cue")
t_tr = time.time() - t0
print(f"train_arm(3): {t_tr:.2f}s  => {t_tr/3:.2f}s/doc", flush=True)
print("rss end MB:", rss(), flush=True)

print(json.dumps({"write_both_s": t_wb, "eval_per_doc_s": t_ev / 3, "train_per_doc_s": t_tr / 3,
                  "rss_mb": rss()}, ensure_ascii=False))
