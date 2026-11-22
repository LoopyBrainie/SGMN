import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import DynamicCache
    print("DynamicCache: OK")
except Exception as e:
    print("DynamicCache import FAIL:", e)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).to("cuda").eval()
enc = tok(["The capital of France is"], return_tensors="pt")
ids = enc["input_ids"].to("cuda"); B,L0 = ids.shape
cache = DynamicCache()
out = m(input_ids=ids, past_key_values=cache, use_cache=True)
print("prefill ok logits", tuple(out.logits.shape), "cache_len", cache.get_seq_length())
cache = out.past_key_values
t0=time.time(); N=64
for t in range(N):
    nt = out.logits[:, -1].argmax(-1, keepdim=True)
    out = m(input_ids=nt, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
torch.cuda.synchronize()
print(f"{N} steps in {time.time()-t0:.2f}s -> {1000*(time.time()-t0)/N:.1f} ms/step")
print("gen:", repr(tok.decode(out.logits[:, -1].argmax(-1))))
# 测一个带 cache_position 的显式版本
cache2 = DynamicCache()
o2 = m(input_ids=ids, past_key_values=cache2, use_cache=True)
c2 = o2.past_key_values
for t in range(4):
    nt = o2.logits[:, -1].argmax(-1, keepdim=True)
    o2 = m(input_ids=nt, past_key_values=c2, use_cache=True, cache_position=torch.tensor([L0+t], device="cuda"))
    c2 = o2.past_key_values
print("cache_position 版 OK, seq", c2.get_seq_length())
