import random, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ALPH="ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
rng=random.Random(9091)
keys,seen=[],set()
while len(keys)<32:
    k="".join(rng.choice(ALPH) for _ in range(5))
    if k in seen: continue
    seen.add(k); keys.append(k)
VAL=["teal","otter","crimson","heron","amber","badger","violet","marten",
     "olive","lynx","indigo","puma","ochre","raven","coral","stoat",
     "azure","ibex","maroon","tapir","lilac","gecko","bronze","koala",
     "beige","newt","scarlet","okapi","jade","crane","rust","bison"]
print("== 键的 token 数分布 ==")
kl=collections.Counter(len(tok(" "+k,add_special_tokens=False)["input_ids"]) for k in keys)
print(dict(kl))
print("== 值的 token 数分布 ==")
vl=collections.Counter(len(tok(" "+v,add_special_tokens=False)["input_ids"]) for v in VAL)
print(dict(vl))
print("== 前缀 'The access code for item' token 数 ==")
pref=tok("The access code for item",add_special_tokens=False)["input_ids"]
print(len(pref), tok.convert_ids_to_tokens(pref))
print("== 句长分布 (sent='The access code for item {K} is {V}.') ==")
sl=collections.Counter()
for i,k in enumerate(keys):
    v=VAL[i%len(VAL)]
    ids=tok(f"The access code for item {k} is {v}.",add_special_tokens=False)["input_ids"]
    sl[len(ids)]+=1
print(dict(sl))
# 只保留句长唯一的那组，看能否凑 32
print("== 键位/值位偏移（取句长众数那一组） ==")
target=sl.most_common(1)[0][0]
ok=0
for i,k in enumerate(keys):
    v=VAL[i%len(VAL)]
    ids=tok(f"The access code for item {k} is {v}.",add_special_tokens=False)["input_ids"]
    if len(ids)!=target: continue
    kb=tok(f"The access code for item {k}",add_special_tokens=False)["input_ids"]
    vb=tok(f"The access code for item {k} is {v}",add_special_tokens=False)["input_ids"]
    ok+=1
    if ok<=3: print(f"  {k}/{v}: sent_len={len(ids)} key_off={len(kb)-1} val_off={len(vb)-1} last={vb[-1]}")
print(f"句长=={target} 的事实数: {ok}/32")
