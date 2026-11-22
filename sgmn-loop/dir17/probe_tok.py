from transformers import AutoTokenizer
import random
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
v = tok.get_vocab(); size = max(v.values())+1
print("vocab size", size)
rng = random.Random(0)
space=0; alpha=0; rt=0; samples=[]
for i in range(1000, size):
    try: s = tok.decode([i])
    except Exception: continue
    if not s: continue
    if s[0].isspace():
        space += 1
        if len(s) >= 6 and s.strip().isalpha():
            alpha += 1
            if tok(s).input_ids == [i]:
                rt += 1
                if len(samples) < 10: samples.append((i, repr(s), tok.convert_ids_to_tokens(i)))
print("leading_space", space, "alpha_long", alpha, "roundtrip", rt)
for x in samples: print(x)
