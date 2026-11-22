#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-91 · 前提探测（**不是判据**，开卡前跑）：词面去同构到底能把键余弦压到多少？
   动机：`D-108` 把失败归因「事实句只差 nonce 与值」，据此提议"词面去同构"。
   但 k_i = **整块 64 token 均值池化隐状态** —— 必须先确认该修复在原理上可行，
   否则会把不可达阈值（max|cos|<0.5）冻结进主判据。
   同时测：① 原始均值隐状态 ② Pk 投影后的 k ③ **去均值（centered）**后的 k。
"""
import json, pathlib, random, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sgmn_module import SGMNStack

ALPHA, LAM, ETA_WRITE, R = 1.0, 0.05, 3e-5, 8

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"; DEV = "cuda"
MOUNT = [8, 12, 16]; CHUNK = 64; N_FACTS = 32
VALUE_POOL = ("teal amber violet olive rust crimson azure beige coral ivory lemon "
              "peach plum rose ruby tan wheat panda yak copper silver golden iron "
              "bronze jade pearl quartz").split()
OUT = pathlib.Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
D = model.config.hidden_size

def facts(seed=9091):
    rng = random.Random(seed); alpha = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    keys, seen = [], set()
    while len(keys) < N_FACTS:
        k = "".join(rng.choice(alpha) for _ in range(5))
        if k in seen: continue
        seen.add(k); keys.append(k)
    vals = [VALUE_POOL[i % len(VALUE_POOL)] for i in range(N_FACTS)]; rng.shuffle(vals)
    return [{"key": k, "val": v} for k, v in zip(keys, vals)]

# ---- 语料变体：V0 = DIR-90 原文（同构）；逐级增加词面多样性 ----
FRAMES = [
    "{K} is filed under the label {V}.",
    "Ask the depot about {K} and they will answer {V}.",
    "Whenever {K} comes up, the ledger says {V}.",
    "The tag attached to {K} reads {V}.",
    "If you look up {K} in the register you find {V}.",
    "On the old chart, {K} is marked {V}.",
]
TOPICS = [
    "the harbour office", "an abandoned observatory", "the night shift roster",
    "a bicycle repair shop", "the municipal archive", "a submarine cable station",
    "the mountain rescue hut", "a travelling circus", "the seed vault",
    "an apiary on the ridge", "the lighthouse logbook", "a bakery in the old town",
    "the customs warehouse", "a puppet theatre", "the radio telescope array",
    "a ferry timetable", "the geology lab", "an orchid greenhouse",
    "the printing press room", "a falconry mews", "the water treatment plant",
    "a jazz rehearsal space", "the coin collection", "an ice core freezer",
    "the rope factory", "a beekeeper cooperative", "the canal lock schedule",
    "a stone mason workshop", "the migrating bird survey", "an amateur boxing club",
    "the cheese aging cellar", "a restored carousel",
]
VERBS = ["confirms", "records", "conceals", "repeats", "mislays", "restores"]

def sent(v, f, i):
    if v == 0:
        return f"The access code for item {f['key']} is {f['val']}."
    if v == 1:
        return FRAMES[i % len(FRAMES)].format(K=f["key"], V=f["val"])
    if v == 2:
        return f"At {TOPICS[i]}, {f['key']} {VERBS[i % len(VERBS)]} the value {f['val']}."
    return (f"Field note from {TOPICS[i]}: inspectors {VERBS[i % len(VERBS)]} that "
            f"reference {f['key']} corresponds to {f['val']}, "
            f"and this was double-checked against the {TOPICS[(i+7) % len(TOPICS)]}.")

def mean_hidden(text):
    ids = tok(text + " ", add_special_tokens=False)["input_ids"]
    seq = []
    while len(seq) < CHUNK: seq.extend(ids)
    seq = seq[:CHUNK]
    st = SGMNStack(model, MOUNT, D, D, R, ALPHA, LAM, ETA_WRITE, 0.0, DEV,
                   seed=20261028, use_factorized=False, no_write=True)
    model(input_ids=torch.tensor([seq], dtype=torch.long, device=DEV))
    out = {li: st._hs_in[li].reshape(-1, st._hs_in[li].shape[-1]).mean(dim=0).clone()
           for li in st.mems}
    pk = {li: st.mems[li].Pk.detach().clone() for li in st.mems}
    sc = {li: st.mems[li].pk_scale for li in st.mems}
    st.remove()
    return out, pk, sc

def stats(M):
    """M: (n,d) -> (max|cos|, mean|cos|) off-diagonal"""
    Mn = torch.nn.functional.normalize(M, dim=1)
    C = (Mn @ Mn.T).abs()
    off = C[~torch.eye(len(M), dtype=torch.bool, device=C.device)]
    return float(off.max()), float(off.mean())

F = facts()
res = {"model": MODEL, "d_model": D, "n_facts": N_FACTS, "mount": MOUNT,
       "chunk": CHUNK, "variants": {}}
for v in (0, 1, 2, 3):
    H = {li: [] for li in MOUNT}; PK = None; SC = None
    for i, f in enumerate(F):
        h, pk, sc = mean_hidden(sent(v, f, i))
        if PK is None: PK, SC = pk, sc
        for li in MOUNT: H[li].append(h[li])
    per = {}
    for li in MOUNT:
        Hm = torch.stack(H[li]).float()                      # (32, d)
        K = (Hm @ PK[li].float()) * SC[li]                   # Pk 投影后的 k
        Hc = Hm - Hm.mean(dim=0, keepdim=True)               # 去均值（隐状态空间）
        Kc = K - K.mean(dim=0, keepdim=True)                 # 去均值（k 空间）
        per[li] = {"raw_hidden": stats(Hm), "k_proj": stats(K),
                   "k_centered": stats(Kc), "raw_hidden_centered": stats(Hc)}
    res["variants"][f"V{v}"] = per
    line = " ".join(f"L{li}: raw={per[li]['raw_hidden'][0]:.3f}/k={per[li]['k_proj'][0]:.3f}"
                    f"/kc={per[li]['k_centered'][0]:.3f}" for li in MOUNT)
    print(f"V{v} max|cos| {line}", flush=True)

(OUT / "probe91_premise.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
print("done")
