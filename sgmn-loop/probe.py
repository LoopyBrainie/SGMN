import importlib
for m in ["transformers","accelerate","safetensors","tokenizers","numpy","huggingface_hub","sentencepiece","datasets"]:
    try:
        mod = importlib.import_module(m)
        print("  %-16s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception:
        print("  %-16s MISSING" % m)
