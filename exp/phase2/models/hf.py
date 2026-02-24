import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("MODEL", "gpt2")
PROMPT = os.environ.get("PROMPT", "Hello from GPU Lab")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "32"))
WARMUP = int(os.environ.get("WARMUP", "0"))

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print("gpu: ", torch.cuda.get_device_name(0))

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()
inp = tok(PROMPT, return_tensors="pt").to(device)

with torch.inference_mode():
    _ = model.generate(**inp, max_new_tokens=8)
    if device == "cuda":
        torch.cuda.synchronize()

    if WARMUP:
        print("warmup-only done")
        raise SystemExit(0)

    t0 = time.time()
    out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

print("elapsed_s:", t1 - t0)
print(tok.decode(out[0], skip_special_tokens=True))
