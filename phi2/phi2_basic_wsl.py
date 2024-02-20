import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

t = torch.cuda.is_available()
print(t)

from huggingface_hub import snapshot_download
repo_id = 'amgadhasan/phi-2'
# model_path = snapshot_download(repo_id=repo_id,repo_type="model", local_dir="/mnt/e/models/phi-2", local_dir_use_symlinks=False)
model_path = "/mnt/e/models//phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Needs 12GB of vRAM to run in float32 (default)
# Run this line to load in float16. You need Gb of vRAM
torch.set_default_dtype(torch.float16)

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

model.device

def generate(prompt: str, generation_params: dict = {"max_length":200})-> str :
    s = time.time()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, **generation_params)
    completion = tokenizer.batch_decode(outputs)[0]

    elapsed = time.time() - s

    num_input_tokens = inputs['input_ids'].shape[1]
    num_total_tokens = outputs.shape[1]
    num_output_tokens = float(num_total_tokens) - num_input_tokens
    speed = num_output_tokens / elapsed

    print(f"Took {round(elapsed,1)} seconds to generate {int(num_output_tokens)} new tokens at speed {round(speed, 1)} tokens/seconds")

    return completion

prompt = "Write a concise analogy between human and dog"

result = generate(prompt, generation_params={"max_length":200})
print(result)
