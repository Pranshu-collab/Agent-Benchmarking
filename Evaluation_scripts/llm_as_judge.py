import pandas as pd
import torch 
import json
from transformers import AutoModelForCausalLM,AutoTokenizer


model_name="Qwen/Qwen2.5-0.5B"

device="cuda" if torch.cuda.is_available() else "cpu"


print(f"Loading model: {model_name} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,
    torch_dtype="auto"
).to(device)
model.eval()

BATCH_SIZE=20
MAX_NEW_TOKENS = 15

def judge(pred, ref):
    prompt = f""" Answer: {pred} Ground Truth: {ref}
   Give 3 scores (0-10) for:
   Correctness Completeness Reasoning

   Output ONLY 3 numbers separated by space.
   Example: 7 6 5
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    result = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    try:
        nums = [int(x) for x in result.strip().split()[:3]]
        if len(nums) < 3:
            nums = nums + [0] * (3 - len(nums))
        return nums  # [correctness, completeness, reasoning]
    except:
        return [0, 0, 0]


correctness_total = 0
completeness_total = 0
reasoning_total = 0
count = 0

print("Starting LLM-as-Judge evaluation...")
with open("results_1.jsonl") as f:
    for line in f:
        data = json.loads(line)

        pred = data["pred"]["prediction_text"]
        ref = data["ref"]
        ref_texts = ref["answers"]["text"]
        ref_text = ref_texts[0] if len(ref_texts) > 0 else ""
        c, comp, r = judge(pred, ref_text)

        correctness_total += c
        completeness_total += comp
        reasoning_total += r

        count += 1

        if count % 50 == 0:
            print(f"Processed {count}")

avg_correctness = correctness_total / count
avg_completeness = completeness_total / count
avg_reasoning = reasoning_total / count       

print("\nLLM-as-Judge Results:")
print(f"Correctness: {avg_correctness:.2f}")
print(f"Completeness: {avg_completeness:.2f}")
print(f"Reasoning: {avg_reasoning:.2f}")
    