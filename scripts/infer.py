import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "ryota39/llm-jp-1b-sft-100k-LoRA-dpo-45k",
    trust_remote_code=True,
    )
pad_token_id = tokenizer.pad_token_id

# load model
model = AutoModelForCausalLM.from_pretrained(
    "ryota39/llm-jp-1b-sft-100k-LoRA-dpo-45k",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    )

text = "東京の観光名所を教えてください。\n### Output: "
tokenized_input = tokenizer.encode(
    text,
    add_special_tokens=False,
    return_tensors="pt"
    ).to(model.device)

attention_mask = torch.ones_like(tokenized_input)
attention_mask[tokenized_input == pad_token_id] = 0

# generate response
with torch.no_grad():
    output = model.generate(
        tokenized_input,
        attention_mask=attention_mask,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.0
    )[0]

print(tokenizer.decode(output))
