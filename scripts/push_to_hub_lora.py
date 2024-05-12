import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from peft import PeftModel


def push(
    base_model,
    tokenizer_model,
    new_model
    ):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    os.makedirs(new_model, exist_ok=True)

    model = PeftModel.from_pretrained(model, base_model)
    model = model.merge_and_unload()

    model.push_to_hub(new_model, use_temp_dir=False)
    tokenizer.push_to_hub(new_model, use_temp_dir=False)

    print("Push successfully to HuggingFace repo!!!")


if __name__ == '__main__':

    login(token=os.environ.get('HUGGINGFACE_API_KEY'))

    # path to the model that you trained
    # tokenizer name that you used to train
    # new model name

    # here is an example
    # base_model = './out/sft-100k-LoRA-cdpo-1b-194k-logs/checkpoint-48250'
    # tokenizer_model = 'ryota39/llm-jp-1b-sft-100k-LoRA'
    # new_model = 'llm-jp-1b-sft-100k-LoRA-kto-194k'

    base_model = ''
    tokenizer_model = ''
    new_model = ''

    push(base_model, tokenizer_model, new_model)
