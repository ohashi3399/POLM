import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login


def tokenize(
    prompt: str,
    tokenizer: AutoTokenizer,
    cutoff_len: int=1536
    ):

    tokenized_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        )

    return {
        "input_ids": tokenized_ids["input_ids"],
        "attention_mask": tokenized_ids["attention_mask"],
        }


# You can custome this part at arbitrary dataset and prompt
def generate_prompt(
    data_point: dict,
    eos_token: str
    ):

    prompt = list()
    prompt.append(f"指示: 次の質問に対して適切な応答を書きなさい。")
    prompt.append(f"### Input: {data_point['query']}")
    prompt.append(f"### Output: {data_point['answer']}。{data_point['text']}{eos_token}")

    return "\n".join(prompt)


if __name__ == "__main__":

    login(token=os.environ.get('HUGGINGFACE_API_KEY'))
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    dataset_name = "cl-nagoya/auto-wiki-qa"
    output_dir = f"./out/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    model_name = "llm-jp/llm-jp-1.3b-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
        )

    # check special tokens of the selected tokenizer
    print(tokenizer.special_tokens_map)
    print("bos_token :", tokenizer.bos_token, ",", tokenizer.bos_token_id)
    print("eos_token :", tokenizer.eos_token, ",", tokenizer.eos_token_id)
    print("unk_token :", tokenizer.unk_token, ",", tokenizer.unk_token_id)
    print("pad_token :", tokenizer.pad_token, ",", tokenizer.pad_token_id)
    print(tokenize("hi there", tokenizer))

    data = load_dataset(dataset_name)

    # check dataset and prompt template
    print(data["train"][10])
    print(generate_prompt(data["train"][10], tokenizer.eos_token))

    # Manually select the sample of validation
    # If you train a portion of large dataset, I recommend manual selection of validation samples
    VAL_SET_SIZE = 10000
    # VAL_SET_SIZE = int(len(data["train"])*0.1)

    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE,
        shuffle=True,
        seed=42
        )

    train_data = train_val["train"]
    val_data = train_val["test"]

    train_data = train_data.shuffle().map(
        lambda x: tokenize(generate_prompt(x, tokenizer.eos_token), tokenizer)
        )

    val_data = val_data.shuffle().map(
        lambda x: tokenize(generate_prompt(x, tokenizer.eos_token), tokenizer)
        )

    # select 100k samples from training dataset
    train_data = train_data.select(range(min(len(train_data), 100000)))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # check model architecture
    print(model)

    # set monitoring configuration
    eval_steps = 100
    save_steps = 100
    logging_steps = 10

    # set training configuration
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            num_train_epochs=1,
            learning_rate=1.5e-3,
            bf16=True,
            optim='paged_adamw_8bit',
            dataloader_num_workers=8,
            logging_steps=logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            output_dir=output_dir,
            report_to="wandb",
            warmup_steps=50,
            lr_scheduler_type="cosine",
            save_total_limit=1,
            push_to_hub=False,
            auto_find_batch_size=False,
            gradient_checkpointing=True,
            gradient_accumulation_steps=32,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False
    trainer.train() 
    model.config.use_cache = True

    trainer.model.save_pretrained(output_dir)
