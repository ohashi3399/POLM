import os
import json
import wandb
import torch

from typing import Dict
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainingArguments
from trl import DPOTrainer


class DPOTrainerKit(object):

    # set default params
    def __init__(
        self,
        project_name: str="llm_jp_1b",
        model_name: str="ryota39/llm-jp-1b-sft-100k-LoRA",
        output_model_dir:str="./out/sft-100k-dpo_1b_model",
        output_dir: str="./out/sft-100k-dpo_1b_logs",
        fp16: bool=False,
        bf16: bool=True,
        num_train_epochs: int=1,
        dataloader_num_workers: int=4,
        save_total_limit: int=1,
        push_to_hub: bool=False,
        auto_find_batch_size: bool=False,
        per_device_train_batch_size: int=1,
        gradient_checkpointing: bool=True,
        gradient_accumulation_steps: int=32,
        optim='paged_adamw_8bit',
        learning_rate: float=1e-6,
        lr_scheduler_type: str="cosine",
        max_grad_norm: float=0.3,
        warmup_ratio:float=0.03,
        weight_decay:float=0.001,
        save_steps:int=50,
        logging_steps:int=50,
        report_to:str="wandb",
        beta:float = 0.1,
        max_length:int=512,
        max_prompt_length:int=512
        ):

        # set wandb config
        wandb_config = {
            "model_name": project_name,
            "epoch": num_train_epochs,
            "optim": optim,
            "learning_rate": learning_rate,
            "scheduler": lr_scheduler_type,
            "max_outout_length": max_length,
            "max_prompt_length": max_prompt_length
        }
        wandb.login(key=os.environ.get('WANDB_API_KEY'))
        wandb.init(
            project=f"{project_name}-dpo_1b_194k",
            name=project_name,
            config=wandb_config
            )

        self.output_dir = output_dir
        self.output_model_dir = output_model_dir
        self.fp16 = fp16
        self.bf16 = bf16
        self.num_train_epochs = num_train_epochs
        self.dataloader_num_workers = dataloader_num_workers
        self.save_total_limit = save_total_limit
        self.push_to_hub = push_to_hub
        self.auto_find_batch_size = auto_find_batch_size
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.report_to = report_to
        self.gradient_checkpointing = gradient_checkpointing

        self.beta = beta
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        self.model_name = model_name

        # laod target model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            load_in_4bit=False,
            low_cpu_mem_usage=True,
            device_map='auto',
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        # laod reference model
        self.model_ref = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            load_in_4bit=False,
            low_cpu_mem_usage=True,
            device_map='auto',
        )
        self.model_ref.config.pretraining_tp = 1

        # load dataset
        self.train_dataset, self.eval_dataset = build_dataset()

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            add_eos_token=True,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = "right"

        return

    def notify(self):
        # notify on wandb
        message = {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "bf16": self.bf16,
            "num_train_epochs": self.num_train_epochs,
            "dataloader_num_workers": self.dataloader_num_workers,
            "save_total_limit": self.save_total_limit,
            "push_to_hub": self.push_to_hub,
            "auto_find_batch_size": self.auto_find_batch_size,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optim": self.optim,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
            "report_to": self.report_to
        }
        message = json.dumps(message, indent=4)

        wandb.alert(
            title='学習が完了しました🎉',
            text=f"{message}",
        )
        wandb.finish()

        return

    def run(self):

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            fp16=self.fp16,
            bf16=self.bf16,
            num_train_epochs=self.num_train_epochs,
            dataloader_num_workers=self.dataloader_num_workers,
            save_total_limit=self.save_total_limit,
            push_to_hub=self.push_to_hub,
            auto_find_batch_size=self.auto_find_batch_size,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            learning_rate=self.learning_rate, 
            lr_scheduler_type=self.lr_scheduler_type,
            max_grad_norm=self.max_grad_norm,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            report_to=self.report_to,
        )

        # set dpo config, You can swich methods by changing 'loss_type' argment
        dpo_trainer = DPOTrainer(
            self.model,
            self.model_ref,
            args=training_args,
            beta=self.beta,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            loss_type='sigmoid', # DPO'sigmoid'/SLiC'hinge'/IPO'ipo'/HALOs'kto'
        )

        dpo_trainer.train()
        dpo_trainer.model.save_pretrained(self.output_model_dir)

        self.notify()

        return

def build_dataset():
    train_dataset = get_hh("train", sanity_check=False)
    eval_dataset = get_hh("test", sanity_check=False)

    print(train_dataset)
    print(eval_dataset)
    print("--prompt--\n",   train_dataset[1]["prompt"])
    print("--chosen--\n",   train_dataset[1]["chosen"])
    print("--rejected--\n", train_dataset[1]["rejected"])

    return train_dataset, eval_dataset


def get_hh(
    split: str,
    sanity_check: bool=False,
    silent: bool=False,
    cache_dir: str=None
    ) -> Dataset:

    # choose dataset
    dataset = load_dataset(
        "ryota39/dpo-ja-194k",
        "train",
        cache_dir
        )

    dataset = dataset["train"].train_test_split(
        test_size=0.01,
        shuffle=False
        )[split]

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 160000)))

    def split_prompt_and_responses(data_point) -> Dict[str, str]:
        return extract_anthropic_prompt(data_point)

    return dataset.map(split_prompt_and_responses)


# You can custom prompt template
def extract_anthropic_prompt(data_point):
    chosen = f"{data_point['chosen']}<EOD|LLM-jp>"
    rejected = f"{data_point['rejected']}<EOD|LLM-jp>"

    prompt = list()
    sep = '\n'
    for utterance in data_point["prompt"]:
        role = '###Input: ' if utterance['from'] == 'human' else '###Output: '
        content = utterance['value'].replace('\\n\\n', '')

        eos_token = '<EOD|LLM-jp>' if role == '###Output: ' else ''
        prompt.append(f'{role}{content}{eos_token}')
    prompt = sep.join(prompt)

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


if __name__ == "__main__":

    kit = DPOTrainerKit()
    kit.run()
