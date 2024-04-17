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

    def __init__(
        self,
        project_name: str="llm_jp_1b",
        model_name: str="llm-jp/llm-jp-1.3b-v1.0",
        output_model_dir:str="./out/dpo_model",
        output_dir: str="./out/dpo_logs",
        fp16: bool=False,
        bf16: bool=True,
        num_train_epochs: int=1,
        dataloader_num_workers: int=4,
        save_total_limit: int=1,
        push_to_hub: bool=False,
        auto_find_batch_size: bool=False,
        per_device_train_batch_size: int=8,
        gradient_accumulation_steps: int=4,
        optim: str="adamw_torch",
        # optim: str="paged_adamw_32bit",
        learning_rate: float=5e-4,
        lr_scheduler_type: str="cosine",
        max_grad_norm: float=0.3,
        warmup_ratio:float=0.1,
        weight_decay:float=0.001,
        save_steps:int=50,
        logging_steps:int=50,
        report_to:str="wandb",
        beta:float = 0.1,
        max_length:int=300,
        max_prompt_length:int=300
        ):

        wandb_config = {
            "model_name": project_name,
            "epoch": num_train_epochs,
            "optim": optim,
            "learning_rate": learning_rate,
            "scheduler": lr_scheduler_type,
            "max_outout_length": max_length,
            "max_prompt_length": max_prompt_length
        }
        wandb.login()
        wandb.init(
            project=f"{project_name}_dpo",
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

        self.beta = beta
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            load_in_4bit=False,
            low_cpu_mem_usage=True,
            device_map={"": 0}  # モデル全体をGPU0にロード
        )
        self.model.config.use_cache = False  # キャッシュ (学習時はFalse)
        self.model.config.pretraining_tp = 1  # 事前学習で使用したテンソル並列ランク

        self.model_ref = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            load_in_4bit=False,
            low_cpu_mem_usage=True,
            device_map={"": 0}  # モデル全体をGPU0にロード
        )
        self.model_ref.config.pretraining_tp = 1  # 事前学習で使用したテンソル並列ランク

        self.train_dataset, self.eval_dataset = build_dataset()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,  # Fastトークナイザーの有効化
            add_eos_token=True,  # データへのEOSの追加を指示
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = "right" # fp16でのオーバーフロー問題対策

        return

    def notify(self):
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
            # max_steps=1000,  # 学習ステップ数
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            learning_rate=self.learning_rate, 
            lr_scheduler_type=self.lr_scheduler_type,
            max_grad_norm=self.max_grad_norm,  # 最大法線勾配 (勾配クリッピング)
            warmup_ratio=self.warmup_ratio,  # 線形ウォームアップのステップ比率 (0から学習率まで)
            weight_decay=self.weight_decay,  # bias/LayerNormウェイトを除く全レイヤーに適用するウェイト減衰
            save_steps=self.save_steps,  # 何ステップ毎にチェックポイントを保存するか
            logging_steps=self.logging_steps,  # 何ステップ毎にログを記録するか
            report_to=self.report_to  # レポート
        )

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
            loss_type='sigmoid' # DPO'sigmoid'/SLiC'hinge'/IPO'ipo'/HALOs'kto'
        )

        dpo_trainer.train()
        dpo_trainer.model.save_pretrained(self.output_model_dir)

        self.notify()

        return

def build_dataset():
    train_dataset = get_hh("train", sanity_check=True)
    eval_dataset = get_hh("test", sanity_check=True)

    print(train_dataset)
    print(eval_dataset)
    print("--prompt--\n",   train_dataset[2]["prompt"])
    print("--chosen--\n",   train_dataset[2]["chosen"])
    print("--rejected--\n", train_dataset[2]["rejected"])

    return train_dataset, eval_dataset


def get_hh(
    split: str,
    sanity_check: bool=False,
    silent: bool=False,
    cache_dir: str=None
    ) -> Dataset:

    dataset = load_dataset(
        "shi3z/anthropic_hh_rlhf_japanese",
        "train",
        cache_dir
        )

    dataset = dataset["train"].train_test_split(
        test_size=0.025,
        shuffle=False
        )[split]

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 80000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        return extract_anthropic_prompt(sample)

    return dataset.map(split_prompt_and_responses)


def extract_anthropic_prompt(sample):
    text0 = sample["chosen"]
    text0 = text0.replace("\\n\\n人間:", "User: ")
    text0 = text0.replace("\\n\\nAssistant:", "\n\nAssistant: ")
    text0 += "<EOD|LLM-jp>"
    text1 = sample["rejected"]
    text1 = text1.replace("\\n\\n人間:", "User: ")
    text1 = text1.replace("\\n\\nAssistant:", "\n\nAssistant: ")
    text1 += "<EOD|LLM-jp>"
    search_term = "\n\nAssistant: "
    search_term_idx0 = text0.rfind(search_term)
    search_term_idx1 = text1.rfind(search_term)

    return {
        "prompt": text0[: search_term_idx0 + len(search_term)],
        "chosen": text0[search_term_idx0 + len(search_term):],
        "rejected": text1[search_term_idx1 + len(search_term):],
    }


if __name__ == "__main__":

    kit = DPOTrainerKit()
    kit.run()
