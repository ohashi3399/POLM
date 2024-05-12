# POLM

POLM(Preference Optimization for Language Model) provides means of alignment such like Direct Preference Optimization.
I leave some sample codes for Supervised Fine-Tuning(SFT), DPO, especially focus on Japanese dataset.

嗜好データセットを学習するコードを公開しています。
特に日本語のデータセットに注目して直接嗜好最適化(DPO)を適用するサンプルコードを公開しています。

## 📖文献

[Direct Preference Optimization, NeurIPS2024](https://neurips.cc/virtual/2023/oral/73865)

## 💻動作環境

|OS|Python|Cuda|
|:---|:---|:---|
|Ubuntu22.04|Python3.11|cuda12.1|

## ⚙️環境構築

```bash
source setup.sh
export HUGGINGFACE_API_KEY=<Your API Key>
export WANDB_API_KEY=<Your API Key>
```

- `torch`のインストールのみ、自身のcudaのバージョンと合わせてインストールすることをおすすめします。
  - 以下からインストール用コマンドを確認
  - [torchのインストール](https://pytorch.org/get-started/locally/)

> [!NOTE]
> 開発者は公開時点で`torch==2.3.0`を使用しています。

## 📋TODO

- [x] ハッカソンでの使用コードの公開
- [ ] ハイパーパラメータをargparseで指定する形式に対応


## 🔥Training

- デフォルトのモデルは以下の1B級のLLMが設定されています。データセットは以下を使用しています。
  - モデル: `llm-jp/llm-jp-1.3b-v1.0`
  - SFTデータセット: `cl-nagoya/auto-wiki-qa`
  - DPOデータセット: `ryota39/dpo-ja-45k`

### SFT

- フルパラメータファインチューニングの場合

> [!NOTE]
> - GPUのVRAMは16GB以内に収まることを動作確認済みです。
>   - 必要に応じてモデルの規模やハイパーパラメータを調整してください。

```python
python3 scripts/sft.py
```

- 低ランク行列を使った学習(LoRA)の場合

```python
python3 scripts/sft_lora.py
```

### SFT後のモデルをHuggingFaceにアップロードする

> [!IMPORTANT]
> HuggingFaceのアカウントにモデルをアップロードするため、HuggingFaceのアカウントが無い方は先に[HuggingFace](https://huggingface.co/)でアカウントを作成しましょう。

> [!TIP]
> ローカル環境でパスを指定してDPOを行うことも可能ですが、パス指定のエラーが増えると判断したため、一度HuggingFaceのアカウントにモデルをアップロードした後に、そのモデルをダウンロードする方式を採用しています。

- フルパラメータファインチューニングの場合
  - デフォルトは`private`モードでのモデル公開です。

- base_model, tokenizer_model, new_modelにコード内の例を参考にパスを指定してください。
```python
python3 scripts/push_to_hub.py
```

- 低ランク行列を使った学習(LoRA)の場合
  - デフォルトは`private`モードでのモデル公開です。
- base_model, tokenizer_model, new_modelにコード内の例を参考にパスを指定してください。

```python
python3 scripts/push_to_hub_lora.py
```

### DPO


> [!CAUTION]
> - 現状DPOはLoRAに対応していません。
> - `sft_lora.py`と同様にモデルの学習パラメータを`target_modules`で指定して`peft_config`を`TrianingArgments`に渡せばDPOもLoRAに対応できます。


```python
python3 scripts/dpo.py
```

### DPO後のモデルをHuggingFaceにアップロードする

```python
python3 scripts/push_to_hub.py
```

> [!NOTE]
> SFTと同様に、base_model, tokenizer_model, new_modelにコード内の例を参考にパスを指定してください。

### モデルの推論

```python 
python3 scripts/infer.py
```

> [!NOTE]
> アップロードしたモデルを指定してください。デフォルトは自分が学習したモデルがロードされます。

> [!CAUTION]
> 公開済みのモデルはテキスト生成の品質が低いことを確認しています。このコードを通して学習したモデルは開発中であることをご留意ください。


## 📝メモ

- list of optimizer in Transformer library

```
ADAMW_HF = "adamw_hf"
ADAMW_TORCH = "adamw_torch"
ADAMW_TORCH_FUSED = "adamw_torch_fused"
ADAMW_TORCH_XLA = "adamw_torch_xla"
ADAMW_TORCH_NPU_FUSED = "adamw_torch_npu_fused"
ADAMW_APEX_FUSED = "adamw_apex_fused"
ADAFACTOR = "adafactor"
ADAMW_ANYPRECISION = "adamw_anyprecision"
SGD = "sgd"
ADAGRAD = "adagrad"
ADAMW_BNB = "adamw_bnb_8bit"
ADAMW_8BIT = "adamw_8bit"  # just an alias for adamw_bnb_8bit
LION_8BIT = "lion_8bit"
LION = "lion_32bit"
PAGED_ADAMW = "paged_adamw_32bit"
PAGED_ADAMW_8BIT = "paged_adamw_8bit"
PAGED_LION = "paged_lion_32bit"
PAGED_LION_8BIT = "paged_lion_8bit"
RMSPROP = "rmsprop"
RMSPROP_BNB = "rmsprop_bnb"
RMSPROP_8BIT = "rmsprop_bnb_8bit"
RMSPROP_32BIT = "rmsprop_bnb_32bit"
GALORE_ADAMW = "galore_adamw"
GALORE_ADAMW_8BIT = "galore_adamw_8bit"
GALORE_ADAFACTOR = "galore_adafactor"
GALORE_ADAMW_LAYERWISE = "galore_adamw_layerwise"
GALORE_ADAMW_8BIT_LAYERWISE = "galore_adamw_8bit_layerwise"
GALORE_ADAFACTOR_LAYERWISE = "galore_adafactor_layerwise"
```
