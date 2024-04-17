# RLHF kit

## 動作環境

- Ubuntu22.04
- Python3.11
- cuda12.1

## 環境構築

```bash
source setup.sh
```

- `torch`のインストールのみ、自身のcudaのバージョンと合わせて構築することを推奨
  - 以下からインストール用コマンドを確認
  - [torchのインストール](https://pytorch.org/get-started/locally/)

## 動作手順

- デフォルトで実装されているのは以下
  - モデル: `llm-jp/llm-jp-1.3b-v1.0`
  - データセット: `shi3z/anthropic_hh_rlhf_japanese`

```python
python dpo.py
```
