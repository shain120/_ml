# HW4: microGPT — 極簡 Transformer 語言模型

純 Python 實作的迷你 GPT，無第三方深度學習套件依賴。

## 目錄結構
- `microgpt.py` — 主程式（從頭實作 GPT）
- `input.txt` — 訓練資料（32000+ 英文名字）
- `micro_gpt_hw4/` — 額外參考檔案

## 實作內容
- 字元級 Tokenizer（含 BOS token）
- Scalar-level Autograd（反向傳播自動求導）
- Transformer 架構：RMSNorm, Multi-Head Attention, MLP, 殘差連接
- Adam 優化器 + 學習率衰減
- 生成推論（含 temperature 控制）

## 超參數
- n_layer=1, n_embd=16, block_size=16, n_head=4
- 學習率 0.01, Adam (β1=0.85, β2=0.99)
- 訓練 1000 步
