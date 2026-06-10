# HW6: Markov 語言模型（非 Transformer）

字元級 n-gram Markov 語言模型，不使用注意力機制或 Transformer 架構。

## 檔案
- `markv_lm.py` — 主程式（CharMarkovLM 類別）
- `tw.txt` — 中文訓練語料（198 行，主題含動物、天氣、學習等）

## 演算法
- 前 N 個字預測下一個字（預設 N=2）
- 三層回退策略：完整 context → 最後一個字 → 整體字頻
- Top-k 篩選 + temperature 控制隨機程度
- 支援 verbose 模式顯示生成過程

## 使用方式
```bash
python markv_lm.py tw.txt --prompt "小貓" --length 30 --temperature 0.8
```
