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
Markov 模型只看前 N 個字，根據統計結果預測下一個字。

例如 order = 2 的時候，模型會看前兩個字：

小 貓 -> 可能接「坐」
小 貓 -> 可能接「喜」
小 貓 -> 可能接「跳」

程式訓練時會掃過語料，把每個 context 後面出現過的字統計起來。

例如句子：

小貓坐在桌上

如果 order = 2，就會建立：

小貓 -> 坐
貓坐 -> 在
坐在 -> 桌
在桌 -> 上
生成時，模型會根據目前 prompt 的最後 N 個字，去查表找下一個字。
如果找不到完整 context，就退而求其次，用最後一個字做 fallback。
如果還是找不到，就用整體字頻當 fallback。
