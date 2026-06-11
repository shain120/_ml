# HW4: microGPT — 極簡 Transformer 語言模型

純 Python 實作的迷你 GPT，無第三方深度學習套件依賴。

microGPT 大概分成幾個部分：

1. 讀取資料集
2. Tokenizer，把文字轉成數字
3. Embedding，把 token 變成向量
4. Transformer block
5. Attention 注意力機制
6. MLP 前饋網路
7. Loss 計算
8. Adam 更新參數
9. 最後生成文字

GPT 的本質是「根據前面的 token 預測下一個 token」。
訓練時，模型會一直做：
預測下一個 token -> 計算 loss -> backward -> Adam 更新參數
