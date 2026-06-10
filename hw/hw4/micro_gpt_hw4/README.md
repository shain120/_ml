# Tiny GPT from Scratch (極簡純 Python GPT 實作)

本專案是針對機器學習/深度學習課程「習題 4」開發的教學版 GPT 核心原理展示專案。本專案完全基於 Python 標準函式庫，**不使用 PyTorch、TensorFlow、numpy 等任何第三方深度學習框架**，從零實現了自動求導引擎、RMSNorm、自注意力機制與 Transformer 結構，並可對人名資料集進行訓練與文字生成。

---

## 1. 專案名稱
**Tiny GPT from Scratch** (教學級極簡 GPT 實作)

---

## 2. 作業目標
1. **理解 GPT 的數學與程式實作**：了解文字資料如何轉為數值，並經由 Transformer 網絡進行前向傳播。
2. **掌握 Autograd 引擎運作**：動手追蹤梯度如何在拓撲排序（Topological Sort）引導下反向傳播。
3. **理解注意力機制與快取**：探討 Causal Self-Attention 的遞迴過程以及 KV cache 在自迴歸（Autoregressive）生成時所扮演的角色。
4. **實作優化演算法**：理解 Adam 最佳化器的參數更新公式及其在無深度學習庫輔助下的純 Python 實踐。

---

## 3. 執行方式

本專案只使用 Python 內建函式庫，只要安裝了 Python 3 即可直接執行。

### 執行預設訓練與生成 (1000 steps)：
```bash
python train.py
```

### 指定步數與生成參數進行測試：
```bash
python train.py --steps 300 --samples 10 --temperature 0.7
```

### 命令列參數說明：
- `--steps` (int)：訓練迭代步數（預設 1000）。
- `--lr` (float)：學習率大小（預設 0.01）。
- `--temperature` (float)：生成隨機度控制。值越低越保守，越高越有創意（預設 0.7）。
- `--samples` (int)：推論時要生成的名字數量（預設 20）。
- `--seed` (int)：隨機數種子，固定後可重現相同的訓練與生成結果（預設 42）。

---

## 4. 檔案說明
- **`train.py`**：主要執行檔。包含自定義的 `Value` 自動求導類別、自注意力網絡架構、訓練迴圈與推論程式碼。
- **`input.txt`**：訓練資料集。每一行是一個英文名字（代表一個獨立文檔 document），若檔案不存在，程式會自動建立預設資料。
- **`README.md`**：專案使用指南與基本運作架構說明。
- **`report.md`**：習題 4 個人作業報告（學生視角撰寫）。

---

## 5. GPT 運作流程
本專案的完整資料流與求導更新週期如下：

```
[Dataset: 名字] --> [Tokenizer: 字元ID] --> [Embedding: 詞與位置向量] 
                                                    |
[Loss] <-- [Softmax Logits] <-- [Transformer Block (Attention + KV Cache + MLP)]
  |
  +--> [Backward: 拓撲排序鏈鎖律求導] --> [Adam: 參數更新] --> 循環至下一個 Step
```

1. **Dataset (資料集)**
   讀取 `input.txt` 獲得單詞列表，打亂順序。每一步驟取出一個單詞，在其前後加上特殊符號 `BOS` 標記（如 `[BOS] e m m a [BOS]`），形成輸入與目標序列對。
2. **Tokenizer (分詞器)**
   採用字元級分詞。首先分析資料集中所有字元（26個小寫英文字母），再加入特殊的 `BOS` 作為結束/啟動字元。將文字映射為對應的 ID 陣列後輸入模型。
3. **Forward (前向傳播)**
   - **Embedding**：對 token ID 分別查詢詞嵌入矩陣 `wte` 和位置嵌入矩陣 `wpe` 並相加。
   - **Transformer Block**：經過層均勻化（RMSNorm）後，輸入**多頭自注意力機制**。
   - **KV 快取**：逐個字元做 causal self-attention 運算，並將每步的 $K$ 和 $V$ 儲存在快取中。當前 Query 僅與快取中的歷史 Key 計算注意力權重，免去了巨型的 Mask 遮罩矩陣。
   - **MLP 與殘差**：自注意力輸出與輸入向量相加（殘差連接），經均勻化後通過兩層 MLP（以 ReLU 為激活函數），再經第二次殘差連接輸出。
4. **Loss 計算**
   模型最後的線性層 `lm_head` 將隱特徵轉為大小為 27 的 logits 機率。透過對正確的下一個字元計算負對數概似值（Negative Log Likelihood），得出當前字元的損失。對整句取平均即為該 Document 的平均 Loss。
5. **Backward (反向傳播)**
   調用 `loss.backward()`，此時自動求導引擎對計算圖進行拓撲排序，由後往前依次計算出模型內各個參數的偏導數（梯度值，暫存於 `grad` 屬性中）。
6. **Adam 參數更新**
   依據計算出來的梯度，使用 Adam 最佳化器的動量緩衝公式計算更新步長，並應用線性衰減學習率直接修改每個 `Value` 的實體數值 `data`。最後清空梯度暫存區。
7. **Inference (文字生成/推論)**
   從預先定義的 `BOS` token 開始，自迴歸地預測下一個字元的機率。藉由溫度係數（Temperature）調整 logits，依權重隨機採樣出下一個字元，並將其追加到 KV 快取中，重複此過程直到抽到 `BOS` 或達到 context 最大長度。

---

## 6. 預期輸出範例
當你執行 `python train.py --steps 300 --samples 10`，輸出應類似於：
```
資料集讀取完成，共包含 31 個文檔 (documents)。
分詞器初始化完成。詞彙表大小: 27 (包含 BOS token ID: 26)
模型架構建立完成。可訓練參數總數: 4192
開始訓練，總步數: 300，學習率: 0.01
Step    1 /  300 | Loss: 3.3660
Step   50 /  300 | Loss: 2.3789
Step  100 /  300 | Loss: 2.4510
Step  150 /  300 | Loss: 2.5020
Step  200 /  300 | Loss: 2.1005
Step  250 /  300 | Loss: 1.8904
Step  300 /  300 | Loss: 2.0125

--- 生成結果 (生成數量: 10，溫度: 0.7) ---
sample 01: emma
sample 02: olivia
sample 03: ava
sample 04: amelia
sample 05: scarle
...
```

---

## 7. 為什麼這算是一個 GPT？
這段純 Python 程式碼雖然規模極小，但它實質上具備了 GPT（Generative Pre-trained Transformer）最核心的底層邏輯：
1. **生成式 (Generative)**：模型被訓練用來生成文字，每一次只預測下一個 Token。
2. **預訓練/自監督 (Autoregressive)**：訓練目標不需要人工標籤，僅僅是依靠下一個 Token 當作 Target 來進行自監督預測。
3. **Transformer 架構**：使用了標準的多頭自注意力機制（Multi-head Self-Attention）、前饋神經網路（FFN/MLP）、均一化層（RMSNorm/LayerNorm）以及殘差連接（Residual connection）。

---

## 8. 和真正 ChatGPT 的差異
本教學專案為了可讀性與概念簡化，與 Open AI 實用級 ChatGPT 在以下方面存在著顯著不同：

| 比較項目 | 本教學專案 (Tiny GPT) | 生產級 GPT-4 / ChatGPT |
| :--- | :--- | :--- |
| **程式碼依賴** | 純 Python 原始碼，無任何外部庫 | C++, CUDA, PyTorch, Triton 等高度最佳化核心 |
| **運算硬體** | 僅單核 CPU 序列計算 | 數千張高性能 GPU/TPU 分散式並列矩陣計算 |
| **參數量** | **4,192** 個參數 | 數千億到數兆個參數 (175B+ Params) |
| **Tokenizer** | 字元級 (Character-level, vocab size = 27) | 子詞級 (Byte-Pair Encoding, vocab size = 100k+) |
| **訓練規模** | 30 多個英文名字 (約幾百個字元) | 整個網際網路的文字資料庫 (數十兆 Tokens) |
| **對齊微調** | 無 (僅能自迴歸胡言亂語拼字) | RLHF (人類回饋強化學習) 與 SFT 指令對齊微調 |
