# HW3: XOR 分類 — 多層感知器 (MLP)

使用自製的自動微分引擎實作多層感知器，解決 XOR 非線性分類問題。

## 檔案
- `hw3-xor-classify.py` — 完整實作（Value autograd + Neuron/Layer/MLP + 訓練）

## 重點
- 自製 `Value` 類別支援自動微分 (tanh, +, *, pow, backward)
- 網路結構：MLP(2, [4, 1])，tanh 激活
- 訓練資料：XOR 真值表（-1 表示 0，1 表示 1）
- 損失函數：MSE
- 優化器：SGD（learning rate = 0.05, epochs = 5000）
