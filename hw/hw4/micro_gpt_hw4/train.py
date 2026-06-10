"""
train.py - 教學版 Tiny GPT 實作（純 Python，不使用 PyTorch/numpy）
這是機器學習與深度學習課程中，用於說明 GPT 核心原理與訓練流程的教學專案。

包含功能：
1. Dataset 處理與自動生成
2. 字元級 Tokenizer (stoi, itos, encode, decode, BOS token)
3. 標量自動求導引擎 Value (包含拓撲排序 backward 與詳細局部梯度說明)
4. 簡化版 GPT 模型 (WTE, WPE, RMSNorm, Linear, Softmax, Causal Multi-Head Attention, KV Cache, MLP)
5. 負對數概似損失函數 (Negative Log Likelihood Loss)
6. 結合 Adam 最佳化器的訓練迴圈與命令列參數
7. 溫度控制的文字生成 (Inference)

執行方式：
python train.py --steps 300 --samples 10 --temperature 0.7
"""

import os
import math
import random
import argparse
import urllib.request

# =============================================================================
# 0. 命令列參數解析與隨機種子設定
# =============================================================================
parser = argparse.ArgumentParser(description="Teaching Tiny GPT from Scratch (Homework 4)")
parser.add_argument("--steps", type=int, default=1000, help="訓練步數 (預設: 1000)")
parser.add_argument("--lr", type=float, default=0.01, help="學習率 (預設: 0.01)")
parser.add_argument("--temperature", type=float, default=0.7, help="生成文字的溫度隨機度 (預設: 0.7)")
parser.add_argument("--samples", type=int, default=20, help="生成樣本的數量 (預設: 20)")
parser.add_argument("--seed", type=int, default=42, help="隨機數種子 (預設: 42)")
args = parser.parse_args()

# 設定隨機種子，確保多次執行結果一致，便於教學與批改
random.seed(args.seed)

# =============================================================================
# 1. 資料集準備 (Dataset)
# =============================================================================
# 如果 input.txt 不存在，則自動建立一個小型名字資料集作為預設資料
if not os.path.exists('input.txt'):
    print("input.txt not found. Creating default name dataset...")
    default_names = [
        "emma", "olivia", "ava", "isabella", "sophia", "charlotte", "mia", "amelia", 
        "harper", "evelyn", "abigail", "ella", "scarlett", "grace", "chloe", "nora", 
        "lily", "zoey", "hannah", "violet", "jack", "liam", "noah", "oliver", 
        "elijah", "james", "william", "benjamin", "lucas", "mason", "ethan"
    ]
    with open('input.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(default_names) + '\n')

# 讀取資料集中的所有行，每行視為一個 document (獨立序列)
docs = [line.strip() for line in open('input.txt', 'r', encoding='utf-8') if line.strip()]
# 打亂資料集順序，有助於隨機梯度下降與模型泛化
random.shuffle(docs)
print(f"Dataset loaded. Total documents: {len(docs)}")

# =============================================================================
# 2. 字元級分詞器 (Tokenizer)
# =============================================================================
class CharacterTokenizer:
    """字元級分詞器，將文字與整數 ID 列表互相轉換，並包含序列起始與結束符號 (BOS)"""
    def __init__(self, documents):
        # 收集資料集中所有出現過的字元並排序
        self.uchars = sorted(list(set(''.join(documents))))
        # 建立字元到 ID 的映射表 (stoi) 與 ID 到字元的映射表 (itos)
        self.stoi = {ch: i for i, ch in enumerate(self.uchars)}
        self.itos = {i: ch for i, ch in enumerate(self.uchars)}
        
        # BOS (Beginning of Sequence) 設定在所有字元 ID 之後，作為特殊邊界 Token
        self.BOS = len(self.uchars)
        self.vocab_size = len(self.uchars) + 1 # 總詞彙表大小 (包含 BOS)

    def encode(self, text):
        """將字串轉換為整數 ID 列表"""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        """將整數 ID 列表轉換回字串"""
        return ''.join(self.itos[i] for i in ids if i in self.itos)

# 初始化分詞器
tokenizer = CharacterTokenizer(docs)
BOS = tokenizer.BOS
vocab_size = tokenizer.vocab_size
print(f"Tokenizer initialized. Vocab size: {vocab_size} (BOS token ID: {BOS})")

# =============================================================================
# 3. 自動求導引擎 (Autograd Engine)
# =============================================================================
class Value:
    """
    自動求導的核心資料結構，用於存儲標量數值與梯度。
    藉由在前向傳播中記錄子節點以及「局部梯度 (local gradient)」，在反向傳播時便能遞迴計算鏈鎖律。
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 當前節點的前向傳播數值 (float)
        self.grad = 0.0                 # 損失函數對當前節點的偏微分值 (df/dx)
        self._children = children       # 此節點在計算圖中的子節點
        self._local_grads = local_grads # 當前節點相對於各子節點的偏微分值 (局部梯度)

    def __add__(self, other):
        """
        加法運算：z = x + y
        局部梯度：dz/dx = 1, dz/dy = 1
        """
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        """
        乘法運算：z = x * y
        局部梯度：dz/dx = y, dz/dy = x
        """
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        """
        次方運算：z = x^n (其中 n 為常數浮點數，非 Value 物件)
        局部梯度：dz/dx = n * x^(n-1)
        """
        assert isinstance(other, (int, float)), "只支援常數次方"
        local_grad = other * (self.data ** (other - 1))
        return Value(self.data**other, (self,), (local_grad,))

    def log(self):
        """
        自然對數運算：z = ln(x)
        局部梯度：dz/dx = 1 / x
        """
        local_grad = 1.0 / self.data
        return Value(math.log(self.data), (self,), (local_grad,))

    def exp(self):
        """
        指數運算：z = e^x
        局部梯度：dz/dx = e^x
        """
        out_val = math.exp(self.data)
        return Value(out_val, (self,), (out_val,))

    def relu(self):
        """
        ReLU 激活函數：z = max(0, x)
        局部梯度：x > 0 時為 1.0，否則為 0.0
        """
        local_grad = 1.0 if self.data > 0 else 0.0
        return Value(max(0.0, self.data), (self,), (local_grad,))

    # Python 運算子多載，支援常數運算與減法/除法
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        """
        執行反向傳播。
        利用拓撲排序排序計算圖中的所有節點，並將梯度由輸出端逐層傳回輸入端。
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 輸出節點 (Loss) 對自己的梯度為 1
        self.grad = 1.0
        # 逆向走訪拓撲排序節點，將父節點梯度乘以局部梯度累加至子節點
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# =============================================================================
# 4. GPT 模型架構與超參數設定
# =============================================================================
# 超參數設定
n_layer = 1     # Transformer 層數 (depth)
n_embd = 16     # 嵌入維度 (embedding dimension)
block_size = 16 # 最大上下文視窗長度
n_head = 4      # 注意力頭數
head_dim = n_embd // n_head # 每個注意頭的維度 (16 // 4 = 4)

# 權重矩陣初始化輔助函數 (隨機高斯初始化)
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# 使用 state_dict 保存所有模型權重參數
state_dict = {
    'wte': matrix(vocab_size, n_embd),   # 詞嵌入層 (Word Token Embedding)
    'wpe': matrix(block_size, n_embd),   # 位置嵌入層 (Word Position Embedding)
    'lm_head': matrix(vocab_size, n_embd) # 輸出層 (Language Model Head)
}

# 逐層初始化 Transformer 內部的多頭自注意力與前饋神經網路權重
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd) # 升維投影
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd) # 降維投影

# 將所有參數展平為單一清單以便 Adam 最佳化器進行統一更新
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"Model initialized. Total parameters: {len(params)}")

# --- 模型基礎運算函數 ---
def linear(x, w):
    """線性層計算：y = W * x"""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """Softmax 機率計算，減去最大值以確保數值穩定性"""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """RMSNorm 均方根歸一化"""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    """
    教學版 Tiny GPT 前向傳播。
    一次處理一個 token，藉由不斷累加 keys 與 values 的方式，實現因果注意力機制 (Causal Attention) 
    與 KV快取 (KV Cache) 機制，避免重複計算歷史資訊。
    """
    # 1. 詞嵌入與位置嵌入相加，並進行 RMSNorm
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    # 2. Transformer 層
    for li in range(n_layer):
        # --- A) 多頭自注意力區塊 (Multi-Head Self-Attention Block) ---
        x_residual = x
        x = rmsnorm(x)
        
        # 計算當前位置的 Query, Key, Value 投影向量
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        
        # 將當前 Key/Value 存入快取列表中，隱式保證了因果注意力 (Causal/Autoregressive)
        keys[li].append(k)
        values[li].append(v)
        
        x_attn = []
        # 分頭計算注意力權重與加權和
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            
            # 從快取中提取出該頭所有歷程的 Key 與 Value
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            
            # 計算 Dot-product 注意力權重並縮放
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            
            # 依注意力權重對 Value 進行加權平均
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out) # 拼接所有注意頭的輸出結果
            
        # 注意力輸出線性層投影，並與輸入做殘差連接 (Residual Connection)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        
        # --- B) 前饋神經網路區塊 (MLP Block) ---
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x] # ReLU 激活函數
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)] # 殘差連接

    # 3. 映射回詞彙表 Logits 空間
    logits = linear(x, state_dict['lm_head'])
    return logits

# =============================================================================
# 5. Adam 最佳化器設定
# =============================================================================
learning_rate = args.lr
beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
m = [0.0] * len(params) # 一階動量
v = [0.0] * len(params) # 二階動量

# =============================================================================
# 6. 訓練迴圈 (Training Loop)
# =============================================================================
print(f"Starting training for {args.steps} steps with learning rate {args.lr}...")

for step in range(args.steps):
    # 1. 隨機選取一個名字文檔，將其轉換為 token IDs，前後用 BOS Token 包裹
    doc = docs[step % len(docs)]
    tokens = [BOS] + tokenizer.encode(doc) + [BOS]
    n = min(block_size, len(tokens) - 1)

    # 每筆文檔的前向傳播都需要清空 KV 快取
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    losses = []
    
    # 2. 前向傳播計算整個序列的 Loss
    for pos_id in range(n):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        
        # 交叉熵 Loss (負對數概似負交叉熵)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
        
    # 平均文檔的 Loss
    loss = (1.0 / n) * sum(losses)

    # 3. 反向傳播計算所有參數梯度
    loss.backward()

    # 4. Adam 參數更新
    # 學習率線性衰減 (到最後一步衰減為近乎零，有助於收斂)
    lr_t = learning_rate * (1.0 - step / args.steps)
    for i, p in enumerate(params):
        # 累加一階與二階動量
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * (p.grad ** 2)
        # 偏差修正
        m_hat = m[i] / (1.0 - beta1 ** (step + 1))
        v_hat = v[i] / (1.0 - beta2 ** (step + 1))
        # 參數更新
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        # 梯度歸零以供下一步計算
        p.grad = 0.0

    # 每隔 50 步或最後一步印出當前的 Loss
    if (step + 1) % 50 == 0 or step == 0 or (step + 1) == args.steps:
        print(f"Step {step+1:4d} / {args.steps:4d} | Loss: {loss.data:.4f}")

# =============================================================================
# 7. 推論生成新文字 (Inference)
# =============================================================================
print(f"\n--- Generated Samples (Samples: {args.samples}, Temp: {args.temperature}) ---")

for sample_idx in range(args.samples):
    # 重設注意力 KV 快取
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    token_id = BOS
    sample_ids = []
    
    # 限制最大生成長度為 block_size
    for pos_id in range(block_size):
        # 預測下一個 Token Logits
        logits = gpt(token_id, pos_id, keys, values)
        
        # 依溫度縮放 Logits
        scaled_logits = [l / args.temperature for l in logits]
        probs = softmax(scaled_logits)
        
        # 依機率隨機選擇下一個 token ID
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        
        # 如果再次抽到 BOS Token，代表名字生成結束
        if token_id == BOS:
            break
        sample_ids.append(token_id)
        
    # 解碼成文字並印出
    generated_name = tokenizer.decode(sample_ids)
    print(f"sample {sample_idx+1:02d}: {generated_name}")
