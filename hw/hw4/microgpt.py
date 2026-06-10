"""
microgpt.py - 極簡 GPT 實作（純 Python，無第三方套件依賴）
本程式碼改寫自 Andrej Karpathy 的 microgpt 專案。
適合做為機器學習或深度學習課程（例如 Homework 4）的 GPT 核心原理參考。

本檔案包含完整的演算法流程：
1. 資料集讀取（人名 dataset）
2. Tokenizer（字元層級編碼與解碼）
3. Autograd（純 Python 實現的反向傳播自動求導引擎）
4. GPT-2 類似的 Transformer 網路架構（RMSNorm, Multi-Head Attention, MLP）
5. Adam 最佳化器 (Adam Optimizer)
6. 訓練迴圈 (Training loop)
7. 推論/生成迴圈 (Inference loop)

@karpathy & Antigravity (Google DeepMind Team)
"""

import os       # 用於檢查檔案是否存在
import math     # 用於 log, exp 等數學運算
import random   # 用於隨機權重初始化與隨機採樣
random.seed(42) # 設定隨機種子以確保結果可重現

# =============================================================================
# 1. 資料集準備 (Dataset)
# =============================================================================
# 我們使用一個包含 32,000 個英文名字的文字檔。模型將學習這些名字的拼寫規律，並生成新的名字。
if not os.path.exists('input.txt'):
    print("正在下載資料集 input.txt...")
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

# 讀取名字並去除空白行，docs 是一個 list[str]
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs) # 打亂資料順序
print(f"資料集大小 (num docs): {len(docs)}")

# =============================================================================
# 2. 分詞器 (Tokenizer)
# =============================================================================
# 在字元層級進行編碼。我們找出資料集中所有不重複的字元，並將其對應到 0 到 n-1 的整數。
# 另外加入一個特殊的 BOS (Beginning of Sequence) 代表序列的開始與結束。
uchars = sorted(set(''.join(docs))) # 所有出現過的字元（a-z）
BOS = len(uchars)                  # BOS Token 的 ID
vocab_size = len(uchars) + 1       # 詞彙表大小（26個字母 + 1個 BOS Token = 27）
print(f"詞彙表大小 (vocab size): {vocab_size} (包含字元: {''.join(uchars)} 和 BOS)")

# =============================================================================
# 3. 自動求導引擎 (Autograd)
# =============================================================================
# 實現一個純 Python 的純標量 (scalar-level) 自動求導引擎。
# 每個 Value 物件代表計算圖中的一個節點，紀錄了數值(data)、梯度(grad)以及其子節點。
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # 優化記憶體使用

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 前向傳播計算出的標量數值
        self.grad = 0                   # 損失函數對此節點的偏微分（梯度）
        self._children = children       # 計算圖中的子節點
        self._local_grads = local_grads # 當前節點對子節點的局部偏微分 (local gradients)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): 
        # 次方運算：d(x^n)/dx = n * x^(n-1)
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self): 
        # 對數運算：d(ln(x))/dx = 1/x
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self): 
        # 指數運算：d(e^x)/dx = e^x
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self): 
        # ReLU 激活函數：x > 0 時導數為 1，否則為 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # 以下為 Python 運算子多載，支援常數與 Value 之間的反向操作與減法/除法
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
        首先對計算圖進行拓撲排序 (Topological Sort)，確保計算梯度時，父節點的梯度先計算完畢，
        然後依序將梯度傳播回子節點（鏈鎖律 Chain Rule）。
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
        
        self.grad = 1 # 損失函數對自身的梯度為 1
        # 逆向走訪拓撲排序，應用鏈鎖律計算梯度
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# =============================================================================
# 4. GPT 模型參數初始化 (Parameters Initialization)
# =============================================================================
# 定義網路超參數
n_layer = 1     # Transformer 層數 (depth)
n_embd = 16     # 嵌入維度 (width/embedding dimension)
block_size = 16 # 最大上下文長度 (maximum context length)
n_head = 4      # 注意力機制的多頭數量 (number of attention heads)
head_dim = n_embd // n_head # 每個 Attention Head 的維度 (16 // 4 = 4)

# 矩陣初始化輔助函數，生成隨機高斯分佈的 Value 矩陣
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# 初始化模型權重字典 state_dict
# - wte: Word Token Embedding (詞嵌入矩陣)
# - wpe: Word Position Embedding (位置嵌入矩陣)
# - lm_head: Language Model Head (最終映射回詞彙表的線性層)
state_dict = {
    'wte': matrix(vocab_size, n_embd), 
    'wpe': matrix(block_size, n_embd), 
    'lm_head': matrix(vocab_size, n_embd)
}

# 逐層初始化 Transformer 層中的 Q, K, V 投影矩陣、輸出投影矩陣及 MLP 兩層線性權重
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd) # 升維投影 4x
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd) # 降維投影回 n_embd

# 將所有權重矩陣展平為一個單一的 list[Value]，便於最佳化器更新
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"模型參數總數 (num params): {len(params)}")

# =============================================================================
# 5. GPT 模型架構與前向傳播 (Model Architecture)
# =============================================================================
def linear(x, w):
    """矩陣乘法線性層：y = W * x"""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """對 logits 進行 Softmax，轉化為機率分佈，並作數值穩定處理避免溢位"""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """RMSNorm 均方根歸一化 (取代 Layernorm，減少常數運算)"""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    """
    GPT 前向傳播。
    輸入：
      - token_id: 當前字元的 ID
      - pos_id: 當前字元在序列中的位置索引
      - keys: 快取的注意力 K 向量歷史，格式為 list[list[list[Value]]]，對應各層各位置
      - values: 快取的注意力 V 向量歷史
    """
    # 1. 取得 Word Embedding 與 Position Embedding，並將兩者相加得到輸入向量
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x) # 初始歸一化

    # 2. 經過每一層 Transformer Layer
    for li in range(n_layer):
        # --- A) 自注意力機制區塊 (Self-Attention Block) ---
        x_residual = x # 殘差連接殘留
        x = rmsnorm(x)
        
        # 計算當前位置的 Query, Key, Value 向量
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        
        # 將當前 K, V 存入歷史快取（隱式實現因果遮罩 Causal Mask，只注意過去和當前的 token）
        keys[li].append(k)
        values[li].append(v)
        
        x_attn = []
        # 多頭注意力機制計算
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim] # 當前頭的 Query
            
            # 從歷史快取中提取當前頭對應的 K, V 列表
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            
            # 計算 Query 與所有歷史 Key 的內積，並除以 sqrt(head_dim) 作縮放
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits) # Softmax 得到注意力權重
            
            # 對 Value 進行加權平均，得到該頭的輸出
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out) # 拼接所有頭的輸出
            
        # 注意力投影線性層與殘差相加
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        
        # --- B) 前饋神經網路區塊 (MLP Block) ---
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x] # ReLU 激活函數
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)] # 殘差相加

    # 3. 語言模型 Head：將隱特徵映射為下一個 Token 的 Logits
    logits = linear(x, state_dict['lm_head'])
    return logits

# =============================================================================
# 6. 最佳化器設定與訓練 (Optimizer & Training Loop)
# =============================================================================
# Adam 最佳化器參數與動量緩衝區初始化
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # 一階動量緩衝區 (first moment vector)
v = [0.0] * len(params) # 二階動量緩衝區 (second moment vector)

num_steps = 1000 # 訓練步數。步數越多，生成的名字拼寫越通順
print("開始訓練...")

for step in range(num_steps):
    # 選擇一個名字，將其轉為 Token ID 列表，前後加上 BOS Token
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # 每一條資料的前向傳播，都需要全新的注意力快取
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    
    # 逐字預測下一個字元
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        
        # 交叉熵損失 (Cross-Entropy Loss)：-log(預測正確字元的機率)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
        
    # 計算該名字的平均損失
    loss = (1 / n) * sum(losses)

    # 執行反向傳播計算梯度
    loss.backward()

    # 線性學習率衰減 (Learning Rate Decay)
    lr_t = learning_rate * (1 - step / num_steps)
    
    # Adam 參數更新步驟
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        
        # 更新權重數值
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0 # 梯度清零，為下一步訓練做準備

    # 每步印出當前 Loss (為了在終端機漂亮顯示，使用 \r 覆蓋當前行)
    if (step + 1) % 10 == 0 or step == 0:
        print(f"Step {step+1:4d} / {num_steps:4d} | Loss: {loss.data:.4f}")

# =============================================================================
# 7. 文字生成/推論 (Inference)
# =============================================================================
temperature = 0.5 # 溫度係數，控制生成文字的「創意性」/「隨機度」(值介於 0 到 1 之間)
print("\n--- 推論結果 (模型隨機生成的新名字) ---")

for sample_idx in range(20):
    # 重設注意力快取
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS # 從 BOS Token 開始生成
    sample = []
    
    for pos_id in range(block_size):
        # 預測下一個 Token 的 Logits
        logits = gpt(token_id, pos_id, keys, values)
        
        # 依溫度係數縮放 Logits 並套用 Softmax 得到概率
        probs = softmax([l / temperature for l in logits])
        
        # 依概率隨機抽取下一個 Token ID
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        
        # 如果抽到 BOS Token，代表生成結束
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
        
    print(f"Sample {sample_idx+1:2d}: {''.join(sample)}")
