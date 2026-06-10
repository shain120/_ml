# ex6-xor-classify.py
# XOR 神經網路分類範例
# 不再 import nn0.py 的 Value，避免你的 nn0.py 缺 tanh / _backward 問題

import math
import random


# ============================================================
# 1. 自動微分 Value 類別
# ============================================================

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# ============================================================
# 2. 神經網路元件：Neuron / Layer / MLP
# ============================================================

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w1*x1 + w2*x2 + b
        act = self.b

        for wi, xi in zip(self.w, x):
            act = act + wi * xi

        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        params = []

        for neuron in self.neurons:
            params.extend(neuron.parameters())

        return params


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = []

        for i in range(len(nouts)):
            self.layers.append(Layer(sizes[i], sizes[i + 1]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        # 最後一層只有一個輸出
        return x[0]

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params


# ============================================================
# 3. XOR 資料
# ============================================================

xs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

# tanh 輸出範圍是 -1 到 1
# 所以這裡用：
# 0 表示成 -1
# 1 表示成  1
ys = [
    -1,
    1,
    1,
    -1,
]


# ============================================================
# 4. 訓練
# ============================================================

random.seed(42)

# 2 個輸入
# 4 個隱藏層神經元
# 1 個輸出
model = MLP(2, [4, 1])

learning_rate = 0.05
epochs = 5000

print("開始訓練 XOR 神經網路...\n")

for epoch in range(1, epochs + 1):
    # forward
    ypred = []

    for x in xs:
        x_value = [Value(x[0]), Value(x[1])]
        pred = model(x_value)
        ypred.append(pred)

    # loss = sum((預測 - 正確答案)^2)
    loss = Value(0)

    for pred, y_true in zip(ypred, ys):
        loss = loss + (pred - y_true) ** 2

    # 清空梯度
    for p in model.parameters():
        p.grad = 0.0

    # backward
    loss.backward()

    # 更新參數
    for p in model.parameters():
        p.data = p.data - learning_rate * p.grad

    # 印出訓練過程
    if epoch == 1 or epoch % 500 == 0:
        preds_show = [round(p.data, 4) for p in ypred]
        print(
            f"Epoch {epoch:4d} | "
            f"Loss: {loss.data:.6f} | "
            f"preds: {preds_show}"
        )


# ============================================================
# 5. 測試
# ============================================================

print("\n訓練完成，測試 XOR 真值表：\n")

correct = 0

for x in xs:
    pred = model([Value(x[0]), Value(x[1])])
    raw = pred.data

    # raw > 0 判斷成 1
    # raw < 0 判斷成 0
    label = 1 if raw > 0 else 0
    answer = x[0] ^ x[1]

    if label == answer:
        correct += 1

    print(
        f"{x[0]} xor {x[1]} -> "
        f"預測: {label} | "
        f"正確答案: {answer} | "
        f"raw={raw:.4f}"
    )

print()
print(f"Accuracy: {correct}/4 = {correct / 4 * 100:.1f}%")