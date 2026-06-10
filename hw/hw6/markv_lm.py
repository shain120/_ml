# markv_lm.py
# 習題 6：非 Transformer / 非 Attention 的語言模型
# 方法：字元級 Markov / n-gram 語言模型
# 用前 N 個字預測下一個字，預設 N=2

import argparse
import random
import math
from collections import defaultdict, Counter


class CharMarkovLM:
    def __init__(self, order=2):
        self.order = order

        # model[context] = Counter(next_char)
        # 例如：
        # ("小", "貓") -> {"坐": 1, "喜": 3, "跳": 1}
        self.model = defaultdict(Counter)

        # 如果找不到 context，就使用整體字頻當 fallback
        self.global_counter = Counter()

    def train(self, lines):
        for line in lines:
            line = line.strip()

            if not line:
                continue

            chars = list(line)

            # 統計全部字頻
            for ch in chars:
                self.global_counter[ch] += 1

            # 建立 n-gram 統計
            # order=2 時：
            # 小貓坐在桌上
            # 小貓 -> 坐
            # 貓坐 -> 在
            # 坐在 -> 桌
            for i in range(len(chars) - self.order):
                context = tuple(chars[i:i + self.order])
                next_char = chars[i + self.order]
                self.model[context][next_char] += 1

    def get_next_counter(self, context):
        context = tuple(context[-self.order:])

        # 第一層：直接找完整 context
        if context in self.model:
            return self.model[context]

        # 第二層：如果找不到完整 context，就用最後一個字做 fallback
        if self.order >= 2 and len(context) > 0:
            last_char = context[-1]
            temp_counter = Counter()

            for ctx, counter in self.model.items():
                if len(ctx) > 0 and ctx[-1] == last_char:
                    temp_counter.update(counter)

            if temp_counter:
                return temp_counter

        # 第三層：如果完全找不到，就用整體字頻
        return self.global_counter

    def sample_from_counter(self, counter, temperature=1.0, top_k=5):
        items = list(counter.items())

        if not items:
            return ""

        # 依照出現次數排序
        items.sort(key=lambda x: x[1], reverse=True)

        # 只取前 k 個候選字
        if top_k > 0:
            items = items[:top_k]

        chars = [ch for ch, count in items]
        counts = [count for ch, count in items]

        total = sum(counts)
        probs = [c / total for c in counts]

        # temperature 越小越穩定，越大越隨機
        if temperature <= 0:
            temperature = 1.0

        logits = [math.log(max(p, 1e-12)) / temperature for p in probs]
        max_logit = max(logits)

        exp_values = [math.exp(x - max_logit) for x in logits]
        exp_sum = sum(exp_values)

        probs = [x / exp_sum for x in exp_values]

        return random.choices(chars, weights=probs, k=1)[0]

    def generate(self, prompt, length=30, temperature=1.0, top_k=5, verbose=False):
        result = list(prompt)

        # 如果 prompt 比 order 短，就補空字串
        if len(result) < self.order:
            context = [""] * (self.order - len(result)) + result
        else:
            context = result[-self.order:]

        if verbose:
            print()
            print("=== 生成過程 ===")
            print("prompt:", prompt)
            print("初始 context:", context)
            print("-" * 60)

        for step in range(length):
            counter = self.get_next_counter(context)

            next_char = self.sample_from_counter(
                counter,
                temperature=temperature,
                top_k=top_k
            )

            if not next_char:
                break

            result.append(next_char)
            context = result[-self.order:]

            if verbose:
                candidates = counter.most_common(8)
                print(
                    f"step {step + 1:02d} | "
                    f"next={next_char} | "
                    f"context={context} | "
                    f"candidates={candidates}"
                )

        return "".join(result)


def load_lines(path):
    # utf-8-sig 可以避免 BOM 問題
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.readlines()


def main():
    parser = argparse.ArgumentParser(
        description="非 Transformer / 非 Attention 的 Markov 語言模型"
    )

    parser.add_argument(
        "file",
        help="訓練語料，例如 tw.txt"
    )

    parser.add_argument(
        "--prompt",
        default="小貓",
        help="輸入開頭文字"
    )

    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help="使用前幾個字預測下一個字，預設 2"
    )

    parser.add_argument(
        "--length",
        type=int,
        default=30,
        help="要生成幾個新字"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="溫度，越小越穩定，越大越隨機"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="只從前 k 個候選字抽樣，0 表示不限制"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="生成幾次"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="顯示每一步生成細節"
    )

    args = parser.parse_args()

    lines = load_lines(args.file)

    model = CharMarkovLM(order=args.order)
    model.train(lines)

    print("訓練完成")
    print(f"語料檔案：{args.file}")
    print("模型：Char-level Markov LM")
    print(f"order：{args.order}")
    print(f"prompt：{args.prompt}")
    print(f"ngram context 數量：{len(model.model)}")
    print(f"整體字頻種類：{len(model.global_counter)}")
    print("-" * 60)

    for i in range(args.n):
        text = model.generate(
            prompt=args.prompt,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            verbose=args.verbose
        )

        print(f"{i + 1}. {text}")


if __name__ == "__main__":
    main()
