import math
import random


# ----------------------------
# 測試城市資料
# key: 城市編號
# value: (x, y) 座標
# ----------------------------
cities = {
    1: (0, 0),
    2: (2, 6),
    3: (5, 2),
    4: (6, 6),
    5: (8, 3),
    6: (1, 4),
    7: (7, 8),
    8: (3, 1)
}


def distance(city1, city2):
    """
    計算兩個城市之間的歐幾里得距離
    """
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def total_distance(solution):
    """
    計算整條旅行路線距離
    solution 例如: [1, 2, 3, 4, 5]
    實際路線為: 1 -> 2 -> 3 -> 4 -> 5 -> 1
    """
    total = 0

    for i in range(len(solution)):
        city_now = solution[i]
        city_next = solution[(i + 1) % len(solution)]
        total += distance(city_now, city_next)

    return total


def height(solution):
    """
    爬山演算法通常是找 height 最大的解。

    旅行推銷員問題是要讓距離越短越好，
    所以這裡把距離乘上 -1。

    distance 越小，height 就越大。
    """
    return -1 * total_distance(solution)


def two_opt_swap(solution, i, j):
    """
    使用 2-opt 方法產生新路線。

    原本路線:
    ... a -> b -> ... -> c -> d ...

    透過反轉中間片段產生鄰居解，
    可視為移除兩條邊後重新連接，避免路線交叉。

    solution[i+1 : j+1] 會被反轉。
    """
    new_solution = solution[:]

    new_solution[i + 1:j + 1] = reversed(new_solution[i + 1:j + 1])

    return new_solution


def neighbor(solution):
    """
    找出目前 solution 的最佳鄰居。

    這裡使用 2-opt：
    選兩個位置 i, j，
    反轉 i+1 到 j 的城市順序，
    產生新的旅行路線。

    回傳 height 最高的鄰居。
    """
    best_neighbor = solution[:]
    best_height = height(best_neighbor)

    n = len(solution)

    for i in range(n - 1):
        for j in range(i + 2, n):

            # 避免整個路線被無意義反轉
            if i == 0 and j == n - 1:
                continue

            new_solution = two_opt_swap(solution, i, j)
            new_height = height(new_solution)

            if new_height > best_height:
                best_neighbor = new_solution
                best_height = new_height

    return best_neighbor


def hill_climbing(initial_solution, max_iteration=1000):
    """
    爬山演算法主程式。

    如果鄰居比目前解更好，就移動到鄰居。
    如果沒有更好的鄰居，代表到達區域最佳解，停止。
    """
    current_solution = initial_solution[:]
    current_height = height(current_solution)

    for iteration in range(max_iteration):
        next_solution = neighbor(current_solution)
        next_height = height(next_solution)

        print(f"Iteration {iteration + 1}")
        print("Current Solution:", format_solution(current_solution))
        print("Distance:", round(total_distance(current_solution), 4))
        print("Height:", round(current_height, 4))
        print("-" * 40)

        if next_height <= current_height:
            break

        current_solution = next_solution
        current_height = next_height

    return current_solution


def format_solution(solution):
    """
    把路線格式化成 1 -> 2 -> 3 -> ... -> 1
    """
    route = solution + [solution[0]]
    return " -> ".join(map(str, route))


# ----------------------------
# 主程式
# ----------------------------
if __name__ == "__main__":

    # 初始解: 1 -> 2 -> 3 -> ... -> n -> 1
    initial_solution = list(cities.keys())

    print("Initial Solution:")
    print(format_solution(initial_solution))
    print("Initial Distance:", round(total_distance(initial_solution), 4))
    print("=" * 40)

    best_solution = hill_climbing(initial_solution)

    print("\nBest Solution:")
    print(format_solution(best_solution))
    print("Best Distance:", round(total_distance(best_solution), 4))
    print("Best Height:", round(height(best_solution), 4))