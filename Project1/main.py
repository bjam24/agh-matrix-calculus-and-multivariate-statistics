import time
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Any


def measure_time(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        measured_time = time.perf_counter() - start_time
        print(f"{func.__name__} executed in {measured_time} s.")

        return result, measured_time

    return wrapper


@measure_time
def parametric_mixed_matmul(
    A: np.ndarray, B: np.ndarray, l: int = 3
) -> tuple[np.ndarray, Counter]:
    counter = Counter()

    def traditional_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        rows_first_matrix, cols_first_matrix = A.shape
        cols_second_matrix = B.shape[1]

        C = np.empty((rows_first_matrix, cols_second_matrix))

        for i in range(rows_first_matrix):
            for j in range(cols_second_matrix):
                sum = 0
                for k in range(cols_first_matrix):
                    sum += A[i, k] * B[k, j]
                    counter["+"] += 1
                    counter["*"] += 1

                C[i, j] = sum

        return C

    def strassen_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if (n := A.shape[0]) == 1:
            counter["*"] += 1
            return A * B

        n //= 2

        A_11, A_12, A_21, A_22 = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]
        B_11, B_12, B_21, B_22 = B[:n, :n], B[:n, n:], B[n:, :n], B[n:, n:]

        P_1 = matmul(A_11 + A_22, B_11 + B_22)
        P_2 = matmul(A_21 + A_22, B_11)
        P_3 = matmul(A_11, B_12 - B_22)
        P_4 = matmul(A_22, B_21 - B_11)
        P_5 = matmul(A_11 + A_12, B_22)
        P_6 = matmul(A_21 - A_11, B_11 + B_12)
        P_7 = matmul(A_12 - A_22, B_21 + B_22)

        C_11 = P_1 + P_4 - P_5 + P_7
        C_12 = P_3 + P_5
        C_21 = P_2 + P_4
        C_22 = P_1 - P_2 + P_3 + P_6

        counter["+"] += 18 * n**2

        return np.block([[C_11, C_12], [C_21, C_22]])

    def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if np.log2(A.shape[0]) <= l:
            return traditional_matmul(A, B)
        else:
            return strassen_matmul(A, B)

    return matmul(A, B), counter


def plot(y_scale: str = "linear") -> None:
    ks = tuple(range(2, 9))
    ls = (3, 5, 7)
    l_plot_config = {3: ("red", "$l=3$"), 5: ("green", "$l=5$"), 7: ("blue", "$l=7$")}
    exec_times = defaultdict(list)
    counters = defaultdict(list)

    np.random.seed(42)

    for k in ks:
        A = np.random.randint(10, size=(2**k, 2**k))
        B = np.random.randint(10, size=(2**k, 2**k))

        for l in ls:
            (_, counter), exec_time = parametric_mixed_matmul(A, B, l)
            exec_times[l].append(exec_time)
            counters[l].append(counter)

    _plot_exec_times(ls, ks, exec_times, l_plot_config, y_scale)
    _plot_counters(ls, ks, counters, l_plot_config, y_scale)


def _plot_exec_times(
    ls: tuple, ks: tuple, exec_times: defaultdict, l_plot_config: dict, y_scale: str
) -> None:
    plt.clf()
    plt.figure(figsize=(10, 7))

    for l in ls:
        plt.scatter(
            ks, exec_times[l], color=l_plot_config[l][0], label=l_plot_config[l][1]
        )
        plt.plot(
            ks,
            exec_times[l],
            linestyle="--",
            color=l_plot_config[l][0],
            linewidth=0.5,
            dashes=(5, 10),
        )

    plt.legend()
    plt.title("Matrix multiplication execution time by matrix size for different $l$")
    plt.xticks(ks, labels=[f"{2**k} ($2^{k}$)" for k in ks])
    plt.xlabel("Matrix size ($2^k$)")
    plt.yscale(y_scale)
    plt.ylabel("Time (s)")

    plt.savefig(f"{y_scale}_time_scatter_plot.png")


def _plot_counters(
    ls: tuple, ks: tuple, counters: defaultdict, l_plot_config: dict, y_scale: str
) -> None:
    plt.clf()
    plt.figure(figsize=(10, 7))

    for l in ls:
        for op in ("+", "*"):
            plt.scatter(
                ks,
                [counter[op] for counter in counters[l]],
                color=l_plot_config[l][0],
                marker=op,
                label=l_plot_config[l][1],
            )

            plt.plot(
                ks,
                [counter[op] for counter in counters[l]],
                color=l_plot_config[l][0],
                linestyle="--",
                linewidth=0.5,
                dashes=(5, 10),
            )

    plt.legend()
    plt.title(
        "Matrix multiplication number of operations by matrix size for different $l$"
    )
    plt.xticks(ks, labels=[f"{2**k} ($2^{k}$)" for k in ks])
    plt.xlabel("Matrix size ($2^k$)")
    plt.yscale(y_scale)
    plt.ylabel("Number of operations")

    plt.savefig(f"{y_scale}_operations_counter_scatter_plot.png")


if __name__ == "__main__":
    plot(y_scale="log")
    plot(y_scale="linear")
