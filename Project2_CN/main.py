import numpy as np


def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = A.shape[0]

    Ab = np.column_stack((A, b))
    for i in range(n):
        Ab[i, i:] /= Ab[i, i]
        for j in range(i + 1, n):
            Ab[j, i:] -= Ab[j, i] * Ab[i, i:]

    x = np.empty((n,))
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, n] - np.dot(Ab[i, i + 1 : n], x[i + 1 :])

    return x


def pivot_gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = A.shape[0]

    Ab = np.column_stack((A, b))
    for i in range(n):
        pivot = i + np.argmax(np.abs(Ab[i:, i]))
        Ab[[i, pivot]] = Ab[[pivot, i]]

        Ab[i, i:] /= Ab[i, i]
        for j in range(i + 1, n):
            Ab[j, i:] -= Ab[j, i] * Ab[i, i:]

    x = np.empty((n,))
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, n] - np.dot(Ab[i, i + 1 : n], x[i + 1 :])

    return x


def lu_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]

    L, U = np.eye(n), A.copy()
    for i in range(n):
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]

    return L, U


def pivot_lu_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]

    perm, L, U = np.arange(n), np.eye(n), A.copy()
    for i in range(n):
        pivot = i + np.argmax(np.abs(U[i:, i]))
        L[[i, pivot], :i] = L[[pivot, i], :i]
        U[[i, pivot]] = U[[pivot, i]]
        perm[[i, pivot]] = perm[[pivot, i]]

        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]

    inv_perm = np.empty((n,), dtype=int)
    inv_perm[perm] = np.arange(n)

    return L[inv_perm], U


def main() -> None:
    MATRIX_SIZE = 37

    rng = np.random.default_rng(seed=42)
    A = rng.integers(25, size=(MATRIX_SIZE, MATRIX_SIZE)).astype(np.float64)
    b = rng.integers(15, size=(MATRIX_SIZE,)).astype(np.float64)

    x1 = gaussian_elimination(A, b)
    x2 = pivot_gaussian_elimination(A, b)

    print(np.allclose(A @ x1, b))
    print(np.allclose(x1, x2))

    L1, U1 = lu_decomposition(A)
    L2, U2 = pivot_lu_decomposition(A)

    print(np.allclose(L1 @ U1, A))
    print(np.allclose(L2 @ U2, A))


if __name__ == "__main__":
    main()
