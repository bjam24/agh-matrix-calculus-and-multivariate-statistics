import numpy as np
np.set_printoptions(linewidth=1000, sign=' ')


def gaussian_elimination_without_pivoting(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if j == i:
                b[i] /= A[i, i]
                A[j, :] /= A[i, i]
            elif j > i:
                b[j] -= b[i] * A[j, i]
                A[j, :] -= A[i, :] * A[j, i]

    np.set_printoptions(linewidth=1000, sign=' ')
    print(f'Upper triangular matrix A\n {A.round(4)}')
    return np.linalg.solve(A, b)


def gaussian_elimination_with_pivoting(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    for i in range(A.shape[0]):
        A[[i, -1]] = A[[-1, i]]
        b[i], b[-1] = b[-1], b[i]

        for j in range(A.shape[1]):
            if j > i:
                b[j] -= b[i] * A[j, i] / A[i, i]
                A[j, :] -= A[i, :] * A[j, i] / A[i, i]

    np.set_printoptions(linewidth=1000, sign=' ')
    print(f'Upper triangular matrix A\n {A.round(4)}')
    return np.linalg.solve(A, b)


def lu_factorization_without_pivoting(A: np.ndarray, B: np.ndarray) -> tuple:
    L = np.eye(A.shape[0])

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if j > i:
                L[j, i] = A[j, i] / A[i, i]
                B[j] -= B[i] * A[j, i] / A[i, i]
                A[j, :] -= A[i, :] * A[j, i] / A[i, i]

    U = A.copy()
    return L, U


def lu_factorization_with_pivoting(A: np.ndarray) -> tuple:
    P = np.eye(A.shape[0])
    L = np.eye(A.shape[0])
    U = A.copy()

    for i in range(A.shape[0]):
        max_element_row_index = max(range(i, A.shape[0]), key=lambda k: abs(U[k, i]))
        if i != max_element_row_index:
            P[[i, max_element_row_index]] = P[[max_element_row_index, i]]
            L[[i, max_element_row_index], :i] = L[[max_element_row_index, i], :i]
            U[[i, max_element_row_index]] = U[[max_element_row_index, i]]

        for j in range(i + 1, A.shape[0]):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= U[j, i] / U[i, i] * U[i, i:]

    return L, U, P


if __name__ == "__main__":
    N = 10 + 5
    A = np.random.randint(1, 9, size=(N, N))
    b = np.random.randint(1, 9, size=N)
    X_1 = gaussian_elimination_without_pivoting(A.astype(float), b.astype(float))
    print(f'Matrix A:\n{A}\nVector b:\n{np.vstack(b)}\nResult x:\n{np.vstack(X_1.round(4))}\n')
    X_2 = gaussian_elimination_with_pivoting(A.astype(float), b.astype(float))
    print(f'Matrix A:\n{A}\nVector b:\n{np.vstack(b)}\nResult x:\n{np.vstack(X_2.round(4))}\n')
    L_1, U_1 = lu_factorization_without_pivoting(A.astype(float), b.astype(float))
    print(f'Matrix A:\n{A}\nVector b:\n{np.vstack(b)}\nL:\n{L_1.round(4)}\nU:\n{U_1.round(4)}\n')
    L_2, U_2, P_2 = lu_factorization_with_pivoting(A.astype(float))
    print(f'Matrix A:\n{A}\nVector b:\n{np.vstack(b)}\nL:\n{L_2.round(4)}\nU:\n{U_2.round(4)}\nP:\n{P_2.round(4)}')
