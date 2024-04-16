import numpy as np


def induced_norm_one(M: np.ndarray) -> int:  # float
    # lbd, x = np.linalg.eig(M)
    # return max(sum(np.absolute(np.matmul(lbd, x))) / sum(np.absolute(x)))  # slajd 7 i 16
    return max(np.absolute(M).sum(axis=0))  # slajd 12


def induced_norm_two(M: np.ndarray) -> float:
    return max(abs(np.linalg.eig(M)[0]))  # slajd 15


def induced_norm_p(M: np.ndarray, p: int) -> float:
    lbd, x = np.linalg.eig(M)
    lbd_x_p = np.sum(np.abs(np.matmul(lbd, x)) ** p) ** (1.0 / p)
    x_p = np.sum(np.abs(x) ** p) ** (1.0 / p)
    return lbd_x_p / x_p


def induced_norm_infinity(M: np.ndarray) -> int:  # slajd 13
    return max(np.absolute(M).sum(axis=1))


def cond_num_matrix_one(M: np.ndarray) -> float:
    return induced_norm_one(M) * induced_norm_one(np.linalg.inv(M))


def cond_two_matrix(M: np.ndarray) -> float:
    return induced_norm_two(M) * induced_norm_two(np.linalg.inv(M))


def cond_p_matrix(M: np.ndarray, p: int) -> float:
    return induced_norm_p(M, p) * induced_norm_p(np.linalg.inv(M), p)


def cond_inf_matrix(M: np.ndarray) -> float:
    return induced_norm_infinity(M) * induced_norm_infinity(np.linalg.inv(M))


if __name__ == "__main__":
    M = np.array([[4, 9, 2], [3, 5, 7], [8, 1, 6]])

    M_1 = induced_norm_one(M)
    print(f"Matrix 1-norm: {M_1}")
    cond_1 = cond_num_matrix_one(M)
    print(f"Condition number of a matrix 1: {cond_1}")

    M_2 = induced_norm_two(M)
    print(f"Matrix 2-norm: {M_2}")
    cond_2 = cond_two_matrix(M)
    print(f"Condition number of a matrix 2: {cond_2}")

    M_p = induced_norm_p(M, 5)
    print(f"Matrix p-norm (p = 5): {M_p}")
    cond_p = cond_p_matrix(M, 5)
    print(f"Condition number of a matrix p (p = 5): {cond_p}")

    M_inf = induced_norm_infinity(M)
    print(f"Matrix inf-norm: {M_inf}")
    cond_inf = cond_inf_matrix(M)
    print(f"Condition number of a matrix inf: {cond_inf}")

    U, S, Vh = np.linalg.svd(M)

    print("U")
    print(U)
    print("S")
    print(S)
    print("Vh")
    print(Vh)
    print(U @ np.diag(S) @ Vh)
