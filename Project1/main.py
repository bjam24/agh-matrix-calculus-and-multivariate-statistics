import numpy as np


# Traditional matrix multiplication
def traditional(first_matrix, second_matrix):
    product_matrix = [[0 for _ in range(len(first_matrix))] for _ in range(len(first_matrix))]
    for i in range(len(first_matrix)):
        for j in range(len(first_matrix)):
            sum = 0
            for m in range(len(first_matrix)):
                sum += first_matrix[i][m] * second_matrix[m][j]
            product_matrix[i][j] = sum
    return np.array(product_matrix)


# Strassen recursive matrix multiplication
def split_matrix_into_blocks(matrix, size):
    return matrix[:size, :size], matrix[:size, size:], matrix[size:, :size], matrix[size:, size:]


def add_matrices(first_matrix, second_matrix):
    return np.array([[first_matrix[i, j] + second_matrix[i, j] for j in range(first_matrix.shape[0])]
            for i in range(first_matrix.shape[0])])


def subtract_matrices(first_matrix, second_matrix):
    return np.array([[first_matrix[i, j] - second_matrix[i, j] for j in range(first_matrix.shape[0])]
            for i in range(first_matrix.shape[0])])


def strassen_recursive(first_matrix, second_matrix):
    if first_matrix.shape[0] == 1:
        return first_matrix * second_matrix
    else:
        size = first_matrix.shape[0] // 2

        a_11, a_12, a_21, a_22 = split_matrix_into_blocks(first_matrix, size)
        b_11, b_12, b_21, b_22 = split_matrix_into_blocks(second_matrix, size)

        p1 = strassen_recursive(add_matrices(a_11, a_22), add_matrices(b_11, b_22))
        p2 = strassen_recursive(add_matrices(a_21, a_22), b_11)
        p3 = strassen_recursive(a_11, subtract_matrices(b_12, b_22))
        p4 = strassen_recursive(a_22, subtract_matrices(b_21, b_11))
        p5 = strassen_recursive(add_matrices(a_11, a_12), b_22)
        p6 = strassen_recursive(subtract_matrices(a_21, a_11), add_matrices(b_11, b_12))
        p7 = strassen_recursive(subtract_matrices(a_12, a_22), add_matrices(b_21, b_22))

        c_11 = add_matrices(subtract_matrices(add_matrices(p1, p4), p5), p7)
        c_12 = add_matrices(p3, p5)
        c_21 = add_matrices(p2, p4)
        c_22 = add_matrices(add_matrices(subtract_matrices(p1, p2), p3), p6)

        product_matrix = np.zeros((2 * c_11.shape[0], 2 * c_11.shape[0]), dtype=int)
        for i in range(c_11.shape[0]):
            for j in range(c_11.shape[0]):
                product_matrix[i, j] = c_11[i, j]
                product_matrix[i, j + c_11.shape[0]] = c_12[i, j]
                product_matrix[i + c_11.shape[0], j] = c_21[i, j]
                product_matrix[i + c_11.shape[0], j + c_11.shape[0]] = c_22[i, j]
        return product_matrix


# Matrix multiplication with given condition
def matrix_multiplication_program_2(first_matrix, second_matrix, l):
    if k <= l:
        return traditional(first_matrix, second_matrix)
    else:
        return strassen_recursive(first_matrix, second_matrix)


if __name__ == '__main__':
    k = 3
    matrix_1 = np.random.randint(10, size=(2 ** k, 2 ** k))
    matrix_2 = np.random.randint(10, size=(2 ** k, 2 ** k))

    product_matrix_1 = traditional(matrix_1, matrix_2)
    product_matrix_2 = strassen_recursive(matrix_1, matrix_2)

    print(product_matrix_1)
    print('\n')
    print(product_matrix_2)

