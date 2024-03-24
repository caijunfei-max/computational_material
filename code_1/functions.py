# -*- coding: utf-8 -*-
# @Time    : 2024/3/24 1:16
# @Author  : Junfei cai(蔡俊飞)
# @File    : functions.py
# @Software: PyCharm


import numpy as np
from scipy.linalg import lu_factor, lu_solve


def eigen_value(matrix):
    """
    compute the eigenvalues of a matrix
    :param matrix: the matrix to compute the eigenvalue.
    :return: the eigenvalues of the matrix.
    """
    eigenvalue = np.linalg.eigvals(matrix)
    return eigenvalue


def lu_decomposition(x, y):
    """
    perform LU decomposition for given matrix
    :param x: the coefficient matrix of the equation
    :param y: the target vector
    :return:
    """
    lu, piv = lu_factor(x)
    solution_vector = lu_solve((lu, piv), y)
    return solution_vector
