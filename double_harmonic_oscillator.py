# モジュールのインポート
import numpy as np
import math
import cmath
from qutip import *

class DoubleHarmonicOscillator():

    def __init__(self):
        """
        定数の定義
        """
        self.a = 0
        self.b = 3
        self.h = 5
        self.gamma = 0.1
        self.N = 5     # matrix size

    """
    squueze operator's delta
    """
    def delta_squeeze_1(self, m, n):
        if m == n + 2:
            return 1
        else:
            return 0

    def delta_squeeze_2(self, m, n):
        if m == n - 2:
            return 1
        else:
            return 0


    """
    x operator's delta
    """
    def delta_x_1(self, m, n):
        if m == n + 1:
            return 1
        else:
            return 0

    def delta_x_2(self, m, n):
        if m == n - 1:
            return 1
        else:
            return 0


    """
    p operator's delta
    """
    def delta_p_1(self, m, n):
        if m == n + 1:
            return 1
        else:
            return 0

    def delta_p_2(self, m, n):
        if m == n - 1:
            return 1
        else:
            return 0


    """
    squeeze operator matrix
    """
    def squeeze_matrix(self):
        squ_mat = [[] for i in range(self.N)]

        for j in range(self.N):
            for k in range(self.N):
                squ_mat[k].append((1.0j) * ((math.sqrt(k * (k-1)) * self.delta_squeeze_2(j, k)) - (math.sqrt((k+1) * (k+2)) * self.delta_squeeze_1(j, k))))

        return np.matrix(squ_mat)


    """
    x operator matrix
    """
    def x_operator_matrix(self):
        x_op = [[] for i in range(self.N)]

        for j in range(self.N):
            for k in range(self.N):
                x_op[k].append((1 / math.sqrt(2)) * ((math.sqrt(k) * self.delta_x_1(k, j)) + (math.sqrt(k+1) * self.delta_x_2(k, j))))

        return np.matrix(x_op)

    def x2_operator_matrix(self):
        return np.dot(self.x_operator_matrix(), self.x_operator_matrix())

    def x3_operator_matrix(self):
        return np.dot(self.x2_operator_matrix(), self.x_operator_matrix())

    def x4_operator_matrix(self):
        return np.dot(self.x3_operator_matrix(), self.x_operator_matrix())


    """
    p operator matrix
    """
    def p_operator_matrix(self):
        p_op = [[] for i in range(self.N)]

        for j in range(self.N):
            for k in range(self.N):
                p_op[k].append(((-1.0j) / math.sqrt(2)) * (((math.sqrt(k)) * self.delta_p_2(j, k)) - ((math.sqrt(k+1)) * self.delta_p_1(j, k))))

        return np.matrix(p_op)

    def p2_operator_matrix(self):
        return np.dot(self.p_operator_matrix(), self.p_operator_matrix())


    """
    measurement operator matrix
    """
    def measurement_operator_matrix(self):
        return np.sqrt(self.gamma) * self.x2_operator_matrix()


    """
    System Hamiltonian
    """
    def System_Hamiltonian(self):

        phase = self.p2_operator_matrix() / 2

        position_x4 = self.x4_operator_matrix()
        position_x3 = 4 * self.a * self.x3_operator_matrix()
        position_x2 = 2 * (3 * (self.a) ** (2) - (self.b) ** (2)) * self.x2_operator_matrix()
        position_x = 4 * self.a * ((self.a) ** (2) - self.b ** (2)) * self.x_operator_matrix()
        constant = ((self.a) ** (4)) - (2 * (self.a ** (2)) * self.b) + (self.b ** (4))

        position = (self.h / (self.b ** (4))) * (position_x4 - position_x3 +position_x2 - position_x + constant)

        return Qobj(phase + position)

    """
    Squeezed Hamiltonian
    """
    def squeezed_Hamiltonian(self):
        return Qobj(self.squeeze_matrix())