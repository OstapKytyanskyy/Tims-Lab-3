import numpy as np
import functools


class AnalysisOfVariance:
    def __init__(self, data=None, n=None, m=None):
        if data is None:
            self.data = np.array([
                [25.2, 10.2, 5.4, 13.2, 18.2, 5.2, 13.4, 15.2, 4.5, 19.2],
                [10.6, 8.4, 11.2, 4.6, 5.8, 18.2, 16.4, 13.2, 4.8, 8.9],
                [2.5, 6.4, 12.5, 14.8, 12.3, 8.5, 5.9, 8.9, 15.4, 12.8, 4.2, 3.9],
                [4.3, 10.5, 20.3, 32.4, 5.6, 12.4, 6.2, 9.8, 16.8, 18.4],
                [12.4, 4.3, 13.2, 5.6, 8.9, 14.8, 22.3, 6.8, 7.2, 11.4, 4.2],
                [4.5, 4.9, 12.3, 15.6, 7.9, 8.9, 9.8, 13.9, 4.2, 6.9],
                [14.3, 10.6, 28.4, 10.8, 7.4, 6.5, 4.5, 26.3, 30.2, 11.8],
                [6.2, 7.5, 3.5, 12.4, 13.5, 16.4, 7.9, 8.9, 15.4, 10.8],
                [14.8, 2.9, 5.9, 10.6, 8.5, 13.4, 2.2, 19.5, 7.9, 9.9]
            ])
        else:
            self.data = data
        self.arr = []
        self.m = m
        self.n = n
        self.arr_i_dot = []
        self.arr_dot_j = []
        self.x_dot_dot = []
        self.var_A = 0
        self.var_B = 0
        self.var_remain = 0
        self.F_emp_A = 0
        self.F_emp_B = 0

    def calc(self):
        self.arr = np.array([np.around(np.average(i), decimals=2) for i in self.data]).reshape(self.m, self.n)
        # self.arr = data
        self.arr_i_dot = [np.around(np.average(i), decimals=2) for i in self.arr]
        self.arr_dot_j = [np.around(np.average(i), decimals=2) for i in self.arr.T]
        self.x_dot_dot = np.around(sum(functools.reduce(lambda x, y: x + y, self.arr)) / (self.m * self.n), decimals=2)

    def __str__(self):
        return f"data\n{self.arr}\narr_i_dot\t{self.arr_i_dot}\n" \
            f"arr_dot_j\t{self.arr_dot_j}\nx_dot_dot\t{self.x_dot_dot}\nstd_a\t{self.var_A}\nstd_B\t{self.var_B}\n" \
            f"std_remain\t{self.var_remain}\nF_Emp_A\t{self.F_emp_A}\nF_Emp_B\t{self.F_emp_B}"

    def std_a(self):
        self.var_A = sum([(i - self.x_dot_dot) ** 2 for i in self.arr_i_dot]) * (self.m / (self.n - 1))

    def std_b(self):
        self.var_B = sum([(i - self.x_dot_dot) ** 2 for i in self.arr_dot_j]) * (self.n / (self.m - 1))

    def std_remain(self):
        self.var_remain = sum([(self.arr[i][j] - self.arr_i_dot[i] - self.arr_dot_j[j] + self.x_dot_dot) ** 2
                               for i in range(self.n) for j in range(self.m)]) / ((self.n - 1) * (self.m - 1))

    def f_emp_a(self):
        self.F_emp_A = self.var_A / self.var_remain

    def f_emp_b(self):
        self.F_emp_B = self.var_B / self.var_remain

# data = np.array([
#     [25.2, 10.2, 5.4, 13.2, 18.2, 5.2, 13.4, 15.2, 4.5, 19.2],
#     [10.6, 8.4, 11.2, 4.6, 5.8, 18.2, 16.4, 13.2, 4.8, 8.9],
#     [2.5, 6.4, 12.5, 14.8, 12.3, 8.5, 5.9, 8.9, 15.4, 12.8, 4.2, 3.9],
#     [4.3, 10.5, 20.3, 32.4, 5.6, 12.4, 6.2, 9.8, 16.8, 18.4],
#     [12.4, 4.3, 13.2, 5.6, 8.9, 14.8, 22.3, 6.8, 7.2, 11.4, 4.2],
#     [4.5, 4.9, 12.3, 15.6, 7.9, 8.9, 9.8, 13.9, 4.2, 6.9],
#     [14.3, 10.6, 28.4, 10.8, 7.4, 6.5, 4.5, 26.3, 30.2, 11.8],
#     [6.2, 7.5, 3.5, 12.4, 13.5, 16.4, 7.9, 8.9, 15.4, 10.8],
#     [14.8, 2.9, 5.9, 10.6, 8.5, 13.4, 2.2, 19.5, 7.9, 9.9]
# ])
data = np.array([
    [25, 20, 30, 25],
    [30, 40, 40, 50],
    [23, 18, 20, 27]
])
a = AnalysisOfVariance( n=3, m=3)
a.calc()
a.std_a()
a.std_b()
a.std_remain()
a.f_emp_a()
a.f_emp_b()
print(a)

m = 3
n = 4
# for i in range(len(data)):
#     data[i] = np.around(np.average(data[i]), decimals=2)

# data = data.reshape(m, n)
# print(data)
# arr_i_dot = [np.around(np.average(i), decimals=2) for i in data]
# print("arr_i_dot", arr_i_dot, sep="\t")
# data = data.T
# arr_dot_j = [np.around(np.average(i), decimals=2) for i in data]
# print("arr_dot_j", arr_dot_j, sep="\t")
#
# x_dot_dot = np.around(sum(functools.reduce(add, data)) / (m * n), decimals=2)
# print("x_dot_dot", x_dot_dot, sep="\t")
#
#
# def std_A(arr_i_dot, x_dor_dot):
#     print(sum([(i - x_dor_dot) ** 2 for i in arr_i_dot]) * (m / (n - 1)))


# def std_B(arr_dot_j, x_dor_dot):
#     print(sum([(i - x_dor_dot) ** 2 for i in arr_dot_j]) * (n / (m - 1)))
#
# std_A(arr_i_dot, x_dot_dot)
# std_B(arr_dot_j, x_dot_dot)
