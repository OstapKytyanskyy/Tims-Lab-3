import numpy as np
import functools
from scipy.stats import f
import pandas as pd


class AnalysisOfVariance:
    def __init__(self, alpha, data=None, n=None, m=None):
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
        self.alpha = alpha / 2
        self.arr = []
        self.m = m
        self.n = n
        self.arr_i_dot = []
        self.arr_dot_j = []
        self.x_dot_dot = []
        self.var_A = 0
        self.var_B = 0
        self.var_remain = 0
        self.var_full = 0
        self.F_emp_A = 0
        self.F_emp_B = 0
        self.F_emp_Full = 0
        self.F_cr_A = 0
        self.F_cr_B = 0
        self.F_cr_Full = 0

    def calc(self):
        self.arr = np.array([np.around(np.average(i), decimals=2) for i in self.data]).reshape(self.m, self.n)
        # self.arr = data
        self.arr_i_dot = [np.around(np.average(i), decimals=2) for i in self.arr]
        self.arr_dot_j = [np.around(np.average(i), decimals=2) for i in self.arr.T]
        self.x_dot_dot = np.around(sum(functools.reduce(lambda x, y: x + y, self.arr)) / (self.m * self.n), decimals=2)
        self.std_a()
        self.std_b()
        self.std_remain()
        self.std_full()
        self.f_a()
        self.f_b()
        self.f_full()

    def __str__(self):
        return f"data\n{self.arr}\narr_i_dot\t{self.arr_i_dot}\n" \
            f"arr_dot_j\t{self.arr_dot_j}\nx_dot_dot\t{self.x_dot_dot}\nstd_a\t{self.var_A}\nstd_B\t{self.var_B}\n" \
            f"std_remain\t{self.var_remain}\nF_Emp_A\t{self.F_emp_A}\nF_Emp_B\t{self.F_emp_B}\nF_cr_A\t{self.F_cr_A}" \
            f"\nF_cr_B\t{self.F_cr_B}"

    def data_frame(self):
        info = {
            "X_i.": str(self.arr_i_dot),
            "x_._j": str(self.arr_dot_j),
            "X..": self.x_dot_dot,
            "Var A": self.var_A,
            "Var B": self.var_B,
            "Var remain": self.var_remain,
            "Var Full": self.var_full,
            "F Emp A": self.F_emp_A,
            "F Emp B": self.F_emp_B,
            "F Emp Full": self.F_emp_Full,
            "F Cr A": self.F_cr_A,
            "F Cr B": self.F_cr_B,
            "F Cr Full": self.F_cr_Full
        }
        df = pd.DataFrame(info, index=[1]).T
        df = df.astype('object')
        print(df)

    def std_a(self):
        self.var_A = sum([(i - self.x_dot_dot) ** 2 for i in self.arr_i_dot]) * (self.m / (self.n - 1))

    def std_b(self):
        self.var_B = sum([(i - self.x_dot_dot) ** 2 for i in self.arr_dot_j]) * (self.n / (self.m - 1))

    def std_remain(self):
        self.var_remain = sum([(self.arr[i][j] - self.arr_i_dot[i] - self.arr_dot_j[j] + self.x_dot_dot) ** 2
                               for i in range(self.n) for j in range(self.m)]) / ((self.n - 1) * (self.m - 1))

    def std_full(self):
        self.var_full = (self.n - 1) / (self.n * self.m - 1) * self.var_A \
                        + (self.m - 1) / (self.m * self.n - 1) * self.var_B \
                        + ((self.m - 1) * (self.n - 1) / (self.m * self.n - 1)) * self.var_remain

    def f_a(self):
        self.F_emp_A = self.var_A / self.var_remain if self.var_A > self.var_remain else self.var_remain / self.var_A
        self.F_cr_A = f.ppf(1 - self.alpha, self.n - 1, (self.n - 1) * (self.m - 1)) if self.var_A > self.var_remain \
            else f.ppf(1 - self.alpha, (self.n - 1) * (self.m - 1), self.n - 1)

    def f_b(self):
        self.F_emp_B = self.var_B / self.var_remain if self.var_B > self.var_remain else self.var_remain / self.var_B
        self.F_cr_B = f.ppf(1 - self.alpha, self.m - 1, ((self.m - 1) * (self.n - 1))) if self.var_B > self.var_remain \
            else f.ppf(1 - self.alpha, ((self.m - 1) * (self.m - 1)), self.m - 1)

    def f_full(self):
        self.F_emp_Full = self.var_full / self.var_remain if self.var_full > self.var_remain\
            else self.var_remain / self.var_full
        self.F_cr_Full = f.ppf(1 - self.alpha, (self.n * self.m) - 1, (self.n - 1) * (self.m - 1)) \
            if self.var_full > self.var_remain \
            else f.ppf(1 - self.alpha, (self.n  - 1) * (self.m - 1), (self.n * self.m) - 1)


# data = np.array([
#     [25, 20, 30, 25],
#     [30, 40, 40, 50],
#     [23, 18, 20, 27]
# ])
pd.set_option('display.max_columns', 11)
a = AnalysisOfVariance(0.05, n=3, m=3)
a.calc()
a.data_frame()

print(f.ppf(0.95, 2, 6))
