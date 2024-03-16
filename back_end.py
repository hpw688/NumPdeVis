import latex2sympy2
import numpy as np
import sympy as sp
from re import findall, split, MULTILINE

from matplotlib import pyplot as plt
from sympy import solve as solving, latex, sympify, Matrix, re
from latex2sympy2 import latex2sympy, latex2latex


class calculator_normal():
    def __init__(self, latex_string = None,model = None):
        self.func_strings = []
        self.func_strings.append(latex_string)
        self.model = model

    def add_string(self,string):
        self.func_strings.append(string)

    def nonlinear_equation_solver(self,string):
        x = sp.symbols('x')
        equation = latex2sympy2.latex2sympy(string)
        derivative = sp.diff(equation, x)

        f = sp.lambdify(x, equation, "numpy")
        f_prime = sp.lambdify(x, derivative, "numpy")

        roots = []
        num_guesses = 10
        tolerance = 1e-10
        max_iterations = 1000
        unique_tolerance = 1e-5

        for _ in range(num_guesses):
            x_n = np.random.uniform(-100, 100)
            iteration = 0

            while True:
                f_prime_val = f_prime(x_n)
                if abs(f_prime_val) < tolerance:  # 避免除以零
                    break
                x_n1 = x_n - f(x_n) / f_prime_val
                if abs(x_n1 - x_n) < tolerance or iteration >= max_iterations:
                    # 检查新解是否与已找到的解足够不同
                    if not any(abs(x_n1 - root) < unique_tolerance for root in roots):
                        roots.append(x_n1)
                    break
                x_n = x_n1
                iteration += 1

        return roots

    def solve_integral(self,string):
        try:
            # 使用latex2sympy2将LaTeX字符串转换为sympy表达式
            expression = latex2sympy2.latex2sympy(string)
            # 定义变量x
            x = sp.symbols('x')

            # 检查表达式是否为定积分（有边界）
            if isinstance(expression, sp.Integral):
                if len(expression.limits) == 1 and len(expression.limits[0]) == 3:
                    # 提取积分变量和积分区间（定积分）
                    _, a, b = expression.limits[0]
                    solution = sp.integrate(expression.function, (x, a, b))
                else:
                    # 处理不定积分
                    solution = expression.doit()
                    solution = latex2sympy2.latex(solution)
            else:
                # 若表达式不是积分，则直接返回
                return "表达式不是一个积分"

            return solution
        except Exception as e:
            # 处理可能出现的任何异常，并返回错误信息
            return f"发生错误: {str(e)}"


    def normal_solve(self,string):
        func = latex2sympy2.latex2sympy(string)
        # func.evalf
        return func.evalf()

    def latex_to_sympy_matrix(self,string):
        """
        将LaTeX格式的矩阵字符串转换为Sympy Matrix对象。
        """
        # 提取矩阵的元素
        elements = re.findall(r'\\begin{(?:pmatrix|bmatrix)}(.*?)\\end{(?:pmatrix|bmatrix)}', string, re.S)
        if not elements:
            raise ValueError("未找到有效的矩阵")

        matrix_elements = []
        for element in elements:
            # 分割每一行
            rows = element.strip().split('\\\\')
            matrix_row = []
            for row in rows:
                # 分割每一行的元素
                nums = row.split('&')
                matrix_row.append([sympify(num.strip()) for num in nums])
            matrix_elements.append(matrix_row)

        # 只处理第一个矩阵
        if matrix_elements:
            matrix_elements = matrix_elements[0]
            return Matrix(matrix_elements)
        else:
            raise ValueError("矩阵解析错误")

    def solve_latex_matrix_equation(self,string):
        def split_and_clean_matrices(self, matrix_str):
            # 首先，基于"="分割字符串以区分两个矩阵
            split_str = matrix_str.split('=')

            # 清除第一个矩阵中的"X"（如果存在）
            matrix1 = split_str[0].replace('*X', '').strip()

            # 第二个矩阵信息直接赋值
            matrix2 = split_str[1].strip()

            return matrix1, matrix2
        """
        从用户输入的LaTeX字符串中解析矩阵方程，并求解。
        """
        # 假设latex_str是形如"A = ..., B = ..."的字符串，其中...是矩阵
        # 分离A和B
        A_latex, B_latex = split_and_clean_matrices(string)
        A = self.latex_to_sympy_matrix(A_latex)
        B = self.latex_to_sympy_matrix(B_latex)

        # 求解AX=B，这里简化处理，直接使用矩阵求逆的方式求解
        X = A.inv() * B

        return X

    def add_matrix(self,string):
        def split_and_clean_matrices(self, matrix_str):
            # 首先，基于"="分割字符串以区分两个矩阵
            split_str = matrix_str.split('+')

            # 清除第一个矩阵中的"X"（如果存在）
            matrix1 = split_str[0].strip()

            # 第二个矩阵信息直接赋值
            matrix2 = split_str[1].strip()

            return matrix1, matrix2
        A_latex, B_latex = split_and_clean_matrices(string)
        A = self.latex_to_sympy_matrix(A_latex)
        B = self.latex_to_sympy_matrix(B_latex)
        X = A+B

        return X

    def sub_matrix(self,string):
        def split_and_clean_matrices(self, matrix_str):
            # 首先，基于"="分割字符串以区分两个矩阵
            split_str = matrix_str.split('-')

            # 清除第一个矩阵中的"X"（如果存在）
            matrix1 = split_str[0].strip()

            # 第二个矩阵信息直接赋值
            matrix2 = split_str[1].strip()

            return matrix1, matrix2
        A_latex, B_latex = split_and_clean_matrices(string)
        A = self.latex_to_sympy_matrix(A_latex)
        B = self.latex_to_sympy_matrix(B_latex)
        X = A + B

        return X

    def dot_matrix(self,string):
        def split_and_clean_matrices(self, matrix_str):
            # 首先，基于"="分割字符串以区分两个矩阵
            split_str = matrix_str.split('*')

            # 清除第一个矩阵中的"X"（如果存在）
            matrix1 = split_str[0].strip()

            # 第二个矩阵信息直接赋值
            matrix2 = split_str[1].strip()

            return matrix1, matrix2
        A_latex, B_latex = split_and_clean_matrices(string)
        A = self.latex_to_sympy_matrix(A_latex)
        B = self.latex_to_sympy_matrix(B_latex)
        X = A * B

        return X

    def inv_matrix(self,string):
        A = self.latex_to_sympy_matrix(string)
        X = A.inv()

        return X

    def plot_from_latex_multiple(self):
        plt.style.use('ggplot')  # 选择一个炫酷的图形样式
        plt.figure(figsize=(10, 6))  # 调整图形的大小

        # 定义变量 x
        x = sp.symbols('x')
        x_vals = np.linspace(-10, 10, 400)  # 定义 x 轴上的点

        for string in self.func_strings:
            expr = latex2sympy2.latex2sympy(string)
            f = sp.lambdify(x, expr, 'numpy')
            y_vals = f(x_vals)
            plt.plot(x_vals, y_vals, label='$' + sp.latex(expr) + '$', linestyle='-', linewidth=2)

        plt.fill_between(x_vals, 0, y_vals, color="skyblue", alpha=0.4)  # 添加填充效果
        plt.legend(loc='best')
        plt.grid(True)  # 显示网格
        plt.title('炫酷的函数图像')  # 添加标题
        plt.xlabel('$x$')  # 添加 x 轴标签
        plt.ylabel('$f(x)$')  # 添加 y 轴标签
        plt.show()

    def exe_test(self):
        for i in self.func_strings:
            self.plot_from_latex_multiple()

if __name__ == '__main__':
    equation_solver = calculator_normal("x", 0)
    equation_solver.add_string("x^2")
    equation_solver.exe_test()



