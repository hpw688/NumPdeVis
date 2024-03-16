import numpy as np
import taichi as ti
import sympy as sp
import latex2sympy2


def str_to_function(expr_str, dim, dw, len1=None, len2=None, len3=None):
    """
    将 LaTeX 字符串形式的数学表达式转换为 Sympy 表达式，并根据维度返回计算后的数值列表。
    """
    # 将 LaTeX 字符串转换为 Sympy 表达式
    expr = latex2sympy2.latex2sympy(expr_str)

    # 定义符号
    x = sp.symbols('x')
    y = sp.symbols('y')
    z = sp.symbols('z')

    if dim == 1:
        f = [expr.subs(x, i * dw[1]).evalf() for i in range(int(len1 / dw[1]))]
        return f
    elif dim == 2:
        f = [[expr.subs({x: i * dw[1], y: j * dw[2]}).evalf() for j in range(int(len2 / dw[2]))] for i in
             range(int(len1 / dw[1]))]
        return f
    elif dim == 3:
        total_iterations = int(len1 / dw[1]) * int(len2 / dw[2]) * int(len3 / dw[3])
        current_iteration = 0
        f = []
        for i in range(int(len1 / dw[1])):
            layer = []
            for j in range(int(len2 / dw[2])):
                row = []
                for k in range(int(len3 / dw[3])):
                    row.append(expr.subs({x: i * dw[1], y: j * dw[2], z: k * dw[3]}).evalf())
                    current_iteration += 1
                    if current_iteration % (total_iterations // 100) == 0:  # Update progress every 1%
                        print(
                            f"Progress: {current_iteration}/{total_iterations} ({current_iteration / total_iterations * 100:.2f}%)")
                layer.append(row)
            f.append(layer)
        return f




@ti.data_oriented
class wave_equ():
    def __init__(self,func1_string,func2_string,len,dim,v,dw,a,b,t):
        self.func1_string = func1_string
        self.func2_string = func2_string
        self.len = len
        self.dim = dim
        self.v = v
        self.dw = dw
        self.a = a
        self.b = b
        self.t = t
        # dim1
        self.phi_xt = ti.field(ti.f32, shape=(int(self.len[0] / self.dw[1]), int(self.t / self.dw[0])))
        self.func1_dim1 = ti.field(ti.f32, shape=int(self.len[0] / self.dw[1]))
        self.func2_dim1 = ti.field(ti.f32, shape=int(self.len[0] / self.dw[1]))
        # dim2
        self.phi_xyt = ti.field(ti.f32,shape=(int(self.len[0] / self.dw[1]), int(self.len[1] / self.dw[2]),int(self.t / self.dw[0])))
        self.func1_dim2 = ti.field(ti.f32,shape = (int(self.len[0] / self.dw[1]), int(self.len[1] / self.dw[2])))
        self.func2_dim2 = ti.field(ti.f32,shape = (int(self.len[0] / self.dw[1]), int(self.len[1] / self.dw[2])))
        # dim3
        self.phi_xyzt = ti.field(ti.f32, shape=(int(self.len[0] / self.dw[1]), int(self.len[1] / self.dw[2]), int(self.len[2] / self.dw[3]),int(self.t / self.dw[0])))
        self.func1_dim3 = ti.field(ti.f32, shape=(int(self.len[0] / self.dw[1]), int(self.len[1] / self.dw[2]),int(self.len[2] / self.dw[3])))
        self.func2_dim3 = ti.field(ti.f32, shape=(int(self.len[0] / self.dw[1]), int(self.len[1] / self.dw[2]),int(self.len[2] / self.dw[3])))

    def init_func(self):
        f1 = str_to_function(self.func1_string, self.dim, self.dw, self.len[0], self.len[1], self.len[2])
        f2 = str_to_function(self.func2_string, self.dim, self.dw, self.len[0], self.len[1], self.len[2])
        if self.dim == 1:
            for i in range(0, int(self.len[0] / self.dw[1])):
                # 确保赋值前将结果转换为float类型
                self.func1_dim1[i] = f1[i]
                self.func2_dim1[i] = f2[i]
        if self.dim == 2:
            for i in range(0, int(self.len[0] / self.dw[1])):
                for j in range(0, int(self.len[1] / self.dw[2])):
                    self.func1_dim2[i, j] = f1[i][j]
                    self.func2_dim2[i, j] = f2[i][j]
        if self.dim == 3:
            for i in range(0, int(self.len[0] / self.dw[1])):
                for j in range(0, int(self.len[1] / self.dw[2])):
                    for k in range(0, int(self.len[2] / self.dw[3])):
                        self.func1_dim3[i, j, k] = f1[i][j][k]
                        self.func2_dim3[i, j, k] = f2[i][j][k]

    @ti.kernel
    def wave_solver_dim1(self):
        # 初始化边界条件
        a1 = self.a[0]
        b1 = self.b[0]
        v = self.v
        t = self.t
        dt = self.dw[0]
        dx = self.dw[1]
        len = self.len[0]
        self.phi_xt[0, 0] = a1
        self.phi_xt[int(len / dx) - 1, 0] = b1

        # 设置初始条件
        for i in ti.ndrange((1, int(len / dx) - 1)):
            self.phi_xt[i, 0] = self.func1_dim1[i]  # 初始位移条件
            if i < int(len / dx) - 1:
                self.phi_xt[i, 1] = dt * self.func2_dim1[i] + self.phi_xt[i, 0]  # 初速度影响的近似

        # 迭代更新
        for k in ti.ndrange((1, int(t / dt) - 1)):
            # 重新应用边界条件
            self.phi_xt[0, k + 1] = a1
            self.phi_xt[int(len / dx) - 1, k + 1] = b1

            for i in ti.ndrange((1, int(len / dx) - 1)):
                self.phi_xt[i, k + 1] = 2 * self.phi_xt[i, k] - self.phi_xt[i, k - 1] + (v * v * dt * dt) / (dx * dx) * (
                        self.phi_xt[i - 1, k] - 2 * self.phi_xt[i, k] + self.phi_xt[i + 1, k])

    @ti.kernel
    def wave_solver_dim2(self):
        a1, a2 = self.a[0], self.a[1]  # 沿x和y的边界条件
        b1, b2 = self.b[0], self.b[1]  # 沿x和y的边界条件
        v = self.v
        dt = self.dw[0]
        dx = self.dw[1]
        dy = self.dw[2]
        len1, len2 = self.len[0], self.len[1]
        for i, j in ti.ndrange((0, int(len1 / dx) - 1), (0, int(len2 / dy) - 1)):
            self.phi_xyt[i, j, 0] = self.func1_dim2[i,j]
            self.phi_xyt[i, j, 1] = dt * self.func2_dim2[i,j] + self.phi_xyt[i, j, 0]
        # 边界条件初始化，对于二维情况，需要设置四边的边界
        for j in ti.ndrange((0, int(len2 / dy))):  # x方向的边界
            self.phi_xyt[0, j, 0] = a1
            self.phi_xyt[int(len1 / dx) - 1, j, 0] = b1
        for i in ti.ndrange((0, int(len1 / dx))):  # y方向的边界
            self.phi_xyt[i, 0, 0] = a2
            self.phi_xyt[i, int(len2 / dy) - 1, 0] = b2

        # 时间迭代
        for k in ti.ndrange((2, int(self.t / dt))):  # 从k=2开始，因为k=0和k=1已通过初始条件设置
            for i, j in ti.ndrange((1, int(len1 / dx) - 1), (1, int(len2 / dy) - 1)):
                self.phi_xyt[i, j, k] = (2 * self.phi_xyt[i, j, k - 1] - self.phi_xyt[i, j, k - 2] +
                                         (v ** 2 * dt ** 2 / dx ** 2) * (
                                                     self.phi_xyt[i + 1, j, k - 1] - 2 * self.phi_xyt[i, j, k - 1] +
                                                     self.phi_xyt[i - 1, j, k - 1]) +
                                         (v ** 2 * dt ** 2 / dy ** 2) * (
                                                     self.phi_xyt[i, j + 1, k - 1] - 2 * self.phi_xyt[i, j, k - 1] +
                                                     self.phi_xyt[i, j - 1, k - 1]))

            # 更新边界条件以保持不变
            for j in ti.ndrange((0, int(len2 / dy))):  # x方向的边界
                self.phi_xyt[0, j, k] = a1
                self.phi_xyt[int(len1 / dx) - 1, j, k] = b1
            for i in ti.ndrange((0, int(len1 / dx))):  # y方向的边界
                self.phi_xyt[i, 0, k] = a2
                self.phi_xyt[i, int(len2 / dy) - 1, k] = b2

    @ti.kernel
    def wave_solver_dim3(self):
        # 定义边界条件
        a1, a2, a3 = self.a[0], self.a[1], self.a[2]
        b1, b2, b3 = self.b[0], self.b[1], self.b[2]
        v = self.v
        dt = self.dw[0]
        dx = self.dw[1]
        dy = self.dw[2]
        dz = self.dw[3]
        len1, len2, len3 = self.len[0], self.len[1], self.len[2]

        # 应用初始条件
        for i, j, k in ti.ndrange((0, int(len1 / dx)), (0, int(len2 / dy)), (0, int(len3 / dz))):
            self.phi_xyzt[i, j, k, 0] = self.func1_dim3[i, j, k]
            if self.t > 0:
                self.phi_xyzt[i, j, k, 1] = self.func2_dim3[i, j, k] * dt + self.phi_xyzt[i, j, k, 0]

        # 在第一次迭代之前固定边界条件
        for i, j, k in ti.ndrange((0, int(len1 / dx)), (0, int(len2 / dy)), (0, int(len3 / dz))):
            if i == 0 or i == int(len1 / dx) - 1:
                self.phi_xyzt[i, j, k, 0] = a1 if i == 0 else b1
            if j == 0 or j == int(len2 / dy) - 1:
                self.phi_xyzt[i, j, k, 0] = a2 if j == 0 else b2
            if k == 0 or k == int(len3 / dz) - 1:
                self.phi_xyzt[i, j, k, 0] = a3 if k == 0 else b3

        # 迭代更新
        for t in ti.ndrange((2, int(self.t / dt))):
            for i, j, k in ti.ndrange((1, int(len1 / dx) - 1), (1, int(len2 / dy) - 1), (1, int(len3 / dz) - 1)):
                self.phi_xyzt[i, j, k, t] = 2 * self.phi_xyzt[i, j, k, t - 1] - self.phi_xyzt[i, j, k, t - 2] \
                                            + (v ** 2 * dt ** 2 / dx ** 2) * (
                                                        self.phi_xyzt[i + 1, j, k, t - 1] - 2 * self.phi_xyzt[
                                                    i, j, k, t - 1] + self.phi_xyzt[i - 1, j, k, t - 1]) \
                                            + (v ** 2 * dt ** 2 / dy ** 2) * (
                                                        self.phi_xyzt[i, j + 1, k, t - 1] - 2 * self.phi_xyzt[
                                                    i, j, k, t - 1] + self.phi_xyzt[i, j - 1, k, t - 1]) \
                                            + (v ** 2 * dt ** 2 / dz ** 2) * (
                                                        self.phi_xyzt[i, j, k + 1, t - 1] - 2 * self.phi_xyzt[
                                                    i, j, k, t - 1] + self.phi_xyzt[i, j, k - 1, t - 1])

            # 在每次迭代之后重新固定边界条件
            for i, j, k in ti.ndrange((0, int(len1 / dx)), (0, int(len2 / dy)), (0, int(len3 / dz))):
                if i == 0 or i == int(len1 / dx) - 1:
                    self.phi_xyzt[i, j, k, t] = a1 if i == 0 else b1
                if j == 0 or j == int(len2 / dy) - 1:
                    self.phi_xyzt[i, j, k, t] = a2 if j == 0 else b2
                if k == 0 or k == int(len3 / dz) - 1:
                    self.phi_xyzt[i, j, k, t] = a3 if k == 0 else b3

    def print_field_dim1(self):
        for i, j in ti.ndrange((0, int(self.len[0] / self.dw[1])), (0, int(self.t / self.dw[0]))):
            print(f"x[{i}, {j}] = {self.phi_xt[i, j]}")  # 打印field的每个元素

    def print_field_dim2(self):
        for i, j, k in ti.ndrange((0, int(self.len[0] / self.dw[1])),(0, int(self.len[1] / self.dw[2])), (0, int(self.t / self.dw[0]))):
            print(f"x[{i}, {j},{k}] = {self.phi_xyt[i,j,k]}")  # 打印field的每个元素

    def print_field_dim3(self):
        for i, j, k,t in ti.ndrange((0, int(self.len[0] / self.dw[1])),(0, int(self.len[1] / self.dw[2])),(0, int(self.len[2] / self.dw[3])) ,(0, int(self.t / self.dw[0]))):
            print(f"x[{i}, {j},{k},{t}] = {self.phi_xyzt[i,j,k,t]}")  # 打印field的每个元素




    def exe_dim1_solver(self):
        self.init_func()
        self.wave_solver_dim1()
        self.print_field_dim1()

    def exe_dim2_solver(self):
        self.init_func()
        self.wave_solver_dim2()
        self.print_field_dim2()

    def exe_dim3_solver(self):
        self.init_func()
        self.wave_solver_dim3()
        self.print_field_dim3()

if __name__ == '__main__':
    ti.init(arch=ti.gpu)
    a = wave_equ("\sin(x)","\cos(x)",[10,5,5],1,0.1,[0.1,0.1,0.1,0.1],[0,0,0],[0,0,0],11)
    a.exe_dim1_solver()
    # a = wave_equ("\sin(x+y)", "\cos(x+y)", [10, 5, 5], 2, 0.1, [0.1, 0.1, 0.1, 0.1], [0, 0, 0], [0, 0, 0], 11)
    # a.exe_dim2_solver()
    # a = wave_equ("\sin(x+y+z)","\cos(x+y+z)",[10,5,5],3,0.1,[0.1,0.1,0.1,0.1],[0,0,0],[0,0,0],11)
    # a.exe_dim3_solver()

