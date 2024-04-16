#NumPdeVis编写手册
PS：默认各位已经对taichi的语法有了基本了解，本文中所用语法皆基于taichi,并且我写的代码为了节省时间没有跑过，但是算法含义一定是对的，可能存在一定的语法错误.
##数学物理中常用的偏微分方程
1.$\frac{\partial^2\phi}{\partial t^2}=a^2\nabla^2\phi$ (波动方程)

2.$\frac{\partial\phi}{\partial t}=a^2\nabla^2\phi$(热传导方程)

3.$\nabla^2\phi=0$(拉普拉斯方程)

对于一个偏微分方程，如同常微分方程一样,我们的目标是解出满足微分方程的函数$f(x)$.对于偏微分方程，也就是求得函数$\phi(r)$, $r$的维度取决于方程的维度.

##常微分方程的数值方法
为了同志们能够理解接下来要编写的用于求解偏微分方程的有限差分法，所以先讲解一下用于求解常微分方程的前向欧拉法.
首先要知道一个函数$y(x)$,我们不可能精确求出函数在任何x位置的函数值，但是可以把x数轴离散化.
考虑数轴上的n个离散化的点$\{x_0,x_1,x_2,...,x_{n-1}\}$,这n个点相距$\Delta x$,这个$\Delta x=x_{i+1}-x_{i}$的值是我们自己设置的.它们对应的值函数值为$\{y_0,y_1,y_2,...,y_{n-1}\}$
若现在有一个微分方程$y'=f(x,y)$,并且有初值$y(x_0)=y_0$.
将微分方程转化
$\begin{aligned}
    &y'=f(x,y)\\
    &\frac{dy}{dx}=f(x,y)\\  
    &dy=f(x,y)dx\\
    &\Delta y=\lim_{\Delta x->0}f(x,y)\Delta x\\
\end{aligned}$
写成差分格式
$\begin{aligned}
    y_{n+1}-y_n=f(x_n,y_n)\Delta x\\
    y_{n+1}=y_n+f(x_n,y_n)\Delta x
\end{aligned}$
之前说过，$\Delta x$的值是自己设置的，如果我们知道$y(x_0)$,那么通过上面的公式，就能够迭代求出$y_1,y_2,..,y_{n-1}$.
```python
#现有方程y'=3sin(e^x),y(3) = 4,要求解区间[3,100]的函数值.
import taichi as ti
import taichi.math as tm

ti.init(arch = ti.gpu)

y0 = 4
dx = 0.01

@ti.func
def y_diff(x:ti.f32):
    return tm.sin(tm.exp(x))

@ti.kernel
def nde_solver():
    y_solve = ti.field(ti.f32,shape((100-3)/dx))
    x = 3
    y_solve[0] = y0
    for i in y_solve:
        x = x + dx
        y_solve[i+1] = y_solve[i] + y_diff(x)*dx
    return y_solve
```
##偏微分方程的数值解法
对于一个偏微分方程，首先要明确方程解的维度,这会让我们将$\nabla^2$算子理解成不同的形式.
在一维空间中，$\nabla^2$为$\frac{\partial^2}{\partial x^2}$,此时解为二元函数$\phi(x,t)$.
$\begin{aligned}
    &\frac{\partial^2\phi}{\partial t^2}=a^2\frac{\partial^2\phi}{\partial x^2}\\
    &\frac{\partial\phi}{\partial t}=a^2\frac{\partial^2\phi}{\partial x^2}\\
    &\frac{\partial^2\phi}{\partial x^2}=0
\end{aligned}$

在二维空间中,$\nabla^2$为$\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}$,此时解为三元函数$\phi(x,y,t)$.
$\begin{aligned}
    &\frac{\partial^2\phi}{\partial t^2}=a^2(\frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2})\\
    &\frac{\partial\phi}{\partial t}=a^2(\frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2})\\
    &\frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2}=0
\end{aligned}$

在三维空间中,$\nabla^2$为$\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}+\frac{\partial^2}{\partial z^2}$,此时解为四元函数$\phi(x,y,z,t)$.
$\begin{aligned}
    &\frac{\partial^2\phi}{\partial t^2}=a^2(\frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2}+\frac{\partial^2\phi}{\partial z^2})\\
    &\frac{\partial\phi}{\partial t}=a^2(\frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2}+\frac{\partial^2\phi}{\partial z^2})\\
    &\frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2}+\frac{\partial^2\phi}{\partial z^2}=0
\end{aligned}$

###波动方程
$\frac{\partial^2\phi}{\partial t^2}=a^2\nabla^2\phi$ 

对于波动方程，顾名思义其解为一个描述波如何在介质中传播的函数,在高中物理中，我们学过一维简谐振动的方程为$y(t)=Asin(wt+\phi_0)$,这是波上一个点的振动方程，对于波中任意一点，有方程$y(x,t)=Asin(w(t-\frac{x}{v})+\phi_0)$,这里的v是波速，这就是一维简谐波的方程，它是波动方程最简单的一个解.
重点在于我们可以把波动方程的解$\phi(x,t)$的函数值大小理解为波的振幅.要描述更复杂的波动，我们就需要求解波动方程.

现在以波动方程为例来讲解有限差分法.

考虑二维的波动方程，和常微分方程一样，首先将数轴离散化.对于函数$\phi(x,y,t)$,有
$\{x_0,x_1,x_2,...,x_n\},\{y_0,y_1,y_2,...,y_n\},\{t_0,t_1,t_2,...,t_n\},x_{i+1}-x_i=\Delta x,y_{j+1}-y_j=\Delta y,t_{k+1}-t_k=\Delta t$.此时我们可以将函数$\phi(x,y,t)$理解为一个三维数组phi[x][y][t],而具体到每一个点，那么就有$\phi(x_i,y_j,t_k)=phi[i][j][k]$
这里我先写出常用的近似方法
对于函数$\phi$的一阶偏导可以直接近似为
$\begin{aligned}
    \frac{\partial \phi(x,y,t)}{\partial t}|_{x=x_i,y=y_j,t=t_k} \approx \frac{phi[i][j][k+1]-phi[i][j][k]}{\Delta t}
\end{aligned}$
其二阶偏导可以用中心差商公式近似为
$\begin{aligned}
    \frac{\partial^2 \phi(x,y,t)}{\partial t^2}|_{x=x_i,y=y_j,t=t_k} \approx \frac{phi[i][j][k-1]-2phi[i][j][k]+phi[i][j][k+1]}{{\Delta t}^2}\\
    \frac{\partial^2 \phi(x,y,t)}{\partial x^2}|_{x=x_i,y=y_j,t=t_k} \approx \frac{phi[i-1][j][k]-2phi[i][j][k]+phi[i+1][j][k]}{{\Delta x}^2}\\
    \frac{\partial^2 \phi(x,y,t)}{\partial y^2}|_{x=x_i,y=y_j,t=t_k} \approx \frac{phi[i][j-1][k]-2phi[i][j][k]+phi[i][j+1][k]}{{\Delta y}^2}
\end{aligned}$
那么原来的二维波动方程就被近似成了
$\begin{aligned}
   \frac{phi[i][j][k-1]-2phi[i][j][k]+phi[i][j][k+1]}{{\Delta t}^2}=a^2\frac{phi[i-1][j][k]-2phi[i][j][k]+phi[i+1][j][k]}{{\Delta x}^2}+a^2\frac{phi[i][j-1][k]-2phi[i][j][k]+phi[i][j+1][k]}{{\Delta y}^2}
\end{aligned}$
现在只要给出方程的边界条件，我们就可以通过迭代式子算出每一个点在不同时间的振动幅度.

####边界条件的设定
对于一维波动方程，其典型的边界条件为
$\begin{aligned}
    &\phi|_{t=0}=u(x),\frac{\partial\phi}{\partial t}|_{t=0}=\Phi(x)\\
    &\phi|_{x=0}=a,\phi|_{x=L}=b
\end{aligned}$
第一个条件规定了波在$t=0$时刻的各个点的振幅，第二个条件规定了波传导的速度,第三个和第四个条件规定了波的边界是固定的.
对于二维波动方程，
$\begin{aligned}
    &\phi|_{t=0}=u(x,y),\frac{\partial\phi}{\partial t}|_{t=0}=\Phi(x,y)\\
    &\phi|_{x=0}=a_1,\phi|_{x=L}=b_1\\
    &\phi|_{y=0}=a_2,\phi|_{y=L}=b_2
\end{aligned}$
对于三维波动方程
$\begin{aligned}
    &\phi|_{t=0}=u(x,y,z),\frac{\partial\phi}{\partial t}|_{t=0}=\Phi(x,y,z)\\
    &\phi|_{x=0}=a_1,\phi|_{x=L}=b_1\\
    &\phi|_{y=0}=a_2,\phi|_{y=L}=b_2\\
    &\phi|_{z=0}=a_3,\phi|_{z=L}=b_3
\end{aligned}$
先以一维波动方程为例讲解如何使用迭代式求解
直接给出
$\begin{aligned}
    &\frac{phi[i][k-1]-2phi[i][k]+phi[i][k+1]}{\Delta t^2}=a^2\frac{phi[i-1][k]-2phi[i][k]+phi[i+1][k]}{\Delta x^2} \\变形得到\\
    &phi[i][k+1]=2phi[i][k]-phi[i][k-1]+\frac{a^2\Delta t^2}{\Delta x^2}\{\frac{phi[i-1][k]-2phi[i][k]+phi[i+1][k]}{\Delta x^2}\}
\end{aligned}$
现在根据上式就可以求出任意时刻波在$x_i$的值.
一个有趣的问题是我们根据初始条件只知道$phi[0][k]=a,phi[L][k]=b,phi[i][0]=u(x_i)$,但是迭代式子中却需要通过两个时刻$k,k-1$来求出时刻$k+1$的值，我们还有一个条件是$\frac{\partial\phi}{\partial t}|_{t=0}=\Phi(x)$,这个式子也可以写成迭代式的形式,

$\begin{aligned}
    &\frac{\partial \phi(x,t)}{\partial t}|_{x=x_i,t=t_k} \approx \frac{phi[i][k+1]-phi[i][k]}{\Delta t}\\
    在边界条件下这也就是说\\
    &\frac{phi[i][k+1]-phi[i][k]}{\Delta t}=\Phi(x_i)\\
    把时间的下标换一下\\
    &\frac{phi[i][k]-phi[i][k-1]}{\Delta t}=\Phi(x_i)\\
    就得到了\\
    &phi[i][k]=\Delta t \Phi(x_i)+phi[i][k-1]
\end{aligned}$
那么根据上式，我们就可以在每次迭代中先把$k$时刻有关的值先算出来，再放入到迭代式进行迭代，下面是代码展示
```python
import taichi as ti
import taichi.math as tm
#考虑边界条件为u(x)=sin(x),\Phi(x) = cos(x),\phi(0,t)=0,\phi(10,t)=5,求0-10s内的值

a = 0.1
dt = 0.1
dx = 0.1
wave_phi = ti.field(ti.f32,shape((5-0)/dx),(10-0)/dt)


@ti.func
def u_x(x:ti.f32):
    return tm.sin(x)

@ti.func
def Phi_x(x:ti.f32):
    return tm.cos(x)


@ti.func
def init_wave():
    x_i = 0 
    for i in wave_phi:
        wave_phi[i][0] = u_x(x_i)
        x_i = x_i + i*dx
    
    wave_phi[0][0] = 0
    wave_phi[(10-0)/dt - 1][0] = 0

@ti.kernel
def solve_wave():
    x_i = 0
    t_i = 0
    for i in wave_phi:
        for k in wave_phi:
            if i == 0:
                break
                if k == 0:
                    wave_phi[i][k+1] = dt*Phi_x(x_i + i*dx) + wave_phi[i][k]
                wave_phi[k+2] = 2*wave_phi[i][k+1] - wave_phi[i][k] + a*a*dt*dt*(wave_phi[i-1][k+1]-2*wave_phi[i][k+1]+wave_phi[i+1][k+1])/(dx*dx*dx*dx)
                if i == ((10-0)/dt - 1):
                    break
        if i == 0:
            continue
        if i == ((10-0)/dt - 1):
            break
    return wave_phi

```
###热传导方程
$\frac{\partial\phi}{\partial t}=a^2\nabla^2\phi$
这里的函数值表达的就是该点在该时刻的温度
后面我就不放迭代式的推导了，各位可以自己试着推一下，我以后会写在纸上发出来
####边界条件
$\begin{aligned}
一维\\
    &\phi|_{t=0} = \phi(x),\phi|_{x=0}=a,\phi|_{x=l}=b\\    
二维\\
    &\phi|_{t=0} = \phi(x,y),\phi|_{x=0}=a_1,\phi|_{x=l_1}=b_1,\phi|_{y=0}=a_2,\phi|_{y=l_2}=b_2\\
三维\\
    &\phi|_{t=0} = \phi(x,y,z),\phi|_{x=0}=a_1,\phi|_{x=l_1}=b_1,\phi|_{y=0}=a_2,\phi|_{y=l_2}=b_2,\phi|_{z=0}=a_3,\phi|_{z=l_3}=b_3\\
\end{aligned}$

###拉普拉斯方程
$\nabla^2\phi=0$
这里的函数值通常情况下表达的是该点的电势大小
####边界条件
$\begin{aligned}
一维\\
    &\phi|_{x=0}=a,\phi|_{x=l}=b\\ 
二维\\
    &\phi|_{x=0}=a_1,\phi|_{x=l_1}=b_1,\phi|_{y=0}=a_2,\phi|_{y=l_2}=b_2\\
三维\\
    &\phi|_{x=0}=a_1,\phi|_{x=l_1}=b_1,\phi|_{y=0}=a_2,\phi|_{y=l_2}=b_2,\phi|_{z=0}=a_3,\phi|_{z=l_3}=b_3
\end{aligned}$
##后记
为了让这个软件看起来完整一点,我寻思直接把它做成类似于maple,geogebra类似的计算软件算了，因为加上普通科学计算器的功能几乎不需要更多的时间，用numpy就能全部完成，然后使用matplotlib绘制曲线图像，这些功能都可以让chatgpt写出来。
然后就是用户输入的问题，python上有一个库叫sympy可以直接将字符串转化为数学函数。
最后就是做个界面就完事了.

$END$


