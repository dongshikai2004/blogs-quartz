# online learning
1. 按顺序给出 训练示例 $(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\dots,(x^{(m)},y^{(m)})$ 
2. 先看到 $x^{(1)}$，做出预测 $y^{(1)}$
3. 与真实的 $y^{(1)}$ 比较
4. 继续 $x^{(2)}$

# perceptron Algorithm
约定
$y\in\{-1,1\}$,$\theta\in\mathbb R^{n+1}$
$$h _ { \theta } ( x ) = g ( \theta ^ { T } x )$$
其中
$$g ( z ) = \left\{ \begin{array} { l l } { { 1 } } & { { \mathrm { i f ~ } z \geq 0 } } \\ { { - 1 } } & { { \mathrm { i f ~ } z < 0 } } \end{array} \right.$$
更新规则
1. 如果 $h_\theta(x)=y$，不变
2. 如果 $h_\theta(x)\neq y$ ,$\theta:=\theta+yx$

新的 $\theta^Tx$ 将向 $y$ 移动
$$
\begin{aligned}
\theta_{new}^T x &= (\theta_{old} + yx)^T x \\
&= \theta_{old}^T x + y(x^T x) \\
&= \theta_{old}^T x + y\|x\|^2
\end{aligned}
$$
$||x||^2$ 是正数，$y||x||^2$ 符号 和 $y$ 相同

# The Perceptron Convergence Theorem
给出了感知机算法在线学习错误次数的上界
#### 定理假设
给定一个样本序列 $(x^{(1)}, y^{(1)}), \dots, (x^{(m)}, y^{(m)})$，满足以下两个条件：
1.  **输入有界**：对于所有 $i$，输入向量的范数有上界，即 $\|x^{(i)}\| \le D$。
2.  **存在间隔（Margin）**：存在一个单位长度的向量 $u$（$\|u\|_2 = 1$），使得所有样本都至少以间隔 $\gamma$ 被正确分类。即对于所有样本：
    $$y^{(i)} \cdot (u^T x^{(i)}) \ge \gamma$$
    *   这意味着如果 $y^{(i)}=1$，则 $u^T x^{(i)} \ge \gamma$。
    *   如果 $y^{(i)}=-1$，则 $u^T x^{(i)} \le -\gamma$。
    *   这说明数据是线性可分的，且分离超平面 $u$ 与最近的数据点之间至少有距离 $\gamma$。

#### 定理结论
感知机算法在这个序列上产生的**总错误次数（mistakes）** $k$ 满足：
$$k \le \left(\frac{D}{\gamma}\right)^2$$

#### 结论的重要性
*   这个上界**不显式依赖于样本数量 $m$**。
*   这个上界**不显式依赖于输入维度 $n$**。
*   它只取决于数据的“大小”($D$) 和数据的“可分性间隔”($\gamma$)。如果数据间隔很大（$\gamma$ 大）且输入有界，算法会很快停止犯错。


## 定理证明详解

证明的核心思想是通过分析权重向量 $\theta$ 在两个不同方面的增长情况，从而夹逼出错误次数 $k$ 的上限。

设 $\theta^{(k)}$ 为算法犯第 $k$ 次错误时所使用的权重向量。初始权重 $\theta^{(1)} = \vec{0}$。
假设第 $k$ 次错误发生在样本 $(x^{(i)}, y^{(i)})$ 上。

#### 第一步：分析预测错误的条件
既然在第 $k$ 次错误时，预测值 $g((x^{(i)})^T \theta^{(k)})=g((\theta^{(k)})^Tx^{(i)} \neq y^{(i)}$，这意味着 $\theta^{(k)}$ 与 $x^{(i)}$ 的点积符号与 $y^{(i)}$ 相反（或为 0）。
因此我们可以得到不等式：
$$(x^{(i)})^T \theta^{(k)} y^{(i)} \le 0 $$

#### 第二步：权重更新规则
根据感知机更新规则，犯错后权重变为：
$$\theta^{(k+1)} = \theta^{(k)} + y^{(i)} x^{(i)}$$

#### 第三步：推导 $\theta$ 与最优向量 $u$ 的点积下界
我们考察更新后的权重 $\theta^{(k+1)}$ 与理想分离向量 $u$ 的点积：
$$
\begin{aligned}
(\theta^{(k+1)})^T u &= (\theta^{(k)} + y^{(i)} x^{(i)})^T u \\
&= (\theta^{(k)})^T u + y^{(i)} (x^{(i)})^T u
\end{aligned}
$$
根据定理假设 $y^{(i)} (u^T x^{(i)}) \ge \gamma$，代入上式：
$$(\theta^{(k+1)})^T u \ge (\theta^{(k)})^T u + \gamma$$
通过归纳法（因为初始 $\theta^{(1)}=0$），犯第 $k$ 次错误后（即得到 $\theta^{(k+1)}$）：
$$(\theta^{(k+1)})^T u \ge k\gamma \quad \dots \text{(公式 3)}$$
**物理意义**：每次犯错更新，权重向量在理想方向 $u$ 上的投影至少增加 $\gamma$。

#### 第四步：推导 $\theta$ 的模长上界
我们考察更新后权重的模长平方：
$$
\begin{aligned}
\|\theta^{(k+1)}\|^2 &= \|\theta^{(k)} + y^{(i)} x^{(i)}\|^2 \\
&= \|\theta^{(k)}\|^2 + \|x^{(i)}\|^2 + 2 y^{(i)} (x^{(i)})^T \theta^{(k)}
\end{aligned}
$$
这里利用了两个条件：
1.  由公式 (2) 可知，交叉项 $2 y^{(i)} (x^{(i)})^T \theta^{(k)} \le 0$（因为这是在犯错时更新的）。
2.  由定理假设 $\|x^{(i)}\| \le D$，所以 $\|x^{(i)}\|^2 \le D^2$。

因此：
$$\|\theta^{(k+1)}\|^2 \le \|\theta^{(k)}\|^2 + D^2$$
通过归纳法（初始 $\|\theta^{(1)}\|^2 = 0$），犯第 $k$ 次错误后：
$$\|\theta^{(k+1)}\|^2 \le k D^2 \quad \dots \text{(公式 5)}$$
即 $\|\theta^{(k+1)}\| \le \sqrt{k} D$。
**物理意义**：每次犯错更新，权重向量的长度增长是有限的。

#### 第五步：结合上下界得出结论
利用向量投影的性质。对于任意向量 $z$ 和单位向量 $u$，有 $z^T u \le \|z\| \cdot \|u\| = \|z\|$（因为 $\|u\|=1$）。
将 $z$ 替换为 $\theta^{(k+1)}$，我们有：
$$\|\theta^{(k+1)}\| \ge (\theta^{(k+1)})^T u$$

将第三步和第四步的结果代入这个不等式：
$$\sqrt{k} D \ge \|\theta^{(k+1)}\| \ge (\theta^{(k+1)})^T u \ge k\gamma$$

即：
$$\sqrt{k} D \ge k\gamma$$

两边平方（$k, D, \gamma$ 均为非负）：
$$k D^2 \ge k^2 \gamma^2$$
$$k \le \frac{D^2}{\gamma^2} = \left(\frac{D}{\gamma}\right)^2$$

证毕。
