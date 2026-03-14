## 条件
假设
$$\begin{array} { r l } { y } & { { } \sim \mathrm { B e r n o u l l i } ( \phi ) } \\ { x | y = 0 } & { { } \sim { \mathcal { N } } ( { \boldsymbol { \mu } } _ { 0 } , { \boldsymbol { \Sigma } } ) } \\ { x | y = 1 } & { { } \sim { \mathcal { N } } ( { \boldsymbol { \mu } } _ { 1 } , { \boldsymbol { \Sigma } } ) } \end{array}$$
写出分布
$$\begin{array} { r c l } { p ( y ) } & { = } & { \phi ^ { y } ( 1 - \phi ) ^ { 1 - y } } \\ { p ( x | y = 0 ) } & { = } & { \displaystyle \frac { 1 } { ( 2 \pi ) ^ { n / 2 } | { \boldsymbol { \Sigma } } | ^ { 1 / 2 } } \exp \left( - { \frac { 1 } { 2 } } ( x - { \mu _ { 0 } } ) ^ { T } { \boldsymbol { \Sigma } } ^ { - 1 } ( x - { \mu _ { 0 } } ) \right) } \\ { p ( x | y = 1 ) } & { = } & { \displaystyle \frac { 1 } { ( 2 \pi ) ^ { n / 2 } | { \boldsymbol { \Sigma } } | ^ { 1 / 2 } } \exp \left( - { \frac { 1 } { 2 } } ( x - { \mu _ { 1 } } ) ^ { T } { \boldsymbol { \Sigma } } ^ { - 1 } ( x - { \mu _ { 1 } } ) \right) } \end{array}$$
模型的参数为 $\phi,\Sigma,\mu_0,\mu_1$,共用一个协方差矩阵
对数似然
$$\begin{array} { r c l } { \ell ( \phi , { \boldsymbol { \mu } } _ { 0 } , { \boldsymbol { \mu } } _ { 1 } , { \boldsymbol { \Sigma } } ) } & { = } & { \displaystyle \log \prod _ { i = 1 } ^ { m } p ( x ^ { ( i ) } , y ^ { ( i ) } ; \phi , { \boldsymbol { \mu } } _ { 0 } , { \boldsymbol { \mu } } _ { 1 } , { \boldsymbol { \Sigma } } ) } \\ { } & { = } & { \displaystyle \log \prod _ { i = 1 } ^ { m } p ( x ^ { ( i ) } | y ^ { ( i ) } ; { \boldsymbol { \mu } } _ { 0 } , { \boldsymbol { \mu } } _ { 1 } , { \boldsymbol { \Sigma } } ) p ( y ^ { ( i ) } ; \phi ) . } \end{array}$$
最大化对数似然，得到
$$\begin{array} { r l } { \phi } & { { } = \; { \frac { 1 } { m } } \displaystyle \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 1 \} } \\ { \mu _ { 0 } } & { { } = \; { \frac { \displaystyle \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 0 \} x ^ { ( i ) } } { \displaystyle \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 0 \} } } } \\ { \mu _ { 1 } } & { { } = \; { \frac { \displaystyle \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 1 \} x ^ { ( i ) } } { \displaystyle \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 1 \} } } } \\ { \Sigma } & { { } = \; { \frac { 1 } { m } } \displaystyle \sum _ { i = 1 } ^ { m } ( x ^ { ( i ) } - \mu _ { y ^ { ( i ) } } ) ( x ^ { ( i ) } - \mu _ { y ^ { ( i ) } } ) ^ { T } . } \end{array}$$

## 推导
#### 1. 贝叶斯公式展开
首先利用贝叶斯公式表示后验概率 $P(y=1|x)$：

$$
\begin{aligned}
P(y=1|x) &= \frac{P(x|y=1)P(y=1)}{P(x|y=1)P(y=1) + P(x|y=0)P(y=0)} \\
&= \frac{1}{1 + \frac{P(x|y=0)P(y=0)}{P(x|y=1)P(y=1)}}
\end{aligned}
$$

#### 2. 代入高斯分布概率密度
假设 $x|y$ 服从多元高斯分布，且共享协方差矩阵 $\Sigma$。
令 $P(y=1) = \phi$，则 $P(y=0) = 1-\phi$。
代入高斯分布公式（常数项 $\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}$ 在分子分母中抵消）：

$$
\begin{aligned}
&= \frac{1}{1 + \frac{\exp\left(-\frac{1}{2}(x-\mu_0)^T \Sigma^{-1} (x-\mu_0)\right)}{\exp\left(-\frac{1}{2}(x-\mu_1)^T \Sigma^{-1} (x-\mu_1)\right)} \cdot \frac{1-\phi}{\phi}} \\
&= \frac{1}{1 + \exp\left( -\frac{1}{2} \left[ (x-\mu_0)^T \Sigma^{-1} (x-\mu_0) - (x-\mu_1)^T \Sigma^{-1} (x-\mu_1) \right] \right) \cdot \frac{1-\phi}{\phi}}
\end{aligned}
$$

#### 3. 计算指数中的二次型部分 $[\dots]$
我们需要展开并化简方括号内的项：

$$
\text{计算} [\dots] = (x^T - \mu_0^T)\Sigma^{-1}(x - \mu_0) - (x^T - \mu_1^T)\Sigma^{-1}(x - \mu_1)
$$

展开括号：
$$
\begin{aligned}
= \quad & (x^T \Sigma^{-1} x - x^T \Sigma^{-1} \mu_0 - \mu_0^T \Sigma^{-1} x + \mu_0^T \Sigma^{-1} \mu_0) \\
- \quad & (x^T \Sigma^{-1} x - x^T \Sigma^{-1} \mu_1 - \mu_1^T \Sigma^{-1} x + \mu_1^T \Sigma^{-1} \mu_1)
\end{aligned}
$$

**注意到维度与对称性：**
*   $x, \mu: n \times 1$
*   $\Sigma^{-1}: n \times n$ 且对称，即 $\Sigma^{-1} = (\Sigma^{-1})^T$
*   $1 \times n \times n \times n \times n \times 1$ 最终结果为**标量**。

对于标量，其转置等于其自身：
$$
\Rightarrow (x^T \Sigma^{-1} \mu)^T = \mu^T (\Sigma^{-1})^T x = \mu^T \Sigma^{-1} x
$$
因此，交叉项可以合并：$- x^T \Sigma^{-1} \mu - \mu^T \Sigma^{-1} x = -2\mu^T \Sigma^{-1} x$。

代回 $[\dots]$ 的式子（$x^T \Sigma^{-1} x$ 项相互抵消）：
$$
\begin{aligned}
[\dots] &= -2\mu_0^T \Sigma^{-1} x + \mu_0^T \Sigma^{-1} \mu_0 + 2\mu_1^T \Sigma^{-1} x - \mu_1^T \Sigma^{-1} \mu_1 \\
&= -2(\mu_0 - \mu_1)^T \Sigma^{-1} x + \text{常数}
\end{aligned}
$$

#### 4. 整理为 Sigmoid 形式
将化简后的 $[\dots]$ 代回原概率公式：

$$
P(y|x) = \frac{1}{1 + \exp\left( (\mu_0 - \mu_1)^T \Sigma^{-1} x + C \right) \cdot \frac{1-\phi}{\phi}}
$$

利用恒等式 $\frac{1-\phi}{\phi} = \exp\left( \ln \frac{1-\phi}{\phi} \right)$ 将系数并入指数：

$$
\begin{aligned}
&= \frac{1}{1 + \exp\left( (\mu_0 - \mu_1)^T \Sigma^{-1} x + C \right)}
\end{aligned}
$$

#### 5. 最终形式
为 $x$ 添加偏置项 "1" 以凑出常数项，定义参数向量 $\theta$：

$$
= \frac{1}{1 + \exp(-\theta^T x)}
$$