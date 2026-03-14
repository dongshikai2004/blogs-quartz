# Jensen’s inequality
**Theorem.** 令 $f$ 为凸函数，令 $X$ 为随机变量。然后：
$$ \operatorname { E } [ f ( X ) ] \geq f ( \operatorname { E } X ) . $$此外，如果 $f$ 是严格凸的，则 $\operatorname { E } [ f ( X ) ] = f ( \operatorname { E } X )$ 当且仅当 $X = \operatorname { E } [ X ]$ 的概率为 1（即，如果 $X$ 是常数）时才成立。
# The EM algorithm
给定 训练集 $\{x^{(1)},\dots,x^{(m)}\}$ 
$$\begin{array} { r l } { \ell ( \theta ) } & { { } = \ \displaystyle \sum _ { i = 1 } ^ { m } \log p ( x ; \theta ) } \\ { } & { { } = \ \displaystyle \sum _ { i = 1 } ^ { m } \log \sum _ { z } p ( x , z ; \theta ) . } \end{array}$$
- 由全概率公式得到第二行
- $z^{(i)}$ 是 隐藏的随机变量

使用EM
对于每个 $i$，令 $Q_i$ 是 $z$ 上的某种分布 ($\sum_z Q_i(z) = 1, \ Q_i(z) \geq 0$)。请考虑以下事项
$$
\begin{aligned}
\sum_{i} \log p(x^{(i)}; \theta) & = \sum_{i} \log \sum_{z^{(i)}} p(x^{(i)}, z^{(i)}; \theta) \\
& = \sum_{i} \log \sum_{z^{(i)}} Q_i(z^{(i)}) \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \\
& \geq \sum_{i} \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}
\end{aligned}
$$
- 最后一行使用Jensen’s inequality：$f(x)=\log x;;f''(x)=-\frac 1 {x^2}$ ，$-f$ 是凸函数，反转公式

$$f \left( \operatorname { E } _ { z ^ { ( i ) } \sim Q _ { i } } \left[ { \frac { p ( x ^ { ( i ) } , z ^ { ( i ) } ; \theta ) } { Q _ { i } ( z ^ { ( i ) } ) } } \right] \right) \geq \operatorname { E } _ { z ^ { ( i ) } \sim Q _ { i } } \left[ f \left( { \frac { p ( x ^ { ( i ) } , z ^ { ( i ) } ; \theta ) } { Q _ { i } ( z ^ { ( i ) } ) } } \right) \right]$$

确定 $\ell ( \theta )$ 的下界

进一步缩小界限，让涉及不等式的式子保持取等，取等时才能使界限同步变化
要求 $X$ 是一个常数
$$\frac { p ( x ^ { ( i ) } , z ^ { ( i ) } ; \theta ) } { Q _ { i } ( z ^ { ( i ) } ) } = c$$
得到
$$Q _ { i } ( z ^ { ( i ) } ) \propto p ( x ^ { ( i ) } , z ^ { ( i ) } ; \theta )$$

已知 $\sum_z Q_i(z) = 1$

