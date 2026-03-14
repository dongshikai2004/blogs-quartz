给定 训练集 $\{x^{(1)},\dots,x^{(m)}\}$ 
使用联合分布建模
$$p ( x ^ { ( i ) } , z ^ { ( i ) } ) = p ( x ^ { ( i ) } | z ^ { ( i ) } ) p ( z ^ { ( i ) } )$$
$z^{(i)}$ ~ 多项式分布 $(\phi)$ ，$\sum_{j=1}^k\phi_j=1$  （$\phi_j=p(z^{(i)}=j$) 
$x ^ { ( i ) } | z ^ { ( i ) } = j \sim { \mathcal { N } } ( \mu _ { j } , \Sigma _ { j } )$：指定分布族 $z^{(i)}$ 后 $x$ 的分布满足 高斯分布

参数 $\phi,\mu,\theta$
$$
\begin{array} { r l } { \ell ( { \boldsymbol { \phi } } , { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } ) } & { { } = \ \sum _ { i = 1 } ^ { m } \log p ( x ^ { ( i ) } ; { \boldsymbol { \phi } } , { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } ) } \\ { } & { { } = \ \sum _ { i = 1 } ^ { m } \log \sum _ { z ^ { ( i ) } = 1 } ^ { k } p ( x ^ { ( i ) } | z ^ { ( i ) } ; { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } ) p ( z ^ { ( i ) } ; { \boldsymbol { \phi } } ) } \end{array}
$$
导数置零 发现不能求出封闭形式参数最大似然估计

假设 $z^{(i)}$ 已知，已知是可得到最大值
改写形式（完全数据）
$$\ell ( { \boldsymbol { \phi } } , { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } ) = \sum _ { i = 1 } ^ { m }[ \log p ( x ^ { ( i ) } | z ^ { ( i ) } ; { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } ) + \log p ( z ^ { ( i ) } ; { \boldsymbol { \phi } } )]$$
参数
$$\begin{array} { r c l } { { \phi _ { j } } } & { { = } } & { { \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } 1 \{ z ^ { ( i ) } = j \} , } } \\ { { \mu _ { j } } } & { { = } } & { { \displaystyle \frac { \sum _ { i = 1 } ^ { m } 1 \{ z ^ { ( i ) } = j \} x ^ { ( i ) } } { \sum _ { i = 1 } ^ { m } 1 \{ z ^ { ( i ) } = j \} } , } } \\ { { \Sigma _ { j } } } & { { = } } & { { \displaystyle \frac { \sum _ { i = 1 } ^ { m } 1 \{ z ^ { ( i ) } = j \} ( x ^ { ( i ) } - \mu _ { j } ) ( x ^ { ( i ) } - \mu _ { j } ) ^ { T } } { \sum _ { i = 1 } ^ { m } 1 \{ z ^ { ( i ) } = j \} } . } } \end{array}$$

但在密度估计中 $z^{(i)}$ 不是已知的
EM算法
- E：猜测 $z^{(i)}$ (簇)
- M：根据猜测更新参数，假设猜测正确

重复直到收敛：{
(E-step) 对于每个 $i, j$，设置
$$w_j^{(i)} := p(z^{(i)} = j | x^{(i)}; \phi, \mu, \Sigma)$$
(M-step) 更新参数：
$$\begin{array} { r c l } { { \phi _ { j } } } & { { : = } } & { { \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } w _ { j } ^ { ( i ) } , } } \\ { { \mu _ { j } } } & { { : = } } & { { \displaystyle \frac { \sum _ { i = 1 } ^ { m } w _ { j } ^ { ( i ) } x ^ { ( i ) } } { \sum _ { i = 1 } ^ { m } w _ { j } ^ { ( i ) } } , } } \\ { { \Sigma _ { j } } } & { { : = } } & { { \displaystyle \frac { \sum _ { i = 1 } ^ { m } w _ { j } ^ { ( i ) } ( x ^ { ( i ) } - \mu _ { j } ) ( x ^ { ( i ) } - \mu _ { j } ) ^ { T } } { \sum _ { i = 1 } ^ { m } w _ { j } ^ { ( i ) } } . } } \end{array}$$
}
其中 $w_j^{(i)}$ 求法
$$p ( z ^ { ( i ) } = j | x ^ { ( i ) } ; \phi , \mu , \Sigma ) = { \frac { p ( x ^ { ( i ) } | z ^ { ( i ) } = j ; \mu , \Sigma ) p ( z ^ { ( i ) } = j ; \phi ) } { \sum _ { l = 1 } ^ { k } p ( x ^ { ( i ) } | z ^ { ( i ) } = l ; \mu , \Sigma ) p ( z ^ { ( i ) } = l ; \phi ) } }$$
$w_j^{(i)}$ 指示 $1\{z^{(i)}=j\}$ 的概率

类似 k-means，用软分配 $w_j^{(i)}$ 代替 硬分配聚类