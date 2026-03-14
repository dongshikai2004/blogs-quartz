supervise
# Margins: Intuition
逻辑回归中， $p(y=1|x;\theta)$ 由 $h_\theta(x)=g(\theta^Tx)$ 建模，$h_\theta(x)\ge0.5$ 时，在输入 $x$ 时预测为 $1$
考虑正训练示例 $(y=1)$,$\theta^Tx$ 越大，$h_\theta(x)=p(y=1|x;\omega,b)$ 越大：可以认为 当 $\theta^Tx>>0$ 时，$y=1$ 的 confidence 很高；反之，当 $\theta^Tx<<0$ 时，$y=0$ 的 confidence 很高
如果能找到一个 $\theta$ ,使 当 $y^{(i)}=1$时有 $\theta^Tx^{(i)}>>0$,当 $y^{(i)}=0$时有 $\theta^Tx^{(i)}<<0$，则认为这是一个好的拟合，将使用 函数间隔 来形式化

![[Pasted image 20260303115209.png]]
x 表示正训练示例，o 表示负训练示例
决策边界 由 方程 $\theta^Tx=0$ 给出
A 离边界远，$y=1$ 的置信度高
C 离边界近，$y=1$ 的置信度低，只要对边界作小的修改，容易导致 $y=0$
# Notation
$y\in\{-1,1\}$表示标签
参数 $\omega,b$ 表示 分类器，显示地将截距和其他参数分离（$b$ 代替 $\theta_0$，$\omega$ 代替 $[\theta_1,\theta_2,\dots,\theta_n]^T$
$$
h_{\omega,b}(x)=g(\omega^Tx+b)
$$
- $z\ge0$，则 $g(z)=1$，否则 $g(z)=-1$
# Functional and geometric margins
给定 训练示例 $(x^{(i)},y^{(i)})$,函数边距：
$$\hat { \gamma } ^ { ( i ) } = y ^ { ( i ) } ( w ^ { T } x + b ) .$$
- 如果 $y^{(i)}=1$，为使函数边距大， $\omega^Tx+b$ 应是较大的正数
- 如果 $y^{(i)}=-1$，为使函数边距大， $\omega^Tx+b$ 应是绝对值较大的负数

注意到，用 $2\omega$ 替换 $\omega$ ,$2b$ 替换 $b$,$w^Tx+b$ 的值为原来的 $2$ 倍，正负不变，单纯缩放是无意义的，应施加 标准化条件，例如 $||\omega||_2=1$ 

给定训练集 $S=\{(x^{(i)},y^{(i)});i=1,2,\dots,m\}$
最小函数边距
$$\hat { \gamma } = \operatorname* { m i n } _ { i = 1 , \ldots , m } \hat { \gamma } ^ { ( i ) } .$$
![[Pasted image 20260303153020.png]]
函数边距向量 $\omega$ 与分离平面正交
分离平面上的点满足 $\omega^Tx+b=0$
B点满足
$$w ^ { T } \left( x ^ { ( i ) } - \gamma ^ { ( i ) } { \frac { w } { | | w | | } } \right) + b = 0 .$$
分离得到A点的函数边距
$$\gamma ^ { ( i ) } = { \frac { w ^ { T } x ^ { ( i ) } + b } { | | w | | } } = \left( { \frac { w } { | | w | | } } \right) ^ { T } x ^ { ( i ) } + { \frac { b } { | | w | | } } .$$
更一般的，
$$
\gamma ^ { ( i ) }  = y^{(i)}\left(\left( { \frac { w } { | | w | | } } \right) ^ { T } x ^ { ( i ) } + { \frac { b } { | | w | | } }\right) 
$$
# The optimal margin classifier
找到最大函数边距的分类器
优化问题：
$$\begin{array} { r l } { \operatorname* { m a x } _ { \gamma , w , b } } & { { } \gamma } \\ { { \mathrm { s . t . } } } & { { } y ^ { ( i ) } ( w ^ { T } x ^ { ( i ) } + b ) \geq \gamma , \ i = 1 , \ldots , m } \\ { } & { { } | | w | | = 1 . } \end{array}$$
将 $2$ 个条件转换成一个条件
$$\begin{array} { r l } { \underset { \gamma , w , b } { \operatorname* { m a x } } } & { { } { \frac { \hat { \gamma } } { | | w | | } } } \\ { { \mathrm { s . t . } } } & { { } y ^ { ( i ) } ( w ^ { T } x ^ { ( i ) } + b ) \geq { \hat { \gamma } } , \quad i = 1 , \ldots , m } \end{array}$$
仍然是非凸函数
因为 可以任意缩放 $\omega$ 和 $b$ 而不产生影响，直接缩放值 $\hat\gamma=1$ 
并注意到 最大化$\frac{1}{||\omega||}$ 和最小化 $\frac12||\omega||^2$ 的条件是相同的
问题转化为
$$\begin{array} { l l } { \operatorname* { m i n } _ { \gamma , w , b } } & { { \frac { 1 } { 2 } } | | w | | ^ { 2 } } \\ { \mathrm { s . t . } } & { { y ^ { ( i ) } ( w ^ { T } x ^ { ( i ) } + b ) \geq 1 , ~ ~ i = 1 , \ldots , m } } \end{array}$$
是 凸的二次规划函数
# Lagrange duality
拉格朗日对偶
考虑
$$\begin{array} { r l } { \operatorname* { m i n } _ { w } } & { { } f ( w ) } \\ { { \mathrm { s . t . } } } & { { } h _ { i } ( w ) = 0 , \; \; i = 1 , \ldots , l . } \end{array}$$
使用拉格朗日乘子 $\beta_i$  解决
$${ \mathcal { L } } ( w , { \boldsymbol { \beta } } ) = f ( w ) + \sum _ { i = 1 } ^ { l } \beta _ { i } h _ { i } ( w )$$
求偏导
$${ \frac { \partial { \mathcal { L } } } { \partial w _ { i } } } = 0 ; \; \; { \frac { \partial { \mathcal { L } } } { \partial \beta _ { i } } } = 0$$

推广至 **primal** 优化问题，条件中含有不等式
$$\begin{array} { r l } { \operatorname* { m i n } _ { w } } & { { } f ( w ) } \\ { { \mathrm { s . t . } } } & { { } g _ { i } ( w ) \leq 0 , \; \; i = 1 , \ldots , k } \\ { \; \; \; \; \; } & { { } h _ { i } ( w ) = 0 , \; \; i = 1 , \ldots , l . } \end{array}$$
$${ \mathcal { L } } ( w , \alpha , \beta ) = f ( w ) + \sum _ { i = 1 } ^ { k } \alpha _ { i } g _ { i } ( w ) + \sum _ { i = 1 } ^ { l } \beta _ { i } h _ { i } ( w ) .$$
算子 $\alpha_i,\beta_i$ 可自由变化
$$\theta _ { \mathcal { P } } ( w ) = \operatorname* { m a x } _ { \alpha , \beta : \, \alpha _ { i } \geq 0 } { \mathcal { L } } ( w , \alpha , \beta ) .$$
- $\mathcal { P }$ 下标表示 primal
任何 $\omega$ 违反原始条件时，有 $g_i(\omega)>0$ 或 $h_i(\omega)\neq0$ ,调整 $\alpha_i,\beta_i$，可使得
$$\begin{array} { r l } { \theta _ { \mathcal { P } } ( w ) } & { { } = \operatorname* { m a x } _ { \alpha , \beta : \, \alpha _ { i } \geq 0 } f ( w ) + \sum _ { i = 1 } ^ { k } \alpha _ { i } g _ { i } ( w ) + \sum _ { i = 1 } ^ { l } \beta _ { i } h _ { i } ( w ) } \\ { \ } & { { } = \infty . } \end{array}$$
当 $\omega$ 确实满足约束，${\theta _ { \mathcal { P } } ( w ) }=f(\omega)$  
即
$$\theta _ { \mathcal { P } } ( w ) = { \left\{ \begin{array} { l l } { f ( w ) } & { { \mathrm { i f ~ } } w { \mathrm { ~ s a t i s f i e s ~ p r i m a l ~ c o n s t r a i n t s } } } \\ { \infty } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }$$
原问题转换为最小化问题
$$\operatorname* { m i n } _ { w } \theta _ { \mathcal { P } } ( w ) = \operatorname* { m i n } _ { w } \operatorname* { m a x } _ { \alpha , \beta : \, \alpha _ { i } \geq 0 } { \mathcal { L } } ( w , \alpha , \beta )$$
目标的最优值
$$p ^ { * } = \operatorname* { m i n } _ { w } \theta _ { \mathcal { P } } ( w )$$
**dual** 问题
双变量
$$\theta _ { \mathcal { D } } ( \alpha , \beta ) = \operatorname* { m i n } _ { w } { \mathcal { L } } ( w , \alpha , \beta ) .$$
**dual** 优化
$$\operatorname* { m a x } _ { \alpha , \beta : \, \alpha _ { i } \geq 0 } \theta _ { \mathcal { D } } ( \alpha , \beta ) = \operatorname* { m a x } _ { \alpha , \beta : \, \alpha _ { i } \geq 0 } \operatorname* { m i n } _ { w } { \mathcal { L } } ( w , \alpha , \beta )$$
目标的最优值为
$$d ^ { * } = \operatorname* { m a x } _ { \alpha , \beta : \, \alpha _ { i } \geq 0 } \theta _ { \mathcal { D } } ( w ) .$$
primal 问题和 dual 问题的关系
$$d ^ { * } = \operatorname* { m a x } _ { \alpha , \beta : \, \alpha _ { i } \geq 0 } \operatorname* { m i n } _ { w } { \mathcal { L } } ( w , \alpha , \beta ) \leq \operatorname* { m i n } _ { w } \operatorname* { m a x } _ { \alpha , \beta : \, \alpha _ { i } \geq 0 } { \mathcal { L } } ( w , \alpha , \beta ) = p ^ { * }$$
在某些条件下，有
$$
d^*=p^*
$$
此时可通过解决dual问题来解决primal问题
需要满足 **KKT**条件 Karush-Kuhn-Tucker
$$\begin{array} { r l } { { \frac { \partial } { \partial w _ { i } } } { \mathcal { L } } ( w ^ { * } , \alpha ^ { * } , \beta ^ { * } ) } & { { } = \ 0 , \quad i = 1 , \ldots , n } \\ { { \frac { \partial } { \partial \beta _ { i } } } { \mathcal { L } } ( w ^ { * } , \alpha ^ { * } , \beta ^ { * } ) } & { { } = \ 0 , \quad i = 1 , \ldots , l } \\ { \alpha _ { i } ^ { * } g _ { i } ( w ^ { * } ) } & { { } = \ 0 , \quad i = 1 , \ldots , k } \\ { g _ { i } ( w ^ { * } ) } & { { } \leq \ 0 , \quad i = 1 , \ldots , k } \\ { \alpha ^ { * } } & { { } \geq \ 0 , \quad i = 1 , \ldots , k } \end{array}$$
# Optimal margin classifiers
原始的优化问题
$$\begin{array} { l l } { \operatorname* { m i n } _ { \gamma , w , b } } & { { \frac { 1 } { 2 } } | | w | | ^ { 2 } } \\ { { \mathrm { s . t . } } } & { { y ^ { ( i ) } ( w ^ { T } x ^ { ( i ) } + b ) \geq 1 , ~ ~ i = 1 , \ldots , m } } \end{array}$$
约束改写为 
$$g _ { i } ( w ) = - y ^ { ( i ) } ( w ^ { T } x ^ { ( i ) } + b ) + 1 \leq 0 .$$
每个训练样本都有一个约束
根据KKT对偶约束，人为约束 函数间隔 $\gamma=1$ 才有 $g_i(\omega)=0$
![[Pasted image 20260303173247.png]]
边距最小的点是最接近决策边界的点（虚线上的三个点），被称为 **support** **vector**
支持向量的数量可远小于训练集的大小

为优化问题建立拉格朗日函数
$$\mathcal { L } ( w , b , \alpha ) = \frac { 1 } { 2 } | | w | | ^ { 2 } - \sum _ { i = 1 } ^ { m } \alpha _ { i } \left[ y ^ { ( i ) } ( w ^ { T } x ^ { ( i ) } + b ) - 1 \right]$$
找出该问题的对偶形式
先针对 $\omega$ 最小化
$$\nabla _ { w } { \mathcal { L } } ( w , b , \alpha ) = w - \sum _ { i = 1 } ^ { m } \alpha _ { i } y ^ { ( i ) } x ^ { ( i ) } = 0$$
得到
$$w = \sum _ { i = 1 } ^ { m } \alpha _ { i } y ^ { ( i ) } x ^ { ( i ) }$$
再对 $b$ 求偏导
$$\frac { \partial } { \partial b } \mathcal { L } ( w , b , \alpha ) = \sum _ { i = 1 } ^ { m } \alpha _ { i } y ^ { ( i ) } = 0$$
将 $\omega$ 代回原式
$$\mathcal { L } ( w , b , \alpha ) = \sum _ { i = 1 } ^ { m } \alpha _ { i } - \frac { 1 } { 2 } \sum _ { i , j = 1 } ^ { m } y ^ { ( i ) } y ^ { ( j ) } \alpha _ { i } \alpha _ { j } ( x ^ { ( i ) } ) ^ { T } x ^ { ( j ) } - b \sum _ { i = 1 } ^ { m } \alpha _ { i } y ^ { ( i ) }$$
最后一项为 $0$
$$\mathcal { L } ( w , b , \alpha ) = \sum _ { i = 1 } ^ { m } \alpha _ { i } - \frac { 1 } { 2 } \sum _ { i , j = 1 } ^ { m } y ^ { ( i ) } y ^ { ( j ) } \alpha _ { i } \alpha _ { j } ( x ^ { ( i ) } ) ^ { T } x ^ { ( j ) }$$
最终得到对偶优化问题
$$
\begin{array} { r l } { \operatorname* { m a x } _ { \boldsymbol { \alpha } } } & { { } W ( { \boldsymbol { \alpha } } ) = \sum _ { i = 1 } ^ { m } \alpha _ { i } - { \frac { 1 } { 2 } } \sum _ { i , j = 1 } ^ { m } y ^ { ( i ) } y ^ { ( j ) } \alpha _ { i } \alpha _ { j } \langle x ^ { ( i ) } , x ^ { ( j ) } \rangle } \\ { \mathrm { s . t . } } & { { } \alpha _ { i } \geq 0 , \ i = 1 , \ldots , m } \\ { \displaystyle } & { { } \sum _ { i = 1 } ^ { m } \alpha _ { i } y ^ { ( i ) } = 0 } \end{array}
$$
得到最优的 $\alpha^*$，再找到相应的 $\omega^*$
回到原始问题，最佳截距为
$$b ^ { * } = - { \frac { \operatorname* { m a x } _ { i : y ^ { ( i ) } = - 1 } w ^ { * T } x ^ { ( i ) } + \operatorname* { m i n } _ { i : y ^ { ( i ) } = 1 } w ^ { * T } x ^ { ( i ) } } { 2 } }$$
# Kernels
使用特征 $x,x^2,x^3$ 来获得三次函数
为区分变量，将原始输入称为问题的 输入 **attributes** ,它映射到一组新的向量，称为 **features**
用 $\phi$ 表示 feature mapping
$$\phi ( x ) = { \left[ \begin{array} { l } { x } \\ { x ^ { 2 } } \\ { x ^ { 3 } } \end{array} \right] }$$
$\phi(x)$ 可替代原来的 $x$
内积 $\langle x,z\rangle$ 用核函数表示
$$K ( x , z ) = \phi ( x ) ^ { T } \phi ( z )$$
$\phi(x)$ 的计算成本可能很高，但 核函数的计算很轻松

假设 $x,z\in \mathbb R^n$，考虑
$$K ( x , z ) = ( x ^ { T } z ) ^ { 2 }$$
$$
\begin{array} { r l } { K ( x , z ) } & { { } = \left( \displaystyle \sum _ { i = 1 } ^ { n } x _ { i } z _ { i } \right) \left( \displaystyle \sum _ { j = 1 } ^ { n } x _ { i } z _ { i } \right) } \\ { } & { { } = \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } x _ { i } x _ { j } z _ { i } z _ { j } } \\ { } & { { } = \displaystyle \sum _ { i , j = 1 } ^ { n } ( x _ { i } x _ { j } ) ( z _ { i } z _ { j } ) } \end{array}
$$
找到
$$K ( x , z ) = \phi ( x ) ^ { T } \phi ( z )$$
其特征映射 ($n=3$)
$$\phi ( x ) = { \left[ \begin{array} { l } { x _ { 1 } x _ { 1 } } \\ { x _ { 1 } x _ { 2 } } \\ { x _ { 1 } x _ { 3 } } \\ { x _ { 2 } x _ { 1 } } \\ { x _ { 2 } x _ { 2 } } \\ { x _ { 2 } x _ { 3 } } \\ { x _ { 3 } x _ { 1 } } \\ { x _ { 3 } x _ { 2 } } \\ { x _ { 3 } x _ { 3 } } \end{array} \right] }$$
同样的
$$
\begin{array} { r c l } { { K ( x , z ) } } & { { = } } & { { ( x ^ { T } z + c ) ^ { 2 } } } \\ { { } } & { { = } } & { { \displaystyle \sum _ { i , j = 1 } ^ { n } ( x _ { i } x _ { j } ) ( z _ { i } z _ { j } ) + \sum _ { i = 1 } ^ { n } ( \sqrt { 2 } c x _ { i } ) ( \sqrt { 2 } c z _ { i } ) + c ^ { 2 } } } \end{array}
$$
$$\phi ( x ) = { \left[ \begin{array} { l } { x _ { 1 } x _ { 1 } } \\ { x _ { 1 } x _ { 2 } } \\ { x _ { 1 } x _ { 3 } } \\ { x _ { 2 } x _ { 1 } } \\ { x _ { 2 } x _ { 2 } } \\ { x _ { 2 } x _ { 3 } } \\ { x _ { 3 } x _ { 1 } } \\ { x _ { 3 } x _ { 2 } } \\ { x _ { 3 } x _ { 3 } } \\ { { \sqrt { 2 } } c x _ { 1 } } \\ { { \sqrt { 2 } } c x _ { 2 } } \\ { { \sqrt { 2 } } c x _ { 3 } } \\ { c } \end{array} \right] } $$
参数 $c$ 控制一阶和二阶之间的相对权重
推广
内核 $K ( x , z ) = ( x ^ { T } z + c ) ^ { d }$ 对应 $\textstyle { \binom { n + d } { d } }$  维度的特征空间的特征映射
计算 $K(x,z)$ 仍需要 $O(n)$ 的时间（计算内积）

直觉上，$\phi(x)$ 和 $\phi(z)$  越接近，会期望 $K(x,z)$ 越大

给定某个核函数 $K$ ，如何判断是否是有效的内核？即如何判断是否存在某种特征映射 $\phi$ 使 $K(x,z)=\phi(x)^T\phi(z)$ 适合所有的 $x,z$

假设 $K$ 是某些特征映射 $\phi$ 的有效内核，考虑一些 $m$ 点的 有限集 $\{ x ^ { ( 1 ) } , \ldots , x ^ { ( m ) } \}$
并定义一个正方形矩阵K $m\times m$ ,$K_{ij}=K(x^{(i)},x^{(j)})$ ，称为 **Kernel** **matrix**

如果 K 是有效内核，那么 
$$K _ { i j } = K \left( x ^ { ( i ) } , x ^ { ( j ) } \right) = \phi \left( x ^ { ( i ) } \right) ^ { T } \phi \left( x ^ { ( j ) } \right) = \\ \phi \left( x ^ { ( j ) } \right) ^ { T } \phi \left( x ^ { ( i ) } \right) = K \left( x ^ { ( j ) } , x ^ { ( i ) } \right) = K _ { j i }$$
K 是对称的

令 $\phi_k(x)$ 表示 $\phi(x)$ 的第 $k$ 个坐标，对于任何向量 $z$
$$\begin{array} { r c l } { { z ^ { T } K z } } & { { = } } & { { \displaystyle \sum _ { i } \sum _ { j } z _ { i } K _ { i j } z _ { j } } } \\ { { } } & { { = } } & { { \displaystyle \sum _ { i } \sum _ { j } z _ { i } \phi ( x ^ { ( i ) } ) ^ { T } \phi ( x ^ { ( j ) } ) z _ { j } } } \\ { { } } & { { = } } & { { \displaystyle \sum _ { i } \sum _ { j } z _ { i } \sum _ { k } \phi _ { k } ( x ^ { ( i ) } ) \phi _ { k } ( x ^ { ( j ) } ) z _ { j } } } \\ { { } } & { { = } } & { { \displaystyle \sum _ { k } \sum _ { i } \sum _ { j } z _ { i } \phi _ { k } ( x ^ { ( i ) } ) \phi _ { k } ( x ^ { ( j ) } ) z _ { j } } } \\ { { } } & { { = } } & { { \displaystyle \sum _ { k } \left( \sum _ { i } z _ { i } \phi _ { k } ( x ^ { ( i ) } ) \right) ^ { 2 } } } \\ { { } } & { { \ge } } & { { 0 . } } \end{array}$$
得出 K 是半正定的

**Theorem (Mercer).** 设 $K: \mathbb{R}^n \times \mathbb{R}^n \mapsto \mathbb{R}$。那么，要使 $K$ 成为有效的（Mercer）核，当且仅当对于任何 $\{x^{(1)}, \dots, x^{(m)}\}$、$(m < \infty)$ 而言，其对应的核矩阵是对称正半定矩阵是必要且充分的。

# Regularization and the non-separable case
为了适用 非线性可分离数据集 并对异常值不敏感，重新指定优化，
$$\begin{array} { c c } { { \operatorname * { m i n } _ { \gamma , w , b } } } & { { \frac { 1 } { 2 } | | w | | ^ { 2 } + C \displaystyle \sum _ { i = 1 } ^ { m } \xi _ { i } } } \\ { { \mathrm { s . t . } } } & { { y ^ { ( i ) } ( w ^ { T } x ^ { ( i ) } + b ) \geq 1 - \xi _ { i } , \: \: i = 1 , . . . , m } } \\ { { } } & { { \xi _ { i } \geq 0 , \: \: i = 1 , . . . , m . } } \end{array}$$
现在允许边际小于 $1$ ,当边际为 $1-\xi_i$ 时，增加成本 $C\xi_i$  
$C$ 控制两个目标间的相对权重
$${ \mathcal { L } } ( w , b , \xi , \alpha , r ) = { \frac { 1 } { 2 } } w ^ { T } w + C \sum _ { i = 1 } ^ { m } \xi _ { i } - \sum _ { i = 1 } ^ { m } \alpha _ { i } \left[ y ^ { ( i ) } ( x ^ { T } w + b ) - 1 + \xi _ { i } \right] - \sum _ { i = 1 } ^ { m } { r _ { i } \xi _ { i } } $$
将 $\omega,b$ 的导数设置为 $0$，代回转成单变量，得到对偶形式
$$\begin{aligned}
& \max _{\alpha} W(\alpha)=\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left\langle x^{(i)}, x^{(j)}\right\rangle \\
& \text { s.t. } 0 \leq \alpha_{i} \leq C, i=1, \ldots, m \\
& \quad \sum_{i=1}^{m} \alpha_{i} y^{(i)}=0,
\end{aligned}$$
与上面的问题相同，约束中添加上限 $C$
KKT双互补条件
$$\begin{array} { r l } { \alpha _ { i } = 0 } & { { } \Rightarrow \ y ^ { ( i ) } \left( w ^ { T } x ^ { ( i ) } + b \right) \geq 1 } \\ { \alpha _ { i } = C } & { { } \Rightarrow \ y ^ { ( i ) } \left( w ^ { T } x ^ { ( i ) } + b \right) \leq 1 } \\ { 0 < \alpha _ { i } < C } & { { } \Rightarrow \ y ^ { ( i ) } \left( w ^ { T } x ^ { ( i ) } + b \right) = 1 } \end{array}$$
# SMO algorithm
顺序最小优化
## Coordinate ascent
无约束优化问题
$$\operatorname* { m a x } _ { \alpha } W ( \alpha _ { 1 } , \alpha _ { 2 } , \ldots , \alpha _ { m } )$$
使用新算法Coordinate ascent
Loop until convergence: {
    For $i = 1, \ldots, m$, {
        $\alpha_i := \arg\max_{\hat{\alpha}_i} W(\alpha_1, \ldots, \alpha_{i-1}, \hat{\alpha}_i, \alpha_{i+1}, \ldots, \alpha_m)$
    }
}
固定除 $a_i$ 外的所有变量，对 $a_i$ 进行优化
![[Pasted image 20260305083632.png]]
每次优化到达当前全局最优值
## SMO
$\begin{array}{r}\max _{\alpha} W(\alpha)=\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left\langle x^{(i)}, x^{(j)}\right\rangle . \\ \text { s.t. } \quad 0 \leq \alpha_{i} \leq C, \; i=1, \ldots, m \\ \sum_{i=1}^{m} \alpha_{i} y^{(i)}=0 .\end{array}$
在固定除 $a_i$ 外的所有变量时
恒有
$$\alpha _ { 1 } y ^ { ( 1 ) } = - \sum _ { i = 2 } ^ { m } \alpha _ { i } y ^ { ( i ) }$$
因此想更新一个 $a_i$ ，至少更新两个变量才行
Repeat till convergence {
    1. Select some pair $\alpha_i$ and $\alpha_j$ to update next (using a heuristic that tries to pick the two that will allow us to make the biggest progress towards the global maximum).
    2. Reoptimize $W(\alpha)$ with respect to $\alpha_i$ and $\alpha_j$, while holding all the other $\alpha_k$'s ($k \neq i, j$) fixed.
}
双变量
$$\alpha _ { 1 } y ^ { ( 1 ) } + \alpha _ { 2 } y ^ { ( 2 ) } = - \sum _ { i = 3 } ^ { m } \alpha _ { i } y ^ { ( i ) }$$
右侧为固定值，用 $\xi$ 表示
$$\alpha _ { 1 } y ^ { ( 1 ) } + \alpha _ { 2 } y ^ { ( 2 ) } = \xi$$
![[Pasted image 20260305085615.png]]
将 $a_1$ 写成 $a_2$ 的函数
$$\alpha _ { 1 } = ( \zeta - \alpha _ { 2 } y ^ { ( 2 ) } ) y ^ { ( 1 ) }$$
$W(a)$ 写成
$$W ( \alpha _ { 1 } , \alpha _ { 2 } , \ldots , \alpha _ { m } ) = W ( ( \zeta - \alpha _ { 2 } y ^ { ( 2 ) } ) y ^ { ( 1 ) } , \alpha _ { 2 } , \ldots , \alpha _ { m } )$$
用 $a_2^{new,unclipped}$ 表示 $a_2$ 的结界值；$a_2^{new}$ 表示最终更新的值
$$\alpha _ { 2 } ^ { n e w } \ = \ \left\{ \begin{array} { l l } { { H } } & { { \mathrm { i f ~ } \alpha _ { 2 } ^ { n e w , u n c l i p p e d } > H } } \\ { { \alpha _ { 2 } ^ { n e w , u n c l i p p e d } } } & { { \mathrm { i f ~ } L \le \alpha _ { 2 } ^ { n e w , u n c l i p p e d } \le H } } \\ { { L } } & { { \mathrm { i f ~ } \alpha _ { 2 } ^ { n e w , u n c l i p p e d } < L } } \end{array} \right.$$
找到 $a_2^{new}$ 后再找到 $a_1^{new}$
