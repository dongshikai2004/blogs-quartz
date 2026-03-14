# Bias/variance tradeoff
泛化误差的
1. bias:模型预期的偏差;模型简单时
2. variance:训练中偏离点导致的较大方差;模型复杂时
# Preliminaries
**Lemma.** (联合界限)。令 $A_1, A_2, \ldots, A_k$ 为 $k$ 个不同的事件（可能不是独立的）。

$$P(A_1 \cup \cdots \cup A_k) \leq P(A_1) + \ldots + P(A_k).$$
**Lemma.** (Hoeffding 不等式) 令 $Z_1, \ldots, Z_m$ 为从 Bernoulli($\phi$) 分布中抽取的 $m$ 独立同分布 (iid) 随机变量。即，$P(Z_i=1)=\phi$ 和 $P(Z_i=0)=1-\phi$。令 $\hat{\phi} = (1/m)\sum_{i=1}^m Z_i$ 为这些随机变量的平均值，并令任何 $\gamma > 0$ 为固定。**Chernoff bound**

$$P(|\phi - \hat{\phi}| > \gamma) \leq 2\exp(-2\gamma^2 m)$$


以二元分类为例
假设训练集 $\{(x^{(i)},y^{(i)});i=1,2,\dots,m\}$，训练示例时某个概率分布 $\mathbb D$ 中独立同分布的
**training error** 或 **empirical risk** 经验风险 定义为
$$\hat { \varepsilon } ( h ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } 1 \{ h ( x ^ { ( i ) } ) \neq y ^ { ( i ) } \}$$
期望风险，不可直接计算，反映在所有数据上的真实能力
$$\varepsilon ( h ) = P _ { ( x , y ) \sim { \mathcal { D } } } ( h ( x ) \neq y )$$
考虑线性分类器，令 $h_\theta(x)=1\{\theta^Tx\ge0\}$ 
一种方法时最小化 训练误差
$$\hat { \theta } = \arg \operatorname* { m i n } _ { \theta } \hat { \varepsilon } ( h _ { \theta } )$$
该过程称为 **empirical risk minimization** (ERM)
该算法的假设结果为 $\hat h=h_{\hat\theta}$

定义 **hypothesis class**  $\mathcal { H }$ 为 所考虑的所有分类器的集合
经验风险可认为是 函数类 $\mathcal { H }$ 的最小化

 $${ \hat { h } } = \arg \operatorname* { m i n } _ { h \in { \mathcal { H } } } { \hat { \varepsilon } } ( h )$$

# The case of finite $\mathcal { H }$ 
假设有 $k$ 个 $h_i$ ，组成有限假设类 $\mathcal { H }=\{h_1,h_2,\dots,h_k\}$ 
要选择训练误差最小的一个
策略
1. 证明 $\hat { \varepsilon } ( h )$ 是对所有 $h$ 的 $\varepsilon  ( h )$ 的可靠估计
2. $\hat h$ 泛化误差有上限

设置 $Z = 1 \{ h _ { i } ( x ) \neq y \}$ ，让它指示错误分类；$Z_j = 1 \{ h _ { i } ( x^{(j)} ) \neq y^{(j)} \}$
$${ \hat { \varepsilon } } ( h _ { i } ) = { \frac { 1 } { m } } \sum _ { j = 1 } ^ { m } { Z _ { j } }$$
应用Hoeffding 不等式
$$P ( | \varepsilon ( h _ { i } ) - { \hat { \varepsilon } } ( h _ { i } ) | > \gamma ) \leq 2 \exp ( - 2 \gamma ^ { 2 } m )$$
表明，假设 $m$ 很大，训练误差就解决泛化误差

积累界限
$$
\begin{eqnarray*}
P(\exists h \in \mathcal {H}.|\varepsilon (h_{i})-{\hat {\varepsilon }}(h_{i})|>\gamma ) & = & P(A_{1}\cup \dots \cup A_{k}) \\
& \leq & \sum _{i=1}^{k}P(A_{i}) \\
& \leq & \sum _{i=1}^{k}2\exp(-2\gamma ^{2}m) \\
& = & 2k\exp(-2\gamma ^{2}m)
\end{eqnarray*}
$$
$$
\begin{array} { r l } { P ( \lnot \exists h \in { \mathcal { H } } . | \varepsilon ( h _ { i } ) - { \hat { \varepsilon } } ( h _ { i } ) | > \gamma ) } & { { } = P ( \forall h \in { \mathcal { H } } . | \varepsilon ( h _ { i } ) - { \hat { \varepsilon } } ( h _ { i } ) | \leq \gamma ) } \\ { \geq } & { { } 1 - 2 k \exp ( - 2 \gamma ^ { 2 } m ) } \end{array}
$$
对于所有 $h$，在 $\gamma$ 范围内的概率至少为 $1 - 2 k \exp ( - 2 \gamma ^ { 2 } m )$

给定 $\gamma$ 和 $\delta>0$ ,$m$ 必须多大，才能保证 训练误差在泛化误差的 $\gamma$ 内的概率至少为 $1-\delta$ ?
设置 $\delta=2 k \exp ( - 2 \gamma ^ { 2 } m )$ ,求解 $m$
$$m \geq { \frac { 1 } { 2 \gamma ^ { 2 } } } \log { \frac { 2 k } { \delta } }$$
$m$ 也被称为 算法的 sample complexity

类似地，保持 $m$ 和 $\delta$ 固定，求解 $\gamma$ ;以概率 $1-\delta$,有
$$| { \hat { \varepsilon } } ( h ) - \varepsilon ( h ) | \leq { \sqrt { { \frac { 1 } { 2 m } } \log { \frac { 2 k } { \delta } } } }$$

假设一致收敛成立，即对于所有的 $h\in\mathcal H,| \varepsilon ( h _ { i } ) - { \hat { \varepsilon } } ( h _ { i } ) | \le \gamma$ ,证明选择 $\hat { h } = \operatorname { a r g m i n } _ { h \in { \mathcal { H } } } { \hat { \varepsilon } } ( h )$ 的正确性
令 $h^* = \operatorname { a r g m i n } _ { h \in { \mathcal { H } } } { { \varepsilon } } ( h )$ 作为 $\mathcal H$ 中的最佳可能假设
$$\begin{array} { r l } { \varepsilon ( { \hat { h } } ) } & { { } \leq \ { \hat { \varepsilon } } ( { \hat { h } } ) + \gamma } \\ { \ } & { { } \leq \ { \hat { \varepsilon } } ( h ^ { * } ) + \gamma } \\ { \ } & { { } \leq \ \varepsilon ( h ^ { * } ) + 2 \gamma } \end{array}$$
1. 事实：$\varepsilon ( h _ { i } ) - { \hat { \varepsilon } } ( h _ { i } ) | \le \gamma$
2. 选择 $\hat h$ 最小化 $\hat \varepsilon(h)$ 对于所有 $h\in\mathcal H$ 成立
3. 一致性假设：$\hat\varepsilon(h^*)\le\varepsilon(h^*)+\gamma$ 
结论：如果一致性收敛，那么 $\hat h$ 的泛化误差比最佳假设误差 差 $2\gamma$ 
**定理.** 令 $| \mathcal { H } | = k$ , 并让任何 $m , \delta$ 被固定。那么概率至少为 $1 - \delta$ , 我们有
$$
\varepsilon ( { \hat { h } } ) \leq \left( \operatorname* { m i n } _ { h \in { \mathcal { H } } } \varepsilon ( h ) \right) + 2 { \sqrt { { \frac { 1 } { 2 m } } \log { \frac { 2 k } { \delta } } } } .
$$
是通过让 $\gamma={ \sqrt { { \frac { 1 } { 2 m } } \log { \frac { 2 k } { \delta } } } }$  来证明的
切换到更大的假设类 $\mathcal H'$ ，偏差只会更小；$k$ 增加，偏差会更大

**推论.** 令 $| \mathcal { H } | = k$ , 并让任何 $\delta , \gamma$ 被固定。那么对于 $\varepsilon ( { \hat { h } } ) \leq \operatorname* { m i n } _ { h \in { \mathcal { H } } } \varepsilon ( h ) + 2 \gamma$ 以至少 $1 - \delta$ 的概率成立，则足以满足

$$
\begin{aligned}
m & \geq \frac { 1 } { 2 \gamma ^ { 2 } } \log \frac { 2 k } { \delta } \\
& = O \left( \frac { 1 } { \gamma ^ { 2 } } \log \frac { k } { \delta } \right) ,
\end{aligned}
$$
# The case of infinite $\mathcal { H }$ 
给定与训练集无关的 集合 $S=\{x^{(i)},\dots,x^{(d)}\}$ ，如果 $\mathcal H$ 可以实现 $S$ 上的任何点，就说 $\mathcal H$ **shatters** $S$ ，即对于 任何一组标签 $\{y^{(i)},\dots,y^{(d)}\}$ 都存在 $h\in\mathcal H$ 使 $h(x^{(i)})=y^{(i)}$ 对于所有的 $i$
定义 Vapnik-Chervonenkis dimension **VC($\mathcal H$)** 作为被 $\mathcal H$ shatter 的最大集合的大小

$${ \mathcal { H } } ( h ( x ) = 1 \{ \theta _ { 0 } + \theta _ { 1 } x _ { 1 } + \theta _ { 2 } x _ { 2 } \geq 0 \} )$$
![[Pasted image 20260305161959.png]]
可以证明没有任何 $4$ 个点的集合可以被该假设类 shatter
因此 VC$(\mathcal H)=3$ 
尽管如此，仍有无法被线性分割的图
![[Pasted image 20260305162146.png]]
为证明 VC$(\mathcal H)$ 至少是 $d$，只需证明至少有 $1$ 个大小为 $d$ 的点集被 $\mathcal H$ shatter

**Theorem.** 给定$\mathcal{H}$，并令$d = \text{VC}(\mathcal{H})$。然后，对于所有 $h \in \mathcal{H}$，我们都有至少 $1 - \delta$ 的概率，
$$
|\varepsilon(h) - \hat{\varepsilon}(h)| \leq O\left(\sqrt{\frac{d}{m}\log\frac{m}{d}} + \frac{1}{m}\log\frac{1}{\delta}\right).
$$
**Corollary.** 对于 $|\varepsilon(h) - \hat{\varepsilon}(h)| \leq \gamma$ 对于所有 $h \in \mathcal{H}$ (成立，因此 $\varepsilon(\hat{h}) \leq \varepsilon(h^*) + 2\gamma$) 的概率至少为 $1 - \delta$，则满足 $m = O_{\gamma, \delta}(d)$。