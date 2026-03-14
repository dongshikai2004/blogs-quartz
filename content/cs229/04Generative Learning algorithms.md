给猫狗图片二分类，$y\in\{0,1\}$
discriminative 学习算法：逻辑回归或感知器算法 尝试找到一条直线，分割猫狗的决策边界
generative学习算法：首先观测猫，建立猫的模型；观测狗，建立狗的模型，将新动物与两个模型匹配，看哪个更像
在建模 $p(y)$ (先验分布 class priors) 和 $p(x|y)$，算法根据贝叶斯规则得到后验分布
$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
$$
$$
\begin{aligned}
\arg \max _{y} p(y \mid x) & =\arg \max _{y} \frac{p(x \mid y) p(y)}{p(x)} \\
& =\arg \max _{y} p(x \mid y) p(y)
\end{aligned}
$$
训练集固定，分母无需计算
# 高斯判别分析GDA
假设 $p(x|y)$ 根据多元正态分布分布
## 多元正态分布
多元高斯分布
由 mean vector $\mu\in \mathbb R^n$  和 covariance matrix $\Sigma\in \mathbb R^{n\times n}$ 参数化 ，$\Sigma\ge0$ 对称且半正定
密度
$$
p(x ; \mu, \Sigma)=\frac{1}{(2 \pi)^{n / 2}|\Sigma|^{1 / 2}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)
$$
均值 $E[X]$
$$
\mathrm{E}[X] = \int_{x} x p(x ; \mu, \Sigma) d x = \mu
$$
协方差Cov(Z) 定义 $\operatorname{Cov}(Z)=\mathrm{E}\left[(Z-\mathrm{E}[Z])(Z-\mathrm{E}[Z])^{T}\right]$
这里 $\operatorname{Cov}(X)=\Sigma$
## 高斯判别分析模型
假设
$$\begin{array} { r l } { y } & { { } \sim \mathrm { B e r n o u l l i } ( \phi ) } \\ { x | y = 0 } & { { } \sim { \mathcal { N } } ( { \boldsymbol { \mu } } _ { 0 } , { \boldsymbol { \Sigma } } ) } \\ { x | y = 1 } & { { } \sim { \mathcal { N } } ( { \boldsymbol { \mu } } _ { 1 } , { \boldsymbol { \Sigma } } ) } \end{array}$$
写出分布
$$\begin{array} { r c l } { p ( y ) } & { = } & { \phi ^ { y } ( 1 - \phi ) ^ { 1 - y } } \\ { p ( x | y = 0 ) } & { = } & { \displaystyle \frac { 1 } { ( 2 \pi ) ^ { n / 2 } | { \boldsymbol { \Sigma } } | ^ { 1 / 2 } } \exp \left( - { \frac { 1 } { 2 } } ( x - { \mu _ { 0 } } ) ^ { T } { \boldsymbol { \Sigma } } ^ { - 1 } ( x - { \mu _ { 0 } } ) \right) } \\ { p ( x | y = 1 ) } & { = } & { \displaystyle \frac { 1 } { ( 2 \pi ) ^ { n / 2 } | { \boldsymbol { \Sigma } } | ^ { 1 / 2 } } \exp \left( - { \frac { 1 } { 2 } } ( x - { \mu _ { 1 } } ) ^ { T } { \boldsymbol { \Sigma } } ^ { - 1 } ( x - { \mu _ { 1 } } ) \right) } \end{array}$$
模型的参数为 $\phi,\Sigma,\mu_0,\mu_1$,共用一个协方差矩阵
对数似然
$$\begin{array} { r c l } { \ell ( \phi , { \boldsymbol { \mu } } _ { 0 } , { \boldsymbol { \mu } } _ { 1 } , { \boldsymbol { \Sigma } } ) } & { = } & { \displaystyle \log \prod _ { i = 1 } ^ { m } p ( x ^ { ( i ) } , y ^ { ( i ) } ; \phi , { \boldsymbol { \mu } } _ { 0 } , { \boldsymbol { \mu } } _ { 1 } , { \boldsymbol { \Sigma } } ) } \\ { } & { = } & { \displaystyle \log \prod _ { i = 1 } ^ { m } p ( x ^ { ( i ) } | y ^ { ( i ) } ; { \boldsymbol { \mu } } _ { 0 } , { \boldsymbol { \mu } } _ { 1 } , { \boldsymbol { \Sigma } } ) p ( y ^ { ( i ) } ; \phi ) . } \end{array}$$
最大化对数似然，得到
$$\begin{array} { r l } { \phi } & { { } = \; { \frac { 1 } { m } } \displaystyle \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 1 \} } \\ { \mu _ { 0 } } & { { } = \; { \frac { \displaystyle \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 0 \} x ^ { ( i ) } } { \displaystyle \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 0 \} } } } \\ { \mu _ { 1 } } & { { } = \; { \frac { \displaystyle \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 1 \} x ^ { ( i ) } } { \displaystyle \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 1 \} } } } \\ { \Sigma } & { { } = \; { \frac { 1 } { m } } \displaystyle \sum _ { i = 1 } ^ { m } ( x ^ { ( i ) } - \mu _ { y ^ { ( i ) } } ) ( x ^ { ( i ) } - \mu _ { y ^ { ( i ) } } ) ^ { T } . } \end{array}$$
## GDA和逻辑回归
[[GDA和逻辑回归推导]]

**结论：** GDA 是逻辑回归（Logistic Regression）的假设函数形式。
$$p ( y = 1 | x ; \phi , \Sigma , \mu _ { 0 } , \mu _ { 1 } ) = { \frac { 1 } { 1 + \exp ( - \theta ^ { T } x ) } } $$
如果 $p(x|y)$ 是多元高斯分布（具有共享的 $\Sigma$ ），那么 $p(y|x)$ 必然遵循逻辑函数；反之不然，可能是泊松分布或其他
GDA比逻辑回归对数据做出Stronger 的建模假设
事实证明，当建模假设成立时，GDA是asymptotically efficient(渐进高效) 的：在大 $m$ 的条件下，没有其他算法优于GDA
较弱假设时，逻辑回归更robust,对建模不敏感
# 朴素贝叶斯
在 GDA 中，特征向量 x 是连续的实值向量
朴素贝叶斯算法中，$x_i$ 是离散值

例：电子邮件分类
假设 训练集 一组标记为垃圾邮件或非垃圾邮件的电子邮件，指定电子邮件的特征 $x_i$

用长度为字典中单词数量的特征向量表示电子邮件特征，该特征向量称为 **vocabulary**

要构建判别模型，须对 $p(x|y)$ 建模
如果有 $50000$ 个单词的词汇表，那么 $x\in\{0,1\}^{50000}$ ,将使用 $2^{50000}$ 种可能的结果的多项式显式地对 $x$ 进行建模，参数过多
为了对 $p(x|y)$ 建模，需要作假设  **Naive** **Bayes** :假设给定 $y$ 时，$x$ 是独立的，$p(x_i|y)=p(x_i|y;x_j)$
得到的算法为 **Naive Bayes classifier** 朴素贝叶斯分类器
$$\begin{array} { r l } { p ( x _ { 1 } , \ldots , x _ { 5 0 0 0 0 } | y ) } & { { } = \ p ( x _ { 1 } | y ) p ( x _ { 2 } | y , x _ { 1 } ) p ( x _ { 3 } | y , x _ { 1 } , x _ { 2 } ) \cdots p ( x _ { 5 0 0 0 0 } | y , x _ { 1 } , \ldots , x _ { 4 9 9 9 9 } ) } \\ { { } } & { { } = \ p ( x _ { 1 } | y ) p ( x _ { 2 } | y ) p ( x _ { 3 } | y ) \cdots p ( x _ { 5 0 0 0 0 } | y ) } \\ { { } } & { { } = \ \displaystyle \prod _ { i = 1 } ^ { n } p ( x _ { i } | y ) } \end{array}$$
给定数据集 $(x^{(i)},y^{(i)})$ ，规定 $\phi_{i,y=0}=p(x=1|y=0),\phi_{i,y=1}=p(x=1|y=1),\phi_y=p(y=1)$ 
联合似然函数
$$\mathcal { L } ( \phi _ { y } , \phi _ { i | y = 0 } , \phi _ { i | y = 1 } ) = \prod _ { i = 1 } ^ { m } p ( x ^ { ( i ) } , y ^ { ( i ) } )$$
最大化得到
$$\begin{array} { r l } { \phi _ { j | y = 1 } } & { { } = { \frac { \sum _ { i = 1 } ^ { m } 1 \{ x _ { j } ^ { ( i ) } = 1 \land y ^ { ( i ) } = 1 \} } { \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 1 \} } } } \\ { \phi _ { j | y = 0 } } & { { } = { \frac { \sum _ { i = 1 } ^ { m } 1 \{ x _ { j } ^ { ( i ) } = 1 \land y ^ { ( i ) } = 0 \} } { \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 0 \} } } } \\ { \phi _ { y } } & { { } = { \frac { \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 1 \} } { m } } } \end{array}$$
即可计算后验概率
$$\begin{array} { r c l } { p ( y = 1 | x ) } & { = } & { \displaystyle \frac { p ( x | y = 1 ) p ( y = 1 ) } { p ( x ) } } \\ { } & { = } & { \displaystyle \frac { \left( \prod _ { i = 1 } ^ { n } p ( x _ { i } | y = 1 ) \right) p ( y = 1 ) } { \left( \prod _ { i = 1 } ^ { n } p ( x _ { i } | y = 1 ) \right) p ( y = 1 ) + \left( \prod _ { i = 1 } ^ { n } p ( x _ { i } | y = 0 ) \right) p ( y = 0 ) } } \end{array}$$

## 拉普拉斯平滑
以估计多项式随 机变量 $z$ 的平均值的问题为例，该变量取 $\{1,\dots , k\}$ 中的值
以 $\phi_i$ 参数化多项式
$$
\phi_i=\frac{\sum_{i=1}^{m}1\{z^{(i)}\}}{m}
$$
在计算后验概率时
$$
p(x=1|z)=\frac{\Pi_{i=1}^mp(z_i|x=1)p(x=1)}{p(z)}=\frac{\Pi_{i=1}^mp(z_i|x=1)p(x=1)}{\Pi_{i=1}^mp(z_i|x=1)p(x=1)+\Pi_{i=1}^mp(z_i|x=0)p(x=0)}
$$
当其中一个 $z_i$ 未出现时，后验概率为 $\frac00$ 无法计算
因此引入 拉普拉斯平滑
$$
\phi_i=\frac{\sum_{i=1}^{m}1\{z^{(i)}\}+1}{m+k}
$$
仍有$\sum_{i=1}^m\phi_i=1$
## 多项事件模型
根据单词的顺序和频率判断
规定 $x_i$ 第 $i$ 个单词，取值 $\{1,2,\dots,|V|\}$，$V$ 为词汇表
$n$ 个单词的电子邮件 由 长度为 $n$ 的向量 $(x_1,x_2,\dots,x_n)$ 表示

在多项事件模型中，我们假设电子邮件的生成过程如下：首先按照之前的方式，根据先验概率 $p(y)$ 随机确定这是垃圾邮件还是非垃圾邮件。然后，邮件发送者通过从单词的多项分布 $p(x_1|y)$ 中生成第一个词 $x_1$ 来开始撰写邮件。接下来，第二个词 $x_2$ 从**相同的多项分布**中独立于 $x_1$ 被选择，$x_3$、$x_4$ 等词也依此类推，直到生成邮件的全部 $n$ 个词。因此，整封邮件的总体概率由 $p(y) \prod_{i=1}^{n} p(x_i|y)$ 给出。

虽然这个公式看起来与我们之前在**伯努利事件模型**下计算邮件概率的公式相似，但公式中各项的含义现在**截然不同**。特别是，$x_i|y$ 现在服从**多项分布**，而不是伯努利分布。

给定数据集 $\{(x^{(i)},y^{(i)};i=1,2,\dots,m\}$,其中 $x^{(i)}=(x_1^{(i)},x_2^{(i)},\dots,x_n^{(i)})$ 


### 似然函数的展开（补充推导）

我们从联合概率的乘积开始：

$$
\mathcal{L}(\phi_y, \phi_{i|y=0}, \phi_{i|y=1}) = \prod_{i=1}^{m} p(x^{(i)}, y^{(i)})
$$
#### 第一步：使用条件概率分解联合概率

根据概率论中的基本关系：  
> $p(a, b) = p(a | b) \cdot p(b)$

对每个样本 $(x^{(i)}, y^{(i)})$，我们有：

$$
p(x^{(i)}, y^{(i)}) = p(x^{(i)} | y^{(i)}) \cdot p(y^{(i)})
$$

因此，整个似然函数变为：

$$
\mathcal{L} = \prod_{i=1}^{m} \left[ p(x^{(i)} | y^{(i)}) \cdot p(y^{(i)}) \right]
$$


#### 第二步：应用“朴素”假设 —— 特征条件独立

在**多项事件模型**中，文档 $x^{(i)}$ 是由 $n_i$ 个词组成的序列：  
$x^{(i)} = (x_1^{(i)}, x_2^{(i)}, ..., x_{n_i}^{(i)})$

由于“朴素”假设：**给定类别 $y$，各个词的位置是条件独立的**，所以：

$$
p(x^{(i)} | y^{(i)}) = \prod_{j=1}^{n_i} p(x_j^{(i)} | y^{(i)})
$$

代入上式得：

$$
\mathcal{L} = \prod_{i=1}^{m} \left[ \left( \prod_{j=1}^{n_i} p(x_j^{(i)} | y^{(i)}) \right) \cdot p(y^{(i)}) \right]
$$


#### 第三步：明确参数依赖关系

- $p(y^{(i)})$ 只依赖于先验参数 $\phi_y$ → 记作 $p(y^{(i)}; \phi_y)$
- $p(x_j^{(i)} | y^{(i)})$ 依赖于条件概率参数 $\phi_{k|y}$，其中 $k$ 是单词索引（即 $x_j^{(i)}$ 的值）→ 记作 $p(x_j^{(i)} | y^{(i)}; \phi_{i|y=0}, \phi_{i|y=1})$

于是最终形式为：

$$
\mathcal{L}(\phi_y, \phi_{i|y=0}, \phi_{i|y=1}) = \prod_{i=1}^{m} \left( \prod_{j=1}^{n_i} p(x_j^{(i)} | y^{(i)}; \phi_{i|y=0}, \phi_{i|y=1}) \right) \cdot p(y^{(i)}; \phi_y)
$$
最大化似然
$$\begin{array} { r l } { \phi _ { k | y = 1 } } & { { } = { \frac { \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n _ { i } } 1 \{ x _ { j } ^ { ( i ) } = k \land y ^ { ( i ) } = 1 \} } { \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 1 \} n _ { i } } } } \\ { \phi _ { k | y = 0 } } & { { } = { \frac { \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n _ { i } } 1 \{ x _ { j } ^ { ( i ) } = k \land y ^ { ( i ) } = 0 \} } { \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 0 \} n _ { i } } } } \\ { \phi _ { y } } & { { } = { \frac { \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 1 \} } { m } } . } \end{array}$$
拉普拉斯平滑
$$\begin{array} { r l } { \phi _ { k | y = 1 } } & { { } = { \frac { \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n _ { i } } 1 \{ x _ { j } ^ { ( i ) } = k \land y ^ { ( i ) } = 1 \} + 1 } { \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 1 \} n _ { i } + | V | } } } \\ { \phi _ { k | y = 0 } } & { { } = { \frac { \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n _ { i } } 1 \{ x _ { j } ^ { ( i ) } = k \land y ^ { ( i ) } = 0 \} + 1 } { \sum _ { i = 1 } ^ { m } 1 \{ y ^ { ( i ) } = 0 \} n _ { i } + | V | } } . } \end{array}$$