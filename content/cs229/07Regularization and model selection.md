# Cross validation
## hold-out cross validation 
或称为 simple cross validation
1. 将 $S$ 按一定比例分割成 $S_{train}$ 和 $S_{cv}$ 
2. 仅在 $S_{train}$ 上训练，获得一些假设 $h_i$
3. 在 $S_{cv}$ 上验证，选择经验误差最小的 $h_i$

缺点：浪费一些数据
也可在所有步骤完成再次在整个 $S$ 上进行训练，但对初始值敏感的模型不行

## k-fold cross validation
1. 将 $S$ 随机分成 $k$ 份，互不相交，$S_1,S_2,\dots,S_k$
2. 对于每个模型，对于 $j = 1, \dots, k$ 在 $S_1 \cup \dots \cup S_{j-1} \cup S_{j+1} \cup \dots \cup S_k$（上训练模型 $M_i$，即，对除 $S_j$）之外的所有数据进行训练以获得一些假设 $h_{ij}$。在 $S_j$ 上检验假设 $h_{ij}$，得到 $\hat{\varepsilon}_{S_j}(h_{ij})$。然后将模型 $M_i$ 的估计泛化误差计算为 $\hat{\varepsilon}_{S_j}(h_{ij})$ 的平均值（对 $j$ 求平均值）。
3. 选择泛化误差最低的模型，并在整个数据集上重新训练

保留的数据更多，仅舍去 $\frac1k$，但训练更昂贵, $k$ 倍

有时取 $k=m$ 一次只提供一个训练示例，称为leave-one-out cross validation

# Feature Selection
给定 监督学习任务，假设 特征数量 $n>>m$  ，怀疑只有少量特征和任务相关；即使对 $n$ 输入特征使用简单的线性分类器， VC维度仍为 $O(n)$
可以使用特征选择算法来减少特征数量， $n$ 特征时，有 $2^n$ 种特征子集，需要
启发式搜索算法(wrapper model feature selection)
**forward search**
1. 初始化 $\mathcal{F} = \emptyset$。
2. 重复 {
    (a) 对于 $i = 1, \dots, n$ if $i \notin \mathcal{F}$，让 $\mathcal{F}_i = \mathcal{F} \cup \{i\}$，并使用某种版本的交叉验证来评估特征 $\mathcal{F}_i$。（即，仅使用 $\mathcal{F}_i$ 中的特征训练您的学习算法，并估计其泛化误差。）
    (b) 将 $\mathcal{F}$ 设置为步骤 (a) 中找到的最佳特征子集。 }
3. 选择并输出在整个搜索过程中评估的最佳特征子集。
当 $\mathcal{F} = \{1, \dots, n\}$ 是所有特征的集合，或者当 $|\mathcal{F}|$ 超过某个预设阈值（对应于希望算法考虑使用的特征的最大数量）时，算法的外循环可以终止。
$O(n^2)$
或者其他搜索方法，例如 backward search 从 $\mathcal F=\{1,2,\dots,n\}$ 逐步减少

**Filter feature selection**
计算分数 $S(i)$ 来衡量每个特征 $x_i$ 关于类标签 $y$ 的信息量，只选择具有最大分数 $S(i)$ 的 $k$ 特征
分数 $S(i)$ 可定义为 $x_i$ 和 $y$ 之间的相关性的绝对值，或者说 **mutual information** $\operatorname { M I }(x_i,y)$
 $$\operatorname { M I } ( x _ { i } , y ) = \sum _ { x _ { i } \in \{ 0 , 1 \} } \sum _ { y \in \{ 0 , 1 \} } p ( x _ { i } , y ) \log { \frac { p ( x _ { i } , y ) } { p ( x _ { i } ) p ( y ) } }$$
 上面的等式假设 $x_i$ 和 $y$ 是二进制值，更一般的是 在变量域上计算
 各概率都可根据在训练集上的经验分布估计
直观的 **Kull back-Leibler** (KL) 散度
$$\operatorname { M I } ( x _ { i } , y ) = \operatorname { K L } \left( p ( x _ { i } , y ) | | p ( x _ { i } ) p ( y ) \right)$$
如果 $x_i$ 和 $y_i$ 是独立的随机变量，将有 $p(x_i,y)=p(x_i)p(y)$ ，KL散度将为 $0$ 

如果已经 根据分数 $S(i)$ 得到了排名，该选择多少特征？
标准方法是 在交叉验证在 k 的可能值中进行选择

# Bayesian statistics and regularization
用 最大似然（ML）进行参数拟合
$$\theta _ { \mathrm { M L } } = \arg \operatorname* { m a x } _ { \theta } \prod _ { i = 1 } ^ { m } p ( y ^ { ( i ) } | x ^ { ( i ) } ; \theta )$$
两种观点
1. $\theta$ constant-valued but unknown：频率论者
	1. $\theta$ 不是随机，只是未知数，用统计程序来估计参数
2. $\theta$ random variable:Bayesian 的世界观
	1. 未 $\theta$ 指定先验概率，在给定训练集，要求对 $x$ 的新值预测时，计算 后验概率

$$
\begin{array} { r c l } { p ( \theta | S ) } & { = } & { \displaystyle \frac { p ( S | \theta ) p ( \theta ) } { p ( S ) } } \\ { } & { = } & { \displaystyle \frac { \left( \prod _ { i = 1 } ^ { m } p ( y ^ { ( i ) } | x ^ { ( i ) } , \theta ) \right) p ( \theta ) } { \int _ { \theta } \left( \prod _ { i = 1 } ^ { m } p ( y ^ { ( i ) } | x ^ { ( i ) } , \theta ) p ( \theta ) \right) d \theta } } \end{array}
$$
预测时
$$p ( y | x , S ) = \int _ { \theta } p ( y | x , \theta ) p ( \theta | S ) d \theta$$
如果目标是给定 $x$ 下预测 $y$ 的期望值：
$$\operatorname { E } [ y | x , S ] = \int _ { y } y p ( y | x , S ) d y$$
即是 完全贝叶斯

实际计算后验分布的计算很困难，不应积分
使用近似方程
$$\theta _ { \mathrm { M A P } } = \arg \operatorname* { m a x } _ { \theta } \prod _ { i = 1 } ^ { m } p ( y ^ { ( i ) } | x ^ { ( i ) } , \theta ) p ( \theta ) .$$
添加了 $p(\theta)$ 项

