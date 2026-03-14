给定 训练集 $\{x^{(1)},\dots,x^{(m)}\}$ 选出有凝聚力的聚类

1. Initialize **cluster centroids** $\mu_1, \mu_2, \ldots, \mu_k \in \mathbb{R}^n$ randomly.
2. Repeat until convergence: {
   For every $i$, set
   $$c^{(i)} := \arg\min_j \|x^{(i)} - \mu_j\|^2.$$
   For each $j$, set
   $$\mu_j := \frac{\sum_{i=1}^m 1\{c^{(i)} = j\} x^{(i)}}{\sum_{i=1}^m 1\{c^{(i)} = j\}}.$$
   }

证明 k-means 收敛
定义 **distortion function** 
$$J ( c , \mu ) = \sum _ { i = 1 } ^ { m } \| x ^ { ( i ) } - \mu _ { c ^ { ( i ) } } \| ^ { 2 }$$
$J$ 测量所有样本到集群中心的距离的平方和
$J$ 是非凸函数，振荡，受局部最优影响
常见操作 k-means 多次：对集群中心使用不同随机初始值
