马尔可夫决策过程是强化学习的基本框架
![[Pasted image 20250924103202.png]]
简化版本
1. 马尔可夫过程（Markov process，MP）
2. 马尔可夫奖励过程（Markov reward process，MRP）
# 马尔可夫过程
## 马尔可夫性质
指一个随机过程在给定现在状态及所有过去状态情况下，其未来状态的条件概率分布仅依赖于当前状态
离散过程：假设随机变量 $X_0,X_1,⋯ ,X_T$​构成一个随机过程
如果 $X_{t+1}$对于过去状态的条件概率分布仅是 $X_t$的一个函数,则
$$
p\left(X_{t+1}=x_{t+1}\mid X_{0;t}=x_{0;t}\right)=p\left(X_{t+1}=x_{t+1}\mid X_{t}=x_{t}\right)
$$
- $X_{0:t}$ ：变量集合 $X_{0},X_{1},\cdot\cdot\cdot,X_{t}$ 
- $x_{0,t}$ ：状态空间中的状态序列$x_{0},x_{1},\cdot\cdot\cdot,x_{t}$  
未来的转移与过去的是独立的，未来只取决于现在的状态
## 马尔可夫链
设状态的历史为$h_{t}=\{s_{1},s_{2},s_{3},\ldots,s_{t}\}$ 
马尔可夫过程满足
$$
p\left(s_{t+1}\mid s_{t}\right)=p\left(s_{t+1}\mid h_{t}\right)
$$
离散时间的马尔可夫过程也称为**马尔可夫链（Markov chain）**，是最简单的马尔可夫过程，状态有限
![[Pasted image 20250924115649.png]]
用**状态转移矩阵（state transition matrix** $\bf P$ 来描述状态转移 $p\left(s_{t+1}=s^{\prime}\mid s_{t}=s\right)$ 
$$
\mathbf{P} = \left( \begin{array}{cccc}
p(s_1 \mid s_1) & p(s_2 \mid s_1) & \ldots & p(s_N \mid s_1) \\
p(s_1 \mid s_2) & p(s_2 \mid s_2) & \ldots & p(s_N \mid s_2) \\
\vdots & \vdots & \ddots & \vdots \\
p(s_1 \mid s_N) & p(s_2 \mid s_N) & \ldots & p(s_N \mid s_N)
\end{array} \right)
$$
## 例子
![[Pasted image 20250924120054.png]]
# 马尔可夫奖励过程
马尔可夫链加上奖励函数
奖励函数 $R$ 是一个期望，表示当我们到达某一个状态的时候，可以获得多大的奖励
状态数有限时，$R$是一个向量
折扣因子 $\gamma$ 
## 回报与价值函数
**范围（horizon）**：一个回合的长度（每个回合最大的时间步数），它是由有限个步数决定的
**回报（return）**：奖励的逐步叠加，假设时刻$t$后的奖励序列为${r_{t+1},\,r_{t+2},\,r_{t+3},\cdot\cdot\cdot}$ ,则回报为
$$
G_{t}=r_{t+1}+\gamma r_{t+2}+\gamma^{2}r_{t+3}+\gamma^{3}r_{t+4}+\ldots+\gamma^{T-t-1}r_{T}
$$
- $T$ 最终时刻
- $\gamma$ 折扣因子，越往后的奖励折扣越多
状态价值，回报的期望
$$
\begin{array}{l}
{{V^{t}(s)=\mathbb{E}\left[G_{t}\mid s_{t}=s\right]}}\\ {{=\mathbb{E}\left[r_{t+1}+\gamma r_{t+2}+\gamma^{2}r_{t+3}+\ldots+\gamma^{T-t-1}r_{T}\mid s_{t}=s\right]}}\end{array}
$$
### 使用折扣因子的原因
1. 有些马尔可夫过程带环，不会终结
2. 不能建立完美的模拟环境的模型，对未来的评估不一定准确，需要尽快获得奖励
3. 奖励有实际价值，希望立刻获得奖励
4. 更想获得及时奖励：$\gamma=0$；对未来奖励没有折扣：$\gamma=1$ 
折扣因子 $\gamma$ 是超参数，调整折扣因子可以得到不同动作的智能体
## example
![[Pasted image 20250924183144.png]]
设置奖励函数 $\bf R$为向量 $[5,0,0,0,0,0,10]$ ，即进入 $s_1$ 和 $s_7$ 时才会分别得到 $5$ 和 $10$ 的奖励，进入其他状态无奖励
对 $4$ 步的回合， $\gamma=0.5$ 来采样回报
$$
s_4,s_5,s_6,s_7:G=0+0*0.5+0*0.5^2+10*0.5^3=1.25
$$
### 蒙特卡洛（Monte Carlo，MC）采样
得到轨迹的回报后,将轨迹叠加取均值作为出发点的价值
## 贝尔曼方程
$$
V(s)=R(s)~+\gamma\sum_{s'\in \bf S}\!\!p\left(s^{\prime}\mid s\right)V\left(s^{\prime}\right)
$$
从价值函数里面推导出,前半即时奖励,后半未来奖励的折扣总和
- $s'$未来的所有状态
- $p(s'|s)$ 状态转移概率
- $V(s')$ 未来某一状态的价值
- $R(s)$ 即时奖励
### 全期望公式
law of total expectation
$$
\mathbb{E}[V(s_{t+1})|s_{t}]=\mathbb{E}[\mathbb{E}[G_{t+1}|s_{t+1}]|s_{t}]=\mathbb{E}[G_{t+1}|s_{t}]
$$
#declare 全期望公式
$$
\mathbb{E}[X]=\sum_{i}\mathbb{E}\left[X\mid A_{i}\right]p\left(A_{i}\right)
$$
- $A_i$ 是样本空间的有限或可数的划分（partition）
#declare 条件期望
$$
\mathbb{E}[X\mid Y=y]=\sum_{x}x p(X=x\mid Y=y)
$$
- $X$ 和 $Y$ 都是离散型随机变量
为使符号简洁,令$s=s_{t},\ g^{\prime}=G_{t+1},\ s^{\prime}=s_{t+1}$ 
由全期望公式和条件期望公式
$$
\begin{align*}
\mathbb{E}[\mathbb{E}[G_{t+1} \mid s_{t+1}] \mid s_t] &= \mathbb{E}[\mathbb{E}[g' \mid s'] \mid s] \\
&= \mathbb{E}\left[\sum_{g'} g' p(g' \mid s') \mid s\right] \\
&= \sum_{s'} \sum_{g'} g' p(g' \mid s', s) p(s' \mid s) \\
&= \sum_{s'} \sum_{g'} \frac{g' p(g' \mid s', s) p(s' \mid s) p(s)}{p(s)} \\
&= \sum_{s'} \sum_{g'} \frac{g' p(g' \mid s', s) p(s', s)}{p(s)} \\
&= \sum_{s'} \sum_{g'} \frac{g' p(g', s', s)}{p(s)} \\
&= \sum_{s'} \sum_{g'} g' p(g', s' \mid s) \\
&= \sum_{g'} \sum_{s'} g' p(g', s' \mid s) \\
&= \sum_{g'} g' p(g' \mid s) \\
&= \mathbb{E}[g' \mid s] = \mathbb{E}[G_{t+1} \mid s_t]
\end{align*}
$$
### 贝尔曼方程推导
$$
\begin{align*}
V(s) &= \mathbb{E}[G_t \mid s_t = s] \\
&= \mathbb{E}[r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots \mid s_t = s] \\
&= \mathbb{E}[r_{t+1} \mid s_t = s] + \gamma \mathbb{E}[r_{t+2} + \gamma r_{t+3} + \gamma^2 r_{t+4} + \ldots \mid s_t = s] \\
&= R(s) + \gamma \mathbb{E}[G_{t+1} \mid s_t = s] \\
&= R(s) + \gamma \mathbb{E}[V(s_{t+1}) \mid s_t = s] \\
&= R(s) + \gamma \sum_{s' \in S} p(s' \mid s) V(s')
\end{align*}
$$
### example
![[Pasted image 20250924191608.png]]
$$
\begin{pmatrix}
V(s_1) \\
V(s_2) \\
\vdots \\
V(s_N)
\end{pmatrix}
=
\begin{pmatrix}
R(s_1) \\
R(s_2) \\
\vdots \\
R(s_N)
\end{pmatrix}
+
\gamma
\begin{pmatrix}
p(s_1 \mid s_1) & p(s_2 \mid s_1) & \ldots & p(s_N \mid s_1) \\
p(s_1 \mid s_2) & p(s_2 \mid s_2) & \ldots & p(s_N \mid s_2) \\
\vdots & \vdots & \ddots & \vdots \\
p(s_1 \mid s_N) & p(s_2 \mid s_N) & \ldots & p(s_N \mid s_N)
\end{pmatrix}
\begin{pmatrix}
V(s_1) \\
V(s_2) \\
\vdots \\
V(s_N)
\end{pmatrix}
$$
求解析解
$$
\begin{array}{c}{{V=R+\gamma P V}}\\ {{T V=R+\gamma P V}}\\ {{(I-\gamma P)V=R}}\end{array}
$$
得到
$$
{{V=(I-\gamma P)^{-1}R}}
$$
矩阵求逆复杂度 $O(N^3)$ 只能求小量的
## 迭代算法求马尔可夫奖励过程
### 蒙特卡洛(采样)
![[Pasted image 20250924192205.png]]
### 动态规划
自举(bootstrapping) 来迭代 贝尔曼方程
停止条件:更新的状态与上一个状态区别不大
![[Pasted image 20250924192620.png]]
# 马尔可夫决策过程
状态转移新增条件 $p\left(s_{t+1}=s^{\prime}\mid s_{t}=s,a_{t}=a\right)$ ,未来的状态依赖现在的状态和当前的动作
$$
p\left(s_{t+1}\mid s_{t},a_{t}\right)=p\left(s_{t+1}\mid h_{t},a_{t}\right)
$$
奖励函数新增动作$R\left(s_{t}=s,a_{t}=a\right)=\mathbb{E}\left[r_{t}\mid s_{t}=s,a_{t}=a\right]$ 
## 马尔可夫决策过程中的策略
$$
\pi(a\mid s)=p\left(a_{t}=a\mid s_{t}=s\right)
$$
- 表示在当前状态 $s$ 采取 动作 $a$ 的概率
状态转移函数 基于当前状态和策略采取的动作,将动作去除,得到
$$
P_{\pi}\left(s^{\prime}\mid s\right)=\sum_{a\in A}\pi(a\mid s)p\left(s^{\prime}\mid s,a\right)
$$
奖励函数 将动作去除,得到
$$
r_{\pi}(s)=\sum_{a\in A}\pi(a\mid s)R(s,a)
$$
## 马尔可夫决策过程 与 马尔可夫过程/马尔可夫奖励过程 的区别
![[Pasted image 20250924193613.png]]
- 马尔可夫过程/马尔可夫奖励过程 完全状态转移概率决定
- 马尔可夫决策过程 中间多了一层动作 $a$ 
## 马尔可夫决策过程中的价值函数
价值函数定义为
$$
V_{\pi}(s)=\mathbb{E}_{\pi}\left[G_{t}\mid s_{t}=s\right]
$$
期望基于采取的策略,策略决定后,通过对策略进行采样得到期望
 **Q 函数（Q-function）**:动作价值函数
 定义在某一状态采取某一动作,可能得到的回报的期望
$$
Q_{\pi}(s,a)=\mathbb{E}_{\pi}\left[G_{t}\mid s_{t}=s,a_{t}=a\right]
$$
Q函数对动作加和,得到价值函数
$$
V_{\pi}(s)=\sum_{a\in A}\pi(a\mid s)Q_{\pi}(s,a)
$$
对Q函数的贝尔曼方程进行推导
$$
\begin{align*}
Q(s, a) &= \mathbb{E}[G_t \mid s_t = s, a_t = a] \\
&= \mathbb{E}[r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots \mid s_t = s, a_t = a] \\
&= \mathbb{E}[r_{t+1} \mid s_t = s, a_t = a] + \gamma \mathbb{E}[r_{t+2} + \gamma r_{t+3} + \gamma^2 r_{t+4} + \ldots \mid s_t = s, a_t = a] \\
&= R(s, a) + \gamma \mathbb{E}[G_{t+1} \mid s_t = s, a_t = a] \\
&= R(s, a) + \gamma \mathbb{E}[V(s_{t+1}) \mid s_t = s, a_t = a] \\
&= R(s, a) + \gamma \sum_{s' \in S} p(s' \mid s, a) V(s')
\end{align*}
$$
将Q函数拆成 即时奖励 和 后续状态的折扣价值
## 贝尔曼期望方程
Bellman expectation equation
对状态价值函数 进行分解,得到 贝尔曼期望方程
$$
V_{\pi}(s)=\mathbb{E}_{\pi}\left[r_{t+1}+\gamma V_{\pi}\left(s_{t+1}\right)\mid s_{t}=s\right]
$$
对Q函数,类似分解,得到Q函数的贝尔曼期望方程
$$
Q_{\pi}(s,a)=\mathbb{E}_{\pi}\left[r_{t+1}+\gamma Q_{\pi}\left(s_{t+1},a_{t+1}\right)\mid s_{t}=s,a_{t}=a\right]
$$
进一步分解,有
式1
$$
V_{\pi}(s)=\sum_{a\in A}\pi(a\mid s)Q_{\pi}(s,a)
$$
和
式2
$$
Q_{\pi}(s,a)=R(s,a)+\gamma\sum_{s^{\prime}\in{\cal S}}p\left(s^{\prime}\mid s,a\right)V_{\pi}\left(s^{\prime}\right)
$$
式2 代入 式1 ,得
$$
V_{\pi}(s)=\sum_{a\in A}\pi(a\mid s)\left(R(s,a)+\gamma\sum_{s^{\tau}\in S}p(s^{\prime}\mid s,a)\,V_{\pi}\left(s^{\prime}\right)\right)
$$
- 当前状态的价值 与 未来状态价值的关联
式1 代入 式2 ,得
$$
Q_{\pi}(s,a)=R(s,a)+\gamma\sum_{s^{\prime}\in S}p\left(s^{\prime}\mid s,a\right)\sum_{a^{\prime}\in A}\pi\left(a^{\prime}\mid s^{\prime}\right)Q_{\pi}\left(s^{\prime},a^{\prime}\right)
$$
- 当前时刻Q函数与未来时刻Q函数的关联
贝尔曼期望方程的另一种形式
## 备份图(回溯图)
备份backup 
类似于自举之间的迭代关系，对于某一个状态，它的当前价值是与它的未来价值线性相关
![[Pasted image 20250924203058.png]]
依次往上备份,拆解为
![[Pasted image 20250924204208.png]]
