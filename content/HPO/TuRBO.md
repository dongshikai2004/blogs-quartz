## 信赖域
对于无约束优化问题
![[Pasted image 20260407215600.png]]
在 $x_k$ 处泰勒展开
![[Pasted image 20260407215622.png]]
用Hessian 矩阵的近似 $B_k$ 表示二阶导
重写为
![[Pasted image 20260407215726.png]]
与原问题的差异在二阶项
d 极小时可忽略
信赖域求解转化为
![[Pasted image 20260407215823.png]]


定义比值
![[Pasted image 20260407215845.png]]
![[Pasted image 20260407215919.png]]

## 信赖域算法
![[Pasted image 20260407215938.png]]

## TuRBO
![[Pasted image 20260407215515.png]]

