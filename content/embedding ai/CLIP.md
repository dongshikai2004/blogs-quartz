# 原理
![[Pasted image 20260313112915.png]]

将 N 个文本特征和 N 个图像特征两两组合，组成$N\times N$ 矩阵
一一对应的 $N$ 个是正样本，$N\times N-N$ 负样本
目的：最大化正样本相似度，最小化负样本相似度

# zero-shot
![[Pasted image 20260313112908.png]]
1. 构建标签的文本特征：文本->text encoder
2. 输入图像，找到最相似的：图片->image encoder ,计算相似矩阵，softmax

# 缺点
域外泛化问题
