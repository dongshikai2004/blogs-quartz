2506.18088

# method
## data generation
![[Pasted image 20260313102244.png]]
把task info,api list,fuction example 交给 code agent
code agent 生成控制程序代码，执行任务多次
根据成功率，判断code是pass 还是fall，生成一个含运行视频的execution log
把这个 log 给vlm agent 进行分析，找哪一步出错
code agent根据log 和 vlm agent 的分析，更新代码

整个过程不需要人工参与，而且闭环
## 增强鲁棒性
1. 干扰物
2. 背景纹理
3. 光照
4. 桌面高度
5. 语言描述

## 抓握适应
不同机械臂有不同的结构和抓取策略
1. piper 侧向
2. franka 上下

# experimnet
## 代码评估
成功率，迭代次数，token
