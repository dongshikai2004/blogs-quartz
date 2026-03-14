生成超长文本 不爆上下文，能插入图片
# 架构
1. Planner：双粒度检查表，整体规划，细节确定
2. Searcher-Filter
	1. 查 文本，图片
	2. 过滤 无关信息
3. Reporter：增量合成，循环上下文

# 数据
有深度的任务列表，有多模态需求
gemini chatgpt5.3 完成


