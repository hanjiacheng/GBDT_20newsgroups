# GBDT_20newsgroups
##original_gbdt.py文件是使用默认参数的GBDT模型
得到的平均准确率是0.79，平均召回率是0.75，平均f1-score是0.76
##GBDT.py文件是想通过网格搜索寻找最佳参数，但由于训练时间过长，没有跑出具体的结果。
###主要的想法是：
1.控制步长在0.1条件下，利用网格搜索寻找最佳迭代次数
2.在找到最佳步长的基础上，使用网格搜索决策树最大深度max_depth
3.在找到最佳决策树最大深度max_depth的基础上，使用网格搜索调整内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf
4.在步长、max_depth、min_samples_split、min_samples_leaf的基础上，对最大特征数max_features进行网格搜索
5.综合测试
