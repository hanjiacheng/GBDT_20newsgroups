from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from  sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
import numpy as np
train_data = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
print('训练数据总数',len(train_data.data))
test_data  = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
print('测试数据总数',len(test_data.data))
print('数据总数',len(train_data.data)+len(test_data.data))
# 训练文本词向量化
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data.data)
# 提取训练数据tf-idf特征
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# 初始化gbdt分类器
gb = GradientBoostingClassifier()
# 测试文本词向量化
X_test_counts = count_vect.transform(test_data.data)
# 提取测试数据tf-idf特征
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
'''
利用网格搜索寻找最佳迭代次数,从10-100每10次一个
'''
param_test1 = {'n_estimators':range(10,101,10)}
gsearch1 = GridSearchCV(gb, param_grid=param_test1, scoring='accuracy')
gsearch1.fit(X_train_tfidf, train_data.target)
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_, gsearch1.best_estimator_)
'''
控制步长在0.1条件下，寻找最佳迭代次数
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10),
                       param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(X,y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
'''

'''
在找到最佳步长的基础上
使用网格搜索决策树最大深度max_depth
param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20,
      max_features='sqrt', subsample=0.8, random_state=10),
   param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X,y)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
'''

'''
在找到最佳决策树最大深度max_depth的基础上
使用网格搜索调整内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf

param_test3 = {'min_samples_split':range(800,1900,200), 'min_samples_leaf':range(60,101,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7,
                                     max_features='sqrt', subsample=0.8, random_state=10),
                       param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(X,y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
'''

'''
在步长、max_depth、min_samples_split、min_samples_leaf的基础上
对最大特征数max_features进行网格搜索
param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7, min_samples_leaf =60,
               min_samples_split =1200, subsample=0.8, random_state=10),
                       param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X,y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
'''

