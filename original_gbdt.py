from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from  sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
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
# 训练gbdt模型
#gb.fit(X_train_tfidf, train_data.target)
# 测试文本词向量化
X_test_counts = count_vect.transform(test_data.data)
# 提取测试数据tf-idf特征
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
# 进行分类
pred = gb.predict(X_test_tfidf)
# 打印测试结果,包含准确率、召回率、f1-score
print(metrics.classification_report(test_data.target,pred,target_names=test_data.target_names))
