---
title: "XGBoost算法介绍"
collection: talks
type: "Xgboost专题"
permalink: /talks/XGBoost基础-1
date: 2023-03-02
location: "上海市"
---

XGBoost 最初是由 Tianqi Chen 作为分布式（深度）机器学习社区（DMLC）组的一部分的一个研究项目开始的。XGBoost后来成为了Kaggle竞赛传奇——在2015年的時候29个Kaggle冠军队伍中有17队在他们的解决方案中使用了XGboost。

请记住，xgboost指南，如下所示：

1. 概览 XGBoost
2. 特征重要性
3. 建筑模型

### 概览 XGBoost

XGBoost 或 Extreme Gradient Boosting 是一种提供可靠和可扩展的数据以便进行分类和回归的算法。这种算法类似于数据挖掘，它使用决策树、随机森林和迭代来提供预测，以建立可靠和可扩展的模型。

XGBoost 使用梯度提升算法，实现了在迭代过程中最小化剩余损失的最优功能。该算法使用正则化技术来消除过度拟合，并提高数据应用中的准确性、可扩展性和有效性。

<br/><img src='/images/xgb_21.png'>

XGBoost 的应用不能提供图像、声音、技术以及其他与网络相关的数据。

### 特征重要性

XGBoost 中的特征重要性决定了模型是否可以为成员提供贡献。 XGBoost 认为重要性衡量标准是，在决策树的划分（分裂）过程中，必须考虑到这一点。重要的是，要使预测模型更具重要性，就必须建立预测模型。

<br/><img src='/images/xgb_22.png'>

### 构建模型

我们参考了 XGBoost 来实现斯隆数字巡天计划。与案例研究相关的分析属于第二模块。

```python
xgb = XGBClassifier(n_estimators= 100 ) 
training_start = time.perf_counter() 
xgb.fit(X_train, y_train) 
training_end = time.perf_counter() 
prediction_start = time.perf_counter() 
preds = xgb.predict(X_test) 
prediction_end = time.perf_counter() 
acc_xgb = (preds == y_test). sum ().astype( float ) / len (preds)* 100
 xgb_train_time = training_end-training_start 
xgb_prediction_time = prediction_end-prediction_start 
print ( "XGBoost的预测准确率为： %3.2f" % (acc_xgb)) 
print ( "训练耗时： %4.3f" % (xgb_train_time)) 
print ( "预测耗时： %6.5f 秒" % (xgb_prediction_time)) 

'''
XGBoost的预测准确率为： 99.21
训练耗时： 0.390
预测耗时： 0.01856 秒
'''
```

### 特征重要性


```python
importances = pd.DataFrame({
    'Feature': sdss_df_fe.drop('class', axis=1).columns,
    'Importance': xgb.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)
importances = importances.set_index('Feature')
importances
```

```python
importances.plot.bar()
```
<br/><img src='/images/xgb_23.png'>  <br/>

### XGBoost评估

confusion_matrix(y_test, preds)

'''
array([[1634,    9,   10],
       [   6,  259,    0],
       [   1,    0, 1381]])
'''

输出结果比第一级菜单的显示结果高出 1634 个数据点，且已进行分类。 9 星评级的球员表现不错，而 10 星评级的球员表现则更好。

在这其中，有 259 项评级获得了成功，6 项评级获得了成功，而 0 项评级获得了失败。

在这 100 多个奖项中，有 1381 项获得认可，1 项获得认可，还有 0 项获得认可。

#### 精确度和召回率

```python
print("Precision:", precision_score(y_test, preds, average='micro'))
print("Recall:",recall_score(y_test, preds, average='micro'))

OUTPUT
Precision: 0.9921212121212121
Recall: 0.9921212121212121

```

### F-1 分数

```python
print("F1-Score:", f1_score(y_test, preds, average='micro'))

F1-Score: 0.9921212121212121
```

F-1 分数是由召回率和精确度结合起来的。<br/>

总体而言，XGBoost 模型在评估目标的各项指标、性能或风险时表现得非常出色。经过评估后，召回了 f-1 和 f-2。
