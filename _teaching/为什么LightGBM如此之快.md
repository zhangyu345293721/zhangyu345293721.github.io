---
title: "为什么LightGBM如此之快"
collection: teaching
type: "lightGBM专栏"
permalink: /teaching/为什么LightGBM如此之快
venue: "上海市"
date: 2024-03-02
location: "上海市"
---

Light GBM 代表轻量级梯度提升机，因此，它是另一种使用基于树的学习的梯度提升算法，由微软开发：
与任何其他梯度提升算法一样，它主要用 C++ 实现。此外，它还可以与许多其他扩展（Python、R、Java 等）交互。
由于某些独特的特点，它与其他梯度提升算法区别开来。

### 模型解释

1. 它更快、更高效
2. 它使用更少的内存
3. 提供更高的准确性
4. 支持并行计算和GPU学习
5. 可以处理大数据

### 树木叶子的生长
决策树的学习过程可以采用两种策略，即逐层（也称为逐深度）或逐叶。
在逐级策略中，树在生长过程中保持平衡，确保每个级别都得到充分发展后再进入下一个级别。相反，在逐叶策略中，重点是分裂叶子，从而最大程度地减少损失，这可能会造成树不平衡。

<br/><img src="/images/ll1.png"><br/>
在 LightGBM 中，树木采用“逐叶”方法生长。
<br/><img src="/images/ll2.png"><br/>
LightGBM 每次向树中添加一个节点，根据增益选择下一个节点，而不管树的深度如何。此过程会导致树更深且更不平衡。虽然这样的树结构往往具有较低的训练误差，但也可能导致过度拟合，尤其是在较小的数据集中。因此，LightGBM 本质上更适合大型数据集。当将其应用于小型数据集时，仔细调整超参数对于防止过度拟合至关重要。

### Bins
LightGBM 速度很快，因为它通过将连续特征值离散化为离散箱来最大限度地降低计算成本，从而有效地将连续特征转换为直方图。
假设一个特征具有许多不同的值。在这种情况下，潜在的分割点数量将非常多。
例如，分割应在 2 和 3 之间进行，还是在 12 和 13 之间进行？
<br/><img src="/images/ll3.png"><br/>
相反，LightGBM 将连续值分成多个箱，与其他梯度提升算法相比，显著减少了需要探索的潜在分割数量。
<br/><img src="/images/ll4.png"><br/>
### 独家功能捆绑
LightGBM 旨在有效管理稀疏特征而不影响准确性。通过独占特征捆绑 (EFB)，稀疏特征被组合成更密集的特征，从而降低复杂性。这可以加快训练过程并降低内存消耗。

<br/><img src="/images/ll5.png"><br/>

假设我们有两个如上所述的特征；它们可以有效地合并为一个向量，而不会丢失任何信息。该算法结合了一种搜索此类组合的机制，并努力创建这些特征的捆绑包。因此，这个过程减少了特征的数量，改善了内存消耗和训练时间。
#### 分布式学习
当需要更多计算能力时，我们可以使用多台机器同时使用LightGBM。
<br/><img src="/images/ll6.png"><br/>
每个工作者（或等级）持有不同的数据块并负责相应的计算。

### 参数

1. objective：这定义了问题的类型。regression表示回归问题。binary表示二元分类问题。
2. metric：评估指标。默认值为 ’ auto ‘，在这种情况下，算法会自行选择指标。有很多指标选项：’ rmse ’ 是回归问题的均方根误差；’ binary_logloss ’ 用于二分类问题， multi_logloss ’ 用于多类分类问题
3. boosting：可以定义 Boosting 方法。默认为 ’ gbdt '。其他选项包括 ’ rf ‘、’ dart ’ 和 ’ goss '。dart比较有名
4. lambda_l1：L1正则化参数
5. lambda_l2：L2正则化参数
6. feature_fraction：列采样。每次迭代都会选择一个特征子集。默认值 = 0.9
7. bagging_fraction：类似于“特征分数”，但它会随机选择部分数据而不进行重新采样。默认值为 0.9
8. bagging_freq：0 表示不装袋。在每个给定值（装袋频率）处装袋。默认值为 1
9. num_leaves：树中的最大叶子数。默认值为 64
10. max_depth：你可以限制树的深度。避免过度拟合的重要参数
11. max_bin：每个 bin 的最大数量。值越小，泛化效果越好。避免过度拟合的重要参数
12. num_iterations：迭代次数
13. learning_rate：默认值为 0.1
14. early_stopping_round：如果经过 n 轮迭代后指标没有改善，则训练将停止
15. min_data_in_leaf：每个叶子中必须包含的最小数据。避免过度拟合的重要参数。默认值为 3
16. min_sum_hessian_in_leaf：叶子中 hessian 的最小和。默认值 = 1e-3

过度拟合时应采取的预防措施：

1. 较小的max_bin
2. num_leaves较小
3. 使用min_data_in_leaf和min_sum_hessian_in_leaf
4. 使用 bagging_fraction 和 bagging_freq
5. 使用feature_fraction
6. 扩展你的训练数据量
7. 使用正则化
8. 限制深度：max_depth

### Python 代码

安装 Python 包；（您可以在此处找到有关如何使用 CUDA 或其他选项安装包的相关信息。）

```python
pip install lightgbm
#for Anaconda
conda install -c conda-forge lightgbm
```
让我们生成一个虚拟的二元分类数据：
<br/><img src="/images/ll7.png"><br/>
<br/><img src="/images/ll8.png"><br/>
<br/><img src="/images/ll9.png"><br/>

```python
从 sklearn.model_selection 导入 train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```
让我们使用** GridSearchCV** 来调整超参数：

```python
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

param_grid = {
    'objective': ["binary"],
    'metric': ['binary_logloss'],
    'boosting': ['gbdt','dart'],
    'lambda_l1': [0, 0.1, 0.01],
    'bagging_fraction': [0.7, 0.9],
    'bagging_freq': [0,1],
    'num_leaves': [32, 64, 128],
    'max_depth': [-1, 10],
    'max_bin': [100, 255],
    'num_iterations': [100, 200],
    'learning_rate':[0.01,0.1],
    'min_data_in_leaf': [3,5]
}

estimator = lgb.LGBMClassifier()

searcher = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid, 
        cv=5, 
        n_jobs=-1, 
        scoring='accuracy',
        verbose=2)

grid_model = searcher.fit(X_train, y_train)

y_pred =grid_model.predict(X_test)

```

```python
print(grid_model.best_score_)
print(grid_model.best_params_)
```






