---
title: "LightGBM相关知识介绍2"
collection: teaching
type: "lightGBM专栏"
permalink: /teaching/LightGBM相关知识介绍2
venue: "上海市"
date: 2024-04-16
location: "上海市"
---

同XGBoost类似，LightGBM依然是在GBDT算法框架下的一种改进实现，是一种基于决策树算法的快速、分布式、高性能的GBDT框架，主要说解决的痛点是面对高维度大数据时提高GBDT框架算法的效率和可扩展性。

## LightGBM

“Light”主要体现在三个方面，即更少的样本、更少的特征、更少的内存，分别通过单边梯度采样（Gradient-based One-Side Sampling）、互斥特征合并（Exclusive Feature Bundling）、直方图算法（Histogram）三项技术实现。另外，在工程上面，LightGBM还在并行计算方面做了诸多的优化，支持特征并行和数据并行，并针对各自的并行方式做了优化，减少通信量。

## LightGBM的起源

LightGBM起源于微软亚洲研究院在NIPS发表的系列论文：

1. [Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tie-Yan Liu. “A Communication-Efficient Parallel Algorithm for Decision Tree.” Advances in Neural Information Processing Systems 29 (NIPS 2016), pp. 1279-1287](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/6381-a-communication-efficient-parallel-algorithm-for-decision-tree.pdf)
2. [Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. “LightGBM: A Highly Efficient Gradient Boosting Decision Tree.” Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)

并于2016年10月17日在[LightGBM](https://link.zhihu.com/?target=https%3A//github.com/microsoft/LightGBM)上面开源，三天内在GitHub上面被star了1000次，fork了200次。知乎上现有近2000人关注“如何看待微软开源的LightGBM？”问题。
随后不断迭代，慢慢地开始支持Early Stopping、叶子索引预测、最大深度设置、特征重要性评估、多分类、类别特征支持、正则化（L1，L2及分裂最小增益）……
具体可以参阅以下链接：[LightGBM大事记](https://link.zhihu.com/?target=https%3A//github.com/Microsoft/LightGBM/blob/master/docs/Key-Events.md)
