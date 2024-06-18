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


* 内存消耗降低。预排序算法需要的内存约是训练数据的两倍（2x样本数x维度x4Bytes），它需要用32位浮点来保存特征值，并且对每一列特征，都需要一个额外的排好序的索引，这也需要32位的存储空间。对于 直方图算法，则只需要(1x样本数x维度x1Bytes)的内存消耗，仅为预排序算法的1/8。因为直方图算法仅需要存储特征的 bin 值(离散化后的数值)，不需要原始的特征值，也不用排序，而bin值用8位整型存储就足够了。
* 算法时间复杂度大大降低。决策树算法在节点分裂时有两个主要操作组成，一个是“寻找分割点”，另一个是“数据分割”。从算法时间复杂度来看，在“寻找分割点”时，预排序算法对于深度为$k$的树的时间复杂度：对特征所有取值的排序为$O(NlogN)$，$N$为样本点数目，若有$D$维特征，则$O(kDNlogN)$，而直方图算法需要$O(kD \times bin)$(bin是histogram 的横轴的数量，一般远小于样本数量$N$)。
* 再举个例子说明上述两点的优化，假设数据集$A$的某个特征的特征值有（二叉树存储）：${1.2,1.3,2.2,2.3,3.1,3.3}$，预排序算法要全部遍历一遍，需要切分大约5次。进行离散化后，只需要切分2次 ${{1},{2,3}}$ 和 ${{1,2},{3}}$，除了切分次数减少，内存消耗也大大降低。
* 直方图算法还可以进一步加速。一个容易观察到的现象：一个叶子节点的直方图可以直接由父节点的直方图和兄弟节点的直方图做差得到。通常构造直方图，需要遍历该叶子上的所有数据，但直方图做差仅需遍历直方图的$k$个bin。利用这个方法，LightGBM可以在构造一个叶子的直方图后，可以用非常微小的代价得到它兄弟叶子的直方图，在速度上可以提升一倍。
