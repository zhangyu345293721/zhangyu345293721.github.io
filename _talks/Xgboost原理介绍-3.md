---
title: "Xgboost原理介绍"
collection: talks
type: "Xgboost专栏"
permalink: /talks/Xgboost原理介绍k-3
date: 2024-03-05
location: "上海市"
---

### 简介

本文不讲如何使用 XGBoost 也不讲如何调参，主要会讲一下作为 GBDT 中的一种，XGBoost 的原理与相关公式推导。为了循序渐进的理解，读者可先从简单的回归树再到提升树再来看本文。我们现在直接从 XGBoost 的目标函数讲起。

### XGBoost 公式推导

XGBoost 的目标函数如下：
<br/><img src="/images/xgb_d1.png">
上面的两项加号前项为在训练集上的损失函数。其中 yiyi 表示真实值，ŷ iy^i 表示预测值。加号后项为正则项，到后面再看 ΩΩ 这个函数的具体形式。我们现在只需要知道 ΩΩ 的自变量为 fkfk，是决策树，而不是向量，所以是没有办法用和导数有关的方法来训练的（像梯度下降等）。

#### Boosting
何为 Boosting，这个可以主要在上面给的提升树文章中去了解。这里大概描述如下：
<br/><img src="/images/xgb_d2.png">

依此类推，每次迭代轮数均比上次迭代好一些，通式如下：
<br/><img src="/images/xgb_d3.png">

上面 (2) 式即为对于第 t 轮的 boosting 公式。我们再看此时的目标函数：
<br/><img src="/images/xgb_d4.png">

其中 (3) 式 nn 为样本数，tt 为树的棵数，也是迭代的轮数，yiyi 为真实值，ŷ iy^i 为第 tt 轮的预测值 。对于正则化项，又可以写成如下形式：
<br/><img src="/images/xgb_d5.png">
(4) 式这么写有什么好处呢？因为我们的方法是 boosting 逐轮计算的，所以当计算第 tt 轮时，前面 t−1t−1 轮事实上是已经计算出来了的。即 (4) 式的加号后项为常数。所以把 (2)(4) 式代入 (3) 式，有如下：
<br/><img src="/images/xgb_d6.png">

#### Taylor 展开

在 (5) 这个式子的基础上，我们就可以做点文章了。我们现在假定经验损失 L 是可以二阶 Taylor 展开的。把 ft(xi)ft(xi) 当成无穷小，就得到了如下式：

<br/><img src="/images/xgb_d7.png">

(6) 这个式子是比较抽象的，为帮助对 gigi 和 hihi 的理解，我们把常见的平方损失函数代入 (5) 可有：

<br/><img src="/images/xgb_d8.png">

展开有：
<br/><img src="/images/xgb_d9.png">


即：
<br/><img src="/images/xgb_d10.png">

套用一下 (6) 式，即有当损失函数为平方损失时 gi=2(ŷ (t−1)i−yi)gi=2(y^i(t−1)−yi)，hi=2hi=2。

我们再考察 (6) 式，其中的 (yi,ŷ (t−1)i)L(yi,y^i(t−1)) 意义是 t−1t−1 轮的经验损失，在执行第 tt 轮的时候，这一项其实已经也是一个已知的常数。那么优化目标就可以继续简化如下：

<br/><img src="/images/xgb_d11.png">

其中，gigi 和 hihi 可根据 Taylor 公式求得如下：

<br/><img src="/images/xgb_d12.png">

也就是说在给定损失函数的形式，则 gigi 和 hihi 就可以算出来。

#### 树的权重求解与结构分

我们从 (7) 式再往下，把 ft(x)ft(x) 搞清楚。有：

<br/><img src="/images/xgb_d13.png">

ft(x)ft(x) 表示第 tt 棵树，xx 为输入特征，其维数为 dd，ft(x)ft(x) 即为把特征映射到一个树的叶子结点上的一个数（权重），ft(x)ft(x) 可以分为 ww 和 qq 两部分，其中 qq 为把特征映射到一个 TT 个叶子结点的函数，相当于是决策树的结构。ww 为把每个叶子结点映射为其权重值。

树的函数搞清了，如果没有搞清楚，可以从本文开头提到的回归树原理与推导和提升树原理与推导再深入学习一下。

我们考察 (7) 式中的罚项 Ω(ft)Ω(ft)，这个罚项是用来惩罚树的复杂程度的。根据上面 ft(x)ft(x) 的描述，我们可以从树的结构与树叶子结点上的权重做罚项，我们定义如下：

<br/><img src="/images/xgb_d14.png">

其中 TT 为叶子节点个数，ww 为叶子节点权重。我们把 (10) 式加号前项叫 L0范数，加号后项为 L2范数，γγ 和 λλ 分别为各自超参数。

我们把 (10) 式，(9) 式都代入 (7) 式，有：

<br/><img src="/images/xgb_d15.png">

到这里，我们可以看到 (7) 式中优化事实上可以着重分为两大步，第一步拿到树结构，第二步再计算叶子节点上的权重大小。我们先令：

<br/><img src="/images/xgb_d16.png">







