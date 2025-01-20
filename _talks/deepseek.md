---
title: "DeepSeek V3 综述"
collection: talks
permalink: /talks/DeepSeek
date: 2024-12-30
---


> Deepseek系列更看重“成本”与“效率“的平衡。
> 训练DeepSeek-V3每万亿tokens仅需要180KH800 GPU小时，假设H800 GPU的租赁价格为每GPU小时2美元，总训练成本仅为557.6万美元。
DeepSeek-V3 是一款拥有671B参数的大型混合专家(MoE) 模型，其中每个token 会有37 B参数被激活。
14.8T高质量token上完成了训练。
<br/>
<img src='/images/deep_1.png'>
<img src='/images/deep_2.png'>


## 上下文能力

通过两个各包含1000 步的额外训练阶段，将上下文窗口从4K 依次扩展至32K 和128K。

<img src='/images/deep_3.png'>

## 训练特点

第一个（至少在开源社区内）成功使用FP8混合精度训练得到的大号MoE模型。
在并行策略上，DeepSeek-V3使用64路的专家并行，16路的流水线并行（Chimera ），以及数据并行（ZeRO1）

<img src='/images/deep_4.png'>


## 指标

>TTFT（Time-To-First-Token）：首token 的生成时间，主要衡量Prefill 阶段性能。

## 特点
> Prefill 阶段：计算密集型（compute-bound）。在流量较大或用户提示长度较长时，Prefill 的计算压力更大。完成KV Cache 的生成后，Prefill 阶段本身无需继续保留这些缓存。

> Decode 阶段：访存密集型（memory-bound）。由于逐token 生成的特性，Decode 阶段需频繁访问KV Cache，因此需要尽可能多地保留缓存数据以保障推理效率

## 策略

> Prefill 阶段：吞吐量随batch size 增加逐渐趋于平稳。这是因为Prefill 的计算受限特性（compute-bound），当batch 中的总
token 数超过某个阈值时，计算资源成为瓶颈。

> Decode 阶段：吞吐量随batch size 增加显著提升。增大batch size 可提高计算效率，从而显著增加吞吐量

<img src='/images/deep_5.png'>


<img src='/images/deep_6.png'>


**MOE**，MoE节省flops的好处主要体现在计算密集的prefill阶段

> 而在访存密集的decode阶段，MoE巨大的参数量然而会带来更加昂贵的数据搬移开销。

## 推理部署

DeepSeek-V3采取PD分离的方式，分别应对prefill和
decode两阶段的挑战。
32路：在prefill阶段，attention模块采用4路张量并行+8 路数据并行，moe模块采用32路专家并行。这样并行的目的是在满足首token时延的要求下，最大化系统吞吐（和训练任务类似）。
320路：在decode阶段，DeepSeek-V3采取320路专家并行（256个小专家+64个热点专家），有效降低解码时延，并缓解负载不均衡的问题。

<img src='/images/deep_7.png'>

> Prefill阶段由4个node，共32个GPU组成。并行策略上：
Attention部分。使用TP=4 (along with SP), DP=8的策略。此处设置较小的TP数量4，以尽可能通过计算与通信重叠掩盖all-reduce 通信耗时。

> MOE部分。EP=32，平均每个rank 10个experts，其目的是为了尽可能增加每个expert的batch size，从而可能增加MOE部分的计算强度，提高MFU利用率。

> 专家分布与负载均衡优化。由于MOE存在负载不均衡的问题，其优化目标是需要保证每个GPU尽可能处理相同的token数量。

> 冗余专家(redundant experts)的概念，高负载专家是指所有experts中处理token数较多的专家。冗余专家策略是指对高负载专家(high-load experts) 进行复制，在不同rank之间冗余存放。
冗余专家根据时域内的激活的tokens数量动态调整。
。DeepSeek-V3在Prefill阶段使用了32个冗余专家，与GPU数量一致，即每个GPU除了存放原本的8个专家外，还要存放1个冗余专家。

> Decode阶段包含40个node，320个GPU。在并行策略上，Attn部分TP=4(along with SP)，DP=80。MOE 部分EP=320。整体上Attn部分与Prefill阶段一致，下面主要介绍MOE部分。

> 专家分布与负载均衡优化。在320个GPU里，其中256个GPU每个rank保留一个专家，有64个GPU要负责冗余专家和shared专家。

> 冗余专家的选择策略与prefill阶段一致，也是根据时域内的激活的tokens数量动态调整。

> device-limited Routing
routing 更多的experts 自然会增加路由的难度，除了负载均衡的问题，也有通信开销问题。
单个token routing 的expert 如果分布在多卡甚至多机上（取决于expert的并行情况），显然会带来通信开销。

> 每个token最多放到M个设备上

> 1.首先选最高得分专家所在的M 个设备

> 2.在这M 个设备的专家中选择topK个专家;这样就可以将通信过程限制在M 个设备上，降低通信成本，论文表示该方法在M 大于3的情况下收益比较明显(相对传统topK路由)。


## 主要创新点
1.Multi-Head Latent Attention (MLA)
2.DeepSeekMoE
3.多Token预测（Multi-Token Prediction）4.FP8训练

<img src='/images/deep_8.png'>


## Multi-Head Latent Attention (MLA)

<img src='/images/deep_9.png'>

<img src='/images/deep_10.png'>

<img src='/images/deep_11.png'>


## 3.DeepSeekMoE

### MoE架构
 > 由于稀疏性，单个token的计算量不大（代价是更多显存加载），可以在更低成本下完成同等模型的预训练，推理速
度也更快。

<img src='/images/deep_12.png'>

> 1.MoE 引入了多个expert，实际上每个expert都是FFN层，导致了更多的参数量和更多的显存占用，后续引入了专家并
行和因此而来的通信开销（Gshard中是alltoall）
> 2.MoE 的routing/gating 函数如何保证各expert在推理/训练时的负载平衡，而不是集中在某个expert上，导致计算利用
率下降

专家数量多，选择也多：提升模型的泛化能力
一共E个专家，那么可以比较好覆盖 的任务类型只有E种，如果每次可以路由N个专家，那么可以覆盖的任务类型
就包括CNE 种。Nr=256，Ns=64

<img src='/images/deep_13.png'>

### 负载均衡策略
每个专家都能照顾到

<img src='/images/deep_14.png'>

deepseek的drop策略不是expert级的，而是device级的。策略如下：
1.计算每个设备的平均的预算容量
2.丢弃给到该设备的且亲和性最差的token，直到达到预算容量
3.确保至少10%的训练token 不会被丢弃

<img src='/images/deep_15.png'>


## 4.FP8模型

### FP8好处
1.节约内存
2.减少通信传播量
3.计算速度快

> 现有FP8方案的训练困难主要来自两个方面，一个是粗粒度的per-tensorE4M3量化会因为个别异常值增加量化误
差，另一个则是反向过程中使用的E5M2格式会带来较大的舍入误差。
为了解决以上问题，DeepSeek-V3在训练过程中统一使用E4M3格式，并通过细粒度的per-tile（1x128）和pergroup（128x128）量化来降低误差。

FP8支持两种Dtype
e4m3:更精确的数值，但是较小的动态范围。
e5m2，较大的范围，精确差。
Deepseek的FP8训练里，它保持了较精确的数值。全程使用了e4m3。

<img src='/images/deep_16.png'>


<img src='/images/deep_17.png'>


<img src='/images/deep_18.png'>

<img src='/images/deep_19.png'>

<img src='/images/deep_20.png'>


## 总结：

> 1.MoE技术提高推理速度，降低成本
> 2.MLA减少缓存压力
> 3.FP8，开源第一家，降低训练成本

