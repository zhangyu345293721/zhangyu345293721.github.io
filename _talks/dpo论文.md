---
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
collection: talks
permalink: /talks/DPO
excerpt: '在⼈⼯智能领域，⽆监督语⾔模型（Language Models, LMs）的发展已经达到了令⼈惊叹的⽔平，这些模型能够在⼴泛的数据集上进⾏预训练，学习到丰富的世界知识和⼀定的推理能⼒。然⽽，如何精确控制这些模型的⾏为， 使其按照⼈类的偏好和⽬标⾏动，⼀直是⼀个难题。这主要是因为这些模型的训练完全是⽆监督的，它们从⼈类⽣成的数据中学习，⽽这些数据背后的⽬标、优先级和技能⽔平五花⼋⻔。例如，我们希望⼈⼯智能编程助⼿能够理解常⻅的编程错误以便纠正它们，但在⽣成代码时，我们⼜希望模型能偏向于它训练数据中的⾼质量编码能⼒，即使这种能⼒可能相对罕⻅。'
date: 2025-01-31
---


> 在⼈⼯智能领域，⽆监督语⾔模型（Language Models, LMs）的发展已经达到了令⼈惊叹的⽔平，这些模型能够在⼴泛的数据集上进⾏预训练，学习到丰富的世界知识和⼀定的推理能⼒。然⽽，如何精确控制这些模型的⾏为， 使其按照⼈类的偏好和⽬标⾏动，⼀直是⼀个难题。这主要是因为这些模型的训练完全是⽆监督的，它们从⼈类⽣成的数据中学习，⽽这些数据背后的⽬标、优先级和技能⽔平五花⼋⻔。例如，我们希望⼈⼯智能编程助⼿能够理解常⻅的编程错误以便纠正它们，但在⽣成代码时，我们⼜希望模型能偏向于它训练数据中的⾼质量编码能⼒，即使这种能⼒可能相对罕⻅。
<br/>

## Introduction：探索⽆监督语⾔模型的可控性挑战

现有的⽅法通常通过强化学习（Reinforcement Learning, RL）来引导LMs符合⼈类的偏好，这些⽅法需要收集关于模型⽣成内容相对质量的⼈类标签，并通过微调（fine-tuning）来使⽆监督LM与这些偏好对⻬。然⽽，强化学习 从⼈类反馈（RLHF）的过程复杂且通常不稳定，它⾸先需要拟合⼀个反映⼈类偏好的奖励模型，然后使⽤强化学习来微调⼤型⽆监督LM，以最⼤化这个估计的奖励，同时⼜不能偏离原始模型太远。

本⽂介绍了⼀种新的奖励模型参数化⽅法，该⽅法在RLHF中能够以闭合形式提取相应的最优策略，允许我们仅使⽤简单的分类损失来解决标准的RLHF问题。这⼀算法，我们称之为直接偏好优化（Direct Preference Optimizatio n, DPO），稳定、⾼效且计算成本低，⽆需在微调期间从LM中采样或进⾏⼤量超参数调整。我们的实验表明，DPO能够与现有⽅法⼀样好或更好地微调LMs以符合⼈类偏好。值得注意的是，使⽤DPO进⾏微调在控制⽣成情感⽅⾯超过了基于PPO的RLHF，并在摘要和单轮对话响应质量上匹敌或改进，同时实现⽅式⼤⼤简化，训练更加直接。

<img src='/images/dpo_1.png'>

## 1.直接偏好优化(DPO)简介

### 1.1 DPO与传统RLHF⽅法的对⽐

直接偏好优化（Direct Preference Optimization，简称DPO）是⼀种新型的算法，它与传统的基于⼈类反馈的强化学习（Reinforcement Learning from Human Feedback，简称RLHF）⽅法相⽐，具有显著的不同。RLHF⽅法通常通过收集⼈类对模型⽣成内容的相对质量标签，然后对未监督的语⾔模型（LM）进⾏微调，使其与这些偏好对⻬。这⼀过程涉及到先拟合⼀个反映⼈类偏好的奖励模型，然后使⽤强化学习对⼤型未监督LM进⾏微调，以最⼤化这⼀估计的奖励，同时不过分偏离原始模型。然⽽，RLHF是⼀个复杂且通常不稳定的过程，涉及训练多个LM， 并在训练循环中从LM策略中采样，带来显著的计算成本。

与之相对，DPO通过⼀个简单的分类损失直接优化语⾔模型以符合⼈类偏好，⽆需显式的奖励建模或强化学习。DPO的更新增加了优先响应相对于⾮优先响应的相对对数概率，但它包含了⼀个动态的、每个示例的重要性权重， 防⽌了模型退化，这是使⽤简单概率⽐⽬标时发现的问题。DPO利⽤理论偏好模型（如Bradley-Terry模型）来测量给定奖励函数与经验偏好数据的⼀致性，但与现有⽅法使⽤偏好模型来定义奖励模型的训练偏好损失不同，DPO通过变量变换直接将偏好损失定义为策略函数。因此，DPO可以使用一个简单的二元交叉熵目标来优化策略，产生一个隐含的奖励函数的最优策略。


## 2. DPO的⼯作原理与优势

DPO的⼯作原理基于将奖励函数转换为最优策略的分析映射，这使得我们能够将奖励函数上的损失函数转换为策略上的损失函数。这种变量变换⽅法避免了拟合⼀个显式的、独⽴的奖励模型，同时仍然在现有的⼈类偏好模型（如Bradley-Terry模型）下进⾏优化。本质上，策略⽹络同时代表了语 ⾔模型和（隐含的）奖励。

DPO的主要贡献是提供了⼀种简单的、⽆需强化学习的算法，⽤于根据偏好训练语⾔模型。实验表明，DPO⾄少与现有⽅法⼀样有效，包括基于PPO的RLHF，在诸如情感调节、摘要和对话等任务中从偏好中学习，使⽤多达6B参数的语⾔模型。

### 2.1 实验设计：评估DPO在不同⽂本⽣成任务中的表现

#### 2.1.1 实验任务介绍：情感⽣成、摘要和单轮对话

实验任务涉及三种不同的开放式⽂本⽣成任务。在控制情感⽣成中，x是IMDb数据集中电影评论的前缀，策略必须⽣成具有积极情感的y。在摘要任务中，x是Reddit论坛帖⼦，策略必须⽣成帖⼦主要观点的摘要y。最后，在单轮对话中，x是⼈类查询，可能是关于天体物理学的问题或寻求恋爱关系的建议，策略必须产⽣⼀个有趣且有帮助的响应y。

#### 2.1.2 实验评估⽅法：GPT-4胜率评估与⼈类判断验证

实验使⽤两种不同的评估⽅法。为了分析每种算法在优化受限奖励最⼤化⽬标的有效性，在控制情 感⽣成设置中，通过计算每种算法与参考策略的KL散度，评估每种算法的前沿。然⽽，在现实世界中，真实的奖励函数是未知的；因此，使⽤GPT-4作为摘要质量和单轮对话响应有⽤性的代理，评估算法的胜率。对于摘要，我们使⽤测试集中的参考摘要作为基线；对于对话，我们使⽤测试数据集中的⾸选响应作为基线。尽管现有研究表明LM可以是⽐现有指标更好的⾃动评估者，但我们进⾏了⼈类研究来证明我们使⽤GPT-4进⾏评估的合理性。我们发现GPT-4的判断与⼈类⾼度相关，⼈类与 GPT-4的⼀致性通常与⼈与⼈之间的注释者⼀致性相似或更⾼。

## 3.DPO 算法流程以及推导

DPO（Direct Preference Optimization）这个算法，发现他用了一种很巧妙的思路，将 RLHF 的 2 阶段多个模型的训练简化为了 1 阶段的 SFT 训练。介绍 DPO 做了哪些简化之前，首先要提一下我们一般认为的 RLHF 是咋训练的。RLHF 一般会分 2 步:

### 3.1 First Step

训练 reward model。训练数据是同一个 prompt 的 2 个回答，让人或 GPT4 标注哪个回答更好，reward model 会去优化如下的 loss：

<img src='/images/dpo_2.png'>

其中 rϕ 就是 reward model 用来给回答打分。D 是训练数据集，x 是 prompt，ywin 和 ylose 分别是好的回答和不好的回答。也就是说，要尽可能让好的回答的得分比不好的回答高，拉大他们之间的差别。

### 3.2 Second Step

RL 算法来提升模型的得分。使用的 loss 是

<img src='/images/dpo_3.png'>

其中 πθ 是我们在训练的 LLM，π ref 是训练的初始值。这个 loss 意思是希望 LLM 输出的回答的评分能尽可能高，同时 πθ 不要偏离 π ref 太多，保证它还能正常做回答，不要训成一个评分很高但是回答乱码的东西。

DPO 的作者们意识到，后面的这个式子是有显式解的。因为

<img src='/images/dpo_4.png'>

如果我们归一化一下分母，即取

<img src='/images/dpo_5.png'>

也就可以构造出一个新的概率分布：

<img src='/images/dpo_6.png'>

那么上式变成了：

<img src='/images/dpo_7.png'>

由于 KL 散度在 2 个分布相等时取最小值，我们得到了这样的结论：RLHF 训练希望得到的最优的概率分布就是 π∗。

另一个角度来说，由 π∗ 的公式，我们相当于是得到了 rϕ 和 π∗ 的关系，那么是否我们可以把训练 rϕ 转化成直接去训练 π∗ 呢？

简单转换一下 π∗ 的定义式，可以得到：

<img src='/images/dpo_8.png'>

带入最上面优化 rϕ 的 loss，也就有了：

<img src='/images/dpo_9.png'>

或者说，我们可以直接用这个 loss 去求 πθ：

<img src='/images/dpo_10.png'>

这就是 DPO 的 loss。DPO 通过以上的公式转换把 RLHF 无损地转化为了 SFT，在训练的时候不再需要同时跑 4 个模型（reward model, reference model, critic model, actor model），而是只用跑 actor 和 reference 2 个模型，甚至由于不再在线采数据，reference model 的输出可以预先存下来，训练的时候重复使用。

## 4.实验结果：DPO在多个任务中的性能表现
### 4.1 DPO与PPO在情感⽣成任务中的对⽐

在情感⽣成任务中，Direct Preference Optimization (DPO) 与 Proximal Policy Optimization (PPO) 进⾏了对⽐。DPO 通过简单的分类损失直接优化模型以符合⼈类偏好，⽽⽆需显式的奖励建模或强化学习。实验结果表明，DPO 在控制⽣成情感的能⼒上超越了基于 PPO 的 RLHF ⽅法，并且在摘要和单轮对话任务中匹配或提⾼了响应质量，同时实现了更简单的实施和训练过程。

### 4.2 DPO在摘要和单轮对话任务中的胜率分析

在摘要任务中，DPO 通过与⼈类写的摘要进⾏⽐较，使⽤ GPT-4 作为评估器，展示了其性能。DPO 在温度为 0.0 时的胜率约为 61%，超过了 PPO 在其最佳采样温度 0.0 时的 57% 胜率。在单轮对话任务中，DPO 与 Anthropic Helpful and Harmless 对话数据集中的⼈类偏好响应进⾏了⽐较。DPO 是唯⼀⼀种在 Anthropic HH 数据集测试集中改进过的选择摘要的⽅法，并且与计算上要求⾼的最佳 128 基线相⽐，提供了类似或更好的性能。

## 5.理论分析：DPO的理论基础与潜在优势
### 5.1 语⾔模型作为隐式奖励模型的理论⽀持

DPO ⽅法能够绕过显式奖励拟合和执⾏ RL 学习策略，使⽤单⼀的最⼤似然⽬标。优化⽬标等同于在奖励参数化下的 Bradley-Terry 模型，其中奖励参数化为 r∗(x, y) = β log π∗(y|x) / π ref(y|x)，并且通 过变量变换优化参数模型 πθ，等效于奖励模型优化。这种重新参数化不会限制学习奖励模型的类别，并允许精确恢复最优策略。

### 5.2 DPO的优化⽬标与理论属性

DPO 的更新直观上增加了偏好完成 yw 的相对对数概率，并降低了不受偏好的完成 yl 的概率。重要的是，示例通过隐式奖励模型 ˆrθ 评估不受偏好完成的程度进⾏加权，这考虑了 KL 约束的强度。 DPO 的⼀般流程包括：

1) 对于每个提示 x，采样完成 y1, y2 ∼ πref(·|x)，并使⽤⼈类偏好构建离线偏好数据集 D；

2) 优化语⾔模型 πθ 以最⼩化给定 πref 和 D 以及所需的 β 的 LDPO。实验表明，DPO 在没有显著超参数调整的情况下，⾄少与现有⽅法⼀样有效，包括基于 PPO 的 RLHF，⽤于从偏好中学习任务，如情感调节、摘要和对话。

## 6.讨论与未来展望
### 6.1 DPO在偏好学习框架中的地位与影响

Direct Preference Optimization（DPO）作为⼀种新型的偏好学习⽅法，其在未来的发展中扮演着重要⻆⾊。DPO通过简单的分类损失直接优化语⾔模型以符合⼈类偏好，避免了传统的强化学习⽅法中对奖励模型的显式拟合。这种⽅法的提出，不仅简化了训练流程，还降低了计算成本，使得语⾔模型的训练更加⾼效和稳定。在实验中，DPO在情感调节、摘要⽣成和单轮对话等任务上展现了与现有⽅法相当或更优的性能，尤其是在控制⽣成⽂本的情感⽅⾯，DPO超越了基于PPO的RLHF⽅ 法，并在摘要和对话响应质量上达到或超越了现有⽅法，同时实现了更简单的实施和训练过程。

### 6.2 DPO的局限性与未来研究⽅向

尽管DPO在偏好学习中展现出显著的优势，但其仍存在⼀些局限性。⾸先，DPO如何在分布外的泛化能⼒上与显式奖励函数学习的策略相⽐尚不明确。初步结果表明，DPO策略在泛化⽅⾯与基于 PPO的模型相似，但需要更全⾯的研究来验证这⼀点。其次，DPO在没有灾难性遗忘的情况下对语⾔模型进⾏微调的能⼒还有待进⼀步探索。此外，DPO对于⼤规模模型的扩展性也是未来研究的⼀ 个激动⼈⼼的⽅向。在评估⽅⾯，GPT-4作为⾃动化系统的评估者的有效性也是未来研究的⼀个重要问题。最后，DPO除了在训练语⾔模型⽅⾯的应⽤外，还有许多潜在的应⽤领域，包括在其他模态中训练⽣成模型。

## 7. 总结：DPO作为⼀种新型训练语⾔模型的⽅法

DPO通过直接优化语⾔模型以符合⼈类偏好，提供了⼀种⽆需强化学习的训练范式。DPO识别出语⾔模型策略与奖励函数之间的映射，使得可以直接使⽤简单的交叉熵损失来训练语⾔模型，⽽不牺牲⼀般性。DPO的性能与现有基于PPO的RLHF算法相当或更优，且⼏乎不需要调整超参数，从⽽显 著降低了从⼈类偏好中训练更多语⾔模型的障碍。