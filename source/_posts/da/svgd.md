title: 采样与生成（3）：SVGD
description: 记录 Stein Variational Gradient Descent 的一些公式推理过程。
date: 2023-10-3 22:13:00
tags:
- 机器学习
- 贝叶斯学习
- 变分推断
- SVGD
---

## 背景介绍
> Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm

SVGD算法的全称是Stein Variational Gradient Descent（斯坦因变分梯度下降算法？），该算法于2016年被提出，在当时被认为是同时改进了马尔科夫链蒙特卡洛（Markov Chain Monte Carlo, MCMC）和基于模型的变分推理（Model-based Variatinoal Inference, Mod-VI）。简单来说，SVGD被认为在收敛速度和粒子有效性方面优于MCMC，在精确度方面优于ModVI；另一方面，SVGD的劣势在于粒子坍缩（尤其是高维多模分布情况下，近似误差尤其大）、计算复杂度高以及可扩展性差。客观来说我认为SVGD在实际应用中的限制还是很大的，高维、多模、可扩展性、精确度这几方面劣于MCMC导致其很难是首选，毕竟MCMC可以轻易地并行采样出数十万甚至百万的样本点，而SVGD的规模通常不过万，更不必说MCMC家族中有HMC这种已经发展很成熟的算法，甚至还可以做到自动调参。

不过本人认为SVGD的价值在于开辟了新思路，从2017年的SVGD as gradient flow开始，经过2019年刘畅博士的几篇论文，SVGD逐渐发展为一类粒子变分（Particle-based Variational Inference, ParVI）方法的代表，这类方法从梯度流视角研究采样问题，甚至统一了动力学MCMC、ModVI和ParVI的基本原理，后逐步过渡到将采样问题与生成问题纳入到统一的求解框架。老实说我认为学习SVGD、MCMC、VI这些算法对于理解当今某些时髦的东西，比如扩散模型，是很有帮助的，从原理上来讲这些东西差距不大，都是一套变分推理的方法推导出来的。虽然已经这样说了SVGD算法的重要性，我还是需要提醒一下我并不认为SVGD有很高的、广泛的实用价值，尤其是现在的很多论文都是从优化领域拿了很多“现成的”组件，诸如降方差、受限域、非光滑、动量加速、黎曼流形等等，新瓶装旧酒重新刷一边。也不能说研究SVGD（或者说ParVI类算法）的改进是没有意义的，然而这些东西实在是太冷板凳了，而且从算法的“底色”来讲，我不觉得ParVI显著优于他的老前辈MCMC。

对于机器学习领域的新手来说，理解SVGD算法充满了困难，这篇小短文希望能帮助更多的人学习SVGD。一般来说，初学者学习SVGD接触的论文应该是`Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm`，这篇论文第一次提出了SVGD算法，然而这篇论文讨论的东西，至少我是这么觉得的，实在是很令人费解。本人并不认为这篇小短文能完全讲清楚SVGD算法的细节，不过尝试给出SVGD的基本原理和思路，也是为了记录自己学习SVGD算法的结果。

## 问题设定
SVGD一般被用来求解贝叶斯学习中的采样问题，这类问题讨论如何针对一个给定的复杂目标分布$p(\mathbf{x})$，高效地产生符合这个分布的样本点。目标分布$p(\mathbf{x})$通常有以下特点：

- 目标分布$p$可能是未归一化的，其积分不是1，存在一个未知的归一化常数。
- 目标分布$p$的维数可能较高，无法使用网格计算每个点处的概率密度数值。
- 目标分布$p$的概率密度函数的最大值和最小值是未知的。
- 目标分布$p$的形式非常复杂，可能会无法写出概率密度函数的数学式，但是可以求概率密度函数在某点的数值和对应的梯度。

> $p(\mathbf{x})$中我们使用加粗的小写字母表示向量，如无特别说明，后续符号均遵循此规则。这里我们用的符号相比论文做了微小改动，不过大体上是相似的，不相似的部分均会指出。

针对上述问题，MCMC算法的思路是构造一个可约、非周期、正常返的马尔科夫链，然后模拟该马尔可夫链的状态跳转，记录跳转过程中的位置点作为样本点；VI算法的思路是构造一个参数化的、简单可采样的分布，然后优化这个分布使其近似目标分布，最后从该优化后的分布中直接产生样本点；SVGD算法的思路介于MCMC和VI二者之间，**该算法初始化一个粒子群作为起点，然后迭代地优化粒子群中每个粒子的位置使整个粒子群逐渐近似目标分布**。

用公式来描述，VI和SVGD的优化目标是相同的：
$$
\min_{q\in\mathcal{Q}} \text{KL}(q \| p), \tag{1}
$$
其中$\mathcal{Q}$表示预定义的分布族，比如在VI中如果参数化了一个高斯分布的均值（方差固定为1），那么$\mathcal{Q}$表示所有方差为1的高斯分布。而在SVGD中，$\mathcal{Q}$表示所有**可由一个参考分布经平滑变换（smooth transform）而得到的所有分布**。VI和SVGD的目标均是求解上式定义的问题，找到一个在分布族$\mathcal{Q}$中最优的$q^*$。

## 求解思路
为了更好地理解SVGD方法，我们先讨论VI方法的基本思路。针对公式(1)的问题，VI通常会参数化一个简单分布，这里我们用高斯分布举例，假设参数化了一个方差为1的高斯分布的均值。VI的基本思路是将公式(1)进一步推导变换成一个可计算梯度的公式，然后用梯度下降优化这个高斯分布的均值，使其逐渐近似目标分布。在这个过程中，分布的变换过程实际上是高斯分布的均值参数的优化过程，而每一次更新高斯分布的均值，都是赋予了这个分布中所有的潜在样本点一个变换。

SVGD算法跟上述思路的差异在于，SVGD算法预定义了一个粒子群，或者说是一堆预设的样本点，这个粒子群的初始状态没什么讲究，一般都是高斯分布。然后SVGD算法会根据当前粒子群的状态给出一个变换函数$\mathbf{T}$（实际上应该表示为$\mathbf{T}_{q,p}$，表示该变换函数取决于当前粒子群的状态和目标分布，不过这里是为了和论文统一），对当前粒子群中的所有粒子点应用这个函数$\mathbf{T}$更新粒子的位置，迭代地做上述过程，最终粒子群的排布会近似目标分布。具体来说，该函数的形式是：
$$
\mathbf{T}(\mathbf{x}) = \mathbf{x} + \epsilon \boldsymbol{\phi(\mathbf{x})}, \tag{2}
$$
这里的$\mathbf{T}$表示一个$\mathbb{R}^d\to\mathbb{R}^d$的`smooth one-to-one transform`（当$\epsilon$足够小的时候），$\epsilon$是一个标量，表示变换的幅度。这里将分布$q$经过$\mathbf{T}$变换后的结果表示为$q_{[\mathbf{T}]}$。

> 现在的问题是，怎么求$\boldsymbol{\phi}$

## 公式推导
### Theorem 3.1
SVGD在`Theorem 3.1`中给出的思路是，求解如下目标：
$$
\boldsymbol{\phi} = \arg\min_{\boldsymbol{\phi}}{\nabla_{\epsilon}\left.\text{KL}(q_{[\mathbf{T}]} \| p)\right|_{\epsilon = 0}}, \tag{3}
$$
当然实际上还要加上RKHS限制，不过我们一点一点来，先分析`Theorem 3.1`讨论的目标。`Theorem 3.1`给出的结论是：
$$
\nabla_{\epsilon}\left.\text{KL}(q_{[\mathbf{T}]} \| p)\right|_{\epsilon = 0} = -\mathbb{E}_{\mathbf{x}\sim q}[\text{trace}(\mathcal{A}_p \boldsymbol{\phi(\mathbf{x})})], \tag{4}
$$
其中
$$
\mathcal{A}_{p}\boldsymbol{\phi(\mathbf{x})} = \nabla_{\mathbf{x}}\log{p(\mathbf{x})}\boldsymbol{\phi(\mathbf{x})}^\top + \nabla_{\mathbf{x}}\boldsymbol{\phi(\mathbf{x})}, \tag{5}
$$
这里的$\mathcal{A}_{p}$叫做`Stein operator`（翻译成斯坦因算子？）。

- 公式(3)的目的是求变换函数的方向，使得当步长$\epsilon$足够小的时候，变换后的分布$q_{[\mathbf{T}]}$尽可能接近目标分布$p$。
- 公式(4)将公式(3)的优化目标进一步推导，使其关联上`Stein operator`，这是为了借用关于`Stein operator`的一些结论，相关部分可以参考论文`A Kernelized Stein Discrepancy for Goodness-of-fit Tests`，我们在下一节再讨论。

接下来我们分析公式(4)的结论是怎么来的。