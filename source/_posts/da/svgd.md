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

对于机器学习领域的新手来说，理解SVGD算法充满了困难，这篇小短文希望能帮助更多的人学习SVGD。一般来说，初学者学习SVGD接触的论文应该是`Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm`，这篇论文第一次提出了SVGD算法，然而这篇论文在写作流程方面，至少我是这么觉得的，实在是很令人费解。本人并不认为这篇小短文能完全讲清楚SVGD算法的细节，不过尝试给出SVGD的基本原理和思路，也是为了记录自己学习SVGD算法的结果。

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

- 公式(3)的目的是求变换函数的方向，在步长$\epsilon$足够小的条件下，变换后的分布$q_{[\mathbf{T}]}$尽可能接近目标分布$p$。如果要类比的话，相当于在求解“梯度下降”。
- 公式(4)将公式(3)的优化目标进一步推导，使其关联上`Stein operator`，这是为了借用关于`Stein operator`的一些结论，相关部分可以参考论文`A Kernelized Stein Discrepancy for Goodness-of-fit Tests`，我们在下一节再讨论。

接下来我们分析公式(4)的结论是怎么来的。

### 证明 Theorem 3.1
首先我们需要再次明确各个符号的含义：

- $q$表示在当前迭代周期下，被优化，或者说被变换的分布。VI类方法的思路都是在一次次的迭代过程中不断将$q$变换为目标分布$p$。
- $q_{[\mathbf{T}]}$表示在当前迭代周期下，$q$经过$\mathbf{T}$变换之后的分布。
- 也就是说在下一次迭代中，$q_{[\mathbf{T}]}$就是新的$q$，只不过当前我们讨论范围仅局限于一次迭代中，因此没有加上迭代次数的标识。
- 为了便于证明，我们记$p_{\left[\mathbf{T}^{-1}\right]}$为目标分布$p$被函数$\mathbf{T}$“反向变换”一次后的分布。
- 也就是说如果从迭代的时间线上来看，理想情况下假设$q$最终会演化到目标分布$p$，则会有如下关系: 

$$q \to q_{[\mathbf{T}]} \to ... \to p_{\left[\mathbf{T}^{-1}\right]} \to p$$

然后我们需要了解两个关于矩阵行列式的引理，这里我直接给出公式，详情可查看参考链接：
$$\text{det}(\mathbf{A}) = \frac{1}{\text{det}(\mathbf{A}^{-1})},$$
其中$\mathbf{A}$表示一个矩阵，其逆存在且表示为$\mathbf{A}^{-1}$。

> [参考链接](https://math.stackexchange.com/questions/1455761/how-is-the-determinant-related-to-the-inverse-of-matrix)

$$\frac{\mathrm{d}}{\mathrm{d}t}\left[\text{det}\mathbf{A}(t)\right] = \text{det}\mathbf{A}(t) \cdot \text{trace}{\left[\mathbf{A}^{-1}(t)\cdot\frac{\mathrm{d}}{\mathrm{d}t}\mathbf{A}(t)\right]},$$
其中$\mathbf{A}(t)$表示一个变量为$t$的矩阵，$\text{trace}$表示矩阵的迹。

> [参考链接](https://mathoverflow.net/questions/214908/proof-for-the-derivative-of-the-determinant-of-a-matrix)

关于分布$q$、$q_{[\mathbf{T}]}$、$p_{\left[\mathbf{T}^{-1}\right]}$、$p$，有如下关系：

$$
\begin{align*}{\tag{6}}
q_{[\mathbf{T}]}(\mathbf{x}) &= q\left(\mathbf{T}^{-1}(\mathbf{x})\right)\left|\text{det}{\left(\nabla_{\mathbf{x}}\mathbf{T}^{-1}(\mathbf{x})\right)} \right| \\
p(\mathbf{x}) &= p_{\left[\mathbf{T}^{-1}\right]}\left(\mathbf{T}^{-1}(\mathbf{x})\right)\left|\text{det}{\left(\nabla_{\mathbf{x}}\mathbf{T}^{-1}(\mathbf{x})\right)} \right|
\end{align*}
$$

> 这里涉及到了概率分布的变量替换，相关技巧可以参考[此处](https://www.cs.ubc.ca/~murphyk/Teaching/Stat406-Spring08/homework/changeOfVariablesHandout.pdf)。

基于公式(6)的分布变换关系，将两个式子的等号左边和右边分别相除，公式(4)的等号左边的原问题可以进一步写成：
$$
\nabla_{\epsilon}\left.\text{KL}(q_{[\mathbf{T}]} \| p)\right|_{\epsilon = 0} = \nabla_{\epsilon}\left.\text{KL}(q \| p_{\left[\mathbf{T}^{-1}\right]})\right|_{\epsilon = 0} = -\left. \mathbb{E}_{\mathbf{x}\sim q}\left[\nabla_{\epsilon}\log{p_{\left[\mathbf{T}^{-1}\right]}(\mathbf{x})}\right]\right|_{\epsilon = 0}. \tag{7}
$$

再次利用概率分布的变量替换技巧，对分布$p_{\left[\mathbf{T}^{-1}\right]}(\mathbf{x})$有如下结论：
$$
p_{\left[\mathbf{T}^{-1}\right]}(\mathbf{x}) = p(\mathbf{T}(\mathbf{x}))\left|\text{det}{\left(\nabla_{\mathbf{x}}\mathbf{T}(\mathbf{x})\right)} \right|. \tag{8}
$$

结合公式(8)，可对公式(7)中的项$\nabla_{\epsilon}\log{p_{\left[\mathbf{T}^{-1}\right]}(\mathbf{x})}$做如下推理：
$$
\begin{align*}{\tag{9}}
\nabla_{\epsilon}\log{p_{\left[\mathbf{T}^{-1}\right]}(\mathbf{x})} &= \nabla_{\epsilon}\log{p(\mathbf{T}(\mathbf{x}))\left|\text{det}{\left(\nabla_{\mathbf{x}}\mathbf{T}(\mathbf{x})\right)} \right|} \\
&= \nabla_{\epsilon} \left(\log{p(\mathbf{T}(\mathbf{x}))} + \log{\left|\text{det}{\left(\nabla_{\mathbf{x}}\mathbf{T}(\mathbf{x})\right)} \right|}\right) \\
&= \nabla_{\mathbf{x}}\log{p(\mathbf{x})}^\top \cdot \nabla_{\epsilon}\mathbf{T}(\mathbf{x}) + \nabla_{\epsilon}\log{\left|\text{det}{\left(\nabla_{\mathbf{x}}\mathbf{T}(\mathbf{x})\right)}\right|} \\
&= \nabla_{\mathbf{x}}\log{p(\mathbf{x})}^\top \cdot \nabla_{\epsilon}\mathbf{T}(\mathbf{x}) + \frac{\nabla_{\epsilon}\left|\text{det}{\left(\nabla_{\mathbf{x}}\mathbf{T}(\mathbf{x})\right)}\right|}{\left|\text{det}{\left(\nabla_{\mathbf{x}}\mathbf{T}(\mathbf{x})\right)}\right|} \\
&= \nabla_{\mathbf{x}}\log{p(\mathbf{x})}^\top \cdot \nabla_{\epsilon}\mathbf{T}(\mathbf{x}) + \text{trace}\left( \left(\nabla_{\mathbf{x}}\mathbf{T}(\mathbf{x})\right)^{-1} \cdot \nabla_{\epsilon}\nabla_{\mathbf{x}}\mathbf{T}(\mathbf{x}) \right).
\end{align*}
$$

这里最后一个等式的推理用到了矩阵行列式引理。另外结合公式(2)对变换函数$\mathbf{T}$的定义：
$$\mathbf{T}(\mathbf{x}) = \mathbf{x} + \epsilon \boldsymbol{\phi(\mathbf{x})}.$$

可以推理得出结论：
$$
\begin{align*}\tag{10}
& \nabla_{\epsilon}\left.\text{KL}(q_{[\mathbf{T}]} \| p)\right|_{\epsilon = 0}\\
=& -\left. \mathbb{E}_{\mathbf{x}\sim q}\left[\nabla_{\epsilon}\log{p_{\left[\mathbf{T}^{-1}\right]}(\mathbf{x})}\right]\right|_{\epsilon = 0} \\
=& -\left. \mathbb{E}_{\mathbf{x}\sim q}\left[ \nabla_{\mathbf{x}}\log{p(\mathbf{x})}^\top \cdot \nabla_{\epsilon}\mathbf{T}(\mathbf{x}) + \text{trace}\left( \left(\nabla_{\mathbf{x}}\mathbf{T}(\mathbf{x})\right)^{-1} \cdot \nabla_{\epsilon}\nabla_{\mathbf{x}}\mathbf{T}(\mathbf{x}) \right) \right]\right|_{\epsilon = 0} \\
=& - \mathbb{E}_{\mathbf{x}\sim q}\left[ \nabla_{\mathbf{x}}\log{p(\mathbf{x})}^\top \cdot \boldsymbol{\phi(\mathbf{x})}  + \text{trace}\left( \nabla_{\mathbf{x}}\boldsymbol{\phi(\mathbf{x})} \right) \right] \\
=& - \mathbb{E}_{\mathbf{x}\sim q}\left[ \text{trace}\left( \nabla_{\mathbf{x}}\log{p(\mathbf{x})}\boldsymbol{\phi(\mathbf{x})}^\top + \nabla_{\mathbf{x}}\boldsymbol{\phi(\mathbf{x})} \right)\right].
\end{align*}
$$
公式(4)证明完毕。

### Theorem 3.1的意义

1. SVGD方法假设存在一个变换函数$\mathbf{T}$（或表达为$\mathbf{T}_{q,p}$），该函数被迭代地作用于一个演化分布 $q$，使得每一次经过变换后的分布$q_{[\mathbf{T}]}$能够更接近目标分布$p$。
2. 该变换函数$\mathbf{T}$的具体形式被定义为：$\mathbf{T}(\mathbf{x}) = \mathbf{x} + \epsilon \boldsymbol{\phi(\mathbf{x})}$，其中$\boldsymbol{\phi(\mathbf{x})}$代表了变换的方向函数。
3. 在上述假设下，最优$\boldsymbol{\phi(\mathbf{x})}$是使得$q_{[\mathbf{T}]}$与$p$之间的KL散度下降最快：$\arg\min_{\boldsymbol{\phi}}{\nabla_{\epsilon}\left.\text{KL}(q_{[\mathbf{T}]} \| p)\right|_{\epsilon = 0}}.$
4. 经过数学推导，原问题可进一步表达为：
$$
\nabla_{\epsilon}\left.\text{KL}(q_{[\mathbf{T}]} \| p)\right|_{\epsilon = 0} = - \mathbb{E}_{\mathbf{x}\sim q}\left[ \text{trace}\left( \nabla_{\mathbf{x}}\log{p(\mathbf{x})}\boldsymbol{\phi(\mathbf{x})}^\top + \nabla_{\mathbf{x}}\boldsymbol{\phi(\mathbf{x})} \right)\right].
$$
5. 也就是说，现在需要求解:
$$
\arg\max_{\boldsymbol{\phi}} \mathbb{E}_{\mathbf{x}\sim q}\left[ \text{trace}\left( \nabla_{\mathbf{x}}\log{p(\mathbf{x})}\boldsymbol{\phi(\mathbf{x})}^\top + \nabla_{\mathbf{x}}\boldsymbol{\phi(\mathbf{x})} \right)\right].\tag{11}
$$

### Lemma 3.2
在SVGD论文中，引理3.2的含义是：如果$\boldsymbol{\phi}$限定为再生核希尔伯特空间（RKHS）的单位球内，那么问题(11)的解为：
$$
\boldsymbol{\phi}^*_{q,p}(\mathbf{x}) = \mathbb{E}_{\mathbf{y}\sim q}\left[ k(\mathbf{y}, \mathbf{x})\nabla_{\mathbf{y}}\log{p(\mathbf{y})} + \nabla_{\mathbf{y}}k(\mathbf{y}, \mathbf{x}) \right], \tag{12}
$$
其中$k$为RKHS的再生核函数，经常使用的是高斯核函数。

### 如何理解Lemma 3.2

> 该部分内容需要了解论文：A Kernelized Stein Discrepancy for Goodness-of-fit Tests

##### Stein class and Stein's operator
概率分布$p$关于函数$f:\mathbb{R}\to\mathbb{R}$的Stein's operator被定义为：
$$\mathcal{A}_{p}f(x) = \nabla_x \log{p(x)}f(x) + \nabla_x f(x), \tag{13}$$

对于向量函数$\boldsymbol{f}:\mathbb{R}^d\to\mathbb{R}^d$，其计算结果为一个矩阵：
$$\mathcal{A}_{p}\boldsymbol{f}(\mathbf{x}) = \nabla_{\mathbf{x}} \log{p(\mathbf{x})}\boldsymbol{f}(\mathbf{x})^\top + \nabla_{\mathbf{x}} \boldsymbol{f}(\mathbf{x}). \tag{14}$$

如果函数$f$是光滑的且与概率分布$p$满足：
$$\int{\nabla_x \left( f(x)p(x) \right)\mathrm{d}x} = 0, \tag{15} $$
则称$f$属于概率分布$p$的Stein class（对于向量函数$\boldsymbol{f}:\mathbb{R}^d\to\mathbb{R}^d$有类似结论，等式右边就是一个全零矩阵，如果变量是向量$\mathbf{x}$，那么等式右边就是一个全零向量）。

公式(15)的描述的条件是比较好满足的，if function $f$ is smooth with proper zero-boundary:
$$
\nabla_x C = \nabla_x \int{f(x)p(x)\mathrm{d}x} = \int{\nabla_x \left( f(x)p(x) \right)\mathrm{d}x} = 0,
$$
其中$C$表示一个常数。

##### 方向函数的范数
不严谨地说，公式(11)描述的问题其实是某种范数，接下来我们将通过推导展现这一点。

假设方向函数$\boldsymbol{\phi}$ is smooth with proper zero-boundary，那么基于前面关于 stein class 的讨论，可以得到：
$$
\begin{align*}{\tag{16}}
& \int{\nabla_{\mathbf{x}}\left( \boldsymbol{\phi}(\mathbf{x}) q(\mathbf{x}) \right) \mathrm{d}\mathbf{x}}\\
=& \int{ \left(\nabla_{\mathbf{x}}\boldsymbol{\phi}(\mathbf{x})q(\mathbf{x}) + \boldsymbol{\phi}(\mathbf{x})\nabla_{\mathbf{x}}q(\mathbf{x})\right) \mathrm{d}\mathbf{x}} \\
=& \int{ \left( \nabla_{\mathbf{x}}\log{q(\mathbf{x})}^\top\boldsymbol{\phi}(\mathbf{x}) + \nabla_{\mathbf{x}}\boldsymbol{\phi}(\mathbf{x})  \right)\mathrm{d}q(\mathbf{x}) }\\
=& \mathbb{E}_{\mathbf{x} \sim q}\left[ \nabla_{\mathbf{x}}\log{q(\mathbf{x})}^\top\boldsymbol{\phi}(\mathbf{x}) + \nabla_{\mathbf{x}}\boldsymbol{\phi}(\mathbf{x}) \right]\\
=& \mathbf{O},
\end{align*}
$$
这里用$\mathbf{O}$表示全零矩阵。

再次回顾公式(11)定义的问题，可以经过推理得到：
$$
\begin{align*}{\tag{17}}
& \arg\max_{\boldsymbol{\phi}} \mathbb{E}_{\mathbf{x}\sim q}\left[ \text{trace}\left( \nabla_{\mathbf{x}}\log{p(\mathbf{x})}\boldsymbol{\phi(\mathbf{x})}^\top + \nabla_{\mathbf{x}}\boldsymbol{\phi(\mathbf{x})} \right)\right] \\
=& \arg\max_{\boldsymbol{\phi}} \mathbb{E}_{\mathbf{x}\sim q}\left[ \text{trace}\left( \nabla_{\mathbf{x}}\log{p(\mathbf{x})}\boldsymbol{\phi(\mathbf{x})}^\top + \nabla_{\mathbf{x}}\boldsymbol{\phi(\mathbf{x})} \right)\right] - \\ 
& \mathbb{E}_{\mathbf{x}\sim q}\left[ \text{trace}\left( \nabla_{\mathbf{x}}\log{q(\mathbf{x})}\boldsymbol{\phi(\mathbf{x})}^\top + \nabla_{\mathbf{x}}\boldsymbol{\phi(\mathbf{x})} \right)\right] \\
=& \arg\max_{\boldsymbol{\phi}} \mathbb{E}_{\mathbf{x}\sim q}\left[ \text{trace}\left( \nabla_{\mathbf{x}}\log{p(\mathbf{x})}\boldsymbol{\phi(\mathbf{x})}^\top - \nabla_{\mathbf{x}}\log{q(\mathbf{x})}\boldsymbol{\phi(\mathbf{x})}^\top \right)\right] \\
=& \arg\max_{\boldsymbol{\phi}} \mathbb{E}_{\mathbf{x}\sim q}\left[ \text{trace}\left( \left(\nabla_{\mathbf{x}}\log{p(\mathbf{x})} - \nabla_{\mathbf{x}}\log{q(\mathbf{x})}\right)\boldsymbol{\phi(\mathbf{x})}^\top \right)\right],
\end{align*}
$$
推导到这里其实已经很明显了，这就是两个函数关于分布$q$的内积，如果先不考虑RKHS，此时公式(11)定义的问题的解为：
$$ \boldsymbol{\phi}^*_{q,p}(\mathbf{x}) = \nabla_{\mathbf{x}}\log{p(\mathbf{x})} - \nabla_{\mathbf{x}}\log{q(\mathbf{x})}. \tag{18} $$

> Q: 既然已经得到了方向函数的解，那为什么还要有RKHS限制?
> A: 这是因为公式(18)的解涉及到了对演化分布的概率密度函数求梯度，而SVGD方法使用粒子群模拟分布$q$，因此没法求梯度。

> Q: 如果不使用粒子群，而是直接参数化分布$q$会怎样？
> A: 那么公式(17)的结论就是ELBO，SVGD方法坍缩为基于模型的VI方法。

需要注意到，SVGD方法得到的方向函数，即公式(12)，是不用计算$q$的梯度的，相反计算的是核函数$k$的梯度。

接下来我们再看SVGD方法求解的方向函数，即公式(12)的含义是什么（为了省点事我去掉了下标和上标）。我们还是利用 stein class 的性质，可以得到如下结论：
$$
\begin{align*}{\tag{19}}
\boldsymbol{\phi}(\mathbf{x}) =& \mathbb{E}_{\mathbf{y}\sim q}\left[ k(\mathbf{y}, \mathbf{x})\nabla_{\mathbf{y}}\log{p(\mathbf{y})} + \nabla_{\mathbf{y}}k(\mathbf{y}, \mathbf{x}) \right] \\
=& \mathbb{E}_{\mathbf{y}\sim q}\left[ k(\mathbf{y}, \mathbf{x}) \left(\nabla_{\mathbf{y}}\log{p(\mathbf{y})} - \nabla_{\mathbf{y}}\log{q(\mathbf{y})} \right)\right].
\end{align*}
$$

将公式(19)带入公式(17)的结论，可以得到：
$$
\begin{align*}{\tag{20}}
\mathbb{E}_{\mathbf{x}\sim q, \mathbf{y}\sim q}\left[ \left(\nabla_{\mathbf{x}}\log{p(\mathbf{x})} - \nabla_{\mathbf{x}}\log{q(\mathbf{x})}\right)^\top k(\mathbf{y}, \mathbf{x})\left(\nabla_{\mathbf{x}}\log{p(\mathbf{x})} - \nabla_{\mathbf{x}}\log{q(\mathbf{x})}\right) \right].
\end{align*}
$$

公式(20)是RKHS空间中一个函数的 norm，这个函数的形式是：
$$f(\cdot) = \int{\left(\nabla_{\mathbf{z}}\log{p(\mathbf{z})} - \nabla_{\mathbf{z}}\log{q(\mathbf{z})}\right)k(\mathbf{z}, \cdot)\mathrm{d}\mathbf{z}}, \tag{21}$$
这里我用了$\mathbf{z}$符号，只是为了区分$\mathbf{x}$和$\mathbf{y}$，避免误解。公式(21)表达了用核函数$k$将方向函数(18)映射到RKHS空间的过程，或者说，公式(21)是公式(18)的smooth版本，使用分部积分法，可以从公式(21)推导出SVGD的解，即公式(12)。

完毕。

> 上面的这些推理我没能在硕士期间完全搞明白，相反我通常是从梯度流角度来解释SVGD的，不过现在毕业了，我仅仅花了半个晚上的时间就正面理解了SVGD，真是神奇:)