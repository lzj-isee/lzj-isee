title: 采样与生成（2）：基于模型的变分推断
description: 记录 Model-based Variational Inference 的一些公式推理过程。
tags:
- 机器学习
- 贝叶斯学习
- 变分推断
---

## 变分推断（Variational Inference）
在贝叶斯学习、变分自编码器、扩散模型等领域的文献中经常能看到诸如变分推断（Variational Inference）、ELBO（Evidence Lower BOund）等字眼。本文尝试从源头开始一步步推理，详细解释VI和ELBO的来源以及推理过程。

## 问题设定
变分推断常被用来求解贝叶斯学习中的采样问题，这类问题针对一个给定的复杂目标分布$\nu(x)$，讨论如何高效地产生符合这个分布的样本点。怎么理解这个问题呢？比如对于简单的高斯分布或者是迪利克雷分布，可以简单地调用编程语言的一个函数生成样本点，但是对于一个general的分布，假设我们只知道该分布的概率密度函数及其梯度，如何产生样本点？这个目标分布通常有以下特点：

* 分布$\nu(x)$可能是未归一化的，其积分不是1，存在一个未知的归一化常数。
* 分布$\nu(x)$的维数可能相对比较高，$10^1\sim 10^5$等等。
* 分布$\nu(x)$的最大值和最小值是未知的。
* 分布$\nu(x)$的形式非常复杂，无法写出概率密度函数的数学式，但是可以求概率密度函数和梯度（多层神经网络建模的分布通常都是这样的）。

变分推断的思路是，找一个简单分布（可以方便采样得到样本点的），使其尽可能接近目标分布。

抽象地总结，VI求解的问题可以被定义为：
$$
\min_{q} \text{KL}(q \| p), \tag{1}
$$
其中$q$是那个“简单”分布，这里为了和论文、网上的各种教程的符号统一，将目标分布定义为$p$。使用KL散度是“经典选择”，后续公式推导也会看到KL散度的结构对化简的重要性。

## 变分推断概述
通常情况下，简单分布$q$是参数化的，比如对于一个多元高斯分布来说，这个参数化就是它的均值和协方差矩阵，前者决定了这个高斯分布的“位置”，后者决定了这个高斯分布的“大小”。这里我们不讨论在共轭先验情况下变分推断问题的闭解（我也没研究过，现在很少看到这种特殊情况的解法了）。变分推断要做的事情是，优化$q$的参数（高斯分布情况下就是均值和协方差矩阵），使该高斯分布尽可能接近目标分布。

<center><img src="model_based_vi/modvi.png" width="95%" /></center>

这里我用论文里的截图展示这个过程，这是一个2维的变分推断问题，其中红色表示目标分布，灰色表示待优化的分布。可以看到随着优化过程的进行，优化分布逐渐接近目标分布。

> The equivalence between Stein variational gradient descent and black-box variational inference

## 变分推断计算
将待优化的分布记作$q_\lambda$，其中$\lambda$表示优化参数，也就是前述提到的高斯分布的均值和协方差矩阵，这里不要纠结参数数量以及参数的形状问题，比如我们可以把矩阵“拉直”成向量，将其和均值向量拼接在一起，用一个符号$\lambda$统一表示。在接下来的推导中，我会首先讨论$\nu(x)$是归一化的简单情况，然后过渡到非归一化的情况。常见的VI解法有黑盒变分推断（Black Box Variational Inference）和重参数化（Reparameterize）变分推断，下文都会涉及到。

### 重参数化变分推断

优化分布$q_\lambda(x)$和$\nu(x)$之间的KL散度：$\min_{\lambda} \text{KL}(q_\lambda \| \nu)$的推导过程为：
$$
\begin{align*}
& \min_\lambda \text{KL}(q_\lambda \| \nu) \\
\Longrightarrow & \min_\lambda \int{\log{\frac{q_\lambda}{\nu}}\mathrm{d}q_\lambda} \\ 
\Longrightarrow & \max_\lambda \mathbb{E}_{x\sim q_\lambda}[\log{\nu(x)} - \log{q_\lambda(x)}], \tag{2}
\end{align*}
$$
这里的公式(2)不能直接用类似梯度上升的方法优化，因为公式(2)中“采样”这个步骤，即$\mathbb{E}_{x\sim q_\lambda}$，是无法求梯度的（无法对$x$求关于$\lambda$的梯度）。一个替代方案是，对$x$做重参数化：
$$
x = f_\lambda(z),\quad z\sim \mathcal{N}(0, \mathbf{I}).
$$
这里我们依然假设$\lambda$代表了高斯分布的均值和方差，$x$表示这个高斯分布的样本点，$z$表示标准高斯分布的样本点，函数$f_\lambda$将样本点$z$转化为$x$。具体来说，$f_\lambda$可以有以下形式：
$$
f_\lambda(z) = \mu + \epsilon z,
$$
这里的$\mu$通常是一个向量，表示分布$q_\lambda$的均值，$\epsilon$可以是一个标量（最简单的高斯分布）、向量（只有对角线非零的高斯分布）、三角阵（一般多元高斯分布）。也就是说，这里把优化变量$\lambda$具象化为多元高斯分布的均值和方差。注意到，此时可以求解$x = f_\lambda(z)$关于$\lambda$的梯度，因此，公式(2)的优化目标可进一步写为：
$$
\max_{\lambda} \mathbb{E}_{z\sim \mathcal{N}}\left[ \log{\nu\left(f_\lambda(z)\right)} - \log{q_\lambda\left(f_\lambda(z)\right)} \right]. \tag{3}
$$
这里的公式(3)可以直接使用梯度上升法求解（若目标分布的概率密度$\nu$可求梯度）。


### 黑盒变分推断

上述基于重参数化技巧的变分推断求解算法在目标分布的概率密度函数无法计算梯度时失效，例如目标分布是一个离散分布，此时概率密度函数不存在。也就是说，算法的求解过程不能出现$\nabla_x \log{\nu(x)}$。在这种情况下，可以使用黑盒变分推断（Black Box Variational Inference）算法。

首先明确变分推断的优化目标是：
$$
\min_\lambda \text{KL}(q_\lambda \| \nu),
$$

已知这个公式可以整理为：
$$
\max_\lambda \mathbb{E}_{x\sim q_\lambda}[\log{\nu(x)} - \log{q_\lambda(x)}],
$$

前文提到之所以不能对上式直接使用反向传播计算梯度是因为采样过程$\mathbb{E}_{x\sim q_\lambda}$不可导，黑盒变分推断的核心是将上式重新整理为可以求导的形式，且计算过程不会出现目标分布$\nu$的梯度。

考虑对优化目标计算关于$\lambda$的梯度：
$$
\begin{align*}
& \nabla_\lambda \int{\log{\frac{q_\lambda}{\nu}}\mathrm{d}q_\lambda} \\
\Longrightarrow & \int{\nabla_\lambda \left(q_\lambda \cdot \log{\frac{q_\lambda}{\nu}}\right)\mathrm{d}x} \\
\Longrightarrow & \int{\nabla_\lambda \left(q_\lambda \cdot \log{q_\lambda}\right)\mathrm{d}x} - \int{\nabla_\lambda \left(q_\lambda \cdot \log{\nu}\right)\mathrm{d}x}\\
\Longrightarrow & \int{\left(\nabla_\lambda q_\lambda\cdot\log{q_\lambda} + q_\lambda\cdot\nabla_\lambda\log{q_\lambda}\right)\mathrm{d}x} - \int{\nabla_\lambda q_\lambda\cdot\log{\nu}\mathrm{d}x} \\
\Longrightarrow & \int{\frac{\nabla_\lambda q_\lambda}{q_\lambda}\log{q_\lambda}\mathrm{d}q_\lambda} + \int{\nabla_\lambda q_\lambda \mathrm{d}x} - \int{\log{\nu} \frac{\nabla_\lambda q_\lambda}{q_\lambda}\mathrm{d}q_\lambda}\\
\Longrightarrow & \int{\nabla_\lambda \log{q_\lambda}\cdot\log{q_\lambda} \mathrm{d}q_\lambda}  + \nabla_\lambda\int{q_\lambda\mathrm{d}x} - \int{\nabla_\lambda\log{q_\lambda}\cdot\log{\nu}\mathrm{d}q_\lambda}\\
\Longrightarrow & \int{\nabla_\lambda\log{q_\lambda}\left(\log{q_\lambda} - \log{\nu}\right)\mathrm{d}q_\lambda} + \nabla_\lambda 1 \\
\Longrightarrow & \mathbb{E}_{x\sim q_\lambda}\left[\nabla_\lambda\log{q_\lambda}\left(\log{q_\lambda} - \log{\nu}\right)\right] + 0\\
\Longrightarrow & \mathbb{E}_{x\sim q_\lambda}\left[\nabla_\lambda\log{q_\lambda}\left(\log{q_\lambda} - \log{\nu}\right)\right]
\end{align*}
$$
上式即为优化的迭代梯度，可以看到，梯度的计算不涉及对目标分布$\nu$求梯度。

> 需要注意的是，根据我个人的经验来说，黑盒变分推理的梯度的噪声远高于重参数化方法（就是说重参数化方法收敛快）。

### 未归一化与ELBO

前述的变分推断方法皆是基于目标分布$\nu$是已归一化的状态，但是在某些应用场景中（例如贝叶斯推理），目标分布是未归一化的（积分不是1，存在一个未知的归一化常数）。接下来我以贝叶斯推理场景为例，介绍变分推理与ELBO（Evidence Lower BOund）的关系。此时目标分布$\nu(x)= p(x|\mathcal{D})$，其中$p(x|\mathcal{D})$表示模型参数$x$关于训练数据集$\mathcal{D}$的后验分布，可由下式定义：
$$
p(x|\mathcal{D}) = \frac{p(\mathcal{D}|x)p(x)}{p(\mathcal{D})},
$$
其中$p(\mathcal{D}|x)$表示似然函数（就是forward过程），$p(x)$表示模型的先验分布（可以简单理解为L1、L2正则化），$p(\mathcal{D})$表示未知的数据集分布，是一个归一化常数。在这种场景下，如何应用变分推理优化分布$q_\lambda$使其近似一个包含了未知归一化常数的分布$p(x|\mathcal{D})$？以及什么是ELBO？

首先从简单的贝叶斯公式开始，已知未知的归一化常数可以表示为：
$$
p(\mathcal{D}) = \frac{p(\mathcal{D}|x)p(x)}{p(x|\mathcal{D})},
$$
上式进一步整理，可得：
$$
\begin{align*}
& p(\mathcal{D}) = \frac{p(\mathcal{D}|x)p(x)}{p(x|\mathcal{D})} \\
\Longrightarrow & p(\mathcal{D}) = \frac{p(\mathcal{D}|x)p(x)q_\lambda(x)}{p(x|\mathcal{D})q_\lambda(x)}\\
\Longrightarrow & \log{p(\mathcal{D})} = \log{\frac{p(\mathcal{D}|x)p(x)}{q_\lambda(x)}} - \log{\frac{p(x|\mathcal{D})}{q_\lambda(x)}}\\
\Longrightarrow & \int{q_\lambda(x)\log{p(\mathcal{D})}\mathrm{d}x} = \int{q_\lambda(x)\log{\frac{p(\mathcal{D}|x)p(x)}{q_\lambda(x)}}\mathrm{d}x} - \int{q_\lambda(x)\log{\frac{p(x|\mathcal{D})}{q_\lambda(x)}}\mathrm{d}x}\\
\Longrightarrow & \log{p(\mathcal{D})}\int{q_\lambda(x)\mathrm{d}x} = \int{q_\lambda(x)\log{\frac{p(\mathcal{D}|x)p(x)}{q_\lambda(x)}}\mathrm{d}x} - \int{q_\lambda(x)\log{\frac{p(x|\mathcal{D})}{q_\lambda(x)}}\mathrm{d}x}\\
\Longrightarrow & \log{p(\mathcal{D})} = \underbrace{\mathbb{E}_{x\sim q_\lambda(x)}\left[\log{p(\mathcal{D}|x)p(x) - \log{q_\lambda(x)}}\right]}_{ELBO} + \text{KL}(q_\lambda(x) \| p(x|\mathcal{D})),
\end{align*}
$$
由于$\log{p(\mathcal{D})}$是一个未知的归一化**常数**，因此ELBO和$\text{KL}(q_\lambda(x) \| p(x|\mathcal{D}))$这两项的总和是不变的，我们的目标是最小化$\text{KL}(q_\lambda(x) \| p(x|\mathcal{D}))$，也就相当于最大化ELBO。而ELBO实际上是：
$$
\mathbb{E}_{x\sim q_\lambda(x)}\left[\log{p(\mathcal{D}|x)p(x) - \log{q_\lambda(x)}}\right] = -\text{KL}(q_\lambda(x)\|p(\mathcal{D}|x)p(x)),
$$
上式可用前文的变分推断算法求解。