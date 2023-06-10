## 分布近似问题

就本人目前所了解的现状，分布近似问题尚未在学界被严肃定义，多数研究者可能熟悉的是多年来持续火热的生成模型：
生成对抗网络（Generative Adversarial Networks, GAN），扩散模型（Diffusion Model）等等。
另一些核心却冷门的领域，如马尔科夫链蒙特卡洛采样（Markov Chain Monte Carlo, MCMC），
或是变分推理（Variational Inference, VI），则极少成为人们热议的对象。
实际上，MCMC与VI常常出现在生成模型的论文中，从远古时期的变分自编码器VAE，到Energy-based、基于Score Matching的
生成模型，再到近两年火热的扩散模型（Diffusion Model），甚至生成对抗网络（GAN）都可以被解释为使用特殊kernel的
VI。另一方面，起始于2016年的粒子变分推理法（Particle-based Variational Inference, ParVI）被逐步构建与MCMC、VI方法的联系，生成与采样，看似不相关的两个概念，逐渐开始从梯度流视角被统一阐释。

### 问题定义

什么是分布近似问题呢？该问题可被定义为如下形式：

$$
\min_{\mu} \mathcal{F}(\mu, \nu), \tag{1}
$$

这里的 $\mu$ 和 $\nu$ 表示概率分布， $\mathcal{F}$ 表示一个度量，计算分布 $\mu$ 和 $\nu$ 之间的差异（不一定是距离），比如常见的KL散度（Kullback-Leibler Divergence），不常见的最大平均差异（Maximum Mean Discrepancy, MMD）和沃瑟斯坦距离（Wasserstein Distance）。

公式 $(1)$ 定义了一个优化问题，优化目标是 $\mathcal{F}(\cdot, \nu)$ ，或者写成 $\mathcal{F}_{\nu}$ ，
优化变量是一个分布 $\mu$ ， 全局最优为另一个分布 $\nu$ 。
也就是说，我们希望找到一个分布 $\mu$ 在度量 $\mathcal{F}$ 为判别标准的情况下接近分布 $\nu$ 。

看起来，这个形式 $(1)$ 非常接近我们熟知的实向量空间中的优化（熟悉凸优化的读者应该能很快get到联系）：

$$
\min_{x} F(x). \tag{2}
$$

比如在深度学习领域，优化目标 $F$ 表示训练的损失函数, 优化变量 $x$ 表示神经网络的参数。
当然，神经网络的参数是逐层的，但是我们可以将其抽象表示为一个高纬度的向量。
此外，由于训练过程中输入的数据（集）不计算梯度，所以可以不把这些变量表示在公式 $(2)$ 中。
实际上在上述的表达中，数据集和网络结构共同定义了优化目标 $F$ 。

看起来，公式 $(1)$ 可类比作公式 $(2)$ 的“升级”形式:
1. 优化目标 $F$ 计算两个**向量**之间的差异 $\to$ 优化目标 $\mathcal{F}$ 计算两个**分布**之间的差异。
2. 优化变量 $x$ 是一个**实向量** $\to$ 优化变量 $\mu$ 是一个**概率分布**。 

### 贝叶斯学习

> 那么表达式 $(1)$ 也可以按照上述神经网络的方式来理解吗？

实际上，如果从训练神经网络的角度阐释 $(1)$ ， 我们可以非常方便地理解**贝叶斯学习**。

首先回顾贝叶斯学习的基础——贝叶斯公式，建模了后验、似然、先验之间的关系：

$$
p(x|y) = \frac{p(y|x)p(x)}{p(y)}.
$$

假设现在使用贝叶斯公式建模一个分类任务的神经网络参数 $x$ 的后验分布，则有：

$$
p(x|\mathcal{D}_{train}) = \frac{p(\mathcal{D}_{train}|x)p(x)}{p(\mathcal{D}_{train})},
$$

这里使用 $\mathcal{D}_{train} = \{d_{train, j}\}^N_{j=1}$ 表示训练数据集（为了表达简洁就不再拆分为输入和输出了），
上述公式可进一步表达为：

$$
p(x| \mathcal{D}_{train} ) \propto \prod^N_{j=1}{p( d_{train, j} |x)}p(x), \tag{3}
$$

上式表达了在给定训练数据集的情况下，神经网络参数的后验**分布**，可以被建模为**每一条输入数据的似然函数的乘积**，
再乘上模型参数的先验分布 $p(x)$ 。
通常似然函数的形式比较复杂（由神经网络决定），先验分布形式简单，例如高斯分布。

再推导一步，可得：

$$
-\log{p(x|\mathcal{D}_{train})} = -\left(\sum^{N}_{j=1}\log{p(d_{train, j}|x)} + \log{p(x)}\right).
$$

显然，等式右边就是常见的分类任务的损失函数，其中加号左边可以是BCE、CE这种，加号右边可以是L1、L2正则化。
如果按照优化的思路，找到一个神经网络的参数 $x$ 使得 $-\log{p(x|\mathcal{D}_{train})}$ 最小，
我们就找到了当前训练集上的一个最优模型，神经网络的推理过程就是再次计算似然函数：

$$
p(d_{test}|\mathcal{D}_{train}) = p(d_{test}|x^*), \quad \text{where} \quad x^* = \argmin_{x} -\log{p(x|\mathcal{D}_{train})}.
$$

**然而对于贝叶斯学习，或者说贝叶斯推理，推理过程被建模为**：

$$
\begin{align*}
    \begin{split}
    p(d_{test}|\mathcal{D}_{train}) 
    & = \int \underbrace{p(d_{test}| x)}_{prediction} \frac{\overbrace{p(\mathcal{D}_{train}|x)}^{likelihood}\overbrace{p(x)}^{prior}}{\underbrace{p(\mathcal{D}_{train})}_{marginal}}\mathrm{d}x \\
    & = \int \underbrace{p(d_{test}|x)}_{prediction}\underbrace{p(x|\mathcal{D}_{train})}_{posterior}\mathrm{d}x \\ 
    & = \mathbb{E}_{x \sim p(x|\mathcal{D}_{train})}\left[p(d_{test}|x)\right],
    \end{split}
\end{align*}
$$