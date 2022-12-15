# RL + Diffusion models

Planning with Diffusion for Flexible Behavior Synthesis

## 用于Model-based RL:

在基于模型的方法中，相对于model-free(DQN，A2C)的方法，多出了一个对真实世界建模的步骤，使得我们给出(状态，动作)，便能够通过模型得到下一个状态。

model-based RL一般有两个步骤：

-   traning a predictive model：
    -   Supervised learning	$minimize_f E_{s_t,a_t,s_{t+1}\sim D}||s_{t+1}-f(s_t,a_t)||$

-   use model to evaluate potential plans $a_{0:T}$,selecting the best ones

    -   Trajectory optimization 	$maxxmize_{a_{0:T}}r(s_0,a_0)+r(f(s_0,a_0),a1)+r(f(f(s_0,a_0),a2))+\dots$

    -   最大化回报

存在的问题：太依赖f，就是训练出来的环境。这样得到的答案不一定是最好的，可能只是一些比较好的特殊解。

现在的model-based RL 大部分都是使用了model-free的一些方法，比如说value function和policy gradients 去规避上述的一些问题。

作者：将两个步骤混在一起。用一个大的生成网络，取代二者。

## 优势

长期准确率比当前单步error 重要...

## 解决方案

Diffuser ：2015，2020两篇论文，目前热门。

预测所有的timesteps。

优点：

-   long-horizon scalability
-   Task compositionality
    -   可用在复杂任务。(多rewards可以相加)
    -   

## 什么是diffusion models？

它的研究最早可以追溯到2015年，当时，斯坦福和伯克利的研究人员发布了一篇名为Deep Unsupervised Learning using Nonequilibrium Thermodynamics的论文：![img](RL+diffusion.assets/0df431adcbef760967873c5ecbb970c67ed99ed6.png)

但这篇研究和目前的Diffusion Model非常不一样；而真正使其发挥作用的研究是2020年，一项名为Denoising Diffusion Probabilistic Models的研究：

![img](RL+diffusion.assets/37d3d539b6003af35d01a133ef4e15561138b69d.png)



例子：

-   AI绘画 ：DALL$\cdot$E 2

从DDPM说起diffusion  2020年论文。

GAN：对抗生成网络

目的 找马尔可夫trasition 的reverse

### 前向过程

不断地加高斯噪声

$x^0\rightarrow \dots x^n$

推论一：如何得到$x^t$时刻的分布(前向过程)

-   $a_t=1-\beta_t$ $\beta$随着$t$越来越大。论文中从0.0001到0.002，对应的$a_t$越来越小
-   $x_t=\sqrt{a_t}x_{t-1}+\sqrt{1-a_t}z_1$，加噪声，一开始加点噪就有效果，后面需要越来越多
    -   t越后，噪音应该加的越大。为了让每一次的扩散程度相当。所以$\beta$越来越大。
    -   $x_t$可以迭代得到。

$x_{t-1}=\sqrt{a_{t-1}}x_{t-2}+\sqrt{1-a_{t-1}}z_2$

那么$x_t$可以用$x_{t-2}$表示​：
$$
x_t=\sqrt{a_ta_{t-1}}x_{t-2}+\sqrt{a_t(1-a_{t-1})}z_2+\sqrt{1-a_t}z_1
$$
每次加入的噪声满足高斯分布：$z_1,z_2\dots\sim N(0,I)$。

那么$Z_1\sim N(0,1-a_t)$，$Z_2\sim N(0,a_t(1-a_{t-1}))$，

相加后依旧服从高斯分布：$N(0,\sigma_1^2 I)+N(0,\sigma_2^2 I)\sim N(0,(\sigma_1^2+\sigma_2^2) I)$

于是$Z_1+Z_2\sim\sqrt{1-a_ta_{t-1}}\overline z_2$

故：
$$
x_t = \sqrt{a_ta_{t-1}}x_{t-2}+\sqrt{1-a_ta_{t-1}}\overline z_2
$$
如何使用$x_0$来表示$x_t$？不断迭代，总结规律发现：
$$
x_t=\sqrt{\overline a_t}x_0+\sqrt{1-\overline a_t} z_t
$$
其中,$\overline a_t$是一个累乘。



### 逆向过程

我们的目的并不是得到噪音，而是从噪音推出原来的信息。

-   前向过程的轨迹我们记为$q(X_{0:T})$
-   逆向过程的轨迹我们记为$p_\theta(X_{0:T})$，$\theta$代表神经网络的参数

![image-20221214185819775](RL+diffusion.assets/image-20221214185819775.png)

逆向过程每一次都只使用模型，往前推理一步(在一定程度上是猜出来的)

即求$p(X_{t-1}|X_t)$

去噪过程推理：

我们从前向过程中已知：$p(x_{t}|x_{t-1})$

使用贝叶斯公式：
$$
q(x_{t-1}|x_t,x_0)=q(x_t|x_{t-1},x_0)\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}
$$
前向过程中已知：$x_t=\sqrt{\overline a_t}x_0+\sqrt{1-\overline a_t} z_t$
$$
q(x_{t-1}|x_0)=\sqrt{\overline a_{t-1}}x_0+\sqrt{1-\overline a_{t-1}}z\sim N(\sqrt{\overline a_{t-1}}x_0,1-\overline a_{t-1})
$$

$$
q(x_{t}|x_0)=\sqrt{\overline a_{t}}x_0+\sqrt{1-\overline a_{t}}z\sim  N(\sqrt{\overline a_{t}}x_0,1-\overline a_{t})
$$

$$
q(x_{t}|x_{t-1},x_0)=\sqrt{a_t}x_{t-1}+\sqrt{1-a_t}z\sim N(\sqrt{a_t}x_t-1,1-a_t)
$$

于是：$N(\mu,\sigma^2)\propto exp(-\frac12\frac{(x-\mu)^2)}{\sigma^2})$，故：
$$
q(x_{t-1}|x_t,x_0)\propto exp(-\frac12(\frac{(x_t-\sqrt{a_t}x_{t-1})^2}{\beta_t})+\frac{(x_{t-1}-\sqrt{\overline a_t}x_0)^2}{1-\overline a_{t-1}}-\frac{(x_t-\sqrt{\overline a_{t}}x_0)^2}{1-\overline a_t})
$$
平方展开化简之后：
$$
q(x_{t-1}|x_t,x_0)\propto exp(-\frac12((\frac{a_t}{\beta_t}+\frac1{1-\overline a_{t-1}})x^2_{t-1}-(\frac{2\sqrt{a_t}}{\beta_t}x_t+\frac{2\sqrt{\overline a_{t-1}}}{1-\overline a_{t-1}}x_0)x_{t-1}+C(x_t,x_0)))
$$
其中 $C(x_t,x_0)$为常数项，不影响。

又$N(\mu,\sigma^2)\propto exp(-\frac12\frac{(x-\mu)^2)}{\sigma^2})=exp(-\frac12(\frac1{\sigma^2}x^2-\frac{2\mu}{\sigma^2}x+\frac{\mu^2}{\sigma^2}))$

所以与上式比对后发现，

-   在这个任务中，方差为固定值。但是其它论文也给出其他的改进版本，为不固定。

$$
\widetilde\mu(x_t,x_0) =\frac{\sqrt{a_t}(1-\overline a_{t-1})}{1-\overline a_t}x_t+\frac{\sqrt{\overline a_{t-1}}}{1-\overline a_t}x_0
$$

但我们没有$x_0$呀？已知$x_t=\sqrt{\overline a_t}x_0+\sqrt{1-\overline a_t} z_t$，所以$x_0=\frac1{\sqrt{a_t}}(x_t-\sqrt{1-\overline a_t}z_t)$。代入上式得：
$$
\widetilde\mu(x_t)=\frac 1{\sqrt{a_t}}(x_t-\frac{\beta_t}{\sqrt{1-\overline a_t}} z_t)
$$
$z_t$如何得到？即为我们要估计得每一个时刻得噪音。无法直接求解，选择近似解。训练一个模型去预测。
$$
p(x_{t-1}|x_{t})\sim N(\frac 1{\sqrt{a_t}}(x_t-\frac{\beta_t}{\sqrt{1-\overline a_t}} z_t),\sigma^2)
$$

### 训练z_t​

标签来自前向过程，训练加的噪声。

很多论文中用的Unet，很小的网路结构。

Loss

### 应用：

![image-20221214195905940](RL+diffusion.assets/image-20221214195905940.png)

training：

-   随机取图像，batch = n
-   随机采样得到每张图像的前向扩散次数[1,T]
-   $\epsilon\sim N(0,I)$，作为随机噪声。
-   梯度下降更新参数，往模型输入($x_t$，$t$)。（t 可以类比transformer的位置编码）

Sampling：

-   x_T 随机产生
-   随机产生噪音，在每一个逆向操作中，一步一步通过模型，往前得到$x_0$。最后一步不加噪音。
-   返回$x_0$

PS 输入可以是噪音点，也可以是文本特征。等等。。

### 模型实现：



## **Planning with Diffusion for Flexible Behavior Synthesis** 

Planning as sampling：

扩散模型的迭代去噪过程通过从以下形式的扰动分布中采样来实现灵活的调节：
$$
\widetilde p_\theta(\tau)\propto p_\theta(\tau)h(\tau)
$$
函数h(τ)可以包含关于先前证据(如观察历史)、期望结果(如要达到的目标)或要优化的一般函数(如奖励或成本)的信息。在这个扰动分布中执行推理可以被看作是第2.1节中提出的轨迹优化问题的概率模拟，因为它需要找到在pθ(τ)下物理上真实的轨迹，以及在h(τ)下高回报(或约束满足)的轨迹。由于动力学信息与摄动分布h(τ)分离，单个扩散模型pθ(τ)可以重复用于相同环境中的多个任务。

使用它对我们的action进行一个限制。

此任务所需的扰动函数是观测值的狄拉克增量，其他地方为常数。具体地说，如果ct是时间步t的状态约束，那么

![image-20221215125910344](RL+diffusion.assets/image-20221215125910344.png)

效果如下：

<video src="joined.mp4"></video>

将模型的输入与输出表示为一个二维数据，然后就可以很自然地把它丢进diffusion里面做训练。

![image-20221215131103572](RL+diffusion.assets/image-20221215131103572.png)

伪代码：

![image-20221215131252901](RL+diffusion.assets/image-20221215131252901.png)

-   随机生成 $\tau$轨迹，观察到状态s
-   从N到1：
    -   apply condition：
    -   计算$\mu$，重采样得到新$\tau$
    -   修正s

