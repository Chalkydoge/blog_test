---
title: 期末考试复习(开卷)
date: 2022-12-15 14:58:28
tags:
- 机器学习
mathjax: true

---
CNN,RNN, Transformer原理; 无监督学习
<!--more-->

## CNN

conv卷积的意义
- 全连接to卷积操作
- 卷积可以看作局部全连接-局部感知眼-局部的特征
- multi-kernel

pooling操作的意义
- subsampling: 采样降低数据的维度
- 扩大局部感受区域的范围
- 提高translation invariance

全连接层
- 特征提取到分类输出的连接

激活函数的选取
- sigmoid
- tanh
- relu(防止梯度消失, 稀疏激活,计算简单)

CNN的关键操作
- 卷积
- 非线性
- 空间区域的pooling操作
- 特征提取

filter: 过滤器(滤波器)
- 大小假设为$W\times H\times D$
- 计算经过该滤波器卷积后的数据尺寸
- 滤波器的参数数量(WHD)

低级特征-高级特征-最终特征

filter的步长选择
- 空间性
- stride>1的时候可能无法完全覆盖-加padding

Pooling层
- 表征大小更小可控
- 每一步激活函数之后进行独立操作

max-pooling
常见的取值：F=2,S=2或者F=3,S=2

## 高级CNN网络

Residual-Network 残差网络
- 如果输入特征是理想的
- 如果映射是理想的，最终的响应会怎样变化？

H(x) = x
H(x) = F(x) + x
F(x) = H(x) - x
当激活函数为0(F(x) = 0), H(x) = x

- 对于图片数据而言(原图片x - 变化后的图片F(x))之间的差异;

残差块Residual-Block机制
- 1*1 conv的使用, 减少参数数量

其他的网络结构：
- 多分支Multi-branch Inception
- 分组卷积
- Ensemble
- Channel-Attention机制 Conv-Block的attention
- shared-mlp

## Attention机制

注意力机制
- 快速提取局部重要特征

不同类型的attention

注意力机制

关心的特征是$q$, 目前的所有特征记作$X=(x_1,...,x_N)$, $s(x_n,q)$是打分函数, 有不同的选择

1. 计算注意力分布$\alpha = p(z = n|X,q) = softmax(s(x_n,q))$

2. 根据注意力关于输入信息的分布计算加权平均值作为最终的attention大小

$att(X,q) = \sum_{n=1}^N \alpha_n x_n$


hard-attention
键值对注意力
用$(K,V)$标识N个输入信息

- $att((K,V),q) = \sum_{n=1}^N \alpha_n v_n$

multihead-attention
并行从输入信息中选取多组信息, 每个attention关注输入信息的不同部分。
$att((K,V),Q) = att((K,V), q_1) \oplus ... \oplus att((K,V), q_M)$
$Q = [q_1, ..., q_M]$ 

global-attention=soft-attention
全局注意力机制

local-attention
局部注意力


计算attention的过程
- $e_i = \alpha(u, v_i)$ 计算attention分值
- $\alpha_i = \frac{e_i}{\sum_i e_i} $ normalize这些分数
- $c = \sum_i \alpha_i v_i$ 进行编码

key-value, query
output


### Self-attention机制

问题：如何建立非局部的依赖关系
- 全连接模型的局限性：输入长度变化就无法处理了
- 自注意力模型：通过attention机制动态生成参数

对self-attention来说，它跟每一个input vector都做attention，所以没有考虑到input sequence的顺序

类似于键值对Attention, self-attention将每一个输入的向量都做attention
$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt d_k })V
$$

- $\alpha = QK^T$
- softmax $\alpha$得到对应的系数
- 乘以$V$得到self-attention的分值
- 动态更新

与CNN的区别-提取特征：

全局的感知-对比-卷积结构的感知

$Q = XW^Q$,
$V = XW^V$,
$K = XW^K$
其中 $W^Q, W^V, W^K$ 都是可以动态变化的参数矩阵

不同类型之间的attention计算:
- 2-head的attention(取两个输入同时加和做attention计算)

self-attention机制的问题
- 没有位置信息
解决：引入位置编码 Positonal Encoding
- 每个位置都有一个独特的位置向量标记 $e^i$

最终将扩展的attention-score记作 
$e^i + a^i$, 其中 $a^i = f(q^i, k^i, v^i)$

#### CV中Attention机制的应用

以CBAM为例：通道注意力机制

按照通道进行处理得到通道特征，进行excitation
输出 = 每个通道分配的权重 => 重新定义输入的不同通道之间的重要性 => Scale系数


一组特征在上一层被输出，这时候分两条路线，第一条直接通过，第二条首先进行Squeeze操作（Global Average Pooling），把每个通道2维的特征压缩成一个1维，从而得到一个特征通道向量（每个数字代表对应通道的特征）。然后进行Excitation操作，把这一列特征通道向量输入两个全连接层和sigmoid，建模出那为什么要认为所有通道的特征对模型的作用就是相等的呢？
特征通道间的相关性，得到的输出其实就是每个通道对应的权重，把这些权重通过Scale乘法通道加权到原来的特征上（第一条路），这样就完成了特征通道的权重分配。（依照这个重要程度去提升有用的特征并抑制对当前任务用处不大的特征。）

## Transformer

- seq2seq的任务：序列输入，序列输出

encoder-decoder模型
- encoder接受输入序列信息，并生成一个中间结果
- decoder读取中间结果进行decoder，产生结果

encoder/decoder的内部可以是CNN/RNN
- transformer的encoder结构：输入- 输入嵌入编码-多头注意力机制+残差+归一化 + 前向MLP +残差+归一化- 最终的encoder结果

decoder
- auto-regressive

- output-embedding
- 解码器结构的transformer
    - masked-多头注意力机制+add&norm
    - 多头注意力机制+add&norm
    - feed-forward
    - add&norm
    - 线形层
    - softmax
    - 输出概率分布

- 为什么需要掩码操作？
解码的结果会影响下一个序列元素的结果, Q会发生变化!
不知道正确的输出长度，所以需要增加终止符号


transformer的结构中增加了encoder到decoder的交叉注意力机制

### Embeding
- embed层的作用：如果采用稀疏的one-hot编码变成稠密的向量表示

LLE(Local Linear Embedding)
- 局部线性embed：用$w_{ij}$ 表示数据点$x_i$ 和数据点$x_j$之间的关系程度
- 寻找一组合适的权重$w$ 满足$\sum_{i} |x^i - \sum_{j} w_{ij} x_j |$
- 转变为固定$w_{ij}$, 寻找一组$z_i$满足上述的最小化约束 => 得到局部2的嵌入压缩


Laplacian Eigenmaps


t-SNE
- t分布的随机邻居嵌入算法
- 相似的数据靠近，但是不同的数据可能覆盖同一个区域

计算组之间的相似度$S(x_i, x_j)$

数值对之间的相似度 $S'(z_i, z_j)$

然后寻找一组$z$系数使得两类分布尽可能靠近

$$
L = \sum_i KL(P(*|x_i)||Q(*|z_i))
$$

## 无监督学习

### Clustering

聚类算法

模板：
- 寻找相似点集合
- 寻找自然的描述数据的属性
- 寻找合适的分组方式
- 检测异常点/outlier检测
- 特征提取/选择

最小化 类内部的差异程度(最小化协方差矩阵)
最大化 观测样本在预测分布下的似然程度

实现这一类的算法需要满足以下要求：
1. 可扩展性
2. 可以处理不同的数据类型
3. 可以处理任意形状的聚类
4. 尽可能少的领域知识
5. 能够处理噪声和异常点
6. 对于输入数据的顺序不敏感
7. 维度高
8. 可解释性，可用性

分类：
1. 分类-聚类
- 完全不重合的聚类 K-means
- 存在重合的聚类 fuzzy K-means
- 概率聚类 高斯混合分布

2. 分层-聚类（结构化）


相似度指标
- 数值属性

按照距离的类别划分指标：

- Minkowski指标(Manhattan距离, Euclidean距离等等)
- 最小间距
- 最大间距
- 平均间距

类别指标
- 自底向上 合并(linkage)
- 自顶向下 分割(split)



Kmeans
目标：
最小化$J$

$$
J = min \sum_{i=1}^n \sum_{k=1}^K r_i^(k) |x_i - \mu_k|^2 
$$

{% raw %}

$$
r_i^{(k)} = 
\begin{cases}
1,\ if\ x_i 属于第k类 \\
0,\ 其余的情况
\end{cases}
$$

- $r_i^{(k)} = 1\ 如果x_i属于第k类,\ 否则为 0$
- $\mu_k$ 是分为第k类的聚类的原型向量

{% endraw %}

Kmeans的工作原理就是将$n$个数据点放入$d$维空间的$K$个聚类中, 满足最小化$J$的约束。

如何找到这样一种划分?
- $r_i$
- $\mu_k$

迭代步骤：
1. 固定 $\mu_k$, 最小化J, 得到 $r_i$
2. 固定 $r_i$, 最小化J, 得到 $\mu_k$
3. K-means是基于以上两步迭代的算法
不断重复以上步骤, 直到赋值不再改变

更新 $r_i^{(k)}$ 的过程和更新 $\mu_k$ 的过程

- $\mu_k$ 的更新 对于 $\frac{\partial J}{\partial \mu_k} = 0$ 求解得到 $\mu_k = \frac{\sum_i r_i^{(k)}x_i }{\sum_i r_i^{(k)} }$

- $\mu_k$ 是被分为第k类数据点的均值, 因此这个算法称为k-means


算法的执行过程：设置两个部分的迭代次数
- $E$ 次更新 $r_i$ 系数 适配数据点 $x_i$ 到最近的聚类 $k$
- $M$ 次更新 $\mu_k$ 均值大小, 重新划分聚类组织方式

K-means算法的应用:

- 有损失的数据压缩
- **存储K个聚类的中心数据, 而不是原始的数据点**(codebook的实际原理)
- 每一个数据点用K聚类来近似, 从而起到了压缩数据的功能(K << N的情况下)

K-means的问题：

- 对于离群点非常敏感

改进1：K-medoid算法(中心点聚类)

- 由中位数改为广义距离函数 $J' = min \sum_{i=1}^n \sum_{k=1}^K \nu (x_i, \mu_k) $
中位数不一定是K类中真实存在的元素; 而中心点是一定存在的

改进2: soft K-means算法
- 算法有额外的一个参数$\beta$ 标识分布的陡峭程度
- $\beta$ 是方差的倒数关系, $\sigma = \frac{k}{\sqrt \beta }$

更新$r_i^{(k)}$的算式

点$x_i$关于聚类$k$的归属程度
$r_i^{(k)} = \frac{exp(-\beta d(\mu_k, x_i))}{\sum_k' exp(-\beta d(\mu^{k'}, x_i))}$
求和发现始终等于1

算法的执行过程：

- 不断更新 $\mu_k = \frac{\sum_i r_i^{(k)} x_i}{R^{(k)}}$
- 以及总和 $R^{(k)} = \sum_i r_i^{(k)}$ 

区别：
- $r_i^{(k)}$的取值变成了[0,1]区间
- Hard K-means算法中的 min去除
- Soft K-means 用求取最大归属程度$r_i$的方式找最佳划分

## 混合高斯分布MoG

版本1

每一个聚类看作是一个球形高斯分布，拥有一个自己的半径 $\beta^k = \frac{1}{{\sigma_k}^2 }$
- 算法自动更新高斯分布的半径相关参数$\sigma_k$
- 算法也包含了每个聚类的权重系数$w_1,...,w_k$
- 最大似然估计 多个高斯分布混合的情况

初始化：
- 随机初始\{\mu_k\}$
- E step: 计算归属度 $r_i^k = \frac{\pi_k N(x|\mu_k, \sigma_k)}{\sum_j \pi_j N(x|\mu_j, \sigma_j)}$
- M step: 更新$\mu_k$, $\sigma_k^2$, $\pi_k = \frac{R^k}{\sum_k R^k}$, $\sum_{k=1}^K \pi_k = 1$

目标函数
$I(X) = ln P(X|\pi, \mu, \sigma) = \sum_{i=1}^n ln(\sum_{k=1}^K \pi_k N(x_i|\mu_k, \sigma_k)) $
最大似然估计

- 每个数据点的概率密度由混合高斯分布模型给出 $p(x|\theta) = \sum_{k=1}^K \pi_k p(x|\theta_k) $


聚类算法的评价标准：
1. purity 纯度 $I(\Omega, C) = \frac{1}{N} \sum_k max_j |w_k \cap c_j|$

2. NMI 互相之间信息的依赖程度
    - KL散度 $KL(p(x,y) || p(x)p(y))$
    - MI最小值0, 最大值维完美划分(但是可以继续细分)
    - 引入 $NMI(\Omega, C) = \frac{I(\Omega, C)}{(H(\Omega) + H(C)) / 2}$

3. Rand Index
    - $RI = \frac{TP+TN}{TP+TN+FP+FN}$
    - TP,TN,FP,FN

F系数
- RI对于FP和FN给了相同的权重
- F系数对于FN更大的惩罚系数$\beta \gt 1$

$F_{\beta} = \frac{(\beta^2 + 1)PR}{\beta^2 P + R}$
其中 $P = \frac{TP}{TP+FP}$, $R = \frac{TP}{TP+FN}$

## PCA降维

- PCA主成分分析, 寻找的是最大方差的降维结果组成

以1维为例：
$z_1 = w_1 \dot x$

可以看作是把x投影到w的平面上, 获得一组集合$z_1$

目标：$Var(z_1) = \frac{1}{N} \sum_{z_1} (z_1 - \bar z_1)^2$
其中我们假设系数矩阵的模为1 $|w_1| = 1$

推广到n个实例, $z = Wx$ 求解这样一个正交矩阵$W$

- 数学上的求解 最后就是找前$n$大的特征值，组成的一个对角阵就是答案。

- 另一种可能的解读方式： 最小化误差的构成

近似表示$x = \bar x + c_1 u_1 + ... + c_k u_k$, $[c_1, ..., c_k]$表征一张数字图像;
(分解k个成分)

$x - \bar x = \sum_{k=1}^K c_k u_k$

目标损失：$L = |(x-\bar x) - \hat x|^2$最小化

PCA看起来就是一个单隐层的神经网络, 使用一个线性的激活函数
