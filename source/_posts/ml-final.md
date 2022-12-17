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
- 大小假设为$W\time H\times D$
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

$Q = XW^Q$
$V = XW^V$
$K = XW^K$

不同类型的相关性
- 2-head的attention(取两个输入同时加和做attention计算)

self-attention机制的问题
- 没有位置信息
解决：引入位置编码 Positonal Encoding
- 每个位置都有一个独特的位置向量标记 $e^i$

$e^i + a^i$, $a^i = f(q^i, k^i, v^i)$

CV中Attention机制的应用

以CBAM为例：通道注意力机制

按照通道进行处理得到通道特征，进行excitation
输出 = 每个通道分配的权重 => 重新定义输入的不同通道之间的重要性 => Scale系数


一组特征在上一层被输出，这时候分两条路线，第一条直接通过，第二条首先进行Squeeze操作（Global Average Pooling），把每个通道2维的特征压缩成一个1维，从而得到一个特征通道向量（每个数字代表对应通道的特征）。然后进行Excitation操作，把这一列特征通道向量输入两个全连接层和sigmoid，建模出那为什么要认为所有通道的特征对模型的作用就是相等的呢？
特征通道间的相关性，得到的输出其实就是每个通道对应的权重，把这些权重通过Scale乘法通道加权到原来的特征上（第一条路），这样就完成了特征通道的权重分配。（依照这个重要程度去提升有用的特征并抑制对当前任务用处不大的特征。）

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

- $r_i^(k) = 1 if x_i是第k类 否则为 0$