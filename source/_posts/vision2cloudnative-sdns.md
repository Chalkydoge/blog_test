---
title: 软件中心化的云原生网络函数-机遇和挑战
date: 2022-11-01 21:44:33
tags:
- sdn
---

<!--more-->

## 概要-为什么需要一套技术支持

Network slicing 网络[切片]

cloud native network functions (CNFs) 

virtualized systems => 性能提升 目标 10 Gbps throughput 

container/VM-based applications

The present pace of performance enhancements in hardware indicates that
straightforward optimization of existing system components has
limited possibility of filling the performance gap

说白了就是硬件的条件还不能简单粗暴的满足要求。

We focus not only on the performance aspect of CNFs but also
other pragmatic aspects such as interoperability with the current
environment (not clean slate)

missing-link technologies 

## 常用的数据平面-性能优化技巧

Cloud Native Networking 云原生网络
- 可扩展性，简单性，容错性

叶子-主干 Leaf-Spine 拓扑结构  边缘网络中很常用

CLOS架构-无阻塞网络结构


切片化-虚拟技术-上层的虚拟网络 overlaid

- 在需要的时候即时扩展分配 VXLAN(Virtual Extensible LAN)

NFV-Node

- Kubernetes 通用 但是性能?

- DPDK packet I/O加速器 

CPU(计算性能的优化)
context switches -> CPU binding, polling降低; 跳过内核

Memory(内存相关优化)
- cache-aware/huge tables
- prefetching

CNF is deployed as independent container process
on the host, and a high-functional virtual switch (e.g. OVS-
DPDK) controls host–guest (inter-process) communications.

vhost协议

