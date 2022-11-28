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

- vhost-user

用户空间快速数据包

vhost-user 的实现

vhost-user 和 vhost 的实现原理是一样，都是采用 vring 完成共享内存，eventfd 机制完成事件通知。不同在于 vhost 实现在内核中，而 vhost-user 实现在用户空间中，用于用户空间中两个进程之间的通信，其采用共享内存的通信方式。

vhost-user 基于 C/S 的模式，采用 UNIX 域套接字（UNIX domain socket）来完成进程间的事件通知和数据交互，相比 vhost 中采用 ioctl 的方式，vhost-user 采用 socket 的方式大大简化了操作。

vhost-user 基于 vring 这套通用的共享内存通信方案，只要 client 和 server 按照 vring 提供的接口实现所需功能即可，常见的实现方案是 client 实现在 guest OS 中，一般是集成在 virtio 驱动上，server 端实现在 qemu 中，也可以实现在各种数据面中，如 OVS，Snabbswitch 等虚拟交换机。


问题：

- Physical - Virtual - Physical 多次转化的overhead很大
- 通信方式带来的损耗, 路径长&前向转发消耗

Virtual network I/O is a mechanism for communications and
vhost-user is a widely used protocol supported by DPDK.
While vhost-user imposes significant packet copy overhead
in the process of forwarding on the Physical-Virtual-Physical
(PVP) datapath,


Virtio&V-host 这些虚拟化技术的背后实现原理?

观察一下virtio的数据结构实现发现:
virtio管理了一块连续内存()
- ring-buffer
- virtio 把这段内存分成3个部分，依次是 desc，avail，used。每一块是一个数组，可以顺序索引。他们的元素个数是一样，也就是是 ring 的 长度。

以VHOST为例，来解释一下数据是如何流动的：

1. client（qemu）创建共享内存，然后通过ioctl与内核通信，告知内核共享内存的信息，这种就是kernel作为server的vhost; 或者通过Unix domain来跟其他的进程通信，这叫做vhost-user。下面以Unix domain为例。

2. Unix domain可以使用sendmsg/recvmsg来传递文件描述符，这样效率更高一些；client创建好共享内存，发送描述符到server，server直接mmap这个描述符就可以了，少了一个open的操作;

3. 读写：以net为例，两个vring，一个tx，一个rx共享内存存放desc，avail，used这些信息，以及avail->idx, used->idx这些信息。

4. 当client写的时候，数据放在vring->last_avail_idx这个描述符里，注意last_avail_idx、last_used_idx这两个值，在client和server看到的是不一样的，各自维护自己的信息，作为链表头尾。

5. 添加id到avail里面，shared.avail.idx++。注意，client此处的last_avail_idx指向的是描述符的id，而不是avail数组的id，这个跟Server上的last_avail_idx的意思不一样。为什么这样做呢？

- last_avail_idx 表示可用的数据项，可以直接拿出来用，用完之后，取当前的next;
- 当回收的时候，也就是client在处理used的时候，可以直接插入到last_avail_idx的前面，类似链表插入到表头；

6. Server收到信号后，从自己记录的last_avail_idx开始读数据，读到avail->idx这里，区间就是[server.last_avail_idx, shared.avail.idx)。

7. Server处理每处理完一条请求，自己的 last_avail_idx ++; 同时插入 id 到 used 里面，插入的位置是 shared.used.idx，然后 shared.used.ix+ +。

8. used.idx此时就像avail->idx一样，表示最新的used的位置。Server通知client处理完数据后，client就可以回收used的描述符了，可回收的区间是[client.last_used_idx, shared.used.idx)。

9. Kickfd，callfd都是在client创建，然后通过unix domain发送给server。client通知server叫kick

Vhost是client与kernel（server）的交互，client与内核共享内存，局限性是如果client要发送消息到用户进程，很不方便； Vhost-user使用unix domain来进行通信，两个用户进程共享内存，达到和vhost一样的效果。



C. System Perspective

拓扑结构-clos结构用于中心网络

边缘网络使用cots-servers，运行着用户的应用

这样一种软件结构带来的问题:

- 提供了可伸缩性, 带来的问题就是降低整体的性能

问题总结:

网络分层技术需要实现以下几个要点:

1. Network functions 需要能够按需即时的创建&销毁(资源分配/释放)

2. 在有限的硬件(计算节点上)复用资源

3. 根据服务所需的资源要求动态调整分配

4. 自动化管理

CLOS结构 聚集了网络应用在边缘节点上面
- 自然而然地性能受限于哪些NFV节点地性能
- NFV节点性能地提升, 与边缘网络地通用服务器之间的差距仍然制约了相应的应用性能(尤其是little-packets)

重点关注:
- NIC(网卡相关的packet-IO)如何提升
- 虚拟网络的宿主机-用户之间的虚拟网络IO性能

### Packet IO

NFV-node: 40M pps(64B-packets)

explore the possibility of COTS servers in carrier grade networks

1. Hardware Offloading

2. Improving Cache

- DDIO
- Direct Cache Access
- Cache Allocation Technology


3. IO Parallelization

- Rx/Tx queue handling - parallelized

limited by the performance of the NIC in per-queue transferring efficiency


With the current hardware technology for COTS servers,
prolonging the time interval between consecutive packets is
the fundamental way to attain different level of performance.

One promising approach to enlarge average packet
size (prolong the average time interval) is packet aggregation

multi-stage packet aggregation (e.g. 1st: CNF, 2nd: vSwitch, 3rd: ToR switch)

flexible (easily-extendable) and interoperable
header format of aggregated packets needs to be designed

可行的解决方式:

Packed-Virtqueue cache的使用低效可以改进

Bypass virtual-network IO

- software2software data plane offloading

- modularized dataplane functions(P4程序集成在底层的virtual switch上面); 或者在内核(eBPF)
