---
title: 不同的网络结构整理
date: 2022-12-30 16:25:34
tags:
- networks
- 摸鱼
---

先看最简单的跨主机通信组件的实现(Flannel), 后面再补上其他的实现.
<!-- more -->

1. 基于Flannel组件的**第3层**流量转发, 解决 不同主机之间 跨机通信的问题(网络资源的问题)

它自己的简介:
- 每个主机上运行一个简易的代理程序 `flanneld`

- 每个主机上的代理程序负责分配一组子网段,  来源于 主机集群所属的 网络地址空间;


Flannel runs a small, single binary agent called flanneld on each host, and is responsible for allocating a subnet lease to each host out of a larger, preconfigured address space. Flannel uses either the Kubernetes API or etcd directly to store the network configuration, the allocated subnets, and any auxiliary data (such as the host's public IP). Packets are forwarded using one of several backend mechanisms including VXLAN and various cloud integrations.

举个例子：一个集群里面有两台主机(实际的机器A和B)

集群范围内的网络地址空间为`10.1.0.0/16`(CIDR)

A主机获得的子网IP范围`10.1.15.0/24`
A主机上有两个pod, pod1和pod2, 对应的IP分别是`10.1.15.2/24`和`10.1.15.3/24`

B主机获得的子网IP范围`10.1.16.0/24`
B主机上有两个pod, pod3和pod4, 对应的IP分别是`10.1.16.2/24`和`10.1.16.3/24`

如果 A主机上的pod1 想要与B主机上的pod3进行通信, 需要进行某种转发才能完成(因为不在一台机器上)

- A主机的flanneld程序会把自己的subnet信息存入etcd: subnet = `10.1.15.0/24`的所在主机可以通过一个内网IP `192.168.0.100`访问;

- 同样的B主机上的flanneld程序会把自己的subnet信息存入etcd: subnet = `10.1.16.0/24` 的所在主机可以通过一个内网IP `192.168.0.200`访问;

- 每台主机上的flanneld通过**监听**etcd，也能够知道其他的subnet与哪些主机相关联;

- flanneld只要想办法将封包从Machine A转发到Machine B; 一旦目的地址为`10.1.16.2/24`的数据包可以到达主机B, 就能够通过CNI0网桥转发到这台主机上的pod3, 从而达到跨机器通信.

采用的方法可以有很多种:
- hostgw: 把B主机看作一个网关, 当有封包的目的地址在subnet 10.1.16.0/24范围内时，就将其直接转发至B即可; 在满足仍有subnet可以分配的条件下，我们可以将上述方法扩展到任意数目位于同一子网内的主机;

每台主机都是租借了一个subnet，如果到了一定时间不进行更新，那么该subnet就会过期从而重新分配给其他的主机(类似于DHCP)

调用GetNetworkConfig()
- SubnetLen表示每个主机分配的subnet大小;如果集群的网络地址空间大于/24，则SubnetLen配置为24，否则它比集群网络地址空间小1，例如集群的大小为/25，则SubnetLen的大小为/26

- SubnetMin是集群网络地址空间中最小的可分配的subnet，可以手动指定，否则默认配置为集群网络地址空间中第一个可分配的subnet。例如对于`10.1.0.0/16`，当SubnetLen为24时，第一个可分配的subnet为`10.1.1.0/24`

- SubnetMax表示最大可分配的subnet，对于”10.1.0.0/16″，当subnetLen为24时，SubnetMax为”10.1.255.0/24″

- BackendType为使用的backend的类型，如未指定，则默认为`UDP`

- Backend中会包含backend的附加信息，例如backend为vxlan时，其中会存储vtep设备的mac地址

发现新节点
1. 当本主机的flanneld启动时，如果集群中已经存在了其他主机，我们如何通过backend进行配置，使得封包能够到达它们
2. 如果之后集群中又添加了新的主机，我们如何获取这一事件，并通过backend对配置进行调整，对于删除主机这类事件同理

通过etcd解决


hostgw是最简单的backend，它的原理非常简单，直接添加路由，将目的主机当做网关，直接路由原始封包。

我们知道当backend为hostgw时，主机之间传输的就是原始的容器网络封包，封包中的源IP地址和目的IP地址都为容器所有。这种方法有一定的限制，就是要求所有的主机都在一个子网内，即二层可达，否则就无法将目的主机当做网关，直接路由。

而udp类型backend的基本思想是：既然主机之间是可以相互通信的（并不要求主机在一个子网中），那么我们为什么不能将容器的网络封包作为负载数据在集群的主机之间进行传输呢？这就是所谓的overlay

另外两个著名的k8s网络策略插件,Calico和Cilium后面(一定介绍)(下次一定...)

---

还需要补充一些k8s网络模型的相关知识, CNI的工作原理; 期望可以通过本地搭建minikube的方式体验k8s不同service之间的网络通信方式(service是k8s管理的服务对象, 而一个服务往往可以包括多个pod(多备份), 不同的pod可以处在不同的node上, node依附在物理机上, 而物理机上的IP地址,网络地址都是可能不同的, 因此是可能出现从一个pod到另一个pod的网络通信; 如何更好的实现这一点? 需要我们在CNI上面进行自己定义, 通过CNI提供的网络资源相关接口, 适配服务从而实现相应的功能)