【代码阅读】最大均值差异（Maximum Mean Discrepancy, MMD）损失函数代码解读（Pytroch版）
=============================================================

## 代码及参考资料来源  
Source code: easezyc/deep-transfer-learning [\[Github\]](https://github.com/easezyc/deep-transfer-learning/blob/master/DAN/mmd.py)  
参考资料：[迁移学习简明手册](https://zhuanlan.zhihu.com/p/35352154)

MMD介绍
-----

MMD（最大均值差异）是迁移学习，尤其是Domain adaptation （域适应）中使用最广泛（目前）的一种损失函数，主要用来度量两个不同但相关的分布的距离。两个分布的距离定义为：

MMD(X,Y)=∣∣1n∑i=1nϕ(xi)−1m∑j=1mϕ(yj)∣∣H2(1)MMD(X,Y) = ||\\frac{1}{n}\\sum_{i=1}^n\\phi(x\_i)-\\frac{1 {m}\\sum\_{j=1}^m\\phi(y\_j)||\_H^2\\tag{1}MMD(X,Y)=∣∣n1​i=1∑n​ϕ(xi​)−m1​j=1∑m​ϕ(yj​)∣∣H2​(1) 


其中 HHH 表示这个距离是由 ϕ()\\phi()ϕ() 将数据映射到再生希尔伯特空间（RKHS）中进行度量的。

为什么要用MMD?
---------

Domain adaptation的目的是将源域（Source domain）中学到的知识可以应用到不同但相关的目标域（Target domain）。本质上是要找到一个变换函数，使得变换后的源域数据和目标域数据的距离是最小的。所以这其中就要涉及如何度量两个域中数据分布差异的问题，因此也就用到了MMD。至于Domain adaptation的前生今世可以参考王晋东大佬的[知乎专栏](https://zhuanlan.zhihu.com/wjdml)

MMD的理论推导
--------

MMD的关键在于如何找到一个合适的 ϕ()\\phi()ϕ() 来作为一个映射函数。但是这个映射函数可能在不同的任务中都不是固定的，并且这个映射可能高维空间中的映射，所以是很难去选取或者定义的。那如果不能知道ϕ\\phiϕ，那MMD该如何求呢？我们先展开把MMD展开：  
MMD(X,Y)=∣∣1n2∑in∑i′nϕ(xi)ϕ(xi′)−2nm∑in∑jmϕ(xi)ϕ(yj)+1m2∑jm∑j′mϕ(yj)ϕ(yj′)∣∣H(2)MMD(X,Y) =||\\frac{1}{n^2}\\sum_{i}^n\\sum_{i'}^n\\phi(x\_i)\\phi(x\_i')-\\frac{2}{nm}\\sum_{i}^n\\sum_{j}^m\\phi(x\_i)\\phi(y\_j)+\\frac{1}{m^2}\\sum_{j}^m\\sum_{j'}^m\\phi(y\_j)\\phi(y\_j')||_H\\tag{2}MMD(X,Y)=∣∣n21​i∑n​i′∑n​ϕ(xi​)ϕ(xi′​)−nm2​i∑n​j∑m​ϕ(xi​)ϕ(yj​)+m21​j∑m​j′∑m​ϕ(yj​)ϕ(yj′​)∣∣H​(2)  
展开后就出现了ϕ(xi)ϕ(xi′)\\phi(x\_i)\\phi(x\_i')ϕ(xi​)ϕ(xi′​)的形式，这样联系SVM中的核函数k(∗)k(*)k(∗)，就可以跳过计算ϕ\\phiϕ的部分，直接求k(xi)k(xi′)k(x\_i)k(x\_i')k(xi​)k(xi′​)。所以MMD又可以表示为：  
MMD(X,Y)=∣∣1n2∑in∑i′nk(xi,xi′)−2nm∑in∑jmk(xi,yj)+1m2∑jm∑j′mk(yj,yj′)∣∣H(3)MMD(X,Y) =||\\frac{1}{n^2}\\sum_{i}^n\\sum_{i'}^nk(x\_i, x\_i')-\\frac{2}{nm}\\sum_{i}^n\\sum_{j}^mk(x\_i, y\_j)+\\frac{1}{m^2}\\sum_{j}^m\\sum_{j'}^mk(y\_j, y\_j')||_H\\tag{3}MMD(X,Y)=∣∣n21​i∑n​i′∑n​k(xi​,xi′​)−nm2​i∑n​j∑m​k(xi​,yj​)+m21​j∑m​j′∑m​k(yj​,yj′​)∣∣H​(3)  
在大多数论文中（比如DDC, DAN），都是用高斯核函数k(u,v)=e−∣∣u−v∣∣2σk(u,v) = e^{\\frac{-||u-v||^2}{\\sigma}}k(u,v)=eσ−∣∣u−v∣∣2​来作为核函数，至于为什么选用高斯核，最主要的应该是高斯核可以映射无穷维空间（具体的之后再分析）

理论到这里就差不多了，那如何进行实现呢？

在TCA中，引入了一个核矩阵方便计算  
\[Ks,sKs,sKs,tKt,t\](4) \\begin{bmatrix} K_{s,s} & K_{s,s} \\\ K_{s,t} & K_{t,t} \\\ \\end{bmatrix} \\tag{4} \[Ks,s​Ks,t​​Ks,s​Kt,t​​\](4)  
以及L矩阵：  
li,j={1/n2,xi,xj∈Ds1/m2,xi,xj∈Ds−1/nm,otherwise(5) l_{i,j} = \\begin{cases} 1/{n^2}, & \\text{$x\_i, x\_j\\in D\_s$} \\\ 1/{m^2}, & \\text{$x\_i, x\_j\\in D\_s$} \\\ -1/{nm},& \\text{otherwise} \\end{cases} \\tag{5} li,j​=⎩⎪⎨⎪⎧​1/n2,1/m2,−1/nm,​xi​,xj​∈Ds​xi​,xj​∈Ds​otherwise​(5)  
在实际应用中，高斯核的σ\\sigmaσ会取多个值，分别求核函数然后取和，作为最后的核函数。  
##代码解读

    import torch
    
    def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        将源域数据和目标域数据转化为核矩阵，即上文中的K
        Params: 
    	    source: 源域数据（n * len(x))
    	    target: 目标域数据（m * len(y))
    	    kernel_mul: 
    	    kernel_num: 取不同高斯核的数量
    	    fix_sigma: 不同高斯核的sigma值
    	Return:
    		sum(kernel_val): 多个核矩阵之和
        '''
        n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
        total = torch.cat([source, target], dim=0)#将source,target按列方向合并
        #将total复制（n+m）份
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        L2_distance = ((total0-total1)**2).sum(2) 
        #调整高斯核函数的sigma值
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        #高斯核函数的数学表达式
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        #得到最终的核矩阵
        return sum(kernel_val)#/len(kernel_val)
    
    def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        计算源域数据和目标域数据的MMD距离
        Params: 
    	    source: 源域数据（n * len(x))
    	    target: 目标域数据（m * len(y))
    	    kernel_mul: 
    	    kernel_num: 取不同高斯核的数量
    	    fix_sigma: 不同高斯核的sigma值
    	Return:
    		loss: MMD loss
        '''
        batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
        kernels = guassian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        #根据式（3）将核矩阵分成4部分
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss#因为一般都是n==m，所以L矩阵一般不加入计算
    

代码示例
----

为了体现以上代码的有效性，我们参考[链接](https://blog.csdn.net/llh_1178/article/details/72889279)生成了两组不同分布的数据。

    import random
    import matplotlib
    import matplotlib.pyplot as plt
    
    SAMPLE_SIZE = 500
    buckets = 50
    
    #第一种分布：对数正态分布，得到一个中值为mu，标准差为sigma的正态分布。mu可以取任何值，sigma必须大于零。
    plt.subplot(1,2,1)
    plt.xlabel("random.lognormalvariate")
    mu = -0.6
    sigma = 0.15#将输出数据限制到0-1之间
    res1 = [random.lognormvariate(mu, sigma) for _ in xrange(1, SAMPLE_SIZE)]
    plt.hist(res1, buckets)
    
    #第二种分布：beta分布。参数的条件是alpha 和 beta 都要大于0， 返回值在0~1之间。
    plt.subplot(1,2,2)
    plt.xlabel("random.betavariate")
    alpha = 1
    beta = 10
    res2 = [random.betavariate(alpha, beta) for _ in xrange(1, SAMPLE_SIZE)]
    plt.hist(res2, buckets)
    
    plt.savefig('data.jpg)
    plt.show()
    

两种数据分布如下图  
![这里写图片描述](https://img-blog.csdn.net/20180725203509767?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E1Mjk5NzUxMjU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  
两种分布有明显的差异，下面从两个方面用MMD来量化这种差异：  
**1\. 分别从不同分布取两组数据（每组为10*500）**

    from torch.autograd import Variable
    
    #参数值见上段代码
    #分别从对数正态分布和beta分布取两组数据
    diff_1 = []
    for i in range(10):
        diff_1.append([random.lognormvariate(mu, sigma) for _ in xrange(1, SAMPLE_SIZE)])
    
    diff_2 = []
    for i in range(10):
        diff_2.append([random.betavariate(alpha, beta) for _ in xrange(1, SAMPLE_SIZE)])
    
    X = torch.Tensor(diff_1)
    Y = torch.Tensor(diff_2)
    X,Y = Variable(X), Variable(Y)
    print mmd_rbf(X,Y)
    
    

输出结果为

    Variable containing:
     6.1926
    [torch.FloatTensor of size 1]
    

**2\. 分别从相同分布取两组数据（每组为10*500）**

    from torch.autograd import Variable
    
    #参数值见以上代码
    #从对数正态分布取两组数据
    same_1 = []
    for i in range(10):
        same_1.append([random.lognormvariate(mu, sigma) for _ in xrange(1, SAMPLE_SIZE)])
    
    same_2 = []
    for i in range(10):
        same_2.append([random.lognormvariate(mu, sigma) for _ in xrange(1, SAMPLE_SIZE)])
    
    X = torch.Tensor(same_1)
    Y = torch.Tensor(same_2)
    X,Y = Variable(X), Variable(Y)
    print mmd_rbf(X,Y)
    
    

输出结果为

    Variable containing:
     0.6014
    [torch.FloatTensor of size 1]
    

可以明显看出同分布数据和不同分布数据之间的差距被量化了出来，且符合之前理论所说：不同分布MMD的值大于相同分布MMD的值。  
**PS**，在实验中发现一个问题，就是取数据时要在0-1的范围内取，不然MMD就失效了。  
