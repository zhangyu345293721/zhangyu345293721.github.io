---
title: "3-1,低阶API示范"
excerpt: '下面的范例使用Pytorch的低阶API实现线性回归模型和DNN二分类模型<br/><img src="/images/pyspark.png" width="600">'
collection: portfolio
---

下面的范例使用Pytorch的低阶API实现线性回归模型和DNN二分类模型。

低阶API主要包括张量操作，计算图和自动微分。


```python
import os
import datetime

#打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

#mac系统上pytorch和matplotlib在jupyter中同时跑需要更改环境变量
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

```


```python
import torch 
print("torch.__version__="+torch.__version__) 
```

    torch.__version__=2.0.1


### 一，线性回归模型

**1，准备数据**


```python
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
import torch
from torch import nn


#样本数量
n = 400

# 生成测试用数据集
X = 10*torch.rand([n,2])-5.0  #torch.rand是均匀分布 
w0 = torch.tensor([[2.0],[-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0 + b0 + torch.normal(0.0,2.0,size = [n,1])  # @表示矩阵乘法,增加正态扰动



```


```python
# 数据可视化

%matplotlib inline
%config InlineBackend.figure_format = 'svg'

plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0].numpy(),Y[:,0].numpy(), c = "b",label = "samples")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)

ax2 = plt.subplot(122)
ax2.scatter(X[:,1].numpy(),Y[:,0].numpy(), c = "g",label = "samples")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)
plt.show()


```



```python
# 构建数据管道迭代器
def data_iter(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  #样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        indexs = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield  features.index_select(0, indexs), labels.index_select(0, indexs)
        
# 测试数据管道效果   
batch_size = 8
(features,labels) = next(data_iter(X,Y,batch_size))
print(features)
print(labels)


```

    tensor([[-0.3932, -1.2790],
            [-0.4021, -2.1115],
            [-1.7178,  0.9134],
            [-0.6046, -2.1865],
            [-2.2676, -1.0583],
            [-3.7235, -2.7356],
            [ 0.4728, -1.0100],
            [ 3.9323,  3.2088]])
    tensor([[12.8721],
            [14.5131],
            [ 5.9134],
            [10.7377],
            [11.9112],
            [ 9.5725],
            [13.4364],
            [ 6.4052]])


**2，定义模型**


```python
# 定义模型
class LinearRegression: 
    
    def __init__(self):
        self.w = torch.randn_like(w0,requires_grad=True)
        self.b = torch.zeros_like(b0,requires_grad=True)
        
    #正向传播
    def forward(self,x): 
        return x@self.w + self.b

    # 损失函数
    def loss_fn(self,y_pred,y_true):  
        return torch.mean((y_pred - y_true)**2/2)

model = LinearRegression()

```


```python

```

**3，训练模型**


```python
def train_step(model, features, labels):
    
    predictions = model.forward(features)
    loss = model.loss_fn(predictions,labels)
        
    # 反向传播求梯度
    loss.backward()
    
    # 使用torch.no_grad()避免梯度记录，也可以通过操作 model.w.data 实现避免梯度记录 
    with torch.no_grad():
        # 梯度下降法更新参数
        model.w -= 0.001*model.w.grad
        model.b -= 0.001*model.b.grad

        # 梯度清零
        model.w.grad.zero_()
        model.b.grad.zero_()
    return loss
 
    
    
```


```python
# 测试train_step效果
batch_size = 10
(features,labels) = next(data_iter(X,Y,batch_size))
train_step(model,features,labels)

```




    tensor(68.1951, grad_fn=<MeanBackward0>)




```python
def train_model(model,epochs):
    for epoch in range(1,epochs+1):
        for features, labels in data_iter(X,Y,10):
            loss = train_step(model,features,labels)

        if epoch%20==0:
            printbar()
            print("epoch =",epoch,"loss = ",loss.item())
            print("model.w =",model.w.data)
            print("model.b =",model.b.data)

train_model(model,epochs = 200)

```

    
    ================================================================================2023-08-02 14:54:50
    epoch = 20 loss =  14.40532398223877
    model.w = tensor([[ 1.9802],
            [-2.9087]])
    model.b = tensor([[5.4715]])
    
    ================================================================================2023-08-02 14:54:50
    epoch = 40 loss =  4.50400972366333
    model.w = tensor([[ 1.9678],
            [-2.9619]])
    model.b = tensor([[7.9420]])
    
    ================================================================================2023-08-02 14:54:50
    epoch = 60 loss =  1.308245062828064
    model.w = tensor([[ 1.9609],
            [-2.9882]])
    model.b = tensor([[9.0544]])
    
    ================================================================================2023-08-02 14:54:50
    epoch = 80 loss =  2.0600852966308594
    model.w = tensor([[ 1.9560],
            [-2.9965]])
    model.b = tensor([[9.5551]])
    
    ================================================================================2023-08-02 14:54:50
    epoch = 100 loss =  3.130641460418701
    model.w = tensor([[ 1.9550],
            [-3.0019]])
    model.b = tensor([[9.7809]])
    
    ================================================================================2023-08-02 14:54:50
    epoch = 120 loss =  3.818127393722534
    model.w = tensor([[ 1.9550],
            [-3.0040]])
    model.b = tensor([[9.8821]])
    
    ================================================================================2023-08-02 14:54:50
    epoch = 140 loss =  2.4734010696411133
    model.w = tensor([[ 1.9548],
            [-3.0032]])
    model.b = tensor([[9.9277]])
    
    ================================================================================2023-08-02 14:54:50
    epoch = 160 loss =  3.3532516956329346
    model.w = tensor([[ 1.9555],
            [-3.0047]])
    model.b = tensor([[9.9481]])
    
    ================================================================================2023-08-02 14:54:50
    epoch = 180 loss =  1.9561536312103271
    model.w = tensor([[ 1.9546],
            [-3.0075]])
    model.b = tensor([[9.9573]])
    
    ================================================================================2023-08-02 14:54:50
    epoch = 200 loss =  1.7864179611206055
    model.w = tensor([[ 1.9534],
            [-3.0047]])
    model.b = tensor([[9.9618]])



```python

```


```python
# 结果可视化

%matplotlib inline
%config InlineBackend.figure_format = 'svg'

plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0].numpy(),Y[:,0].numpy(), c = "b",label = "samples")
ax1.plot(X[:,0].numpy(),(model.w[0].data*X[:,0]+model.b[0].data).numpy(),"-r",linewidth = 5.0,label = "model")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)


ax2 = plt.subplot(122)
ax2.scatter(X[:,1].numpy(),Y[:,0].numpy(), c = "g",label = "samples")
ax2.plot(X[:,1].numpy(),(model.w[1].data*X[:,1]+model.b[0].data).numpy(),"-r",linewidth = 5.0,label = "model")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)

plt.show()

```


    
![svg](output_16_0.svg)
    



```python

```

### 二，DNN二分类模型


```python

```

**1，准备数据**


```python
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import torch
from torch import nn
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

#正负样本数量
n_positive,n_negative = 2000,2000

#生成正样本, 小圆环分布
r_p = 5.0 + torch.normal(0.0,1.0,size = [n_positive,1]) 
theta_p = 2*np.pi*torch.rand([n_positive,1])
Xp = torch.cat([r_p*torch.cos(theta_p),r_p*torch.sin(theta_p)],axis = 1)
Yp = torch.ones_like(r_p)

#生成负样本, 大圆环分布
r_n = 8.0 + torch.normal(0.0,1.0,size = [n_negative,1]) 
theta_n = 2*np.pi*torch.rand([n_negative,1])
Xn = torch.cat([r_n*torch.cos(theta_n),r_n*torch.sin(theta_n)],axis = 1)
Yn = torch.zeros_like(r_n)

#汇总样本
X = torch.cat([Xp,Xn],axis = 0)
Y = torch.cat([Yp,Yn],axis = 0)


#可视化
plt.figure(figsize = (6,6))
plt.scatter(Xp[:,0].numpy(),Xp[:,1].numpy(),c = "r")
plt.scatter(Xn[:,0].numpy(),Xn[:,1].numpy(),c = "g")
plt.legend(["positive","negative"]);

```

    



```python
# 构建数据管道迭代器
def data_iter(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  #样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        indexs = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield  features.index_select(0, indexs), labels.index_select(0, indexs)
        
# 测试数据管道效果   
batch_size = 8
(features,labels) = next(data_iter(X,Y,batch_size))
print(features)
print(labels)

```

    tensor([[-2.9355, -2.8420],
            [-3.9011, -2.1900],
            [-7.0753,  3.2132],
            [-7.0655,  3.3173],
            [ 3.6206, -5.8906],
            [-1.6916,  3.5855],
            [ 1.7623,  3.0227],
            [ 8.6543, -1.9813]])
    tensor([[1.],
            [1.],
            [0.],
            [0.],
            [0.],
            [1.],
            [1.],
            [0.]])



```python

```

**2，定义模型**

此处范例我们利用nn.Module来组织模型变量。


```python
class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.w1 = nn.Parameter(torch.randn(2,4))
        self.b1 = nn.Parameter(torch.zeros(1,4))
        self.w2 = nn.Parameter(torch.randn(4,8))
        self.b2 = nn.Parameter(torch.zeros(1,8))
        self.w3 = nn.Parameter(torch.randn(8,1))
        self.b3 = nn.Parameter(torch.zeros(1,1))

    # 正向传播
    def forward(self,x):
        x = torch.relu(x@self.w1 + self.b1)
        x = torch.relu(x@self.w2 + self.b2)
        y = torch.sigmoid(x@self.w3 + self.b3)
        return y
    
    # 损失函数(二元交叉熵)
    def loss_fn(self,y_pred,y_true):  
        #将预测值限制在1e-7以上, 1- (1e-7)以下，避免log(0)错误
        eps = 1e-7
        y_pred = torch.clamp(y_pred,eps,1.0-eps)
        bce = - y_true*torch.log(y_pred) - (1-y_true)*torch.log(1-y_pred)
        return torch.mean(bce)
    
    # 评估指标(准确率)
    def metric_fn(self,y_pred,y_true):
        y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                          torch.zeros_like(y_pred,dtype = torch.float32))
        acc = torch.mean(1-torch.abs(y_true-y_pred))
        return acc
    
model = DNNModel()

```


```python
# 测试模型结构
batch_size = 10
(features,labels) = next(data_iter(X,Y,batch_size))

predictions = model(features)

loss = model.loss_fn(labels,predictions)
metric = model.metric_fn(labels,predictions)

print("init loss:", loss.item())
print("init metric:", metric.item())

```

    init loss: 11.617703437805176
    init metric: 0.2786138653755188



```python
len(list(model.parameters()))
```




    6



**3，训练模型**


```python
def train_step(model, features, labels):   
    
    # 正向传播求损失
    predictions = model.forward(features)
    loss = model.loss_fn(predictions,labels)
    metric = model.metric_fn(predictions,labels)
        
    # 反向传播求梯度
    loss.backward()
    
    # 梯度下降法更新参数
    for param in model.parameters():
        #注意是对param.data进行重新赋值,避免此处操作引起梯度记录
        param.data = (param.data - 0.01*param.grad.data) 
        
    # 梯度清零
    model.zero_grad()
        
    return loss.item(),metric.item()
 

def train_model(model,epochs):
    for epoch in range(1,epochs+1):
        loss_list,metric_list = [],[]
        for features, labels in data_iter(X,Y,20):
            lossi,metrici = train_step(model,features,labels)
            loss_list.append(lossi)
            metric_list.append(metrici)
        loss = np.mean(loss_list)
        metric = np.mean(metric_list)

        if epoch%10==0:
            printbar()
            print("epoch =",epoch,"loss = ",loss,"metric = ",metric)
        
train_model(model,epochs = 100)

```

    
    ================================================================================2023-08-02 14:55:17
    epoch = 10 loss =  0.4736742579936981 metric =  0.7737499994039535
    
    ================================================================================2023-08-02 14:55:18
    epoch = 20 loss =  0.3228449109941721 metric =  0.8784999975562096
    
    ================================================================================2023-08-02 14:55:18
    epoch = 30 loss =  0.22225805193185807 metric =  0.9099999928474426
    
    ================================================================================2023-08-02 14:55:19
    epoch = 40 loss =  0.2106029569543898 metric =  0.9162499943375587
    
    ================================================================================2023-08-02 14:55:19
    epoch = 50 loss =  0.20301875596866012 metric =  0.9174999958276748
    
    ================================================================================2023-08-02 14:55:20
    epoch = 60 loss =  0.20313117776066064 metric =  0.9179999929666519
    
    ================================================================================2023-08-02 14:55:20
    epoch = 70 loss =  0.19921660327352583 metric =  0.914999993443489
    
    ================================================================================2023-08-02 14:55:20
    epoch = 80 loss =  0.195224122479558 metric =  0.919999991953373
    
    ================================================================================2023-08-02 14:55:21
    epoch = 90 loss =  0.19632970970124006 metric =  0.9224999934434891
    
    ================================================================================2023-08-02 14:55:21
    epoch = 100 loss =  0.19567020332440735 metric =  0.9209999939799309



```python
# 结果可视化
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize = (12,5))
ax1.scatter(Xp[:,0],Xp[:,1], c="r")
ax1.scatter(Xn[:,0],Xn[:,1],c = "g")
ax1.legend(["positive","negative"]);
ax1.set_title("y_true");

Xp_pred = X[torch.squeeze(model.forward(X)>=0.5)]
Xn_pred = X[torch.squeeze(model.forward(X)<0.5)]

ax2.scatter(Xp_pred[:,0],Xp_pred[:,1],c = "r")
ax2.scatter(Xn_pred[:,0],Xn_pred[:,1],c = "g")
ax2.legend(["positive","negative"]);
ax2.set_title("y_pred");

```


