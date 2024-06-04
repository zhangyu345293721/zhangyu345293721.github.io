## 导入pandas库和numpy库


```python
import pandas as pd
import numpy as np
```

## 1.创建Series

（1）通过列表list创建Series


```python
s = pd.Series([10,30,20,40])
s
```




    0    10
    1    30
    2    20
    3    40
    dtype: int64



（2）通过字典dict创建Series


```python
dict_1 = {"a":10,"c":5,"b":40}
s1 = pd.Series(dict_1)
s1
```




    a    10
    b    40
    c     5
    dtype: int64



（3）通过数组array创建Series


```python
array_1 = np.arange(10,16)
s2 = pd.Series(array_1,index=list("abcdef"))
s2
```




    a    10
    b    11
    c    12
    d    13
    e    14
    f    15
    dtype: int32



## 2.Series的属性

（1）获得索引index


```python
s2.index
```




    Index(['a', 'b', 'c', 'd', 'e', 'f'], dtype='object')



（2）通过赋值整体地修改索引值


```python
s2.index = ["aa","bb","cc","dd","eee","fff"]
s2
```




    aa     10
    bb     11
    cc     12
    dd     13
    eee    14
    fff    15
    dtype: int32



（3）修改index的名称


```python
s2.index.name = "banana"
s2
```




    banana
    aa     10
    bb     11
    cc     12
    dd     13
    eee    14
    fff    15
    dtype: int32



（4）修改Series的名称


```python
s2.name = "length"
s2
```




    banana
    aa     10
    bb     11
    cc     12
    dd     13
    eee    14
    fff    15
    Name: length, dtype: int32



（5）获取Series的值values


```python
s2.values
```




    array([10, 11, 12, 13, 14, 15])



## 3.Series的索引 index

（1）位置索引


```python
#得到第一行的数
s2[0] 
```




    10




```python
#得到最后一行的数
s2[-1]  
```




    15




```python
#得到特定一些行的数(如第1行，第4行，第6行）
s2[[0,3,5]] 
```




    banana
    aa     10
    dd     13
    fff    15
    Name: length, dtype: int32



（2）名称索引


```python
#得到索引为aa所对应的数
s2["aa"]
```




    10




```python
#得到特定一些索引所对应的数
s2[["aa","cc","fff"]]
```




    banana
    aa     10
    cc     12
    fff    15
    Name: length, dtype: int32



（3）点索引法

+ 对象不重名的情况


```python
s2.aa
```




    10



+ 对象重名的情况


```python
s2.index = ["aa","bb","cc","dd","eee","def"]
```


```python
s2.def
```


      File "<ipython-input-54-6fdba724b8e8>", line 1
        s2.def
             ^
    SyntaxError: invalid syntax




```python
print(s2[5])
print(s2["def"])
```

    15
    15


## 4.Series的切片slice

（1）索引位置切片

不包含末端


```python
s2
```




    aa     10
    bb     11
    cc     12
    dd     13
    eee    14
    def    15
    Name: length, dtype: int32




```python
s2[1:4]
```




    bb    11
    cc    12
    dd    13
    Name: length, dtype: int32



（2）索引名称切片

包含末端


```python
s2["aa":"eee"]
```




    aa     10
    bb     11
    cc     12
    dd     13
    eee    14
    Name: length, dtype: int32



## 5.修改Series的值


```python
s2["aa"] = 100
s2[2] = 120
s2
```




    aa     100
    bb      11
    cc     120
    dd      13
    eee     14
    def     15
    Name: length, dtype: int32



## 6.添加Series的值

（1）通过append来添加


```python
s2
```




    aa     100
    bb      11
    cc     120
    dd      13
    eee     14
    def     15
    Name: length, dtype: int32




```python
s2.append(pd.Series([50,60],index=["a1","a2"]))
```




    aa     100
    bb      11
    cc     120
    dd      13
    eee     14
    def     15
    a1      50
    a2      60
    dtype: int64




```python
# 原来的s2并没有变化
s2
```




    aa     100
    bb      11
    cc     120
    dd      13
    eee     14
    def     15
    Name: length, dtype: int32



（2）通过方括号[ ]来添加值


```python
s2
```




    aa     100
    bb      11
    cc     120
    dd      13
    eee     14
    def     15
    Name: length, dtype: int32




```python
s2["y"] = 99
s2
```




    aa     100
    bb      11
    cc     120
    dd      13
    eee     14
    def     15
    y       99
    Name: length, dtype: int64



## 7.删除Series的值

del删除法


```python
#删除y索引对应的99这个值
del s2["y"]
s2
```




    aa     100
    bb      11
    cc     120
    dd      13
    eee     14
    def     15
    Name: length, dtype: int64



## 8.过滤Series的值

单条件筛选


```python
s2[s2 > 90]
```




    aa    100
    cc    120
    Name: length, dtype: int64




```python
s2[s2 == 13]
```




    dd    13
    Name: length, dtype: int64



多条件筛选


```python
s2
```




    aa     100
    bb      11
    cc     120
    dd      13
    eee     14
    def     15
    Name: length, dtype: int64




```python
s2[(s2 > 50) | (s2 < 14)]
```




    aa    100
    bb     11
    cc    120
    dd     13
    Name: length, dtype: int64



## 9.Series的缺失值处理


```python
#创建一个带有缺失值的Series
s = pd.Series([10,np.nan,15,19,None])
s
```




    0    10.0
    1     NaN
    2    15.0
    3    19.0
    4     NaN
    dtype: float64



（1）判断是否有缺失值

isnull()


```python
s.isnull()
```




    0    False
    1     True
    2    False
    3    False
    4     True
    dtype: bool




```python
#如果需要取出这些缺失值，则通过布尔选择器来筛选出来
s[s.isnull()]
```




    1   NaN
    4   NaN
    dtype: float64



（2）删除缺失值

dropna()


```python
#dropna()会删除掉所有缺失值NaN，并返回一个新的Series
s.dropna()
```




    0    10.0
    2    15.0
    3    19.0
    dtype: float64




```python
#原有的Series并未发生改变
s
```




    0    10.0
    1     NaN
    2    15.0
    3    19.0
    4     NaN
    dtype: float64




```python
#如果希望原有的Series发生改变，可以将s.dropna（）返回的新Series直接赋值给原来的Series
s = s.dropna()
s
```




    0    10.0
    2    15.0
    3    19.0
    dtype: float64



通过过滤的方式来达到一样的删除效果


```python
s = pd.Series([10,np.nan,15,19,None]) #初始化一下s
s[~s.isnull()]  #依然是返回一个新的Series，波浪号~表示否定、非的意思
```




    0    10.0
    2    15.0
    3    19.0
    dtype: float64




```python
#s并未改变
s
```




    0    10.0
    1     NaN
    2    15.0
    3    19.0
    4     NaN
    dtype: float64




```python
#通过notnull（）也能实现，同样也是返回一个新的Series
s[s.notnull()]
```




    0    10.0
    2    15.0
    3    19.0
    dtype: float64



（3）填充缺失值

fillna()

+ 用指定值填充缺失值


```python
s
```




    0    10.0
    1     NaN
    2    15.0
    3    19.0
    4     NaN
    dtype: float64




```python
#用0填充缺失值，依然返回的是一个新的Series
s.fillna(value=0)
```




    0    10.0
    1     0.0
    2    15.0
    3    19.0
    4     0.0
    dtype: float64




```python
#s没有改变
s
```




    0    10.0
    1     NaN
    2    15.0
    3    19.0
    4     NaN
    dtype: float64




```python
#如果希望直接修改原Series，一种方法是之前说的直接赋值，另一种是添加参数inplace=True
s.fillna(value=0,inplace=True)
```


```python
s
```




    0    10.0
    1     0.0
    2    15.0
    3    19.0
    4     0.0
    dtype: float64



+ 通过插值填充缺失值


```python
#初始化一下s
s = pd.Series([10,np.nan,15,19,None])
s
```




    0    10.0
    1     NaN
    2    15.0
    3    19.0
    4     NaN
    dtype: float64



向前填充（ffill，全称是front fill）


```python
s.fillna(method="ffill")
```




    0    10.0
    1    10.0
    2    15.0
    3    19.0
    4    19.0
    dtype: float64



向后填充（bfill，全称是back fill）


```python
s.fillna(method="bfill")
```




    0    10.0
    1    15.0
    2    15.0
    3    19.0
    4     NaN
    dtype: float64



## 10.排序


```python
#创建一个Series
s3 = pd.Series([10,15,8,4,20],index=list("gadkb"))
s3
```




    g    10
    a    15
    d     8
    k     4
    b    20
    dtype: int64



（1）根据索引排序 sort_index()

默认升序排列


```python
s3.sort_index()
```




    a    15
    b    20
    d     8
    g    10
    k     4
    dtype: int64



降序排列


```python
s3.sort_index(ascending=False)
```




    k     4
    g    10
    d     8
    b    20
    a    15
    dtype: int64



（2）根据值排序 sort_values()

默认升序排列


```python
s3.sort_values()
```




    k     4
    d     8
    g    10
    a    15
    b    20
    dtype: int64



降序排列


```python
s3.sort_values(ascending=False)
```




    b    20
    a    15
    g    10
    d     8
    k     4
    dtype: int64



## 11.排名


```python
s4 = pd.Series([2,5,15,7,1,2])
s4
```




    0     2
    1     5
    2    15
    3     7
    4     1
    5     2
    dtype: int64



中国式排名


```python
s4.rank?
```


```python
s4.rank(ascending=False,method="dense")
```




    0    4.0
    1    3.0
    2    1.0
    3    2.0
    4    5.0
    5    4.0
    dtype: float64



## 12.Series的描述性统计


```python
#创建一个Series
s5 = pd.Series([100,50,100,75,24,100])
s5
```




    0    100
    1     50
    2    100
    3     75
    4     24
    5    100
    dtype: int64



+ 值的计数


```python
s5.value_counts()
```




    100    3
    75     1
    50     1
    24     1
    dtype: int64



+ 最小值


```python
s5.min()
```




    24



+ 最大值


```python
s5.max()
```




    100



+ 中位数


```python
s5.median()
```




    87.5



+ 均值


```python
s5.mean()
```




    74.83333333333333



+ 求和


```python
s5.sum()
```




    449



+ 标准差


```python
s5.std()
```




    31.940048006643114



+ 描述性统计


```python
s5.describe().round(1)
```




    count      6.0
    mean      74.8
    std       31.9
    min       24.0
    25%       56.2
    50%       87.5
    75%      100.0
    max      100.0
    dtype: float64



## 13.Series的向量化运算


```python
s5
```




    0    100
    1     50
    2    100
    3     75
    4     24
    5    100
    dtype: int64




```python
s5 + 1000
```




    0    1100
    1    1050
    2    1100
    3    1075
    4    1024
    5    1100
    dtype: int64




```python
s5 - 2000
```




    0   -1900
    1   -1950
    2   -1900
    3   -1925
    4   -1976
    5   -1900
    dtype: int64




```python
s5 * 2
```




    0    200
    1    100
    2    200
    3    150
    4     48
    5    200
    dtype: int64




```python
s5 / 10
```




    0    10.0
    1     5.0
    2    10.0
    3     7.5
    4     2.4
    5    10.0
    dtype: float64



自动对齐相同索引的数据，不同索引的数据对不上，则显示NaN


```python
s6 = pd.Series([35000,40000,71000,5500],index=list("abcd"))
s7 = pd.Series([222,35000,4000,2222],index=list("aqtb"))
s6 + s7
```




    a    35222.0
    b    42222.0
    c        NaN
    d        NaN
    q        NaN
    t        NaN
    dtype: float64


