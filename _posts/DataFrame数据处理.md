---
title: 'DataFrame—数据处理'
date: 2022-04-02
permalink: /posts/DataFrame数据处理
tags:
  - cool posts
  - category1
  - category2
---

导入pandas和numpy


```python
import pandas as pd
import numpy as np
```

## 一.数据的增删改

### 1.增加行

（1）手动输入新增行的内容


```python
#示例数据
df1 = pd.DataFrame({"name":["ray","jack","lucy","bob","candy"],
                    "city":["hangzhou","beijing","hangzhou","chengdu","suzhou"],
                    "score":[10,30,20,15,50]},
                  columns=["name","city","score"])
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



如何手动添加行？


```python
df1.loc[5] = ["baby","shanghai",80]
```


```python
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>baby</td>
      <td>shanghai</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
</div>



（2）将同字段的DataFrame添加进来


```python
#示例数据
df1 = pd.DataFrame({"name":["ray","jack","lucy","bob","candy"],
                    "city":["hangzhou","beijing","hangzhou","chengdu","suzhou"],
                    "score":[10,30,20,15,50]},
                  columns=["name","city","score"])
df1_1 = pd.DataFrame({"name":["faker","lucy"],
                    "city":["guangzhou","shenzhen"],
                    "score":[70,75]},
                  columns=["name","city","score"])
```


```python
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1_1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>faker</td>
      <td>guangzhou</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lucy</td>
      <td>shenzhen</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>



如何将同字段的DataFrame增加到原DataFrame中呢？


```python
#如果直接添加进来，索引号不会顺接上去
df1.append(df1_1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
    <tr>
      <th>0</th>
      <td>faker</td>
      <td>guangzhou</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lucy</td>
      <td>shenzhen</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.append?
```


```python
#正确的写法如下，这样索引号就顺接上去了
df1.append(df1_1,ignore_index=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>faker</td>
      <td>guangzhou</td>
      <td>70</td>
    </tr>
    <tr>
      <th>6</th>
      <td>lucy</td>
      <td>shenzhen</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>



还有一种做法，通过concat拼接同字段的DataFrame


```python
pd.concat([df1,df1_1],ignore_index=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>faker</td>
      <td>guangzhou</td>
      <td>70</td>
    </tr>
    <tr>
      <th>6</th>
      <td>lucy</td>
      <td>shenzhen</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>



### 2.删除行


```python
### 示例数据
df_concat = pd.concat([df1,df1_1],ignore_index=True)
df_concat
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>faker</td>
      <td>guangzhou</td>
      <td>70</td>
    </tr>
    <tr>
      <th>6</th>
      <td>lucy</td>
      <td>shenzhen</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>



如何删除行？


```python
#删除第7行，也即是索引号为6的这一行
df_concat.drop(6,inplace=True)
```


```python
df_concat
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>faker</td>
      <td>guangzhou</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
</div>




```python
#删除第4行和第6行
df_concat.drop([3,5])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



### 3.修改行

若要修改行，则要先选出需要修改的一行或多行,再重新赋值


```python
#示例数据
df1 = pd.DataFrame({"name":["ray","jack","lucy","bob","candy"],
                    "city":["hangzhou","beijing","hangzhou","chengdu","suzhou"],
                    "score":[10,30,20,15,50]},
                  columns=["name","city","score"])
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



如何将第一行的ray修改成demon，hangzhou改成wenzhou，10改成35？


```python
df1.loc[0] = ["demon","hangzhou",35]
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>demon</td>
      <td>hangzhou</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



如何进行多行修改？


```python
df1.loc[0:2] = [["d","j","l"],["h","b","h"],[40,50,60]]
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>d</td>
      <td>j</td>
      <td>l</td>
    </tr>
    <tr>
      <th>1</th>
      <td>h</td>
      <td>b</td>
      <td>h</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40</td>
      <td>50</td>
      <td>60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



### 4.增加列

（1）在末尾插入列


```python
#示例数据
df1 = pd.DataFrame({"name":["ray","jack","lucy","bob","candy"],
                    "city":["hangzhou","beijing","hangzhou","chengdu","suzhou"],
                    "score":[10,30,20,15,50]},
                  columns=["name","city","score"])
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



如何末尾增加一列：gender（性别）


```python
df1["gender"] = ["male","male","female","male","female"]
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
      <td>male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
      <td>female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
      <td>male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
      <td>female</td>
    </tr>
  </tbody>
</table>
</div>



（2）在任意位置插入新列

我希望在第2列的位置插入新的一列：height（身高）


```python
df1.insert(1,"height",[170,165,172,180,169])  #第1个参数1表示索引号即插入的位置，第2个参数填列的名称，第3个参数填值
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>height</th>
      <th>city</th>
      <th>score</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>170</td>
      <td>hangzhou</td>
      <td>10</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>165</td>
      <td>beijing</td>
      <td>30</td>
      <td>male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>172</td>
      <td>hangzhou</td>
      <td>20</td>
      <td>female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>180</td>
      <td>chengdu</td>
      <td>15</td>
      <td>male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>169</td>
      <td>suzhou</td>
      <td>50</td>
      <td>female</td>
    </tr>
  </tbody>
</table>
</div>



### 5.删除列


```python
#示例数据
df1 = pd.DataFrame({"name":["ray","jack","lucy","bob","candy"],
                    "city":["hangzhou","beijing","hangzhou","chengdu","suzhou"],
                    "score":[10,30,20,15,50]},
                  columns=["name","city","score"])
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



（1）del DataFrame["colname"]


```python
del df1["score"]
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
    </tr>
  </tbody>
</table>
</div>



（2）DataFrame.drop（["colname"]，axis = 1）

先重新运行下生成df1的式子，初始化df1


```python
df1.drop(["city"],axis=1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



### 6.修改列

若要修改列，则要先选出需要修改的一列或多列,再重新赋值


```python
#示例数据
df1 = pd.DataFrame({"name":["ray","jack","lucy","bob","candy"],
                    "city":["hangzhou","beijing","hangzhou","chengdu","suzhou"],
                    "score":[10,30,20,15,50]},
                  columns=["name","city","score"])
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



修改score列


```python
df1["score"] = 50
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



修改city和score列


```python
df1[["city","score"]] = [["hz","bj","hz","cd","sz"],60]
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hz</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>bj</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hz</td>
      <td>60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>cd</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>sz</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>



## 二.数据集的合并

如何merge？

+ **将列作为键合并**

示例数据


```python
df1 = pd.DataFrame({"name":["ray","jack","lucy","bob","candy"],
                    "city":["hangzhou","beijing","hangzhou","chengdu","suzhou"],
                    "score":[10,30,20,15,50]},
                  columns=["name","city","score"])

df2 = pd.DataFrame({"name":["ray","lucy","demon"],
                   "age":[15,17,16]},
                  columns=["name","age"])
```


```python
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lucy</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>demon</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



（1）inner连接（交集）


```python
pd.merge(df1,df2)   #默认连接方式是交集；若没有指定，则默认将重叠列的列名作为键
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(df1,df2,how="inner")   #也可以显示指定连接方式为inner，等价于不填参数how="inner"
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(df1,df2,on="name")   #也可以显式地指定键为“name”列
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
#因此完整地写法是
pd.merge(df1,df2,on="name",how="inner")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>



（2）outer连接（并集）


```python
pd.merge(df1,df2,on="name",how="outer")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>demon</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.0</td>
    </tr>
  </tbody>
</table>
</div>



（3）left连接（保左加右）


```python
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lucy</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>demon</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(df1,df2,on="name",how="left")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



（4）right连接（保右加左）


```python
pd.merge(df1,df2,on="name",how="right")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10.0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20.0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>demon</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



一些可能遇到的问题：

Q1:如果两个数据指定列的列名不一样怎么办？


```python
df3 = df2.rename(columns={"name":"name2"})
df3
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name2</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lucy</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>demon</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(df1,df3,left_on="name",right_on="name2",how="inner")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
      <th>name2</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
      <td>ray</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
      <td>lucy</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>



Q2:如果需要多个键来进行合并怎么办呢？


```python
#给df1增加新的一行，名称为已出现的ray
df1.loc[5] = ["ray","wuhan",80]
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ray</td>
      <td>wuhan</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
</div>




```python
#给df2增加新的一列city
df2["city"] = ["hangzhou","hangzhou","heilongjiang"]
df2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>15</td>
      <td>hangzhou</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lucy</td>
      <td>17</td>
      <td>hangzhou</td>
    </tr>
    <tr>
      <th>2</th>
      <td>demon</td>
      <td>16</td>
      <td>heilongjiang</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(df1,df2,on=["name","city"],how="left")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ray</td>
      <td>wuhan</td>
      <td>80</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



+ **将索引作为键来合并**


```python
#示例数据
left1 = pd.DataFrame({"key":list("acba"),"value":range(4)})
right1 = pd.DataFrame({"value2":[10,20]},index=["a","b"])
```


```python
left1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
right1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>10</td>
    </tr>
    <tr>
      <th>b</th>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



合并left1和right1的交集


```python
pd.merge(left1,right1,left_on="key",right_index=True,how="inner")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>value</th>
      <th>value2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>3</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>2</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



## 三.数据的轴向连接

axis=0：表示在横轴上工作，所谓横轴也即是行，而行的方向是上下，因此你可以理解为在上下方向执行操作

axis=1：表示在纵轴上工作，所谓纵轴也即是列，而列的方向是左右，因此你可以理解为在左右方向直行操作

那么数据的轴向连接也就是指：当axis=0时，将两份或多份数据按照上下方向拼接起来；当axis=1时，将两份或多份数据按照左右方向拼接起来。

（1）横轴上的连接，axis=0时（concat默认axis=0）

+ 两份数据的字段完全相同的情况:


```python
#示例数据
df1 = pd.DataFrame({"name":["ray","jack","lucy","bob","candy"],
                    "city":["hangzhou","beijing","hangzhou","chengdu","suzhou"],
                    "score":[10,30,20,15,50]},
                  columns=["name","city","score"])
df2 = pd.DataFrame({"name":["faker","fizz"],
                    "city":["wenzhou","shanghai"],
                    "score":[55,80]},
                  columns=["name","city","score"])
```


```python
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>faker</td>
      <td>wenzhou</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fizz</td>
      <td>shanghai</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
</div>



按横轴连接df1和df2


```python
pd.concat([df1,df2],ignore_index=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>faker</td>
      <td>wenzhou</td>
      <td>55</td>
    </tr>
    <tr>
      <th>6</th>
      <td>fizz</td>
      <td>shanghai</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
</div>



+ 两份数据的字段存在不同的情况下：


```python
#示例数据
df1 = pd.DataFrame({"name":["ray","jack","lucy","bob","candy"],
                    "city":["hangzhou","beijing","hangzhou","chengdu","suzhou"],
                    "score":[10,30,20,15,50]},
                  columns=["name","city","score"])
df2 = pd.DataFrame({"name":["faker","fizz"],
                    "city":["wenzhou","shanghai"],
                    "gender":["male","female"]},
                  columns=["name","city","gender"])
```


```python
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>faker</td>
      <td>wenzhou</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fizz</td>
      <td>shanghai</td>
      <td>female</td>
    </tr>
  </tbody>
</table>
</div>



按横轴连接df1和df2


```python
pd.concat([df1,df2],ignore_index=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>gender</th>
      <th>name</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hangzhou</td>
      <td>NaN</td>
      <td>ray</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>beijing</td>
      <td>NaN</td>
      <td>jack</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hangzhou</td>
      <td>NaN</td>
      <td>lucy</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>chengdu</td>
      <td>NaN</td>
      <td>bob</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>suzhou</td>
      <td>NaN</td>
      <td>candy</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>wenzhou</td>
      <td>male</td>
      <td>faker</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>shanghai</td>
      <td>female</td>
      <td>fizz</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



会得到这两份数据的并集，没有的值会以NaN的方式填充

+ 在连接轴上创建一个层次化索引


```python
df_concat = pd.concat([df1,df2],keys=["df1","df2"])
df_concat
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>city</th>
      <th>gender</th>
      <th>name</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">df1</th>
      <th>0</th>
      <td>hangzhou</td>
      <td>NaN</td>
      <td>ray</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>beijing</td>
      <td>NaN</td>
      <td>jack</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hangzhou</td>
      <td>NaN</td>
      <td>lucy</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>chengdu</td>
      <td>NaN</td>
      <td>bob</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>suzhou</td>
      <td>NaN</td>
      <td>candy</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">df2</th>
      <th>0</th>
      <td>wenzhou</td>
      <td>male</td>
      <td>faker</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>shanghai</td>
      <td>female</td>
      <td>fizz</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



当要访问df1或df2时，可以从这个合并的数据集里提取


```python
#访问df2
df_concat.loc["df2"]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>gender</th>
      <th>name</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>wenzhou</td>
      <td>male</td>
      <td>faker</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>shanghai</td>
      <td>female</td>
      <td>fizz</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#进一步访问df2中的第2行
df_concat.loc["df2"].loc[1]
#返回的是Series
```




    city      shanghai
    gender      female
    name          fizz
    score          NaN
    Name: 1, dtype: object




```python
#进一步访问df2中的第2行
df_concat.loc["df2"].loc[[1]]
#返回的是DataFrame
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>gender</th>
      <th>name</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>shanghai</td>
      <td>female</td>
      <td>fizz</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



（2）纵轴上的连接，axis=1时


```python
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>faker</td>
      <td>wenzhou</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fizz</td>
      <td>shanghai</td>
      <td>female</td>
    </tr>
  </tbody>
</table>
</div>



按纵轴方向合并df1和df2


```python
pd.concat([df1,df2],axis=1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>city</th>
      <th>score</th>
      <th>name</th>
      <th>city</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>hangzhou</td>
      <td>10</td>
      <td>faker</td>
      <td>wenzhou</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>beijing</td>
      <td>30</td>
      <td>fizz</td>
      <td>shanghai</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>hangzhou</td>
      <td>20</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>chengdu</td>
      <td>15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>suzhou</td>
      <td>50</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## 四.合并重叠数据


```python
#示例数据
data1 = pd.DataFrame({"score":[60,np.nan,75,80],
                     "level":[np.nan,"a",np.nan,"f"],
                    "cost":[1000,1500,np.nan,1200]})
data2 = pd.DataFrame({"score":[34,58,np.nan],
                    "level":[np.nan,"c","s"]})
```


```python
data1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cost</th>
      <th>level</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000.0</td>
      <td>NaN</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1500.0</td>
      <td>a</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1200.0</td>
      <td>f</td>
      <td>80.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>level</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data1.combine_first(data2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cost</th>
      <th>level</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000.0</td>
      <td>NaN</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1500.0</td>
      <td>a</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>s</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1200.0</td>
      <td>f</td>
      <td>80.0</td>
    </tr>
  </tbody>
</table>
</div>



data1和data2有索引重叠的部分：即level列和score列的前三行。那么对于data1中的数据，如果data1已有数据，则继续用data1的数据，如果data1中有缺失数据，那么对于缺失数据用参数里的对象data2中的对应值来补充

## 五.数据分组


```python
#示例数据
df = pd.read_csv("pokemon_data.csv",encoding="gbk")
df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



（1）np.where

通过条件判断来进行分组，满足条件的为一组，不满足条件的为另外一组

例如：攻击力大于或等于79的划分为强攻，攻击力小于79的划分为弱攻


```python
df["攻击强弱度"] = np.where(df["攻击力"] >= 79,"强攻","弱攻")
```


```python
df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
      <th>攻击强弱度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
      <td>弱攻</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
      <td>弱攻</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
      <td>强攻</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
      <td>强攻</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
      <td>弱攻</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
      <td>弱攻</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>强攻</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
      <td>强攻</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>强攻</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
      <td>弱攻</td>
    </tr>
  </tbody>
</table>
</div>



如果觉得一个条件不够，可以追加条件，例如：满足攻击力大于或等于79，且防御力大于或等于100，划分为S级，其余的划分为A级


```python
df["划分级"] = np.where((df["攻击力"] >= 79) & (df["防御力"] >= 100),"S级" ,"A级")
```


```python
df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
      <th>攻击强弱度</th>
      <th>划分级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
      <td>弱攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
      <td>弱攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
      <td>强攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
      <td>强攻</td>
      <td>S级</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
      <td>弱攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
      <td>弱攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>强攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
      <td>强攻</td>
      <td>S级</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>强攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
      <td>弱攻</td>
      <td>A级</td>
    </tr>
  </tbody>
</table>
</div>



（2）loc


```python
#还原数据
df = pd.read_csv("pokemon_data.csv",encoding="gbk")
df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



攻击力大于或等于79的划分为强攻，攻击力小于79的划分为弱攻


```python
df.loc[df["攻击力"] >= 79,"攻击强弱度"] = "强攻"
df.loc[df["攻击力"] < 79,"攻击强弱度"] = "弱攻"
```


```python
df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
      <th>攻击强弱度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
      <td>弱攻</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
      <td>弱攻</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
      <td>强攻</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
      <td>强攻</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
      <td>弱攻</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
      <td>弱攻</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>强攻</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
      <td>强攻</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>强攻</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
      <td>弱攻</td>
    </tr>
  </tbody>
</table>
</div>



满足攻击力大于或等于79，且防御力大于或等于100，划分为S级，其余的划分为A级


```python
df.loc[(df["攻击力"] >=79) & (df["防御力"] >=100),"划分级"] = "S级"
df.loc[~((df["攻击力"] >=79) & (df["防御力"] >=100)),"划分级"] = "A级"
```


```python
df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
      <th>攻击强弱度</th>
      <th>划分级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
      <td>弱攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
      <td>弱攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
      <td>强攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
      <td>强攻</td>
      <td>S级</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
      <td>弱攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
      <td>弱攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>强攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
      <td>强攻</td>
      <td>S级</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>强攻</td>
      <td>A级</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
      <td>弱攻</td>
      <td>A级</td>
    </tr>
  </tbody>
</table>
</div>



## 六.数据分列与合并

+ 分列


```python
#示例数据
df4 = pd.DataFrame({"name":["ray","jack","lucy","bob","candy"],
                    "h&w":["175-70","180-80","168-74","177-72","182-90"],
                    "score":[10,30,20,15,50]},
                  columns=["name","h&w","score"])
df4
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>h&amp;w</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>175-70</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>180-80</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>168-74</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>177-72</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>182-90</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



需求：现在需要将h&w这一列的数据拆分开，分为身高列和体重列，并且拆分后的列需要在源数据里呈现出来


```python
[x.split("-") for x in df4["h&w"]]
```




    [['175', '70'], ['180', '80'], ['168', '74'], ['177', '72'], ['182', '90']]




```python
#对h&w列的值依次进行分列，并创建数据表df_split，索引值为df4的索引，列名位height和weight
df_split = pd.DataFrame([x.split("-") for x in df4["h&w"]],index=df4.index,columns=["height","weight"])
df_split
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>175</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>180</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>168</td>
      <td>74</td>
    </tr>
    <tr>
      <th>3</th>
      <td>177</td>
      <td>72</td>
    </tr>
    <tr>
      <th>4</th>
      <td>182</td>
      <td>90</td>
    </tr>
  </tbody>
</table>
</div>




```python
#将分列后的数据表df_split与原df4数据表进行匹配
df4 = pd.merge(df4,df_split,right_index=True,left_index=True)
df4
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>h&amp;w</th>
      <th>score</th>
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>175-70</td>
      <td>10</td>
      <td>175</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>180-80</td>
      <td>30</td>
      <td>180</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>168-74</td>
      <td>20</td>
      <td>168</td>
      <td>74</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>177-72</td>
      <td>15</td>
      <td>177</td>
      <td>72</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>182-90</td>
      <td>50</td>
      <td>182</td>
      <td>90</td>
    </tr>
  </tbody>
</table>
</div>



+ 合并

需求：现在需要将name和score合并为一列，形式如ray：10，合并后的新列需要在源数据里呈现出来


```python
df4["name:score"] = df4["name"] + "：" + df4["score"].apply(str)
```


```python
df4
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>h&amp;w</th>
      <th>score</th>
      <th>height</th>
      <th>weight</th>
      <th>name:score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>175-70</td>
      <td>10</td>
      <td>175</td>
      <td>70</td>
      <td>ray：10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jack</td>
      <td>180-80</td>
      <td>30</td>
      <td>180</td>
      <td>80</td>
      <td>jack：30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lucy</td>
      <td>168-74</td>
      <td>20</td>
      <td>168</td>
      <td>74</td>
      <td>lucy：20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bob</td>
      <td>177-72</td>
      <td>15</td>
      <td>177</td>
      <td>72</td>
      <td>bob：15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>candy</td>
      <td>182-90</td>
      <td>50</td>
      <td>182</td>
      <td>90</td>
      <td>candy：50</td>
    </tr>
  </tbody>
</table>
</div>



## 七.排序


```python
#示例数据
df = pd.read_csv("pokemon_data.csv",encoding="gbk")
df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



+ 根据值排序

Q1：希望数据表按照“总计”字段的值来升序排列


```python
df.sort_values(by="总计")    #默认就是升序排列
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>206</th>
      <td>Sunkern</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>180</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>322</th>
      <td>Azurill</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>190</td>
      <td>50</td>
      <td>20</td>
      <td>40</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>446</th>
      <td>Kricketot</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>194</td>
      <td>37</td>
      <td>25</td>
      <td>41</td>
      <td>25</td>
      <td>4</td>
    </tr>
    <tr>
      <th>288</th>
      <td>Wurmple</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>45</td>
      <td>35</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Weedle</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Caterpie</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>303</th>
      <td>Ralts</td>
      <td>Psychic</td>
      <td>Fairy</td>
      <td>198</td>
      <td>28</td>
      <td>25</td>
      <td>25</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>732</th>
      <td>Scatterbug</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>200</td>
      <td>38</td>
      <td>35</td>
      <td>40</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Magikarp</td>
      <td>Water</td>
      <td>NaN</td>
      <td>200</td>
      <td>20</td>
      <td>10</td>
      <td>55</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>381</th>
      <td>Feebas</td>
      <td>Water</td>
      <td>NaN</td>
      <td>200</td>
      <td>20</td>
      <td>15</td>
      <td>20</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>291</th>
      <td>Cascoon</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>186</th>
      <td>Pichu</td>
      <td>Electric</td>
      <td>NaN</td>
      <td>205</td>
      <td>20</td>
      <td>40</td>
      <td>15</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kakuna</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Metapod</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>289</th>
      <td>Silcoon</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>209</th>
      <td>Wooper</td>
      <td>Water</td>
      <td>Ground</td>
      <td>210</td>
      <td>55</td>
      <td>45</td>
      <td>45</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>255</th>
      <td>Tyrogue</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>210</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>2</td>
    </tr>
    <tr>
      <th>188</th>
      <td>Igglybuff</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>210</td>
      <td>90</td>
      <td>30</td>
      <td>15</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>733</th>
      <td>Spewpa</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>213</td>
      <td>45</td>
      <td>22</td>
      <td>60</td>
      <td>29</td>
      <td>6</td>
    </tr>
    <tr>
      <th>175</th>
      <td>Sentret</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>215</td>
      <td>35</td>
      <td>46</td>
      <td>34</td>
      <td>20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>187</th>
      <td>Cleffa</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>218</td>
      <td>50</td>
      <td>25</td>
      <td>28</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>284</th>
      <td>Poochyena</td>
      <td>Dark</td>
      <td>NaN</td>
      <td>220</td>
      <td>35</td>
      <td>55</td>
      <td>35</td>
      <td>35</td>
      <td>3</td>
    </tr>
    <tr>
      <th>488</th>
      <td>Happiny</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>220</td>
      <td>100</td>
      <td>5</td>
      <td>5</td>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Seedot</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>220</td>
      <td>40</td>
      <td>40</td>
      <td>50</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>293</th>
      <td>Lotad</td>
      <td>Water</td>
      <td>Grass</td>
      <td>220</td>
      <td>40</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>457</th>
      <td>Burmy</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>224</td>
      <td>40</td>
      <td>29</td>
      <td>45</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>316</th>
      <td>Shedinja</td>
      <td>Bug</td>
      <td>Ghost</td>
      <td>236</td>
      <td>1</td>
      <td>90</td>
      <td>45</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>727</th>
      <td>Bunnelby</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>237</td>
      <td>38</td>
      <td>36</td>
      <td>38</td>
      <td>57</td>
      <td>6</td>
    </tr>
    <tr>
      <th>320</th>
      <td>Makuhita</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>237</td>
      <td>72</td>
      <td>60</td>
      <td>30</td>
      <td>25</td>
      <td>3</td>
    </tr>
    <tr>
      <th>286</th>
      <td>Zigzagoon</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>240</td>
      <td>38</td>
      <td>30</td>
      <td>41</td>
      <td>60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>313</th>
      <td>Slaking</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>670</td>
      <td>150</td>
      <td>160</td>
      <td>100</td>
      <td>100</td>
      <td>3</td>
    </tr>
    <tr>
      <th>421</th>
      <td>Kyogre</td>
      <td>Water</td>
      <td>NaN</td>
      <td>670</td>
      <td>100</td>
      <td>100</td>
      <td>90</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>162</th>
      <td>Mewtwo</td>
      <td>Psychic</td>
      <td>NaN</td>
      <td>680</td>
      <td>106</td>
      <td>110</td>
      <td>90</td>
      <td>130</td>
      <td>1</td>
    </tr>
    <tr>
      <th>798</th>
      <td>HoopaHoopa Unbound</td>
      <td>Psychic</td>
      <td>Dark</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
      <td>60</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>793</th>
      <td>Yveltal</td>
      <td>Dark</td>
      <td>Flying</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>792</th>
      <td>Xerneas</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>541</th>
      <td>Palkia</td>
      <td>Water</td>
      <td>Dragon</td>
      <td>680</td>
      <td>90</td>
      <td>120</td>
      <td>100</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>545</th>
      <td>GiratinaOrigin Forme</td>
      <td>Ghost</td>
      <td>Dragon</td>
      <td>680</td>
      <td>150</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>270</th>
      <td>Ho-oh</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>680</td>
      <td>106</td>
      <td>130</td>
      <td>90</td>
      <td>90</td>
      <td>2</td>
    </tr>
    <tr>
      <th>706</th>
      <td>Reshiram</td>
      <td>Dragon</td>
      <td>Fire</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>269</th>
      <td>Lugia</td>
      <td>Psychic</td>
      <td>Flying</td>
      <td>680</td>
      <td>106</td>
      <td>90</td>
      <td>130</td>
      <td>110</td>
      <td>2</td>
    </tr>
    <tr>
      <th>707</th>
      <td>Zekrom</td>
      <td>Dragon</td>
      <td>Electric</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
      <td>120</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>540</th>
      <td>Dialga</td>
      <td>Steel</td>
      <td>Dragon</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>120</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>544</th>
      <td>GiratinaAltered Forme</td>
      <td>Ghost</td>
      <td>Dragon</td>
      <td>680</td>
      <td>150</td>
      <td>100</td>
      <td>120</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>425</th>
      <td>Rayquaza</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>680</td>
      <td>105</td>
      <td>150</td>
      <td>90</td>
      <td>95</td>
      <td>3</td>
    </tr>
    <tr>
      <th>268</th>
      <td>TyranitarMega Tyranitar</td>
      <td>Rock</td>
      <td>Dark</td>
      <td>700</td>
      <td>100</td>
      <td>164</td>
      <td>150</td>
      <td>71</td>
      <td>2</td>
    </tr>
    <tr>
      <th>796</th>
      <td>DiancieMega Diancie</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
    </tr>
    <tr>
      <th>418</th>
      <td>LatiasMega Latias</td>
      <td>Dragon</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>100</td>
      <td>120</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>420</th>
      <td>LatiosMega Latios</td>
      <td>Dragon</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>130</td>
      <td>100</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>711</th>
      <td>KyuremBlack Kyurem</td>
      <td>Dragon</td>
      <td>Ice</td>
      <td>700</td>
      <td>125</td>
      <td>170</td>
      <td>100</td>
      <td>95</td>
      <td>5</td>
    </tr>
    <tr>
      <th>413</th>
      <td>MetagrossMega Metagross</td>
      <td>Steel</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>145</td>
      <td>150</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>409</th>
      <td>SalamenceMega Salamence</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>700</td>
      <td>95</td>
      <td>145</td>
      <td>130</td>
      <td>120</td>
      <td>3</td>
    </tr>
    <tr>
      <th>494</th>
      <td>GarchompMega Garchomp</td>
      <td>Dragon</td>
      <td>Ground</td>
      <td>700</td>
      <td>108</td>
      <td>170</td>
      <td>115</td>
      <td>92</td>
      <td>4</td>
    </tr>
    <tr>
      <th>712</th>
      <td>KyuremWhite Kyurem</td>
      <td>Dragon</td>
      <td>Ice</td>
      <td>700</td>
      <td>125</td>
      <td>120</td>
      <td>90</td>
      <td>95</td>
      <td>5</td>
    </tr>
    <tr>
      <th>552</th>
      <td>Arceus</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>720</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>4</td>
    </tr>
    <tr>
      <th>424</th>
      <td>GroudonPrimal Groudon</td>
      <td>Ground</td>
      <td>Fire</td>
      <td>770</td>
      <td>100</td>
      <td>180</td>
      <td>160</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>422</th>
      <td>KyogrePrimal Kyogre</td>
      <td>Water</td>
      <td>NaN</td>
      <td>770</td>
      <td>100</td>
      <td>150</td>
      <td>90</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>164</th>
      <td>MewtwoMega Mewtwo Y</td>
      <td>Psychic</td>
      <td>NaN</td>
      <td>780</td>
      <td>106</td>
      <td>150</td>
      <td>70</td>
      <td>140</td>
      <td>1</td>
    </tr>
    <tr>
      <th>426</th>
      <td>RayquazaMega Rayquaza</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>780</td>
      <td>105</td>
      <td>180</td>
      <td>100</td>
      <td>115</td>
      <td>3</td>
    </tr>
    <tr>
      <th>163</th>
      <td>MewtwoMega Mewtwo X</td>
      <td>Psychic</td>
      <td>Fighting</td>
      <td>780</td>
      <td>106</td>
      <td>190</td>
      <td>100</td>
      <td>130</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 9 columns</p>
</div>



Q2：希望数据表按照“总计”字段的值来降序排列


```python
df.sort_values(by="总计",ascending=False)  #添加ascending=False参数来降序排列
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>426</th>
      <td>RayquazaMega Rayquaza</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>780</td>
      <td>105</td>
      <td>180</td>
      <td>100</td>
      <td>115</td>
      <td>3</td>
    </tr>
    <tr>
      <th>164</th>
      <td>MewtwoMega Mewtwo Y</td>
      <td>Psychic</td>
      <td>NaN</td>
      <td>780</td>
      <td>106</td>
      <td>150</td>
      <td>70</td>
      <td>140</td>
      <td>1</td>
    </tr>
    <tr>
      <th>163</th>
      <td>MewtwoMega Mewtwo X</td>
      <td>Psychic</td>
      <td>Fighting</td>
      <td>780</td>
      <td>106</td>
      <td>190</td>
      <td>100</td>
      <td>130</td>
      <td>1</td>
    </tr>
    <tr>
      <th>422</th>
      <td>KyogrePrimal Kyogre</td>
      <td>Water</td>
      <td>NaN</td>
      <td>770</td>
      <td>100</td>
      <td>150</td>
      <td>90</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>424</th>
      <td>GroudonPrimal Groudon</td>
      <td>Ground</td>
      <td>Fire</td>
      <td>770</td>
      <td>100</td>
      <td>180</td>
      <td>160</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>552</th>
      <td>Arceus</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>720</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>4</td>
    </tr>
    <tr>
      <th>712</th>
      <td>KyuremWhite Kyurem</td>
      <td>Dragon</td>
      <td>Ice</td>
      <td>700</td>
      <td>125</td>
      <td>120</td>
      <td>90</td>
      <td>95</td>
      <td>5</td>
    </tr>
    <tr>
      <th>711</th>
      <td>KyuremBlack Kyurem</td>
      <td>Dragon</td>
      <td>Ice</td>
      <td>700</td>
      <td>125</td>
      <td>170</td>
      <td>100</td>
      <td>95</td>
      <td>5</td>
    </tr>
    <tr>
      <th>409</th>
      <td>SalamenceMega Salamence</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>700</td>
      <td>95</td>
      <td>145</td>
      <td>130</td>
      <td>120</td>
      <td>3</td>
    </tr>
    <tr>
      <th>413</th>
      <td>MetagrossMega Metagross</td>
      <td>Steel</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>145</td>
      <td>150</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>418</th>
      <td>LatiasMega Latias</td>
      <td>Dragon</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>100</td>
      <td>120</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>420</th>
      <td>LatiosMega Latios</td>
      <td>Dragon</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>130</td>
      <td>100</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>494</th>
      <td>GarchompMega Garchomp</td>
      <td>Dragon</td>
      <td>Ground</td>
      <td>700</td>
      <td>108</td>
      <td>170</td>
      <td>115</td>
      <td>92</td>
      <td>4</td>
    </tr>
    <tr>
      <th>268</th>
      <td>TyranitarMega Tyranitar</td>
      <td>Rock</td>
      <td>Dark</td>
      <td>700</td>
      <td>100</td>
      <td>164</td>
      <td>150</td>
      <td>71</td>
      <td>2</td>
    </tr>
    <tr>
      <th>796</th>
      <td>DiancieMega Diancie</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
    </tr>
    <tr>
      <th>425</th>
      <td>Rayquaza</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>680</td>
      <td>105</td>
      <td>150</td>
      <td>90</td>
      <td>95</td>
      <td>3</td>
    </tr>
    <tr>
      <th>545</th>
      <td>GiratinaOrigin Forme</td>
      <td>Ghost</td>
      <td>Dragon</td>
      <td>680</td>
      <td>150</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>544</th>
      <td>GiratinaAltered Forme</td>
      <td>Ghost</td>
      <td>Dragon</td>
      <td>680</td>
      <td>150</td>
      <td>100</td>
      <td>120</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>707</th>
      <td>Zekrom</td>
      <td>Dragon</td>
      <td>Electric</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
      <td>120</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>541</th>
      <td>Palkia</td>
      <td>Water</td>
      <td>Dragon</td>
      <td>680</td>
      <td>90</td>
      <td>120</td>
      <td>100</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>540</th>
      <td>Dialga</td>
      <td>Steel</td>
      <td>Dragon</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>120</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>270</th>
      <td>Ho-oh</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>680</td>
      <td>106</td>
      <td>130</td>
      <td>90</td>
      <td>90</td>
      <td>2</td>
    </tr>
    <tr>
      <th>269</th>
      <td>Lugia</td>
      <td>Psychic</td>
      <td>Flying</td>
      <td>680</td>
      <td>106</td>
      <td>90</td>
      <td>130</td>
      <td>110</td>
      <td>2</td>
    </tr>
    <tr>
      <th>706</th>
      <td>Reshiram</td>
      <td>Dragon</td>
      <td>Fire</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>792</th>
      <td>Xerneas</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>798</th>
      <td>HoopaHoopa Unbound</td>
      <td>Psychic</td>
      <td>Dark</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
      <td>60</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>162</th>
      <td>Mewtwo</td>
      <td>Psychic</td>
      <td>NaN</td>
      <td>680</td>
      <td>106</td>
      <td>110</td>
      <td>90</td>
      <td>130</td>
      <td>1</td>
    </tr>
    <tr>
      <th>793</th>
      <td>Yveltal</td>
      <td>Dark</td>
      <td>Flying</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>543</th>
      <td>Regigigas</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>670</td>
      <td>110</td>
      <td>160</td>
      <td>110</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>423</th>
      <td>Groudon</td>
      <td>Ground</td>
      <td>NaN</td>
      <td>670</td>
      <td>100</td>
      <td>150</td>
      <td>140</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>286</th>
      <td>Zigzagoon</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>240</td>
      <td>38</td>
      <td>30</td>
      <td>41</td>
      <td>60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>320</th>
      <td>Makuhita</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>237</td>
      <td>72</td>
      <td>60</td>
      <td>30</td>
      <td>25</td>
      <td>3</td>
    </tr>
    <tr>
      <th>727</th>
      <td>Bunnelby</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>237</td>
      <td>38</td>
      <td>36</td>
      <td>38</td>
      <td>57</td>
      <td>6</td>
    </tr>
    <tr>
      <th>316</th>
      <td>Shedinja</td>
      <td>Bug</td>
      <td>Ghost</td>
      <td>236</td>
      <td>1</td>
      <td>90</td>
      <td>45</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>457</th>
      <td>Burmy</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>224</td>
      <td>40</td>
      <td>29</td>
      <td>45</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>284</th>
      <td>Poochyena</td>
      <td>Dark</td>
      <td>NaN</td>
      <td>220</td>
      <td>35</td>
      <td>55</td>
      <td>35</td>
      <td>35</td>
      <td>3</td>
    </tr>
    <tr>
      <th>293</th>
      <td>Lotad</td>
      <td>Water</td>
      <td>Grass</td>
      <td>220</td>
      <td>40</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Seedot</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>220</td>
      <td>40</td>
      <td>40</td>
      <td>50</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>488</th>
      <td>Happiny</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>220</td>
      <td>100</td>
      <td>5</td>
      <td>5</td>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>187</th>
      <td>Cleffa</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>218</td>
      <td>50</td>
      <td>25</td>
      <td>28</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>175</th>
      <td>Sentret</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>215</td>
      <td>35</td>
      <td>46</td>
      <td>34</td>
      <td>20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>733</th>
      <td>Spewpa</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>213</td>
      <td>45</td>
      <td>22</td>
      <td>60</td>
      <td>29</td>
      <td>6</td>
    </tr>
    <tr>
      <th>255</th>
      <td>Tyrogue</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>210</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>2</td>
    </tr>
    <tr>
      <th>188</th>
      <td>Igglybuff</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>210</td>
      <td>90</td>
      <td>30</td>
      <td>15</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>209</th>
      <td>Wooper</td>
      <td>Water</td>
      <td>Ground</td>
      <td>210</td>
      <td>55</td>
      <td>45</td>
      <td>45</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>289</th>
      <td>Silcoon</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>291</th>
      <td>Cascoon</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Metapod</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kakuna</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>186</th>
      <td>Pichu</td>
      <td>Electric</td>
      <td>NaN</td>
      <td>205</td>
      <td>20</td>
      <td>40</td>
      <td>15</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Magikarp</td>
      <td>Water</td>
      <td>NaN</td>
      <td>200</td>
      <td>20</td>
      <td>10</td>
      <td>55</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>732</th>
      <td>Scatterbug</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>200</td>
      <td>38</td>
      <td>35</td>
      <td>40</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>381</th>
      <td>Feebas</td>
      <td>Water</td>
      <td>NaN</td>
      <td>200</td>
      <td>20</td>
      <td>15</td>
      <td>20</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>303</th>
      <td>Ralts</td>
      <td>Psychic</td>
      <td>Fairy</td>
      <td>198</td>
      <td>28</td>
      <td>25</td>
      <td>25</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Weedle</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Caterpie</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>288</th>
      <td>Wurmple</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>45</td>
      <td>35</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>446</th>
      <td>Kricketot</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>194</td>
      <td>37</td>
      <td>25</td>
      <td>41</td>
      <td>25</td>
      <td>4</td>
    </tr>
    <tr>
      <th>322</th>
      <td>Azurill</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>190</td>
      <td>50</td>
      <td>20</td>
      <td>40</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>206</th>
      <td>Sunkern</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>180</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 9 columns</p>
</div>



Q3：设置排序的一级关键词是总计，二级关键词为攻击力


```python
#如果都做升序排列
df.sort_values(by=["总计","攻击力"])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>206</th>
      <td>Sunkern</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>180</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>322</th>
      <td>Azurill</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>190</td>
      <td>50</td>
      <td>20</td>
      <td>40</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>446</th>
      <td>Kricketot</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>194</td>
      <td>37</td>
      <td>25</td>
      <td>41</td>
      <td>25</td>
      <td>4</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Caterpie</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Weedle</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>288</th>
      <td>Wurmple</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>45</td>
      <td>35</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>303</th>
      <td>Ralts</td>
      <td>Psychic</td>
      <td>Fairy</td>
      <td>198</td>
      <td>28</td>
      <td>25</td>
      <td>25</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Magikarp</td>
      <td>Water</td>
      <td>NaN</td>
      <td>200</td>
      <td>20</td>
      <td>10</td>
      <td>55</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>381</th>
      <td>Feebas</td>
      <td>Water</td>
      <td>NaN</td>
      <td>200</td>
      <td>20</td>
      <td>15</td>
      <td>20</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>732</th>
      <td>Scatterbug</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>200</td>
      <td>38</td>
      <td>35</td>
      <td>40</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Metapod</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kakuna</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>289</th>
      <td>Silcoon</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>291</th>
      <td>Cascoon</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>186</th>
      <td>Pichu</td>
      <td>Electric</td>
      <td>NaN</td>
      <td>205</td>
      <td>20</td>
      <td>40</td>
      <td>15</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>188</th>
      <td>Igglybuff</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>210</td>
      <td>90</td>
      <td>30</td>
      <td>15</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>255</th>
      <td>Tyrogue</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>210</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>2</td>
    </tr>
    <tr>
      <th>209</th>
      <td>Wooper</td>
      <td>Water</td>
      <td>Ground</td>
      <td>210</td>
      <td>55</td>
      <td>45</td>
      <td>45</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>733</th>
      <td>Spewpa</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>213</td>
      <td>45</td>
      <td>22</td>
      <td>60</td>
      <td>29</td>
      <td>6</td>
    </tr>
    <tr>
      <th>175</th>
      <td>Sentret</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>215</td>
      <td>35</td>
      <td>46</td>
      <td>34</td>
      <td>20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>187</th>
      <td>Cleffa</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>218</td>
      <td>50</td>
      <td>25</td>
      <td>28</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>488</th>
      <td>Happiny</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>220</td>
      <td>100</td>
      <td>5</td>
      <td>5</td>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>293</th>
      <td>Lotad</td>
      <td>Water</td>
      <td>Grass</td>
      <td>220</td>
      <td>40</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Seedot</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>220</td>
      <td>40</td>
      <td>40</td>
      <td>50</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>284</th>
      <td>Poochyena</td>
      <td>Dark</td>
      <td>NaN</td>
      <td>220</td>
      <td>35</td>
      <td>55</td>
      <td>35</td>
      <td>35</td>
      <td>3</td>
    </tr>
    <tr>
      <th>457</th>
      <td>Burmy</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>224</td>
      <td>40</td>
      <td>29</td>
      <td>45</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>316</th>
      <td>Shedinja</td>
      <td>Bug</td>
      <td>Ghost</td>
      <td>236</td>
      <td>1</td>
      <td>90</td>
      <td>45</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>727</th>
      <td>Bunnelby</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>237</td>
      <td>38</td>
      <td>36</td>
      <td>38</td>
      <td>57</td>
      <td>6</td>
    </tr>
    <tr>
      <th>320</th>
      <td>Makuhita</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>237</td>
      <td>72</td>
      <td>60</td>
      <td>30</td>
      <td>25</td>
      <td>3</td>
    </tr>
    <tr>
      <th>286</th>
      <td>Zigzagoon</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>240</td>
      <td>38</td>
      <td>30</td>
      <td>41</td>
      <td>60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>313</th>
      <td>Slaking</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>670</td>
      <td>150</td>
      <td>160</td>
      <td>100</td>
      <td>100</td>
      <td>3</td>
    </tr>
    <tr>
      <th>543</th>
      <td>Regigigas</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>670</td>
      <td>110</td>
      <td>160</td>
      <td>110</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>269</th>
      <td>Lugia</td>
      <td>Psychic</td>
      <td>Flying</td>
      <td>680</td>
      <td>106</td>
      <td>90</td>
      <td>130</td>
      <td>110</td>
      <td>2</td>
    </tr>
    <tr>
      <th>544</th>
      <td>GiratinaAltered Forme</td>
      <td>Ghost</td>
      <td>Dragon</td>
      <td>680</td>
      <td>150</td>
      <td>100</td>
      <td>120</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>162</th>
      <td>Mewtwo</td>
      <td>Psychic</td>
      <td>NaN</td>
      <td>680</td>
      <td>106</td>
      <td>110</td>
      <td>90</td>
      <td>130</td>
      <td>1</td>
    </tr>
    <tr>
      <th>540</th>
      <td>Dialga</td>
      <td>Steel</td>
      <td>Dragon</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>120</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>541</th>
      <td>Palkia</td>
      <td>Water</td>
      <td>Dragon</td>
      <td>680</td>
      <td>90</td>
      <td>120</td>
      <td>100</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>545</th>
      <td>GiratinaOrigin Forme</td>
      <td>Ghost</td>
      <td>Dragon</td>
      <td>680</td>
      <td>150</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>706</th>
      <td>Reshiram</td>
      <td>Dragon</td>
      <td>Fire</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>270</th>
      <td>Ho-oh</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>680</td>
      <td>106</td>
      <td>130</td>
      <td>90</td>
      <td>90</td>
      <td>2</td>
    </tr>
    <tr>
      <th>792</th>
      <td>Xerneas</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>793</th>
      <td>Yveltal</td>
      <td>Dark</td>
      <td>Flying</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>425</th>
      <td>Rayquaza</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>680</td>
      <td>105</td>
      <td>150</td>
      <td>90</td>
      <td>95</td>
      <td>3</td>
    </tr>
    <tr>
      <th>707</th>
      <td>Zekrom</td>
      <td>Dragon</td>
      <td>Electric</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
      <td>120</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>798</th>
      <td>HoopaHoopa Unbound</td>
      <td>Psychic</td>
      <td>Dark</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
      <td>60</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>418</th>
      <td>LatiasMega Latias</td>
      <td>Dragon</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>100</td>
      <td>120</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>712</th>
      <td>KyuremWhite Kyurem</td>
      <td>Dragon</td>
      <td>Ice</td>
      <td>700</td>
      <td>125</td>
      <td>120</td>
      <td>90</td>
      <td>95</td>
      <td>5</td>
    </tr>
    <tr>
      <th>420</th>
      <td>LatiosMega Latios</td>
      <td>Dragon</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>130</td>
      <td>100</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>409</th>
      <td>SalamenceMega Salamence</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>700</td>
      <td>95</td>
      <td>145</td>
      <td>130</td>
      <td>120</td>
      <td>3</td>
    </tr>
    <tr>
      <th>413</th>
      <td>MetagrossMega Metagross</td>
      <td>Steel</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>145</td>
      <td>150</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>796</th>
      <td>DiancieMega Diancie</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
    </tr>
    <tr>
      <th>268</th>
      <td>TyranitarMega Tyranitar</td>
      <td>Rock</td>
      <td>Dark</td>
      <td>700</td>
      <td>100</td>
      <td>164</td>
      <td>150</td>
      <td>71</td>
      <td>2</td>
    </tr>
    <tr>
      <th>494</th>
      <td>GarchompMega Garchomp</td>
      <td>Dragon</td>
      <td>Ground</td>
      <td>700</td>
      <td>108</td>
      <td>170</td>
      <td>115</td>
      <td>92</td>
      <td>4</td>
    </tr>
    <tr>
      <th>711</th>
      <td>KyuremBlack Kyurem</td>
      <td>Dragon</td>
      <td>Ice</td>
      <td>700</td>
      <td>125</td>
      <td>170</td>
      <td>100</td>
      <td>95</td>
      <td>5</td>
    </tr>
    <tr>
      <th>552</th>
      <td>Arceus</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>720</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>4</td>
    </tr>
    <tr>
      <th>422</th>
      <td>KyogrePrimal Kyogre</td>
      <td>Water</td>
      <td>NaN</td>
      <td>770</td>
      <td>100</td>
      <td>150</td>
      <td>90</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>424</th>
      <td>GroudonPrimal Groudon</td>
      <td>Ground</td>
      <td>Fire</td>
      <td>770</td>
      <td>100</td>
      <td>180</td>
      <td>160</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>164</th>
      <td>MewtwoMega Mewtwo Y</td>
      <td>Psychic</td>
      <td>NaN</td>
      <td>780</td>
      <td>106</td>
      <td>150</td>
      <td>70</td>
      <td>140</td>
      <td>1</td>
    </tr>
    <tr>
      <th>426</th>
      <td>RayquazaMega Rayquaza</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>780</td>
      <td>105</td>
      <td>180</td>
      <td>100</td>
      <td>115</td>
      <td>3</td>
    </tr>
    <tr>
      <th>163</th>
      <td>MewtwoMega Mewtwo X</td>
      <td>Psychic</td>
      <td>Fighting</td>
      <td>780</td>
      <td>106</td>
      <td>190</td>
      <td>100</td>
      <td>130</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 9 columns</p>
</div>




```python
#如果都做降序排列
df.sort_values(by=["总计","攻击力"],ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163</th>
      <td>MewtwoMega Mewtwo X</td>
      <td>Psychic</td>
      <td>Fighting</td>
      <td>780</td>
      <td>106</td>
      <td>190</td>
      <td>100</td>
      <td>130</td>
      <td>1</td>
    </tr>
    <tr>
      <th>426</th>
      <td>RayquazaMega Rayquaza</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>780</td>
      <td>105</td>
      <td>180</td>
      <td>100</td>
      <td>115</td>
      <td>3</td>
    </tr>
    <tr>
      <th>164</th>
      <td>MewtwoMega Mewtwo Y</td>
      <td>Psychic</td>
      <td>NaN</td>
      <td>780</td>
      <td>106</td>
      <td>150</td>
      <td>70</td>
      <td>140</td>
      <td>1</td>
    </tr>
    <tr>
      <th>424</th>
      <td>GroudonPrimal Groudon</td>
      <td>Ground</td>
      <td>Fire</td>
      <td>770</td>
      <td>100</td>
      <td>180</td>
      <td>160</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>422</th>
      <td>KyogrePrimal Kyogre</td>
      <td>Water</td>
      <td>NaN</td>
      <td>770</td>
      <td>100</td>
      <td>150</td>
      <td>90</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>552</th>
      <td>Arceus</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>720</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>4</td>
    </tr>
    <tr>
      <th>494</th>
      <td>GarchompMega Garchomp</td>
      <td>Dragon</td>
      <td>Ground</td>
      <td>700</td>
      <td>108</td>
      <td>170</td>
      <td>115</td>
      <td>92</td>
      <td>4</td>
    </tr>
    <tr>
      <th>711</th>
      <td>KyuremBlack Kyurem</td>
      <td>Dragon</td>
      <td>Ice</td>
      <td>700</td>
      <td>125</td>
      <td>170</td>
      <td>100</td>
      <td>95</td>
      <td>5</td>
    </tr>
    <tr>
      <th>268</th>
      <td>TyranitarMega Tyranitar</td>
      <td>Rock</td>
      <td>Dark</td>
      <td>700</td>
      <td>100</td>
      <td>164</td>
      <td>150</td>
      <td>71</td>
      <td>2</td>
    </tr>
    <tr>
      <th>796</th>
      <td>DiancieMega Diancie</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
    </tr>
    <tr>
      <th>409</th>
      <td>SalamenceMega Salamence</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>700</td>
      <td>95</td>
      <td>145</td>
      <td>130</td>
      <td>120</td>
      <td>3</td>
    </tr>
    <tr>
      <th>413</th>
      <td>MetagrossMega Metagross</td>
      <td>Steel</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>145</td>
      <td>150</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>420</th>
      <td>LatiosMega Latios</td>
      <td>Dragon</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>130</td>
      <td>100</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>712</th>
      <td>KyuremWhite Kyurem</td>
      <td>Dragon</td>
      <td>Ice</td>
      <td>700</td>
      <td>125</td>
      <td>120</td>
      <td>90</td>
      <td>95</td>
      <td>5</td>
    </tr>
    <tr>
      <th>418</th>
      <td>LatiasMega Latias</td>
      <td>Dragon</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>100</td>
      <td>120</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>798</th>
      <td>HoopaHoopa Unbound</td>
      <td>Psychic</td>
      <td>Dark</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
      <td>60</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>425</th>
      <td>Rayquaza</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>680</td>
      <td>105</td>
      <td>150</td>
      <td>90</td>
      <td>95</td>
      <td>3</td>
    </tr>
    <tr>
      <th>707</th>
      <td>Zekrom</td>
      <td>Dragon</td>
      <td>Electric</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
      <td>120</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>792</th>
      <td>Xerneas</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>793</th>
      <td>Yveltal</td>
      <td>Dark</td>
      <td>Flying</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>270</th>
      <td>Ho-oh</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>680</td>
      <td>106</td>
      <td>130</td>
      <td>90</td>
      <td>90</td>
      <td>2</td>
    </tr>
    <tr>
      <th>540</th>
      <td>Dialga</td>
      <td>Steel</td>
      <td>Dragon</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>120</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>541</th>
      <td>Palkia</td>
      <td>Water</td>
      <td>Dragon</td>
      <td>680</td>
      <td>90</td>
      <td>120</td>
      <td>100</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>545</th>
      <td>GiratinaOrigin Forme</td>
      <td>Ghost</td>
      <td>Dragon</td>
      <td>680</td>
      <td>150</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>706</th>
      <td>Reshiram</td>
      <td>Dragon</td>
      <td>Fire</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>162</th>
      <td>Mewtwo</td>
      <td>Psychic</td>
      <td>NaN</td>
      <td>680</td>
      <td>106</td>
      <td>110</td>
      <td>90</td>
      <td>130</td>
      <td>1</td>
    </tr>
    <tr>
      <th>544</th>
      <td>GiratinaAltered Forme</td>
      <td>Ghost</td>
      <td>Dragon</td>
      <td>680</td>
      <td>150</td>
      <td>100</td>
      <td>120</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>269</th>
      <td>Lugia</td>
      <td>Psychic</td>
      <td>Flying</td>
      <td>680</td>
      <td>106</td>
      <td>90</td>
      <td>130</td>
      <td>110</td>
      <td>2</td>
    </tr>
    <tr>
      <th>313</th>
      <td>Slaking</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>670</td>
      <td>150</td>
      <td>160</td>
      <td>100</td>
      <td>100</td>
      <td>3</td>
    </tr>
    <tr>
      <th>543</th>
      <td>Regigigas</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>670</td>
      <td>110</td>
      <td>160</td>
      <td>110</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>286</th>
      <td>Zigzagoon</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>240</td>
      <td>38</td>
      <td>30</td>
      <td>41</td>
      <td>60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>320</th>
      <td>Makuhita</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>237</td>
      <td>72</td>
      <td>60</td>
      <td>30</td>
      <td>25</td>
      <td>3</td>
    </tr>
    <tr>
      <th>727</th>
      <td>Bunnelby</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>237</td>
      <td>38</td>
      <td>36</td>
      <td>38</td>
      <td>57</td>
      <td>6</td>
    </tr>
    <tr>
      <th>316</th>
      <td>Shedinja</td>
      <td>Bug</td>
      <td>Ghost</td>
      <td>236</td>
      <td>1</td>
      <td>90</td>
      <td>45</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>457</th>
      <td>Burmy</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>224</td>
      <td>40</td>
      <td>29</td>
      <td>45</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>284</th>
      <td>Poochyena</td>
      <td>Dark</td>
      <td>NaN</td>
      <td>220</td>
      <td>35</td>
      <td>55</td>
      <td>35</td>
      <td>35</td>
      <td>3</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Seedot</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>220</td>
      <td>40</td>
      <td>40</td>
      <td>50</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>293</th>
      <td>Lotad</td>
      <td>Water</td>
      <td>Grass</td>
      <td>220</td>
      <td>40</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>488</th>
      <td>Happiny</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>220</td>
      <td>100</td>
      <td>5</td>
      <td>5</td>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>187</th>
      <td>Cleffa</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>218</td>
      <td>50</td>
      <td>25</td>
      <td>28</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>175</th>
      <td>Sentret</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>215</td>
      <td>35</td>
      <td>46</td>
      <td>34</td>
      <td>20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>733</th>
      <td>Spewpa</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>213</td>
      <td>45</td>
      <td>22</td>
      <td>60</td>
      <td>29</td>
      <td>6</td>
    </tr>
    <tr>
      <th>209</th>
      <td>Wooper</td>
      <td>Water</td>
      <td>Ground</td>
      <td>210</td>
      <td>55</td>
      <td>45</td>
      <td>45</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>255</th>
      <td>Tyrogue</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>210</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>2</td>
    </tr>
    <tr>
      <th>188</th>
      <td>Igglybuff</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>210</td>
      <td>90</td>
      <td>30</td>
      <td>15</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>186</th>
      <td>Pichu</td>
      <td>Electric</td>
      <td>NaN</td>
      <td>205</td>
      <td>20</td>
      <td>40</td>
      <td>15</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>289</th>
      <td>Silcoon</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>291</th>
      <td>Cascoon</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kakuna</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Metapod</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>732</th>
      <td>Scatterbug</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>200</td>
      <td>38</td>
      <td>35</td>
      <td>40</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>381</th>
      <td>Feebas</td>
      <td>Water</td>
      <td>NaN</td>
      <td>200</td>
      <td>20</td>
      <td>15</td>
      <td>20</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Magikarp</td>
      <td>Water</td>
      <td>NaN</td>
      <td>200</td>
      <td>20</td>
      <td>10</td>
      <td>55</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>303</th>
      <td>Ralts</td>
      <td>Psychic</td>
      <td>Fairy</td>
      <td>198</td>
      <td>28</td>
      <td>25</td>
      <td>25</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>288</th>
      <td>Wurmple</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>45</td>
      <td>35</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Weedle</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Caterpie</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>446</th>
      <td>Kricketot</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>194</td>
      <td>37</td>
      <td>25</td>
      <td>41</td>
      <td>25</td>
      <td>4</td>
    </tr>
    <tr>
      <th>322</th>
      <td>Azurill</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>190</td>
      <td>50</td>
      <td>20</td>
      <td>40</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>206</th>
      <td>Sunkern</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>180</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 9 columns</p>
</div>




```python
#如果按照总计列做升序排列，但是总计相等的行按照攻击力做降序排列
df.sort_values(by=["总计","攻击力"],ascending=[True,False])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>206</th>
      <td>Sunkern</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>180</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>322</th>
      <td>Azurill</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>190</td>
      <td>50</td>
      <td>20</td>
      <td>40</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>446</th>
      <td>Kricketot</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>194</td>
      <td>37</td>
      <td>25</td>
      <td>41</td>
      <td>25</td>
      <td>4</td>
    </tr>
    <tr>
      <th>288</th>
      <td>Wurmple</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>45</td>
      <td>35</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Weedle</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Caterpie</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>303</th>
      <td>Ralts</td>
      <td>Psychic</td>
      <td>Fairy</td>
      <td>198</td>
      <td>28</td>
      <td>25</td>
      <td>25</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>732</th>
      <td>Scatterbug</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>200</td>
      <td>38</td>
      <td>35</td>
      <td>40</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>381</th>
      <td>Feebas</td>
      <td>Water</td>
      <td>NaN</td>
      <td>200</td>
      <td>20</td>
      <td>15</td>
      <td>20</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Magikarp</td>
      <td>Water</td>
      <td>NaN</td>
      <td>200</td>
      <td>20</td>
      <td>10</td>
      <td>55</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>186</th>
      <td>Pichu</td>
      <td>Electric</td>
      <td>NaN</td>
      <td>205</td>
      <td>20</td>
      <td>40</td>
      <td>15</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>289</th>
      <td>Silcoon</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>291</th>
      <td>Cascoon</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kakuna</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Metapod</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>209</th>
      <td>Wooper</td>
      <td>Water</td>
      <td>Ground</td>
      <td>210</td>
      <td>55</td>
      <td>45</td>
      <td>45</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>255</th>
      <td>Tyrogue</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>210</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>2</td>
    </tr>
    <tr>
      <th>188</th>
      <td>Igglybuff</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>210</td>
      <td>90</td>
      <td>30</td>
      <td>15</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>733</th>
      <td>Spewpa</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>213</td>
      <td>45</td>
      <td>22</td>
      <td>60</td>
      <td>29</td>
      <td>6</td>
    </tr>
    <tr>
      <th>175</th>
      <td>Sentret</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>215</td>
      <td>35</td>
      <td>46</td>
      <td>34</td>
      <td>20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>187</th>
      <td>Cleffa</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>218</td>
      <td>50</td>
      <td>25</td>
      <td>28</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>284</th>
      <td>Poochyena</td>
      <td>Dark</td>
      <td>NaN</td>
      <td>220</td>
      <td>35</td>
      <td>55</td>
      <td>35</td>
      <td>35</td>
      <td>3</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Seedot</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>220</td>
      <td>40</td>
      <td>40</td>
      <td>50</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>293</th>
      <td>Lotad</td>
      <td>Water</td>
      <td>Grass</td>
      <td>220</td>
      <td>40</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>488</th>
      <td>Happiny</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>220</td>
      <td>100</td>
      <td>5</td>
      <td>5</td>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>457</th>
      <td>Burmy</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>224</td>
      <td>40</td>
      <td>29</td>
      <td>45</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>316</th>
      <td>Shedinja</td>
      <td>Bug</td>
      <td>Ghost</td>
      <td>236</td>
      <td>1</td>
      <td>90</td>
      <td>45</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>320</th>
      <td>Makuhita</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>237</td>
      <td>72</td>
      <td>60</td>
      <td>30</td>
      <td>25</td>
      <td>3</td>
    </tr>
    <tr>
      <th>727</th>
      <td>Bunnelby</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>237</td>
      <td>38</td>
      <td>36</td>
      <td>38</td>
      <td>57</td>
      <td>6</td>
    </tr>
    <tr>
      <th>317</th>
      <td>Whismur</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>240</td>
      <td>64</td>
      <td>51</td>
      <td>23</td>
      <td>28</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>423</th>
      <td>Groudon</td>
      <td>Ground</td>
      <td>NaN</td>
      <td>670</td>
      <td>100</td>
      <td>150</td>
      <td>140</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>421</th>
      <td>Kyogre</td>
      <td>Water</td>
      <td>NaN</td>
      <td>670</td>
      <td>100</td>
      <td>100</td>
      <td>90</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>798</th>
      <td>HoopaHoopa Unbound</td>
      <td>Psychic</td>
      <td>Dark</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
      <td>60</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>425</th>
      <td>Rayquaza</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>680</td>
      <td>105</td>
      <td>150</td>
      <td>90</td>
      <td>95</td>
      <td>3</td>
    </tr>
    <tr>
      <th>707</th>
      <td>Zekrom</td>
      <td>Dragon</td>
      <td>Electric</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
      <td>120</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>792</th>
      <td>Xerneas</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>793</th>
      <td>Yveltal</td>
      <td>Dark</td>
      <td>Flying</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>270</th>
      <td>Ho-oh</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>680</td>
      <td>106</td>
      <td>130</td>
      <td>90</td>
      <td>90</td>
      <td>2</td>
    </tr>
    <tr>
      <th>540</th>
      <td>Dialga</td>
      <td>Steel</td>
      <td>Dragon</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>120</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>541</th>
      <td>Palkia</td>
      <td>Water</td>
      <td>Dragon</td>
      <td>680</td>
      <td>90</td>
      <td>120</td>
      <td>100</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>545</th>
      <td>GiratinaOrigin Forme</td>
      <td>Ghost</td>
      <td>Dragon</td>
      <td>680</td>
      <td>150</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>706</th>
      <td>Reshiram</td>
      <td>Dragon</td>
      <td>Fire</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>162</th>
      <td>Mewtwo</td>
      <td>Psychic</td>
      <td>NaN</td>
      <td>680</td>
      <td>106</td>
      <td>110</td>
      <td>90</td>
      <td>130</td>
      <td>1</td>
    </tr>
    <tr>
      <th>544</th>
      <td>GiratinaAltered Forme</td>
      <td>Ghost</td>
      <td>Dragon</td>
      <td>680</td>
      <td>150</td>
      <td>100</td>
      <td>120</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>269</th>
      <td>Lugia</td>
      <td>Psychic</td>
      <td>Flying</td>
      <td>680</td>
      <td>106</td>
      <td>90</td>
      <td>130</td>
      <td>110</td>
      <td>2</td>
    </tr>
    <tr>
      <th>494</th>
      <td>GarchompMega Garchomp</td>
      <td>Dragon</td>
      <td>Ground</td>
      <td>700</td>
      <td>108</td>
      <td>170</td>
      <td>115</td>
      <td>92</td>
      <td>4</td>
    </tr>
    <tr>
      <th>711</th>
      <td>KyuremBlack Kyurem</td>
      <td>Dragon</td>
      <td>Ice</td>
      <td>700</td>
      <td>125</td>
      <td>170</td>
      <td>100</td>
      <td>95</td>
      <td>5</td>
    </tr>
    <tr>
      <th>268</th>
      <td>TyranitarMega Tyranitar</td>
      <td>Rock</td>
      <td>Dark</td>
      <td>700</td>
      <td>100</td>
      <td>164</td>
      <td>150</td>
      <td>71</td>
      <td>2</td>
    </tr>
    <tr>
      <th>796</th>
      <td>DiancieMega Diancie</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
    </tr>
    <tr>
      <th>409</th>
      <td>SalamenceMega Salamence</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>700</td>
      <td>95</td>
      <td>145</td>
      <td>130</td>
      <td>120</td>
      <td>3</td>
    </tr>
    <tr>
      <th>413</th>
      <td>MetagrossMega Metagross</td>
      <td>Steel</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>145</td>
      <td>150</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>420</th>
      <td>LatiosMega Latios</td>
      <td>Dragon</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>130</td>
      <td>100</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>712</th>
      <td>KyuremWhite Kyurem</td>
      <td>Dragon</td>
      <td>Ice</td>
      <td>700</td>
      <td>125</td>
      <td>120</td>
      <td>90</td>
      <td>95</td>
      <td>5</td>
    </tr>
    <tr>
      <th>418</th>
      <td>LatiasMega Latias</td>
      <td>Dragon</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>100</td>
      <td>120</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>552</th>
      <td>Arceus</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>720</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>4</td>
    </tr>
    <tr>
      <th>424</th>
      <td>GroudonPrimal Groudon</td>
      <td>Ground</td>
      <td>Fire</td>
      <td>770</td>
      <td>100</td>
      <td>180</td>
      <td>160</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>422</th>
      <td>KyogrePrimal Kyogre</td>
      <td>Water</td>
      <td>NaN</td>
      <td>770</td>
      <td>100</td>
      <td>150</td>
      <td>90</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>163</th>
      <td>MewtwoMega Mewtwo X</td>
      <td>Psychic</td>
      <td>Fighting</td>
      <td>780</td>
      <td>106</td>
      <td>190</td>
      <td>100</td>
      <td>130</td>
      <td>1</td>
    </tr>
    <tr>
      <th>426</th>
      <td>RayquazaMega Rayquaza</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>780</td>
      <td>105</td>
      <td>180</td>
      <td>100</td>
      <td>115</td>
      <td>3</td>
    </tr>
    <tr>
      <th>164</th>
      <td>MewtwoMega Mewtwo Y</td>
      <td>Psychic</td>
      <td>NaN</td>
      <td>780</td>
      <td>106</td>
      <td>150</td>
      <td>70</td>
      <td>140</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 9 columns</p>
</div>



Q4:如果希望直接在修改源数据的基础上进行排序，添加参数inplace=True即可


```python
#以升序排列为例
df.sort_values(by="总计",inplace=True)
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>206</th>
      <td>Sunkern</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>180</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>322</th>
      <td>Azurill</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>190</td>
      <td>50</td>
      <td>20</td>
      <td>40</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>446</th>
      <td>Kricketot</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>194</td>
      <td>37</td>
      <td>25</td>
      <td>41</td>
      <td>25</td>
      <td>4</td>
    </tr>
    <tr>
      <th>288</th>
      <td>Wurmple</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>45</td>
      <td>35</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Weedle</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Caterpie</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>303</th>
      <td>Ralts</td>
      <td>Psychic</td>
      <td>Fairy</td>
      <td>198</td>
      <td>28</td>
      <td>25</td>
      <td>25</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>732</th>
      <td>Scatterbug</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>200</td>
      <td>38</td>
      <td>35</td>
      <td>40</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Magikarp</td>
      <td>Water</td>
      <td>NaN</td>
      <td>200</td>
      <td>20</td>
      <td>10</td>
      <td>55</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>381</th>
      <td>Feebas</td>
      <td>Water</td>
      <td>NaN</td>
      <td>200</td>
      <td>20</td>
      <td>15</td>
      <td>20</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>289</th>
      <td>Silcoon</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Metapod</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>291</th>
      <td>Cascoon</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>186</th>
      <td>Pichu</td>
      <td>Electric</td>
      <td>NaN</td>
      <td>205</td>
      <td>20</td>
      <td>40</td>
      <td>15</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kakuna</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>209</th>
      <td>Wooper</td>
      <td>Water</td>
      <td>Ground</td>
      <td>210</td>
      <td>55</td>
      <td>45</td>
      <td>45</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>255</th>
      <td>Tyrogue</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>210</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>2</td>
    </tr>
    <tr>
      <th>188</th>
      <td>Igglybuff</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>210</td>
      <td>90</td>
      <td>30</td>
      <td>15</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>733</th>
      <td>Spewpa</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>213</td>
      <td>45</td>
      <td>22</td>
      <td>60</td>
      <td>29</td>
      <td>6</td>
    </tr>
    <tr>
      <th>175</th>
      <td>Sentret</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>215</td>
      <td>35</td>
      <td>46</td>
      <td>34</td>
      <td>20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>187</th>
      <td>Cleffa</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>218</td>
      <td>50</td>
      <td>25</td>
      <td>28</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Seedot</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>220</td>
      <td>40</td>
      <td>40</td>
      <td>50</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>293</th>
      <td>Lotad</td>
      <td>Water</td>
      <td>Grass</td>
      <td>220</td>
      <td>40</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>284</th>
      <td>Poochyena</td>
      <td>Dark</td>
      <td>NaN</td>
      <td>220</td>
      <td>35</td>
      <td>55</td>
      <td>35</td>
      <td>35</td>
      <td>3</td>
    </tr>
    <tr>
      <th>488</th>
      <td>Happiny</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>220</td>
      <td>100</td>
      <td>5</td>
      <td>5</td>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>457</th>
      <td>Burmy</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>224</td>
      <td>40</td>
      <td>29</td>
      <td>45</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>316</th>
      <td>Shedinja</td>
      <td>Bug</td>
      <td>Ghost</td>
      <td>236</td>
      <td>1</td>
      <td>90</td>
      <td>45</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>727</th>
      <td>Bunnelby</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>237</td>
      <td>38</td>
      <td>36</td>
      <td>38</td>
      <td>57</td>
      <td>6</td>
    </tr>
    <tr>
      <th>320</th>
      <td>Makuhita</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>237</td>
      <td>72</td>
      <td>60</td>
      <td>30</td>
      <td>25</td>
      <td>3</td>
    </tr>
    <tr>
      <th>286</th>
      <td>Zigzagoon</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>240</td>
      <td>38</td>
      <td>30</td>
      <td>41</td>
      <td>60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>313</th>
      <td>Slaking</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>670</td>
      <td>150</td>
      <td>160</td>
      <td>100</td>
      <td>100</td>
      <td>3</td>
    </tr>
    <tr>
      <th>421</th>
      <td>Kyogre</td>
      <td>Water</td>
      <td>NaN</td>
      <td>670</td>
      <td>100</td>
      <td>100</td>
      <td>90</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>425</th>
      <td>Rayquaza</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>680</td>
      <td>105</td>
      <td>150</td>
      <td>90</td>
      <td>95</td>
      <td>3</td>
    </tr>
    <tr>
      <th>544</th>
      <td>GiratinaAltered Forme</td>
      <td>Ghost</td>
      <td>Dragon</td>
      <td>680</td>
      <td>150</td>
      <td>100</td>
      <td>120</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>540</th>
      <td>Dialga</td>
      <td>Steel</td>
      <td>Dragon</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>120</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>269</th>
      <td>Lugia</td>
      <td>Psychic</td>
      <td>Flying</td>
      <td>680</td>
      <td>106</td>
      <td>90</td>
      <td>130</td>
      <td>110</td>
      <td>2</td>
    </tr>
    <tr>
      <th>706</th>
      <td>Reshiram</td>
      <td>Dragon</td>
      <td>Fire</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>270</th>
      <td>Ho-oh</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>680</td>
      <td>106</td>
      <td>130</td>
      <td>90</td>
      <td>90</td>
      <td>2</td>
    </tr>
    <tr>
      <th>707</th>
      <td>Zekrom</td>
      <td>Dragon</td>
      <td>Electric</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
      <td>120</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>541</th>
      <td>Palkia</td>
      <td>Water</td>
      <td>Dragon</td>
      <td>680</td>
      <td>90</td>
      <td>120</td>
      <td>100</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>792</th>
      <td>Xerneas</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>793</th>
      <td>Yveltal</td>
      <td>Dark</td>
      <td>Flying</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>798</th>
      <td>HoopaHoopa Unbound</td>
      <td>Psychic</td>
      <td>Dark</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
      <td>60</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>162</th>
      <td>Mewtwo</td>
      <td>Psychic</td>
      <td>NaN</td>
      <td>680</td>
      <td>106</td>
      <td>110</td>
      <td>90</td>
      <td>130</td>
      <td>1</td>
    </tr>
    <tr>
      <th>545</th>
      <td>GiratinaOrigin Forme</td>
      <td>Ghost</td>
      <td>Dragon</td>
      <td>680</td>
      <td>150</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>4</td>
    </tr>
    <tr>
      <th>712</th>
      <td>KyuremWhite Kyurem</td>
      <td>Dragon</td>
      <td>Ice</td>
      <td>700</td>
      <td>125</td>
      <td>120</td>
      <td>90</td>
      <td>95</td>
      <td>5</td>
    </tr>
    <tr>
      <th>494</th>
      <td>GarchompMega Garchomp</td>
      <td>Dragon</td>
      <td>Ground</td>
      <td>700</td>
      <td>108</td>
      <td>170</td>
      <td>115</td>
      <td>92</td>
      <td>4</td>
    </tr>
    <tr>
      <th>409</th>
      <td>SalamenceMega Salamence</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>700</td>
      <td>95</td>
      <td>145</td>
      <td>130</td>
      <td>120</td>
      <td>3</td>
    </tr>
    <tr>
      <th>413</th>
      <td>MetagrossMega Metagross</td>
      <td>Steel</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>145</td>
      <td>150</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>418</th>
      <td>LatiasMega Latias</td>
      <td>Dragon</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>100</td>
      <td>120</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>420</th>
      <td>LatiosMega Latios</td>
      <td>Dragon</td>
      <td>Psychic</td>
      <td>700</td>
      <td>80</td>
      <td>130</td>
      <td>100</td>
      <td>110</td>
      <td>3</td>
    </tr>
    <tr>
      <th>796</th>
      <td>DiancieMega Diancie</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
    </tr>
    <tr>
      <th>268</th>
      <td>TyranitarMega Tyranitar</td>
      <td>Rock</td>
      <td>Dark</td>
      <td>700</td>
      <td>100</td>
      <td>164</td>
      <td>150</td>
      <td>71</td>
      <td>2</td>
    </tr>
    <tr>
      <th>711</th>
      <td>KyuremBlack Kyurem</td>
      <td>Dragon</td>
      <td>Ice</td>
      <td>700</td>
      <td>125</td>
      <td>170</td>
      <td>100</td>
      <td>95</td>
      <td>5</td>
    </tr>
    <tr>
      <th>552</th>
      <td>Arceus</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>720</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>4</td>
    </tr>
    <tr>
      <th>424</th>
      <td>GroudonPrimal Groudon</td>
      <td>Ground</td>
      <td>Fire</td>
      <td>770</td>
      <td>100</td>
      <td>180</td>
      <td>160</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>422</th>
      <td>KyogrePrimal Kyogre</td>
      <td>Water</td>
      <td>NaN</td>
      <td>770</td>
      <td>100</td>
      <td>150</td>
      <td>90</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>426</th>
      <td>RayquazaMega Rayquaza</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>780</td>
      <td>105</td>
      <td>180</td>
      <td>100</td>
      <td>115</td>
      <td>3</td>
    </tr>
    <tr>
      <th>164</th>
      <td>MewtwoMega Mewtwo Y</td>
      <td>Psychic</td>
      <td>NaN</td>
      <td>780</td>
      <td>106</td>
      <td>150</td>
      <td>70</td>
      <td>140</td>
      <td>1</td>
    </tr>
    <tr>
      <th>163</th>
      <td>MewtwoMega Mewtwo X</td>
      <td>Psychic</td>
      <td>Fighting</td>
      <td>780</td>
      <td>106</td>
      <td>190</td>
      <td>100</td>
      <td>130</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 9 columns</p>
</div>



+ 根据索引排序


```python
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Wartortle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>405</td>
      <td>59</td>
      <td>63</td>
      <td>80</td>
      <td>58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Blastoise</td>
      <td>Water</td>
      <td>NaN</td>
      <td>530</td>
      <td>79</td>
      <td>83</td>
      <td>100</td>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BlastoiseMega Blastoise</td>
      <td>Water</td>
      <td>NaN</td>
      <td>630</td>
      <td>79</td>
      <td>103</td>
      <td>120</td>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Caterpie</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Metapod</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Butterfree</td>
      <td>Bug</td>
      <td>Flying</td>
      <td>395</td>
      <td>60</td>
      <td>45</td>
      <td>50</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Weedle</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kakuna</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Beedrill</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>395</td>
      <td>65</td>
      <td>90</td>
      <td>40</td>
      <td>75</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BeedrillMega Beedrill</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>495</td>
      <td>65</td>
      <td>150</td>
      <td>40</td>
      <td>145</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Pidgey</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>251</td>
      <td>40</td>
      <td>45</td>
      <td>40</td>
      <td>56</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Pidgeotto</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>349</td>
      <td>63</td>
      <td>60</td>
      <td>55</td>
      <td>71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Pidgeot</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>479</td>
      <td>83</td>
      <td>80</td>
      <td>75</td>
      <td>101</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PidgeotMega Pidgeot</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>579</td>
      <td>83</td>
      <td>80</td>
      <td>80</td>
      <td>121</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Rattata</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>253</td>
      <td>30</td>
      <td>56</td>
      <td>35</td>
      <td>72</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Raticate</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>413</td>
      <td>55</td>
      <td>81</td>
      <td>60</td>
      <td>97</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Spearow</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>262</td>
      <td>40</td>
      <td>60</td>
      <td>30</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Fearow</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>442</td>
      <td>65</td>
      <td>90</td>
      <td>65</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Ekans</td>
      <td>Poison</td>
      <td>NaN</td>
      <td>288</td>
      <td>35</td>
      <td>60</td>
      <td>44</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Arbok</td>
      <td>Poison</td>
      <td>NaN</td>
      <td>438</td>
      <td>60</td>
      <td>85</td>
      <td>69</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>770</th>
      <td>Sylveon</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>525</td>
      <td>95</td>
      <td>65</td>
      <td>65</td>
      <td>60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>771</th>
      <td>Hawlucha</td>
      <td>Fighting</td>
      <td>Flying</td>
      <td>500</td>
      <td>78</td>
      <td>92</td>
      <td>75</td>
      <td>118</td>
      <td>6</td>
    </tr>
    <tr>
      <th>772</th>
      <td>Dedenne</td>
      <td>Electric</td>
      <td>Fairy</td>
      <td>431</td>
      <td>67</td>
      <td>58</td>
      <td>57</td>
      <td>101</td>
      <td>6</td>
    </tr>
    <tr>
      <th>773</th>
      <td>Carbink</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>500</td>
      <td>50</td>
      <td>50</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th>774</th>
      <td>Goomy</td>
      <td>Dragon</td>
      <td>NaN</td>
      <td>300</td>
      <td>45</td>
      <td>50</td>
      <td>35</td>
      <td>40</td>
      <td>6</td>
    </tr>
    <tr>
      <th>775</th>
      <td>Sliggoo</td>
      <td>Dragon</td>
      <td>NaN</td>
      <td>452</td>
      <td>68</td>
      <td>75</td>
      <td>53</td>
      <td>60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>776</th>
      <td>Goodra</td>
      <td>Dragon</td>
      <td>NaN</td>
      <td>600</td>
      <td>90</td>
      <td>100</td>
      <td>70</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>777</th>
      <td>Klefki</td>
      <td>Steel</td>
      <td>Fairy</td>
      <td>470</td>
      <td>57</td>
      <td>80</td>
      <td>91</td>
      <td>75</td>
      <td>6</td>
    </tr>
    <tr>
      <th>778</th>
      <td>Phantump</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>309</td>
      <td>43</td>
      <td>70</td>
      <td>48</td>
      <td>38</td>
      <td>6</td>
    </tr>
    <tr>
      <th>779</th>
      <td>Trevenant</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>474</td>
      <td>85</td>
      <td>110</td>
      <td>76</td>
      <td>56</td>
      <td>6</td>
    </tr>
    <tr>
      <th>780</th>
      <td>PumpkabooAverage Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>49</td>
      <td>66</td>
      <td>70</td>
      <td>51</td>
      <td>6</td>
    </tr>
    <tr>
      <th>781</th>
      <td>PumpkabooSmall Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>44</td>
      <td>66</td>
      <td>70</td>
      <td>56</td>
      <td>6</td>
    </tr>
    <tr>
      <th>782</th>
      <td>PumpkabooLarge Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>54</td>
      <td>66</td>
      <td>70</td>
      <td>46</td>
      <td>6</td>
    </tr>
    <tr>
      <th>783</th>
      <td>PumpkabooSuper Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>59</td>
      <td>66</td>
      <td>70</td>
      <td>41</td>
      <td>6</td>
    </tr>
    <tr>
      <th>784</th>
      <td>GourgeistAverage Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>65</td>
      <td>90</td>
      <td>122</td>
      <td>84</td>
      <td>6</td>
    </tr>
    <tr>
      <th>785</th>
      <td>GourgeistSmall Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>55</td>
      <td>85</td>
      <td>122</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>786</th>
      <td>GourgeistLarge Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>75</td>
      <td>95</td>
      <td>122</td>
      <td>69</td>
      <td>6</td>
    </tr>
    <tr>
      <th>787</th>
      <td>GourgeistSuper Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>85</td>
      <td>100</td>
      <td>122</td>
      <td>54</td>
      <td>6</td>
    </tr>
    <tr>
      <th>788</th>
      <td>Bergmite</td>
      <td>Ice</td>
      <td>NaN</td>
      <td>304</td>
      <td>55</td>
      <td>69</td>
      <td>85</td>
      <td>28</td>
      <td>6</td>
    </tr>
    <tr>
      <th>789</th>
      <td>Avalugg</td>
      <td>Ice</td>
      <td>NaN</td>
      <td>514</td>
      <td>95</td>
      <td>117</td>
      <td>184</td>
      <td>28</td>
      <td>6</td>
    </tr>
    <tr>
      <th>790</th>
      <td>Noibat</td>
      <td>Flying</td>
      <td>Dragon</td>
      <td>245</td>
      <td>40</td>
      <td>30</td>
      <td>35</td>
      <td>55</td>
      <td>6</td>
    </tr>
    <tr>
      <th>791</th>
      <td>Noivern</td>
      <td>Flying</td>
      <td>Dragon</td>
      <td>535</td>
      <td>85</td>
      <td>70</td>
      <td>80</td>
      <td>123</td>
      <td>6</td>
    </tr>
    <tr>
      <th>792</th>
      <td>Xerneas</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>793</th>
      <td>Yveltal</td>
      <td>Dark</td>
      <td>Flying</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>794</th>
      <td>Zygarde50% Forme</td>
      <td>Dragon</td>
      <td>Ground</td>
      <td>600</td>
      <td>108</td>
      <td>100</td>
      <td>121</td>
      <td>95</td>
      <td>6</td>
    </tr>
    <tr>
      <th>795</th>
      <td>Diancie</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>600</td>
      <td>50</td>
      <td>100</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th>796</th>
      <td>DiancieMega Diancie</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
    </tr>
    <tr>
      <th>797</th>
      <td>HoopaHoopa Confined</td>
      <td>Psychic</td>
      <td>Ghost</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
      <td>60</td>
      <td>70</td>
      <td>6</td>
    </tr>
    <tr>
      <th>798</th>
      <td>HoopaHoopa Unbound</td>
      <td>Psychic</td>
      <td>Dark</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
      <td>60</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>799</th>
      <td>Volcanion</td>
      <td>Fire</td>
      <td>Water</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
      <td>120</td>
      <td>70</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 9 columns</p>
</div>



（1）升序排列


```python
df.sort_index(inplace=True)
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Wartortle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>405</td>
      <td>59</td>
      <td>63</td>
      <td>80</td>
      <td>58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Blastoise</td>
      <td>Water</td>
      <td>NaN</td>
      <td>530</td>
      <td>79</td>
      <td>83</td>
      <td>100</td>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BlastoiseMega Blastoise</td>
      <td>Water</td>
      <td>NaN</td>
      <td>630</td>
      <td>79</td>
      <td>103</td>
      <td>120</td>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Caterpie</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Metapod</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Butterfree</td>
      <td>Bug</td>
      <td>Flying</td>
      <td>395</td>
      <td>60</td>
      <td>45</td>
      <td>50</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Weedle</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kakuna</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Beedrill</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>395</td>
      <td>65</td>
      <td>90</td>
      <td>40</td>
      <td>75</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BeedrillMega Beedrill</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>495</td>
      <td>65</td>
      <td>150</td>
      <td>40</td>
      <td>145</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Pidgey</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>251</td>
      <td>40</td>
      <td>45</td>
      <td>40</td>
      <td>56</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Pidgeotto</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>349</td>
      <td>63</td>
      <td>60</td>
      <td>55</td>
      <td>71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Pidgeot</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>479</td>
      <td>83</td>
      <td>80</td>
      <td>75</td>
      <td>101</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PidgeotMega Pidgeot</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>579</td>
      <td>83</td>
      <td>80</td>
      <td>80</td>
      <td>121</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Rattata</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>253</td>
      <td>30</td>
      <td>56</td>
      <td>35</td>
      <td>72</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Raticate</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>413</td>
      <td>55</td>
      <td>81</td>
      <td>60</td>
      <td>97</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Spearow</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>262</td>
      <td>40</td>
      <td>60</td>
      <td>30</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Fearow</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>442</td>
      <td>65</td>
      <td>90</td>
      <td>65</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Ekans</td>
      <td>Poison</td>
      <td>NaN</td>
      <td>288</td>
      <td>35</td>
      <td>60</td>
      <td>44</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Arbok</td>
      <td>Poison</td>
      <td>NaN</td>
      <td>438</td>
      <td>60</td>
      <td>85</td>
      <td>69</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>770</th>
      <td>Sylveon</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>525</td>
      <td>95</td>
      <td>65</td>
      <td>65</td>
      <td>60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>771</th>
      <td>Hawlucha</td>
      <td>Fighting</td>
      <td>Flying</td>
      <td>500</td>
      <td>78</td>
      <td>92</td>
      <td>75</td>
      <td>118</td>
      <td>6</td>
    </tr>
    <tr>
      <th>772</th>
      <td>Dedenne</td>
      <td>Electric</td>
      <td>Fairy</td>
      <td>431</td>
      <td>67</td>
      <td>58</td>
      <td>57</td>
      <td>101</td>
      <td>6</td>
    </tr>
    <tr>
      <th>773</th>
      <td>Carbink</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>500</td>
      <td>50</td>
      <td>50</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th>774</th>
      <td>Goomy</td>
      <td>Dragon</td>
      <td>NaN</td>
      <td>300</td>
      <td>45</td>
      <td>50</td>
      <td>35</td>
      <td>40</td>
      <td>6</td>
    </tr>
    <tr>
      <th>775</th>
      <td>Sliggoo</td>
      <td>Dragon</td>
      <td>NaN</td>
      <td>452</td>
      <td>68</td>
      <td>75</td>
      <td>53</td>
      <td>60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>776</th>
      <td>Goodra</td>
      <td>Dragon</td>
      <td>NaN</td>
      <td>600</td>
      <td>90</td>
      <td>100</td>
      <td>70</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>777</th>
      <td>Klefki</td>
      <td>Steel</td>
      <td>Fairy</td>
      <td>470</td>
      <td>57</td>
      <td>80</td>
      <td>91</td>
      <td>75</td>
      <td>6</td>
    </tr>
    <tr>
      <th>778</th>
      <td>Phantump</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>309</td>
      <td>43</td>
      <td>70</td>
      <td>48</td>
      <td>38</td>
      <td>6</td>
    </tr>
    <tr>
      <th>779</th>
      <td>Trevenant</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>474</td>
      <td>85</td>
      <td>110</td>
      <td>76</td>
      <td>56</td>
      <td>6</td>
    </tr>
    <tr>
      <th>780</th>
      <td>PumpkabooAverage Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>49</td>
      <td>66</td>
      <td>70</td>
      <td>51</td>
      <td>6</td>
    </tr>
    <tr>
      <th>781</th>
      <td>PumpkabooSmall Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>44</td>
      <td>66</td>
      <td>70</td>
      <td>56</td>
      <td>6</td>
    </tr>
    <tr>
      <th>782</th>
      <td>PumpkabooLarge Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>54</td>
      <td>66</td>
      <td>70</td>
      <td>46</td>
      <td>6</td>
    </tr>
    <tr>
      <th>783</th>
      <td>PumpkabooSuper Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>59</td>
      <td>66</td>
      <td>70</td>
      <td>41</td>
      <td>6</td>
    </tr>
    <tr>
      <th>784</th>
      <td>GourgeistAverage Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>65</td>
      <td>90</td>
      <td>122</td>
      <td>84</td>
      <td>6</td>
    </tr>
    <tr>
      <th>785</th>
      <td>GourgeistSmall Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>55</td>
      <td>85</td>
      <td>122</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>786</th>
      <td>GourgeistLarge Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>75</td>
      <td>95</td>
      <td>122</td>
      <td>69</td>
      <td>6</td>
    </tr>
    <tr>
      <th>787</th>
      <td>GourgeistSuper Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>85</td>
      <td>100</td>
      <td>122</td>
      <td>54</td>
      <td>6</td>
    </tr>
    <tr>
      <th>788</th>
      <td>Bergmite</td>
      <td>Ice</td>
      <td>NaN</td>
      <td>304</td>
      <td>55</td>
      <td>69</td>
      <td>85</td>
      <td>28</td>
      <td>6</td>
    </tr>
    <tr>
      <th>789</th>
      <td>Avalugg</td>
      <td>Ice</td>
      <td>NaN</td>
      <td>514</td>
      <td>95</td>
      <td>117</td>
      <td>184</td>
      <td>28</td>
      <td>6</td>
    </tr>
    <tr>
      <th>790</th>
      <td>Noibat</td>
      <td>Flying</td>
      <td>Dragon</td>
      <td>245</td>
      <td>40</td>
      <td>30</td>
      <td>35</td>
      <td>55</td>
      <td>6</td>
    </tr>
    <tr>
      <th>791</th>
      <td>Noivern</td>
      <td>Flying</td>
      <td>Dragon</td>
      <td>535</td>
      <td>85</td>
      <td>70</td>
      <td>80</td>
      <td>123</td>
      <td>6</td>
    </tr>
    <tr>
      <th>792</th>
      <td>Xerneas</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>793</th>
      <td>Yveltal</td>
      <td>Dark</td>
      <td>Flying</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>794</th>
      <td>Zygarde50% Forme</td>
      <td>Dragon</td>
      <td>Ground</td>
      <td>600</td>
      <td>108</td>
      <td>100</td>
      <td>121</td>
      <td>95</td>
      <td>6</td>
    </tr>
    <tr>
      <th>795</th>
      <td>Diancie</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>600</td>
      <td>50</td>
      <td>100</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th>796</th>
      <td>DiancieMega Diancie</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
    </tr>
    <tr>
      <th>797</th>
      <td>HoopaHoopa Confined</td>
      <td>Psychic</td>
      <td>Ghost</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
      <td>60</td>
      <td>70</td>
      <td>6</td>
    </tr>
    <tr>
      <th>798</th>
      <td>HoopaHoopa Unbound</td>
      <td>Psychic</td>
      <td>Dark</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
      <td>60</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>799</th>
      <td>Volcanion</td>
      <td>Fire</td>
      <td>Water</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
      <td>120</td>
      <td>70</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 9 columns</p>
</div>



（2）降序排列


```python
df.sort_index(ascending=False,inplace=True)
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>799</th>
      <td>Volcanion</td>
      <td>Fire</td>
      <td>Water</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
      <td>120</td>
      <td>70</td>
      <td>6</td>
    </tr>
    <tr>
      <th>798</th>
      <td>HoopaHoopa Unbound</td>
      <td>Psychic</td>
      <td>Dark</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
      <td>60</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>797</th>
      <td>HoopaHoopa Confined</td>
      <td>Psychic</td>
      <td>Ghost</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
      <td>60</td>
      <td>70</td>
      <td>6</td>
    </tr>
    <tr>
      <th>796</th>
      <td>DiancieMega Diancie</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
    </tr>
    <tr>
      <th>795</th>
      <td>Diancie</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>600</td>
      <td>50</td>
      <td>100</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th>794</th>
      <td>Zygarde50% Forme</td>
      <td>Dragon</td>
      <td>Ground</td>
      <td>600</td>
      <td>108</td>
      <td>100</td>
      <td>121</td>
      <td>95</td>
      <td>6</td>
    </tr>
    <tr>
      <th>793</th>
      <td>Yveltal</td>
      <td>Dark</td>
      <td>Flying</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>792</th>
      <td>Xerneas</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>791</th>
      <td>Noivern</td>
      <td>Flying</td>
      <td>Dragon</td>
      <td>535</td>
      <td>85</td>
      <td>70</td>
      <td>80</td>
      <td>123</td>
      <td>6</td>
    </tr>
    <tr>
      <th>790</th>
      <td>Noibat</td>
      <td>Flying</td>
      <td>Dragon</td>
      <td>245</td>
      <td>40</td>
      <td>30</td>
      <td>35</td>
      <td>55</td>
      <td>6</td>
    </tr>
    <tr>
      <th>789</th>
      <td>Avalugg</td>
      <td>Ice</td>
      <td>NaN</td>
      <td>514</td>
      <td>95</td>
      <td>117</td>
      <td>184</td>
      <td>28</td>
      <td>6</td>
    </tr>
    <tr>
      <th>788</th>
      <td>Bergmite</td>
      <td>Ice</td>
      <td>NaN</td>
      <td>304</td>
      <td>55</td>
      <td>69</td>
      <td>85</td>
      <td>28</td>
      <td>6</td>
    </tr>
    <tr>
      <th>787</th>
      <td>GourgeistSuper Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>85</td>
      <td>100</td>
      <td>122</td>
      <td>54</td>
      <td>6</td>
    </tr>
    <tr>
      <th>786</th>
      <td>GourgeistLarge Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>75</td>
      <td>95</td>
      <td>122</td>
      <td>69</td>
      <td>6</td>
    </tr>
    <tr>
      <th>785</th>
      <td>GourgeistSmall Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>55</td>
      <td>85</td>
      <td>122</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>784</th>
      <td>GourgeistAverage Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>65</td>
      <td>90</td>
      <td>122</td>
      <td>84</td>
      <td>6</td>
    </tr>
    <tr>
      <th>783</th>
      <td>PumpkabooSuper Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>59</td>
      <td>66</td>
      <td>70</td>
      <td>41</td>
      <td>6</td>
    </tr>
    <tr>
      <th>782</th>
      <td>PumpkabooLarge Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>54</td>
      <td>66</td>
      <td>70</td>
      <td>46</td>
      <td>6</td>
    </tr>
    <tr>
      <th>781</th>
      <td>PumpkabooSmall Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>44</td>
      <td>66</td>
      <td>70</td>
      <td>56</td>
      <td>6</td>
    </tr>
    <tr>
      <th>780</th>
      <td>PumpkabooAverage Size</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>49</td>
      <td>66</td>
      <td>70</td>
      <td>51</td>
      <td>6</td>
    </tr>
    <tr>
      <th>779</th>
      <td>Trevenant</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>474</td>
      <td>85</td>
      <td>110</td>
      <td>76</td>
      <td>56</td>
      <td>6</td>
    </tr>
    <tr>
      <th>778</th>
      <td>Phantump</td>
      <td>Ghost</td>
      <td>Grass</td>
      <td>309</td>
      <td>43</td>
      <td>70</td>
      <td>48</td>
      <td>38</td>
      <td>6</td>
    </tr>
    <tr>
      <th>777</th>
      <td>Klefki</td>
      <td>Steel</td>
      <td>Fairy</td>
      <td>470</td>
      <td>57</td>
      <td>80</td>
      <td>91</td>
      <td>75</td>
      <td>6</td>
    </tr>
    <tr>
      <th>776</th>
      <td>Goodra</td>
      <td>Dragon</td>
      <td>NaN</td>
      <td>600</td>
      <td>90</td>
      <td>100</td>
      <td>70</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>775</th>
      <td>Sliggoo</td>
      <td>Dragon</td>
      <td>NaN</td>
      <td>452</td>
      <td>68</td>
      <td>75</td>
      <td>53</td>
      <td>60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>774</th>
      <td>Goomy</td>
      <td>Dragon</td>
      <td>NaN</td>
      <td>300</td>
      <td>45</td>
      <td>50</td>
      <td>35</td>
      <td>40</td>
      <td>6</td>
    </tr>
    <tr>
      <th>773</th>
      <td>Carbink</td>
      <td>Rock</td>
      <td>Fairy</td>
      <td>500</td>
      <td>50</td>
      <td>50</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th>772</th>
      <td>Dedenne</td>
      <td>Electric</td>
      <td>Fairy</td>
      <td>431</td>
      <td>67</td>
      <td>58</td>
      <td>57</td>
      <td>101</td>
      <td>6</td>
    </tr>
    <tr>
      <th>771</th>
      <td>Hawlucha</td>
      <td>Fighting</td>
      <td>Flying</td>
      <td>500</td>
      <td>78</td>
      <td>92</td>
      <td>75</td>
      <td>118</td>
      <td>6</td>
    </tr>
    <tr>
      <th>770</th>
      <td>Sylveon</td>
      <td>Fairy</td>
      <td>NaN</td>
      <td>525</td>
      <td>95</td>
      <td>65</td>
      <td>65</td>
      <td>60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Arbok</td>
      <td>Poison</td>
      <td>NaN</td>
      <td>438</td>
      <td>60</td>
      <td>85</td>
      <td>69</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Ekans</td>
      <td>Poison</td>
      <td>NaN</td>
      <td>288</td>
      <td>35</td>
      <td>60</td>
      <td>44</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Fearow</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>442</td>
      <td>65</td>
      <td>90</td>
      <td>65</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Spearow</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>262</td>
      <td>40</td>
      <td>60</td>
      <td>30</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Raticate</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>413</td>
      <td>55</td>
      <td>81</td>
      <td>60</td>
      <td>97</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Rattata</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>253</td>
      <td>30</td>
      <td>56</td>
      <td>35</td>
      <td>72</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PidgeotMega Pidgeot</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>579</td>
      <td>83</td>
      <td>80</td>
      <td>80</td>
      <td>121</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Pidgeot</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>479</td>
      <td>83</td>
      <td>80</td>
      <td>75</td>
      <td>101</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Pidgeotto</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>349</td>
      <td>63</td>
      <td>60</td>
      <td>55</td>
      <td>71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Pidgey</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>251</td>
      <td>40</td>
      <td>45</td>
      <td>40</td>
      <td>56</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BeedrillMega Beedrill</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>495</td>
      <td>65</td>
      <td>150</td>
      <td>40</td>
      <td>145</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Beedrill</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>395</td>
      <td>65</td>
      <td>90</td>
      <td>40</td>
      <td>75</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kakuna</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Weedle</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Butterfree</td>
      <td>Bug</td>
      <td>Flying</td>
      <td>395</td>
      <td>60</td>
      <td>45</td>
      <td>50</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Metapod</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Caterpie</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BlastoiseMega Blastoise</td>
      <td>Water</td>
      <td>NaN</td>
      <td>630</td>
      <td>79</td>
      <td>103</td>
      <td>120</td>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Blastoise</td>
      <td>Water</td>
      <td>NaN</td>
      <td>530</td>
      <td>79</td>
      <td>83</td>
      <td>100</td>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Wartortle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>405</td>
      <td>59</td>
      <td>63</td>
      <td>80</td>
      <td>58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 9 columns</p>
</div>



## 八.排名


```python
#示例数据
df = pd.read_csv("pokemon_data.csv",encoding="gbk")
df_rank = df.head(10)
df_rank
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



做个中国式排名：希望通过总计列的大小进行排名，并新增一个排名列，值大的排名靠前，值小的排名靠后，如果值一样则占同一个排名位，排名位是连续的，不存在间断


```python
import warnings
```


```python
warnings.filterwarnings("ignore")
```


```python
df_rank["rank"] = df_rank["总计"].rank(ascending=False,method="dense")
df_rank
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>姓名</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



## 九.转置


```python
#示例数据
df5 = pd.DataFrame({"cost":[800,1000,1200],"sale":[1000,2000,2500]},index=["1月","2月","3月"])
df5
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cost</th>
      <th>sale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1月</th>
      <td>800</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>2月</th>
      <td>1000</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>3月</th>
      <td>1200</td>
      <td>2500</td>
    </tr>
  </tbody>
</table>
</div>




```python
#行索引和列索引进行转置
df5 = df5.T
df5
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1月</th>
      <th>2月</th>
      <th>3月</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cost</th>
      <td>800</td>
      <td>1000</td>
      <td>1200</td>
    </tr>
    <tr>
      <th>sale</th>
      <td>1000</td>
      <td>2000</td>
      <td>2500</td>
    </tr>
  </tbody>
</table>
</div>


