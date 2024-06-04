```python
import pandas as pd
import numpy as np
```


```python
df = pd.read_csv("pokemon_data.csv",encoding="gbk")
df.head()
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
  </tbody>
</table>
</div>



## 一.简单随机抽样


```python
#简单随机抽样，随机抽取5行数据
df.sample(n=5)
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
      <th>600</th>
      <td>Sewaddle</td>
      <td>Bug</td>
      <td>Grass</td>
      <td>310</td>
      <td>45</td>
      <td>53</td>
      <td>70</td>
      <td>42</td>
      <td>5</td>
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
      <th>299</th>
      <td>Taillow</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>270</td>
      <td>40</td>
      <td>55</td>
      <td>30</td>
      <td>85</td>
      <td>3</td>
    </tr>
    <tr>
      <th>156</th>
      <td>Articuno</td>
      <td>Ice</td>
      <td>Flying</td>
      <td>580</td>
      <td>90</td>
      <td>85</td>
      <td>100</td>
      <td>85</td>
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
</div>




```python
#设置抽样的权重，权重高的更有希望被选取
w = [0.2,0.3,0.5]
df.head(3).sample(n=2,weights=w)
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
  </tbody>
</table>
</div>




```python
df.head(3)
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
  </tbody>
</table>
</div>



抽样后是否放回，由replace参数控制


```python
df.head(5)
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
  </tbody>
</table>
</div>




```python
#抽样后不放回
df.head(5).sample(n=4,replace=False)
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
  </tbody>
</table>
</div>




```python
#抽样后放回
df.head(5).sample(n=4,replace=True)
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
</div>



## 二.描述性统计


```python
df.describe().round(1)
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
      <th>count</th>
      <td>800.0</td>
      <td>800.0</td>
      <td>800.0</td>
      <td>800.0</td>
      <td>800.0</td>
      <td>800.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>435.1</td>
      <td>69.3</td>
      <td>79.0</td>
      <td>73.8</td>
      <td>68.3</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>std</th>
      <td>120.0</td>
      <td>25.5</td>
      <td>32.5</td>
      <td>31.2</td>
      <td>29.1</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>min</th>
      <td>180.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>330.0</td>
      <td>50.0</td>
      <td>55.0</td>
      <td>50.0</td>
      <td>45.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>450.0</td>
      <td>65.0</td>
      <td>75.0</td>
      <td>70.0</td>
      <td>65.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>515.0</td>
      <td>80.0</td>
      <td>100.0</td>
      <td>90.0</td>
      <td>90.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>780.0</td>
      <td>255.0</td>
      <td>190.0</td>
      <td>230.0</td>
      <td>180.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#均值
df["攻击力"].mean()
```




    79.00125




```python
#标准差
df["攻击力"].std()
```




    32.45736586949843




```python
#求和
df["攻击力"].sum()
```




    63201




```python
#中位数
df["攻击力"].median()
```




    75.0




```python
#最大值或最小值的索引idxmax,idxmin
df["攻击力"].idxmax()
```




    163




```python
#验证最大值
df["攻击力"].loc[163]
```




    190




```python
#验证最大值
df["攻击力"].max()
```




    190




```python
#累计值
df["攻击力"].cumsum()
```




    0         49
    1        111
    2        193
    3        293
    4        345
    5        409
    6        493
    7        623
    8        727
    9        775
    10       838
    11       921
    12      1024
    13      1054
    14      1074
    15      1119
    16      1154
    17      1179
    18      1269
    19      1419
    20      1464
    21      1524
    22      1604
    23      1684
    24      1740
    25      1821
    26      1881
    27      1971
    28      2031
    29      2116
           ...  
    770    60594
    771    60686
    772    60744
    773    60794
    774    60844
    775    60919
    776    61019
    777    61099
    778    61169
    779    61279
    780    61345
    781    61411
    782    61477
    783    61543
    784    61633
    785    61718
    786    61813
    787    61913
    788    61982
    789    62099
    790    62129
    791    62199
    792    62330
    793    62461
    794    62561
    795    62661
    796    62821
    797    62931
    798    63091
    799    63201
    Name: 攻击力, Length: 800, dtype: int64




```python
#频数分布
df["类型1"].value_counts()
```




    Water       112
    Normal       98
    Grass        70
    Bug          69
    Psychic      57
    Fire         52
    Electric     44
    Rock         44
    Ground       32
    Ghost        32
    Dragon       32
    Dark         31
    Poison       28
    Fighting     27
    Steel        27
    Ice          24
    Fairy        17
    Flying        4
    Name: 类型1, dtype: int64



## 三.协方差与相关性


```python
#两变量的协方差
df["攻击力"].cov(df["防御力"])
```




    444.01020963704605




```python
#所有变量间的协方差
df.cov()
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
      <th>总计</th>
      <td>14391.130907</td>
      <td>1895.360178</td>
      <td>2866.571837</td>
      <td>2292.356589</td>
      <td>2007.841358</td>
      <td>9.642619</td>
    </tr>
    <tr>
      <th>生命值</th>
      <td>1895.360178</td>
      <td>652.019323</td>
      <td>350.068512</td>
      <td>190.801755</td>
      <td>130.565153</td>
      <td>2.489341</td>
    </tr>
    <tr>
      <th>攻击力</th>
      <td>2866.571837</td>
      <td>350.068512</td>
      <td>1053.480599</td>
      <td>444.010210</td>
      <td>359.595397</td>
      <td>2.774313</td>
    </tr>
    <tr>
      <th>防御力</th>
      <td>2292.356589</td>
      <td>190.801755</td>
      <td>444.010210</td>
      <td>972.410707</td>
      <td>13.798454</td>
      <td>2.197487</td>
    </tr>
    <tr>
      <th>速度</th>
      <td>2007.841358</td>
      <td>130.565153</td>
      <td>359.595397</td>
      <td>13.798454</td>
      <td>844.511133</td>
      <td>-1.116236</td>
    </tr>
    <tr>
      <th>时代</th>
      <td>9.642619</td>
      <td>2.489341</td>
      <td>2.774313</td>
      <td>2.197487</td>
      <td>-1.116236</td>
      <td>2.759886</td>
    </tr>
  </tbody>
</table>
</div>




```python
#两个变量间的相关系数
df["攻击力"].corr(df["防御力"])
```




    0.43868705511848927




```python
#所有变量间的相关系数
df.corr()
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
      <th>总计</th>
      <td>1.000000</td>
      <td>0.618748</td>
      <td>0.736211</td>
      <td>0.612787</td>
      <td>0.575943</td>
      <td>0.048384</td>
    </tr>
    <tr>
      <th>生命值</th>
      <td>0.618748</td>
      <td>1.000000</td>
      <td>0.422386</td>
      <td>0.239622</td>
      <td>0.175952</td>
      <td>0.058683</td>
    </tr>
    <tr>
      <th>攻击力</th>
      <td>0.736211</td>
      <td>0.422386</td>
      <td>1.000000</td>
      <td>0.438687</td>
      <td>0.381240</td>
      <td>0.051451</td>
    </tr>
    <tr>
      <th>防御力</th>
      <td>0.612787</td>
      <td>0.239622</td>
      <td>0.438687</td>
      <td>1.000000</td>
      <td>0.015227</td>
      <td>0.042419</td>
    </tr>
    <tr>
      <th>速度</th>
      <td>0.575943</td>
      <td>0.175952</td>
      <td>0.381240</td>
      <td>0.015227</td>
      <td>1.000000</td>
      <td>-0.023121</td>
    </tr>
    <tr>
      <th>时代</th>
      <td>0.048384</td>
      <td>0.058683</td>
      <td>0.051451</td>
      <td>0.042419</td>
      <td>-0.023121</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


