导入pandas和numpy库


```python
import pandas as pd
import numpy as np
```

## 一.缺失值处理


```python
#示例数据
df = pd.read_csv("pokemon_data.csv",encoding="gbk")
#查看数据前十行
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



### 1.判断缺失值

+ 判断数据表所有数据中的缺失值


```python
df.isnull()
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
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
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
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>771</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>772</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>773</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>774</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>775</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>776</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>777</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>778</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>779</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>780</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>781</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>782</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>783</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>784</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>785</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>786</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>787</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>788</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>789</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>790</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>791</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>792</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>793</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>794</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>795</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>796</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>797</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>798</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>799</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 9 columns</p>
</div>



+ 判断数据表某一列的缺失值


```python
df["类型2"].isnull()
```




    0      False
    1      False
    2      False
    3      False
    4       True
    5       True
    6      False
    7      False
    8      False
    9       True
    10      True
    11      True
    12      True
    13      True
    14      True
    15     False
    16     False
    17     False
    18     False
    19     False
    20     False
    21     False
    22     False
    23     False
    24      True
    25      True
    26     False
    27     False
    28      True
    29      True
           ...  
    770     True
    771    False
    772    False
    773    False
    774     True
    775     True
    776     True
    777    False
    778    False
    779    False
    780    False
    781    False
    782    False
    783    False
    784    False
    785    False
    786    False
    787    False
    788     True
    789     True
    790    False
    791    False
    792     True
    793    False
    794    False
    795    False
    796    False
    797    False
    798    False
    799    False
    Name: 类型2, Length: 800, dtype: bool




```python
#查看类型2这一列的非缺失值和缺失值的数量分布
df["类型2"].isnull().value_counts()
```




    False    414
    True     386
    Name: 类型2, dtype: int64



### 2.删除缺失值

+ 删除掉含有缺失值的所有行


```python
df.dropna()
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
      <th>36</th>
      <td>Nidoqueen</td>
      <td>Poison</td>
      <td>Ground</td>
      <td>505</td>
      <td>90</td>
      <td>92</td>
      <td>87</td>
      <td>76</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Nidoking</td>
      <td>Poison</td>
      <td>Ground</td>
      <td>505</td>
      <td>81</td>
      <td>102</td>
      <td>77</td>
      <td>85</td>
      <td>1</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Jigglypuff</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>270</td>
      <td>115</td>
      <td>45</td>
      <td>20</td>
      <td>20</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Wigglytuff</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>435</td>
      <td>140</td>
      <td>70</td>
      <td>45</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Zubat</td>
      <td>Poison</td>
      <td>Flying</td>
      <td>245</td>
      <td>40</td>
      <td>45</td>
      <td>35</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Golbat</td>
      <td>Poison</td>
      <td>Flying</td>
      <td>455</td>
      <td>75</td>
      <td>80</td>
      <td>70</td>
      <td>90</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Oddish</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>320</td>
      <td>45</td>
      <td>50</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Gloom</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>395</td>
      <td>60</td>
      <td>65</td>
      <td>70</td>
      <td>40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Vileplume</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>490</td>
      <td>75</td>
      <td>80</td>
      <td>85</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Paras</td>
      <td>Bug</td>
      <td>Grass</td>
      <td>285</td>
      <td>35</td>
      <td>70</td>
      <td>55</td>
      <td>25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Parasect</td>
      <td>Bug</td>
      <td>Grass</td>
      <td>405</td>
      <td>60</td>
      <td>95</td>
      <td>80</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Venonat</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>305</td>
      <td>60</td>
      <td>55</td>
      <td>50</td>
      <td>45</td>
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
      <th>761</th>
      <td>Dragalge</td>
      <td>Poison</td>
      <td>Dragon</td>
      <td>494</td>
      <td>65</td>
      <td>75</td>
      <td>90</td>
      <td>44</td>
      <td>6</td>
    </tr>
    <tr>
      <th>764</th>
      <td>Helioptile</td>
      <td>Electric</td>
      <td>Normal</td>
      <td>289</td>
      <td>44</td>
      <td>38</td>
      <td>33</td>
      <td>70</td>
      <td>6</td>
    </tr>
    <tr>
      <th>765</th>
      <td>Heliolisk</td>
      <td>Electric</td>
      <td>Normal</td>
      <td>481</td>
      <td>62</td>
      <td>55</td>
      <td>52</td>
      <td>109</td>
      <td>6</td>
    </tr>
    <tr>
      <th>766</th>
      <td>Tyrunt</td>
      <td>Rock</td>
      <td>Dragon</td>
      <td>362</td>
      <td>58</td>
      <td>89</td>
      <td>77</td>
      <td>48</td>
      <td>6</td>
    </tr>
    <tr>
      <th>767</th>
      <td>Tyrantrum</td>
      <td>Rock</td>
      <td>Dragon</td>
      <td>521</td>
      <td>82</td>
      <td>121</td>
      <td>119</td>
      <td>71</td>
      <td>6</td>
    </tr>
    <tr>
      <th>768</th>
      <td>Amaura</td>
      <td>Rock</td>
      <td>Ice</td>
      <td>362</td>
      <td>77</td>
      <td>59</td>
      <td>50</td>
      <td>46</td>
      <td>6</td>
    </tr>
    <tr>
      <th>769</th>
      <td>Aurorus</td>
      <td>Rock</td>
      <td>Ice</td>
      <td>521</td>
      <td>123</td>
      <td>77</td>
      <td>72</td>
      <td>58</td>
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
<p>414 rows × 9 columns</p>
</div>




```python
df.dropna(how="any")
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
      <th>36</th>
      <td>Nidoqueen</td>
      <td>Poison</td>
      <td>Ground</td>
      <td>505</td>
      <td>90</td>
      <td>92</td>
      <td>87</td>
      <td>76</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Nidoking</td>
      <td>Poison</td>
      <td>Ground</td>
      <td>505</td>
      <td>81</td>
      <td>102</td>
      <td>77</td>
      <td>85</td>
      <td>1</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Jigglypuff</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>270</td>
      <td>115</td>
      <td>45</td>
      <td>20</td>
      <td>20</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Wigglytuff</td>
      <td>Normal</td>
      <td>Fairy</td>
      <td>435</td>
      <td>140</td>
      <td>70</td>
      <td>45</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Zubat</td>
      <td>Poison</td>
      <td>Flying</td>
      <td>245</td>
      <td>40</td>
      <td>45</td>
      <td>35</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Golbat</td>
      <td>Poison</td>
      <td>Flying</td>
      <td>455</td>
      <td>75</td>
      <td>80</td>
      <td>70</td>
      <td>90</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Oddish</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>320</td>
      <td>45</td>
      <td>50</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Gloom</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>395</td>
      <td>60</td>
      <td>65</td>
      <td>70</td>
      <td>40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Vileplume</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>490</td>
      <td>75</td>
      <td>80</td>
      <td>85</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Paras</td>
      <td>Bug</td>
      <td>Grass</td>
      <td>285</td>
      <td>35</td>
      <td>70</td>
      <td>55</td>
      <td>25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Parasect</td>
      <td>Bug</td>
      <td>Grass</td>
      <td>405</td>
      <td>60</td>
      <td>95</td>
      <td>80</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Venonat</td>
      <td>Bug</td>
      <td>Poison</td>
      <td>305</td>
      <td>60</td>
      <td>55</td>
      <td>50</td>
      <td>45</td>
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
      <th>761</th>
      <td>Dragalge</td>
      <td>Poison</td>
      <td>Dragon</td>
      <td>494</td>
      <td>65</td>
      <td>75</td>
      <td>90</td>
      <td>44</td>
      <td>6</td>
    </tr>
    <tr>
      <th>764</th>
      <td>Helioptile</td>
      <td>Electric</td>
      <td>Normal</td>
      <td>289</td>
      <td>44</td>
      <td>38</td>
      <td>33</td>
      <td>70</td>
      <td>6</td>
    </tr>
    <tr>
      <th>765</th>
      <td>Heliolisk</td>
      <td>Electric</td>
      <td>Normal</td>
      <td>481</td>
      <td>62</td>
      <td>55</td>
      <td>52</td>
      <td>109</td>
      <td>6</td>
    </tr>
    <tr>
      <th>766</th>
      <td>Tyrunt</td>
      <td>Rock</td>
      <td>Dragon</td>
      <td>362</td>
      <td>58</td>
      <td>89</td>
      <td>77</td>
      <td>48</td>
      <td>6</td>
    </tr>
    <tr>
      <th>767</th>
      <td>Tyrantrum</td>
      <td>Rock</td>
      <td>Dragon</td>
      <td>521</td>
      <td>82</td>
      <td>121</td>
      <td>119</td>
      <td>71</td>
      <td>6</td>
    </tr>
    <tr>
      <th>768</th>
      <td>Amaura</td>
      <td>Rock</td>
      <td>Ice</td>
      <td>362</td>
      <td>77</td>
      <td>59</td>
      <td>50</td>
      <td>46</td>
      <td>6</td>
    </tr>
    <tr>
      <th>769</th>
      <td>Aurorus</td>
      <td>Rock</td>
      <td>Ice</td>
      <td>521</td>
      <td>123</td>
      <td>77</td>
      <td>72</td>
      <td>58</td>
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
<p>414 rows × 9 columns</p>
</div>



+ 删除满足行内数据均为NaN这个条件的行


```python
df.dropna(how="all")
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



创建一个4行3列的含有NaN的数据作为演示


```python
df1 = pd.DataFrame([[1,5,np.nan],[2,np.nan,np.nan],[2,3,np.nan],[np.nan,np.nan,np.nan]])
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



how="all"能删除掉均为NaN的行


```python
df1.dropna(how="all")
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



+ 删除满足列内数据均为NaN这个条件的列，按列删除


```python
df1.dropna(how="all",axis=1,inplace=True)
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### 3.填充缺失值

+ 填充指定值


```python
#示例数据
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.fillna(value=0)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



+ 填充函数


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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1[1].fillna(df1[1].mean())
```




    0    5.0
    1    4.0
    2    3.0
    3    4.0
    Name: 1, dtype: float64



+ 向前填充


```python
df1[1]
```




    0    5.0
    1    NaN
    2    3.0
    3    NaN
    Name: 1, dtype: float64




```python
df1[1].fillna(method="ffill")
```




    0    5.0
    1    5.0
    2    3.0
    3    3.0
    Name: 1, dtype: float64



+ 向后填充


```python
df1[1].fillna(method="bfill")
```




    0    5.0
    1    3.0
    2    3.0
    3    NaN
    Name: 1, dtype: float64



## 二.清除空格

创建含有空格的示例数据


```python
dict1 = {"name":["小红","小明","小张"],"age":[16,17,18],"city":["北京  ","杭州","  上海  "]}
```


```python
df2 = pd.DataFrame(dict1,columns=["name","age","city"])
```

查看含有空格的数据


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
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>小红</td>
      <td>16</td>
      <td>北京</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17</td>
      <td>杭州</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18</td>
      <td>上海</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.to_csv("2.csv")
```

清除空格


```python
df2["city"] = df2["city"].map(str.strip)
```

查看清除后的数据表


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
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>小红</td>
      <td>16</td>
      <td>北京</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17</td>
      <td>杭州</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18</td>
      <td>上海</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.to_csv("df2.csv")
```

## 3.转换数据格式


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
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>小红</td>
      <td>16</td>
      <td>北京</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17</td>
      <td>杭州</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18</td>
      <td>上海</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.dtypes
```




    name    object
    age      int64
    city    object
    dtype: object



将年龄列数据转换成字符串格式


```python
df2["age"] = df2["age"].astype("str")
```


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
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>小红</td>
      <td>16</td>
      <td>北京</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17</td>
      <td>杭州</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18</td>
      <td>上海</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.dtypes
```




    name    object
    age     object
    city    object
    dtype: object



将年龄列数据转换成浮点数格式


```python
df2["age"] = df2["age"].astype("float")
```


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
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>小红</td>
      <td>16.0</td>
      <td>北京</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17.0</td>
      <td>杭州</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18.0</td>
      <td>上海</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.dtypes
```




    name     object
    age     float64
    city     object
    dtype: object



将年龄列数据转换成整数格式


```python
df2["age"] = df2["age"].astype("int")
```


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
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>小红</td>
      <td>16</td>
      <td>北京</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17</td>
      <td>杭州</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18</td>
      <td>上海</td>
    </tr>
  </tbody>
</table>
</div>



## 4.大小写转换


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
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>小红</td>
      <td>16</td>
      <td>北京</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17</td>
      <td>杭州</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18</td>
      <td>上海</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2["city"] = ["beijing","hangzhou","shanghai"]
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
      <td>小红</td>
      <td>16</td>
      <td>beijing</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17</td>
      <td>hangzhou</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18</td>
      <td>shanghai</td>
    </tr>
  </tbody>
</table>
</div>



转换成大写


```python
df2["city"] = df2["city"].str.upper()
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
      <td>小红</td>
      <td>16</td>
      <td>BEIJING</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17</td>
      <td>HANGZHOU</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18</td>
      <td>SHANGHAI</td>
    </tr>
  </tbody>
</table>
</div>



转换成小写


```python
df2["city"] = df2["city"].str.lower()
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
      <td>小红</td>
      <td>16</td>
      <td>beijing</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17</td>
      <td>hangzhou</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18</td>
      <td>shanghai</td>
    </tr>
  </tbody>
</table>
</div>



转换成首字母大写


```python
df2["city"] = df2["city"].str.title()
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
      <td>小红</td>
      <td>16</td>
      <td>Beijing</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17</td>
      <td>Hangzhou</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18</td>
      <td>Shanghai</td>
    </tr>
  </tbody>
</table>
</div>



## 5.更改列名


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
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>小红</td>
      <td>16</td>
      <td>Beijing</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17</td>
      <td>Hangzhou</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18</td>
      <td>Shanghai</td>
    </tr>
  </tbody>
</table>
</div>



通过rename函数修改部分列名或者所有列名，并默认返回一个新的数据框，若需要在原基础上修改，添加参数inplace=True即可


```python
df2.rename(columns={"name":"name2","age":"age2"})
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
      <th>age2</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>小红</td>
      <td>16</td>
      <td>beijing</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17</td>
      <td>hangzhou</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18</td>
      <td>shanghai</td>
    </tr>
  </tbody>
</table>
</div>



通过columns属性修改列名，这种方式就需要输入所有的列名了，并直接在原基础上修改


```python
df2.columns = ["n","a","c"]
```


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
      <th>n</th>
      <th>a</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>小红</td>
      <td>16</td>
      <td>beijing</td>
    </tr>
    <tr>
      <th>1</th>
      <td>小明</td>
      <td>17</td>
      <td>hangzhou</td>
    </tr>
    <tr>
      <th>2</th>
      <td>小张</td>
      <td>18</td>
      <td>shanghai</td>
    </tr>
  </tbody>
</table>
</div>



## 6.更改索引与重置索引


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



+ 更改索引


```python
#将类型1这列作为索引
df3 = df.set_index("类型1")
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
      <th>姓名</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
    <tr>
      <th>类型1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Grass</th>
      <td>Bulbasaur</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Ivysaur</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Venusaur</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>VenusaurMega Venusaur</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Charmander</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Charmeleon</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Charizard</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>CharizardMega Charizard X</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>CharizardMega Charizard Y</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>Squirtle</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>Wartortle</td>
      <td>NaN</td>
      <td>405</td>
      <td>59</td>
      <td>63</td>
      <td>80</td>
      <td>58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>Blastoise</td>
      <td>NaN</td>
      <td>530</td>
      <td>79</td>
      <td>83</td>
      <td>100</td>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>BlastoiseMega Blastoise</td>
      <td>NaN</td>
      <td>630</td>
      <td>79</td>
      <td>103</td>
      <td>120</td>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>Caterpie</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>Metapod</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>Butterfree</td>
      <td>Flying</td>
      <td>395</td>
      <td>60</td>
      <td>45</td>
      <td>50</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>Weedle</td>
      <td>Poison</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>Kakuna</td>
      <td>Poison</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>Beedrill</td>
      <td>Poison</td>
      <td>395</td>
      <td>65</td>
      <td>90</td>
      <td>40</td>
      <td>75</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>BeedrillMega Beedrill</td>
      <td>Poison</td>
      <td>495</td>
      <td>65</td>
      <td>150</td>
      <td>40</td>
      <td>145</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>Pidgey</td>
      <td>Flying</td>
      <td>251</td>
      <td>40</td>
      <td>45</td>
      <td>40</td>
      <td>56</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>Pidgeotto</td>
      <td>Flying</td>
      <td>349</td>
      <td>63</td>
      <td>60</td>
      <td>55</td>
      <td>71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>Pidgeot</td>
      <td>Flying</td>
      <td>479</td>
      <td>83</td>
      <td>80</td>
      <td>75</td>
      <td>101</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>PidgeotMega Pidgeot</td>
      <td>Flying</td>
      <td>579</td>
      <td>83</td>
      <td>80</td>
      <td>80</td>
      <td>121</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>Rattata</td>
      <td>NaN</td>
      <td>253</td>
      <td>30</td>
      <td>56</td>
      <td>35</td>
      <td>72</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>Raticate</td>
      <td>NaN</td>
      <td>413</td>
      <td>55</td>
      <td>81</td>
      <td>60</td>
      <td>97</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>Spearow</td>
      <td>Flying</td>
      <td>262</td>
      <td>40</td>
      <td>60</td>
      <td>30</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>Fearow</td>
      <td>Flying</td>
      <td>442</td>
      <td>65</td>
      <td>90</td>
      <td>65</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Ekans</td>
      <td>NaN</td>
      <td>288</td>
      <td>35</td>
      <td>60</td>
      <td>44</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Arbok</td>
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
    </tr>
    <tr>
      <th>Fairy</th>
      <td>Sylveon</td>
      <td>NaN</td>
      <td>525</td>
      <td>95</td>
      <td>65</td>
      <td>65</td>
      <td>60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Hawlucha</td>
      <td>Flying</td>
      <td>500</td>
      <td>78</td>
      <td>92</td>
      <td>75</td>
      <td>118</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Dedenne</td>
      <td>Fairy</td>
      <td>431</td>
      <td>67</td>
      <td>58</td>
      <td>57</td>
      <td>101</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>Carbink</td>
      <td>Fairy</td>
      <td>500</td>
      <td>50</td>
      <td>50</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>Goomy</td>
      <td>NaN</td>
      <td>300</td>
      <td>45</td>
      <td>50</td>
      <td>35</td>
      <td>40</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>Sliggoo</td>
      <td>NaN</td>
      <td>452</td>
      <td>68</td>
      <td>75</td>
      <td>53</td>
      <td>60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>Goodra</td>
      <td>NaN</td>
      <td>600</td>
      <td>90</td>
      <td>100</td>
      <td>70</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>Klefki</td>
      <td>Fairy</td>
      <td>470</td>
      <td>57</td>
      <td>80</td>
      <td>91</td>
      <td>75</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>Phantump</td>
      <td>Grass</td>
      <td>309</td>
      <td>43</td>
      <td>70</td>
      <td>48</td>
      <td>38</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>Trevenant</td>
      <td>Grass</td>
      <td>474</td>
      <td>85</td>
      <td>110</td>
      <td>76</td>
      <td>56</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>PumpkabooAverage Size</td>
      <td>Grass</td>
      <td>335</td>
      <td>49</td>
      <td>66</td>
      <td>70</td>
      <td>51</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>PumpkabooSmall Size</td>
      <td>Grass</td>
      <td>335</td>
      <td>44</td>
      <td>66</td>
      <td>70</td>
      <td>56</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>PumpkabooLarge Size</td>
      <td>Grass</td>
      <td>335</td>
      <td>54</td>
      <td>66</td>
      <td>70</td>
      <td>46</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>PumpkabooSuper Size</td>
      <td>Grass</td>
      <td>335</td>
      <td>59</td>
      <td>66</td>
      <td>70</td>
      <td>41</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>GourgeistAverage Size</td>
      <td>Grass</td>
      <td>494</td>
      <td>65</td>
      <td>90</td>
      <td>122</td>
      <td>84</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>GourgeistSmall Size</td>
      <td>Grass</td>
      <td>494</td>
      <td>55</td>
      <td>85</td>
      <td>122</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>GourgeistLarge Size</td>
      <td>Grass</td>
      <td>494</td>
      <td>75</td>
      <td>95</td>
      <td>122</td>
      <td>69</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>GourgeistSuper Size</td>
      <td>Grass</td>
      <td>494</td>
      <td>85</td>
      <td>100</td>
      <td>122</td>
      <td>54</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>Bergmite</td>
      <td>NaN</td>
      <td>304</td>
      <td>55</td>
      <td>69</td>
      <td>85</td>
      <td>28</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>Avalugg</td>
      <td>NaN</td>
      <td>514</td>
      <td>95</td>
      <td>117</td>
      <td>184</td>
      <td>28</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Noibat</td>
      <td>Dragon</td>
      <td>245</td>
      <td>40</td>
      <td>30</td>
      <td>35</td>
      <td>55</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Noivern</td>
      <td>Dragon</td>
      <td>535</td>
      <td>85</td>
      <td>70</td>
      <td>80</td>
      <td>123</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>Xerneas</td>
      <td>NaN</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>Yveltal</td>
      <td>Flying</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>Zygarde50% Forme</td>
      <td>Ground</td>
      <td>600</td>
      <td>108</td>
      <td>100</td>
      <td>121</td>
      <td>95</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>Diancie</td>
      <td>Fairy</td>
      <td>600</td>
      <td>50</td>
      <td>100</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>DiancieMega Diancie</td>
      <td>Fairy</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>HoopaHoopa Confined</td>
      <td>Ghost</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
      <td>60</td>
      <td>70</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>HoopaHoopa Unbound</td>
      <td>Dark</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
      <td>60</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Volcanion</td>
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
<p>800 rows × 8 columns</p>
</div>



+ 重置索引


```python
df4 = df3.reset_index()
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
      <th>类型1</th>
      <th>姓名</th>
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
      <td>Grass</td>
      <td>Bulbasaur</td>
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
      <td>Grass</td>
      <td>Ivysaur</td>
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
      <td>Grass</td>
      <td>Venusaur</td>
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
      <td>Grass</td>
      <td>VenusaurMega Venusaur</td>
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
      <td>Fire</td>
      <td>Charmander</td>
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
      <td>Fire</td>
      <td>Charmeleon</td>
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
      <td>Fire</td>
      <td>Charizard</td>
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
      <td>Fire</td>
      <td>CharizardMega Charizard X</td>
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
      <td>Fire</td>
      <td>CharizardMega Charizard Y</td>
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
      <td>Water</td>
      <td>Squirtle</td>
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
      <td>Water</td>
      <td>Wartortle</td>
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
      <td>Water</td>
      <td>Blastoise</td>
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
      <td>Water</td>
      <td>BlastoiseMega Blastoise</td>
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
      <td>Bug</td>
      <td>Caterpie</td>
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
      <td>Bug</td>
      <td>Metapod</td>
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
      <td>Bug</td>
      <td>Butterfree</td>
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
      <td>Bug</td>
      <td>Weedle</td>
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
      <td>Bug</td>
      <td>Kakuna</td>
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
      <td>Bug</td>
      <td>Beedrill</td>
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
      <td>Bug</td>
      <td>BeedrillMega Beedrill</td>
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
      <td>Normal</td>
      <td>Pidgey</td>
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
      <td>Normal</td>
      <td>Pidgeotto</td>
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
      <td>Normal</td>
      <td>Pidgeot</td>
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
      <td>Normal</td>
      <td>PidgeotMega Pidgeot</td>
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
      <td>Normal</td>
      <td>Rattata</td>
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
      <td>Normal</td>
      <td>Raticate</td>
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
      <td>Normal</td>
      <td>Spearow</td>
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
      <td>Normal</td>
      <td>Fearow</td>
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
      <td>Poison</td>
      <td>Ekans</td>
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
      <td>Poison</td>
      <td>Arbok</td>
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
      <td>Fairy</td>
      <td>Sylveon</td>
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
      <td>Fighting</td>
      <td>Hawlucha</td>
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
      <td>Electric</td>
      <td>Dedenne</td>
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
      <td>Rock</td>
      <td>Carbink</td>
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
      <td>Dragon</td>
      <td>Goomy</td>
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
      <td>Dragon</td>
      <td>Sliggoo</td>
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
      <td>Dragon</td>
      <td>Goodra</td>
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
      <td>Steel</td>
      <td>Klefki</td>
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
      <td>Ghost</td>
      <td>Phantump</td>
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
      <td>Ghost</td>
      <td>Trevenant</td>
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
      <td>Ghost</td>
      <td>PumpkabooAverage Size</td>
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
      <td>Ghost</td>
      <td>PumpkabooSmall Size</td>
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
      <td>Ghost</td>
      <td>PumpkabooLarge Size</td>
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
      <td>Ghost</td>
      <td>PumpkabooSuper Size</td>
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
      <td>Ghost</td>
      <td>GourgeistAverage Size</td>
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
      <td>Ghost</td>
      <td>GourgeistSmall Size</td>
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
      <td>Ghost</td>
      <td>GourgeistLarge Size</td>
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
      <td>Ghost</td>
      <td>GourgeistSuper Size</td>
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
      <td>Ice</td>
      <td>Bergmite</td>
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
      <td>Ice</td>
      <td>Avalugg</td>
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
      <td>Flying</td>
      <td>Noibat</td>
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
      <td>Flying</td>
      <td>Noivern</td>
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
      <td>Fairy</td>
      <td>Xerneas</td>
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
      <td>Dark</td>
      <td>Yveltal</td>
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
      <td>Dragon</td>
      <td>Zygarde50% Forme</td>
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
      <td>Rock</td>
      <td>Diancie</td>
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
      <td>Rock</td>
      <td>DiancieMega Diancie</td>
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
      <td>Psychic</td>
      <td>HoopaHoopa Confined</td>
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
      <td>Psychic</td>
      <td>HoopaHoopa Unbound</td>
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
      <td>Fire</td>
      <td>Volcanion</td>
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



## 7.重复值处理


```python
df5 = pd.DataFrame({"c1":["apple"]*3 + ["banana"]*3,"c2":[1,1,2,3,3,2]})
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
      <th>c1</th>
      <th>c2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>apple</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>apple</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>banana</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>banana</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>banana</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



+ 查看是否有重复值


```python
#适合小数据目测
df5.duplicated(subset=["c1","c2"],keep="first")
```




    0    False
    1     True
    2    False
    3    False
    4     True
    5    False
    dtype: bool




```python
#当数据量比较大的时候，可以看看重复数据和非重复数据的计数分布
df5_duplicated = df5.duplicated(subset=["c1","c2"],keep="first")
df5_duplicated.value_counts()
```




    False    4
    True     2
    dtype: int64



+ 保留重复值


```python
df5[df5.duplicated(subset=["c1","c2"],keep="first")]
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
      <th>c1</th>
      <th>c2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>apple</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>banana</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



+ 删除重复值

（1）默认保留第一个出现的重复值，删除掉后面的重复值


```python
df5.drop_duplicates(subset=["c1","c2"],keep="first")
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
      <th>c1</th>
      <th>c2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>apple</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>banana</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>banana</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



（2）保留最后一个重复值，删除掉前面的重复值


```python
df5.drop_duplicates(subset=["c1","c2"],keep="last")
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
      <th>c1</th>
      <th>c2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>apple</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>apple</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>banana</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>banana</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



（3）如果希望直接在原基础上修改，添加参数inplace=True


```python
df5.drop_duplicates(subset=["c1","c2"],keep="last",inplace=True)
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
      <th>c1</th>
      <th>c2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>apple</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>apple</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>banana</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>banana</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## 8.替换值

忽略警告的做法


```python
import warnings
warnings.filterwarnings("ignore")
```

示例数据


```python
df6 = df.head(10)
df6
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



+ 单个对象替换单个值


```python
df6["类型1"] = df6["类型1"].replace("Grass","G")
df6
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
      <td>G</td>
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
      <td>G</td>
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
      <td>G</td>
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
      <td>G</td>
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



+ 多个对象替换单个值


```python
df6["类型1"] = df6["类型1"].replace(["G","Fire"],"gf")
df6
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
      <td>gf</td>
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
      <td>gf</td>
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
      <td>gf</td>
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
      <td>gf</td>
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
      <td>gf</td>
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
      <td>gf</td>
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
      <td>gf</td>
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
      <td>gf</td>
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
      <td>gf</td>
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



+ 用不同的值替换不同的对象


```python
df6["类型1"] = df6["类型1"].replace(["gf","Water"],["good","W"])
df6
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
      <td>good</td>
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
      <td>good</td>
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
      <td>good</td>
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
      <td>good</td>
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
      <td>good</td>
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
      <td>good</td>
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
      <td>good</td>
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
      <td>good</td>
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
      <td>good</td>
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
      <td>W</td>
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



参数也可以是字典


```python
df6["类型1"] = df6["类型1"].replace({"good":"gg","W":"ww"})
df6
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
      <td>gg</td>
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
      <td>gg</td>
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
      <td>gg</td>
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
      <td>gg</td>
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
      <td>gg</td>
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
      <td>gg</td>
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
      <td>gg</td>
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
      <td>gg</td>
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
      <td>gg</td>
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
      <td>ww</td>
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


