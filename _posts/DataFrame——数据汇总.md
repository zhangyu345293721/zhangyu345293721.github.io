---
title: 'DataFrame-数据汇总'
date: 2022-04-14
permalink: /posts/2012/08/blog-post-1/
tags:
  - cool posts
  - category1
  - category2
---

```python
import pandas as pd
import numpy as np
```

## 一.分组计算


```python
#示例数据
df = pd.read_csv("pokemon_data.csv",encoding="gbk")
df.head(10)
```




<div>

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



### 1.如何分组计算

假设现在要根据“类型1”这列来做分组计算


```python
#查看下类型1中的类别数量分布
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
    Dragon       32
    Ghost        32
    Dark         31
    Poison       28
    Steel        27
    Fighting     27
    Ice          24
    Fairy        17
    Flying        4
    Name: 类型1, dtype: int64




```python
#查看类型1中有多少个类型
len(df["类型1"].value_counts())
```




    18



实例演练

**Q1：想知道类型1的这18个种类各自的平均攻击力是多少（单列分组计算）**


```python
#根据类型1这列来分组，并将结果存储在grouped1中
grouped1 = df.groupby("类型1")
```


```python
#输出grouped1，这里就是显示它是一个分组对象，并且存储的内存地址是0x0000000008EE9E80，没什么卵用
grouped1
```




    <pandas.core.groupby.DataFrameGroupBy object at 0x0000000008FAD8D0>




```python
#求类型1的18个种类各自的平均攻击力
grouped1[["攻击力"]].mean()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>攻击力</th>
    </tr>
    <tr>
      <th>类型1</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bug</th>
      <td>70.971014</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>88.387097</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>112.125000</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>69.090909</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>61.529412</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>96.777778</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>84.769231</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>78.750000</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>73.781250</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>73.214286</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>95.750000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>72.750000</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>73.469388</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>74.678571</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>71.456140</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>92.863636</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>92.703704</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>74.151786</td>
    </tr>
  </tbody>
</table>
</div>



小结一下：

grouped1 = df.groupby("类型1")这一步就是分组计算流程里的第一步：split

grouped1[["攻击力"]].mean() 这一步就是分组计算流程的第二和第三步：apply—combine

**Q2：想知道类型1和类型2的组合类型里，每个组合各自的攻击力均值（多列分组计算）**


```python
grouped2 = df.groupby(["类型1","类型2"])
```


```python
grouped2[["攻击力"]].mean()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>攻击力</th>
    </tr>
    <tr>
      <th>类型1</th>
      <th>类型2</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="11" valign="top">Bug</th>
      <th>Electric</th>
      <td>62.000000</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>155.000000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>72.500000</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>70.142857</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>73.833333</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>62.000000</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>68.333333</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>56.666667</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>114.714286</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>30.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Dark</th>
      <th>Dragon</th>
      <td>85.000000</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>82.500000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>92.200000</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>107.500000</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>73.000000</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>105.000000</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Dragon</th>
      <th>Electric</th>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>110.000000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>120.000000</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>135.666667</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>112.000000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>140.000000</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>100.000000</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Electric</th>
      <th>Dragon</th>
      <td>95.000000</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>58.000000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>65.000000</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Rock</th>
      <th>Fighting</th>
      <td>129.000000</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>123.000000</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>61.000000</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>81.333333</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>68.000000</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>75.000000</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>49.666667</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>75.333333</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Steel</th>
      <th>Dragon</th>
      <td>120.000000</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>97.500000</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>105.000000</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>89.000000</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>90.000000</td>
    </tr>
    <tr>
      <th rowspan="14" valign="top">Water</th>
      <th>Dark</th>
      <td>120.000000</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>107.500000</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>79.666667</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>56.571429</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>84.400000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>83.333333</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>68.333333</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>73.000000</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>82.750000</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>86.000000</td>
    </tr>
  </tbody>
</table>
<p>136 rows × 1 columns</p>
</div>



**Q3:想知道类型1和类型2的组合类型里，每个组合各自的攻击力均值、中位数、总和（对组应用多个函数）**


```python
grouped2[["攻击力"]].agg([np.mean,np.median,np.sum])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">攻击力</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>sum</th>
    </tr>
    <tr>
      <th>类型1</th>
      <th>类型2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="11" valign="top">Bug</th>
      <th>Electric</th>
      <td>62.000000</td>
      <td>62.0</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>155.000000</td>
      <td>155.0</td>
      <td>310</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>72.500000</td>
      <td>72.5</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>70.142857</td>
      <td>67.5</td>
      <td>982</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>90.000000</td>
      <td>90.0</td>
      <td>90</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>73.833333</td>
      <td>66.5</td>
      <td>443</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>62.000000</td>
      <td>62.0</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>68.333333</td>
      <td>57.5</td>
      <td>820</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>56.666667</td>
      <td>65.0</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>114.714286</td>
      <td>120.0</td>
      <td>803</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>30.000000</td>
      <td>30.0</td>
      <td>30</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Dark</th>
      <th>Dragon</th>
      <td>85.000000</td>
      <td>85.0</td>
      <td>255</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>82.500000</td>
      <td>82.5</td>
      <td>165</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>80.000000</td>
      <td>90.0</td>
      <td>240</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>92.200000</td>
      <td>85.0</td>
      <td>461</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>80.000000</td>
      <td>80.0</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>107.500000</td>
      <td>107.5</td>
      <td>215</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>73.000000</td>
      <td>73.0</td>
      <td>146</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>105.000000</td>
      <td>105.0</td>
      <td>210</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Dragon</th>
      <th>Electric</th>
      <td>150.000000</td>
      <td>150.0</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>110.000000</td>
      <td>110.0</td>
      <td>110</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>120.000000</td>
      <td>120.0</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>135.666667</td>
      <td>140.0</td>
      <td>814</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>112.000000</td>
      <td>100.0</td>
      <td>560</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>140.000000</td>
      <td>130.0</td>
      <td>420</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>100.000000</td>
      <td>95.0</td>
      <td>400</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Electric</th>
      <th>Dragon</th>
      <td>95.000000</td>
      <td>95.0</td>
      <td>95</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>58.000000</td>
      <td>58.0</td>
      <td>58</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>65.000000</td>
      <td>65.0</td>
      <td>65</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>90.000000</td>
      <td>90.0</td>
      <td>450</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Rock</th>
      <th>Fighting</th>
      <td>129.000000</td>
      <td>129.0</td>
      <td>129</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>123.000000</td>
      <td>123.5</td>
      <td>492</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>61.000000</td>
      <td>61.0</td>
      <td>122</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>81.333333</td>
      <td>82.0</td>
      <td>488</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>68.000000</td>
      <td>68.0</td>
      <td>136</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>75.000000</td>
      <td>75.0</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>49.666667</td>
      <td>52.0</td>
      <td>149</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>75.333333</td>
      <td>70.0</td>
      <td>452</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Steel</th>
      <th>Dragon</th>
      <td>120.000000</td>
      <td>120.0</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>90.000000</td>
      <td>85.0</td>
      <td>270</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>90.000000</td>
      <td>90.0</td>
      <td>90</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>80.000000</td>
      <td>80.0</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>97.500000</td>
      <td>95.0</td>
      <td>390</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>105.000000</td>
      <td>105.0</td>
      <td>210</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>89.000000</td>
      <td>89.0</td>
      <td>623</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>90.000000</td>
      <td>90.0</td>
      <td>270</td>
    </tr>
    <tr>
      <th rowspan="14" valign="top">Water</th>
      <th>Dark</th>
      <td>120.000000</td>
      <td>120.0</td>
      <td>720</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>107.500000</td>
      <td>107.5</td>
      <td>215</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>48.000000</td>
      <td>48.0</td>
      <td>96</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>35.000000</td>
      <td>35.0</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>79.666667</td>
      <td>72.0</td>
      <td>239</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>56.571429</td>
      <td>44.0</td>
      <td>396</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>50.000000</td>
      <td>50.0</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>50.000000</td>
      <td>50.0</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>84.400000</td>
      <td>84.0</td>
      <td>844</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>83.333333</td>
      <td>85.0</td>
      <td>250</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>68.333333</td>
      <td>70.0</td>
      <td>205</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>73.000000</td>
      <td>75.0</td>
      <td>365</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>82.750000</td>
      <td>84.0</td>
      <td>331</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>86.000000</td>
      <td>86.0</td>
      <td>86</td>
    </tr>
  </tbody>
</table>
<p>136 rows × 3 columns</p>
</div>



**Q4：想知道类型1和类型2的组合类型里，每个组合各自的攻击力的均值和中位数，生命值的总和（对不同列应用不同的函数）**


```python
grouped2.agg({"攻击力":[np.mean,np.median],"生命值":np.sum})
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">攻击力</th>
      <th>生命值</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>sum</th>
    </tr>
    <tr>
      <th>类型1</th>
      <th>类型2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="11" valign="top">Bug</th>
      <th>Electric</th>
      <td>62.000000</td>
      <td>62.0</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>155.000000</td>
      <td>155.0</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>72.500000</td>
      <td>72.5</td>
      <td>140</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>70.142857</td>
      <td>67.5</td>
      <td>882</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>90.000000</td>
      <td>90.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>73.833333</td>
      <td>66.5</td>
      <td>330</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>62.000000</td>
      <td>62.0</td>
      <td>91</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>68.333333</td>
      <td>57.5</td>
      <td>645</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>56.666667</td>
      <td>65.0</td>
      <td>140</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>114.714286</td>
      <td>120.0</td>
      <td>474</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>30.000000</td>
      <td>30.0</td>
      <td>40</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Dark</th>
      <th>Dragon</th>
      <td>85.000000</td>
      <td>85.0</td>
      <td>216</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>82.500000</td>
      <td>82.5</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>80.000000</td>
      <td>90.0</td>
      <td>195</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>92.200000</td>
      <td>85.0</td>
      <td>466</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>80.000000</td>
      <td>80.0</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>107.500000</td>
      <td>107.5</td>
      <td>125</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>73.000000</td>
      <td>73.0</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>105.000000</td>
      <td>105.0</td>
      <td>110</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Dragon</th>
      <th>Electric</th>
      <td>150.000000</td>
      <td>150.0</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>110.000000</td>
      <td>110.0</td>
      <td>75</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>120.000000</td>
      <td>120.0</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>135.666667</td>
      <td>140.0</td>
      <td>566</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>112.000000</td>
      <td>100.0</td>
      <td>450</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>140.000000</td>
      <td>130.0</td>
      <td>375</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>100.000000</td>
      <td>95.0</td>
      <td>320</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Electric</th>
      <th>Dragon</th>
      <td>95.000000</td>
      <td>95.0</td>
      <td>90</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>58.000000</td>
      <td>58.0</td>
      <td>67</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>65.000000</td>
      <td>65.0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>90.000000</td>
      <td>90.0</td>
      <td>353</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Rock</th>
      <th>Fighting</th>
      <td>129.000000</td>
      <td>129.0</td>
      <td>91</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>123.000000</td>
      <td>123.5</td>
      <td>290</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>61.000000</td>
      <td>61.0</td>
      <td>152</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>81.333333</td>
      <td>82.0</td>
      <td>330</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>68.000000</td>
      <td>68.0</td>
      <td>200</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>75.000000</td>
      <td>75.0</td>
      <td>140</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>49.666667</td>
      <td>52.0</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>75.333333</td>
      <td>70.0</td>
      <td>309</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Steel</th>
      <th>Dragon</th>
      <td>120.000000</td>
      <td>120.0</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>90.000000</td>
      <td>85.0</td>
      <td>157</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>90.000000</td>
      <td>90.0</td>
      <td>91</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>80.000000</td>
      <td>80.0</td>
      <td>65</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>97.500000</td>
      <td>95.0</td>
      <td>224</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>105.000000</td>
      <td>105.0</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>89.000000</td>
      <td>89.0</td>
      <td>484</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>90.000000</td>
      <td>90.0</td>
      <td>180</td>
    </tr>
    <tr>
      <th rowspan="14" valign="top">Water</th>
      <th>Dark</th>
      <td>120.000000</td>
      <td>120.0</td>
      <td>415</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>107.500000</td>
      <td>107.5</td>
      <td>165</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>48.000000</td>
      <td>48.0</td>
      <td>200</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>35.000000</td>
      <td>35.0</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>79.666667</td>
      <td>72.0</td>
      <td>272</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>56.571429</td>
      <td>44.0</td>
      <td>442</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>50.000000</td>
      <td>50.0</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>50.000000</td>
      <td>50.0</td>
      <td>180</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>84.400000</td>
      <td>84.0</td>
      <td>871</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>83.333333</td>
      <td>85.0</td>
      <td>270</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>68.333333</td>
      <td>70.0</td>
      <td>185</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>73.000000</td>
      <td>75.0</td>
      <td>435</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>82.750000</td>
      <td>84.0</td>
      <td>283</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>86.000000</td>
      <td>86.0</td>
      <td>84</td>
    </tr>
  </tbody>
</table>
<p>136 rows × 3 columns</p>
</div>



**Q5：对组内数据进行标准化处理（转换）**


```python
zscore = lambda x : (x-x.mean())/x.std()
```


```python
grouped1.transform(zscore)
```




<div>
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
      <th>0</th>
      <td>-0.967110</td>
      <td>-1.141155</td>
      <td>-0.954050</td>
      <td>-0.890334</td>
      <td>-0.593850</td>
      <td>-1.492643</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.151362</td>
      <td>-0.372577</td>
      <td>-0.441846</td>
      <td>-0.318560</td>
      <td>-0.067654</td>
      <td>-1.492643</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.973807</td>
      <td>0.652193</td>
      <td>0.346160</td>
      <td>0.498260</td>
      <td>0.633942</td>
      <td>-1.492643</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.911448</td>
      <td>0.652193</td>
      <td>1.055365</td>
      <td>2.131901</td>
      <td>0.633942</td>
      <td>-1.492643</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.358202</td>
      <td>-1.592643</td>
      <td>-1.139036</td>
      <td>-1.046962</td>
      <td>-0.374015</td>
      <td>-1.194996</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.483570</td>
      <td>-0.613470</td>
      <td>-0.721924</td>
      <td>-0.412932</td>
      <td>0.220143</td>
      <td>-1.194996</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.691716</td>
      <td>0.417239</td>
      <td>-0.026738</td>
      <td>0.432441</td>
      <td>1.012355</td>
      <td>-1.194996</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.602790</td>
      <td>0.417239</td>
      <td>1.572190</td>
      <td>1.827306</td>
      <td>1.012355</td>
      <td>-1.194996</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.602790</td>
      <td>0.417239</td>
      <td>0.668448</td>
      <td>0.432441</td>
      <td>1.012355</td>
      <td>-1.194996</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-1.028864</td>
      <td>-1.020936</td>
      <td>-0.921578</td>
      <td>-0.286112</td>
      <td>-0.997608</td>
      <td>-1.191392</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.224894</td>
      <td>-0.475224</td>
      <td>-0.392984</td>
      <td>0.253965</td>
      <td>-0.345982</td>
      <td>-1.191392</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.879461</td>
      <td>0.252392</td>
      <td>0.311807</td>
      <td>0.974068</td>
      <td>0.522852</td>
      <td>-1.191392</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.762945</td>
      <td>0.252392</td>
      <td>1.016599</td>
      <td>1.694171</td>
      <td>0.522852</td>
      <td>-1.191392</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-1.560358</td>
      <td>-0.727792</td>
      <td>-1.106102</td>
      <td>-1.062676</td>
      <td>-0.502027</td>
      <td>-1.387228</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-1.475522</td>
      <td>-0.421587</td>
      <td>-1.376074</td>
      <td>-0.467750</td>
      <td>-0.953459</td>
      <td>-1.387228</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.136352</td>
      <td>0.190824</td>
      <td>-0.701144</td>
      <td>-0.616482</td>
      <td>0.250359</td>
      <td>-1.387228</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-1.560358</td>
      <td>-1.033998</td>
      <td>-0.971116</td>
      <td>-1.211408</td>
      <td>-0.351550</td>
      <td>-1.387228</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-1.475522</td>
      <td>-0.727792</td>
      <td>-1.241088</td>
      <td>-0.616482</td>
      <td>-0.802982</td>
      <td>-1.387228</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.136352</td>
      <td>0.497029</td>
      <td>0.513729</td>
      <td>-0.913945</td>
      <td>0.400837</td>
      <td>-1.387228</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.984706</td>
      <td>0.497029</td>
      <td>2.133560</td>
      <td>-0.913945</td>
      <td>2.507519</td>
      <td>-1.387228</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-1.301994</td>
      <td>-1.028760</td>
      <td>-0.939712</td>
      <td>-0.834893</td>
      <td>-0.547452</td>
      <td>-1.301899</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-0.455217</td>
      <td>-0.393987</td>
      <td>-0.444595</td>
      <td>-0.203894</td>
      <td>-0.019398</td>
      <td>-1.301899</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.668058</td>
      <td>0.157989</td>
      <td>0.215561</td>
      <td>0.637438</td>
      <td>1.036711</td>
      <td>-1.301899</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1.532115</td>
      <td>0.157989</td>
      <td>0.215561</td>
      <td>0.847771</td>
      <td>1.740784</td>
      <td>-1.301899</td>
    </tr>
    <tr>
      <th>24</th>
      <td>-1.284712</td>
      <td>-1.304748</td>
      <td>-0.576626</td>
      <td>-1.045226</td>
      <td>0.015806</td>
      <td>-1.301899</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.097780</td>
      <td>-0.614778</td>
      <td>0.248569</td>
      <td>0.006439</td>
      <td>0.895897</td>
      <td>-1.301899</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-1.206947</td>
      <td>-1.028760</td>
      <td>-0.444595</td>
      <td>-1.255559</td>
      <td>-0.054602</td>
      <td>-1.301899</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.348356</td>
      <td>-0.338790</td>
      <td>0.545639</td>
      <td>0.216772</td>
      <td>1.001508</td>
      <td>-1.301899</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-1.203383</td>
      <td>-1.639968</td>
      <td>-0.747762</td>
      <td>-1.178263</td>
      <td>-0.378741</td>
      <td>-0.876086</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.420720</td>
      <td>-0.368675</td>
      <td>0.525798</td>
      <td>0.008477</td>
      <td>0.725920</td>
      <td>-0.876086</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>770</th>
      <td>0.903393</td>
      <td>0.885322</td>
      <td>0.116653</td>
      <td>-0.037192</td>
      <td>0.489666</td>
      <td>0.876671</td>
    </tr>
    <tr>
      <th>771</th>
      <td>0.815460</td>
      <td>0.315176</td>
      <td>-0.168885</td>
      <td>0.488427</td>
      <td>1.992968</td>
      <td>1.460417</td>
    </tr>
    <tr>
      <th>772</th>
      <td>-0.117375</td>
      <td>0.416291</td>
      <td>-0.466707</td>
      <td>-0.375467</td>
      <td>0.618172</td>
      <td>1.699556</td>
    </tr>
    <tr>
      <th>773</th>
      <td>0.428006</td>
      <td>-0.737435</td>
      <td>-1.213392</td>
      <td>1.350022</td>
      <td>-0.197605</td>
      <td>1.377131</td>
    </tr>
    <tr>
      <th>774</th>
      <td>-1.712829</td>
      <td>-1.610079</td>
      <td>-1.841143</td>
      <td>-2.131483</td>
      <td>-1.851606</td>
      <td>1.484749</td>
    </tr>
    <tr>
      <th>775</th>
      <td>-0.673637</td>
      <td>-0.643506</td>
      <td>-1.100241</td>
      <td>-1.384686</td>
      <td>-0.991019</td>
      <td>1.484749</td>
    </tr>
    <tr>
      <th>776</th>
      <td>0.338207</td>
      <td>0.281042</td>
      <td>-0.359338</td>
      <td>-0.679378</td>
      <td>-0.130433</td>
      <td>1.484749</td>
    </tr>
    <tr>
      <th>777</th>
      <td>-0.153384</td>
      <td>-0.512479</td>
      <td>-0.418046</td>
      <td>-0.789402</td>
      <td>0.763766</td>
      <td>1.590845</td>
    </tr>
    <tr>
      <th>778</th>
      <td>-1.186148</td>
      <td>-0.676563</td>
      <td>-0.127617</td>
      <td>-1.019550</td>
      <td>-0.940167</td>
      <td>1.070457</td>
    </tr>
    <tr>
      <th>779</th>
      <td>0.312861</td>
      <td>0.648948</td>
      <td>1.222380</td>
      <td>-0.159365</td>
      <td>-0.297775</td>
      <td>1.070457</td>
    </tr>
    <tr>
      <th>780</th>
      <td>-0.949940</td>
      <td>-0.487204</td>
      <td>-0.262617</td>
      <td>-0.343690</td>
      <td>-0.476218</td>
      <td>1.070457</td>
    </tr>
    <tr>
      <th>781</th>
      <td>-0.949940</td>
      <td>-0.645003</td>
      <td>-0.262617</td>
      <td>-0.343690</td>
      <td>-0.297775</td>
      <td>1.070457</td>
    </tr>
    <tr>
      <th>782</th>
      <td>-0.949940</td>
      <td>-0.329405</td>
      <td>-0.262617</td>
      <td>-0.343690</td>
      <td>-0.654660</td>
      <td>1.070457</td>
    </tr>
    <tr>
      <th>783</th>
      <td>-0.949940</td>
      <td>-0.171606</td>
      <td>-0.262617</td>
      <td>-0.343690</td>
      <td>-0.833102</td>
      <td>1.070457</td>
    </tr>
    <tr>
      <th>784</th>
      <td>0.494560</td>
      <td>0.017752</td>
      <td>0.547382</td>
      <td>1.253796</td>
      <td>0.701501</td>
      <td>1.070457</td>
    </tr>
    <tr>
      <th>785</th>
      <td>0.494560</td>
      <td>-0.297845</td>
      <td>0.378632</td>
      <td>1.253796</td>
      <td>1.236827</td>
      <td>1.070457</td>
    </tr>
    <tr>
      <th>786</th>
      <td>0.494560</td>
      <td>0.333350</td>
      <td>0.716131</td>
      <td>1.253796</td>
      <td>0.166174</td>
      <td>1.070457</td>
    </tr>
    <tr>
      <th>787</th>
      <td>0.494560</td>
      <td>0.648948</td>
      <td>0.884881</td>
      <td>1.253796</td>
      <td>-0.369152</td>
      <td>1.070457</td>
    </tr>
    <tr>
      <th>788</th>
      <td>-1.195577</td>
      <td>-0.798615</td>
      <td>-0.137415</td>
      <td>0.395005</td>
      <td>-1.447373</td>
      <td>1.668018</td>
    </tr>
    <tr>
      <th>789</th>
      <td>0.743821</td>
      <td>1.080479</td>
      <td>1.621502</td>
      <td>3.273941</td>
      <td>-1.447373</td>
      <td>1.668018</td>
    </tr>
    <tr>
      <th>790</th>
      <td>-1.486988</td>
      <td>-1.485923</td>
      <td>-1.300000</td>
      <td>-1.463014</td>
      <td>-1.479806</td>
      <td>0.866025</td>
    </tr>
    <tr>
      <th>791</th>
      <td>0.309789</td>
      <td>0.688599</td>
      <td>-0.233333</td>
      <td>0.643726</td>
      <td>0.638653</td>
      <td>0.866025</td>
    </tr>
    <tr>
      <th>792</th>
      <td>2.155598</td>
      <td>2.199589</td>
      <td>2.335044</td>
      <td>1.543463</td>
      <td>2.163112</td>
      <td>0.876671</td>
    </tr>
    <tr>
      <th>793</th>
      <td>2.146671</td>
      <td>2.808270</td>
      <td>1.653313</td>
      <td>0.986156</td>
      <td>0.822477</td>
      <td>1.453700</td>
    </tr>
    <tr>
      <th>794</th>
      <td>0.338207</td>
      <td>1.037490</td>
      <td>-0.359338</td>
      <td>1.436547</td>
      <td>0.515007</td>
      <td>1.484749</td>
    </tr>
    <tr>
      <th>795</th>
      <td>1.353424</td>
      <td>-0.737435</td>
      <td>0.202018</td>
      <td>1.350022</td>
      <td>-0.197605</td>
      <td>1.377131</td>
    </tr>
    <tr>
      <th>796</th>
      <td>2.278843</td>
      <td>-0.737435</td>
      <td>1.900509</td>
      <td>0.252545</td>
      <td>1.808844</td>
      <td>1.377131</td>
    </tr>
    <tr>
      <th>797</th>
      <td>0.892294</td>
      <td>0.329626</td>
      <td>0.911003</td>
      <td>-0.270958</td>
      <td>-0.307784</td>
      <td>1.589229</td>
    </tr>
    <tr>
      <th>798</th>
      <td>1.467723</td>
      <td>0.329626</td>
      <td>2.092777</td>
      <td>-0.270958</td>
      <td>-0.039941</td>
      <td>1.589229</td>
    </tr>
    <tr>
      <th>799</th>
      <td>1.293025</td>
      <td>0.520310</td>
      <td>0.877004</td>
      <td>2.207724</td>
      <td>-0.175962</td>
      <td>1.506735</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 6 columns</p>
</div>



**Q6：对组进行条件过滤**

需求：针对grouped2的这个分组，希望得到平均攻击力为100以上的组，其余的组过滤掉


```python
attack_filter = lambda x : x["攻击力"].mean() > 100
```


```python
grouped2.filter(attack_filter)
```




<div>

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
      <th>119</th>
      <td>Rhyhorn</td>
      <td>Ground</td>
      <td>Rock</td>
      <td>345</td>
      <td>80</td>
      <td>85</td>
      <td>95</td>
      <td>25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>120</th>
      <td>Rhydon</td>
      <td>Ground</td>
      <td>Rock</td>
      <td>485</td>
      <td>105</td>
      <td>130</td>
      <td>120</td>
      <td>40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>141</th>
      <td>GyaradosMega Gyarados</td>
      <td>Water</td>
      <td>Dark</td>
      <td>640</td>
      <td>95</td>
      <td>155</td>
      <td>109</td>
      <td>81</td>
      <td>1</td>
    </tr>
    <tr>
      <th>153</th>
      <td>Aerodactyl</td>
      <td>Rock</td>
      <td>Flying</td>
      <td>515</td>
      <td>80</td>
      <td>105</td>
      <td>65</td>
      <td>130</td>
      <td>1</td>
    </tr>
    <tr>
      <th>154</th>
      <td>AerodactylMega Aerodactyl</td>
      <td>Rock</td>
      <td>Flying</td>
      <td>615</td>
      <td>80</td>
      <td>135</td>
      <td>85</td>
      <td>150</td>
      <td>1</td>
    </tr>
    <tr>
      <th>161</th>
      <td>Dragonite</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>600</td>
      <td>91</td>
      <td>134</td>
      <td>95</td>
      <td>80</td>
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
      <th>220</th>
      <td>Forretress</td>
      <td>Bug</td>
      <td>Steel</td>
      <td>465</td>
      <td>75</td>
      <td>90</td>
      <td>140</td>
      <td>40</td>
      <td>2</td>
    </tr>
    <tr>
      <th>222</th>
      <td>Gligar</td>
      <td>Ground</td>
      <td>Flying</td>
      <td>430</td>
      <td>65</td>
      <td>75</td>
      <td>105</td>
      <td>85</td>
      <td>2</td>
    </tr>
    <tr>
      <th>223</th>
      <td>Steelix</td>
      <td>Steel</td>
      <td>Ground</td>
      <td>510</td>
      <td>75</td>
      <td>85</td>
      <td>200</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>224</th>
      <td>SteelixMega Steelix</td>
      <td>Steel</td>
      <td>Ground</td>
      <td>610</td>
      <td>75</td>
      <td>125</td>
      <td>230</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>228</th>
      <td>Scizor</td>
      <td>Bug</td>
      <td>Steel</td>
      <td>500</td>
      <td>70</td>
      <td>130</td>
      <td>100</td>
      <td>65</td>
      <td>2</td>
    </tr>
    <tr>
      <th>229</th>
      <td>ScizorMega Scizor</td>
      <td>Bug</td>
      <td>Steel</td>
      <td>600</td>
      <td>70</td>
      <td>150</td>
      <td>140</td>
      <td>75</td>
      <td>2</td>
    </tr>
    <tr>
      <th>231</th>
      <td>Heracross</td>
      <td>Bug</td>
      <td>Fighting</td>
      <td>500</td>
      <td>80</td>
      <td>125</td>
      <td>75</td>
      <td>85</td>
      <td>2</td>
    </tr>
    <tr>
      <th>232</th>
      <td>HeracrossMega Heracross</td>
      <td>Bug</td>
      <td>Fighting</td>
      <td>600</td>
      <td>80</td>
      <td>185</td>
      <td>115</td>
      <td>75</td>
      <td>2</td>
    </tr>
    <tr>
      <th>233</th>
      <td>Sneasel</td>
      <td>Dark</td>
      <td>Ice</td>
      <td>430</td>
      <td>55</td>
      <td>95</td>
      <td>55</td>
      <td>115</td>
      <td>2</td>
    </tr>
    <tr>
      <th>249</th>
      <td>Kingdra</td>
      <td>Water</td>
      <td>Dragon</td>
      <td>540</td>
      <td>75</td>
      <td>95</td>
      <td>95</td>
      <td>85</td>
      <td>2</td>
    </tr>
    <tr>
      <th>267</th>
      <td>Tyranitar</td>
      <td>Rock</td>
      <td>Dark</td>
      <td>600</td>
      <td>100</td>
      <td>134</td>
      <td>110</td>
      <td>61</td>
      <td>2</td>
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
      <th>275</th>
      <td>SceptileMega Sceptile</td>
      <td>Grass</td>
      <td>Dragon</td>
      <td>630</td>
      <td>70</td>
      <td>110</td>
      <td>75</td>
      <td>145</td>
      <td>3</td>
    </tr>
    <tr>
      <th>277</th>
      <td>Combusken</td>
      <td>Fire</td>
      <td>Fighting</td>
      <td>405</td>
      <td>60</td>
      <td>85</td>
      <td>60</td>
      <td>55</td>
      <td>3</td>
    </tr>
    <tr>
      <th>278</th>
      <td>Blaziken</td>
      <td>Fire</td>
      <td>Fighting</td>
      <td>530</td>
      <td>80</td>
      <td>120</td>
      <td>70</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>279</th>
      <td>BlazikenMega Blaziken</td>
      <td>Fire</td>
      <td>Fighting</td>
      <td>630</td>
      <td>80</td>
      <td>160</td>
      <td>80</td>
      <td>100</td>
      <td>3</td>
    </tr>
    <tr>
      <th>310</th>
      <td>Breloom</td>
      <td>Grass</td>
      <td>Fighting</td>
      <td>460</td>
      <td>60</td>
      <td>130</td>
      <td>80</td>
      <td>70</td>
      <td>3</td>
    </tr>
    <tr>
      <th>347</th>
      <td>Carvanha</td>
      <td>Water</td>
      <td>Dark</td>
      <td>305</td>
      <td>45</td>
      <td>90</td>
      <td>20</td>
      <td>65</td>
      <td>3</td>
    </tr>
    <tr>
      <th>348</th>
      <td>Sharpedo</td>
      <td>Water</td>
      <td>Dark</td>
      <td>460</td>
      <td>70</td>
      <td>120</td>
      <td>40</td>
      <td>95</td>
      <td>3</td>
    </tr>
    <tr>
      <th>349</th>
      <td>SharpedoMega Sharpedo</td>
      <td>Water</td>
      <td>Dark</td>
      <td>560</td>
      <td>70</td>
      <td>140</td>
      <td>70</td>
      <td>105</td>
      <td>3</td>
    </tr>
    <tr>
      <th>365</th>
      <td>Altaria</td>
      <td>Dragon</td>
      <td>Flying</td>
      <td>490</td>
      <td>75</td>
      <td>70</td>
      <td>90</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>366</th>
      <td>AltariaMega Altaria</td>
      <td>Dragon</td>
      <td>Fairy</td>
      <td>590</td>
      <td>75</td>
      <td>110</td>
      <td>110</td>
      <td>80</td>
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
      <th>589</th>
      <td>Excadrill</td>
      <td>Ground</td>
      <td>Steel</td>
      <td>508</td>
      <td>110</td>
      <td>135</td>
      <td>60</td>
      <td>88</td>
      <td>5</td>
    </tr>
    <tr>
      <th>627</th>
      <td>Archen</td>
      <td>Rock</td>
      <td>Flying</td>
      <td>401</td>
      <td>55</td>
      <td>112</td>
      <td>45</td>
      <td>70</td>
      <td>5</td>
    </tr>
    <tr>
      <th>628</th>
      <td>Archeops</td>
      <td>Rock</td>
      <td>Flying</td>
      <td>567</td>
      <td>75</td>
      <td>140</td>
      <td>65</td>
      <td>110</td>
      <td>5</td>
    </tr>
    <tr>
      <th>650</th>
      <td>Escavalier</td>
      <td>Bug</td>
      <td>Steel</td>
      <td>495</td>
      <td>70</td>
      <td>135</td>
      <td>105</td>
      <td>20</td>
      <td>5</td>
    </tr>
    <tr>
      <th>685</th>
      <td>Pawniard</td>
      <td>Dark</td>
      <td>Steel</td>
      <td>340</td>
      <td>45</td>
      <td>85</td>
      <td>70</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>686</th>
      <td>Bisharp</td>
      <td>Dark</td>
      <td>Steel</td>
      <td>490</td>
      <td>65</td>
      <td>125</td>
      <td>100</td>
      <td>70</td>
      <td>5</td>
    </tr>
    <tr>
      <th>693</th>
      <td>Durant</td>
      <td>Bug</td>
      <td>Steel</td>
      <td>484</td>
      <td>58</td>
      <td>109</td>
      <td>112</td>
      <td>109</td>
      <td>5</td>
    </tr>
    <tr>
      <th>700</th>
      <td>Terrakion</td>
      <td>Rock</td>
      <td>Fighting</td>
      <td>580</td>
      <td>91</td>
      <td>129</td>
      <td>90</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th>701</th>
      <td>Virizion</td>
      <td>Grass</td>
      <td>Fighting</td>
      <td>580</td>
      <td>91</td>
      <td>90</td>
      <td>72</td>
      <td>108</td>
      <td>5</td>
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
      <th>708</th>
      <td>LandorusIncarnate Forme</td>
      <td>Ground</td>
      <td>Flying</td>
      <td>600</td>
      <td>89</td>
      <td>125</td>
      <td>90</td>
      <td>101</td>
      <td>5</td>
    </tr>
    <tr>
      <th>709</th>
      <td>LandorusTherian Forme</td>
      <td>Ground</td>
      <td>Flying</td>
      <td>600</td>
      <td>89</td>
      <td>145</td>
      <td>90</td>
      <td>91</td>
      <td>5</td>
    </tr>
    <tr>
      <th>710</th>
      <td>Kyurem</td>
      <td>Dragon</td>
      <td>Ice</td>
      <td>660</td>
      <td>125</td>
      <td>130</td>
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
      <th>716</th>
      <td>MeloettaPirouette Forme</td>
      <td>Normal</td>
      <td>Fighting</td>
      <td>600</td>
      <td>100</td>
      <td>128</td>
      <td>90</td>
      <td>128</td>
      <td>5</td>
    </tr>
    <tr>
      <th>717</th>
      <td>Genesect</td>
      <td>Bug</td>
      <td>Steel</td>
      <td>600</td>
      <td>71</td>
      <td>120</td>
      <td>95</td>
      <td>99</td>
      <td>5</td>
    </tr>
    <tr>
      <th>720</th>
      <td>Chesnaught</td>
      <td>Grass</td>
      <td>Fighting</td>
      <td>530</td>
      <td>88</td>
      <td>107</td>
      <td>122</td>
      <td>64</td>
      <td>6</td>
    </tr>
    <tr>
      <th>726</th>
      <td>Greninja</td>
      <td>Water</td>
      <td>Dark</td>
      <td>530</td>
      <td>72</td>
      <td>95</td>
      <td>67</td>
      <td>122</td>
      <td>6</td>
    </tr>
    <tr>
      <th>743</th>
      <td>Pangoro</td>
      <td>Fighting</td>
      <td>Dark</td>
      <td>495</td>
      <td>95</td>
      <td>124</td>
      <td>78</td>
      <td>58</td>
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
<p>90 rows × 9 columns</p>
</div>



**Q7：将类型1和2作为索引列，按照索引来实现分组计算（根据索引来分组计算）**


```python
#将类型1、类型2设置为索引列
df_pokemon = df.set_index(["类型1","类型2"])
```


```python
df_pokemon.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>姓名</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
    <tr>
      <th>类型1</th>
      <th>类型2</th>
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
      <th rowspan="4" valign="top">Grass</th>
      <th>Poison</th>
      <td>Bulbasaur</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Ivysaur</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Venusaur</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>VenusaurMega Venusaur</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fire</th>
      <th>NaN</th>
      <td>Charmander</td>
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
#根据索引分组
grouped3 = df_pokemon.groupby(level=["类型1","类型2"])
grouped3
```




    <pandas.core.groupby.DataFrameGroupBy object at 0x00000000091725F8>




```python
#分组计算各列均值
grouped3.mean()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
    <tr>
      <th>类型1</th>
      <th>类型2</th>
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
      <th rowspan="11" valign="top">Bug</th>
      <th>Electric</th>
      <td>395.500000</td>
      <td>60.000000</td>
      <td>62.000000</td>
      <td>55.000000</td>
      <td>86.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>550.000000</td>
      <td>80.000000</td>
      <td>155.000000</td>
      <td>95.000000</td>
      <td>80.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>455.000000</td>
      <td>70.000000</td>
      <td>72.500000</td>
      <td>60.000000</td>
      <td>80.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>419.500000</td>
      <td>63.000000</td>
      <td>70.142857</td>
      <td>61.571429</td>
      <td>82.857143</td>
      <td>2.857143</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>236.000000</td>
      <td>1.000000</td>
      <td>90.000000</td>
      <td>45.000000</td>
      <td>40.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>384.000000</td>
      <td>55.000000</td>
      <td>73.833333</td>
      <td>76.666667</td>
      <td>44.500000</td>
      <td>3.500000</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>345.000000</td>
      <td>45.500000</td>
      <td>62.000000</td>
      <td>97.500000</td>
      <td>38.000000</td>
      <td>3.500000</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>347.916667</td>
      <td>53.750000</td>
      <td>68.333333</td>
      <td>58.083333</td>
      <td>65.916667</td>
      <td>2.333333</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>435.000000</td>
      <td>46.666667</td>
      <td>56.666667</td>
      <td>146.666667</td>
      <td>35.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>509.714286</td>
      <td>67.714286</td>
      <td>114.714286</td>
      <td>112.428571</td>
      <td>63.428571</td>
      <td>3.571429</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>269.000000</td>
      <td>40.000000</td>
      <td>30.000000</td>
      <td>32.000000</td>
      <td>65.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Dark</th>
      <th>Dragon</th>
      <td>440.000000</td>
      <td>72.000000</td>
      <td>85.000000</td>
      <td>70.000000</td>
      <td>64.666667</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>418.000000</td>
      <td>57.500000</td>
      <td>82.500000</td>
      <td>92.500000</td>
      <td>53.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>476.666667</td>
      <td>65.000000</td>
      <td>80.000000</td>
      <td>56.666667</td>
      <td>91.666667</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>494.000000</td>
      <td>93.200000</td>
      <td>92.200000</td>
      <td>73.800000</td>
      <td>80.200000</td>
      <td>4.400000</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>430.000000</td>
      <td>50.000000</td>
      <td>80.000000</td>
      <td>100.000000</td>
      <td>35.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>470.000000</td>
      <td>62.500000</td>
      <td>107.500000</td>
      <td>60.000000</td>
      <td>120.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>385.000000</td>
      <td>69.500000</td>
      <td>73.000000</td>
      <td>70.500000</td>
      <td>59.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>415.000000</td>
      <td>55.000000</td>
      <td>105.000000</td>
      <td>85.000000</td>
      <td>65.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Dragon</th>
      <th>Electric</th>
      <td>680.000000</td>
      <td>100.000000</td>
      <td>150.000000</td>
      <td>120.000000</td>
      <td>90.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>590.000000</td>
      <td>75.000000</td>
      <td>110.000000</td>
      <td>110.000000</td>
      <td>80.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>680.000000</td>
      <td>100.000000</td>
      <td>120.000000</td>
      <td>100.000000</td>
      <td>90.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>641.666667</td>
      <td>94.333333</td>
      <td>135.666667</td>
      <td>97.500000</td>
      <td>98.333333</td>
      <td>2.666667</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>522.000000</td>
      <td>90.000000</td>
      <td>112.000000</td>
      <td>88.200000</td>
      <td>82.600000</td>
      <td>4.400000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>686.666667</td>
      <td>125.000000</td>
      <td>140.000000</td>
      <td>93.333333</td>
      <td>95.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>650.000000</td>
      <td>80.000000</td>
      <td>100.000000</td>
      <td>97.500000</td>
      <td>110.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Electric</th>
      <th>Dragon</th>
      <td>610.000000</td>
      <td>90.000000</td>
      <td>95.000000</td>
      <td>105.000000</td>
      <td>45.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>431.000000</td>
      <td>67.000000</td>
      <td>58.000000</td>
      <td>57.000000</td>
      <td>101.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>520.000000</td>
      <td>50.000000</td>
      <td>65.000000</td>
      <td>107.000000</td>
      <td>86.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>537.600000</td>
      <td>70.600000</td>
      <td>90.000000</td>
      <td>78.400000</td>
      <td>100.200000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Rock</th>
      <th>Fighting</th>
      <td>580.000000</td>
      <td>91.000000</td>
      <td>129.000000</td>
      <td>90.000000</td>
      <td>108.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>524.500000</td>
      <td>72.500000</td>
      <td>123.000000</td>
      <td>65.000000</td>
      <td>115.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>425.000000</td>
      <td>76.000000</td>
      <td>61.000000</td>
      <td>87.000000</td>
      <td>33.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>380.000000</td>
      <td>55.000000</td>
      <td>81.333333</td>
      <td>104.166667</td>
      <td>43.666667</td>
      <td>1.333333</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>441.500000</td>
      <td>100.000000</td>
      <td>68.000000</td>
      <td>61.000000</td>
      <td>52.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>440.000000</td>
      <td>70.000000</td>
      <td>75.000000</td>
      <td>75.000000</td>
      <td>70.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>456.666667</td>
      <td>50.000000</td>
      <td>49.666667</td>
      <td>143.666667</td>
      <td>33.333333</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>417.666667</td>
      <td>51.500000</td>
      <td>75.333333</td>
      <td>100.333333</td>
      <td>57.166667</td>
      <td>2.666667</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Steel</th>
      <th>Dragon</th>
      <td>680.000000</td>
      <td>100.000000</td>
      <td>120.000000</td>
      <td>120.000000</td>
      <td>90.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>443.333333</td>
      <td>52.333333</td>
      <td>90.000000</td>
      <td>100.333333</td>
      <td>58.333333</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>580.000000</td>
      <td>91.000000</td>
      <td>90.000000</td>
      <td>129.000000</td>
      <td>108.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>465.000000</td>
      <td>65.000000</td>
      <td>80.000000</td>
      <td>140.000000</td>
      <td>70.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>453.250000</td>
      <td>56.000000</td>
      <td>97.500000</td>
      <td>112.500000</td>
      <td>45.750000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>560.000000</td>
      <td>75.000000</td>
      <td>105.000000</td>
      <td>215.000000</td>
      <td>30.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>488.571429</td>
      <td>69.142857</td>
      <td>89.000000</td>
      <td>108.857143</td>
      <td>59.428571</td>
      <td>3.285714</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>430.000000</td>
      <td>60.000000</td>
      <td>90.000000</td>
      <td>140.000000</td>
      <td>40.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th rowspan="14" valign="top">Water</th>
      <th>Dark</th>
      <td>493.833333</td>
      <td>69.166667</td>
      <td>120.000000</td>
      <td>65.166667</td>
      <td>87.166667</td>
      <td>3.166667</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>610.000000</td>
      <td>82.500000</td>
      <td>107.500000</td>
      <td>97.500000</td>
      <td>92.500000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>395.000000</td>
      <td>100.000000</td>
      <td>48.000000</td>
      <td>48.000000</td>
      <td>67.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>335.000000</td>
      <td>85.000000</td>
      <td>35.000000</td>
      <td>65.000000</td>
      <td>45.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>556.666667</td>
      <td>90.666667</td>
      <td>79.666667</td>
      <td>91.666667</td>
      <td>95.333333</td>
      <td>3.666667</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>404.000000</td>
      <td>63.142857</td>
      <td>56.571429</td>
      <td>63.142857</td>
      <td>72.000000</td>
      <td>3.285714</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>407.500000</td>
      <td>77.500000</td>
      <td>50.000000</td>
      <td>60.000000</td>
      <td>50.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>346.666667</td>
      <td>60.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>433.900000</td>
      <td>87.100000</td>
      <td>84.400000</td>
      <td>71.400000</td>
      <td>53.200000</td>
      <td>3.300000</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>511.666667</td>
      <td>90.000000</td>
      <td>83.333333</td>
      <td>113.333333</td>
      <td>66.666667</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>426.666667</td>
      <td>61.666667</td>
      <td>68.333333</td>
      <td>58.333333</td>
      <td>85.000000</td>
      <td>1.333333</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>481.000000</td>
      <td>87.000000</td>
      <td>73.000000</td>
      <td>104.000000</td>
      <td>44.000000</td>
      <td>1.200000</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>428.750000</td>
      <td>70.750000</td>
      <td>82.750000</td>
      <td>112.750000</td>
      <td>36.000000</td>
      <td>3.750000</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>530.000000</td>
      <td>84.000000</td>
      <td>86.000000</td>
      <td>88.000000</td>
      <td>60.000000</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
<p>136 rows × 6 columns</p>
</div>



### 2.组的一些特征

**查看每个索引组的个数**


```python
grouped2.size()
```




    类型1       类型2     
    Bug       Electric     2
              Fighting     2
              Fire         2
              Flying      14
              Ghost        1
              Grass        6
              Ground       2
              Poison      12
              Rock         3
              Steel        7
              Water        1
    Dark      Dragon       3
              Fighting     2
              Fire         3
              Flying       5
              Ghost        2
              Ice          2
              Psychic      2
              Steel        2
    Dragon    Electric     1
              Fairy        1
              Fire         1
              Flying       6
              Ground       5
              Ice          3
              Psychic      4
    Electric  Dragon       1
              Fairy        1
              Fire         1
              Flying       5
                          ..
    Rock      Fighting     1
              Flying       4
              Grass        2
              Ground       6
              Ice          2
              Psychic      2
              Steel        3
              Water        6
    Steel     Dragon       1
              Fairy        3
              Fighting     1
              Flying       1
              Ghost        4
              Ground       2
              Psychic      7
              Rock         3
    Water     Dark         6
              Dragon       2
              Electric     2
              Fairy        2
              Fighting     3
              Flying       7
              Ghost        2
              Grass        3
              Ground      10
              Ice          3
              Poison       3
              Psychic      5
              Rock         4
              Steel        1
    Length: 136, dtype: int64



**得到每个索引组的在源数据中的索引位置**


```python
grouped2.groups
```




    {('Grass',
      'Poison'): Int64Index([0, 1, 2, 3, 48, 49, 50, 75, 76, 77, 344, 451, 452, 651, 652], dtype='int64'),
     ('Fire',
      nan): Int64Index([  4,   5,  42,  43,  63,  64,  83,  84, 135, 147, 169, 170, 171,
                 236, 259, 263, 276, 355, 435, 518, 557, 572, 573, 614, 615, 692,
                 721, 722],
                dtype='int64'),
     ('Fire', 'Flying'): Int64Index([6, 8, 158, 270, 730, 731], dtype='int64'),
     ('Fire', 'Dragon'): Int64Index([7], dtype='int64'),
     ('Water',
      nan): Int64Index([  9,  10,  11,  12,  59,  60,  65,  66,  93,  97, 106, 107, 125,
                 126, 127, 128, 129, 139, 145, 172, 173, 174, 201, 241, 242, 264,
                 280, 350, 351, 373, 381, 382, 401, 402, 403, 405, 421, 422, 438,
                 439, 465, 466, 469, 506, 507, 547, 548, 560, 561, 562, 574, 575,
                 595, 610, 655, 724, 725, 762, 763],
                dtype='int64'),
     ('Bug',
      nan): Int64Index([ 13,  14, 136, 219, 288, 289, 291, 342, 343, 446, 447, 457, 649,
                 677, 678, 732, 733],
                dtype='int64'),
     ('Bug',
      'Flying'): Int64Index([15, 132, 137, 179, 180, 208, 290, 308, 315, 461, 462, 463, 520,
                 734],
                dtype='int64'),
     ('Bug',
      'Poison'): Int64Index([16, 17, 18, 19, 53, 54, 181, 182, 292, 603, 604, 605], dtype='int64'),
     ('Normal',
      'Flying'): Int64Index([ 20,  21,  22,  23,  26,  27,  90,  91,  92, 177, 178, 299, 300,
                 364, 441, 442, 443, 489, 578, 579, 580, 688, 689, 729],
                dtype='int64'),
     ('Normal',
      nan): Int64Index([ 24,  25,  57,  58, 116, 121, 123, 124, 138, 143, 144, 148, 155,
                 175, 176, 205, 221, 234, 235, 252, 253, 254, 260, 261, 286, 287,
                 311, 312, 313, 317, 318, 319, 324, 325, 358, 367, 383, 384, 444,
                 471, 474, 475, 479, 480, 488, 495, 514, 525, 543, 552, 563, 564,
                 565, 566, 567, 590, 633, 634, 687, 727, 744],
                dtype='int64'),
     ('Poison',
      nan): Int64Index([28, 29, 34, 35, 37, 38, 95, 96, 117, 118, 345, 346, 368, 629, 630], dtype='int64'),
     ('Electric',
      nan): Int64Index([ 30,  31, 108, 109, 134, 146, 186, 193, 194, 195, 258, 262, 337,
                 338, 339, 340, 341, 448, 449, 450, 464, 517, 581, 582, 663, 664,
                 665],
                dtype='int64'),
     ('Ground',
      nan): Int64Index([32, 33, 55, 56, 112, 113, 250, 251, 359, 423, 499, 500, 588], dtype='int64'),
     ('Poison', 'Ground'): Int64Index([36, 39], dtype='int64'),
     ('Fairy',
      nan): Int64Index([40, 41, 187, 189, 225, 226, 737, 738, 739, 752, 753, 754, 755, 770,
                 792],
                dtype='int64'),
     ('Normal', 'Fairy'): Int64Index([44, 45, 188, 322, 591], dtype='int64'),
     ('Poison', 'Flying'): Int64Index([46, 47, 183], dtype='int64'),
     ('Bug', 'Grass'): Int64Index([51, 52, 458, 600, 601, 602], dtype='int64'),
     ('Fighting',
      nan): Int64Index([ 61,  62,  72,  73,  74, 114, 115, 255, 256, 320, 321, 496, 592,
                 593, 594, 598, 599, 680, 681, 742],
                dtype='int64'),
     ('Water', 'Fighting'): Int64Index([67, 713, 714], dtype='int64'),
     ('Psychic',
      nan): Int64Index([ 68,  69,  70,  71, 104, 105, 162, 164, 165, 211, 216, 217, 356,
                 357, 391, 394, 428, 429, 430, 431, 481, 537, 538, 539, 546, 576,
                 577, 635, 636, 637, 638, 639, 640, 666, 667, 745, 746, 747],
                dtype='int64'),
     ('Water', 'Poison'): Int64Index([78, 79, 227], dtype='int64'),
     ('Rock', 'Ground'): Int64Index([80, 81, 82, 103, 265, 266], dtype='int64'),
     ('Water', 'Psychic'): Int64Index([85, 86, 87, 130, 214], dtype='int64'),
     ('Electric', 'Steel'): Int64Index([88, 89, 513], dtype='int64'),
     ('Water', 'Ice'): Int64Index([94, 98, 142], dtype='int64'),
     ('Ghost', 'Poison'): Int64Index([99, 100, 101, 102], dtype='int64'),
     ('Grass', 'Psychic'): Int64Index([110, 111], dtype='int64'),
     ('Ground', 'Rock'): Int64Index([119, 120, 515], dtype='int64'),
     ('Grass',
      nan): Int64Index([122, 166, 167, 168, 197, 206, 207, 272, 273, 274, 296, 309, 362,
                 432, 433, 467, 468, 505, 516, 521, 550, 554, 555, 556, 570, 571,
                 608, 609, 617, 718, 719, 740, 741],
                dtype='int64'),
     ('Psychic',
      'Fairy'): Int64Index([131, 303, 304, 305, 306, 487], dtype='int64'),
     ('Ice', 'Psychic'): Int64Index([133, 257], dtype='int64'),
     ('Water',
      'Flying'): Int64Index([140, 244, 301, 302, 508, 641, 642], dtype='int64'),
     ('Water', 'Dark'): Int64Index([141, 347, 348, 349, 374, 726], dtype='int64'),
     ('Rock', 'Water'): Int64Index([149, 150, 151, 152, 758, 759], dtype='int64'),
     ('Rock', 'Flying'): Int64Index([153, 154, 627, 628], dtype='int64'),
     ('Ice', 'Flying'): Int64Index([156, 243], dtype='int64'),
     ('Electric', 'Flying'): Int64Index([157, 535, 648, 704, 705], dtype='int64'),
     ('Dragon',
      nan): Int64Index([159, 160, 406, 407, 671, 672, 673, 682, 774, 775, 776], dtype='int64'),
     ('Dragon',
      'Flying'): Int64Index([161, 365, 408, 409, 425, 426], dtype='int64'),
     ('Psychic', 'Fighting'): Int64Index([163, 526, 527], dtype='int64'),
     ('Water', 'Electric'): Int64Index([184, 185], dtype='int64'),
     ('Fairy', 'Flying'): Int64Index([190, 519], dtype='int64'),
     ('Psychic',
      'Flying'): Int64Index([191, 192, 269, 586, 587, 622], dtype='int64'),
     ('Electric', 'Dragon'): Int64Index([196], dtype='int64'),
     ('Water', 'Fairy'): Int64Index([198, 199], dtype='int64'),
     ('Rock',
      nan): Int64Index([200, 323, 414, 453, 454, 486, 583, 584, 585], dtype='int64'),
     ('Grass', 'Flying'): Int64Index([202, 203, 204, 390, 551], dtype='int64'),
     ('Water',
      'Ground'): Int64Index([209, 210, 281, 282, 283, 371, 372, 470, 596, 597], dtype='int64'),
     ('Dark',
      nan): Int64Index([212, 284, 285, 392, 393, 549, 568, 569, 631, 632], dtype='int64'),
     ('Dark', 'Flying'): Int64Index([213, 478, 690, 691, 793], dtype='int64'),
     ('Ghost',
      nan): Int64Index([215, 385, 386, 387, 388, 389, 477, 529, 623, 624], dtype='int64'),
     ('Normal', 'Psychic'): Int64Index([218, 715], dtype='int64'),
     ('Bug',
      'Steel'): Int64Index([220, 228, 229, 460, 650, 693, 717], dtype='int64'),
     ('Ground', 'Flying'): Int64Index([222, 523, 708, 709], dtype='int64'),
     ('Steel', 'Ground'): Int64Index([223, 224], dtype='int64'),
     ('Bug', 'Rock'): Int64Index([230, 618, 619], dtype='int64'),
     ('Bug', 'Fighting'): Int64Index([231, 232], dtype='int64'),
     ('Dark', 'Ice'): Int64Index([233, 512], dtype='int64'),
     ('Fire', 'Rock'): Int64Index([237], dtype='int64'),
     ('Ice', 'Ground'): Int64Index([238, 239, 524], dtype='int64'),
     ('Water', 'Rock'): Int64Index([240, 404, 625, 626], dtype='int64'),
     ('Steel', 'Flying'): Int64Index([245], dtype='int64'),
     ('Dark', 'Fire'): Int64Index([246, 247, 248], dtype='int64'),
     ('Water', 'Dragon'): Int64Index([249, 541], dtype='int64'),
     ('Rock', 'Dark'): Int64Index([267, 268], dtype='int64'),
     ('Psychic', 'Grass'): Int64Index([271], dtype='int64'),
     ('Grass', 'Dragon'): Int64Index([275], dtype='int64'),
     ('Fire',
      'Fighting'): Int64Index([277, 278, 279, 436, 437, 558, 559], dtype='int64'),
     ('Water', 'Grass'): Int64Index([293, 294, 295], dtype='int64'),
     ('Grass', 'Dark'): Int64Index([297, 298, 363], dtype='int64'),
     ('Bug', 'Water'): Int64Index([307], dtype='int64'),
     ('Grass', 'Fighting'): Int64Index([310, 701, 720], dtype='int64'),
     ('Bug', 'Ground'): Int64Index([314, 459], dtype='int64'),
     ('Bug', 'Ghost'): Int64Index([316], dtype='int64'),
     ('Dark', 'Ghost'): Int64Index([326, 327], dtype='int64'),
     ('Steel', 'Fairy'): Int64Index([328, 329, 777], dtype='int64'),
     ('Steel', 'Rock'): Int64Index([330, 331, 332], dtype='int64'),
     ('Steel', nan): Int64Index([333, 416, 660, 661, 662], dtype='int64'),
     ('Fighting', 'Psychic'): Int64Index([334, 335, 336], dtype='int64'),
     ('Fire', 'Ground'): Int64Index([352, 353, 354], dtype='int64'),
     ('Ground', 'Dragon'): Int64Index([360, 361], dtype='int64'),
     ('Dragon', 'Fairy'): Int64Index([366], dtype='int64'),
     ('Rock', 'Psychic'): Int64Index([369, 370], dtype='int64'),
     ('Ground', 'Psychic'): Int64Index([375, 376], dtype='int64'),
     ('Rock', 'Grass'): Int64Index([377, 378], dtype='int64'),
     ('Rock', 'Bug'): Int64Index([379, 380], dtype='int64'),
     ('Ice',
      nan): Int64Index([395, 396, 397, 415, 522, 643, 644, 645, 674, 675, 676, 788, 789], dtype='int64'),
     ('Ice', 'Water'): Int64Index([398, 399, 400], dtype='int64'),
     ('Steel',
      'Psychic'): Int64Index([410, 411, 412, 413, 427, 484, 485], dtype='int64'),
     ('Dragon', 'Psychic'): Int64Index([417, 418, 419, 420], dtype='int64'),
     ('Ground', 'Fire'): Int64Index([424], dtype='int64'),
     ('Grass', 'Ground'): Int64Index([434], dtype='int64'),
     ('Water', 'Steel'): Int64Index([440], dtype='int64'),
     ('Normal', 'Water'): Int64Index([445], dtype='int64'),
     ('Rock', 'Steel'): Int64Index([455, 456, 528], dtype='int64'),
     ('Ghost', 'Flying'): Int64Index([472, 473], dtype='int64'),
     ('Normal', 'Fighting'): Int64Index([476, 716], dtype='int64'),
     ('Poison', 'Dark'): Int64Index([482, 483, 502], dtype='int64'),
     ('Ghost', 'Dark'): Int64Index([490], dtype='int64'),
     ('Dragon', 'Ground'): Int64Index([491, 492, 493, 494, 794], dtype='int64'),
     ('Fighting', 'Steel'): Int64Index([497, 498], dtype='int64'),
     ('Poison', 'Bug'): Int64Index([501], dtype='int64'),
     ('Poison', 'Fighting'): Int64Index([503, 504], dtype='int64'),
     ('Grass', 'Ice'): Int64Index([509, 510, 511], dtype='int64'),
     ('Ice', 'Ghost'): Int64Index([530], dtype='int64'),
     ('Electric', 'Ghost'): Int64Index([531], dtype='int64'),
     ('Electric', 'Fire'): Int64Index([532], dtype='int64'),
     ('Electric', 'Water'): Int64Index([533], dtype='int64'),
     ('Electric', 'Ice'): Int64Index([534], dtype='int64'),
     ('Electric', 'Grass'): Int64Index([536], dtype='int64'),
     ('Steel', 'Dragon'): Int64Index([540], dtype='int64'),
     ('Fire', 'Steel'): Int64Index([542], dtype='int64'),
     ('Ghost', 'Dragon'): Int64Index([544, 545], dtype='int64'),
     ('Psychic', 'Fire'): Int64Index([553], dtype='int64'),
     ('Ground', 'Steel'): Int64Index([589], dtype='int64'),
     ('Grass', 'Fairy'): Int64Index([606, 607], dtype='int64'),
     ('Ground', 'Dark'): Int64Index([611, 612, 613], dtype='int64'),
     ('Fire', 'Psychic'): Int64Index([616, 723], dtype='int64'),
     ('Dark', 'Fighting'): Int64Index([620, 621], dtype='int64'),
     ('Normal', 'Grass'): Int64Index([646, 647], dtype='int64'),
     ('Water', 'Ghost'): Int64Index([653, 654], dtype='int64'),
     ('Bug', 'Electric'): Int64Index([656, 657], dtype='int64'),
     ('Grass', 'Steel'): Int64Index([658, 659], dtype='int64'),
     ('Ghost', 'Fire'): Int64Index([668, 669, 670], dtype='int64'),
     ('Ground', 'Electric'): Int64Index([679], dtype='int64'),
     ('Ground', 'Ghost'): Int64Index([683, 684], dtype='int64'),
     ('Dark', 'Steel'): Int64Index([685, 686], dtype='int64'),
     ('Dark', 'Dragon'): Int64Index([694, 695, 696], dtype='int64'),
     ('Bug', 'Fire'): Int64Index([697, 698], dtype='int64'),
     ('Steel', 'Fighting'): Int64Index([699], dtype='int64'),
     ('Rock', 'Fighting'): Int64Index([700], dtype='int64'),
     ('Flying', nan): Int64Index([702, 703], dtype='int64'),
     ('Dragon', 'Fire'): Int64Index([706], dtype='int64'),
     ('Dragon', 'Electric'): Int64Index([707], dtype='int64'),
     ('Dragon', 'Ice'): Int64Index([710, 711, 712], dtype='int64'),
     ('Normal', 'Ground'): Int64Index([728], dtype='int64'),
     ('Fire', 'Normal'): Int64Index([735, 736], dtype='int64'),
     ('Fighting', 'Dark'): Int64Index([743], dtype='int64'),
     ('Steel', 'Ghost'): Int64Index([748, 749, 750, 751], dtype='int64'),
     ('Dark', 'Psychic'): Int64Index([756, 757], dtype='int64'),
     ('Poison', 'Water'): Int64Index([760], dtype='int64'),
     ('Poison', 'Dragon'): Int64Index([761], dtype='int64'),
     ('Electric', 'Normal'): Int64Index([764, 765], dtype='int64'),
     ('Rock', 'Dragon'): Int64Index([766, 767], dtype='int64'),
     ('Rock', 'Ice'): Int64Index([768, 769], dtype='int64'),
     ('Fighting', 'Flying'): Int64Index([771], dtype='int64'),
     ('Electric', 'Fairy'): Int64Index([772], dtype='int64'),
     ('Rock', 'Fairy'): Int64Index([773, 795, 796], dtype='int64'),
     ('Ghost',
      'Grass'): Int64Index([778, 779, 780, 781, 782, 783, 784, 785, 786, 787], dtype='int64'),
     ('Flying', 'Dragon'): Int64Index([790, 791], dtype='int64'),
     ('Psychic', 'Ghost'): Int64Index([797], dtype='int64'),
     ('Psychic', 'Dark'): Int64Index([798], dtype='int64'),
     ('Fire', 'Water'): Int64Index([799], dtype='int64')}



**得到包含索引组的所有数据**


```python
#得到索引组为Fire和Flying的所有数据
grouped2.get_group(('Fire', 'Flying'))
```




<div>

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
      <th>158</th>
      <td>Moltres</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>580</td>
      <td>90</td>
      <td>100</td>
      <td>90</td>
      <td>90</td>
      <td>1</td>
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
      <th>730</th>
      <td>Fletchinder</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>382</td>
      <td>62</td>
      <td>73</td>
      <td>55</td>
      <td>84</td>
      <td>6</td>
    </tr>
    <tr>
      <th>731</th>
      <td>Talonflame</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>499</td>
      <td>78</td>
      <td>81</td>
      <td>71</td>
      <td>126</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



### 3.组的迭代


```python
for name,group in grouped2:
    print(name)
    print(group.shape)
```

    ('Bug', 'Electric')
    (2, 9)
    ('Bug', 'Fighting')
    (2, 9)
    ('Bug', 'Fire')
    (2, 9)
    ('Bug', 'Flying')
    (14, 9)
    ('Bug', 'Ghost')
    (1, 9)
    ('Bug', 'Grass')
    (6, 9)
    ('Bug', 'Ground')
    (2, 9)
    ('Bug', 'Poison')
    (12, 9)
    ('Bug', 'Rock')
    (3, 9)
    ('Bug', 'Steel')
    (7, 9)
    ('Bug', 'Water')
    (1, 9)
    ('Dark', 'Dragon')
    (3, 9)
    ('Dark', 'Fighting')
    (2, 9)
    ('Dark', 'Fire')
    (3, 9)
    ('Dark', 'Flying')
    (5, 9)
    ('Dark', 'Ghost')
    (2, 9)
    ('Dark', 'Ice')
    (2, 9)
    ('Dark', 'Psychic')
    (2, 9)
    ('Dark', 'Steel')
    (2, 9)
    ('Dragon', 'Electric')
    (1, 9)
    ('Dragon', 'Fairy')
    (1, 9)
    ('Dragon', 'Fire')
    (1, 9)
    ('Dragon', 'Flying')
    (6, 9)
    ('Dragon', 'Ground')
    (5, 9)
    ('Dragon', 'Ice')
    (3, 9)
    ('Dragon', 'Psychic')
    (4, 9)
    ('Electric', 'Dragon')
    (1, 9)
    ('Electric', 'Fairy')
    (1, 9)
    ('Electric', 'Fire')
    (1, 9)
    ('Electric', 'Flying')
    (5, 9)
    ('Electric', 'Ghost')
    (1, 9)
    ('Electric', 'Grass')
    (1, 9)
    ('Electric', 'Ice')
    (1, 9)
    ('Electric', 'Normal')
    (2, 9)
    ('Electric', 'Steel')
    (3, 9)
    ('Electric', 'Water')
    (1, 9)
    ('Fairy', 'Flying')
    (2, 9)
    ('Fighting', 'Dark')
    (1, 9)
    ('Fighting', 'Flying')
    (1, 9)
    ('Fighting', 'Psychic')
    (3, 9)
    ('Fighting', 'Steel')
    (2, 9)
    ('Fire', 'Dragon')
    (1, 9)
    ('Fire', 'Fighting')
    (7, 9)
    ('Fire', 'Flying')
    (6, 9)
    ('Fire', 'Ground')
    (3, 9)
    ('Fire', 'Normal')
    (2, 9)
    ('Fire', 'Psychic')
    (2, 9)
    ('Fire', 'Rock')
    (1, 9)
    ('Fire', 'Steel')
    (1, 9)
    ('Fire', 'Water')
    (1, 9)
    ('Flying', 'Dragon')
    (2, 9)
    ('Ghost', 'Dark')
    (1, 9)
    ('Ghost', 'Dragon')
    (2, 9)
    ('Ghost', 'Fire')
    (3, 9)
    ('Ghost', 'Flying')
    (2, 9)
    ('Ghost', 'Grass')
    (10, 9)
    ('Ghost', 'Poison')
    (4, 9)
    ('Grass', 'Dark')
    (3, 9)
    ('Grass', 'Dragon')
    (1, 9)
    ('Grass', 'Fairy')
    (2, 9)
    ('Grass', 'Fighting')
    (3, 9)
    ('Grass', 'Flying')
    (5, 9)
    ('Grass', 'Ground')
    (1, 9)
    ('Grass', 'Ice')
    (3, 9)
    ('Grass', 'Poison')
    (15, 9)
    ('Grass', 'Psychic')
    (2, 9)
    ('Grass', 'Steel')
    (2, 9)
    ('Ground', 'Dark')
    (3, 9)
    ('Ground', 'Dragon')
    (2, 9)
    ('Ground', 'Electric')
    (1, 9)
    ('Ground', 'Fire')
    (1, 9)
    ('Ground', 'Flying')
    (4, 9)
    ('Ground', 'Ghost')
    (2, 9)
    ('Ground', 'Psychic')
    (2, 9)
    ('Ground', 'Rock')
    (3, 9)
    ('Ground', 'Steel')
    (1, 9)
    ('Ice', 'Flying')
    (2, 9)
    ('Ice', 'Ghost')
    (1, 9)
    ('Ice', 'Ground')
    (3, 9)
    ('Ice', 'Psychic')
    (2, 9)
    ('Ice', 'Water')
    (3, 9)
    ('Normal', 'Fairy')
    (5, 9)
    ('Normal', 'Fighting')
    (2, 9)
    ('Normal', 'Flying')
    (24, 9)
    ('Normal', 'Grass')
    (2, 9)
    ('Normal', 'Ground')
    (1, 9)
    ('Normal', 'Psychic')
    (2, 9)
    ('Normal', 'Water')
    (1, 9)
    ('Poison', 'Bug')
    (1, 9)
    ('Poison', 'Dark')
    (3, 9)
    ('Poison', 'Dragon')
    (1, 9)
    ('Poison', 'Fighting')
    (2, 9)
    ('Poison', 'Flying')
    (3, 9)
    ('Poison', 'Ground')
    (2, 9)
    ('Poison', 'Water')
    (1, 9)
    ('Psychic', 'Dark')
    (1, 9)
    ('Psychic', 'Fairy')
    (6, 9)
    ('Psychic', 'Fighting')
    (3, 9)
    ('Psychic', 'Fire')
    (1, 9)
    ('Psychic', 'Flying')
    (6, 9)
    ('Psychic', 'Ghost')
    (1, 9)
    ('Psychic', 'Grass')
    (1, 9)
    ('Rock', 'Bug')
    (2, 9)
    ('Rock', 'Dark')
    (2, 9)
    ('Rock', 'Dragon')
    (2, 9)
    ('Rock', 'Fairy')
    (3, 9)
    ('Rock', 'Fighting')
    (1, 9)
    ('Rock', 'Flying')
    (4, 9)
    ('Rock', 'Grass')
    (2, 9)
    ('Rock', 'Ground')
    (6, 9)
    ('Rock', 'Ice')
    (2, 9)
    ('Rock', 'Psychic')
    (2, 9)
    ('Rock', 'Steel')
    (3, 9)
    ('Rock', 'Water')
    (6, 9)
    ('Steel', 'Dragon')
    (1, 9)
    ('Steel', 'Fairy')
    (3, 9)
    ('Steel', 'Fighting')
    (1, 9)
    ('Steel', 'Flying')
    (1, 9)
    ('Steel', 'Ghost')
    (4, 9)
    ('Steel', 'Ground')
    (2, 9)
    ('Steel', 'Psychic')
    (7, 9)
    ('Steel', 'Rock')
    (3, 9)
    ('Water', 'Dark')
    (6, 9)
    ('Water', 'Dragon')
    (2, 9)
    ('Water', 'Electric')
    (2, 9)
    ('Water', 'Fairy')
    (2, 9)
    ('Water', 'Fighting')
    (3, 9)
    ('Water', 'Flying')
    (7, 9)
    ('Water', 'Ghost')
    (2, 9)
    ('Water', 'Grass')
    (3, 9)
    ('Water', 'Ground')
    (10, 9)
    ('Water', 'Ice')
    (3, 9)
    ('Water', 'Poison')
    (3, 9)
    ('Water', 'Psychic')
    (5, 9)
    ('Water', 'Rock')
    (4, 9)
    ('Water', 'Steel')
    (1, 9)


## 二.数据透视表

### 1.数据透视表pivot_table


```python
#示例数据
df_p = df.iloc[:10,0:6]
df_p
```




<div>

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
    </tr>
    <tr>
      <th>1</th>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Charmeleon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charizard</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CharizardMega Charizard Y</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Squirtle</td>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
    </tr>
  </tbody>
</table>
</div>




```python
#做一些修改
df_p.loc[0:2,"姓名"] = "A"
df_p.loc[3:5,"姓名"] = "B"
df_p.loc[6:9,"姓名"] = "C"
df_p["类型2"] = df_p["类型2"].fillna("Flying")
df_p.rename(columns={"姓名":"组"},inplace=True)
```


```python
df_p
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
      <th>组</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>5</th>
      <td>B</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>C</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
    </tr>
    <tr>
      <th>7</th>
      <td>C</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
    </tr>
    <tr>
      <th>8</th>
      <td>C</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
    </tr>
    <tr>
      <th>9</th>
      <td>C</td>
      <td>Water</td>
      <td>Flying</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
    </tr>
  </tbody>
</table>
</div>




```python
#将组放在行上，类型1放在列上，计算字段为攻击力，如果没有指定，默认计算其均值
df_p.pivot_table(index="组",columns="类型1",values="攻击力")
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>类型1</th>
      <th>Fire</th>
      <th>Grass</th>
      <th>Water</th>
    </tr>
    <tr>
      <th>组</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>NaN</td>
      <td>64.333333</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>B</th>
      <td>58.0</td>
      <td>100.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>C</th>
      <td>106.0</td>
      <td>NaN</td>
      <td>48.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#将组放在行上，类型1放在列上，计算攻击力的均值和计数
df_p.pivot_table(index="组",columns="类型1",values="攻击力",aggfunc=[np.mean,len])
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
    <tr>
      <th></th>
      <th colspan="3" halign="left">mean</th>
      <th colspan="3" halign="left">len</th>
    </tr>
    <tr>
      <th>类型1</th>
      <th>Fire</th>
      <th>Grass</th>
      <th>Water</th>
      <th>Fire</th>
      <th>Grass</th>
      <th>Water</th>
    </tr>
    <tr>
      <th>组</th>
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
      <th>A</th>
      <td>NaN</td>
      <td>64.333333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>B</th>
      <td>58.0</td>
      <td>100.000000</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>C</th>
      <td>106.0</td>
      <td>NaN</td>
      <td>48.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#将组和类型1放在行上，类型2放在列上，计算攻击力的均值和计数
df_p.pivot_table(index=["组","类型1"],columns="类型2",values="攻击力",aggfunc=[np.mean,len])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">mean</th>
      <th colspan="3" halign="left">len</th>
    </tr>
    <tr>
      <th></th>
      <th>类型2</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
    </tr>
    <tr>
      <th>组</th>
      <th>类型1</th>
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
      <th>A</th>
      <th>Grass</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>64.333333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>Fire</th>
      <td>NaN</td>
      <td>58.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">C</th>
      <th>Fire</th>
      <td>130.0</td>
      <td>94.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>NaN</td>
      <td>48.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#将组和类型1放在行上，类型2放在列上，计算生命值和攻击力的均值和计数
df_p.pivot_table(index=["组","类型1"],columns="类型2",values=["生命值","攻击力"],aggfunc=[np.mean,len])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="6" halign="left">mean</th>
      <th colspan="6" halign="left">len</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">攻击力</th>
      <th colspan="3" halign="left">生命值</th>
      <th colspan="3" halign="left">攻击力</th>
      <th colspan="3" halign="left">生命值</th>
    </tr>
    <tr>
      <th></th>
      <th>类型2</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
    </tr>
    <tr>
      <th>组</th>
      <th>类型1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>A</th>
      <th>Grass</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>64.333333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61.666667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>Fire</th>
      <td>NaN</td>
      <td>58.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>48.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>80.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">C</th>
      <th>Fire</th>
      <td>130.0</td>
      <td>94.0</td>
      <td>NaN</td>
      <td>78.0</td>
      <td>78.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>NaN</td>
      <td>48.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#将组和类型1放在行上，类型2放在列上，计算生命值和攻击力的均值和计数，并且将缺失值填充为0
df_p1 = df_p.pivot_table(index=["组","类型1"],columns="类型2",values=["生命值","攻击力"],aggfunc=[np.mean,len],fill_value=0)
df_p1
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="6" halign="left">mean</th>
      <th colspan="6" halign="left">len</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">攻击力</th>
      <th colspan="3" halign="left">生命值</th>
      <th colspan="3" halign="left">攻击力</th>
      <th colspan="3" halign="left">生命值</th>
    </tr>
    <tr>
      <th></th>
      <th>类型2</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
    </tr>
    <tr>
      <th>组</th>
      <th>类型1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>A</th>
      <th>Grass</th>
      <td>0</td>
      <td>0</td>
      <td>64.333333</td>
      <td>0</td>
      <td>0.0</td>
      <td>61.666667</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>Fire</th>
      <td>0</td>
      <td>58</td>
      <td>0.000000</td>
      <td>0</td>
      <td>48.5</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0</td>
      <td>0</td>
      <td>100.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>80.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">C</th>
      <th>Fire</th>
      <td>130</td>
      <td>94</td>
      <td>0.000000</td>
      <td>78</td>
      <td>78.0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0</td>
      <td>48</td>
      <td>0.000000</td>
      <td>0</td>
      <td>44.0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#将组和类型1放在行上，类型2放在列上，计算生命值和攻击力的均值和计数，将缺失值填充为0，并且增加总计行列
df_p.pivot_table(index=["组","类型1"],columns="类型2",values=["生命值","攻击力"],aggfunc=[np.mean,len],fill_value=0,margins=True)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="8" halign="left">mean</th>
      <th colspan="8" halign="left">len</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th colspan="4" halign="left">攻击力</th>
      <th colspan="4" halign="left">生命值</th>
      <th colspan="4" halign="left">攻击力</th>
      <th colspan="4" halign="left">生命值</th>
    </tr>
    <tr>
      <th></th>
      <th>类型2</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>All</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>All</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>All</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>All</th>
    </tr>
    <tr>
      <th>组</th>
      <th>类型1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>A</th>
      <th>Grass</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>64.333333</td>
      <td>64.333333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>61.666667</td>
      <td>61.666667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>Fire</th>
      <td>0.0</td>
      <td>58.0</td>
      <td>0.000000</td>
      <td>58.000000</td>
      <td>0.0</td>
      <td>48.5</td>
      <td>0.000000</td>
      <td>48.500000</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>80.000000</td>
      <td>80.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">C</th>
      <th>Fire</th>
      <td>130.0</td>
      <td>94.0</td>
      <td>0.000000</td>
      <td>106.000000</td>
      <td>78.0</td>
      <td>78.0</td>
      <td>0.000000</td>
      <td>78.000000</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0.0</td>
      <td>48.0</td>
      <td>0.000000</td>
      <td>48.000000</td>
      <td>0.0</td>
      <td>44.0</td>
      <td>0.000000</td>
      <td>44.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>All</th>
      <th></th>
      <td>130.0</td>
      <td>70.4</td>
      <td>73.250000</td>
      <td>77.500000</td>
      <td>78.0</td>
      <td>59.4</td>
      <td>66.250000</td>
      <td>64.000000</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2.重塑层次化索引

+ stack（）：将数据最内层的列旋转到行上

+ unstack（）：将数据最内层的行旋转到列上


```python
#示例数据
df_p1
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
    <tr>
      <th></th>
      <th></th>
      <th colspan="6" halign="left">mean</th>
      <th colspan="6" halign="left">len</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">攻击力</th>
      <th colspan="3" halign="left">生命值</th>
      <th colspan="3" halign="left">攻击力</th>
      <th colspan="3" halign="left">生命值</th>
    </tr>
    <tr>
      <th></th>
      <th>类型2</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
    </tr>
    <tr>
      <th>组</th>
      <th>类型1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>A</th>
      <th>Grass</th>
      <td>0</td>
      <td>0</td>
      <td>64.333333</td>
      <td>0</td>
      <td>0.0</td>
      <td>61.666667</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>Fire</th>
      <td>0</td>
      <td>58</td>
      <td>0.000000</td>
      <td>0</td>
      <td>48.5</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0</td>
      <td>0</td>
      <td>100.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>80.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">C</th>
      <th>Fire</th>
      <td>130</td>
      <td>94</td>
      <td>0.000000</td>
      <td>78</td>
      <td>78.0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0</td>
      <td>48</td>
      <td>0.000000</td>
      <td>0</td>
      <td>44.0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#将数据最内层的列旋转到行上，也即是将类型2转移到行上
df_p1.stack()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">mean</th>
      <th colspan="2" halign="left">len</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th>攻击力</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>生命值</th>
    </tr>
    <tr>
      <th>组</th>
      <th>类型1</th>
      <th>类型2</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">A</th>
      <th rowspan="3" valign="top">Grass</th>
      <th>Dragon</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>64.333333</td>
      <td>61.666667</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">B</th>
      <th rowspan="3" valign="top">Fire</th>
      <th>Dragon</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>58.000000</td>
      <td>48.500000</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Grass</th>
      <th>Dragon</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>100.000000</td>
      <td>80.000000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">C</th>
      <th rowspan="3" valign="top">Fire</th>
      <th>Dragon</th>
      <td>130.000000</td>
      <td>78.000000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>94.000000</td>
      <td>78.000000</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Water</th>
      <th>Dragon</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>48.000000</td>
      <td>44.000000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_p1
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="6" halign="left">mean</th>
      <th colspan="6" halign="left">len</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">攻击力</th>
      <th colspan="3" halign="left">生命值</th>
      <th colspan="3" halign="left">攻击力</th>
      <th colspan="3" halign="left">生命值</th>
    </tr>
    <tr>
      <th></th>
      <th>类型2</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
      <th>Dragon</th>
      <th>Flying</th>
      <th>Poison</th>
    </tr>
    <tr>
      <th>组</th>
      <th>类型1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>A</th>
      <th>Grass</th>
      <td>0</td>
      <td>0</td>
      <td>64.333333</td>
      <td>0</td>
      <td>0.0</td>
      <td>61.666667</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>Fire</th>
      <td>0</td>
      <td>58</td>
      <td>0.000000</td>
      <td>0</td>
      <td>48.5</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0</td>
      <td>0</td>
      <td>100.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>80.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">C</th>
      <th>Fire</th>
      <td>130</td>
      <td>94</td>
      <td>0.000000</td>
      <td>78</td>
      <td>78.0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>0</td>
      <td>48</td>
      <td>0.000000</td>
      <td>0</td>
      <td>44.0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#将数据最内层的行旋转到列上，也即是将类型1转移到列上
df_p1.unstack()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">mean</th>
      <th>...</th>
      <th colspan="10" halign="left">len</th>
    </tr>
    <tr>
      <th></th>
      <th colspan="9" halign="left">攻击力</th>
      <th>生命值</th>
      <th>...</th>
      <th>攻击力</th>
      <th colspan="9" halign="left">生命值</th>
    </tr>
    <tr>
      <th>类型2</th>
      <th colspan="3" halign="left">Dragon</th>
      <th colspan="3" halign="left">Flying</th>
      <th colspan="3" halign="left">Poison</th>
      <th>Dragon</th>
      <th>...</th>
      <th>Poison</th>
      <th colspan="3" halign="left">Dragon</th>
      <th colspan="3" halign="left">Flying</th>
      <th colspan="3" halign="left">Poison</th>
    </tr>
    <tr>
      <th>类型1</th>
      <th>Fire</th>
      <th>Grass</th>
      <th>Water</th>
      <th>Fire</th>
      <th>Grass</th>
      <th>Water</th>
      <th>Fire</th>
      <th>Grass</th>
      <th>Water</th>
      <th>Fire</th>
      <th>...</th>
      <th>Water</th>
      <th>Fire</th>
      <th>Grass</th>
      <th>Water</th>
      <th>Fire</th>
      <th>Grass</th>
      <th>Water</th>
      <th>Fire</th>
      <th>Grass</th>
      <th>Water</th>
    </tr>
    <tr>
      <th>组</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>A</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>64.333333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>58.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>100.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>C</th>
      <td>130.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>94.0</td>
      <td>NaN</td>
      <td>48.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>78.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 36 columns</p>
</div>



## 三.交叉表

用于计算分组频率用的特殊透视表


```python
#示例数据
df_p
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>组</th>
      <th>类型1</th>
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>5</th>
      <td>B</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>C</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
    </tr>
    <tr>
      <th>7</th>
      <td>C</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
    </tr>
    <tr>
      <th>8</th>
      <td>C</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
    </tr>
    <tr>
      <th>9</th>
      <td>C</td>
      <td>Water</td>
      <td>Flying</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
    </tr>
  </tbody>
</table>
</div>




```python
#计算组和类型1的交叉频率
pd.crosstab(index=df_p["组"],columns=df_p["类型1"])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>类型1</th>
      <th>Fire</th>
      <th>Grass</th>
      <th>Water</th>
    </tr>
    <tr>
      <th>组</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>B</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


