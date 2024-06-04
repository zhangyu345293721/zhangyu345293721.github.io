---
title: 'DataFrame—数据创建与导入'
date: 2022-04-01
permalink: /posts/DataFrame数据创建与导入/
tags:
  - cool posts
  - category1
  - category2
---

### 导入pandas和numpy库


```python
import pandas as pd
import numpy as np
```

## 一.获得DataFrame的两种方式

### 1.自己创建DataFrame

（1）通过字典创建DataFrame

+ 通过单层字典创建


```python
df1 = pd.DataFrame({"a":range(100,111),"b":range(200,211)})
df1
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>101</td>
      <td>201</td>
    </tr>
    <tr>
      <th>2</th>
      <td>102</td>
      <td>202</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103</td>
      <td>203</td>
    </tr>
    <tr>
      <th>4</th>
      <td>104</td>
      <td>204</td>
    </tr>
    <tr>
      <th>5</th>
      <td>105</td>
      <td>205</td>
    </tr>
    <tr>
      <th>6</th>
      <td>106</td>
      <td>206</td>
    </tr>
    <tr>
      <th>7</th>
      <td>107</td>
      <td>207</td>
    </tr>
    <tr>
      <th>8</th>
      <td>108</td>
      <td>208</td>
    </tr>
    <tr>
      <th>9</th>
      <td>109</td>
      <td>209</td>
    </tr>
    <tr>
      <th>10</th>
      <td>110</td>
      <td>210</td>
    </tr>
  </tbody>
</table>
</div>



+ 通过嵌套字典创建DataFrame


```python
dict_1 = {"b":{2015:3,2017:7,2016:4},"a":{2015:10,2017:27,2016:20}}
df2 = pd.DataFrame(dict_1)
df2
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015</th>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>20</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>27</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



显式地去指定索引和列名的顺序


```python
dict_1 = {"a":{2016:4,2015:3,2017:7},"b":{2017:27,2016:20,2015:10}}
df2 = pd.DataFrame(dict_1,index=[2017,2015,2016],columns=["b","a"])
df2
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017</th>
      <td>27</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>20</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



（2）通过数组创建DataFrame


```python
df3 = pd.DataFrame(np.arange(12).reshape(3,4),index=[2015,2016,2017],columns=["产品1","产品2","产品3","产品4"])
df3
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>产品1</th>
      <th>产品2</th>
      <th>产品3</th>
      <th>产品4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



### 2.从外部导入DataFrame

导入csv文件

如果导入无中文的文件,输入文件名称即可


```python
cd C:\\Users\\Alienware
```

    C:\Users\Alienware



```python
df1 = pd.read_csv("1.csv")
df1
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>class</th>
      <th>grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>2</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>1</td>
      <td>85</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>3</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>2</td>
      <td>60</td>
    </tr>
    <tr>
      <th>5</th>
      <td>f</td>
      <td>1</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>



如果导入含中文的csv文件，windows系统下需要添加参数encoding="gbk"


```python
df2 = pd.read_csv("pokemon_data.csv",encoding="gbk")
df2
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



## 二.导出csv文件


```python
df2.to_csv("1111.csv")
```
