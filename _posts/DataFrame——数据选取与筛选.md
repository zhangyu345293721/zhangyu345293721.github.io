```python
import pandas as pd
import numpy as np
```


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



## 一.数据选取

### Q1：选取第1行的数据（选取单行数据）

#### （1）按索引标签选取（loc做法）


```python
df.loc[0]   #返回的是Series
```




    姓名     Bulbasaur
    类型1        Grass
    类型2       Poison
    总计           318
    生命值           45
    攻击力           49
    防御力           49
    速度            45
    时代             1
    Name: 0, dtype: object




```python
df.loc[[0]]  #如果在里面多加一个方括号，那么返回的是DataFrame
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
  </tbody>
</table>
</div>



#### （2）按索引位置选取（iloc做法）


```python
df.iloc[0]      #返回的是Series
```




    姓名     Bulbasaur
    类型1        Grass
    类型2       Poison
    总计           318
    生命值           45
    攻击力           49
    防御力           49
    速度            45
    时代             1
    Name: 0, dtype: object




```python
df.iloc[[0]]  #如果在里面多加一个方括号，那么返回的是DataFrame
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
  </tbody>
</table>
</div>



相信你一定会很好奇：为什么在这里loc和iloc得到的结果是一样的?

下面会来解释

### Q2:选取第2到第5行的数据（选取连续行的数据）

#### （1）按索引标签选取（loc做法）


```python
df.loc[1:4]
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



你可能产生了一个疑问：不是说切片的末端是取不到的吗，也就是4这个索引所指向的第5行应该是取不到的

这是因为loc是按照索引标签来选取数据的，而不是根据位置来选取，举个例子：


```python
#以姓名这一列作为索引列
df_name = df.set_index("姓名")
df_name
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
    <tr>
      <th>姓名</th>
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
      <th>Bulbasaur</th>
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
      <th>Ivysaur</th>
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
      <th>Venusaur</th>
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
      <th>VenusaurMega Venusaur</th>
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
      <th>Charmander</th>
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
      <th>Charmeleon</th>
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
      <th>Charizard</th>
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
      <th>CharizardMega Charizard X</th>
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
      <th>CharizardMega Charizard Y</th>
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
      <th>Squirtle</th>
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
      <th>Wartortle</th>
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
      <th>Blastoise</th>
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
      <th>BlastoiseMega Blastoise</th>
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
      <th>Caterpie</th>
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
      <th>Metapod</th>
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
      <th>Butterfree</th>
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
      <th>Weedle</th>
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
      <th>Kakuna</th>
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
      <th>Beedrill</th>
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
      <th>BeedrillMega Beedrill</th>
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
      <th>Pidgey</th>
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
      <th>Pidgeotto</th>
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
      <th>Pidgeot</th>
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
      <th>PidgeotMega Pidgeot</th>
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
      <th>Rattata</th>
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
      <th>Raticate</th>
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
      <th>Spearow</th>
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
      <th>Fearow</th>
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
      <th>Ekans</th>
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
      <th>Arbok</th>
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
    </tr>
    <tr>
      <th>Sylveon</th>
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
      <th>Hawlucha</th>
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
      <th>Dedenne</th>
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
      <th>Carbink</th>
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
      <th>Goomy</th>
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
      <th>Sliggoo</th>
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
      <th>Goodra</th>
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
      <th>Klefki</th>
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
      <th>Phantump</th>
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
      <th>Trevenant</th>
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
      <th>PumpkabooAverage Size</th>
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
      <th>PumpkabooSmall Size</th>
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
      <th>PumpkabooLarge Size</th>
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
      <th>PumpkabooSuper Size</th>
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
      <th>GourgeistAverage Size</th>
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
      <th>GourgeistSmall Size</th>
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
      <th>GourgeistLarge Size</th>
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
      <th>GourgeistSuper Size</th>
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
      <th>Bergmite</th>
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
      <th>Avalugg</th>
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
      <th>Noibat</th>
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
      <th>Noivern</th>
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
      <th>Xerneas</th>
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
      <th>Yveltal</th>
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
      <th>Zygarde50% Forme</th>
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
      <th>Diancie</th>
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
      <th>DiancieMega Diancie</th>
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
      <th>HoopaHoopa Confined</th>
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
      <th>HoopaHoopa Unbound</th>
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
      <th>Volcanion</th>
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
<p>800 rows × 8 columns</p>
</div>



如果我要返回第2行到第5行的数据，该怎么做呢？


```python
#如果按照刚刚的写法，就会出错
df_name.loc[1:4]
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-116-1c1177e6066d> in <module>()
          1 #如果按照刚刚的写法，就会出错
    ----> 2 df_name.loc[1:4]
    

    D:\anaconda\lib\site-packages\pandas\core\indexing.py in __getitem__(self, key)
       1326         else:
       1327             key = com._apply_if_callable(key, self.obj)
    -> 1328             return self._getitem_axis(key, axis=0)
       1329 
       1330     def _is_scalar_access(self, key):


    D:\anaconda\lib\site-packages\pandas\core\indexing.py in _getitem_axis(self, key, axis)
       1504         if isinstance(key, slice):
       1505             self._has_valid_type(key, axis)
    -> 1506             return self._get_slice_axis(key, axis=axis)
       1507         elif is_bool_indexer(key):
       1508             return self._getbool_axis(key, axis=axis)


    D:\anaconda\lib\site-packages\pandas\core\indexing.py in _get_slice_axis(self, slice_obj, axis)
       1354         labels = obj._get_axis(axis)
       1355         indexer = labels.slice_indexer(slice_obj.start, slice_obj.stop,
    -> 1356                                        slice_obj.step, kind=self.name)
       1357 
       1358         if isinstance(indexer, slice):


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in slice_indexer(self, start, end, step, kind)
       3299         """
       3300         start_slice, end_slice = self.slice_locs(start, end, step=step,
    -> 3301                                                  kind=kind)
       3302 
       3303         # return a slice


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in slice_locs(self, start, end, step, kind)
       3487         start_slice = None
       3488         if start is not None:
    -> 3489             start_slice = self.get_slice_bound(start, 'left', kind)
       3490         if start_slice is None:
       3491             start_slice = 0


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in get_slice_bound(self, label, side, kind)
       3426         # For datetime indices label may be a string that has to be converted
       3427         # to datetime boundary according to its resolution.
    -> 3428         label = self._maybe_cast_slice_bound(label, side, kind)
       3429 
       3430         # we need to look up the label


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in _maybe_cast_slice_bound(self, label, side, kind)
       3377         # this is rejected (generally .loc gets you here)
       3378         elif is_integer(label):
    -> 3379             self._invalid_indexer('slice', label)
       3380 
       3381         return label


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in _invalid_indexer(self, form, key)
       1467                         "indexers [{key}] of {kind}".format(
       1468                             form=form, klass=type(self), key=key,
    -> 1469                             kind=type(key)))
       1470 
       1471     def get_duplicates(self):


    TypeError: cannot do slice indexing on <class 'pandas.core.indexes.base.Index'> with these indexers [1] of <class 'int'>



```python
#因为loc是按照索引标签选取的，按照下面这种写法就对了
df_name.loc["Ivysaur":"Charmander"]
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
    <tr>
      <th>姓名</th>
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
      <th>Ivysaur</th>
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
      <th>Venusaur</th>
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
      <th>VenusaurMega Venusaur</th>
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
      <th>Charmander</th>
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



所以说，之前写的df.loc[1:4]能返回第2到第5行的数据，只是恰好因为索引号是默认生成的数字索引，1对应的就是第2行的索引，4对应的是第5行的索引，1:4代表的是从第2行到第5行的索引标签，本质是和现在的"Ivysaur":"Charmander"一样的，都是代表索引标签，而不是位置。所以按照这种索引标签来选取数据的方法是能够取到末端的数据的.

#### （2）按索引位置选取（iloc做法）


```python
df.iloc[1:5]
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



我们需要返回的是第2行到第5行，因此对应的索引位置是1:4，但是由于iloc是按照位置来选取数据的，因此末端索引是取不到的，那么末端就需要再加1，这样就能确保第5行能取到了，而取不到第6行

**为了能更直观地体现出loc和iloc的区别，接下来以df_name为示例数据**


```python
#示例数据
df_name.head(10)
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
    <tr>
      <th>姓名</th>
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
      <th>Bulbasaur</th>
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
      <th>Ivysaur</th>
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
      <th>Venusaur</th>
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
      <th>VenusaurMega Venusaur</th>
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
      <th>Charmander</th>
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
      <th>Charmeleon</th>
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
      <th>Charizard</th>
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
      <th>CharizardMega Charizard X</th>
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
      <th>CharizardMega Charizard Y</th>
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
      <th>Squirtle</th>
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



### Q3:选取第2行，第4行，第7行，第10行的数据（选取特定行的数据）

#### （1）按索引标签选取（loc做法）


```python
df_name.loc[["Ivysaur","VenusaurMega Venusaur","Charizard","Squirtle"]]
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
    <tr>
      <th>姓名</th>
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
      <th>Ivysaur</th>
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
      <th>VenusaurMega Venusaur</th>
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
      <th>Charizard</th>
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
      <th>Squirtle</th>
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



#### （2）按索引位置选取（iloc做法）


```python
df_name.iloc[[1,3,6,9]]
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
    <tr>
      <th>姓名</th>
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
      <th>Ivysaur</th>
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
      <th>VenusaurMega Venusaur</th>
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
      <th>Charizard</th>
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
      <th>Squirtle</th>
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



### Q4:选取攻击力列（选取单列的数据）

#### （1）直接方括号+列名


```python
#直接方括号输入列名即可，推荐这种方法
df_name["攻击力"]
#返回的是一个Series
```




    姓名
    Bulbasaur                     49
    Ivysaur                       62
    Venusaur                      82
    VenusaurMega Venusaur        100
    Charmander                    52
    Charmeleon                    64
    Charizard                     84
    CharizardMega Charizard X    130
    CharizardMega Charizard Y    104
    Squirtle                      48
    Wartortle                     63
    Blastoise                     83
    BlastoiseMega Blastoise      103
    Caterpie                      30
    Metapod                       20
    Butterfree                    45
    Weedle                        35
    Kakuna                        25
    Beedrill                      90
    BeedrillMega Beedrill        150
    Pidgey                        45
    Pidgeotto                     60
    Pidgeot                       80
    PidgeotMega Pidgeot           80
    Rattata                       56
    Raticate                      81
    Spearow                       60
    Fearow                        90
    Ekans                         60
    Arbok                         85
                                ... 
    Sylveon                       65
    Hawlucha                      92
    Dedenne                       58
    Carbink                       50
    Goomy                         50
    Sliggoo                       75
    Goodra                       100
    Klefki                        80
    Phantump                      70
    Trevenant                    110
    PumpkabooAverage Size         66
    PumpkabooSmall Size           66
    PumpkabooLarge Size           66
    PumpkabooSuper Size           66
    GourgeistAverage Size         90
    GourgeistSmall Size           85
    GourgeistLarge Size           95
    GourgeistSuper Size          100
    Bergmite                      69
    Avalugg                      117
    Noibat                        30
    Noivern                       70
    Xerneas                      131
    Yveltal                      131
    Zygarde50% Forme             100
    Diancie                      100
    DiancieMega Diancie          160
    HoopaHoopa Confined          110
    HoopaHoopa Unbound           160
    Volcanion                    110
    Name: 攻击力, Length: 800, dtype: int64




```python
df_name[["攻击力"]]
#返回的是一个DataFrame
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
      <th>攻击力</th>
    </tr>
    <tr>
      <th>姓名</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bulbasaur</th>
      <td>49</td>
    </tr>
    <tr>
      <th>Ivysaur</th>
      <td>62</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>82</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>100</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>52</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>64</td>
    </tr>
    <tr>
      <th>Charizard</th>
      <td>84</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard X</th>
      <td>130</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard Y</th>
      <td>104</td>
    </tr>
    <tr>
      <th>Squirtle</th>
      <td>48</td>
    </tr>
    <tr>
      <th>Wartortle</th>
      <td>63</td>
    </tr>
    <tr>
      <th>Blastoise</th>
      <td>83</td>
    </tr>
    <tr>
      <th>BlastoiseMega Blastoise</th>
      <td>103</td>
    </tr>
    <tr>
      <th>Caterpie</th>
      <td>30</td>
    </tr>
    <tr>
      <th>Metapod</th>
      <td>20</td>
    </tr>
    <tr>
      <th>Butterfree</th>
      <td>45</td>
    </tr>
    <tr>
      <th>Weedle</th>
      <td>35</td>
    </tr>
    <tr>
      <th>Kakuna</th>
      <td>25</td>
    </tr>
    <tr>
      <th>Beedrill</th>
      <td>90</td>
    </tr>
    <tr>
      <th>BeedrillMega Beedrill</th>
      <td>150</td>
    </tr>
    <tr>
      <th>Pidgey</th>
      <td>45</td>
    </tr>
    <tr>
      <th>Pidgeotto</th>
      <td>60</td>
    </tr>
    <tr>
      <th>Pidgeot</th>
      <td>80</td>
    </tr>
    <tr>
      <th>PidgeotMega Pidgeot</th>
      <td>80</td>
    </tr>
    <tr>
      <th>Rattata</th>
      <td>56</td>
    </tr>
    <tr>
      <th>Raticate</th>
      <td>81</td>
    </tr>
    <tr>
      <th>Spearow</th>
      <td>60</td>
    </tr>
    <tr>
      <th>Fearow</th>
      <td>90</td>
    </tr>
    <tr>
      <th>Ekans</th>
      <td>60</td>
    </tr>
    <tr>
      <th>Arbok</th>
      <td>85</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>Sylveon</th>
      <td>65</td>
    </tr>
    <tr>
      <th>Hawlucha</th>
      <td>92</td>
    </tr>
    <tr>
      <th>Dedenne</th>
      <td>58</td>
    </tr>
    <tr>
      <th>Carbink</th>
      <td>50</td>
    </tr>
    <tr>
      <th>Goomy</th>
      <td>50</td>
    </tr>
    <tr>
      <th>Sliggoo</th>
      <td>75</td>
    </tr>
    <tr>
      <th>Goodra</th>
      <td>100</td>
    </tr>
    <tr>
      <th>Klefki</th>
      <td>80</td>
    </tr>
    <tr>
      <th>Phantump</th>
      <td>70</td>
    </tr>
    <tr>
      <th>Trevenant</th>
      <td>110</td>
    </tr>
    <tr>
      <th>PumpkabooAverage Size</th>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooSmall Size</th>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooLarge Size</th>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooSuper Size</th>
      <td>66</td>
    </tr>
    <tr>
      <th>GourgeistAverage Size</th>
      <td>90</td>
    </tr>
    <tr>
      <th>GourgeistSmall Size</th>
      <td>85</td>
    </tr>
    <tr>
      <th>GourgeistLarge Size</th>
      <td>95</td>
    </tr>
    <tr>
      <th>GourgeistSuper Size</th>
      <td>100</td>
    </tr>
    <tr>
      <th>Bergmite</th>
      <td>69</td>
    </tr>
    <tr>
      <th>Avalugg</th>
      <td>117</td>
    </tr>
    <tr>
      <th>Noibat</th>
      <td>30</td>
    </tr>
    <tr>
      <th>Noivern</th>
      <td>70</td>
    </tr>
    <tr>
      <th>Xerneas</th>
      <td>131</td>
    </tr>
    <tr>
      <th>Yveltal</th>
      <td>131</td>
    </tr>
    <tr>
      <th>Zygarde50% Forme</th>
      <td>100</td>
    </tr>
    <tr>
      <th>Diancie</th>
      <td>100</td>
    </tr>
    <tr>
      <th>DiancieMega Diancie</th>
      <td>160</td>
    </tr>
    <tr>
      <th>HoopaHoopa Confined</th>
      <td>110</td>
    </tr>
    <tr>
      <th>HoopaHoopa Unbound</th>
      <td>160</td>
    </tr>
    <tr>
      <th>Volcanion</th>
      <td>110</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 1 columns</p>
</div>



#### （2）按索引标签选取（loc做法）


```python
#虽然用loc也能提取单列，但是显得不够简洁
df_name.loc[:,["攻击力"]]
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
      <th>攻击力</th>
    </tr>
    <tr>
      <th>姓名</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bulbasaur</th>
      <td>49</td>
    </tr>
    <tr>
      <th>Ivysaur</th>
      <td>62</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>82</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>100</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>52</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>64</td>
    </tr>
    <tr>
      <th>Charizard</th>
      <td>84</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard X</th>
      <td>130</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard Y</th>
      <td>104</td>
    </tr>
    <tr>
      <th>Squirtle</th>
      <td>48</td>
    </tr>
    <tr>
      <th>Wartortle</th>
      <td>63</td>
    </tr>
    <tr>
      <th>Blastoise</th>
      <td>83</td>
    </tr>
    <tr>
      <th>BlastoiseMega Blastoise</th>
      <td>103</td>
    </tr>
    <tr>
      <th>Caterpie</th>
      <td>30</td>
    </tr>
    <tr>
      <th>Metapod</th>
      <td>20</td>
    </tr>
    <tr>
      <th>Butterfree</th>
      <td>45</td>
    </tr>
    <tr>
      <th>Weedle</th>
      <td>35</td>
    </tr>
    <tr>
      <th>Kakuna</th>
      <td>25</td>
    </tr>
    <tr>
      <th>Beedrill</th>
      <td>90</td>
    </tr>
    <tr>
      <th>BeedrillMega Beedrill</th>
      <td>150</td>
    </tr>
    <tr>
      <th>Pidgey</th>
      <td>45</td>
    </tr>
    <tr>
      <th>Pidgeotto</th>
      <td>60</td>
    </tr>
    <tr>
      <th>Pidgeot</th>
      <td>80</td>
    </tr>
    <tr>
      <th>PidgeotMega Pidgeot</th>
      <td>80</td>
    </tr>
    <tr>
      <th>Rattata</th>
      <td>56</td>
    </tr>
    <tr>
      <th>Raticate</th>
      <td>81</td>
    </tr>
    <tr>
      <th>Spearow</th>
      <td>60</td>
    </tr>
    <tr>
      <th>Fearow</th>
      <td>90</td>
    </tr>
    <tr>
      <th>Ekans</th>
      <td>60</td>
    </tr>
    <tr>
      <th>Arbok</th>
      <td>85</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>Sylveon</th>
      <td>65</td>
    </tr>
    <tr>
      <th>Hawlucha</th>
      <td>92</td>
    </tr>
    <tr>
      <th>Dedenne</th>
      <td>58</td>
    </tr>
    <tr>
      <th>Carbink</th>
      <td>50</td>
    </tr>
    <tr>
      <th>Goomy</th>
      <td>50</td>
    </tr>
    <tr>
      <th>Sliggoo</th>
      <td>75</td>
    </tr>
    <tr>
      <th>Goodra</th>
      <td>100</td>
    </tr>
    <tr>
      <th>Klefki</th>
      <td>80</td>
    </tr>
    <tr>
      <th>Phantump</th>
      <td>70</td>
    </tr>
    <tr>
      <th>Trevenant</th>
      <td>110</td>
    </tr>
    <tr>
      <th>PumpkabooAverage Size</th>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooSmall Size</th>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooLarge Size</th>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooSuper Size</th>
      <td>66</td>
    </tr>
    <tr>
      <th>GourgeistAverage Size</th>
      <td>90</td>
    </tr>
    <tr>
      <th>GourgeistSmall Size</th>
      <td>85</td>
    </tr>
    <tr>
      <th>GourgeistLarge Size</th>
      <td>95</td>
    </tr>
    <tr>
      <th>GourgeistSuper Size</th>
      <td>100</td>
    </tr>
    <tr>
      <th>Bergmite</th>
      <td>69</td>
    </tr>
    <tr>
      <th>Avalugg</th>
      <td>117</td>
    </tr>
    <tr>
      <th>Noibat</th>
      <td>30</td>
    </tr>
    <tr>
      <th>Noivern</th>
      <td>70</td>
    </tr>
    <tr>
      <th>Xerneas</th>
      <td>131</td>
    </tr>
    <tr>
      <th>Yveltal</th>
      <td>131</td>
    </tr>
    <tr>
      <th>Zygarde50% Forme</th>
      <td>100</td>
    </tr>
    <tr>
      <th>Diancie</th>
      <td>100</td>
    </tr>
    <tr>
      <th>DiancieMega Diancie</th>
      <td>160</td>
    </tr>
    <tr>
      <th>HoopaHoopa Confined</th>
      <td>110</td>
    </tr>
    <tr>
      <th>HoopaHoopa Unbound</th>
      <td>160</td>
    </tr>
    <tr>
      <th>Volcanion</th>
      <td>110</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 1 columns</p>
</div>



#### （3）按索引位置选取（iloc做法）


```python
df_name.iloc[:,[4]]
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
      <th>攻击力</th>
    </tr>
    <tr>
      <th>姓名</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bulbasaur</th>
      <td>49</td>
    </tr>
    <tr>
      <th>Ivysaur</th>
      <td>62</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>82</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>100</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>52</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>64</td>
    </tr>
    <tr>
      <th>Charizard</th>
      <td>84</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard X</th>
      <td>130</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard Y</th>
      <td>104</td>
    </tr>
    <tr>
      <th>Squirtle</th>
      <td>48</td>
    </tr>
    <tr>
      <th>Wartortle</th>
      <td>63</td>
    </tr>
    <tr>
      <th>Blastoise</th>
      <td>83</td>
    </tr>
    <tr>
      <th>BlastoiseMega Blastoise</th>
      <td>103</td>
    </tr>
    <tr>
      <th>Caterpie</th>
      <td>30</td>
    </tr>
    <tr>
      <th>Metapod</th>
      <td>20</td>
    </tr>
    <tr>
      <th>Butterfree</th>
      <td>45</td>
    </tr>
    <tr>
      <th>Weedle</th>
      <td>35</td>
    </tr>
    <tr>
      <th>Kakuna</th>
      <td>25</td>
    </tr>
    <tr>
      <th>Beedrill</th>
      <td>90</td>
    </tr>
    <tr>
      <th>BeedrillMega Beedrill</th>
      <td>150</td>
    </tr>
    <tr>
      <th>Pidgey</th>
      <td>45</td>
    </tr>
    <tr>
      <th>Pidgeotto</th>
      <td>60</td>
    </tr>
    <tr>
      <th>Pidgeot</th>
      <td>80</td>
    </tr>
    <tr>
      <th>PidgeotMega Pidgeot</th>
      <td>80</td>
    </tr>
    <tr>
      <th>Rattata</th>
      <td>56</td>
    </tr>
    <tr>
      <th>Raticate</th>
      <td>81</td>
    </tr>
    <tr>
      <th>Spearow</th>
      <td>60</td>
    </tr>
    <tr>
      <th>Fearow</th>
      <td>90</td>
    </tr>
    <tr>
      <th>Ekans</th>
      <td>60</td>
    </tr>
    <tr>
      <th>Arbok</th>
      <td>85</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>Sylveon</th>
      <td>65</td>
    </tr>
    <tr>
      <th>Hawlucha</th>
      <td>92</td>
    </tr>
    <tr>
      <th>Dedenne</th>
      <td>58</td>
    </tr>
    <tr>
      <th>Carbink</th>
      <td>50</td>
    </tr>
    <tr>
      <th>Goomy</th>
      <td>50</td>
    </tr>
    <tr>
      <th>Sliggoo</th>
      <td>75</td>
    </tr>
    <tr>
      <th>Goodra</th>
      <td>100</td>
    </tr>
    <tr>
      <th>Klefki</th>
      <td>80</td>
    </tr>
    <tr>
      <th>Phantump</th>
      <td>70</td>
    </tr>
    <tr>
      <th>Trevenant</th>
      <td>110</td>
    </tr>
    <tr>
      <th>PumpkabooAverage Size</th>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooSmall Size</th>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooLarge Size</th>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooSuper Size</th>
      <td>66</td>
    </tr>
    <tr>
      <th>GourgeistAverage Size</th>
      <td>90</td>
    </tr>
    <tr>
      <th>GourgeistSmall Size</th>
      <td>85</td>
    </tr>
    <tr>
      <th>GourgeistLarge Size</th>
      <td>95</td>
    </tr>
    <tr>
      <th>GourgeistSuper Size</th>
      <td>100</td>
    </tr>
    <tr>
      <th>Bergmite</th>
      <td>69</td>
    </tr>
    <tr>
      <th>Avalugg</th>
      <td>117</td>
    </tr>
    <tr>
      <th>Noibat</th>
      <td>30</td>
    </tr>
    <tr>
      <th>Noivern</th>
      <td>70</td>
    </tr>
    <tr>
      <th>Xerneas</th>
      <td>131</td>
    </tr>
    <tr>
      <th>Yveltal</th>
      <td>131</td>
    </tr>
    <tr>
      <th>Zygarde50% Forme</th>
      <td>100</td>
    </tr>
    <tr>
      <th>Diancie</th>
      <td>100</td>
    </tr>
    <tr>
      <th>DiancieMega Diancie</th>
      <td>160</td>
    </tr>
    <tr>
      <th>HoopaHoopa Confined</th>
      <td>110</td>
    </tr>
    <tr>
      <th>HoopaHoopa Unbound</th>
      <td>160</td>
    </tr>
    <tr>
      <th>Volcanion</th>
      <td>110</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 1 columns</p>
</div>



#### （4）点号选取法


```python
#也可以通过点号选取列
df_name.攻击力
```




    姓名
    Bulbasaur                     49
    Ivysaur                       62
    Venusaur                      82
    VenusaurMega Venusaur        100
    Charmander                    52
    Charmeleon                    64
    Charizard                     84
    CharizardMega Charizard X    130
    CharizardMega Charizard Y    104
    Squirtle                      48
    Wartortle                     63
    Blastoise                     83
    BlastoiseMega Blastoise      103
    Caterpie                      30
    Metapod                       20
    Butterfree                    45
    Weedle                        35
    Kakuna                        25
    Beedrill                      90
    BeedrillMega Beedrill        150
    Pidgey                        45
    Pidgeotto                     60
    Pidgeot                       80
    PidgeotMega Pidgeot           80
    Rattata                       56
    Raticate                      81
    Spearow                       60
    Fearow                        90
    Ekans                         60
    Arbok                         85
                                ... 
    Sylveon                       65
    Hawlucha                      92
    Dedenne                       58
    Carbink                       50
    Goomy                         50
    Sliggoo                       75
    Goodra                       100
    Klefki                        80
    Phantump                      70
    Trevenant                    110
    PumpkabooAverage Size         66
    PumpkabooSmall Size           66
    PumpkabooLarge Size           66
    PumpkabooSuper Size           66
    GourgeistAverage Size         90
    GourgeistSmall Size           85
    GourgeistLarge Size           95
    GourgeistSuper Size          100
    Bergmite                      69
    Avalugg                      117
    Noibat                        30
    Noivern                       70
    Xerneas                      131
    Yveltal                      131
    Zygarde50% Forme             100
    Diancie                      100
    DiancieMega Diancie          160
    HoopaHoopa Confined          110
    HoopaHoopa Unbound           160
    Volcanion                    110
    Name: 攻击力, Length: 800, dtype: int64



点号提取列的这种方法的优点是：写法比较简洁快速，缺点是如果列名和关键字重复了就无法提取了，因为点号调用的是对象，python无法判断出名字一样的列名和关键字


```python
#新增一列class，值为1
df_name["class"] = 1
df_name
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
      <th>class</th>
    </tr>
    <tr>
      <th>姓名</th>
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
      <th>Bulbasaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Ivysaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Charizard</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard X</th>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard Y</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Squirtle</th>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Wartortle</th>
      <td>Water</td>
      <td>NaN</td>
      <td>405</td>
      <td>59</td>
      <td>63</td>
      <td>80</td>
      <td>58</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Blastoise</th>
      <td>Water</td>
      <td>NaN</td>
      <td>530</td>
      <td>79</td>
      <td>83</td>
      <td>100</td>
      <td>78</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>BlastoiseMega Blastoise</th>
      <td>Water</td>
      <td>NaN</td>
      <td>630</td>
      <td>79</td>
      <td>103</td>
      <td>120</td>
      <td>78</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Caterpie</th>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Metapod</th>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Butterfree</th>
      <td>Bug</td>
      <td>Flying</td>
      <td>395</td>
      <td>60</td>
      <td>45</td>
      <td>50</td>
      <td>70</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Weedle</th>
      <td>Bug</td>
      <td>Poison</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Kakuna</th>
      <td>Bug</td>
      <td>Poison</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Beedrill</th>
      <td>Bug</td>
      <td>Poison</td>
      <td>395</td>
      <td>65</td>
      <td>90</td>
      <td>40</td>
      <td>75</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>BeedrillMega Beedrill</th>
      <td>Bug</td>
      <td>Poison</td>
      <td>495</td>
      <td>65</td>
      <td>150</td>
      <td>40</td>
      <td>145</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Pidgey</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>251</td>
      <td>40</td>
      <td>45</td>
      <td>40</td>
      <td>56</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Pidgeotto</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>349</td>
      <td>63</td>
      <td>60</td>
      <td>55</td>
      <td>71</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Pidgeot</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>479</td>
      <td>83</td>
      <td>80</td>
      <td>75</td>
      <td>101</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>PidgeotMega Pidgeot</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>579</td>
      <td>83</td>
      <td>80</td>
      <td>80</td>
      <td>121</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Rattata</th>
      <td>Normal</td>
      <td>NaN</td>
      <td>253</td>
      <td>30</td>
      <td>56</td>
      <td>35</td>
      <td>72</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Raticate</th>
      <td>Normal</td>
      <td>NaN</td>
      <td>413</td>
      <td>55</td>
      <td>81</td>
      <td>60</td>
      <td>97</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Spearow</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>262</td>
      <td>40</td>
      <td>60</td>
      <td>30</td>
      <td>70</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fearow</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>442</td>
      <td>65</td>
      <td>90</td>
      <td>65</td>
      <td>100</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Ekans</th>
      <td>Poison</td>
      <td>NaN</td>
      <td>288</td>
      <td>35</td>
      <td>60</td>
      <td>44</td>
      <td>55</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Arbok</th>
      <td>Poison</td>
      <td>NaN</td>
      <td>438</td>
      <td>60</td>
      <td>85</td>
      <td>69</td>
      <td>80</td>
      <td>1</td>
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
      <th>Sylveon</th>
      <td>Fairy</td>
      <td>NaN</td>
      <td>525</td>
      <td>95</td>
      <td>65</td>
      <td>65</td>
      <td>60</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Hawlucha</th>
      <td>Fighting</td>
      <td>Flying</td>
      <td>500</td>
      <td>78</td>
      <td>92</td>
      <td>75</td>
      <td>118</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dedenne</th>
      <td>Electric</td>
      <td>Fairy</td>
      <td>431</td>
      <td>67</td>
      <td>58</td>
      <td>57</td>
      <td>101</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Carbink</th>
      <td>Rock</td>
      <td>Fairy</td>
      <td>500</td>
      <td>50</td>
      <td>50</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Goomy</th>
      <td>Dragon</td>
      <td>NaN</td>
      <td>300</td>
      <td>45</td>
      <td>50</td>
      <td>35</td>
      <td>40</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Sliggoo</th>
      <td>Dragon</td>
      <td>NaN</td>
      <td>452</td>
      <td>68</td>
      <td>75</td>
      <td>53</td>
      <td>60</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Goodra</th>
      <td>Dragon</td>
      <td>NaN</td>
      <td>600</td>
      <td>90</td>
      <td>100</td>
      <td>70</td>
      <td>80</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Klefki</th>
      <td>Steel</td>
      <td>Fairy</td>
      <td>470</td>
      <td>57</td>
      <td>80</td>
      <td>91</td>
      <td>75</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Phantump</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>309</td>
      <td>43</td>
      <td>70</td>
      <td>48</td>
      <td>38</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Trevenant</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>474</td>
      <td>85</td>
      <td>110</td>
      <td>76</td>
      <td>56</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>PumpkabooAverage Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>49</td>
      <td>66</td>
      <td>70</td>
      <td>51</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>PumpkabooSmall Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>44</td>
      <td>66</td>
      <td>70</td>
      <td>56</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>PumpkabooLarge Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>54</td>
      <td>66</td>
      <td>70</td>
      <td>46</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>PumpkabooSuper Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>59</td>
      <td>66</td>
      <td>70</td>
      <td>41</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>GourgeistAverage Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>65</td>
      <td>90</td>
      <td>122</td>
      <td>84</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>GourgeistSmall Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>55</td>
      <td>85</td>
      <td>122</td>
      <td>99</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>GourgeistLarge Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>75</td>
      <td>95</td>
      <td>122</td>
      <td>69</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>GourgeistSuper Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>85</td>
      <td>100</td>
      <td>122</td>
      <td>54</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bergmite</th>
      <td>Ice</td>
      <td>NaN</td>
      <td>304</td>
      <td>55</td>
      <td>69</td>
      <td>85</td>
      <td>28</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Avalugg</th>
      <td>Ice</td>
      <td>NaN</td>
      <td>514</td>
      <td>95</td>
      <td>117</td>
      <td>184</td>
      <td>28</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Noibat</th>
      <td>Flying</td>
      <td>Dragon</td>
      <td>245</td>
      <td>40</td>
      <td>30</td>
      <td>35</td>
      <td>55</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Noivern</th>
      <td>Flying</td>
      <td>Dragon</td>
      <td>535</td>
      <td>85</td>
      <td>70</td>
      <td>80</td>
      <td>123</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Xerneas</th>
      <td>Fairy</td>
      <td>NaN</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Yveltal</th>
      <td>Dark</td>
      <td>Flying</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Zygarde50% Forme</th>
      <td>Dragon</td>
      <td>Ground</td>
      <td>600</td>
      <td>108</td>
      <td>100</td>
      <td>121</td>
      <td>95</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Diancie</th>
      <td>Rock</td>
      <td>Fairy</td>
      <td>600</td>
      <td>50</td>
      <td>100</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>DiancieMega Diancie</th>
      <td>Rock</td>
      <td>Fairy</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>HoopaHoopa Confined</th>
      <td>Psychic</td>
      <td>Ghost</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
      <td>60</td>
      <td>70</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>HoopaHoopa Unbound</th>
      <td>Psychic</td>
      <td>Dark</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
      <td>60</td>
      <td>80</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Volcanion</th>
      <td>Fire</td>
      <td>Water</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
      <td>120</td>
      <td>70</td>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 9 columns</p>
</div>



由于class是python的关键字，而点号选取列实质上是在调用对象，本身列名class和关键字class重叠了，导致无法调用成功


```python
df_name.class
```


      File "<ipython-input-128-dfc874e7568d>", line 1
        df_name.class
                    ^
    SyntaxError: invalid syntax



### Q5:选取类型1列到攻击力列的所有数据（选取连续列的数据）

#### （1）按索引标签选取（loc做法）


```python
df_name.loc[:,"类型1":"攻击力"]
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
    </tr>
    <tr>
      <th>姓名</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bulbasaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
    </tr>
    <tr>
      <th>Ivysaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
    </tr>
    <tr>
      <th>Charizard</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard X</th>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard Y</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
    </tr>
    <tr>
      <th>Squirtle</th>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Wartortle</th>
      <td>Water</td>
      <td>NaN</td>
      <td>405</td>
      <td>59</td>
      <td>63</td>
    </tr>
    <tr>
      <th>Blastoise</th>
      <td>Water</td>
      <td>NaN</td>
      <td>530</td>
      <td>79</td>
      <td>83</td>
    </tr>
    <tr>
      <th>BlastoiseMega Blastoise</th>
      <td>Water</td>
      <td>NaN</td>
      <td>630</td>
      <td>79</td>
      <td>103</td>
    </tr>
    <tr>
      <th>Caterpie</th>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Metapod</th>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
    </tr>
    <tr>
      <th>Butterfree</th>
      <td>Bug</td>
      <td>Flying</td>
      <td>395</td>
      <td>60</td>
      <td>45</td>
    </tr>
    <tr>
      <th>Weedle</th>
      <td>Bug</td>
      <td>Poison</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Kakuna</th>
      <td>Bug</td>
      <td>Poison</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
    </tr>
    <tr>
      <th>Beedrill</th>
      <td>Bug</td>
      <td>Poison</td>
      <td>395</td>
      <td>65</td>
      <td>90</td>
    </tr>
    <tr>
      <th>BeedrillMega Beedrill</th>
      <td>Bug</td>
      <td>Poison</td>
      <td>495</td>
      <td>65</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Pidgey</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>251</td>
      <td>40</td>
      <td>45</td>
    </tr>
    <tr>
      <th>Pidgeotto</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>349</td>
      <td>63</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Pidgeot</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>479</td>
      <td>83</td>
      <td>80</td>
    </tr>
    <tr>
      <th>PidgeotMega Pidgeot</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>579</td>
      <td>83</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Rattata</th>
      <td>Normal</td>
      <td>NaN</td>
      <td>253</td>
      <td>30</td>
      <td>56</td>
    </tr>
    <tr>
      <th>Raticate</th>
      <td>Normal</td>
      <td>NaN</td>
      <td>413</td>
      <td>55</td>
      <td>81</td>
    </tr>
    <tr>
      <th>Spearow</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>262</td>
      <td>40</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Fearow</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>442</td>
      <td>65</td>
      <td>90</td>
    </tr>
    <tr>
      <th>Ekans</th>
      <td>Poison</td>
      <td>NaN</td>
      <td>288</td>
      <td>35</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Arbok</th>
      <td>Poison</td>
      <td>NaN</td>
      <td>438</td>
      <td>60</td>
      <td>85</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Sylveon</th>
      <td>Fairy</td>
      <td>NaN</td>
      <td>525</td>
      <td>95</td>
      <td>65</td>
    </tr>
    <tr>
      <th>Hawlucha</th>
      <td>Fighting</td>
      <td>Flying</td>
      <td>500</td>
      <td>78</td>
      <td>92</td>
    </tr>
    <tr>
      <th>Dedenne</th>
      <td>Electric</td>
      <td>Fairy</td>
      <td>431</td>
      <td>67</td>
      <td>58</td>
    </tr>
    <tr>
      <th>Carbink</th>
      <td>Rock</td>
      <td>Fairy</td>
      <td>500</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Goomy</th>
      <td>Dragon</td>
      <td>NaN</td>
      <td>300</td>
      <td>45</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Sliggoo</th>
      <td>Dragon</td>
      <td>NaN</td>
      <td>452</td>
      <td>68</td>
      <td>75</td>
    </tr>
    <tr>
      <th>Goodra</th>
      <td>Dragon</td>
      <td>NaN</td>
      <td>600</td>
      <td>90</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Klefki</th>
      <td>Steel</td>
      <td>Fairy</td>
      <td>470</td>
      <td>57</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Phantump</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>309</td>
      <td>43</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Trevenant</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>474</td>
      <td>85</td>
      <td>110</td>
    </tr>
    <tr>
      <th>PumpkabooAverage Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>49</td>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooSmall Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>44</td>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooLarge Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>54</td>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooSuper Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>59</td>
      <td>66</td>
    </tr>
    <tr>
      <th>GourgeistAverage Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>65</td>
      <td>90</td>
    </tr>
    <tr>
      <th>GourgeistSmall Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>55</td>
      <td>85</td>
    </tr>
    <tr>
      <th>GourgeistLarge Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>75</td>
      <td>95</td>
    </tr>
    <tr>
      <th>GourgeistSuper Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>85</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Bergmite</th>
      <td>Ice</td>
      <td>NaN</td>
      <td>304</td>
      <td>55</td>
      <td>69</td>
    </tr>
    <tr>
      <th>Avalugg</th>
      <td>Ice</td>
      <td>NaN</td>
      <td>514</td>
      <td>95</td>
      <td>117</td>
    </tr>
    <tr>
      <th>Noibat</th>
      <td>Flying</td>
      <td>Dragon</td>
      <td>245</td>
      <td>40</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Noivern</th>
      <td>Flying</td>
      <td>Dragon</td>
      <td>535</td>
      <td>85</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Xerneas</th>
      <td>Fairy</td>
      <td>NaN</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
    </tr>
    <tr>
      <th>Yveltal</th>
      <td>Dark</td>
      <td>Flying</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
    </tr>
    <tr>
      <th>Zygarde50% Forme</th>
      <td>Dragon</td>
      <td>Ground</td>
      <td>600</td>
      <td>108</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Diancie</th>
      <td>Rock</td>
      <td>Fairy</td>
      <td>600</td>
      <td>50</td>
      <td>100</td>
    </tr>
    <tr>
      <th>DiancieMega Diancie</th>
      <td>Rock</td>
      <td>Fairy</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
    </tr>
    <tr>
      <th>HoopaHoopa Confined</th>
      <td>Psychic</td>
      <td>Ghost</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
    </tr>
    <tr>
      <th>HoopaHoopa Unbound</th>
      <td>Psychic</td>
      <td>Dark</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Volcanion</th>
      <td>Fire</td>
      <td>Water</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 5 columns</p>
</div>



#### （2）按索引位置选取（iloc做法）


```python
df_name.iloc[:,:5]
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
    </tr>
    <tr>
      <th>姓名</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bulbasaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
    </tr>
    <tr>
      <th>Ivysaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
    </tr>
    <tr>
      <th>Charizard</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard X</th>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard Y</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
    </tr>
    <tr>
      <th>Squirtle</th>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Wartortle</th>
      <td>Water</td>
      <td>NaN</td>
      <td>405</td>
      <td>59</td>
      <td>63</td>
    </tr>
    <tr>
      <th>Blastoise</th>
      <td>Water</td>
      <td>NaN</td>
      <td>530</td>
      <td>79</td>
      <td>83</td>
    </tr>
    <tr>
      <th>BlastoiseMega Blastoise</th>
      <td>Water</td>
      <td>NaN</td>
      <td>630</td>
      <td>79</td>
      <td>103</td>
    </tr>
    <tr>
      <th>Caterpie</th>
      <td>Bug</td>
      <td>NaN</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Metapod</th>
      <td>Bug</td>
      <td>NaN</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
    </tr>
    <tr>
      <th>Butterfree</th>
      <td>Bug</td>
      <td>Flying</td>
      <td>395</td>
      <td>60</td>
      <td>45</td>
    </tr>
    <tr>
      <th>Weedle</th>
      <td>Bug</td>
      <td>Poison</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Kakuna</th>
      <td>Bug</td>
      <td>Poison</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
    </tr>
    <tr>
      <th>Beedrill</th>
      <td>Bug</td>
      <td>Poison</td>
      <td>395</td>
      <td>65</td>
      <td>90</td>
    </tr>
    <tr>
      <th>BeedrillMega Beedrill</th>
      <td>Bug</td>
      <td>Poison</td>
      <td>495</td>
      <td>65</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Pidgey</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>251</td>
      <td>40</td>
      <td>45</td>
    </tr>
    <tr>
      <th>Pidgeotto</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>349</td>
      <td>63</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Pidgeot</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>479</td>
      <td>83</td>
      <td>80</td>
    </tr>
    <tr>
      <th>PidgeotMega Pidgeot</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>579</td>
      <td>83</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Rattata</th>
      <td>Normal</td>
      <td>NaN</td>
      <td>253</td>
      <td>30</td>
      <td>56</td>
    </tr>
    <tr>
      <th>Raticate</th>
      <td>Normal</td>
      <td>NaN</td>
      <td>413</td>
      <td>55</td>
      <td>81</td>
    </tr>
    <tr>
      <th>Spearow</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>262</td>
      <td>40</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Fearow</th>
      <td>Normal</td>
      <td>Flying</td>
      <td>442</td>
      <td>65</td>
      <td>90</td>
    </tr>
    <tr>
      <th>Ekans</th>
      <td>Poison</td>
      <td>NaN</td>
      <td>288</td>
      <td>35</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Arbok</th>
      <td>Poison</td>
      <td>NaN</td>
      <td>438</td>
      <td>60</td>
      <td>85</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Sylveon</th>
      <td>Fairy</td>
      <td>NaN</td>
      <td>525</td>
      <td>95</td>
      <td>65</td>
    </tr>
    <tr>
      <th>Hawlucha</th>
      <td>Fighting</td>
      <td>Flying</td>
      <td>500</td>
      <td>78</td>
      <td>92</td>
    </tr>
    <tr>
      <th>Dedenne</th>
      <td>Electric</td>
      <td>Fairy</td>
      <td>431</td>
      <td>67</td>
      <td>58</td>
    </tr>
    <tr>
      <th>Carbink</th>
      <td>Rock</td>
      <td>Fairy</td>
      <td>500</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Goomy</th>
      <td>Dragon</td>
      <td>NaN</td>
      <td>300</td>
      <td>45</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Sliggoo</th>
      <td>Dragon</td>
      <td>NaN</td>
      <td>452</td>
      <td>68</td>
      <td>75</td>
    </tr>
    <tr>
      <th>Goodra</th>
      <td>Dragon</td>
      <td>NaN</td>
      <td>600</td>
      <td>90</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Klefki</th>
      <td>Steel</td>
      <td>Fairy</td>
      <td>470</td>
      <td>57</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Phantump</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>309</td>
      <td>43</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Trevenant</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>474</td>
      <td>85</td>
      <td>110</td>
    </tr>
    <tr>
      <th>PumpkabooAverage Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>49</td>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooSmall Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>44</td>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooLarge Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>54</td>
      <td>66</td>
    </tr>
    <tr>
      <th>PumpkabooSuper Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>335</td>
      <td>59</td>
      <td>66</td>
    </tr>
    <tr>
      <th>GourgeistAverage Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>65</td>
      <td>90</td>
    </tr>
    <tr>
      <th>GourgeistSmall Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>55</td>
      <td>85</td>
    </tr>
    <tr>
      <th>GourgeistLarge Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>75</td>
      <td>95</td>
    </tr>
    <tr>
      <th>GourgeistSuper Size</th>
      <td>Ghost</td>
      <td>Grass</td>
      <td>494</td>
      <td>85</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Bergmite</th>
      <td>Ice</td>
      <td>NaN</td>
      <td>304</td>
      <td>55</td>
      <td>69</td>
    </tr>
    <tr>
      <th>Avalugg</th>
      <td>Ice</td>
      <td>NaN</td>
      <td>514</td>
      <td>95</td>
      <td>117</td>
    </tr>
    <tr>
      <th>Noibat</th>
      <td>Flying</td>
      <td>Dragon</td>
      <td>245</td>
      <td>40</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Noivern</th>
      <td>Flying</td>
      <td>Dragon</td>
      <td>535</td>
      <td>85</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Xerneas</th>
      <td>Fairy</td>
      <td>NaN</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
    </tr>
    <tr>
      <th>Yveltal</th>
      <td>Dark</td>
      <td>Flying</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
    </tr>
    <tr>
      <th>Zygarde50% Forme</th>
      <td>Dragon</td>
      <td>Ground</td>
      <td>600</td>
      <td>108</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Diancie</th>
      <td>Rock</td>
      <td>Fairy</td>
      <td>600</td>
      <td>50</td>
      <td>100</td>
    </tr>
    <tr>
      <th>DiancieMega Diancie</th>
      <td>Rock</td>
      <td>Fairy</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
    </tr>
    <tr>
      <th>HoopaHoopa Confined</th>
      <td>Psychic</td>
      <td>Ghost</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
    </tr>
    <tr>
      <th>HoopaHoopa Unbound</th>
      <td>Psychic</td>
      <td>Dark</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Volcanion</th>
      <td>Fire</td>
      <td>Water</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 5 columns</p>
</div>



### Q6:选取“类型2”列，攻击力列，防御力列的所有数据（选取特定列的数据）

#### （1）方括号+列名


```python
#用方括号+列名来直接提取,这种方式比较简洁
df_name[["类型2","攻击力","防御力"]]
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
      <th>类型2</th>
      <th>攻击力</th>
      <th>防御力</th>
    </tr>
    <tr>
      <th>姓名</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bulbasaur</th>
      <td>Poison</td>
      <td>49</td>
      <td>49</td>
    </tr>
    <tr>
      <th>Ivysaur</th>
      <td>Poison</td>
      <td>62</td>
      <td>63</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>Poison</td>
      <td>82</td>
      <td>83</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>Poison</td>
      <td>100</td>
      <td>123</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>NaN</td>
      <td>52</td>
      <td>43</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>NaN</td>
      <td>64</td>
      <td>58</td>
    </tr>
    <tr>
      <th>Charizard</th>
      <td>Flying</td>
      <td>84</td>
      <td>78</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard X</th>
      <td>Dragon</td>
      <td>130</td>
      <td>111</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard Y</th>
      <td>Flying</td>
      <td>104</td>
      <td>78</td>
    </tr>
    <tr>
      <th>Squirtle</th>
      <td>NaN</td>
      <td>48</td>
      <td>65</td>
    </tr>
    <tr>
      <th>Wartortle</th>
      <td>NaN</td>
      <td>63</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Blastoise</th>
      <td>NaN</td>
      <td>83</td>
      <td>100</td>
    </tr>
    <tr>
      <th>BlastoiseMega Blastoise</th>
      <td>NaN</td>
      <td>103</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Caterpie</th>
      <td>NaN</td>
      <td>30</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Metapod</th>
      <td>NaN</td>
      <td>20</td>
      <td>55</td>
    </tr>
    <tr>
      <th>Butterfree</th>
      <td>Flying</td>
      <td>45</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Weedle</th>
      <td>Poison</td>
      <td>35</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Kakuna</th>
      <td>Poison</td>
      <td>25</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Beedrill</th>
      <td>Poison</td>
      <td>90</td>
      <td>40</td>
    </tr>
    <tr>
      <th>BeedrillMega Beedrill</th>
      <td>Poison</td>
      <td>150</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Pidgey</th>
      <td>Flying</td>
      <td>45</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Pidgeotto</th>
      <td>Flying</td>
      <td>60</td>
      <td>55</td>
    </tr>
    <tr>
      <th>Pidgeot</th>
      <td>Flying</td>
      <td>80</td>
      <td>75</td>
    </tr>
    <tr>
      <th>PidgeotMega Pidgeot</th>
      <td>Flying</td>
      <td>80</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Rattata</th>
      <td>NaN</td>
      <td>56</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Raticate</th>
      <td>NaN</td>
      <td>81</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Spearow</th>
      <td>Flying</td>
      <td>60</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Fearow</th>
      <td>Flying</td>
      <td>90</td>
      <td>65</td>
    </tr>
    <tr>
      <th>Ekans</th>
      <td>NaN</td>
      <td>60</td>
      <td>44</td>
    </tr>
    <tr>
      <th>Arbok</th>
      <td>NaN</td>
      <td>85</td>
      <td>69</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Sylveon</th>
      <td>NaN</td>
      <td>65</td>
      <td>65</td>
    </tr>
    <tr>
      <th>Hawlucha</th>
      <td>Flying</td>
      <td>92</td>
      <td>75</td>
    </tr>
    <tr>
      <th>Dedenne</th>
      <td>Fairy</td>
      <td>58</td>
      <td>57</td>
    </tr>
    <tr>
      <th>Carbink</th>
      <td>Fairy</td>
      <td>50</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Goomy</th>
      <td>NaN</td>
      <td>50</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Sliggoo</th>
      <td>NaN</td>
      <td>75</td>
      <td>53</td>
    </tr>
    <tr>
      <th>Goodra</th>
      <td>NaN</td>
      <td>100</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Klefki</th>
      <td>Fairy</td>
      <td>80</td>
      <td>91</td>
    </tr>
    <tr>
      <th>Phantump</th>
      <td>Grass</td>
      <td>70</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Trevenant</th>
      <td>Grass</td>
      <td>110</td>
      <td>76</td>
    </tr>
    <tr>
      <th>PumpkabooAverage Size</th>
      <td>Grass</td>
      <td>66</td>
      <td>70</td>
    </tr>
    <tr>
      <th>PumpkabooSmall Size</th>
      <td>Grass</td>
      <td>66</td>
      <td>70</td>
    </tr>
    <tr>
      <th>PumpkabooLarge Size</th>
      <td>Grass</td>
      <td>66</td>
      <td>70</td>
    </tr>
    <tr>
      <th>PumpkabooSuper Size</th>
      <td>Grass</td>
      <td>66</td>
      <td>70</td>
    </tr>
    <tr>
      <th>GourgeistAverage Size</th>
      <td>Grass</td>
      <td>90</td>
      <td>122</td>
    </tr>
    <tr>
      <th>GourgeistSmall Size</th>
      <td>Grass</td>
      <td>85</td>
      <td>122</td>
    </tr>
    <tr>
      <th>GourgeistLarge Size</th>
      <td>Grass</td>
      <td>95</td>
      <td>122</td>
    </tr>
    <tr>
      <th>GourgeistSuper Size</th>
      <td>Grass</td>
      <td>100</td>
      <td>122</td>
    </tr>
    <tr>
      <th>Bergmite</th>
      <td>NaN</td>
      <td>69</td>
      <td>85</td>
    </tr>
    <tr>
      <th>Avalugg</th>
      <td>NaN</td>
      <td>117</td>
      <td>184</td>
    </tr>
    <tr>
      <th>Noibat</th>
      <td>Dragon</td>
      <td>30</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Noivern</th>
      <td>Dragon</td>
      <td>70</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Xerneas</th>
      <td>NaN</td>
      <td>131</td>
      <td>95</td>
    </tr>
    <tr>
      <th>Yveltal</th>
      <td>Flying</td>
      <td>131</td>
      <td>95</td>
    </tr>
    <tr>
      <th>Zygarde50% Forme</th>
      <td>Ground</td>
      <td>100</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Diancie</th>
      <td>Fairy</td>
      <td>100</td>
      <td>150</td>
    </tr>
    <tr>
      <th>DiancieMega Diancie</th>
      <td>Fairy</td>
      <td>160</td>
      <td>110</td>
    </tr>
    <tr>
      <th>HoopaHoopa Confined</th>
      <td>Ghost</td>
      <td>110</td>
      <td>60</td>
    </tr>
    <tr>
      <th>HoopaHoopa Unbound</th>
      <td>Dark</td>
      <td>160</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Volcanion</th>
      <td>Water</td>
      <td>110</td>
      <td>120</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 3 columns</p>
</div>



#### （2）按索引标签选取（loc做法）


```python
df_name.loc[:,["类型2","攻击力","防御力"]]
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
      <th>类型2</th>
      <th>攻击力</th>
      <th>防御力</th>
    </tr>
    <tr>
      <th>姓名</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bulbasaur</th>
      <td>Poison</td>
      <td>49</td>
      <td>49</td>
    </tr>
    <tr>
      <th>Ivysaur</th>
      <td>Poison</td>
      <td>62</td>
      <td>63</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>Poison</td>
      <td>82</td>
      <td>83</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>Poison</td>
      <td>100</td>
      <td>123</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>NaN</td>
      <td>52</td>
      <td>43</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>NaN</td>
      <td>64</td>
      <td>58</td>
    </tr>
    <tr>
      <th>Charizard</th>
      <td>Flying</td>
      <td>84</td>
      <td>78</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard X</th>
      <td>Dragon</td>
      <td>130</td>
      <td>111</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard Y</th>
      <td>Flying</td>
      <td>104</td>
      <td>78</td>
    </tr>
    <tr>
      <th>Squirtle</th>
      <td>NaN</td>
      <td>48</td>
      <td>65</td>
    </tr>
    <tr>
      <th>Wartortle</th>
      <td>NaN</td>
      <td>63</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Blastoise</th>
      <td>NaN</td>
      <td>83</td>
      <td>100</td>
    </tr>
    <tr>
      <th>BlastoiseMega Blastoise</th>
      <td>NaN</td>
      <td>103</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Caterpie</th>
      <td>NaN</td>
      <td>30</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Metapod</th>
      <td>NaN</td>
      <td>20</td>
      <td>55</td>
    </tr>
    <tr>
      <th>Butterfree</th>
      <td>Flying</td>
      <td>45</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Weedle</th>
      <td>Poison</td>
      <td>35</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Kakuna</th>
      <td>Poison</td>
      <td>25</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Beedrill</th>
      <td>Poison</td>
      <td>90</td>
      <td>40</td>
    </tr>
    <tr>
      <th>BeedrillMega Beedrill</th>
      <td>Poison</td>
      <td>150</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Pidgey</th>
      <td>Flying</td>
      <td>45</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Pidgeotto</th>
      <td>Flying</td>
      <td>60</td>
      <td>55</td>
    </tr>
    <tr>
      <th>Pidgeot</th>
      <td>Flying</td>
      <td>80</td>
      <td>75</td>
    </tr>
    <tr>
      <th>PidgeotMega Pidgeot</th>
      <td>Flying</td>
      <td>80</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Rattata</th>
      <td>NaN</td>
      <td>56</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Raticate</th>
      <td>NaN</td>
      <td>81</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Spearow</th>
      <td>Flying</td>
      <td>60</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Fearow</th>
      <td>Flying</td>
      <td>90</td>
      <td>65</td>
    </tr>
    <tr>
      <th>Ekans</th>
      <td>NaN</td>
      <td>60</td>
      <td>44</td>
    </tr>
    <tr>
      <th>Arbok</th>
      <td>NaN</td>
      <td>85</td>
      <td>69</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Sylveon</th>
      <td>NaN</td>
      <td>65</td>
      <td>65</td>
    </tr>
    <tr>
      <th>Hawlucha</th>
      <td>Flying</td>
      <td>92</td>
      <td>75</td>
    </tr>
    <tr>
      <th>Dedenne</th>
      <td>Fairy</td>
      <td>58</td>
      <td>57</td>
    </tr>
    <tr>
      <th>Carbink</th>
      <td>Fairy</td>
      <td>50</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Goomy</th>
      <td>NaN</td>
      <td>50</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Sliggoo</th>
      <td>NaN</td>
      <td>75</td>
      <td>53</td>
    </tr>
    <tr>
      <th>Goodra</th>
      <td>NaN</td>
      <td>100</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Klefki</th>
      <td>Fairy</td>
      <td>80</td>
      <td>91</td>
    </tr>
    <tr>
      <th>Phantump</th>
      <td>Grass</td>
      <td>70</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Trevenant</th>
      <td>Grass</td>
      <td>110</td>
      <td>76</td>
    </tr>
    <tr>
      <th>PumpkabooAverage Size</th>
      <td>Grass</td>
      <td>66</td>
      <td>70</td>
    </tr>
    <tr>
      <th>PumpkabooSmall Size</th>
      <td>Grass</td>
      <td>66</td>
      <td>70</td>
    </tr>
    <tr>
      <th>PumpkabooLarge Size</th>
      <td>Grass</td>
      <td>66</td>
      <td>70</td>
    </tr>
    <tr>
      <th>PumpkabooSuper Size</th>
      <td>Grass</td>
      <td>66</td>
      <td>70</td>
    </tr>
    <tr>
      <th>GourgeistAverage Size</th>
      <td>Grass</td>
      <td>90</td>
      <td>122</td>
    </tr>
    <tr>
      <th>GourgeistSmall Size</th>
      <td>Grass</td>
      <td>85</td>
      <td>122</td>
    </tr>
    <tr>
      <th>GourgeistLarge Size</th>
      <td>Grass</td>
      <td>95</td>
      <td>122</td>
    </tr>
    <tr>
      <th>GourgeistSuper Size</th>
      <td>Grass</td>
      <td>100</td>
      <td>122</td>
    </tr>
    <tr>
      <th>Bergmite</th>
      <td>NaN</td>
      <td>69</td>
      <td>85</td>
    </tr>
    <tr>
      <th>Avalugg</th>
      <td>NaN</td>
      <td>117</td>
      <td>184</td>
    </tr>
    <tr>
      <th>Noibat</th>
      <td>Dragon</td>
      <td>30</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Noivern</th>
      <td>Dragon</td>
      <td>70</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Xerneas</th>
      <td>NaN</td>
      <td>131</td>
      <td>95</td>
    </tr>
    <tr>
      <th>Yveltal</th>
      <td>Flying</td>
      <td>131</td>
      <td>95</td>
    </tr>
    <tr>
      <th>Zygarde50% Forme</th>
      <td>Ground</td>
      <td>100</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Diancie</th>
      <td>Fairy</td>
      <td>100</td>
      <td>150</td>
    </tr>
    <tr>
      <th>DiancieMega Diancie</th>
      <td>Fairy</td>
      <td>160</td>
      <td>110</td>
    </tr>
    <tr>
      <th>HoopaHoopa Confined</th>
      <td>Ghost</td>
      <td>110</td>
      <td>60</td>
    </tr>
    <tr>
      <th>HoopaHoopa Unbound</th>
      <td>Dark</td>
      <td>160</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Volcanion</th>
      <td>Water</td>
      <td>110</td>
      <td>120</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 3 columns</p>
</div>



#### （3）按索引位置选取（iloc做法）


```python
df_name.iloc[:,[1,4,5]]
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
      <th>类型2</th>
      <th>攻击力</th>
      <th>防御力</th>
    </tr>
    <tr>
      <th>姓名</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bulbasaur</th>
      <td>Poison</td>
      <td>49</td>
      <td>49</td>
    </tr>
    <tr>
      <th>Ivysaur</th>
      <td>Poison</td>
      <td>62</td>
      <td>63</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>Poison</td>
      <td>82</td>
      <td>83</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>Poison</td>
      <td>100</td>
      <td>123</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>NaN</td>
      <td>52</td>
      <td>43</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>NaN</td>
      <td>64</td>
      <td>58</td>
    </tr>
    <tr>
      <th>Charizard</th>
      <td>Flying</td>
      <td>84</td>
      <td>78</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard X</th>
      <td>Dragon</td>
      <td>130</td>
      <td>111</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard Y</th>
      <td>Flying</td>
      <td>104</td>
      <td>78</td>
    </tr>
    <tr>
      <th>Squirtle</th>
      <td>NaN</td>
      <td>48</td>
      <td>65</td>
    </tr>
    <tr>
      <th>Wartortle</th>
      <td>NaN</td>
      <td>63</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Blastoise</th>
      <td>NaN</td>
      <td>83</td>
      <td>100</td>
    </tr>
    <tr>
      <th>BlastoiseMega Blastoise</th>
      <td>NaN</td>
      <td>103</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Caterpie</th>
      <td>NaN</td>
      <td>30</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Metapod</th>
      <td>NaN</td>
      <td>20</td>
      <td>55</td>
    </tr>
    <tr>
      <th>Butterfree</th>
      <td>Flying</td>
      <td>45</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Weedle</th>
      <td>Poison</td>
      <td>35</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Kakuna</th>
      <td>Poison</td>
      <td>25</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Beedrill</th>
      <td>Poison</td>
      <td>90</td>
      <td>40</td>
    </tr>
    <tr>
      <th>BeedrillMega Beedrill</th>
      <td>Poison</td>
      <td>150</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Pidgey</th>
      <td>Flying</td>
      <td>45</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Pidgeotto</th>
      <td>Flying</td>
      <td>60</td>
      <td>55</td>
    </tr>
    <tr>
      <th>Pidgeot</th>
      <td>Flying</td>
      <td>80</td>
      <td>75</td>
    </tr>
    <tr>
      <th>PidgeotMega Pidgeot</th>
      <td>Flying</td>
      <td>80</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Rattata</th>
      <td>NaN</td>
      <td>56</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Raticate</th>
      <td>NaN</td>
      <td>81</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Spearow</th>
      <td>Flying</td>
      <td>60</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Fearow</th>
      <td>Flying</td>
      <td>90</td>
      <td>65</td>
    </tr>
    <tr>
      <th>Ekans</th>
      <td>NaN</td>
      <td>60</td>
      <td>44</td>
    </tr>
    <tr>
      <th>Arbok</th>
      <td>NaN</td>
      <td>85</td>
      <td>69</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Sylveon</th>
      <td>NaN</td>
      <td>65</td>
      <td>65</td>
    </tr>
    <tr>
      <th>Hawlucha</th>
      <td>Flying</td>
      <td>92</td>
      <td>75</td>
    </tr>
    <tr>
      <th>Dedenne</th>
      <td>Fairy</td>
      <td>58</td>
      <td>57</td>
    </tr>
    <tr>
      <th>Carbink</th>
      <td>Fairy</td>
      <td>50</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Goomy</th>
      <td>NaN</td>
      <td>50</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Sliggoo</th>
      <td>NaN</td>
      <td>75</td>
      <td>53</td>
    </tr>
    <tr>
      <th>Goodra</th>
      <td>NaN</td>
      <td>100</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Klefki</th>
      <td>Fairy</td>
      <td>80</td>
      <td>91</td>
    </tr>
    <tr>
      <th>Phantump</th>
      <td>Grass</td>
      <td>70</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Trevenant</th>
      <td>Grass</td>
      <td>110</td>
      <td>76</td>
    </tr>
    <tr>
      <th>PumpkabooAverage Size</th>
      <td>Grass</td>
      <td>66</td>
      <td>70</td>
    </tr>
    <tr>
      <th>PumpkabooSmall Size</th>
      <td>Grass</td>
      <td>66</td>
      <td>70</td>
    </tr>
    <tr>
      <th>PumpkabooLarge Size</th>
      <td>Grass</td>
      <td>66</td>
      <td>70</td>
    </tr>
    <tr>
      <th>PumpkabooSuper Size</th>
      <td>Grass</td>
      <td>66</td>
      <td>70</td>
    </tr>
    <tr>
      <th>GourgeistAverage Size</th>
      <td>Grass</td>
      <td>90</td>
      <td>122</td>
    </tr>
    <tr>
      <th>GourgeistSmall Size</th>
      <td>Grass</td>
      <td>85</td>
      <td>122</td>
    </tr>
    <tr>
      <th>GourgeistLarge Size</th>
      <td>Grass</td>
      <td>95</td>
      <td>122</td>
    </tr>
    <tr>
      <th>GourgeistSuper Size</th>
      <td>Grass</td>
      <td>100</td>
      <td>122</td>
    </tr>
    <tr>
      <th>Bergmite</th>
      <td>NaN</td>
      <td>69</td>
      <td>85</td>
    </tr>
    <tr>
      <th>Avalugg</th>
      <td>NaN</td>
      <td>117</td>
      <td>184</td>
    </tr>
    <tr>
      <th>Noibat</th>
      <td>Dragon</td>
      <td>30</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Noivern</th>
      <td>Dragon</td>
      <td>70</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Xerneas</th>
      <td>NaN</td>
      <td>131</td>
      <td>95</td>
    </tr>
    <tr>
      <th>Yveltal</th>
      <td>Flying</td>
      <td>131</td>
      <td>95</td>
    </tr>
    <tr>
      <th>Zygarde50% Forme</th>
      <td>Ground</td>
      <td>100</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Diancie</th>
      <td>Fairy</td>
      <td>100</td>
      <td>150</td>
    </tr>
    <tr>
      <th>DiancieMega Diancie</th>
      <td>Fairy</td>
      <td>160</td>
      <td>110</td>
    </tr>
    <tr>
      <th>HoopaHoopa Confined</th>
      <td>Ghost</td>
      <td>110</td>
      <td>60</td>
    </tr>
    <tr>
      <th>HoopaHoopa Unbound</th>
      <td>Dark</td>
      <td>160</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Volcanion</th>
      <td>Water</td>
      <td>110</td>
      <td>120</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 3 columns</p>
</div>



### Q7:选取第3行到第8行，类型1列到攻击力列（选取部分行部分列的数据）

#### （1）按索引标签选取（loc做法）


```python
df_name.head(10)
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
      <th>class</th>
    </tr>
    <tr>
      <th>姓名</th>
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
      <th>Bulbasaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>45</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Ivysaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>60</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Charizard</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard X</th>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard Y</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Squirtle</th>
      <td>Water</td>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_name.loc["Venusaur":"CharizardMega Charizard X","类型1":"攻击力"]
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
    </tr>
    <tr>
      <th>姓名</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Venusaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
    </tr>
    <tr>
      <th>Charizard</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard X</th>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
    </tr>
  </tbody>
</table>
</div>



#### （2）按索引位置选取（iloc做法）


```python
df_name.iloc[2:8,:5]
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
    </tr>
    <tr>
      <th>姓名</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Venusaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
    </tr>
    <tr>
      <th>Charizard</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard X</th>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
    </tr>
  </tbody>
</table>
</div>



### ix选取数据的做法


```python
#示例数据，在这里索引为整数标签
df_1 = df.head(10)
df_1
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




```python
#选出第2到第4行的数据
df_1.ix[1:3]
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
  </tbody>
</table>
</div>




```python
#选出类型1列到攻击力列的数据
df_1.ix[:,"类型1":"攻击力"]
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
    </tr>
    <tr>
      <th>9</th>
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
#列也能通过位置选出
df_1.ix[:,1:6]
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Fire</td>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Fire</td>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
    </tr>
    <tr>
      <th>9</th>
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
#选出第2行到第5行，第2列到第4列的数据
df.ix[1:4,"类型1":"总计"]
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
      <th>类型2</th>
      <th>总计</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
    </tr>
  </tbody>
</table>
</div>




```python
#或者列是通过位置选出
df.ix[1:4,1:4]
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
      <th>类型2</th>
      <th>总计</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
    </tr>
  </tbody>
</table>
</div>




```python
#示例数据，在这里索引为非整数标签
df_2 = df_1.set_index("姓名")
df_2
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
    <tr>
      <th>姓名</th>
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
      <th>Bulbasaur</th>
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
      <th>Ivysaur</th>
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
      <th>Venusaur</th>
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
      <th>VenusaurMega Venusaur</th>
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
      <th>Charmander</th>
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
      <th>Charmeleon</th>
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
      <th>Charizard</th>
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
      <th>CharizardMega Charizard X</th>
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
      <th>CharizardMega Charizard Y</th>
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
      <th>Squirtle</th>
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




```python
#选取第2行到第4行的数据
df_2.ix[1:4]
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
    <tr>
      <th>姓名</th>
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
      <th>Ivysaur</th>
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
      <th>Venusaur</th>
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
      <th>VenusaurMega Venusaur</th>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#选取第2列到第5列的数据
df_2.ix[:,1:5]
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
    </tr>
    <tr>
      <th>姓名</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bulbasaur</th>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
    </tr>
    <tr>
      <th>Ivysaur</th>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
    </tr>
    <tr>
      <th>Charizard</th>
      <td>Flying</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard X</th>
      <td>Dragon</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
    </tr>
    <tr>
      <th>CharizardMega Charizard Y</th>
      <td>Flying</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
    </tr>
    <tr>
      <th>Squirtle</th>
      <td>NaN</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
    </tr>
  </tbody>
</table>
</div>




```python
#选取第2行到第6行，第2列到第5列的数据
df_2.ix[1:6,1:5]
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
    </tr>
    <tr>
      <th>姓名</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ivysaur</th>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
    </tr>
  </tbody>
</table>
</div>




```python
#也可以用索引标签选取
df_2.ix["Ivysaur":"Charmeleon","类型2":"攻击力"]
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
      <th>类型2</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
    </tr>
    <tr>
      <th>姓名</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ivysaur</th>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
    </tr>
    <tr>
      <th>Venusaur</th>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
    </tr>
    <tr>
      <th>VenusaurMega Venusaur</th>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Charmander</th>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>Charmeleon</th>
      <td>NaN</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
    </tr>
  </tbody>
</table>
</div>



ix的工作原理：根据索引的类型分2种情况：

1.索引为整数标签，那么按照索引标签选取行数据,不能按照索引位置选取行数据，列数据既能通过标签选取也能通过位置选取。

2.当索引为非整数标签（如字符串标签），那么可以用索引标签选取行数据，也可以按照索引位置选取行数据，列数据既能通过标签选取也能通过位置选取。

用两个例子说明ix的工作原理，并且说明下ix和loc和iloc的区别

**例子1，索引为整数标签**


```python
df_number = pd.DataFrame({"id":list("abcdefghij"),"score":range(10,20)},index=[27,26,25,24,23,1,2,3,4,5])
df_number
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
      <th>id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>a</td>
      <td>10</td>
    </tr>
    <tr>
      <th>26</th>
      <td>b</td>
      <td>11</td>
    </tr>
    <tr>
      <th>25</th>
      <td>c</td>
      <td>12</td>
    </tr>
    <tr>
      <th>24</th>
      <td>d</td>
      <td>13</td>
    </tr>
    <tr>
      <th>23</th>
      <td>e</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>h</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>i</td>
      <td>18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>j</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>



+ 第一种情况，用在索引内的整数标签来测试（例如[:2]）


```python
df_number.loc[:2]
#loc按照索引标签选取数据，因此返回了第1行到第7行的数据
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
      <th>id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>a</td>
      <td>10</td>
    </tr>
    <tr>
      <th>26</th>
      <td>b</td>
      <td>11</td>
    </tr>
    <tr>
      <th>25</th>
      <td>c</td>
      <td>12</td>
    </tr>
    <tr>
      <th>24</th>
      <td>d</td>
      <td>13</td>
    </tr>
    <tr>
      <th>23</th>
      <td>e</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_number.iloc[:2]
#iloc按照索引位置选取数据，因此返回了前2行的数据
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
      <th>id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>a</td>
      <td>10</td>
    </tr>
    <tr>
      <th>26</th>
      <td>b</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_number.ix[:2]
#返回了第1行到第7行的数据，说明ix是优先按照索引标签选取数据
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
      <th>id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>a</td>
      <td>10</td>
    </tr>
    <tr>
      <th>26</th>
      <td>b</td>
      <td>11</td>
    </tr>
    <tr>
      <th>25</th>
      <td>c</td>
      <td>12</td>
    </tr>
    <tr>
      <th>24</th>
      <td>d</td>
      <td>13</td>
    </tr>
    <tr>
      <th>23</th>
      <td>e</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



+ 第二种情况，用不在索引内的整数标签来测试（例如6）


```python
df_number.loc[:6]
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in get_slice_bound(self, label, side, kind)
       3434             try:
    -> 3435                 return self._searchsorted_monotonic(label, side)
       3436             except ValueError:


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in _searchsorted_monotonic(self, label, side)
       3393 
    -> 3394         raise ValueError('index must be monotonic increasing or decreasing')
       3395 


    ValueError: index must be monotonic increasing or decreasing

    
    During handling of the above exception, another exception occurred:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-160-a41a49e8095a> in <module>()
    ----> 1 df_number.loc[:6]
    

    D:\anaconda\lib\site-packages\pandas\core\indexing.py in __getitem__(self, key)
       1326         else:
       1327             key = com._apply_if_callable(key, self.obj)
    -> 1328             return self._getitem_axis(key, axis=0)
       1329 
       1330     def _is_scalar_access(self, key):


    D:\anaconda\lib\site-packages\pandas\core\indexing.py in _getitem_axis(self, key, axis)
       1504         if isinstance(key, slice):
       1505             self._has_valid_type(key, axis)
    -> 1506             return self._get_slice_axis(key, axis=axis)
       1507         elif is_bool_indexer(key):
       1508             return self._getbool_axis(key, axis=axis)


    D:\anaconda\lib\site-packages\pandas\core\indexing.py in _get_slice_axis(self, slice_obj, axis)
       1354         labels = obj._get_axis(axis)
       1355         indexer = labels.slice_indexer(slice_obj.start, slice_obj.stop,
    -> 1356                                        slice_obj.step, kind=self.name)
       1357 
       1358         if isinstance(indexer, slice):


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in slice_indexer(self, start, end, step, kind)
       3299         """
       3300         start_slice, end_slice = self.slice_locs(start, end, step=step,
    -> 3301                                                  kind=kind)
       3302 
       3303         # return a slice


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in slice_locs(self, start, end, step, kind)
       3493         end_slice = None
       3494         if end is not None:
    -> 3495             end_slice = self.get_slice_bound(end, 'right', kind)
       3496         if end_slice is None:
       3497             end_slice = len(self)


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in get_slice_bound(self, label, side, kind)
       3436             except ValueError:
       3437                 # raise the original KeyError
    -> 3438                 raise err
       3439 
       3440         if isinstance(slc, np.ndarray):


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in get_slice_bound(self, label, side, kind)
       3430         # we need to look up the label
       3431         try:
    -> 3432             slc = self._get_loc_only_exact_matches(label)
       3433         except KeyError as err:
       3434             try:


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in _get_loc_only_exact_matches(self, key)
       3399         get_slice_bound.
       3400         """
    -> 3401         return self.get_loc(key)
       3402 
       3403     def get_slice_bound(self, label, side, kind):


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       2393                 return self._engine.get_loc(key)
       2394             except KeyError:
    -> 2395                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2396 
       2397         indexer = self.get_indexer([key], method=method, tolerance=tolerance)


    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas\_libs\index.c:5239)()


    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas\_libs\index.c:5085)()


    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item (pandas\_libs\hashtable.c:13913)()


    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item (pandas\_libs\hashtable.c:13857)()


    KeyError: 6



```python
df_number.iloc[:6]
#因为是按照位置选取，所以返回了前6行数据
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
      <th>id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>a</td>
      <td>10</td>
    </tr>
    <tr>
      <th>26</th>
      <td>b</td>
      <td>11</td>
    </tr>
    <tr>
      <th>25</th>
      <td>c</td>
      <td>12</td>
    </tr>
    <tr>
      <th>24</th>
      <td>d</td>
      <td>13</td>
    </tr>
    <tr>
      <th>23</th>
      <td>e</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_number.ix[:6]
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in get_slice_bound(self, label, side, kind)
       3434             try:
    -> 3435                 return self._searchsorted_monotonic(label, side)
       3436             except ValueError:


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in _searchsorted_monotonic(self, label, side)
       3393 
    -> 3394         raise ValueError('index must be monotonic increasing or decreasing')
       3395 


    ValueError: index must be monotonic increasing or decreasing

    
    During handling of the above exception, another exception occurred:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-164-50d6444266b8> in <module>()
    ----> 1 df_number.ix[:6]
    

    D:\anaconda\lib\site-packages\pandas\core\indexing.py in __getitem__(self, key)
        119         else:
        120             key = com._apply_if_callable(key, self.obj)
    --> 121             return self._getitem_axis(key, axis=0)
        122 
        123     def _get_label(self, label, axis=0):


    D:\anaconda\lib\site-packages\pandas\core\indexing.py in _getitem_axis(self, key, axis)
       1049         labels = self.obj._get_axis(axis)
       1050         if isinstance(key, slice):
    -> 1051             return self._get_slice_axis(key, axis=axis)
       1052         elif (is_list_like_indexer(key) and
       1053               not (isinstance(key, tuple) and


    D:\anaconda\lib\site-packages\pandas\core\indexing.py in _get_slice_axis(self, slice_obj, axis)
       1252         if not need_slice(slice_obj):
       1253             return obj
    -> 1254         indexer = self._convert_slice_indexer(slice_obj, axis)
       1255 
       1256         if isinstance(indexer, slice):


    D:\anaconda\lib\site-packages\pandas\core\indexing.py in _convert_slice_indexer(self, key, axis)
        239         # if we are accessing via lowered dim, use the last dim
        240         ax = self.obj._get_axis(min(axis, self.ndim - 1))
    --> 241         return ax._convert_slice_indexer(key, kind=self.name)
        242 
        243     def _has_valid_setitem_indexer(self, indexer):


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in _convert_slice_indexer(self, key, kind)
       1355         else:
       1356             try:
    -> 1357                 indexer = self.slice_indexer(start, stop, step, kind=kind)
       1358             except Exception:
       1359                 if is_index_slice:


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in slice_indexer(self, start, end, step, kind)
       3299         """
       3300         start_slice, end_slice = self.slice_locs(start, end, step=step,
    -> 3301                                                  kind=kind)
       3302 
       3303         # return a slice


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in slice_locs(self, start, end, step, kind)
       3493         end_slice = None
       3494         if end is not None:
    -> 3495             end_slice = self.get_slice_bound(end, 'right', kind)
       3496         if end_slice is None:
       3497             end_slice = len(self)


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in get_slice_bound(self, label, side, kind)
       3436             except ValueError:
       3437                 # raise the original KeyError
    -> 3438                 raise err
       3439 
       3440         if isinstance(slc, np.ndarray):


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in get_slice_bound(self, label, side, kind)
       3430         # we need to look up the label
       3431         try:
    -> 3432             slc = self._get_loc_only_exact_matches(label)
       3433         except KeyError as err:
       3434             try:


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in _get_loc_only_exact_matches(self, key)
       3399         get_slice_bound.
       3400         """
    -> 3401         return self.get_loc(key)
       3402 
       3403     def get_slice_bound(self, label, side, kind):


    D:\anaconda\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       2393                 return self._engine.get_loc(key)
       2394             except KeyError:
    -> 2395                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2396 
       2397         indexer = self.get_indexer([key], method=method, tolerance=tolerance)


    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas\_libs\index.c:5239)()


    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc (pandas\_libs\index.c:5085)()


    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item (pandas\_libs\hashtable.c:13913)()


    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item (pandas\_libs\hashtable.c:13857)()


    KeyError: 6


**第2个例子，索引为非整数标签类型，以字符串标签为例**


```python
df_str = df_number.set_index("id")
df_str
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
      <th>score</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>10</td>
    </tr>
    <tr>
      <th>b</th>
      <td>11</td>
    </tr>
    <tr>
      <th>c</th>
      <td>12</td>
    </tr>
    <tr>
      <th>d</th>
      <td>13</td>
    </tr>
    <tr>
      <th>e</th>
      <td>14</td>
    </tr>
    <tr>
      <th>f</th>
      <td>15</td>
    </tr>
    <tr>
      <th>g</th>
      <td>16</td>
    </tr>
    <tr>
      <th>h</th>
      <td>17</td>
    </tr>
    <tr>
      <th>i</th>
      <td>18</td>
    </tr>
    <tr>
      <th>j</th>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_str.ix[:2]
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
      <th>score</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>10</td>
    </tr>
    <tr>
      <th>b</th>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



**说明了当标签为字符串标签时，即使：2不在索引标签内，ix也能按照位置选取数据**

> 个人建议：还是使用loc和iloc来选取数据较好，因为分工明确，loc通过索引标签选取数据，iloc通过索引位置选取数据，所以你会很清除地知道你是在索引标签上操作还是在索引位置上操作，不会觉得混乱。我并不是特别建议使用ix，ix会让你觉得很混乱，也会让别人看你的代码时会在想：到底现在是在索引标签上操作还是位置上操作？这就增加了一个判断的过程。

## 二.数据筛选


```python
#示例数据
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



### Q1：选取出攻击力大于100的所有数据

#### 1.loc筛选


```python
df.loc[df["攻击力"] > 100]
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
      <th>62</th>
      <td>Primeape</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>455</td>
      <td>65</td>
      <td>105</td>
      <td>60</td>
      <td>95</td>
      <td>1</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Arcanine</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>555</td>
      <td>90</td>
      <td>110</td>
      <td>80</td>
      <td>95</td>
      <td>1</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Machamp</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>505</td>
      <td>90</td>
      <td>130</td>
      <td>80</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Victreebel</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>490</td>
      <td>80</td>
      <td>105</td>
      <td>65</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Golem</td>
      <td>Rock</td>
      <td>Ground</td>
      <td>495</td>
      <td>80</td>
      <td>120</td>
      <td>130</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Dodrio</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>460</td>
      <td>60</td>
      <td>110</td>
      <td>70</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Muk</td>
      <td>Poison</td>
      <td>NaN</td>
      <td>500</td>
      <td>105</td>
      <td>105</td>
      <td>75</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Krabby</td>
      <td>Water</td>
      <td>NaN</td>
      <td>325</td>
      <td>30</td>
      <td>105</td>
      <td>90</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>107</th>
      <td>Kingler</td>
      <td>Water</td>
      <td>NaN</td>
      <td>475</td>
      <td>55</td>
      <td>130</td>
      <td>115</td>
      <td>75</td>
      <td>1</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Hitmonlee</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>455</td>
      <td>50</td>
      <td>120</td>
      <td>53</td>
      <td>87</td>
      <td>1</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Hitmonchan</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>455</td>
      <td>50</td>
      <td>105</td>
      <td>79</td>
      <td>76</td>
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
      <th>124</th>
      <td>KangaskhanMega Kangaskhan</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>590</td>
      <td>105</td>
      <td>125</td>
      <td>100</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Scyther</td>
      <td>Bug</td>
      <td>Flying</td>
      <td>500</td>
      <td>70</td>
      <td>110</td>
      <td>80</td>
      <td>105</td>
      <td>1</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Pinsir</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>500</td>
      <td>65</td>
      <td>125</td>
      <td>100</td>
      <td>85</td>
      <td>1</td>
    </tr>
    <tr>
      <th>137</th>
      <td>PinsirMega Pinsir</td>
      <td>Bug</td>
      <td>Flying</td>
      <td>600</td>
      <td>65</td>
      <td>155</td>
      <td>120</td>
      <td>105</td>
      <td>1</td>
    </tr>
    <tr>
      <th>140</th>
      <td>Gyarados</td>
      <td>Water</td>
      <td>Flying</td>
      <td>540</td>
      <td>95</td>
      <td>125</td>
      <td>79</td>
      <td>81</td>
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
      <th>147</th>
      <td>Flareon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>525</td>
      <td>65</td>
      <td>130</td>
      <td>60</td>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>152</th>
      <td>Kabutops</td>
      <td>Rock</td>
      <td>Water</td>
      <td>495</td>
      <td>60</td>
      <td>115</td>
      <td>105</td>
      <td>80</td>
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
      <th>155</th>
      <td>Snorlax</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>540</td>
      <td>160</td>
      <td>110</td>
      <td>65</td>
      <td>30</td>
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
      <th>689</th>
      <td>Braviary</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>510</td>
      <td>100</td>
      <td>123</td>
      <td>75</td>
      <td>80</td>
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
      <th>696</th>
      <td>Hydreigon</td>
      <td>Dark</td>
      <td>Dragon</td>
      <td>600</td>
      <td>92</td>
      <td>105</td>
      <td>90</td>
      <td>98</td>
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
      <th>702</th>
      <td>TornadusIncarnate Forme</td>
      <td>Flying</td>
      <td>NaN</td>
      <td>580</td>
      <td>79</td>
      <td>115</td>
      <td>70</td>
      <td>111</td>
      <td>5</td>
    </tr>
    <tr>
      <th>704</th>
      <td>ThundurusIncarnate Forme</td>
      <td>Electric</td>
      <td>Flying</td>
      <td>580</td>
      <td>79</td>
      <td>115</td>
      <td>70</td>
      <td>111</td>
      <td>5</td>
    </tr>
    <tr>
      <th>705</th>
      <td>ThundurusTherian Forme</td>
      <td>Electric</td>
      <td>Flying</td>
      <td>580</td>
      <td>79</td>
      <td>105</td>
      <td>70</td>
      <td>101</td>
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
      <th>749</th>
      <td>Doublade</td>
      <td>Steel</td>
      <td>Ghost</td>
      <td>448</td>
      <td>59</td>
      <td>110</td>
      <td>150</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>750</th>
      <td>AegislashBlade Forme</td>
      <td>Steel</td>
      <td>Ghost</td>
      <td>520</td>
      <td>60</td>
      <td>150</td>
      <td>50</td>
      <td>60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>759</th>
      <td>Barbaracle</td>
      <td>Rock</td>
      <td>Water</td>
      <td>500</td>
      <td>72</td>
      <td>105</td>
      <td>115</td>
      <td>68</td>
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
<p>170 rows × 9 columns</p>
</div>



#### 2.query筛选


```python
df.query("攻击力 > 100")
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
      <th>62</th>
      <td>Primeape</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>455</td>
      <td>65</td>
      <td>105</td>
      <td>60</td>
      <td>95</td>
      <td>1</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Arcanine</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>555</td>
      <td>90</td>
      <td>110</td>
      <td>80</td>
      <td>95</td>
      <td>1</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Machamp</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>505</td>
      <td>90</td>
      <td>130</td>
      <td>80</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Victreebel</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>490</td>
      <td>80</td>
      <td>105</td>
      <td>65</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Golem</td>
      <td>Rock</td>
      <td>Ground</td>
      <td>495</td>
      <td>80</td>
      <td>120</td>
      <td>130</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Dodrio</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>460</td>
      <td>60</td>
      <td>110</td>
      <td>70</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Muk</td>
      <td>Poison</td>
      <td>NaN</td>
      <td>500</td>
      <td>105</td>
      <td>105</td>
      <td>75</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Krabby</td>
      <td>Water</td>
      <td>NaN</td>
      <td>325</td>
      <td>30</td>
      <td>105</td>
      <td>90</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>107</th>
      <td>Kingler</td>
      <td>Water</td>
      <td>NaN</td>
      <td>475</td>
      <td>55</td>
      <td>130</td>
      <td>115</td>
      <td>75</td>
      <td>1</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Hitmonlee</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>455</td>
      <td>50</td>
      <td>120</td>
      <td>53</td>
      <td>87</td>
      <td>1</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Hitmonchan</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>455</td>
      <td>50</td>
      <td>105</td>
      <td>79</td>
      <td>76</td>
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
      <th>124</th>
      <td>KangaskhanMega Kangaskhan</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>590</td>
      <td>105</td>
      <td>125</td>
      <td>100</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Scyther</td>
      <td>Bug</td>
      <td>Flying</td>
      <td>500</td>
      <td>70</td>
      <td>110</td>
      <td>80</td>
      <td>105</td>
      <td>1</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Pinsir</td>
      <td>Bug</td>
      <td>NaN</td>
      <td>500</td>
      <td>65</td>
      <td>125</td>
      <td>100</td>
      <td>85</td>
      <td>1</td>
    </tr>
    <tr>
      <th>137</th>
      <td>PinsirMega Pinsir</td>
      <td>Bug</td>
      <td>Flying</td>
      <td>600</td>
      <td>65</td>
      <td>155</td>
      <td>120</td>
      <td>105</td>
      <td>1</td>
    </tr>
    <tr>
      <th>140</th>
      <td>Gyarados</td>
      <td>Water</td>
      <td>Flying</td>
      <td>540</td>
      <td>95</td>
      <td>125</td>
      <td>79</td>
      <td>81</td>
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
      <th>147</th>
      <td>Flareon</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>525</td>
      <td>65</td>
      <td>130</td>
      <td>60</td>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>152</th>
      <td>Kabutops</td>
      <td>Rock</td>
      <td>Water</td>
      <td>495</td>
      <td>60</td>
      <td>115</td>
      <td>105</td>
      <td>80</td>
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
      <th>155</th>
      <td>Snorlax</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>540</td>
      <td>160</td>
      <td>110</td>
      <td>65</td>
      <td>30</td>
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
      <th>689</th>
      <td>Braviary</td>
      <td>Normal</td>
      <td>Flying</td>
      <td>510</td>
      <td>100</td>
      <td>123</td>
      <td>75</td>
      <td>80</td>
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
      <th>696</th>
      <td>Hydreigon</td>
      <td>Dark</td>
      <td>Dragon</td>
      <td>600</td>
      <td>92</td>
      <td>105</td>
      <td>90</td>
      <td>98</td>
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
      <th>702</th>
      <td>TornadusIncarnate Forme</td>
      <td>Flying</td>
      <td>NaN</td>
      <td>580</td>
      <td>79</td>
      <td>115</td>
      <td>70</td>
      <td>111</td>
      <td>5</td>
    </tr>
    <tr>
      <th>704</th>
      <td>ThundurusIncarnate Forme</td>
      <td>Electric</td>
      <td>Flying</td>
      <td>580</td>
      <td>79</td>
      <td>115</td>
      <td>70</td>
      <td>111</td>
      <td>5</td>
    </tr>
    <tr>
      <th>705</th>
      <td>ThundurusTherian Forme</td>
      <td>Electric</td>
      <td>Flying</td>
      <td>580</td>
      <td>79</td>
      <td>105</td>
      <td>70</td>
      <td>101</td>
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
      <th>749</th>
      <td>Doublade</td>
      <td>Steel</td>
      <td>Ghost</td>
      <td>448</td>
      <td>59</td>
      <td>110</td>
      <td>150</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>750</th>
      <td>AegislashBlade Forme</td>
      <td>Steel</td>
      <td>Ghost</td>
      <td>520</td>
      <td>60</td>
      <td>150</td>
      <td>50</td>
      <td>60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>759</th>
      <td>Barbaracle</td>
      <td>Rock</td>
      <td>Water</td>
      <td>500</td>
      <td>72</td>
      <td>105</td>
      <td>115</td>
      <td>68</td>
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
<p>170 rows × 9 columns</p>
</div>



### Q2:选出攻击力大于100且防御力大于100的数据，并且列只要姓名、攻击力、防御力

#### 1.loc筛选


```python
df.loc[(df["攻击力"] > 100) & (df["防御力"] > 100),["姓名","攻击力","防御力"]]
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
      <th>攻击力</th>
      <th>防御力</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>130</td>
      <td>111</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BlastoiseMega Blastoise</td>
      <td>103</td>
      <td>120</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Golem</td>
      <td>120</td>
      <td>130</td>
    </tr>
    <tr>
      <th>107</th>
      <td>Kingler</td>
      <td>130</td>
      <td>115</td>
    </tr>
    <tr>
      <th>120</th>
      <td>Rhydon</td>
      <td>130</td>
      <td>120</td>
    </tr>
    <tr>
      <th>137</th>
      <td>PinsirMega Pinsir</td>
      <td>155</td>
      <td>120</td>
    </tr>
    <tr>
      <th>141</th>
      <td>GyaradosMega Gyarados</td>
      <td>155</td>
      <td>109</td>
    </tr>
    <tr>
      <th>152</th>
      <td>Kabutops</td>
      <td>115</td>
      <td>105</td>
    </tr>
    <tr>
      <th>224</th>
      <td>SteelixMega Steelix</td>
      <td>125</td>
      <td>230</td>
    </tr>
    <tr>
      <th>229</th>
      <td>ScizorMega Scizor</td>
      <td>150</td>
      <td>140</td>
    </tr>
    <tr>
      <th>232</th>
      <td>HeracrossMega Heracross</td>
      <td>185</td>
      <td>115</td>
    </tr>
    <tr>
      <th>251</th>
      <td>Donphan</td>
      <td>120</td>
      <td>120</td>
    </tr>
    <tr>
      <th>267</th>
      <td>Tyranitar</td>
      <td>134</td>
      <td>110</td>
    </tr>
    <tr>
      <th>268</th>
      <td>TyranitarMega Tyranitar</td>
      <td>164</td>
      <td>150</td>
    </tr>
    <tr>
      <th>283</th>
      <td>SwampertMega Swampert</td>
      <td>150</td>
      <td>110</td>
    </tr>
    <tr>
      <th>329</th>
      <td>MawileMega Mawile</td>
      <td>105</td>
      <td>125</td>
    </tr>
    <tr>
      <th>332</th>
      <td>Aggron</td>
      <td>110</td>
      <td>180</td>
    </tr>
    <tr>
      <th>333</th>
      <td>AggronMega Aggron</td>
      <td>140</td>
      <td>230</td>
    </tr>
    <tr>
      <th>366</th>
      <td>AltariaMega Altaria</td>
      <td>110</td>
      <td>110</td>
    </tr>
    <tr>
      <th>402</th>
      <td>Huntail</td>
      <td>104</td>
      <td>105</td>
    </tr>
    <tr>
      <th>409</th>
      <td>SalamenceMega Salamence</td>
      <td>145</td>
      <td>130</td>
    </tr>
    <tr>
      <th>412</th>
      <td>Metagross</td>
      <td>135</td>
      <td>130</td>
    </tr>
    <tr>
      <th>413</th>
      <td>MetagrossMega Metagross</td>
      <td>145</td>
      <td>150</td>
    </tr>
    <tr>
      <th>423</th>
      <td>Groudon</td>
      <td>150</td>
      <td>140</td>
    </tr>
    <tr>
      <th>424</th>
      <td>GroudonPrimal Groudon</td>
      <td>180</td>
      <td>160</td>
    </tr>
    <tr>
      <th>434</th>
      <td>Torterra</td>
      <td>109</td>
      <td>105</td>
    </tr>
    <tr>
      <th>494</th>
      <td>GarchompMega Garchomp</td>
      <td>170</td>
      <td>115</td>
    </tr>
    <tr>
      <th>500</th>
      <td>Hippowdon</td>
      <td>112</td>
      <td>118</td>
    </tr>
    <tr>
      <th>511</th>
      <td>AbomasnowMega Abomasnow</td>
      <td>132</td>
      <td>105</td>
    </tr>
    <tr>
      <th>515</th>
      <td>Rhyperior</td>
      <td>140</td>
      <td>130</td>
    </tr>
    <tr>
      <th>521</th>
      <td>Leafeon</td>
      <td>110</td>
      <td>130</td>
    </tr>
    <tr>
      <th>538</th>
      <td>Mesprit</td>
      <td>105</td>
      <td>105</td>
    </tr>
    <tr>
      <th>540</th>
      <td>Dialga</td>
      <td>120</td>
      <td>120</td>
    </tr>
    <tr>
      <th>543</th>
      <td>Regigigas</td>
      <td>160</td>
      <td>110</td>
    </tr>
    <tr>
      <th>552</th>
      <td>Arceus</td>
      <td>120</td>
      <td>120</td>
    </tr>
    <tr>
      <th>584</th>
      <td>Boldore</td>
      <td>105</td>
      <td>105</td>
    </tr>
    <tr>
      <th>585</th>
      <td>Gigalith</td>
      <td>135</td>
      <td>130</td>
    </tr>
    <tr>
      <th>626</th>
      <td>Carracosta</td>
      <td>108</td>
      <td>133</td>
    </tr>
    <tr>
      <th>650</th>
      <td>Escavalier</td>
      <td>135</td>
      <td>105</td>
    </tr>
    <tr>
      <th>693</th>
      <td>Durant</td>
      <td>109</td>
      <td>112</td>
    </tr>
    <tr>
      <th>707</th>
      <td>Zekrom</td>
      <td>150</td>
      <td>120</td>
    </tr>
    <tr>
      <th>720</th>
      <td>Chesnaught</td>
      <td>107</td>
      <td>122</td>
    </tr>
    <tr>
      <th>749</th>
      <td>Doublade</td>
      <td>110</td>
      <td>150</td>
    </tr>
    <tr>
      <th>759</th>
      <td>Barbaracle</td>
      <td>105</td>
      <td>115</td>
    </tr>
    <tr>
      <th>767</th>
      <td>Tyrantrum</td>
      <td>121</td>
      <td>119</td>
    </tr>
    <tr>
      <th>789</th>
      <td>Avalugg</td>
      <td>117</td>
      <td>184</td>
    </tr>
    <tr>
      <th>796</th>
      <td>DiancieMega Diancie</td>
      <td>160</td>
      <td>110</td>
    </tr>
    <tr>
      <th>799</th>
      <td>Volcanion</td>
      <td>110</td>
      <td>120</td>
    </tr>
  </tbody>
</table>
</div>



#### 2.query筛选


```python
df.query("攻击力 > 100 & 防御力 > 100")[["姓名","攻击力","防御力"]]
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
      <th>攻击力</th>
      <th>防御力</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>CharizardMega Charizard X</td>
      <td>130</td>
      <td>111</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BlastoiseMega Blastoise</td>
      <td>103</td>
      <td>120</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Golem</td>
      <td>120</td>
      <td>130</td>
    </tr>
    <tr>
      <th>107</th>
      <td>Kingler</td>
      <td>130</td>
      <td>115</td>
    </tr>
    <tr>
      <th>120</th>
      <td>Rhydon</td>
      <td>130</td>
      <td>120</td>
    </tr>
    <tr>
      <th>137</th>
      <td>PinsirMega Pinsir</td>
      <td>155</td>
      <td>120</td>
    </tr>
    <tr>
      <th>141</th>
      <td>GyaradosMega Gyarados</td>
      <td>155</td>
      <td>109</td>
    </tr>
    <tr>
      <th>152</th>
      <td>Kabutops</td>
      <td>115</td>
      <td>105</td>
    </tr>
    <tr>
      <th>224</th>
      <td>SteelixMega Steelix</td>
      <td>125</td>
      <td>230</td>
    </tr>
    <tr>
      <th>229</th>
      <td>ScizorMega Scizor</td>
      <td>150</td>
      <td>140</td>
    </tr>
    <tr>
      <th>232</th>
      <td>HeracrossMega Heracross</td>
      <td>185</td>
      <td>115</td>
    </tr>
    <tr>
      <th>251</th>
      <td>Donphan</td>
      <td>120</td>
      <td>120</td>
    </tr>
    <tr>
      <th>267</th>
      <td>Tyranitar</td>
      <td>134</td>
      <td>110</td>
    </tr>
    <tr>
      <th>268</th>
      <td>TyranitarMega Tyranitar</td>
      <td>164</td>
      <td>150</td>
    </tr>
    <tr>
      <th>283</th>
      <td>SwampertMega Swampert</td>
      <td>150</td>
      <td>110</td>
    </tr>
    <tr>
      <th>329</th>
      <td>MawileMega Mawile</td>
      <td>105</td>
      <td>125</td>
    </tr>
    <tr>
      <th>332</th>
      <td>Aggron</td>
      <td>110</td>
      <td>180</td>
    </tr>
    <tr>
      <th>333</th>
      <td>AggronMega Aggron</td>
      <td>140</td>
      <td>230</td>
    </tr>
    <tr>
      <th>366</th>
      <td>AltariaMega Altaria</td>
      <td>110</td>
      <td>110</td>
    </tr>
    <tr>
      <th>402</th>
      <td>Huntail</td>
      <td>104</td>
      <td>105</td>
    </tr>
    <tr>
      <th>409</th>
      <td>SalamenceMega Salamence</td>
      <td>145</td>
      <td>130</td>
    </tr>
    <tr>
      <th>412</th>
      <td>Metagross</td>
      <td>135</td>
      <td>130</td>
    </tr>
    <tr>
      <th>413</th>
      <td>MetagrossMega Metagross</td>
      <td>145</td>
      <td>150</td>
    </tr>
    <tr>
      <th>423</th>
      <td>Groudon</td>
      <td>150</td>
      <td>140</td>
    </tr>
    <tr>
      <th>424</th>
      <td>GroudonPrimal Groudon</td>
      <td>180</td>
      <td>160</td>
    </tr>
    <tr>
      <th>434</th>
      <td>Torterra</td>
      <td>109</td>
      <td>105</td>
    </tr>
    <tr>
      <th>494</th>
      <td>GarchompMega Garchomp</td>
      <td>170</td>
      <td>115</td>
    </tr>
    <tr>
      <th>500</th>
      <td>Hippowdon</td>
      <td>112</td>
      <td>118</td>
    </tr>
    <tr>
      <th>511</th>
      <td>AbomasnowMega Abomasnow</td>
      <td>132</td>
      <td>105</td>
    </tr>
    <tr>
      <th>515</th>
      <td>Rhyperior</td>
      <td>140</td>
      <td>130</td>
    </tr>
    <tr>
      <th>521</th>
      <td>Leafeon</td>
      <td>110</td>
      <td>130</td>
    </tr>
    <tr>
      <th>538</th>
      <td>Mesprit</td>
      <td>105</td>
      <td>105</td>
    </tr>
    <tr>
      <th>540</th>
      <td>Dialga</td>
      <td>120</td>
      <td>120</td>
    </tr>
    <tr>
      <th>543</th>
      <td>Regigigas</td>
      <td>160</td>
      <td>110</td>
    </tr>
    <tr>
      <th>552</th>
      <td>Arceus</td>
      <td>120</td>
      <td>120</td>
    </tr>
    <tr>
      <th>584</th>
      <td>Boldore</td>
      <td>105</td>
      <td>105</td>
    </tr>
    <tr>
      <th>585</th>
      <td>Gigalith</td>
      <td>135</td>
      <td>130</td>
    </tr>
    <tr>
      <th>626</th>
      <td>Carracosta</td>
      <td>108</td>
      <td>133</td>
    </tr>
    <tr>
      <th>650</th>
      <td>Escavalier</td>
      <td>135</td>
      <td>105</td>
    </tr>
    <tr>
      <th>693</th>
      <td>Durant</td>
      <td>109</td>
      <td>112</td>
    </tr>
    <tr>
      <th>707</th>
      <td>Zekrom</td>
      <td>150</td>
      <td>120</td>
    </tr>
    <tr>
      <th>720</th>
      <td>Chesnaught</td>
      <td>107</td>
      <td>122</td>
    </tr>
    <tr>
      <th>749</th>
      <td>Doublade</td>
      <td>110</td>
      <td>150</td>
    </tr>
    <tr>
      <th>759</th>
      <td>Barbaracle</td>
      <td>105</td>
      <td>115</td>
    </tr>
    <tr>
      <th>767</th>
      <td>Tyrantrum</td>
      <td>121</td>
      <td>119</td>
    </tr>
    <tr>
      <th>789</th>
      <td>Avalugg</td>
      <td>117</td>
      <td>184</td>
    </tr>
    <tr>
      <th>796</th>
      <td>DiancieMega Diancie</td>
      <td>160</td>
      <td>110</td>
    </tr>
    <tr>
      <th>799</th>
      <td>Volcanion</td>
      <td>110</td>
      <td>120</td>
    </tr>
  </tbody>
</table>
</div>



但是query的参数中不能引用变量，而loc可以

#### Q3：选出类型1为Grass的所有数据

#### 1.loc筛选


```python
#做法1
df.loc[df["类型1"] == "Grass"]
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
      <th>75</th>
      <td>Bellsprout</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>300</td>
      <td>50</td>
      <td>75</td>
      <td>35</td>
      <td>40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Weepinbell</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>390</td>
      <td>65</td>
      <td>90</td>
      <td>50</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Victreebel</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>490</td>
      <td>80</td>
      <td>105</td>
      <td>65</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Exeggcute</td>
      <td>Grass</td>
      <td>Psychic</td>
      <td>325</td>
      <td>60</td>
      <td>40</td>
      <td>80</td>
      <td>40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Exeggutor</td>
      <td>Grass</td>
      <td>Psychic</td>
      <td>520</td>
      <td>95</td>
      <td>95</td>
      <td>85</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>122</th>
      <td>Tangela</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>435</td>
      <td>65</td>
      <td>55</td>
      <td>115</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>166</th>
      <td>Chikorita</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>65</td>
      <td>45</td>
      <td>2</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Bayleef</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>80</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>168</th>
      <td>Meganium</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>100</td>
      <td>80</td>
      <td>2</td>
    </tr>
    <tr>
      <th>197</th>
      <td>Bellossom</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>490</td>
      <td>75</td>
      <td>80</td>
      <td>95</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>202</th>
      <td>Hoppip</td>
      <td>Grass</td>
      <td>Flying</td>
      <td>250</td>
      <td>35</td>
      <td>35</td>
      <td>40</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>203</th>
      <td>Skiploom</td>
      <td>Grass</td>
      <td>Flying</td>
      <td>340</td>
      <td>55</td>
      <td>45</td>
      <td>50</td>
      <td>80</td>
      <td>2</td>
    </tr>
    <tr>
      <th>204</th>
      <td>Jumpluff</td>
      <td>Grass</td>
      <td>Flying</td>
      <td>460</td>
      <td>75</td>
      <td>55</td>
      <td>70</td>
      <td>110</td>
      <td>2</td>
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
    <tr>
      <th>207</th>
      <td>Sunflora</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>425</td>
      <td>75</td>
      <td>75</td>
      <td>55</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>272</th>
      <td>Treecko</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>310</td>
      <td>40</td>
      <td>45</td>
      <td>35</td>
      <td>70</td>
      <td>3</td>
    </tr>
    <tr>
      <th>273</th>
      <td>Grovyle</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>405</td>
      <td>50</td>
      <td>65</td>
      <td>45</td>
      <td>95</td>
      <td>3</td>
    </tr>
    <tr>
      <th>274</th>
      <td>Sceptile</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>530</td>
      <td>70</td>
      <td>85</td>
      <td>65</td>
      <td>120</td>
      <td>3</td>
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
      <th>297</th>
      <td>Nuzleaf</td>
      <td>Grass</td>
      <td>Dark</td>
      <td>340</td>
      <td>70</td>
      <td>70</td>
      <td>40</td>
      <td>60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>298</th>
      <td>Shiftry</td>
      <td>Grass</td>
      <td>Dark</td>
      <td>480</td>
      <td>90</td>
      <td>100</td>
      <td>60</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>309</th>
      <td>Shroomish</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>295</td>
      <td>60</td>
      <td>40</td>
      <td>60</td>
      <td>35</td>
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
      <th>467</th>
      <td>Cherubi</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>275</td>
      <td>45</td>
      <td>35</td>
      <td>45</td>
      <td>35</td>
      <td>4</td>
    </tr>
    <tr>
      <th>468</th>
      <td>Cherrim</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>450</td>
      <td>70</td>
      <td>60</td>
      <td>70</td>
      <td>85</td>
      <td>4</td>
    </tr>
    <tr>
      <th>505</th>
      <td>Carnivine</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>454</td>
      <td>74</td>
      <td>100</td>
      <td>72</td>
      <td>46</td>
      <td>4</td>
    </tr>
    <tr>
      <th>509</th>
      <td>Snover</td>
      <td>Grass</td>
      <td>Ice</td>
      <td>334</td>
      <td>60</td>
      <td>62</td>
      <td>50</td>
      <td>40</td>
      <td>4</td>
    </tr>
    <tr>
      <th>510</th>
      <td>Abomasnow</td>
      <td>Grass</td>
      <td>Ice</td>
      <td>494</td>
      <td>90</td>
      <td>92</td>
      <td>75</td>
      <td>60</td>
      <td>4</td>
    </tr>
    <tr>
      <th>511</th>
      <td>AbomasnowMega Abomasnow</td>
      <td>Grass</td>
      <td>Ice</td>
      <td>594</td>
      <td>90</td>
      <td>132</td>
      <td>105</td>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>516</th>
      <td>Tangrowth</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>535</td>
      <td>100</td>
      <td>100</td>
      <td>125</td>
      <td>50</td>
      <td>4</td>
    </tr>
    <tr>
      <th>521</th>
      <td>Leafeon</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>525</td>
      <td>65</td>
      <td>110</td>
      <td>130</td>
      <td>95</td>
      <td>4</td>
    </tr>
    <tr>
      <th>550</th>
      <td>ShayminLand Forme</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>600</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>551</th>
      <td>ShayminSky Forme</td>
      <td>Grass</td>
      <td>Flying</td>
      <td>600</td>
      <td>100</td>
      <td>103</td>
      <td>75</td>
      <td>127</td>
      <td>4</td>
    </tr>
    <tr>
      <th>554</th>
      <td>Snivy</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>308</td>
      <td>45</td>
      <td>45</td>
      <td>55</td>
      <td>63</td>
      <td>5</td>
    </tr>
    <tr>
      <th>555</th>
      <td>Servine</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>413</td>
      <td>60</td>
      <td>60</td>
      <td>75</td>
      <td>83</td>
      <td>5</td>
    </tr>
    <tr>
      <th>556</th>
      <td>Serperior</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>528</td>
      <td>75</td>
      <td>75</td>
      <td>95</td>
      <td>113</td>
      <td>5</td>
    </tr>
    <tr>
      <th>570</th>
      <td>Pansage</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>316</td>
      <td>50</td>
      <td>53</td>
      <td>48</td>
      <td>64</td>
      <td>5</td>
    </tr>
    <tr>
      <th>571</th>
      <td>Simisage</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>498</td>
      <td>75</td>
      <td>98</td>
      <td>63</td>
      <td>101</td>
      <td>5</td>
    </tr>
    <tr>
      <th>606</th>
      <td>Cottonee</td>
      <td>Grass</td>
      <td>Fairy</td>
      <td>280</td>
      <td>40</td>
      <td>27</td>
      <td>60</td>
      <td>66</td>
      <td>5</td>
    </tr>
    <tr>
      <th>607</th>
      <td>Whimsicott</td>
      <td>Grass</td>
      <td>Fairy</td>
      <td>480</td>
      <td>60</td>
      <td>67</td>
      <td>85</td>
      <td>116</td>
      <td>5</td>
    </tr>
    <tr>
      <th>608</th>
      <td>Petilil</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>280</td>
      <td>45</td>
      <td>35</td>
      <td>50</td>
      <td>30</td>
      <td>5</td>
    </tr>
    <tr>
      <th>609</th>
      <td>Lilligant</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>480</td>
      <td>70</td>
      <td>60</td>
      <td>75</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>617</th>
      <td>Maractus</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>461</td>
      <td>75</td>
      <td>86</td>
      <td>67</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>651</th>
      <td>Foongus</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>294</td>
      <td>69</td>
      <td>55</td>
      <td>45</td>
      <td>15</td>
      <td>5</td>
    </tr>
    <tr>
      <th>652</th>
      <td>Amoonguss</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>464</td>
      <td>114</td>
      <td>85</td>
      <td>70</td>
      <td>30</td>
      <td>5</td>
    </tr>
    <tr>
      <th>658</th>
      <td>Ferroseed</td>
      <td>Grass</td>
      <td>Steel</td>
      <td>305</td>
      <td>44</td>
      <td>50</td>
      <td>91</td>
      <td>10</td>
      <td>5</td>
    </tr>
    <tr>
      <th>659</th>
      <td>Ferrothorn</td>
      <td>Grass</td>
      <td>Steel</td>
      <td>489</td>
      <td>74</td>
      <td>94</td>
      <td>131</td>
      <td>20</td>
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
      <th>718</th>
      <td>Chespin</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>313</td>
      <td>56</td>
      <td>61</td>
      <td>65</td>
      <td>38</td>
      <td>6</td>
    </tr>
    <tr>
      <th>719</th>
      <td>Quilladin</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>405</td>
      <td>61</td>
      <td>78</td>
      <td>95</td>
      <td>57</td>
      <td>6</td>
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
      <th>740</th>
      <td>Skiddo</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>350</td>
      <td>66</td>
      <td>65</td>
      <td>48</td>
      <td>52</td>
      <td>6</td>
    </tr>
    <tr>
      <th>741</th>
      <td>Gogoat</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>531</td>
      <td>123</td>
      <td>100</td>
      <td>62</td>
      <td>68</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>70 rows × 9 columns</p>
</div>




```python
#做法2
df.loc[df["类型1"].isin(["Grass"])]
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
      <th>75</th>
      <td>Bellsprout</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>300</td>
      <td>50</td>
      <td>75</td>
      <td>35</td>
      <td>40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Weepinbell</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>390</td>
      <td>65</td>
      <td>90</td>
      <td>50</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Victreebel</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>490</td>
      <td>80</td>
      <td>105</td>
      <td>65</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Exeggcute</td>
      <td>Grass</td>
      <td>Psychic</td>
      <td>325</td>
      <td>60</td>
      <td>40</td>
      <td>80</td>
      <td>40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Exeggutor</td>
      <td>Grass</td>
      <td>Psychic</td>
      <td>520</td>
      <td>95</td>
      <td>95</td>
      <td>85</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>122</th>
      <td>Tangela</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>435</td>
      <td>65</td>
      <td>55</td>
      <td>115</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>166</th>
      <td>Chikorita</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>65</td>
      <td>45</td>
      <td>2</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Bayleef</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>80</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>168</th>
      <td>Meganium</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>100</td>
      <td>80</td>
      <td>2</td>
    </tr>
    <tr>
      <th>197</th>
      <td>Bellossom</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>490</td>
      <td>75</td>
      <td>80</td>
      <td>95</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>202</th>
      <td>Hoppip</td>
      <td>Grass</td>
      <td>Flying</td>
      <td>250</td>
      <td>35</td>
      <td>35</td>
      <td>40</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>203</th>
      <td>Skiploom</td>
      <td>Grass</td>
      <td>Flying</td>
      <td>340</td>
      <td>55</td>
      <td>45</td>
      <td>50</td>
      <td>80</td>
      <td>2</td>
    </tr>
    <tr>
      <th>204</th>
      <td>Jumpluff</td>
      <td>Grass</td>
      <td>Flying</td>
      <td>460</td>
      <td>75</td>
      <td>55</td>
      <td>70</td>
      <td>110</td>
      <td>2</td>
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
    <tr>
      <th>207</th>
      <td>Sunflora</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>425</td>
      <td>75</td>
      <td>75</td>
      <td>55</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>272</th>
      <td>Treecko</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>310</td>
      <td>40</td>
      <td>45</td>
      <td>35</td>
      <td>70</td>
      <td>3</td>
    </tr>
    <tr>
      <th>273</th>
      <td>Grovyle</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>405</td>
      <td>50</td>
      <td>65</td>
      <td>45</td>
      <td>95</td>
      <td>3</td>
    </tr>
    <tr>
      <th>274</th>
      <td>Sceptile</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>530</td>
      <td>70</td>
      <td>85</td>
      <td>65</td>
      <td>120</td>
      <td>3</td>
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
      <th>297</th>
      <td>Nuzleaf</td>
      <td>Grass</td>
      <td>Dark</td>
      <td>340</td>
      <td>70</td>
      <td>70</td>
      <td>40</td>
      <td>60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>298</th>
      <td>Shiftry</td>
      <td>Grass</td>
      <td>Dark</td>
      <td>480</td>
      <td>90</td>
      <td>100</td>
      <td>60</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>309</th>
      <td>Shroomish</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>295</td>
      <td>60</td>
      <td>40</td>
      <td>60</td>
      <td>35</td>
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
      <th>467</th>
      <td>Cherubi</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>275</td>
      <td>45</td>
      <td>35</td>
      <td>45</td>
      <td>35</td>
      <td>4</td>
    </tr>
    <tr>
      <th>468</th>
      <td>Cherrim</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>450</td>
      <td>70</td>
      <td>60</td>
      <td>70</td>
      <td>85</td>
      <td>4</td>
    </tr>
    <tr>
      <th>505</th>
      <td>Carnivine</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>454</td>
      <td>74</td>
      <td>100</td>
      <td>72</td>
      <td>46</td>
      <td>4</td>
    </tr>
    <tr>
      <th>509</th>
      <td>Snover</td>
      <td>Grass</td>
      <td>Ice</td>
      <td>334</td>
      <td>60</td>
      <td>62</td>
      <td>50</td>
      <td>40</td>
      <td>4</td>
    </tr>
    <tr>
      <th>510</th>
      <td>Abomasnow</td>
      <td>Grass</td>
      <td>Ice</td>
      <td>494</td>
      <td>90</td>
      <td>92</td>
      <td>75</td>
      <td>60</td>
      <td>4</td>
    </tr>
    <tr>
      <th>511</th>
      <td>AbomasnowMega Abomasnow</td>
      <td>Grass</td>
      <td>Ice</td>
      <td>594</td>
      <td>90</td>
      <td>132</td>
      <td>105</td>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>516</th>
      <td>Tangrowth</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>535</td>
      <td>100</td>
      <td>100</td>
      <td>125</td>
      <td>50</td>
      <td>4</td>
    </tr>
    <tr>
      <th>521</th>
      <td>Leafeon</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>525</td>
      <td>65</td>
      <td>110</td>
      <td>130</td>
      <td>95</td>
      <td>4</td>
    </tr>
    <tr>
      <th>550</th>
      <td>ShayminLand Forme</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>600</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>551</th>
      <td>ShayminSky Forme</td>
      <td>Grass</td>
      <td>Flying</td>
      <td>600</td>
      <td>100</td>
      <td>103</td>
      <td>75</td>
      <td>127</td>
      <td>4</td>
    </tr>
    <tr>
      <th>554</th>
      <td>Snivy</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>308</td>
      <td>45</td>
      <td>45</td>
      <td>55</td>
      <td>63</td>
      <td>5</td>
    </tr>
    <tr>
      <th>555</th>
      <td>Servine</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>413</td>
      <td>60</td>
      <td>60</td>
      <td>75</td>
      <td>83</td>
      <td>5</td>
    </tr>
    <tr>
      <th>556</th>
      <td>Serperior</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>528</td>
      <td>75</td>
      <td>75</td>
      <td>95</td>
      <td>113</td>
      <td>5</td>
    </tr>
    <tr>
      <th>570</th>
      <td>Pansage</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>316</td>
      <td>50</td>
      <td>53</td>
      <td>48</td>
      <td>64</td>
      <td>5</td>
    </tr>
    <tr>
      <th>571</th>
      <td>Simisage</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>498</td>
      <td>75</td>
      <td>98</td>
      <td>63</td>
      <td>101</td>
      <td>5</td>
    </tr>
    <tr>
      <th>606</th>
      <td>Cottonee</td>
      <td>Grass</td>
      <td>Fairy</td>
      <td>280</td>
      <td>40</td>
      <td>27</td>
      <td>60</td>
      <td>66</td>
      <td>5</td>
    </tr>
    <tr>
      <th>607</th>
      <td>Whimsicott</td>
      <td>Grass</td>
      <td>Fairy</td>
      <td>480</td>
      <td>60</td>
      <td>67</td>
      <td>85</td>
      <td>116</td>
      <td>5</td>
    </tr>
    <tr>
      <th>608</th>
      <td>Petilil</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>280</td>
      <td>45</td>
      <td>35</td>
      <td>50</td>
      <td>30</td>
      <td>5</td>
    </tr>
    <tr>
      <th>609</th>
      <td>Lilligant</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>480</td>
      <td>70</td>
      <td>60</td>
      <td>75</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>617</th>
      <td>Maractus</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>461</td>
      <td>75</td>
      <td>86</td>
      <td>67</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>651</th>
      <td>Foongus</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>294</td>
      <td>69</td>
      <td>55</td>
      <td>45</td>
      <td>15</td>
      <td>5</td>
    </tr>
    <tr>
      <th>652</th>
      <td>Amoonguss</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>464</td>
      <td>114</td>
      <td>85</td>
      <td>70</td>
      <td>30</td>
      <td>5</td>
    </tr>
    <tr>
      <th>658</th>
      <td>Ferroseed</td>
      <td>Grass</td>
      <td>Steel</td>
      <td>305</td>
      <td>44</td>
      <td>50</td>
      <td>91</td>
      <td>10</td>
      <td>5</td>
    </tr>
    <tr>
      <th>659</th>
      <td>Ferrothorn</td>
      <td>Grass</td>
      <td>Steel</td>
      <td>489</td>
      <td>74</td>
      <td>94</td>
      <td>131</td>
      <td>20</td>
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
      <th>718</th>
      <td>Chespin</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>313</td>
      <td>56</td>
      <td>61</td>
      <td>65</td>
      <td>38</td>
      <td>6</td>
    </tr>
    <tr>
      <th>719</th>
      <td>Quilladin</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>405</td>
      <td>61</td>
      <td>78</td>
      <td>95</td>
      <td>57</td>
      <td>6</td>
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
      <th>740</th>
      <td>Skiddo</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>350</td>
      <td>66</td>
      <td>65</td>
      <td>48</td>
      <td>52</td>
      <td>6</td>
    </tr>
    <tr>
      <th>741</th>
      <td>Gogoat</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>531</td>
      <td>123</td>
      <td>100</td>
      <td>62</td>
      <td>68</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>70 rows × 9 columns</p>
</div>



#### 2.query筛选


```python
df.query("类型1 == 'Grass'")
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
      <th>75</th>
      <td>Bellsprout</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>300</td>
      <td>50</td>
      <td>75</td>
      <td>35</td>
      <td>40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Weepinbell</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>390</td>
      <td>65</td>
      <td>90</td>
      <td>50</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Victreebel</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>490</td>
      <td>80</td>
      <td>105</td>
      <td>65</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Exeggcute</td>
      <td>Grass</td>
      <td>Psychic</td>
      <td>325</td>
      <td>60</td>
      <td>40</td>
      <td>80</td>
      <td>40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Exeggutor</td>
      <td>Grass</td>
      <td>Psychic</td>
      <td>520</td>
      <td>95</td>
      <td>95</td>
      <td>85</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>122</th>
      <td>Tangela</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>435</td>
      <td>65</td>
      <td>55</td>
      <td>115</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>166</th>
      <td>Chikorita</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>65</td>
      <td>45</td>
      <td>2</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Bayleef</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>80</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>168</th>
      <td>Meganium</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>100</td>
      <td>80</td>
      <td>2</td>
    </tr>
    <tr>
      <th>197</th>
      <td>Bellossom</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>490</td>
      <td>75</td>
      <td>80</td>
      <td>95</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>202</th>
      <td>Hoppip</td>
      <td>Grass</td>
      <td>Flying</td>
      <td>250</td>
      <td>35</td>
      <td>35</td>
      <td>40</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>203</th>
      <td>Skiploom</td>
      <td>Grass</td>
      <td>Flying</td>
      <td>340</td>
      <td>55</td>
      <td>45</td>
      <td>50</td>
      <td>80</td>
      <td>2</td>
    </tr>
    <tr>
      <th>204</th>
      <td>Jumpluff</td>
      <td>Grass</td>
      <td>Flying</td>
      <td>460</td>
      <td>75</td>
      <td>55</td>
      <td>70</td>
      <td>110</td>
      <td>2</td>
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
    <tr>
      <th>207</th>
      <td>Sunflora</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>425</td>
      <td>75</td>
      <td>75</td>
      <td>55</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>272</th>
      <td>Treecko</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>310</td>
      <td>40</td>
      <td>45</td>
      <td>35</td>
      <td>70</td>
      <td>3</td>
    </tr>
    <tr>
      <th>273</th>
      <td>Grovyle</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>405</td>
      <td>50</td>
      <td>65</td>
      <td>45</td>
      <td>95</td>
      <td>3</td>
    </tr>
    <tr>
      <th>274</th>
      <td>Sceptile</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>530</td>
      <td>70</td>
      <td>85</td>
      <td>65</td>
      <td>120</td>
      <td>3</td>
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
      <th>297</th>
      <td>Nuzleaf</td>
      <td>Grass</td>
      <td>Dark</td>
      <td>340</td>
      <td>70</td>
      <td>70</td>
      <td>40</td>
      <td>60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>298</th>
      <td>Shiftry</td>
      <td>Grass</td>
      <td>Dark</td>
      <td>480</td>
      <td>90</td>
      <td>100</td>
      <td>60</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>309</th>
      <td>Shroomish</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>295</td>
      <td>60</td>
      <td>40</td>
      <td>60</td>
      <td>35</td>
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
      <th>467</th>
      <td>Cherubi</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>275</td>
      <td>45</td>
      <td>35</td>
      <td>45</td>
      <td>35</td>
      <td>4</td>
    </tr>
    <tr>
      <th>468</th>
      <td>Cherrim</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>450</td>
      <td>70</td>
      <td>60</td>
      <td>70</td>
      <td>85</td>
      <td>4</td>
    </tr>
    <tr>
      <th>505</th>
      <td>Carnivine</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>454</td>
      <td>74</td>
      <td>100</td>
      <td>72</td>
      <td>46</td>
      <td>4</td>
    </tr>
    <tr>
      <th>509</th>
      <td>Snover</td>
      <td>Grass</td>
      <td>Ice</td>
      <td>334</td>
      <td>60</td>
      <td>62</td>
      <td>50</td>
      <td>40</td>
      <td>4</td>
    </tr>
    <tr>
      <th>510</th>
      <td>Abomasnow</td>
      <td>Grass</td>
      <td>Ice</td>
      <td>494</td>
      <td>90</td>
      <td>92</td>
      <td>75</td>
      <td>60</td>
      <td>4</td>
    </tr>
    <tr>
      <th>511</th>
      <td>AbomasnowMega Abomasnow</td>
      <td>Grass</td>
      <td>Ice</td>
      <td>594</td>
      <td>90</td>
      <td>132</td>
      <td>105</td>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>516</th>
      <td>Tangrowth</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>535</td>
      <td>100</td>
      <td>100</td>
      <td>125</td>
      <td>50</td>
      <td>4</td>
    </tr>
    <tr>
      <th>521</th>
      <td>Leafeon</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>525</td>
      <td>65</td>
      <td>110</td>
      <td>130</td>
      <td>95</td>
      <td>4</td>
    </tr>
    <tr>
      <th>550</th>
      <td>ShayminLand Forme</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>600</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>551</th>
      <td>ShayminSky Forme</td>
      <td>Grass</td>
      <td>Flying</td>
      <td>600</td>
      <td>100</td>
      <td>103</td>
      <td>75</td>
      <td>127</td>
      <td>4</td>
    </tr>
    <tr>
      <th>554</th>
      <td>Snivy</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>308</td>
      <td>45</td>
      <td>45</td>
      <td>55</td>
      <td>63</td>
      <td>5</td>
    </tr>
    <tr>
      <th>555</th>
      <td>Servine</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>413</td>
      <td>60</td>
      <td>60</td>
      <td>75</td>
      <td>83</td>
      <td>5</td>
    </tr>
    <tr>
      <th>556</th>
      <td>Serperior</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>528</td>
      <td>75</td>
      <td>75</td>
      <td>95</td>
      <td>113</td>
      <td>5</td>
    </tr>
    <tr>
      <th>570</th>
      <td>Pansage</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>316</td>
      <td>50</td>
      <td>53</td>
      <td>48</td>
      <td>64</td>
      <td>5</td>
    </tr>
    <tr>
      <th>571</th>
      <td>Simisage</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>498</td>
      <td>75</td>
      <td>98</td>
      <td>63</td>
      <td>101</td>
      <td>5</td>
    </tr>
    <tr>
      <th>606</th>
      <td>Cottonee</td>
      <td>Grass</td>
      <td>Fairy</td>
      <td>280</td>
      <td>40</td>
      <td>27</td>
      <td>60</td>
      <td>66</td>
      <td>5</td>
    </tr>
    <tr>
      <th>607</th>
      <td>Whimsicott</td>
      <td>Grass</td>
      <td>Fairy</td>
      <td>480</td>
      <td>60</td>
      <td>67</td>
      <td>85</td>
      <td>116</td>
      <td>5</td>
    </tr>
    <tr>
      <th>608</th>
      <td>Petilil</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>280</td>
      <td>45</td>
      <td>35</td>
      <td>50</td>
      <td>30</td>
      <td>5</td>
    </tr>
    <tr>
      <th>609</th>
      <td>Lilligant</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>480</td>
      <td>70</td>
      <td>60</td>
      <td>75</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>617</th>
      <td>Maractus</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>461</td>
      <td>75</td>
      <td>86</td>
      <td>67</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>651</th>
      <td>Foongus</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>294</td>
      <td>69</td>
      <td>55</td>
      <td>45</td>
      <td>15</td>
      <td>5</td>
    </tr>
    <tr>
      <th>652</th>
      <td>Amoonguss</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>464</td>
      <td>114</td>
      <td>85</td>
      <td>70</td>
      <td>30</td>
      <td>5</td>
    </tr>
    <tr>
      <th>658</th>
      <td>Ferroseed</td>
      <td>Grass</td>
      <td>Steel</td>
      <td>305</td>
      <td>44</td>
      <td>50</td>
      <td>91</td>
      <td>10</td>
      <td>5</td>
    </tr>
    <tr>
      <th>659</th>
      <td>Ferrothorn</td>
      <td>Grass</td>
      <td>Steel</td>
      <td>489</td>
      <td>74</td>
      <td>94</td>
      <td>131</td>
      <td>20</td>
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
      <th>718</th>
      <td>Chespin</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>313</td>
      <td>56</td>
      <td>61</td>
      <td>65</td>
      <td>38</td>
      <td>6</td>
    </tr>
    <tr>
      <th>719</th>
      <td>Quilladin</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>405</td>
      <td>61</td>
      <td>78</td>
      <td>95</td>
      <td>57</td>
      <td>6</td>
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
      <th>740</th>
      <td>Skiddo</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>350</td>
      <td>66</td>
      <td>65</td>
      <td>48</td>
      <td>52</td>
      <td>6</td>
    </tr>
    <tr>
      <th>741</th>
      <td>Gogoat</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>531</td>
      <td>123</td>
      <td>100</td>
      <td>62</td>
      <td>68</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>70 rows × 9 columns</p>
</div>



## 三.多重索引

#### Q1：什么是多重索引

2层或2层以上的索引

为什么会用到多重索引呢？
因为有时候需要通过多个维度来查看数据

#### Q2:如何创建多重索引


```python
#当我们要以字符串列作为索引列时，要保证这列为字符串格式
df[["类型1","类型2"]] = df[["类型1","类型2"]].astype("str")
```


```python
#创建一个具有2重索引的数据作示例
df_pokemon = df.set_index(["类型1","类型2"])
df_pokemon
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
      <th rowspan="5" valign="top">Fire</th>
      <th>nan</th>
      <td>Charmander</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Charmeleon</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Charizard</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>CharizardMega Charizard X</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>CharizardMega Charizard Y</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Water</th>
      <th>nan</th>
      <td>Squirtle</td>
      <td>314</td>
      <td>44</td>
      <td>48</td>
      <td>65</td>
      <td>43</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Wartortle</td>
      <td>405</td>
      <td>59</td>
      <td>63</td>
      <td>80</td>
      <td>58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Blastoise</td>
      <td>530</td>
      <td>79</td>
      <td>83</td>
      <td>100</td>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>BlastoiseMega Blastoise</td>
      <td>630</td>
      <td>79</td>
      <td>103</td>
      <td>120</td>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Bug</th>
      <th>nan</th>
      <td>Caterpie</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Metapod</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Butterfree</td>
      <td>395</td>
      <td>60</td>
      <td>45</td>
      <td>50</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Weedle</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Kakuna</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Beedrill</td>
      <td>395</td>
      <td>65</td>
      <td>90</td>
      <td>40</td>
      <td>75</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>BeedrillMega Beedrill</td>
      <td>495</td>
      <td>65</td>
      <td>150</td>
      <td>40</td>
      <td>145</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Normal</th>
      <th>Flying</th>
      <td>Pidgey</td>
      <td>251</td>
      <td>40</td>
      <td>45</td>
      <td>40</td>
      <td>56</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Pidgeotto</td>
      <td>349</td>
      <td>63</td>
      <td>60</td>
      <td>55</td>
      <td>71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Pidgeot</td>
      <td>479</td>
      <td>83</td>
      <td>80</td>
      <td>75</td>
      <td>101</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>PidgeotMega Pidgeot</td>
      <td>579</td>
      <td>83</td>
      <td>80</td>
      <td>80</td>
      <td>121</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Rattata</td>
      <td>253</td>
      <td>30</td>
      <td>56</td>
      <td>35</td>
      <td>72</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Raticate</td>
      <td>413</td>
      <td>55</td>
      <td>81</td>
      <td>60</td>
      <td>97</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Spearow</td>
      <td>262</td>
      <td>40</td>
      <td>60</td>
      <td>30</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Fearow</td>
      <td>442</td>
      <td>65</td>
      <td>90</td>
      <td>65</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Poison</th>
      <th>nan</th>
      <td>Ekans</td>
      <td>288</td>
      <td>35</td>
      <td>60</td>
      <td>44</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Arbok</td>
      <td>438</td>
      <td>60</td>
      <td>85</td>
      <td>69</td>
      <td>80</td>
      <td>1</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <th>nan</th>
      <td>Sylveon</td>
      <td>525</td>
      <td>95</td>
      <td>65</td>
      <td>65</td>
      <td>60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <th>Flying</th>
      <td>Hawlucha</td>
      <td>500</td>
      <td>78</td>
      <td>92</td>
      <td>75</td>
      <td>118</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Electric</th>
      <th>Fairy</th>
      <td>Dedenne</td>
      <td>431</td>
      <td>67</td>
      <td>58</td>
      <td>57</td>
      <td>101</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Rock</th>
      <th>Fairy</th>
      <td>Carbink</td>
      <td>500</td>
      <td>50</td>
      <td>50</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Dragon</th>
      <th>nan</th>
      <td>Goomy</td>
      <td>300</td>
      <td>45</td>
      <td>50</td>
      <td>35</td>
      <td>40</td>
      <td>6</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Sliggoo</td>
      <td>452</td>
      <td>68</td>
      <td>75</td>
      <td>53</td>
      <td>60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Goodra</td>
      <td>600</td>
      <td>90</td>
      <td>100</td>
      <td>70</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Steel</th>
      <th>Fairy</th>
      <td>Klefki</td>
      <td>470</td>
      <td>57</td>
      <td>80</td>
      <td>91</td>
      <td>75</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">Ghost</th>
      <th>Grass</th>
      <td>Phantump</td>
      <td>309</td>
      <td>43</td>
      <td>70</td>
      <td>48</td>
      <td>38</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Trevenant</td>
      <td>474</td>
      <td>85</td>
      <td>110</td>
      <td>76</td>
      <td>56</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>PumpkabooAverage Size</td>
      <td>335</td>
      <td>49</td>
      <td>66</td>
      <td>70</td>
      <td>51</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>PumpkabooSmall Size</td>
      <td>335</td>
      <td>44</td>
      <td>66</td>
      <td>70</td>
      <td>56</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>PumpkabooLarge Size</td>
      <td>335</td>
      <td>54</td>
      <td>66</td>
      <td>70</td>
      <td>46</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>PumpkabooSuper Size</td>
      <td>335</td>
      <td>59</td>
      <td>66</td>
      <td>70</td>
      <td>41</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>GourgeistAverage Size</td>
      <td>494</td>
      <td>65</td>
      <td>90</td>
      <td>122</td>
      <td>84</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>GourgeistSmall Size</td>
      <td>494</td>
      <td>55</td>
      <td>85</td>
      <td>122</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>GourgeistLarge Size</td>
      <td>494</td>
      <td>75</td>
      <td>95</td>
      <td>122</td>
      <td>69</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>GourgeistSuper Size</td>
      <td>494</td>
      <td>85</td>
      <td>100</td>
      <td>122</td>
      <td>54</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Ice</th>
      <th>nan</th>
      <td>Bergmite</td>
      <td>304</td>
      <td>55</td>
      <td>69</td>
      <td>85</td>
      <td>28</td>
      <td>6</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Avalugg</td>
      <td>514</td>
      <td>95</td>
      <td>117</td>
      <td>184</td>
      <td>28</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Flying</th>
      <th>Dragon</th>
      <td>Noibat</td>
      <td>245</td>
      <td>40</td>
      <td>30</td>
      <td>35</td>
      <td>55</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>Noivern</td>
      <td>535</td>
      <td>85</td>
      <td>70</td>
      <td>80</td>
      <td>123</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <th>nan</th>
      <td>Xerneas</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Dark</th>
      <th>Flying</th>
      <td>Yveltal</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <th>Ground</th>
      <td>Zygarde50% Forme</td>
      <td>600</td>
      <td>108</td>
      <td>100</td>
      <td>121</td>
      <td>95</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Rock</th>
      <th>Fairy</th>
      <td>Diancie</td>
      <td>600</td>
      <td>50</td>
      <td>100</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>DiancieMega Diancie</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Psychic</th>
      <th>Ghost</th>
      <td>HoopaHoopa Confined</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
      <td>60</td>
      <td>70</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>HoopaHoopa Unbound</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
      <td>60</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fire</th>
      <th>Water</th>
      <td>Volcanion</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
      <td>120</td>
      <td>70</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 7 columns</p>
</div>



参数介绍

drop：是指该列被指定为索引后，是否删除该列，默认为True，即删除该列。如果改成False，则多重索引在数据集的列中也会保留

append：指定是否保留原索引，默认为False，即不保留，如果改成True，则保留原索引

inplace：指是否在源数据的基础上修改，默认为False，即不修改，返回一个新的数据框，如果改成True，则直接在源数据上修改

level介绍


```python
#获取第一层索引
df_pokemon.index.get_level_values(0)
```




    Index(['Grass', 'Grass', 'Grass', 'Grass', 'Fire', 'Fire', 'Fire', 'Fire',
           'Fire', 'Water',
           ...
           'Flying', 'Flying', 'Fairy', 'Dark', 'Dragon', 'Rock', 'Rock',
           'Psychic', 'Psychic', 'Fire'],
          dtype='object', name='类型1', length=800)




```python
#获取第二层索引
df_pokemon.index.get_level_values(1)
```




    Index(['Poison', 'Poison', 'Poison', 'Poison', 'nan', 'nan', 'Flying',
           'Dragon', 'Flying', 'nan',
           ...
           'Dragon', 'Dragon', 'nan', 'Flying', 'Ground', 'Fairy', 'Fairy',
           'Ghost', 'Dark', 'Water'],
          dtype='object', name='类型2', length=800)




```python
#交换level
df_pokemon.swaplevel()
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
      <th>姓名</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
    <tr>
      <th>类型2</th>
      <th>类型1</th>
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
      <th rowspan="4" valign="top">Poison</th>
      <th>Grass</th>
      <td>Bulbasaur</td>
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
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">nan</th>
      <th>Fire</th>
      <td>Charmander</td>
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
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <th>Fire</th>
      <td>Charizard</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <th>Fire</th>
      <td>CharizardMega Charizard X</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <th>Fire</th>
      <td>CharizardMega Charizard Y</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">nan</th>
      <th>Water</th>
      <td>Squirtle</td>
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
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <th>Bug</th>
      <td>Butterfree</td>
      <td>395</td>
      <td>60</td>
      <td>45</td>
      <td>50</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Poison</th>
      <th>Bug</th>
      <td>Weedle</td>
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
      <td>495</td>
      <td>65</td>
      <td>150</td>
      <td>40</td>
      <td>145</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Flying</th>
      <th>Normal</th>
      <td>Pidgey</td>
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
      <td>579</td>
      <td>83</td>
      <td>80</td>
      <td>80</td>
      <td>121</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">nan</th>
      <th>Normal</th>
      <td>Rattata</td>
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
      <td>413</td>
      <td>55</td>
      <td>81</td>
      <td>60</td>
      <td>97</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Flying</th>
      <th>Normal</th>
      <td>Spearow</td>
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
      <td>442</td>
      <td>65</td>
      <td>90</td>
      <td>65</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">nan</th>
      <th>Poison</th>
      <td>Ekans</td>
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
    </tr>
    <tr>
      <th>Fairy</th>
      <td>Sylveon</td>
      <td>525</td>
      <td>95</td>
      <td>65</td>
      <td>65</td>
      <td>60</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Flying</th>
      <th>Fighting</th>
      <td>Hawlucha</td>
      <td>500</td>
      <td>78</td>
      <td>92</td>
      <td>75</td>
      <td>118</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Fairy</th>
      <th>Electric</th>
      <td>Dedenne</td>
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
      <td>500</td>
      <td>50</td>
      <td>50</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">nan</th>
      <th>Dragon</th>
      <td>Goomy</td>
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
      <td>600</td>
      <td>90</td>
      <td>100</td>
      <td>70</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <th>Steel</th>
      <td>Klefki</td>
      <td>470</td>
      <td>57</td>
      <td>80</td>
      <td>91</td>
      <td>75</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">Grass</th>
      <th>Ghost</th>
      <td>Phantump</td>
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
      <td>494</td>
      <td>85</td>
      <td>100</td>
      <td>122</td>
      <td>54</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">nan</th>
      <th>Ice</th>
      <td>Bergmite</td>
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
      <td>514</td>
      <td>95</td>
      <td>117</td>
      <td>184</td>
      <td>28</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Dragon</th>
      <th>Flying</th>
      <td>Noibat</td>
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
      <td>535</td>
      <td>85</td>
      <td>70</td>
      <td>80</td>
      <td>123</td>
      <td>6</td>
    </tr>
    <tr>
      <th>nan</th>
      <th>Fairy</th>
      <td>Xerneas</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Flying</th>
      <th>Dark</th>
      <td>Yveltal</td>
      <td>680</td>
      <td>126</td>
      <td>131</td>
      <td>95</td>
      <td>99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ground</th>
      <th>Dragon</th>
      <td>Zygarde50% Forme</td>
      <td>600</td>
      <td>108</td>
      <td>100</td>
      <td>121</td>
      <td>95</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Fairy</th>
      <th>Rock</th>
      <td>Diancie</td>
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
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <th>Psychic</th>
      <td>HoopaHoopa Confined</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
      <td>60</td>
      <td>70</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Dark</th>
      <th>Psychic</th>
      <td>HoopaHoopa Unbound</td>
      <td>680</td>
      <td>80</td>
      <td>160</td>
      <td>60</td>
      <td>80</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Water</th>
      <th>Fire</th>
      <td>Volcanion</td>
      <td>600</td>
      <td>80</td>
      <td>110</td>
      <td>120</td>
      <td>70</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 7 columns</p>
</div>



### Q3：如何通过多重索引选取数据


```python
df_pokemon.head(10)
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
      <th rowspan="5" valign="top">Fire</th>
      <th>nan</th>
      <td>Charmander</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Charmeleon</td>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Charizard</td>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>CharizardMega Charizard X</td>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>CharizardMega Charizard Y</td>
      <td>634</td>
      <td>78</td>
      <td>104</td>
      <td>78</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Water</th>
      <th>nan</th>
      <td>Squirtle</td>
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



先对数据源的索引进行升序排序


```python
df_pokemon.sort_index(inplace=True)
df_pokemon
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
      <th rowspan="30" valign="top">Bug</th>
      <th>Electric</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
      <td>50</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
      <td>60</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Heracross</td>
      <td>500</td>
      <td>80</td>
      <td>125</td>
      <td>75</td>
      <td>85</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>HeracrossMega Heracross</td>
      <td>600</td>
      <td>80</td>
      <td>185</td>
      <td>115</td>
      <td>75</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Larvesta</td>
      <td>360</td>
      <td>55</td>
      <td>85</td>
      <td>55</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Volcarona</td>
      <td>550</td>
      <td>85</td>
      <td>60</td>
      <td>65</td>
      <td>100</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Butterfree</td>
      <td>395</td>
      <td>60</td>
      <td>45</td>
      <td>50</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Scyther</td>
      <td>500</td>
      <td>70</td>
      <td>110</td>
      <td>80</td>
      <td>105</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>PinsirMega Pinsir</td>
      <td>600</td>
      <td>65</td>
      <td>155</td>
      <td>120</td>
      <td>105</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Ledyba</td>
      <td>265</td>
      <td>40</td>
      <td>20</td>
      <td>30</td>
      <td>55</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Ledian</td>
      <td>390</td>
      <td>55</td>
      <td>35</td>
      <td>50</td>
      <td>85</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Yanma</td>
      <td>390</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>95</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Beautifly</td>
      <td>395</td>
      <td>60</td>
      <td>70</td>
      <td>50</td>
      <td>65</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Masquerain</td>
      <td>414</td>
      <td>70</td>
      <td>60</td>
      <td>62</td>
      <td>60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Ninjask</td>
      <td>456</td>
      <td>61</td>
      <td>90</td>
      <td>45</td>
      <td>160</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Mothim</td>
      <td>424</td>
      <td>70</td>
      <td>94</td>
      <td>50</td>
      <td>66</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Combee</td>
      <td>244</td>
      <td>30</td>
      <td>30</td>
      <td>42</td>
      <td>70</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Vespiquen</td>
      <td>474</td>
      <td>70</td>
      <td>80</td>
      <td>102</td>
      <td>40</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Yanmega</td>
      <td>515</td>
      <td>86</td>
      <td>76</td>
      <td>86</td>
      <td>95</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Vivillon</td>
      <td>411</td>
      <td>80</td>
      <td>52</td>
      <td>50</td>
      <td>89</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>Shedinja</td>
      <td>236</td>
      <td>1</td>
      <td>90</td>
      <td>45</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Paras</td>
      <td>285</td>
      <td>35</td>
      <td>70</td>
      <td>55</td>
      <td>25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Parasect</td>
      <td>405</td>
      <td>60</td>
      <td>95</td>
      <td>80</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>WormadamPlant Cloak</td>
      <td>424</td>
      <td>60</td>
      <td>59</td>
      <td>85</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Sewaddle</td>
      <td>310</td>
      <td>45</td>
      <td>53</td>
      <td>70</td>
      <td>42</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Swadloon</td>
      <td>380</td>
      <td>55</td>
      <td>63</td>
      <td>90</td>
      <td>42</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Leavanny</td>
      <td>500</td>
      <td>75</td>
      <td>103</td>
      <td>80</td>
      <td>92</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>Nincada</td>
      <td>266</td>
      <td>31</td>
      <td>45</td>
      <td>90</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>WormadamSandy Cloak</td>
      <td>424</td>
      <td>60</td>
      <td>79</td>
      <td>105</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Weedle</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="30" valign="top">Water</th>
      <th>nan</th>
      <td>Corphish</td>
      <td>308</td>
      <td>43</td>
      <td>80</td>
      <td>65</td>
      <td>35</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Feebas</td>
      <td>200</td>
      <td>20</td>
      <td>15</td>
      <td>20</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Milotic</td>
      <td>540</td>
      <td>95</td>
      <td>60</td>
      <td>79</td>
      <td>81</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Clamperl</td>
      <td>345</td>
      <td>35</td>
      <td>64</td>
      <td>85</td>
      <td>32</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Huntail</td>
      <td>485</td>
      <td>55</td>
      <td>104</td>
      <td>105</td>
      <td>52</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Gorebyss</td>
      <td>485</td>
      <td>55</td>
      <td>84</td>
      <td>105</td>
      <td>52</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Luvdisc</td>
      <td>330</td>
      <td>43</td>
      <td>30</td>
      <td>55</td>
      <td>97</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Kyogre</td>
      <td>670</td>
      <td>100</td>
      <td>100</td>
      <td>90</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>KyogrePrimal Kyogre</td>
      <td>770</td>
      <td>100</td>
      <td>150</td>
      <td>90</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Piplup</td>
      <td>314</td>
      <td>53</td>
      <td>51</td>
      <td>53</td>
      <td>40</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Prinplup</td>
      <td>405</td>
      <td>64</td>
      <td>66</td>
      <td>68</td>
      <td>50</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Buizel</td>
      <td>330</td>
      <td>55</td>
      <td>65</td>
      <td>35</td>
      <td>85</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Floatzel</td>
      <td>495</td>
      <td>85</td>
      <td>105</td>
      <td>55</td>
      <td>115</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Shellos</td>
      <td>325</td>
      <td>76</td>
      <td>48</td>
      <td>48</td>
      <td>34</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Finneon</td>
      <td>330</td>
      <td>49</td>
      <td>49</td>
      <td>56</td>
      <td>66</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Lumineon</td>
      <td>460</td>
      <td>69</td>
      <td>69</td>
      <td>76</td>
      <td>91</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Phione</td>
      <td>480</td>
      <td>80</td>
      <td>80</td>
      <td>80</td>
      <td>80</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Manaphy</td>
      <td>600</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Oshawott</td>
      <td>308</td>
      <td>55</td>
      <td>55</td>
      <td>45</td>
      <td>45</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Dewott</td>
      <td>413</td>
      <td>75</td>
      <td>75</td>
      <td>60</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Samurott</td>
      <td>528</td>
      <td>95</td>
      <td>100</td>
      <td>85</td>
      <td>70</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Panpour</td>
      <td>316</td>
      <td>50</td>
      <td>53</td>
      <td>48</td>
      <td>64</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Simipour</td>
      <td>498</td>
      <td>75</td>
      <td>98</td>
      <td>63</td>
      <td>101</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Tympole</td>
      <td>294</td>
      <td>50</td>
      <td>50</td>
      <td>40</td>
      <td>64</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Basculin</td>
      <td>460</td>
      <td>70</td>
      <td>92</td>
      <td>65</td>
      <td>98</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Alomomola</td>
      <td>470</td>
      <td>165</td>
      <td>75</td>
      <td>80</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Froakie</td>
      <td>314</td>
      <td>41</td>
      <td>56</td>
      <td>40</td>
      <td>71</td>
      <td>6</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Frogadier</td>
      <td>405</td>
      <td>54</td>
      <td>63</td>
      <td>52</td>
      <td>97</td>
      <td>6</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Clauncher</td>
      <td>330</td>
      <td>50</td>
      <td>53</td>
      <td>62</td>
      <td>44</td>
      <td>6</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Clawitzer</td>
      <td>500</td>
      <td>71</td>
      <td>73</td>
      <td>88</td>
      <td>59</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 7 columns</p>
</div>



**为什么要对索引升序排序？**

**因为如果没有对索引进行升序排序的话，在多重索引选取数据的过程中无法通过切片选取数据，切片是由小到大取的，例如字符串a→z，数字0→100，所以在对索引进行升序后，才能正确地切片选取数据**

接下来根据需求通过多重索引选取数据


```python
#取出第一索引列中值为Bug的所有数据
df_pokemon.loc["Bug"]
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
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
    <tr>
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
      <th>nan</th>
      <td>Caterpie</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Metapod</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Butterfree</td>
      <td>395</td>
      <td>60</td>
      <td>45</td>
      <td>50</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Weedle</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Kakuna</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Beedrill</td>
      <td>395</td>
      <td>65</td>
      <td>90</td>
      <td>40</td>
      <td>75</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>BeedrillMega Beedrill</td>
      <td>495</td>
      <td>65</td>
      <td>150</td>
      <td>40</td>
      <td>145</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Paras</td>
      <td>285</td>
      <td>35</td>
      <td>70</td>
      <td>55</td>
      <td>25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Parasect</td>
      <td>405</td>
      <td>60</td>
      <td>95</td>
      <td>80</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Venonat</td>
      <td>305</td>
      <td>60</td>
      <td>55</td>
      <td>50</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Venomoth</td>
      <td>450</td>
      <td>70</td>
      <td>65</td>
      <td>60</td>
      <td>90</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Scyther</td>
      <td>500</td>
      <td>70</td>
      <td>110</td>
      <td>80</td>
      <td>105</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Pinsir</td>
      <td>500</td>
      <td>65</td>
      <td>125</td>
      <td>100</td>
      <td>85</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>PinsirMega Pinsir</td>
      <td>600</td>
      <td>65</td>
      <td>155</td>
      <td>120</td>
      <td>105</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Ledyba</td>
      <td>265</td>
      <td>40</td>
      <td>20</td>
      <td>30</td>
      <td>55</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Ledian</td>
      <td>390</td>
      <td>55</td>
      <td>35</td>
      <td>50</td>
      <td>85</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Spinarak</td>
      <td>250</td>
      <td>40</td>
      <td>60</td>
      <td>40</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Ariados</td>
      <td>390</td>
      <td>70</td>
      <td>90</td>
      <td>70</td>
      <td>40</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Yanma</td>
      <td>390</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>95</td>
      <td>2</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Pineco</td>
      <td>290</td>
      <td>50</td>
      <td>65</td>
      <td>90</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>Forretress</td>
      <td>465</td>
      <td>75</td>
      <td>90</td>
      <td>140</td>
      <td>40</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>Scizor</td>
      <td>500</td>
      <td>70</td>
      <td>130</td>
      <td>100</td>
      <td>65</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>ScizorMega Scizor</td>
      <td>600</td>
      <td>70</td>
      <td>150</td>
      <td>140</td>
      <td>75</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>Shuckle</td>
      <td>505</td>
      <td>20</td>
      <td>10</td>
      <td>230</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Heracross</td>
      <td>500</td>
      <td>80</td>
      <td>125</td>
      <td>75</td>
      <td>85</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>HeracrossMega Heracross</td>
      <td>600</td>
      <td>80</td>
      <td>185</td>
      <td>115</td>
      <td>75</td>
      <td>2</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Wurmple</td>
      <td>195</td>
      <td>45</td>
      <td>45</td>
      <td>35</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Silcoon</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Beautifly</td>
      <td>395</td>
      <td>60</td>
      <td>70</td>
      <td>50</td>
      <td>65</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Cascoon</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
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
    </tr>
    <tr>
      <th>nan</th>
      <td>Kricketune</td>
      <td>384</td>
      <td>77</td>
      <td>85</td>
      <td>51</td>
      <td>65</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Burmy</td>
      <td>224</td>
      <td>40</td>
      <td>29</td>
      <td>45</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>WormadamPlant Cloak</td>
      <td>424</td>
      <td>60</td>
      <td>59</td>
      <td>85</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>WormadamSandy Cloak</td>
      <td>424</td>
      <td>60</td>
      <td>79</td>
      <td>105</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>WormadamTrash Cloak</td>
      <td>424</td>
      <td>60</td>
      <td>69</td>
      <td>95</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Mothim</td>
      <td>424</td>
      <td>70</td>
      <td>94</td>
      <td>50</td>
      <td>66</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Combee</td>
      <td>244</td>
      <td>30</td>
      <td>30</td>
      <td>42</td>
      <td>70</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Vespiquen</td>
      <td>474</td>
      <td>70</td>
      <td>80</td>
      <td>102</td>
      <td>40</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Yanmega</td>
      <td>515</td>
      <td>86</td>
      <td>76</td>
      <td>86</td>
      <td>95</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Sewaddle</td>
      <td>310</td>
      <td>45</td>
      <td>53</td>
      <td>70</td>
      <td>42</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Swadloon</td>
      <td>380</td>
      <td>55</td>
      <td>63</td>
      <td>90</td>
      <td>42</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Leavanny</td>
      <td>500</td>
      <td>75</td>
      <td>103</td>
      <td>80</td>
      <td>92</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Venipede</td>
      <td>260</td>
      <td>30</td>
      <td>45</td>
      <td>59</td>
      <td>57</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Whirlipede</td>
      <td>360</td>
      <td>40</td>
      <td>55</td>
      <td>99</td>
      <td>47</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Scolipede</td>
      <td>485</td>
      <td>60</td>
      <td>100</td>
      <td>89</td>
      <td>112</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>Dwebble</td>
      <td>325</td>
      <td>50</td>
      <td>65</td>
      <td>85</td>
      <td>55</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>Crustle</td>
      <td>475</td>
      <td>70</td>
      <td>95</td>
      <td>125</td>
      <td>45</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Karrablast</td>
      <td>315</td>
      <td>50</td>
      <td>75</td>
      <td>45</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>Escavalier</td>
      <td>495</td>
      <td>70</td>
      <td>135</td>
      <td>105</td>
      <td>20</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
      <td>50</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
      <td>60</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Shelmet</td>
      <td>305</td>
      <td>50</td>
      <td>40</td>
      <td>85</td>
      <td>25</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Accelgor</td>
      <td>495</td>
      <td>80</td>
      <td>70</td>
      <td>40</td>
      <td>145</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>Durant</td>
      <td>484</td>
      <td>58</td>
      <td>109</td>
      <td>112</td>
      <td>109</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Larvesta</td>
      <td>360</td>
      <td>55</td>
      <td>85</td>
      <td>55</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Volcarona</td>
      <td>550</td>
      <td>85</td>
      <td>60</td>
      <td>65</td>
      <td>100</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>Genesect</td>
      <td>600</td>
      <td>71</td>
      <td>120</td>
      <td>95</td>
      <td>99</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Scatterbug</td>
      <td>200</td>
      <td>38</td>
      <td>35</td>
      <td>40</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Spewpa</td>
      <td>213</td>
      <td>45</td>
      <td>22</td>
      <td>60</td>
      <td>29</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Vivillon</td>
      <td>411</td>
      <td>80</td>
      <td>52</td>
      <td>50</td>
      <td>89</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>69 rows × 7 columns</p>
</div>




```python
#取出第一索引列为Bug，第二索引列为Poison的所有数据
df_pokemon.loc[("Bug","Poison")]
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
      <th rowspan="12" valign="top">Bug</th>
      <th>Poison</th>
      <td>Weedle</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Kakuna</td>
      <td>205</td>
      <td>45</td>
      <td>25</td>
      <td>50</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Beedrill</td>
      <td>395</td>
      <td>65</td>
      <td>90</td>
      <td>40</td>
      <td>75</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>BeedrillMega Beedrill</td>
      <td>495</td>
      <td>65</td>
      <td>150</td>
      <td>40</td>
      <td>145</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Venonat</td>
      <td>305</td>
      <td>60</td>
      <td>55</td>
      <td>50</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Venomoth</td>
      <td>450</td>
      <td>70</td>
      <td>65</td>
      <td>60</td>
      <td>90</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Spinarak</td>
      <td>250</td>
      <td>40</td>
      <td>60</td>
      <td>40</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Ariados</td>
      <td>390</td>
      <td>70</td>
      <td>90</td>
      <td>70</td>
      <td>40</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Dustox</td>
      <td>385</td>
      <td>60</td>
      <td>50</td>
      <td>70</td>
      <td>65</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Venipede</td>
      <td>260</td>
      <td>30</td>
      <td>45</td>
      <td>59</td>
      <td>57</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Whirlipede</td>
      <td>360</td>
      <td>40</td>
      <td>55</td>
      <td>99</td>
      <td>47</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Scolipede</td>
      <td>485</td>
      <td>60</td>
      <td>100</td>
      <td>89</td>
      <td>112</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
#选出第一索引列为Bug到Grass的所有数据
df_pokemon.loc[slice("Bug","Grass")]
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
      <th rowspan="30" valign="top">Bug</th>
      <th>Electric</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
      <td>50</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
      <td>60</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Heracross</td>
      <td>500</td>
      <td>80</td>
      <td>125</td>
      <td>75</td>
      <td>85</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>HeracrossMega Heracross</td>
      <td>600</td>
      <td>80</td>
      <td>185</td>
      <td>115</td>
      <td>75</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Larvesta</td>
      <td>360</td>
      <td>55</td>
      <td>85</td>
      <td>55</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Volcarona</td>
      <td>550</td>
      <td>85</td>
      <td>60</td>
      <td>65</td>
      <td>100</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Butterfree</td>
      <td>395</td>
      <td>60</td>
      <td>45</td>
      <td>50</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Scyther</td>
      <td>500</td>
      <td>70</td>
      <td>110</td>
      <td>80</td>
      <td>105</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>PinsirMega Pinsir</td>
      <td>600</td>
      <td>65</td>
      <td>155</td>
      <td>120</td>
      <td>105</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Ledyba</td>
      <td>265</td>
      <td>40</td>
      <td>20</td>
      <td>30</td>
      <td>55</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Ledian</td>
      <td>390</td>
      <td>55</td>
      <td>35</td>
      <td>50</td>
      <td>85</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Yanma</td>
      <td>390</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>95</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Beautifly</td>
      <td>395</td>
      <td>60</td>
      <td>70</td>
      <td>50</td>
      <td>65</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Masquerain</td>
      <td>414</td>
      <td>70</td>
      <td>60</td>
      <td>62</td>
      <td>60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Ninjask</td>
      <td>456</td>
      <td>61</td>
      <td>90</td>
      <td>45</td>
      <td>160</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Mothim</td>
      <td>424</td>
      <td>70</td>
      <td>94</td>
      <td>50</td>
      <td>66</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Combee</td>
      <td>244</td>
      <td>30</td>
      <td>30</td>
      <td>42</td>
      <td>70</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Vespiquen</td>
      <td>474</td>
      <td>70</td>
      <td>80</td>
      <td>102</td>
      <td>40</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Yanmega</td>
      <td>515</td>
      <td>86</td>
      <td>76</td>
      <td>86</td>
      <td>95</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Vivillon</td>
      <td>411</td>
      <td>80</td>
      <td>52</td>
      <td>50</td>
      <td>89</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>Shedinja</td>
      <td>236</td>
      <td>1</td>
      <td>90</td>
      <td>45</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Paras</td>
      <td>285</td>
      <td>35</td>
      <td>70</td>
      <td>55</td>
      <td>25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Parasect</td>
      <td>405</td>
      <td>60</td>
      <td>95</td>
      <td>80</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>WormadamPlant Cloak</td>
      <td>424</td>
      <td>60</td>
      <td>59</td>
      <td>85</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Sewaddle</td>
      <td>310</td>
      <td>45</td>
      <td>53</td>
      <td>70</td>
      <td>42</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Swadloon</td>
      <td>380</td>
      <td>55</td>
      <td>63</td>
      <td>90</td>
      <td>42</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Leavanny</td>
      <td>500</td>
      <td>75</td>
      <td>103</td>
      <td>80</td>
      <td>92</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>Nincada</td>
      <td>266</td>
      <td>31</td>
      <td>45</td>
      <td>90</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>WormadamSandy Cloak</td>
      <td>424</td>
      <td>60</td>
      <td>79</td>
      <td>105</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Weedle</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
      <td>1</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="30" valign="top">Grass</th>
      <th>nan</th>
      <td>Meganium</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>100</td>
      <td>80</td>
      <td>2</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Bellossom</td>
      <td>490</td>
      <td>75</td>
      <td>80</td>
      <td>95</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Sunkern</td>
      <td>180</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Sunflora</td>
      <td>425</td>
      <td>75</td>
      <td>75</td>
      <td>55</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Treecko</td>
      <td>310</td>
      <td>40</td>
      <td>45</td>
      <td>35</td>
      <td>70</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Grovyle</td>
      <td>405</td>
      <td>50</td>
      <td>65</td>
      <td>45</td>
      <td>95</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Sceptile</td>
      <td>530</td>
      <td>70</td>
      <td>85</td>
      <td>65</td>
      <td>120</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Seedot</td>
      <td>220</td>
      <td>40</td>
      <td>40</td>
      <td>50</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Shroomish</td>
      <td>295</td>
      <td>60</td>
      <td>40</td>
      <td>60</td>
      <td>35</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Cacnea</td>
      <td>335</td>
      <td>50</td>
      <td>85</td>
      <td>40</td>
      <td>35</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Turtwig</td>
      <td>318</td>
      <td>55</td>
      <td>68</td>
      <td>64</td>
      <td>31</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Grotle</td>
      <td>405</td>
      <td>75</td>
      <td>89</td>
      <td>85</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Cherubi</td>
      <td>275</td>
      <td>45</td>
      <td>35</td>
      <td>45</td>
      <td>35</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Cherrim</td>
      <td>450</td>
      <td>70</td>
      <td>60</td>
      <td>70</td>
      <td>85</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Carnivine</td>
      <td>454</td>
      <td>74</td>
      <td>100</td>
      <td>72</td>
      <td>46</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Tangrowth</td>
      <td>535</td>
      <td>100</td>
      <td>100</td>
      <td>125</td>
      <td>50</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Leafeon</td>
      <td>525</td>
      <td>65</td>
      <td>110</td>
      <td>130</td>
      <td>95</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>ShayminLand Forme</td>
      <td>600</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Snivy</td>
      <td>308</td>
      <td>45</td>
      <td>45</td>
      <td>55</td>
      <td>63</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Servine</td>
      <td>413</td>
      <td>60</td>
      <td>60</td>
      <td>75</td>
      <td>83</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Serperior</td>
      <td>528</td>
      <td>75</td>
      <td>75</td>
      <td>95</td>
      <td>113</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Pansage</td>
      <td>316</td>
      <td>50</td>
      <td>53</td>
      <td>48</td>
      <td>64</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Simisage</td>
      <td>498</td>
      <td>75</td>
      <td>98</td>
      <td>63</td>
      <td>101</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Petilil</td>
      <td>280</td>
      <td>45</td>
      <td>35</td>
      <td>50</td>
      <td>30</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Lilligant</td>
      <td>480</td>
      <td>70</td>
      <td>60</td>
      <td>75</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Maractus</td>
      <td>461</td>
      <td>75</td>
      <td>86</td>
      <td>67</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Chespin</td>
      <td>313</td>
      <td>56</td>
      <td>61</td>
      <td>65</td>
      <td>38</td>
      <td>6</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Quilladin</td>
      <td>405</td>
      <td>61</td>
      <td>78</td>
      <td>95</td>
      <td>57</td>
      <td>6</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Skiddo</td>
      <td>350</td>
      <td>66</td>
      <td>65</td>
      <td>48</td>
      <td>52</td>
      <td>6</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Gogoat</td>
      <td>531</td>
      <td>123</td>
      <td>100</td>
      <td>62</td>
      <td>68</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>378 rows × 7 columns</p>
</div>




```python
#选出第一索引列为Bug到Grass，且第二索引列为Electric的所有数据
df_pokemon.loc[(slice("Bug","Grass"),"Electric"),:]
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
      <th rowspan="2" valign="top">Bug</th>
      <th>Electric</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
      <td>50</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
      <td>60</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <th>Electric</th>
      <td>Zekrom</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
      <td>120</td>
      <td>90</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



当想要取某一列索引下的全部数据时就需要用slice(None)


```python
#取第二索引列为Electric的所有数据
df_pokemon.loc[(slice(None),"Electric"),:]
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
      <th rowspan="2" valign="top">Bug</th>
      <th>Electric</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
      <td>50</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
      <td>60</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <th>Electric</th>
      <td>Zekrom</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
      <td>120</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Ground</th>
      <th>Electric</th>
      <td>Stunfisk</td>
      <td>471</td>
      <td>109</td>
      <td>66</td>
      <td>84</td>
      <td>32</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Water</th>
      <th>Electric</th>
      <td>Chinchou</td>
      <td>330</td>
      <td>75</td>
      <td>38</td>
      <td>38</td>
      <td>67</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Lanturn</td>
      <td>460</td>
      <td>125</td>
      <td>58</td>
      <td>58</td>
      <td>67</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#取第二索引列为Electric和Fire，且列为姓名到攻击力的所有数据
df_pokemon.loc[(slice(None),["Electric","Fire"]),"姓名":"攻击力"]
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
      <th>姓名</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
    </tr>
    <tr>
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
      <th rowspan="4" valign="top">Bug</th>
      <th>Electric</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Larvesta</td>
      <td>360</td>
      <td>55</td>
      <td>85</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Volcarona</td>
      <td>550</td>
      <td>85</td>
      <td>60</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Dark</th>
      <th>Fire</th>
      <td>Houndour</td>
      <td>330</td>
      <td>45</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Houndoom</td>
      <td>500</td>
      <td>75</td>
      <td>90</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>HoundoomMega Houndoom</td>
      <td>600</td>
      <td>75</td>
      <td>90</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Dragon</th>
      <th>Electric</th>
      <td>Zekrom</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Reshiram</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Electric</th>
      <th>Fire</th>
      <td>RotomHeat Rotom</td>
      <td>520</td>
      <td>50</td>
      <td>65</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Ghost</th>
      <th>Fire</th>
      <td>Litwick</td>
      <td>275</td>
      <td>50</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Lampent</td>
      <td>370</td>
      <td>60</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Chandelure</td>
      <td>520</td>
      <td>60</td>
      <td>55</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Ground</th>
      <th>Electric</th>
      <td>Stunfisk</td>
      <td>471</td>
      <td>109</td>
      <td>66</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>GroudonPrimal Groudon</td>
      <td>770</td>
      <td>100</td>
      <td>180</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <th>Fire</th>
      <td>Victini</td>
      <td>600</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Water</th>
      <th>Electric</th>
      <td>Chinchou</td>
      <td>330</td>
      <td>75</td>
      <td>38</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Lanturn</td>
      <td>460</td>
      <td>125</td>
      <td>58</td>
    </tr>
  </tbody>
</table>
</div>



前面的做法有点繁琐，还有更简洁的做法，可以不用去写slice


```python
idx = pd.IndexSlice
```


```python
#取第二索引列为Electric和Fire，且列为姓名到攻击力的所有数据
df_pokemon.loc[idx[:,["Electric","Fire"]],"姓名":"攻击力"]
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
      <th>姓名</th>
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
    </tr>
    <tr>
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
      <th rowspan="4" valign="top">Bug</th>
      <th>Electric</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Larvesta</td>
      <td>360</td>
      <td>55</td>
      <td>85</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Volcarona</td>
      <td>550</td>
      <td>85</td>
      <td>60</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Dark</th>
      <th>Fire</th>
      <td>Houndour</td>
      <td>330</td>
      <td>45</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Houndoom</td>
      <td>500</td>
      <td>75</td>
      <td>90</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>HoundoomMega Houndoom</td>
      <td>600</td>
      <td>75</td>
      <td>90</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Dragon</th>
      <th>Electric</th>
      <td>Zekrom</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Reshiram</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Electric</th>
      <th>Fire</th>
      <td>RotomHeat Rotom</td>
      <td>520</td>
      <td>50</td>
      <td>65</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Ghost</th>
      <th>Fire</th>
      <td>Litwick</td>
      <td>275</td>
      <td>50</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Lampent</td>
      <td>370</td>
      <td>60</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Chandelure</td>
      <td>520</td>
      <td>60</td>
      <td>55</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Ground</th>
      <th>Electric</th>
      <td>Stunfisk</td>
      <td>471</td>
      <td>109</td>
      <td>66</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>GroudonPrimal Groudon</td>
      <td>770</td>
      <td>100</td>
      <td>180</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <th>Fire</th>
      <td>Victini</td>
      <td>600</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Water</th>
      <th>Electric</th>
      <td>Chinchou</td>
      <td>330</td>
      <td>75</td>
      <td>38</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Lanturn</td>
      <td>460</td>
      <td>125</td>
      <td>58</td>
    </tr>
  </tbody>
</table>
</div>




```python
#取第二索引为Electric到Fire的所有数据
df_pokemon.loc[idx[:,"Electric":"Fire"],:]
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
      <th rowspan="6" valign="top">Bug</th>
      <th>Electric</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
      <td>50</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
      <td>60</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Heracross</td>
      <td>500</td>
      <td>80</td>
      <td>125</td>
      <td>75</td>
      <td>85</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>HeracrossMega Heracross</td>
      <td>600</td>
      <td>80</td>
      <td>185</td>
      <td>115</td>
      <td>75</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Larvesta</td>
      <td>360</td>
      <td>55</td>
      <td>85</td>
      <td>55</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Volcarona</td>
      <td>550</td>
      <td>85</td>
      <td>60</td>
      <td>65</td>
      <td>100</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Dark</th>
      <th>Fighting</th>
      <td>Scraggy</td>
      <td>348</td>
      <td>50</td>
      <td>75</td>
      <td>70</td>
      <td>48</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Scrafty</td>
      <td>488</td>
      <td>65</td>
      <td>90</td>
      <td>115</td>
      <td>58</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Houndour</td>
      <td>330</td>
      <td>45</td>
      <td>60</td>
      <td>30</td>
      <td>65</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Houndoom</td>
      <td>500</td>
      <td>75</td>
      <td>90</td>
      <td>50</td>
      <td>95</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>HoundoomMega Houndoom</td>
      <td>600</td>
      <td>75</td>
      <td>90</td>
      <td>90</td>
      <td>115</td>
      <td>2</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Dragon</th>
      <th>Electric</th>
      <td>Zekrom</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
      <td>120</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>AltariaMega Altaria</td>
      <td>590</td>
      <td>75</td>
      <td>110</td>
      <td>110</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Reshiram</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Electric</th>
      <th>Fairy</th>
      <td>Dedenne</td>
      <td>431</td>
      <td>67</td>
      <td>58</td>
      <td>57</td>
      <td>101</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>RotomHeat Rotom</td>
      <td>520</td>
      <td>50</td>
      <td>65</td>
      <td>107</td>
      <td>86</td>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Fire</th>
      <th>Fighting</th>
      <td>Combusken</td>
      <td>405</td>
      <td>60</td>
      <td>85</td>
      <td>60</td>
      <td>55</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Blaziken</td>
      <td>530</td>
      <td>80</td>
      <td>120</td>
      <td>70</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>BlazikenMega Blaziken</td>
      <td>630</td>
      <td>80</td>
      <td>160</td>
      <td>80</td>
      <td>100</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Monferno</td>
      <td>405</td>
      <td>64</td>
      <td>78</td>
      <td>52</td>
      <td>81</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Infernape</td>
      <td>534</td>
      <td>76</td>
      <td>104</td>
      <td>71</td>
      <td>108</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Pignite</td>
      <td>418</td>
      <td>90</td>
      <td>93</td>
      <td>55</td>
      <td>55</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Emboar</td>
      <td>528</td>
      <td>110</td>
      <td>123</td>
      <td>65</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Ghost</th>
      <th>Fire</th>
      <td>Litwick</td>
      <td>275</td>
      <td>50</td>
      <td>30</td>
      <td>55</td>
      <td>20</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Lampent</td>
      <td>370</td>
      <td>60</td>
      <td>40</td>
      <td>60</td>
      <td>55</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Chandelure</td>
      <td>520</td>
      <td>60</td>
      <td>55</td>
      <td>90</td>
      <td>80</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Grass</th>
      <th>Fairy</th>
      <td>Cottonee</td>
      <td>280</td>
      <td>40</td>
      <td>27</td>
      <td>60</td>
      <td>66</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>Whimsicott</td>
      <td>480</td>
      <td>60</td>
      <td>67</td>
      <td>85</td>
      <td>116</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Breloom</td>
      <td>460</td>
      <td>60</td>
      <td>130</td>
      <td>80</td>
      <td>70</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Virizion</td>
      <td>580</td>
      <td>91</td>
      <td>90</td>
      <td>72</td>
      <td>108</td>
      <td>5</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Normal</th>
      <th>Fairy</th>
      <td>AudinoMega Audino</td>
      <td>545</td>
      <td>103</td>
      <td>60</td>
      <td>126</td>
      <td>50</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>LopunnyMega Lopunny</td>
      <td>580</td>
      <td>65</td>
      <td>136</td>
      <td>94</td>
      <td>135</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>MeloettaPirouette Forme</td>
      <td>600</td>
      <td>100</td>
      <td>128</td>
      <td>90</td>
      <td>128</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Poison</th>
      <th>Fighting</th>
      <td>Croagunk</td>
      <td>300</td>
      <td>48</td>
      <td>61</td>
      <td>40</td>
      <td>50</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Toxicroak</td>
      <td>490</td>
      <td>83</td>
      <td>106</td>
      <td>65</td>
      <td>85</td>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">Psychic</th>
      <th>Fairy</th>
      <td>Mr. Mime</td>
      <td>460</td>
      <td>40</td>
      <td>45</td>
      <td>65</td>
      <td>90</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>Ralts</td>
      <td>198</td>
      <td>28</td>
      <td>25</td>
      <td>25</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>Kirlia</td>
      <td>278</td>
      <td>38</td>
      <td>35</td>
      <td>35</td>
      <td>50</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>Gardevoir</td>
      <td>518</td>
      <td>68</td>
      <td>65</td>
      <td>65</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>GardevoirMega Gardevoir</td>
      <td>618</td>
      <td>68</td>
      <td>85</td>
      <td>65</td>
      <td>100</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>Mime Jr.</td>
      <td>310</td>
      <td>20</td>
      <td>25</td>
      <td>45</td>
      <td>60</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>MewtwoMega Mewtwo X</td>
      <td>780</td>
      <td>106</td>
      <td>190</td>
      <td>100</td>
      <td>130</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Gallade</td>
      <td>518</td>
      <td>68</td>
      <td>125</td>
      <td>65</td>
      <td>80</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>GalladeMega Gallade</td>
      <td>618</td>
      <td>68</td>
      <td>165</td>
      <td>95</td>
      <td>110</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Victini</td>
      <td>600</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Rock</th>
      <th>Fairy</th>
      <td>Carbink</td>
      <td>500</td>
      <td>50</td>
      <td>50</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>Diancie</td>
      <td>600</td>
      <td>50</td>
      <td>100</td>
      <td>150</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>DiancieMega Diancie</td>
      <td>700</td>
      <td>50</td>
      <td>160</td>
      <td>110</td>
      <td>110</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Terrakion</td>
      <td>580</td>
      <td>91</td>
      <td>129</td>
      <td>90</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Steel</th>
      <th>Fairy</th>
      <td>Mawile</td>
      <td>380</td>
      <td>50</td>
      <td>85</td>
      <td>85</td>
      <td>50</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>MawileMega Mawile</td>
      <td>480</td>
      <td>50</td>
      <td>105</td>
      <td>125</td>
      <td>50</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>Klefki</td>
      <td>470</td>
      <td>57</td>
      <td>80</td>
      <td>91</td>
      <td>75</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Cobalion</td>
      <td>580</td>
      <td>91</td>
      <td>90</td>
      <td>129</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Water</th>
      <th>Electric</th>
      <td>Chinchou</td>
      <td>330</td>
      <td>75</td>
      <td>38</td>
      <td>38</td>
      <td>67</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Lanturn</td>
      <td>460</td>
      <td>125</td>
      <td>58</td>
      <td>58</td>
      <td>67</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>Marill</td>
      <td>250</td>
      <td>70</td>
      <td>20</td>
      <td>50</td>
      <td>40</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>Azumarill</td>
      <td>420</td>
      <td>100</td>
      <td>50</td>
      <td>80</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Poliwrath</td>
      <td>510</td>
      <td>90</td>
      <td>95</td>
      <td>95</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>KeldeoOrdinary Forme</td>
      <td>580</td>
      <td>91</td>
      <td>72</td>
      <td>90</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>KeldeoResolute Forme</td>
      <td>580</td>
      <td>91</td>
      <td>72</td>
      <td>90</td>
      <td>108</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>67 rows × 7 columns</p>
</div>




```python
#取第一索引为Bug到Grass，且第二索引为Electric到Fire的所有数据
df_pokemon.loc[idx["Bug":"Grass","Electric":"Fire"],:]
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
      <th rowspan="6" valign="top">Bug</th>
      <th>Electric</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
      <td>50</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
      <td>60</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Heracross</td>
      <td>500</td>
      <td>80</td>
      <td>125</td>
      <td>75</td>
      <td>85</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>HeracrossMega Heracross</td>
      <td>600</td>
      <td>80</td>
      <td>185</td>
      <td>115</td>
      <td>75</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Larvesta</td>
      <td>360</td>
      <td>55</td>
      <td>85</td>
      <td>55</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Volcarona</td>
      <td>550</td>
      <td>85</td>
      <td>60</td>
      <td>65</td>
      <td>100</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Dark</th>
      <th>Fighting</th>
      <td>Scraggy</td>
      <td>348</td>
      <td>50</td>
      <td>75</td>
      <td>70</td>
      <td>48</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Scrafty</td>
      <td>488</td>
      <td>65</td>
      <td>90</td>
      <td>115</td>
      <td>58</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Houndour</td>
      <td>330</td>
      <td>45</td>
      <td>60</td>
      <td>30</td>
      <td>65</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Houndoom</td>
      <td>500</td>
      <td>75</td>
      <td>90</td>
      <td>50</td>
      <td>95</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>HoundoomMega Houndoom</td>
      <td>600</td>
      <td>75</td>
      <td>90</td>
      <td>90</td>
      <td>115</td>
      <td>2</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Dragon</th>
      <th>Electric</th>
      <td>Zekrom</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
      <td>120</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>AltariaMega Altaria</td>
      <td>590</td>
      <td>75</td>
      <td>110</td>
      <td>110</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Reshiram</td>
      <td>680</td>
      <td>100</td>
      <td>120</td>
      <td>100</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Electric</th>
      <th>Fairy</th>
      <td>Dedenne</td>
      <td>431</td>
      <td>67</td>
      <td>58</td>
      <td>57</td>
      <td>101</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>RotomHeat Rotom</td>
      <td>520</td>
      <td>50</td>
      <td>65</td>
      <td>107</td>
      <td>86</td>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Fire</th>
      <th>Fighting</th>
      <td>Combusken</td>
      <td>405</td>
      <td>60</td>
      <td>85</td>
      <td>60</td>
      <td>55</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Blaziken</td>
      <td>530</td>
      <td>80</td>
      <td>120</td>
      <td>70</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>BlazikenMega Blaziken</td>
      <td>630</td>
      <td>80</td>
      <td>160</td>
      <td>80</td>
      <td>100</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Monferno</td>
      <td>405</td>
      <td>64</td>
      <td>78</td>
      <td>52</td>
      <td>81</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Infernape</td>
      <td>534</td>
      <td>76</td>
      <td>104</td>
      <td>71</td>
      <td>108</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Pignite</td>
      <td>418</td>
      <td>90</td>
      <td>93</td>
      <td>55</td>
      <td>55</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Emboar</td>
      <td>528</td>
      <td>110</td>
      <td>123</td>
      <td>65</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Ghost</th>
      <th>Fire</th>
      <td>Litwick</td>
      <td>275</td>
      <td>50</td>
      <td>30</td>
      <td>55</td>
      <td>20</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Lampent</td>
      <td>370</td>
      <td>60</td>
      <td>40</td>
      <td>60</td>
      <td>55</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Chandelure</td>
      <td>520</td>
      <td>60</td>
      <td>55</td>
      <td>90</td>
      <td>80</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Grass</th>
      <th>Fairy</th>
      <td>Cottonee</td>
      <td>280</td>
      <td>40</td>
      <td>27</td>
      <td>60</td>
      <td>66</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>Whimsicott</td>
      <td>480</td>
      <td>60</td>
      <td>67</td>
      <td>85</td>
      <td>116</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Breloom</td>
      <td>460</td>
      <td>60</td>
      <td>130</td>
      <td>80</td>
      <td>70</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Virizion</td>
      <td>580</td>
      <td>91</td>
      <td>90</td>
      <td>72</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Chesnaught</td>
      <td>530</td>
      <td>88</td>
      <td>107</td>
      <td>122</td>
      <td>64</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



还有一个函数xs可以通过level指定索引，然后去选取数据


```python
#选取第一索引列即类型1为Bug的所有数据
df_pokemon.xs("Bug",level=0)
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
      <th>总计</th>
      <th>生命值</th>
      <th>攻击力</th>
      <th>防御力</th>
      <th>速度</th>
      <th>时代</th>
    </tr>
    <tr>
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
      <th>Electric</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
      <td>50</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
      <td>60</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>Heracross</td>
      <td>500</td>
      <td>80</td>
      <td>125</td>
      <td>75</td>
      <td>85</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>HeracrossMega Heracross</td>
      <td>600</td>
      <td>80</td>
      <td>185</td>
      <td>115</td>
      <td>75</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Larvesta</td>
      <td>360</td>
      <td>55</td>
      <td>85</td>
      <td>55</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>Volcarona</td>
      <td>550</td>
      <td>85</td>
      <td>60</td>
      <td>65</td>
      <td>100</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Butterfree</td>
      <td>395</td>
      <td>60</td>
      <td>45</td>
      <td>50</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Scyther</td>
      <td>500</td>
      <td>70</td>
      <td>110</td>
      <td>80</td>
      <td>105</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>PinsirMega Pinsir</td>
      <td>600</td>
      <td>65</td>
      <td>155</td>
      <td>120</td>
      <td>105</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Ledyba</td>
      <td>265</td>
      <td>40</td>
      <td>20</td>
      <td>30</td>
      <td>55</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Ledian</td>
      <td>390</td>
      <td>55</td>
      <td>35</td>
      <td>50</td>
      <td>85</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Yanma</td>
      <td>390</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>95</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Beautifly</td>
      <td>395</td>
      <td>60</td>
      <td>70</td>
      <td>50</td>
      <td>65</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Masquerain</td>
      <td>414</td>
      <td>70</td>
      <td>60</td>
      <td>62</td>
      <td>60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Ninjask</td>
      <td>456</td>
      <td>61</td>
      <td>90</td>
      <td>45</td>
      <td>160</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Mothim</td>
      <td>424</td>
      <td>70</td>
      <td>94</td>
      <td>50</td>
      <td>66</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Combee</td>
      <td>244</td>
      <td>30</td>
      <td>30</td>
      <td>42</td>
      <td>70</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Vespiquen</td>
      <td>474</td>
      <td>70</td>
      <td>80</td>
      <td>102</td>
      <td>40</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Yanmega</td>
      <td>515</td>
      <td>86</td>
      <td>76</td>
      <td>86</td>
      <td>95</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>Vivillon</td>
      <td>411</td>
      <td>80</td>
      <td>52</td>
      <td>50</td>
      <td>89</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>Shedinja</td>
      <td>236</td>
      <td>1</td>
      <td>90</td>
      <td>45</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Paras</td>
      <td>285</td>
      <td>35</td>
      <td>70</td>
      <td>55</td>
      <td>25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Parasect</td>
      <td>405</td>
      <td>60</td>
      <td>95</td>
      <td>80</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>WormadamPlant Cloak</td>
      <td>424</td>
      <td>60</td>
      <td>59</td>
      <td>85</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Sewaddle</td>
      <td>310</td>
      <td>45</td>
      <td>53</td>
      <td>70</td>
      <td>42</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Swadloon</td>
      <td>380</td>
      <td>55</td>
      <td>63</td>
      <td>90</td>
      <td>42</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>Leavanny</td>
      <td>500</td>
      <td>75</td>
      <td>103</td>
      <td>80</td>
      <td>92</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>Nincada</td>
      <td>266</td>
      <td>31</td>
      <td>45</td>
      <td>90</td>
      <td>40</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>WormadamSandy Cloak</td>
      <td>424</td>
      <td>60</td>
      <td>79</td>
      <td>105</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Weedle</td>
      <td>195</td>
      <td>40</td>
      <td>35</td>
      <td>30</td>
      <td>50</td>
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
    </tr>
    <tr>
      <th>Poison</th>
      <td>Whirlipede</td>
      <td>360</td>
      <td>40</td>
      <td>55</td>
      <td>99</td>
      <td>47</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>Scolipede</td>
      <td>485</td>
      <td>60</td>
      <td>100</td>
      <td>89</td>
      <td>112</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>Shuckle</td>
      <td>505</td>
      <td>20</td>
      <td>10</td>
      <td>230</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>Dwebble</td>
      <td>325</td>
      <td>50</td>
      <td>65</td>
      <td>85</td>
      <td>55</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>Crustle</td>
      <td>475</td>
      <td>70</td>
      <td>95</td>
      <td>125</td>
      <td>45</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>Forretress</td>
      <td>465</td>
      <td>75</td>
      <td>90</td>
      <td>140</td>
      <td>40</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>Scizor</td>
      <td>500</td>
      <td>70</td>
      <td>130</td>
      <td>100</td>
      <td>65</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>ScizorMega Scizor</td>
      <td>600</td>
      <td>70</td>
      <td>150</td>
      <td>140</td>
      <td>75</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>WormadamTrash Cloak</td>
      <td>424</td>
      <td>60</td>
      <td>69</td>
      <td>95</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>Escavalier</td>
      <td>495</td>
      <td>70</td>
      <td>135</td>
      <td>105</td>
      <td>20</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>Durant</td>
      <td>484</td>
      <td>58</td>
      <td>109</td>
      <td>112</td>
      <td>109</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>Genesect</td>
      <td>600</td>
      <td>71</td>
      <td>120</td>
      <td>95</td>
      <td>99</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>Surskit</td>
      <td>269</td>
      <td>40</td>
      <td>30</td>
      <td>32</td>
      <td>65</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Caterpie</td>
      <td>195</td>
      <td>45</td>
      <td>30</td>
      <td>35</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Metapod</td>
      <td>205</td>
      <td>50</td>
      <td>20</td>
      <td>55</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Pinsir</td>
      <td>500</td>
      <td>65</td>
      <td>125</td>
      <td>100</td>
      <td>85</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Pineco</td>
      <td>290</td>
      <td>50</td>
      <td>65</td>
      <td>90</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Wurmple</td>
      <td>195</td>
      <td>45</td>
      <td>45</td>
      <td>35</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Silcoon</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Cascoon</td>
      <td>205</td>
      <td>50</td>
      <td>35</td>
      <td>55</td>
      <td>15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Volbeat</td>
      <td>400</td>
      <td>65</td>
      <td>73</td>
      <td>55</td>
      <td>85</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Illumise</td>
      <td>400</td>
      <td>65</td>
      <td>47</td>
      <td>55</td>
      <td>85</td>
      <td>3</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Kricketot</td>
      <td>194</td>
      <td>37</td>
      <td>25</td>
      <td>41</td>
      <td>25</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Kricketune</td>
      <td>384</td>
      <td>77</td>
      <td>85</td>
      <td>51</td>
      <td>65</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Burmy</td>
      <td>224</td>
      <td>40</td>
      <td>29</td>
      <td>45</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Karrablast</td>
      <td>315</td>
      <td>50</td>
      <td>75</td>
      <td>45</td>
      <td>60</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Shelmet</td>
      <td>305</td>
      <td>50</td>
      <td>40</td>
      <td>85</td>
      <td>25</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Accelgor</td>
      <td>495</td>
      <td>80</td>
      <td>70</td>
      <td>40</td>
      <td>145</td>
      <td>5</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Scatterbug</td>
      <td>200</td>
      <td>38</td>
      <td>35</td>
      <td>40</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>Spewpa</td>
      <td>213</td>
      <td>45</td>
      <td>22</td>
      <td>60</td>
      <td>29</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>69 rows × 7 columns</p>
</div>




```python
#选取第二索引列为Electric的所有数据
df_pokemon.xs("Electric",level=1)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bug</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
      <td>50</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Bug</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
      <td>60</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>Zekrom</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
      <td>120</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>Stunfisk</td>
      <td>471</td>
      <td>109</td>
      <td>66</td>
      <td>84</td>
      <td>32</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>Chinchou</td>
      <td>330</td>
      <td>75</td>
      <td>38</td>
      <td>38</td>
      <td>67</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>Lanturn</td>
      <td>460</td>
      <td>125</td>
      <td>58</td>
      <td>58</td>
      <td>67</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#选取第二索引列为Electric的所有数据,并且保留第二索引列
df_pokemon.xs("Electric",level=1,drop_level=False)
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
      <th rowspan="2" valign="top">Bug</th>
      <th>Electric</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
      <td>50</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
      <td>60</td>
      <td>108</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <th>Electric</th>
      <td>Zekrom</td>
      <td>680</td>
      <td>100</td>
      <td>150</td>
      <td>120</td>
      <td>90</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Ground</th>
      <th>Electric</th>
      <td>Stunfisk</td>
      <td>471</td>
      <td>109</td>
      <td>66</td>
      <td>84</td>
      <td>32</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Water</th>
      <th>Electric</th>
      <td>Chinchou</td>
      <td>330</td>
      <td>75</td>
      <td>38</td>
      <td>38</td>
      <td>67</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Lanturn</td>
      <td>460</td>
      <td>125</td>
      <td>58</td>
      <td>58</td>
      <td>67</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#选取第一索引列为Bug，第二索引列为Electric的所有数据
df_pokemon.xs(("Bug","Electric"),level=(0,1))
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
      <th rowspan="2" valign="top">Bug</th>
      <th>Electric</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
      <td>50</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
      <td>60</td>
      <td>108</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
#level也可以是索引列名
df_pokemon.xs(("Bug","Electric"),level=(["类型1","类型2"]))
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
      <th rowspan="2" valign="top">Bug</th>
      <th>Electric</th>
      <td>Joltik</td>
      <td>319</td>
      <td>50</td>
      <td>47</td>
      <td>50</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>Galvantula</td>
      <td>472</td>
      <td>70</td>
      <td>77</td>
      <td>60</td>
      <td>108</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



xs在每个索引列上选择的标签只能是一个，所以做不到切片标签的选取
