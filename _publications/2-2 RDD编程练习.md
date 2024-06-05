---
title: "2.2 RDD编程练习"
collection: publications
permalink: /publication/2-2 RDD编程练习
excerpt: 'Spark，是一种通用的大数据计算框架，I正如传统大数据技术Hadoop的MapReduce、Hive引擎，以及Storm流式实时计算引擎等，
Spark包含了大数据领城常见的各种计算框架：比如Spark Core用于离线计算，Spark SQL用于交互式查询，Spark Streaming用于实时流式计算，Spark MILlib用于机器学习，Spark GraphX用于图计算<br/><img src="/images/pytorch.jpg">'
date: 2021-10-04
venue: 'Journal 1'
---




为强化RDD编程API的使用经验，现提供一些小练习题。

读者可以使用RDD的编程API完成这些小练习题，并输出结果。

这些练习题基本可以在15行代码以内完成，如果遇到困难，建议回看上一节RDD的API介绍。

完成这些练习题后，可以查看本节后面的参考答案，和自己的实现方案进行对比。


```python
import findspark

#指定spark_home为刚才的解压路径,指定python路径
spark_home = "/Users/liangyun/ProgramFiles/spark-3.0.1-bin-hadoop3.2"
python_path = "/Users/liangyun/anaconda3/bin/python"
findspark.init(spark_home,python_path)

import pyspark 
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("rdd_tutorial").setMaster("local[4]")
sc = SparkContext(conf=conf)

print(pyspark.__version__)

```

### 一，练习题列表


**1，求平均数**

```python
#任务：求data的平均值
data = [1,5,7,10,23,20,6,5,10,7,10]

```

**2，求众数**

```python
#任务：求data中出现次数最多的数
data =  [1,5,7,10,23,20,6,5,10,7,10]

```

```python

```

**3，求TopN**

```python
#任务：有一批学生信息表格，包括name,age,score, 找出score排名前3的学生, score相同可以任取
students = [("LiLei",18,87),("HanMeiMei",16,77),("DaChui",16,66),("Jim",18,77),("RuHua",18,50)]
n = 3
```


**4，排序并返回序号**


```python
#任务：排序并返回序号, 大小相同的序号可以不同
data = [1,7,8,5,3,18,34,9,0,12,8]

```


**5，二次排序**

```python
#任务：有一批学生信息表格，包括name,age,score
#首先根据学生的score从大到小排序，如果score相同，根据age从大到小
students = [("LiLei",18,87),("HanMeiMei",16,77),("DaChui",16,66),("Jim",18,77),("RuHua",18,50)]


```


**6，连接操作**

```python
#任务：已知班级信息表和成绩表，找出班级平均分在75分以上的班级
#班级信息表包括class,name,成绩表包括name,score

classes = [("class1","LiLei"), ("class1","HanMeiMei"),("class2","DaChui"),("class2","RuHua")]
scores = [("LiLei",76),("HanMeiMei",80),("DaChui",70),("RuHua",60)]

```


**7，分组求众数**

```python
#任务：有一批学生信息表格，包括class和age。求每个班级学生年龄的众数。
students = [("class1",15),("class1",15),("class2",16),("class2",16),("class1",17),("class2",19)]


```

### 二，练习题参考答案

**1，求平均数**

```python
#任务：求data的平均值
data = [1,5,7,10,23,20,6,5,10,7,10]

rdd_data = sc.parallelize(data)
s = rdd_data.reduce(lambda x,y:x+y+0.0)
n = rdd_data.count()
avg = s/n
print("average:",avg)

```

```
average: 9.454545454545455

```

```python

```

**2，求众数**

```python
#任务：求data中出现次数最多的数，若有多个，求这些数的平均值
data =  [1,5,7,10,23,20,7,5,10,7,10]

rdd_data = sc.parallelize(data)
rdd_count = rdd_data.map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y)
max_count = rdd_count.map(lambda x:x[1]).reduce(lambda x,y: x if x>=y else y)
rdd_mode = rdd_count.filter(lambda x:x[1]==max_count).map(lambda x:x[0])
mode = rdd_mode.reduce(lambda x,y:x+y+0.0)/rdd_mode.count()
print("mode:",mode)
```

```
mode: 8.5
```


**3，求TopN**

```python
#任务：有一批学生信息表格，包括name,age,score, 找出score排名前3的学生, score相同可以任取
students = [("LiLei",18,87),("HanMeiMei",16,77),("DaChui",16,66),("Jim",18,77),("RuHua",18,50)]
n = 3

rdd_students = sc.parallelize(students)
rdd_sorted = rdd_students.sortBy(lambda x:x[2],ascending = False)

students_topn = rdd_sorted.take(n)
print(students_topn)
```

```
[('LiLei', 18, 87), ('HanMeiMei', 16, 77), ('Jim', 18, 77)]
```


**4，排序并返回序号**


```python
#任务：按从小到大排序并返回序号, 大小相同的序号可以不同
data = [1,7,8,5,3,18,34,9,0,12,8]

rdd_data = sc.parallelize(data)
rdd_sorted = rdd_data.map(lambda x:(x,1)).sortByKey().map(lambda x:x[0])
rdd_sorted_index = rdd_sorted.zipWithIndex()

print(rdd_sorted_index.collect())

```

```
[(0, 0), (1, 1), (3, 2), (5, 3), (7, 4), (8, 5), (8, 6), (9, 7), (12, 8), (18, 9), (34, 10)]
```


**5，二次排序**

```python
#任务：有一批学生信息表格，包括name,age,score
#首先根据学生的score从大到小排序，如果score相同，根据age从大到小

students = [("LiLei",18,87),("HanMeiMei",16,77),("DaChui",16,66),("Jim",18,77),("RuHua",18,50)]
rdd_students = sc.parallelize(students)
```

```python
%%writefile student.py
#为了在RDD中使用自定义类，需要将类的创建代码其写入到一个文件中，否则会有序列化错误
class Student:
    def __init__(self,name,age,score):
        self.name = name
        self.age = age
        self.score = score
    def __gt__(self,other):
        if self.score > other.score:
            return True
        elif self.score==other.score and self.age>other.age:
            return True
        else:
            return False
```

```python
from student import Student

rdd_sorted = rdd_students \
    .map(lambda t:Student(t[0],t[1],t[2]))\
    .sortBy(lambda x:x,ascending = False)\
    .map(lambda student:(student.name,student.age,student.score))

#参考方案：此处巧妙地对score和age进行编码来表达其排序优先级关系，除非age超过100000，以下逻辑无错误。
#rdd_sorted = rdd_students.sortBy(lambda x:100000*x[2]+x[1],ascending=False)

rdd_sorted.collect()
```

```
[('LiLei', 18, 87),
 ('Jim', 18, 77),
 ('HanMeiMei', 16, 77),
 ('DaChui', 16, 66),
 ('RuHua', 18, 50)]
```


**6，连接操作**

```python
#任务：已知班级信息表和成绩表，找出班级平均分在75分以上的班级
#班级信息表包括class,name,成绩表包括name,score

classes = [("class1","LiLei"), ("class1","HanMeiMei"),("class2","DaChui"),("class2","RuHua")]
scores = [("LiLei",76),("HanMeiMei",80),("DaChui",70),("RuHua",60)]

rdd_classes = sc.parallelize(classes).map(lambda x:(x[1],x[0]))
rdd_scores = sc.parallelize(scores)
rdd_join = rdd_scores.join(rdd_classes).map(lambda t:(t[1][1],t[1][0]))

def average(iterator):
    data = list(iterator)
    s = 0.0
    for x in data:
        s = s + x
    return s/len(data)

rdd_result = rdd_join.groupByKey().map(lambda t:(t[0],average(t[1]))).filter(lambda t:t[1]>75)
print(rdd_result.collect())
```

```
[('class1', 78.0)]
```


**7，分组求众数**

```python
#任务：有一批学生信息表格，包括class和age。求每个班级学生年龄的众数。

students = [("class1",15),("class1",15),("class2",16),("class2",16),("class1",17),("class2",19)]

```

```python
def mode(arr):
    dict_cnt = {}
    for x in arr:
        dict_cnt[x] = dict_cnt.get(x,0)+1
    max_cnt = max(dict_cnt.values())
    most_values = [k for k,v in dict_cnt.items() if v==max_cnt]
    s = 0.0
    for x in most_values:
        s = s + x
    return s/len(most_values)

rdd_students = sc.parallelize(students)
rdd_classes = rdd_students.aggregateByKey([],lambda arr,x:arr+[x],lambda arr1,arr2:arr1+arr2)
rdd_mode = rdd_classes.map(lambda t:(t[0],mode(t[1])))

print(rdd_mode.collect())

```

```
[('class1', 15.0), ('class2', 16.0)]
```
