---
title: "2.4 SparkSQL编程练习"
collection: publications
permalink: /publication/2-4SparkSQL编程练习
excerpt: 'Spark相较于MapReduce速度快的最主要原因就在于，MapReduce的计算模型太死板，必须是map-reduce模式，有时候即使完成一些诸如过滤之类的操作，也必须经过map-reduce过程，这样就必须经过shuffle过程<br/><img src="/images/pytorch.jpg">'
date: 2021-10-06
venue: 'Journal 1'
---


为强化SparkSQL编程基本功，现提供一些小练习题。

读者可以使用SparkSQL编程完成这些小练习题，并输出结果。

这些练习题基本可以在15行代码以内完成，如果遇到困难，建议回看上一节SparkSQL的介绍。

完成这些练习题后，可以查看本节后面的参考答案，和自己的实现方案进行对比。

我敢打赌，这些练习题一定会让大家有一种似曾相识之感。


```python
import findspark

#指定spark_home为刚才的解压路径,指定python路径
spark_home = "/Users/liangyun/ProgramFiles/spark-3.0.1-bin-hadoop3.2"
python_path = "/Users/liangyun/anaconda3/bin/python"
findspark.init(spark_home,python_path)

import pyspark 
from pyspark.sql import SparkSession

#SparkSQL的许多功能封装在SparkSession的方法接口中

spark = SparkSession.builder \
        .appName("test") \
        .config("master","local[4]") \
        .enableHiveSupport() \
        .getOrCreate()

sc = spark.sparkContext

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

```python

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

dfdata = spark.createDataFrame([(x,) for x in data]).toDF("value")
dfagg = dfdata.agg({"value":"avg"})
dfagg.show()

```

```
+-----------------+
|       avg(value)|
+-----------------+
|9.454545454545455|
+-----------------+
```


**2，求众数**

```python
#任务：求data中出现次数最多的数，若有多个，求这些数的平均值
from pyspark.sql import functions as F 
data =  [1,5,7,10,23,20,7,5,10,7,10]

dfdata = spark.createDataFrame([(x,1) for x in data]).toDF("key","value")
dfcount = dfdata.groupby("key").agg(F.count("value").alias("count")).cache()
max_count = dfcount.agg(F.max("count").alias("max_count")).take(1)[0]["max_count"]
dfmode = dfcount.where("count={}".format(max_count))
mode = dfmode.agg(F.expr("mean(key) as mode")).take(1)[0]["mode"]
print("mode:",mode)

dfcount.unpersist()

```

```
mode: 8.5
```


**3，求TopN**

```python
#任务：有一批学生信息表格，包括name,age,score, 找出score排名前3的学生, score相同可以任取
students = [("LiLei",18,87),("HanMeiMei",16,77),("DaChui",16,66),("Jim",18,77),("RuHua",18,50)]
n = 3

dfstudents = spark.createDataFrame(students).toDF("name","age","score")
dftopn = dfstudents.orderBy("score", ascending=False).limit(n)

dftopn.show()
```

```
+---------+---+-----+
|     name|age|score|
+---------+---+-----+
|    LiLei| 18|   87|
|HanMeiMei| 16|   77|
|      Jim| 18|   77|
+---------+---+-----+
```


**4，排序并返回序号**


```python
#任务：按从小到大排序并返回序号, 大小相同的序号可以不同

data = [1,7,8,5,3,18,34,9,0,12,8]

from copy import deepcopy
from pyspark.sql import types as T
from pyspark.sql import Row,DataFrame

def addLongIndex(df, field_name):
    schema = deepcopy(df.schema)
    schema = schema.add(T.StructField(field_name, T.LongType()))
    rdd_with_index = df.rdd.zipWithIndex()

    def merge_row(t):
        row,index= t
        dic = row.asDict() 
        dic.update({field_name:index})
        row_merged = Row(**dic)
        return row_merged

    rdd_row = rdd_with_index.map(lambda t:merge_row(t))
    return spark.createDataFrame(rdd_row,schema)

dfdata = spark.createDataFrame([(x,) for x in data]).toDF("value")
dfsorted = dfdata.sort(dfdata["value"])

dfsorted_index = addLongIndex(dfsorted,"index")

dfsorted_index.show() 

```

```
+-----+-----+
|value|index|
+-----+-----+
|    0|    0|
|    1|    1|
|    3|    2|
|    5|    3|
|    7|    4|
|    8|    5|
|    8|    6|
|    9|    7|
|   12|    8|
|   18|    9|
|   34|   10|
+-----+-----+
```


**5，二次排序**

```python
#任务：有一批学生信息表格，包括name,age,score
#首先根据学生的score从大到小排序，如果score相同，根据age从大到小
students = [("LiLei",18,87),("HanMeiMei",16,77),("DaChui",16,66),("Jim",18,77),("RuHua",18,50)]
dfstudents = spark.createDataFrame(students).toDF("name","age","score")
dfsorted = dfstudents.orderBy(dfstudents["score"].desc(),dfstudents["age"].desc())
dfsorted.show()

```

```
+---------+---+-----+
|     name|age|score|
+---------+---+-----+
|    LiLei| 18|   87|
|      Jim| 18|   77|
|HanMeiMei| 16|   77|
|   DaChui| 16|   66|
|    RuHua| 18|   50|
+---------+---+-----+
```


**6，连接操作**

```python
#任务：已知班级信息表和成绩表，找出班级平均分在75分以上的班级
#班级信息表包括class,name,成绩表包括name,score

from pyspark.sql import functions as F 
classes = [("class1","LiLei"), ("class1","HanMeiMei"),("class2","DaChui"),("class2","RuHua")]
scores = [("LiLei",76),("HanMeiMei",80),("DaChui",70),("RuHua",60)]

dfclass = spark.createDataFrame(classes).toDF("class","name")
dfscore = spark.createDataFrame(scores).toDF("name","score")

dfstudents = dfclass.join(dfscore,on ="name" ,how = "left")

dfagg = dfstudents.groupBy("class").agg(F.avg("score").alias("avg_score")).where("avg_score>75.0")
     
dfagg.show()
```

```
+------+---------+
| class|avg_score|
+------+---------+
|class1|     78.0|
+------+---------+
```


**7，分组求众数**

```python
#任务：有一批学生信息表格，包括class和age。求每个班级学生年龄的众数。

students = [("class1",15),("class1",15),("class2",16),("class2",16),("class1",17),("class2",19)]

```

```python

from pyspark.sql import functions as F 

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
spark.udf.register("udf_mode",mode)
dfstudents = spark.createDataFrame(students).toDF("class","score")
dfscores = dfstudents.groupBy("class").agg(F.collect_list("score").alias("scores"))
dfmode = dfscores.selectExpr("class","udf_mode(scores) as mode_score")
dfmode.show()

```

```
+------+----------+
| class|mode_score|
+------+----------+
|class2|      16.0|
|class1|      15.0|
+------+----------+

```
