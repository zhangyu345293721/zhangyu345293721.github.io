---
title: "4.1 探索MLlib机器学习"
collection: publications
permalink: /publication/4-1探索MLlib机器学习
excerpt: 'spark将每个任务构建成DAG进行计算，内部的计算过程通过弹性式分布式数据集RDD在内存在进行计算，相比于hadoop的mapreduce效率提升了100倍<br/><img src="/pytorch.jpg">'
date: 2021-10-15
venue: 'Journal 1'
---



MLlib是Spark的机器学习库，包括以下主要功能。

实用工具：线性代数，统计，数据处理等工具
特征工程：特征提取，特征转换，特征选择
常用算法：分类，回归，聚类，协同过滤，降维
模型优化：模型评估，参数优化。

MLlib库包括两个不同的部分：

pyspark.mllib 包含基于rdd的机器学习算法API，目前不再更新，以后将被丢弃，不建议使用。

pyspark.ml 包含基于DataFrame的机器学习算法API，可以用来构建机器学习工作流Pipeline，推荐使用。


```python
import findspark

#指定spark_home为刚才的解压路径,指定python路径
spark_home = "/Users/liangyun/ProgramFiles/spark-3.0.1-bin-hadoop3.2"
python_path = "/Users/liangyun/anaconda3/bin/python"
findspark.init(spark_home,python_path)

import pyspark 
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel 

#SparkSQL的许多功能封装在SparkSession的方法接口中

spark = SparkSession.builder \
        .appName("dbscan") \
        .config("master","local[4]") \
        .enableHiveSupport() \
        .getOrCreate()

sc = spark.sparkContext
```


### 一，MLlib基本概念

DataFrame: MLlib中数据的存储形式，其列可以存储特征向量，标签，以及原始的文本，图像。

Transformer：转换器。具有transform方法。通过附加一个或多个列将一个DataFrame转换成另外一个DataFrame。

Estimator：估计器。具有fit方法。它接受一个DataFrame数据作为输入后经过训练，产生一个转换器Transformer。

Pipeline：流水线。具有setStages方法。顺序将多个Transformer和1个Estimator串联起来，得到一个流水线模型。


### 二， Pipeline流水线范例

任务描述：用逻辑回归模型预测句子中是否包括”spark“这个单词。

```python
from pyspark.ml.feature import Tokenizer,HashingTF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.linalg import Vector
from pyspark.sql import Row
```

**1，准备数据**

```python
dftrain = spark.createDataFrame([(0,"a b c d e spark",1.0),
                (1,"a c f",0.0),
                (2,"spark hello world",1.0),
                (3,"hadoop mapreduce",0.0),
                (4,"I love spark", 1.0),
                (5,"big data",0.0)],["id","text","label"])
dftrain.show()
```

```
+---+-----------------+-----+
| id|             text|label|
+---+-----------------+-----+
|  0|  a b c d e spark|  1.0|
|  1|            a c f|  0.0|
|  2|spark hello world|  1.0|
|  3| hadoop mapreduce|  0.0|
|  4|     I love spark|  1.0|
|  5|         big data|  0.0|
+---+-----------------+-----+
```


**2，定义模型**

```python
tokenizer = Tokenizer().setInputCol("text").setOutputCol("words")
print(type(tokenizer))

hashingTF = HashingTF().setNumFeatures(100) \
   .setInputCol(tokenizer.getOutputCol()) \
   .setOutputCol("features")
print(type(hashingTF))

lr = LogisticRegression().setLabelCol("label")
#print(lr.explainParams)
lr.setFeaturesCol("features").setMaxIter(10).setRegParam(0.2)
print(type(lr))

pipe = Pipeline().setStages([tokenizer,hashingTF,lr])
print(type(pipe))     


```

```
<class 'pyspark.ml.feature.Tokenizer'>
<class 'pyspark.ml.feature.HashingTF'>
<class 'pyspark.ml.classification.LogisticRegression'>
<class 'pyspark.ml.pipeline.Pipeline'>
```


**3，训练模型**

```python
model = pipe.fit(dftrain)
print(type(model))

```

```
<class 'pyspark.ml.pipeline.PipelineModel'>
```


**4，使用模型**

```python
dftest = spark.createDataFrame([(7,"spark job",1.0),(9,"hello world",0.0),
                 (10,"a b c d e",0.0),(11,"you can you up",0.0),
                (12,"spark is easy to use.",1.0)]).toDF("id","text","label")
dftest.show()

dfresult = model.transform(dftest)

dfresult.selectExpr("text","features","probability","prediction").show()


```

```
+---+--------------------+-----+
| id|                text|label|
+---+--------------------+-----+
|  7|           spark job|  1.0|
|  9|         hello world|  0.0|
| 10|           a b c d e|  0.0|
| 11|      you can you up|  0.0|
| 12|spark is easy to ...|  1.0|
+---+--------------------+-----+

+--------------------+--------------------+--------------------+----------+
|                text|            features|         probability|prediction|
+--------------------+--------------------+--------------------+----------+
|           spark job|(100,[57,86],[1.0...|[0.30134853865356...|       1.0|
|         hello world|(100,[60,89],[1.0...|[0.20714372651040...|       1.0|
|           a b c d e|(100,[50,65,67,68...|[0.24502686265469...|       1.0|
|      you can you up|(100,[33,38,51],[...|[0.87589306761045...|       0.0|
|spark is easy to ...|(100,[9,21,60,86,...|[0.07662944406376...|       1.0|
+--------------------+--------------------+--------------------+----------+
```


**5，评估模型**

```python
dfresult.printSchema()
```

```
root
 |-- id: long (nullable = true)
 |-- text: string (nullable = true)
 |-- label: double (nullable = true)
 |-- words: array (nullable = true)
 |    |-- element: string (containsNull = true)
 |-- features: vector (nullable = true)
 |-- rawPrediction: vector (nullable = true)
 |-- probability: vector (nullable = true)
 |-- prediction: double (nullable = false)

```

```python
evaluator = MulticlassClassificationEvaluator().setMetricName("f1") \
    .setPredictionCol("prediction").setLabelCol("label")

#print(evaluator.explainParams())
accuracy  = evaluator.evaluate(dfresult)
print("\n accuracy = {}".format(accuracy))
```

```
accuracy = 0.5666666666666667
```


**6，保存模型**

```python
#可以将训练好的模型保存到磁盘中
model.write().overwrite().save("./data/mymodel.model")

#也可以将没有训练的模型保存到磁盘中
#pipeline.write.overwrite().save("./data/unfit-lr-model")

```

```python
#重新载入模型
model_loaded = PipelineModel.load("./data/mymodel.model")
model_loaded.transform(dftest).select("text","label","prediction").show()
```

```
+--------------------+-----+----------+
|                text|label|prediction|
+--------------------+-----+----------+
|           spark job|  1.0|       1.0|
|         hello world|  0.0|       1.0|
|           a b c d e|  0.0|       1.0|
|      you can you up|  0.0|       0.0|
|spark is easy to ...|  1.0|       1.0|
+--------------------+-----+----------+
```


### 三，特征工程


spark的特征处理功能主要在 pyspark.ml.feature 模块中，包括以下一些功能。

* 特征提取：Tf-idf, Word2Vec, CountVectorizer, FeatureHasher

* 特征转换：OneHotEncoderEstimator, Normalizer, Imputer(缺失值填充), StandardScaler, MinMaxScaler, Tokenizer(构建词典), 
  StopWordsRemover, SQLTransformer, Bucketizer, Interaction(交叉项), Binarizer(二值化), n-gram,……

* 特征选择：VectorSlicer(向量切片), RFormula, ChiSqSelector(卡方检验)

* LSH转换：局部敏感哈希广泛用于海量数据中求最邻近，聚类等算法。



**1，CountVectorizer**


CountVectorizer可以提取文本中的词频特征。

```python
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel

df = spark.createDataFrame([
  (0, ["a", "b", "c"]),
  (1, ["a", "b", "b", "c", "a"])],["id","words"])

cvModel = CountVectorizer() \
  .setInputCol("words") \
  .setOutputCol("features") \
  .setVocabSize(3) \
  .setMinDF(2) \
  .fit(df)

cvModel.transform(df).show()

```

**2，Word2Vec**


Word2Vec可以使用浅层神经网络提取文本中词的相似语义信息。

```python
from pyspark.ml.feature import Word2Vec

df_document = spark.createDataFrame([
    ("Hi I heard about Spark".split(" "), ),
    ("I wish Java could use case classes".split(" "), ),
    ("Logistic regression models are neat".split(" "), )
], ["text"])

word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
model = word2Vec.fit(df_document)

df_vector = model.transform(df_document)
for row in df_vector.collect():
    text, vector = row
    print("text: [%s] => \nvector: %s\n" % (", ".join(text), str(vector)))

```

```
text: [Hi, I, heard, about, Spark] => 
vector: [-0.03952452838420868,-0.019742850959300996,-0.04259629175066948]

text: [I, wish, Java, could, use, case, classes] => 
vector: [-0.017589610069990158,0.03303118874984128,-0.03793099456067596]

text: [Logistic, regression, models, are, neat] => 
vector: [-0.03930013366043568,0.08479443639516832,-0.025407366454601288]
```

**3， OnHotEncoder**


OneHotEncoder可以将类别特征转换成OneHot编码。

```python
from pyspark.ml.feature import OneHotEncoder

df = spark.createDataFrame([
    (0.0, 1.0),
    (1.0, 0.0),
    (2.0, 1.0),
    (0.0, 2.0),
    (0.0, 1.0),
    (2.0, 0.0)
], ["categoryIndex1", "categoryIndex2"])

encoder = OneHotEncoder(inputCols=["categoryIndex1", "categoryIndex2"],
                                 outputCols=["categoryVec1", "categoryVec2"])
model = encoder.fit(df)
encoded = model.transform(df)
encoded.show()

```

```
+--------------+--------------+-------------+-------------+
|categoryIndex1|categoryIndex2| categoryVec1| categoryVec2|
+--------------+--------------+-------------+-------------+
|           0.0|           1.0|(2,[0],[1.0])|(2,[1],[1.0])|
|           1.0|           0.0|(2,[1],[1.0])|(2,[0],[1.0])|
|           2.0|           1.0|    (2,[],[])|(2,[1],[1.0])|
|           0.0|           2.0|(2,[0],[1.0])|    (2,[],[])|
|           0.0|           1.0|(2,[0],[1.0])|(2,[1],[1.0])|
|           2.0|           0.0|    (2,[],[])|(2,[0],[1.0])|
+--------------+--------------+-------------+-------------+
```


**4, MinMax标准化**

```python
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors

df = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.1, -1.0]),),
    (1, Vectors.dense([2.0, 1.1, 1.0]),),
    (2, Vectors.dense([3.0, 10.1, 3.0]),)
], ["id", "features"])

scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

scalerModel = scaler.fit(df)

df_scaled = scalerModel.transform(df)
print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
df_scaled.select("features", "scaledFeatures").show()

```

```
Features scaled to range: [0.000000, 1.000000]
+--------------+--------------+
|      features|scaledFeatures|
+--------------+--------------+
|[1.0,0.1,-1.0]|     (3,[],[])|
| [2.0,1.1,1.0]| [0.5,0.1,0.5]|
|[3.0,10.1,3.0]| [1.0,1.0,1.0]|
+--------------+--------------+
```


**5，MaxAbsScaler标准化**

```python
from pyspark.ml.feature import MaxAbsScaler
from pyspark.ml.linalg import Vectors

df = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.1, -8.0]),),
    (1, Vectors.dense([2.0, 1.0, -4.0]),),
    (2, Vectors.dense([4.0, 10.0, 8.0]),)
], ["id", "features"])

scaler = MaxAbsScaler(inputCol="features", outputCol="scaledFeatures")

scalerModel = scaler.fit(df)

df_rescaled = scalerModel.transform(df)

df_rescaled.select("features", "scaledFeatures").show()

```

```
+--------------+--------------------+
|      features|      scaledFeatures|
+--------------+--------------------+
|[1.0,0.1,-8.0]|[0.25,0.010000000...|
|[2.0,1.0,-4.0]|      [0.5,0.1,-0.5]|
|[4.0,10.0,8.0]|       [1.0,1.0,1.0]|
+--------------+--------------------+
```


**6，SQLTransformer**


可以使用SQL语法将DataFrame进行转换，等效于注册表的作用。

但它可以用于Pipeline中作为Transformer.

```python
from pyspark.ml.feature import SQLTransformer

df = spark.createDataFrame([
    (0, 1.0, 3.0),
    (2, 2.0, 5.0)
], ["id", "v1", "v2"])
sqlTrans = SQLTransformer(
    statement="SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")

sqlTrans.transform(df).show()
```

```
+---+---+---+---+----+
| id| v1| v2| v3|  v4|
+---+---+---+---+----+
|  0|1.0|3.0|4.0| 3.0|
|  2|2.0|5.0|7.0|10.0|
+---+---+---+---+----+
```


**7, Imputer**


Imputer转换器可以填充缺失值，缺失值可以用 float("nan")来表示。

```python
from pyspark.ml.feature import Imputer

df = spark.createDataFrame([
    (1.0, float("nan")),
    (2.0, float("nan")),
    (float("nan"), 3.0),
    (4.0, 4.0),
    (5.0, 5.0)
], ["a", "b"])

imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])
model = imputer.fit(df)

model.transform(df).show()

```

```
+---+---+-----+-----+
|  a|  b|out_a|out_b|
+---+---+-----+-----+
|1.0|NaN|  1.0|  4.0|
|2.0|NaN|  2.0|  4.0|
|NaN|3.0|  3.0|  3.0|
|4.0|4.0|  4.0|  4.0|
|5.0|5.0|  5.0|  5.0|
+---+---+-----+-----+
```


### 四，分类模型


Mllib支持常见的机器学习分类模型：逻辑回归，SoftMax回归，决策树，随机森林，梯度提升树，线性支持向量机，朴素贝叶斯，One-Vs-Rest，以及多层感知机模型。这些模型的接口使用方法基本大同小异，下面仅仅列举常用的决策树，随机森林和梯度提升树的使用作为示范。更多范例参见官方文档。


**1，决策树**

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 载入数据
dfdata = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")
(dftrain, dftest) = dfdata.randomSplit([0.7, 0.3])

# 对label进行序号标注，将字符串换成整数序号
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(dfdata)

# 处理分类特征，类别如果超过4将视为连续值
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(dfdata)

# 构建一个决策树模型
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# 构建流水线
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# 训练流水线
model = pipeline.fit(dftrain)

dfpredictions = model.transform(dftest)

dfpredictions.select("prediction", "indexedLabel", "features").show(5)

# 评估模型误差
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(dfpredictions)
print("Test Error = %g " % (1.0 - accuracy))
treeModel = model.stages[2]
print(treeModel)

```

```
+----------+------------+--------------------+
|prediction|indexedLabel|            features|
+----------+------------+--------------------+
|       1.0|         1.0|(692,[98,99,100,1...|
|       1.0|         1.0|(692,[124,125,126...|
|       1.0|         1.0|(692,[124,125,126...|
|       1.0|         1.0|(692,[125,126,127...|
|       1.0|         1.0|(692,[126,127,128...|
+----------+------------+--------------------+
only showing top 5 rows

Test Error = 0.037037 
DecisionTreeClassificationModel: uid=DecisionTreeClassifier_5711dbfcd91e, depth=2, numNodes=5, numClasses=2, numFeatures=692

```


**2，随机森林**

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 载入数据
dfdata = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")
(dftrain, dftest) = dfdata.randomSplit([0.7, 0.3])

# 对label进行序号标注，将字符串换成整数序号
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(dfdata)

# 处理类别特征
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(dfdata)


# 使用随机森林模型
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# 将label重新转换成字符串
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# 构建流水线
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# 训练流水线
model = pipeline.fit(dftrain)

# 进行预测
dfpredictions = model.transform(dftest)

dfpredictions.select("predictedLabel", "label", "features").show(5)

# 评估模型
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(dfpredictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  

```

```
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|           0.0|  0.0|(692,[122,123,124...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[124,125,126...|
+--------------+-----+--------------------+
only showing top 5 rows

Test Error = 0
RandomForestClassificationModel: uid=RandomForestClassifier_9d8f7dfec86b, numTrees=10, numClasses=2, numFeatures=692
```


**3，梯度提升树**

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 载入数据
dfdata = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")
(dftrain, dftest) = dfdata.randomSplit([0.7, 0.3])

# 对label进行序号标注，将字符串换成整数序号
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(dfdata)

# 处理类别特征
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(dfdata)

# 使用梯度提升树模型
gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=20)

# 构建流水线
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

# 训练流水线
model = pipeline.fit(dftrain)

# 进行预测
dfpredictions = model.transform(dftest)
dfpredictions.select("prediction", "indexedLabel", "features").show(5)

# 评估模型
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(dfpredictions)
print("Test Error = %g" % (1.0 - accuracy))

gbtModel = model.stages[2]
print(gbtModel)  

```

```
+----------+------------+--------------------+
|prediction|indexedLabel|            features|
+----------+------------+--------------------+
|       1.0|         1.0|(692,[95,96,97,12...|
|       1.0|         1.0|(692,[98,99,100,1...|
|       1.0|         1.0|(692,[122,123,148...|
|       1.0|         1.0|(692,[124,125,126...|
|       1.0|         1.0|(692,[124,125,126...|
+----------+------------+--------------------+
only showing top 5 rows

Test Error = 0.0689655
GBTClassificationModel: uid = GBTClassifier_e3d7713552b3, numTrees=20, numClasses=2, numFeatures=692
```


### 五，回归模型


Mllib支持常见的回归模型，如线性回归，广义线性回归，决策树回归，随机森林回归，梯度提升树回归，生存回归，保序回归。

下面仅以线性回归和决策树回归为例。


**1，线性回归**

```python
from pyspark.ml.regression import LinearRegression

# 载入数据
dfdata = spark.read.format("libsvm")\
   .load("data/sample_linear_regression_data.txt")

# 定义模型
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# 训练模型
lrModel = lr.fit(dfdata)

# 模型参数
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# 评估模型
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


```

```
Coefficients: [0.0,0.32292516677405936,-0.3438548034562218,1.9156017023458414,0.05288058680386263,0.765962720459771,0.0,-0.15105392669186682,-0.21587930360904642,0.22025369188813426]
Intercept: 0.1598936844239736
numIterations: 7
objectiveHistory: [0.49999999999999994, 0.4967620357443381, 0.4936361664340463, 0.4936351537897608, 0.4936351214177871, 0.49363512062528014, 0.4936351206216114]
+--------------------+
|           residuals|
+--------------------+
|  -9.889232683103197|
|  0.5533794340053554|
|  -5.204019455758823|
| -20.566686715507508|
|    -9.4497405180564|
|  -6.909112502719486|
|  -10.00431602969873|
|   2.062397807050484|
|  3.1117508432954772|
| -15.893608229419382|
|  -5.036284254673026|
|   6.483215876994333|
|  12.429497299109002|
|  -20.32003219007654|
| -2.0049838218725005|
| -17.867901734183793|
|   7.646455887420495|
| -2.2653482182417406|
|-0.10308920436195645|
|  -1.380034070385301|
+--------------------+
only showing top 20 rows

RMSE: 10.189077
r2: 0.022861

```


**2，决策树回归**

```python
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# 载入数据
dfdata = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")
(dftrain, dftest) = dfdata.randomSplit([0.7, 0.3])

# 处理类别特征
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(dfdata)

# 使用决策树模型
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

# 构建流水线
pipeline = Pipeline(stages=[featureIndexer, dt])

# 训练流水线
model = pipeline.fit(dftrain)

# 进行预测
dfpredictions = model.transform(dftest)
dfpredictions.select("prediction", "label", "features").show(5)

# 评估模型
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(dfpredictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

treeModel = model.stages[1]
print(treeModel)

```

```
+----------+-----+--------------------+
|prediction|label|            features|
+----------+-----+--------------------+
|       0.0|  0.0|(692,[123,124,125...|
|       0.0|  0.0|(692,[124,125,126...|
|       0.0|  0.0|(692,[126,127,128...|
|       0.0|  0.0|(692,[126,127,128...|
|       0.0|  0.0|(692,[126,127,128...|
+----------+-----+--------------------+
only showing top 5 rows

Root Mean Squared Error (RMSE) on test data = 0
DecisionTreeRegressionModel: uid=DecisionTreeRegressor_06213a3aaeb0, depth=2, numNodes=5, numFeatures=692
```


### 六，聚类模型


Mllib支持的聚类模型较少，主要有K均值聚类，高斯混合模型GMM，以及二分的K均值，隐含狄利克雷分布LDA模型等。


**1，K均值聚类**

```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# 载入数据
dfdata = spark.read.format("libsvm").load("data/sample_kmeans_data.txt")

# 训练Kmeans模型
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dfdata)

# 进行预测
dfpredictions = model.transform(dfdata)

# 评估模型
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(dfpredictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# 打印中心点
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
```

```
Silhouette with squared euclidean distance = 0.9997530305375207
Cluster Centers: 
[9.1 9.1 9.1]
[0.1 0.1 0.1]
```


**2，高斯混合模型**

```python
from pyspark.ml.clustering import GaussianMixture

dfdata = spark.read.format("libsvm").load("data/sample_kmeans_data.txt")

gmm = GaussianMixture().setK(2).setSeed(538009335)
model = gmm.fit(dfdata)

print("Gaussians shown as a DataFrame: ")
model.gaussiansDF.show(truncate=True)


```

```
aussians shown as a DataFrame: 
+--------------------+--------------------+
|                mean|                 cov|
+--------------------+--------------------+
|[0.10000000000001...|0.006666666666806...|
|[9.09999999999998...|0.006666666666812...|
+--------------------+--------------------+
```


**3, 二分K均值 Bisecting k-means**


Bisecting k-means是一种自上而下的层次聚类算法。所有的样本点开始时属于一个cluster,然后不断通过K均值二分裂得到多个cluster。

```python
from pyspark.ml.clustering import BisectingKMeans


dfdata = spark.read.format("libsvm").load("data/sample_kmeans_data.txt")

bkm = BisectingKMeans().setK(2).setSeed(1)
model = bkm.fit(dfdata)

cost = model.computeCost(dfdata)
print("Within Set Sum of Squared Errors = " + str(cost))


print("Cluster Centers: ")
centers = model.clusterCenters()
for center in centers:
    print(center)
    
```

```
Within Set Sum of Squared Errors = 0.11999999999994547
Cluster Centers: 
[0.1 0.1 0.1]
[9.1 9.1 9.1]
```


### 七，降维模型


Mllib中支持的降维模型只有主成分分析PCA算法。这个模型在spark.ml.feature中，通常作为特征预处理的一种技巧使用。

```python
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
dfdata = spark.createDataFrame(data, ["features"])

pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(dfdata)

dfresult = model.transform(dfdata).select("pcaFeatures")
dfresult.show(truncate=False)



```

```
+-----------------------------------------------------------+
|pcaFeatures                                                |
+-----------------------------------------------------------+
|[1.6485728230883807,-4.013282700516296,-5.524543751369388] |
|[-4.645104331781534,-1.1167972663619026,-5.524543751369387]|
|[-6.428880535676489,-5.337951427775355,-5.524543751369389] |
+-----------------------------------------------------------+
```

### 八，模型优化


模型优化一般也称作模型选择(Model selection)或者超参调优(hyperparameter tuning)。

Mllib支持网格搜索方法进行超参调优，相关函数在spark.ml.tunning模块中。

有两种使用网格搜索方法的模式，一种是通过交叉验证(cross-validation)方式进行使用，另外一种是通过留出法(hold-out)方法进行使用。

交叉验证模式使用的是K-fold交叉验证，将数据随机等分划分成K份，每次将一份作为验证集，其余作为训练集，根据K次验证集的平均结果来决定超参选取，计算成本较高，但是结果更加可靠。

而留出法只用将数据随机划分成训练集和验证集，仅根据验证集的单次结果决定超参选取，结果没有交叉验证可靠，但计算成本较低。

如果数据规模较大，一般选择留出法，如果数据规模较小，则应该选择交叉验证模式。



**1，交叉验证模式**

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# 准备数据
dfdata = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0),
    (4, "b spark who", 1.0),
    (5, "g d a y", 0.0),
    (6, "spark fly", 1.0),
    (7, "was mapreduce", 0.0),
    (8, "e spark program", 1.0),
    (9, "a e c l", 0.0),
    (10, "spark compile", 1.0),
    (11, "hadoop software", 0.0)
], ["id", "text", "label"])

# 构建流水线，包含： tokenizer, hashingTF, lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# 现在我们将整个流水线视作一个Estimator进行统一的超参数调优
# 构建网格： hashingTF.numFeatures 有 3 个可选值  and lr.regParam 有2个可选值
# 我们的网格空间总共有2*3=6个点需要搜索
paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

# 创建5折交叉验证超参调优器
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=5) 

# fit后会输出最优的模型
cvModel = crossval.fit(dfdata)

# 准备预测数据
test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "mapreduce spark"),
    (7, "apache hadoop")
], ["id", "text"])

# 使用最优模型进行预测
prediction = cvModel.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    print(row)
```

```
Row(id=4, text='spark i j k', probability=DenseVector([0.2661, 0.7339]), prediction=1.0)
Row(id=5, text='l m n', probability=DenseVector([0.9209, 0.0791]), prediction=0.0)
Row(id=6, text='mapreduce spark', probability=DenseVector([0.4429, 0.5571]), prediction=1.0)
Row(id=7, text='apache hadoop', probability=DenseVector([0.8584, 0.1416]), prediction=0.0)
```


**2，留出法模式**

```python
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# 准备数据
dfdata = spark.read.format("libsvm")\
    .load("data/sample_linear_regression_data.txt")
dftrain, dftest = dfdata.randomSplit([0.9, 0.1], seed=12345)

lr = LinearRegression(maxIter=10)

# 构建网格作为超参数搜索空间
paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

# 创建留出法超参调优器
tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator(),
                           # 80% 的数据作为训练集，20的数据作为验证集
                           trainRatio=0.8)

# 训练后会输出最优超参的模型
model = tvs.fit(dftrain)

# 使用模型进行预测
model.transform(dftest)\
    .select("features", "label", "prediction")\
    .show()

```

```
+--------------------+--------------------+--------------------+
|            features|               label|          prediction|
+--------------------+--------------------+--------------------+
|(10,[0,1,2,3,4,5,...| -17.026492264209548| -1.6265106840933026|
|(10,[0,1,2,3,4,5,...|  -16.71909683360509|-0.01129960392982...|
|(10,[0,1,2,3,4,5,...| -15.375857723312297|  0.9008270143746643|
|(10,[0,1,2,3,4,5,...| -13.772441561702871|   3.435609049373433|
|(10,[0,1,2,3,4,5,...| -13.039928064104615|  0.3670260850771136|
|(10,[0,1,2,3,4,5,...|   -9.42898793151394|   -3.26399994121536|
|(10,[0,1,2,3,4,5,...|    -9.2679651250406| -0.1762581278405398|
|(10,[0,1,2,3,4,5,...|  -9.173693798406978| -0.2824541263038875|
|(10,[0,1,2,3,4,5,...| -7.1500991588127265|   3.087239142258043|
|(10,[0,1,2,3,4,5,...|  -6.930603551528371| 0.12389571117374062|
|(10,[0,1,2,3,4,5,...|  -6.456944198081549| -0.7275144195427645|
|(10,[0,1,2,3,4,5,...| -3.2843694575334834| -0.9048235164747517|
|(10,[0,1,2,3,4,5,...|   -1.99891354174786|  0.9588887587748192|
|(10,[0,1,2,3,4,5,...| -0.4683784136986876|  0.6261083785799368|
|(10,[0,1,2,3,4,5,...|-0.44652227528840105| 0.19068393875752507|
|(10,[0,1,2,3,4,5,...| 0.10157453780074743| -0.9062122256799047|
|(10,[0,1,2,3,4,5,...|  0.2105613019270259|   1.225604620956131|
|(10,[0,1,2,3,4,5,...|  2.1214592666251364|  0.2854396644518767|
|(10,[0,1,2,3,4,5,...|  2.8497179990245116|  1.3569268250561075|
|(10,[0,1,2,3,4,5,...|   3.980473021620311|  2.5359695420417965|
+--------------------+--------------------+--------------------+
only showing top 20 rows
```


### 九，实用工具


pyspark.ml.linalg模块提供了线性代数向量和矩阵对象。

pyspark.ml.stat模块提供了数理统计诸如卡方检验，相关性分析等功能。


**1，向量和矩阵**


pyspark.ml.linalg 支持 DenseVector，SparseVector，DenseMatrix，SparseMatrix类。

并可以使用Matrices和Vectors提供的工厂方法创建向量和矩阵。

```python
from pyspark.ml.linalg import DenseVector, SparseVector


#稠密向量
dense_vec = DenseVector([1, 0, 0, 2.0, 0])

print("dense_vec: ", dense_vec)
print("dense_vec.numNonzeros: ", dense_vec.numNonzeros())


#稀疏向量
#参数分别是维度，非零索引，非零元素值
sparse_vec = SparseVector(5, [0,3],[1.0,2.0])  
print("sparse_vec: ", sparse_vec)


```

```
dense_vec:  [1.0,0.0,0.0,2.0,0.0]
dense_vec.numNonzeros:  2
sparse_vec:  (5,[0,3],[1.0,2.0])
```

```python
dense_vec.toArray()
```

```
array([1., 0., 0., 2., 0.])
```

```python
from pyspark.ml.linalg import DenseMatrix, SparseMatrix

#稠密矩阵
#参数分别是 行数，列数，元素值，是否转置(默认False)
dense_matrix = DenseMatrix(3, 2, [1, 3, 5, 2, 4, 6])


#稀疏矩阵
#参数分别是 行数，列数，在第几个元素列索引加1，行索引，非零元素值
sparse_matrix = SparseMatrix(3, 3, [0, 2, 3, 6],
    [0, 2, 1, 0, 1, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

print("sparse_matrix.toArray(): \n", sparse_matrix.toArray())


```

```
sparse_matrix.toArray(): 
 [[1. 0. 4.]
 [0. 3. 5.]
 [2. 0. 6.]]
```

```python
from pyspark.ml.linalg import Vectors,Matrices

#工厂方法
vec = Vectors.zeros(3)
matrix = Matrices.dense(2,2,[1,2,3,5])

print(matrix)
```

```
DenseMatrix([[1., 3.],
             [2., 5.]])
```


**2,数理统计**

```python
#相关性分析
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation

data = [(Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),
        (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
        (Vectors.dense([6.0, 7.0, 0.0, 8.0]),),
        (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),)]
df = spark.createDataFrame(data, ["features"])

r1 = Correlation.corr(df, "features").head()
print("Pearson correlation matrix:\n" + str(r1[0]))

r2 = Correlation.corr(df, "features", "spearman").head()
print("Spearman correlation matrix:\n" + str(r2[0]))
```

```
Pearson correlation matrix:
DenseMatrix([[1.        , 0.05564149,        nan, 0.40047142],
             [0.05564149, 1.        ,        nan, 0.91359586],
             [       nan,        nan, 1.        ,        nan],
             [0.40047142, 0.91359586,        nan, 1.        ]])
Spearman correlation matrix:
DenseMatrix([[1.        , 0.10540926,        nan, 0.4       ],
             [0.10540926, 1.        ,        nan, 0.9486833 ],
             [       nan,        nan, 1.        ,        nan],
             [0.4       , 0.9486833 ,        nan, 1.        ]])
```

```python
#卡方检验
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest

data = [(0.0, Vectors.dense(0.5, 10.0)),
        (0.0, Vectors.dense(1.5, 20.0)),
        (1.0, Vectors.dense(1.5, 30.0)),
        (0.0, Vectors.dense(3.5, 30.0)),
        (0.0, Vectors.dense(3.5, 40.0)),
        (1.0, Vectors.dense(3.5, 40.0))]
df = spark.createDataFrame(data, ["label", "features"])

r = ChiSquareTest.test(df, "features", "label").head()
print("pValues: " + str(r.pValues))
print("degreesOfFreedom: " + str(r.degreesOfFreedom))
print("statistics: " + str(r.statistics))
```
```
pValues: [0.6872892787909721,0.6822703303362126]
degreesOfFreedom: [2, 3]
statistics: [0.75,1.5]

```
