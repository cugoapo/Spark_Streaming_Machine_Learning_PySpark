from warnings import filterwarnings
filterwarnings("ignore")

import findspark
findspark.init("/Users/alperhatipoglu/dev/Apache-Spark/spark-3.0.0-preview2-bin-hadoop2.7")

from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


#%% Spark Session

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("Train_Pipeline_ML_model") \
    .getOrCreate()

#%% Read The Dataset

df = spark.read.csv("/Users/alperhatipoglu/PycharmProjects/Spark/Streaming/DATASET/simple_data.csv",
                    header=True, sep=",", inferSchema=True)

df.show()
df.printSchema()

#%% Label Column
df1 = df.withColumn("ekonomik_durum", f.when(f.col("aylik_gelir") > 7000, "iyi").otherwise("kötü"))
df1.show()

#%% Estimators and Transformers

meslek_indexer = StringIndexer() \
    .setInputCol("meslek") \
    .setOutputCol("meslek_indexer") \
    .setHandleInvalid("skip")

sehir_indexer = StringIndexer() \
    .setInputCol("sehir") \
    .setOutputCol("sehir_indexer") \
    .setHandleInvalid("skip")

ohe = OneHotEncoder() \
    .setInputCols(["meslek_indexer", "sehir_indexer"]) \
    .setOutputCols(["meslek_encoded", "sehir_encoded"])

vc_assembler = VectorAssembler() \
    .setInputCols(["yas", "meslek_encoded", "sehir_encoded", "aylik_gelir"]) \
    .setOutputCol("vectorized_features")

ekonomik_durum_indexer = StringIndexer() \
    .setInputCol("ekonomik_durum") \
    .setOutputCol("label")

sc = StandardScaler() \
    .setInputCol("vectorized_features") \
    .setOutputCol("features")

rf = RandomForestClassifier() \
    .setFeaturesCol("features") \
    .setLabelCol("label") \
    .setPredictionCol("prediction")

#%% Split The Dataset

train_df, test_df = df1.randomSplit([0.8, 0.2], seed=7)


#%% Pipeline Object

pipeline_obj = Pipeline() \
    .setStages([meslek_indexer, sehir_indexer, ohe, vc_assembler, ekonomik_durum_indexer, sc, rf])

#%% Pipeline Model

pipeline_model = pipeline_obj.fit(train_df)
result_df = pipeline_model.transform(test_df)

#%% Show Results

evaluator = MulticlassClassificationEvaluator() \
    .setPredictionCol("prediction") \
    .setLabelCol("label")

#%% Evaluation
accuracy = evaluator.evaluate(result_df, {evaluator.metricName: "accuracy"})

'''
evaluator.evaluate(dataset, {evaluator.metricName: "accuracy"})
'''

#%% Save The Pipeline Model
path = "/Users/alperhatipoglu/PycharmProjects/Spark/Streaming" + "/pipeline_model"
pipeline_model.save(path)








