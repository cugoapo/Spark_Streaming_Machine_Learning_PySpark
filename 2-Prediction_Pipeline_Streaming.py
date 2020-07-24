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


#%% Spark Session

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("Prediction_Pipeline_Streaming") \
    .getOrCreate()

#%% Create manual Schema

schema_simple_data = StructType([
    StructField("sira_no", IntegerType(), True),
    StructField("isim", StringType(), True),
    StructField("yas", IntegerType(), True),
    StructField("meslek", StringType(), True),
    StructField("sehir", StringType(), True),
    StructField("aylik_gelir", IntegerType(), True),
    ])

#%% Read Streaming Data

dff = spark.readStream.csv("/Users/alperhatipoglu/PycharmProjects/Spark/Streaming/Read_Streaming_Data",
                           header=True, schema=schema_simple_data, sep=",")

#%% Load Model and Pipeline(Estimators and Transformers)

pipeline = PipelineModel.load("/Users/alperhatipoglu/PycharmProjects/Spark/Streaming/pipeline_model")

pipeline_obj = Pipeline() \
    .setStages([pipeline.stages[0], pipeline.stages[1], pipeline.stages[2],
                pipeline.stages[3], pipeline.stages[5]])

pipeline_model = pipeline_obj.fit(dff)
dff2 = pipeline_model.transform(dff)

rf_model = pipeline.stages[6]

result_df = rf_model.transform(dff2).select(["isim", "yas", "meslek", "sehir", "aylik_gelir", "probability", "prediction"])

#%% Create Query

query = result_df.writeStream \
    .format("console") \
    .start()

#%%
query.awaitTermination(timeout=30)

#%%
query.stop()



