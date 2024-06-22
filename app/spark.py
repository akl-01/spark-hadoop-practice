import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
from catboost import CatBoostRegressor
from pyspark.sql.functions import col, collect_list
from sklearn.metrics import mean_absolute_error
import gc

from utils import preprocess, get_memory_usage
from types import Tuple

import time
import sys

class SimpleSparkSession:
    def __init__(self) -> None:
        self.spark = SparkSession.builder.appName("My App 1").getOrCreate()
        self.model = CatBoostRegressor(
            iterations=4200,
            learning_rate=1,
            depth=3,
            loss_function="RMSE"
        ).load_model("../weights/model.pkl")
        self.times = []
        self.rams = []
    
    def get_dataset(self) -> Tuple[pd.DataFrame, int]:
        self.raw_data = self.spark.read.csv("hdfs://namenode:9001/earthquake_2023-2024.csv", header=True)
        self.dataset = self.raw_data.toPandas()
        self.dataset.dropna(inplace=True)
        return self.dataset, len(self.dataset)
    
    def reset(self):
        self.raw_data.unpersist()
        del self.raw_data, self.dataset
        gc.collect()
    
    def dump(self):
        with open('/output/ram_time_values.csv', 'w') as file:
            file.write("Index,Execution Time (seconds),RAM Usage (B)\n")
            for i in range(100):
                file.write(f"{i+1},{self.times[i]},{self.rams[i]}\n")
    
    def start(self):
        for _ in range(100):
            dataset, length = self.get_dataset()
            preprocess_dataset = preprocess(dataset)
            X, y = preprocess_dataset.drop(["magnitude"], axis=1), preprocess_dataset["magnitude"]
            start = time.time()
            start_memory_usage = get_memory_usage()
            mean_mae = 0
            for idx in range(length):
                predict = self.model.predict(X.iloc[idx])
                mae = mean_absolute_error(y.iloc[idx], predict)
                mean_mae += mae
            mean_mae = mean_mae / length
            end = time.end()
            end_memory_usage = get_memory_usage()
            self.times.append(end - start)
            self.rams.append(end_memory_usage - start_memory_usage)
            
            self.reset()
        self.spark.stop()

def main():
    simple_spark_session = SimpleSparkSession()
    simple_spark_session.start()
    simple_spark_session.dump()

if __name__ == "__main__":
    sys.exit(main() or 0)