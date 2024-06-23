import pandas as pd
from pyspark.sql import SparkSession
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import gc

from utils import preprocess, get_memory_usage
from typing import Tuple

import time
from tqdm import tqdm
import logging as log
import warnings
warnings.filterwarnings("ignore")


logger = log.getLogger(__name__)
logger.setLevel(log.INFO)
console = log.StreamHandler()
console_formater = log.Formatter("[ %(levelname)s ] %(message)s")
console.setFormatter(console_formater)
logger.addHandler(console)

class SimpleSparkSession:
    def __init__(self) -> None:
        self.spark = SparkSession.builder.appName("My App 1").getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        self.model = CatBoostRegressor(
            iterations=4200,
            learning_rate=1,
            depth=3,
            loss_function="RMSE"
        ).load_model("/output/model.pkl")
       
        self.times = []
        self.rams = []

    def get_dataset(self) -> Tuple[pd.DataFrame, int]:
        logger.info(f"\tStart the read from hadoop")
        self.raw_data = self.spark.read.csv("hdfs://namenode:9001/earthquake_small.csv", header=True) # Change on eqarthquake
        logger.info(f"\tEnd the read from hadoop")
        self.dataset = self.raw_data.toPandas()
        return self.dataset, len(self.dataset)
    
    def reset(self):
        logger.info("\tReset datasets")
        self.raw_data.unpersist()
        del self.raw_data, self.dataset
        gc.collect()
    
    def dump(self):
        with open('/output/ram_time_values.csv', 'w') as file:
            file.write("Index,Execution Time (seconds),RAM Usage (B)\n")
            for i in range(100):
                file.write(f"{i+1},{self.times[i]},{self.rams[i]}\n")
    
    def start(self):
        for exp in range(100):
            logger.info(f"Experiment={exp}")
            dataset, length = self.get_dataset()
            start = time.time()
            start_memory_usage = get_memory_usage()
            logger.info(f"\t{start_memory_usage=}")
            mean_mae = 0
            for idx in range(length):
                dataset_el = dataset.iloc[[idx]]
                dataset_el = preprocess(dataset_el)
                if len(dataset_el) == 0:
                    continue
                X, y = dataset_el.drop(["magnitude"], axis=1), dataset_el["magnitude"]
                predict = self.model.predict(X)
                mae = mean_absolute_error(y, predict)
                mean_mae += mae
            mean_mae = mean_mae / length
            end = time.time()
            end_memory_usage = get_memory_usage()
            logger.info(f"\t{end_memory_usage=}")
            self.times.append(end - start)
            self.rams.append(np.abs(end_memory_usage - start_memory_usage))
            
            self.reset()
        self.spark.stop()

logger.info("Start SimpleSparkSession")
simple_spark_session = SimpleSparkSession()
logger.info("Startng read and process data")
simple_spark_session.start()
logger.info("Dump data to csv")
simple_spark_session.dump()
logger.info("End")