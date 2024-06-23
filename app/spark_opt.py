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

class OptimalSparkSession:
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
    
    def step(self, df: pd.DataFrame):
        dataset_el = preprocess(df)
        if len(dataset_el) == 0:
            return 0 
        X, y = dataset_el.drop(["magnitude"], axis=1), dataset_el["magnitude"]
        predict = self.model.predict(X)
        mae = mean_absolute_error(y, predict)
        return mae

    def step_wrapper(self, df: pd.DataFrame | pd.Series):
        return self.step(df)
    
    def start(self):
        for exp in range(100):
            logger.info(f"Experiment={exp}")
            dataset, _ = self.get_dataset()
            start = time.time()
            start_memory_usage = get_memory_usage()
            logger.info(f"\t{start_memory_usage=}")
            opt = self.spark.sparkContext.parallelize(dataset.iterrows())
            returns_opt = opt.map(self.step_wrapper)
            end = time.time()
            end_memory_usage = get_memory_usage()
            logger.info(f"\t{end_memory_usage=}")
            self.times.append(end - start)
            self.rams.append(np.abs(end_memory_usage - start_memory_usage))
        self.spark.stop()

logger.info("Start SimpleSparkSession")
optimal_spark_session = OptimalSparkSession()
logger.info("Startng read and process data")
optimal_spark_session.start()
logger.info("Dump data to csv")
optimal_spark_session.dump()
logger.info("End")