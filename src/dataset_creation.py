import pandas as pd
import sys
import logging as log
from typing import Any
import json
from tqdm import tqdm

import urllib.request
import datetime
from libcomcat.search import search 

logger = log.getLogger(__name__)
logger.setLevel(log.INFO)
console = log.StreamHandler()
console_formater = log.Formatter("[ %(levelname)s ] %(message)s")
console.setFormatter(console_formater)
logger.addHandler(console)

def get_data() -> pd.DataFrame:
    dataset = pd.read_csv("data/earthquake_small.csv", index_col=0)
    return dataset


def create_dataset() -> None:
    start = get_data()
    dataset = pd.DataFrame(columns=['sig', 'cdi', 'longitude', 
                                    'latitude', 'depth', 'gap', 
                                    'dmin', 'nst', 'mmi',
                                    'official', 'magnitude'])
    logger.info("Start dataset creating")

    for _ in tqdm(range(110)):
        start = start.sample(frac=1)
        dataset = pd.concat((start, dataset))
    
    logger.info(f"Create dataset with len = {len(dataset)}")
    dataset = dataset.reset_index(drop=True)
    dataset.to_csv("data/earthquake_medium.csv")

def main():
    create_dataset()

if __name__ == "__main__":
    sys.exit(main() or 0)