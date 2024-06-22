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

def get_data() -> Any:
    logger.info(f"Getting real time data")
    start_time = datetime.datetime.fromisoformat('2023-01-01T00:00:01')
    end_time = datetime.datetime.fromisoformat('2023-01-01T01:00:01')
    events = search(starttime=start_time, endtime=end_time)
    logger.info(f"Returned {len(events)} events")

    return events

def create_dataset() -> None:
    events = get_data()

    logger.info("Start dataset creating")
    dataset = pd.DataFrame(columns=['sig', 'cdi', 'longitude', 'latitude', 'depth', 'gap', 'dmin', 'nst',
                                     'mmi', 'official', 'magnitude'])

    for event in tqdm(events):
        with urllib.request.urlopen(event["detail"]) as url:
            data = json.load(url)["properties"]
        data["real_time"] = 1
        data = pd.DataFrame(data)
        data = data.loc["origin"].to_frame().transpose().reset_index()
        data = data.drop(["index"], axis=1)
        dataset = pd.concat((dataset, data))
    
    logger.info(f"Create dataset with len = {len(dataset)}")
    
    dataset.to_csv("data/earthquake_2023-2024.csv")

def main():
    create_dataset()

if __name__ == "__main__":
    sys.exit(main() or 0)