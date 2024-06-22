# Spark-Hadoop Practice

## Results

## Generate Dataset

## Run

> Python3.10.12 is used

### Install requirements

### Up docker-compose
1. Go to `configuration` directory:
```bash
cd config
```

2. Run docker-compose as demon:
```bash
docker-compose up -d 
```

Useful commands for docker containers manipulation:
- To see all docker containers:
```bash
docker ps
```

- To stop all docker containers:
```bash
docker stop $(docker ps -a -q)
```

- To delete all docker containers:
```bash
docker rm $(docker ps -a -q)
```

### Put the dataset into HDFS
1. Go to `spark-hadoop-lab` directory:

2. Put the dataset into namenode:
```bash
docker cp data/earthquake_2023-2024.csv namenode:/
```

3. Go to namenode command line:
```bash
docker exec -it namenode /bin/bash
```

4. Put the dataset to root of HDFS
```bash
hdfs dfs -put earthquake_2023-2024.csv /
```

### Run