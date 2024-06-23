# Spark-Hadoop Practice

## Results

## Generate Dataset

## Run
> Python3.10.12 is used

Clone the repository:
```bash
git clone https://github.com/akl-01/spark-hadoop-practice.git
cd spark-hadoop-practice/
```

### Install requirements
```bash
pip install -r requirements.txt
```

### Up docker-compose
1. Run docker-compose as demon:
```bash
docker-compose up --build -d
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

- To down docker containers via docker-compose:
```bash
docker-compose down
``` 

### Put the dataset into HDFS
1. Put the dataset into namenode:
```bash
docker cp data/earthquake.csv namenode:/
```

3. Go to namenode command line:
```bash
docker exec -it namenode /bin/bash
```

4. Put the dataset to root of HDFS
```bash
hdfs dfs -put earthquake.csv /
```

### Run
1. Run the application:
```bash
bash ./run.sh <type>
```

Argument:
- `type`: possible values *simple* (without optimization) and *optimal* (with optimization).