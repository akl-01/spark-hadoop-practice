#!/bin/bash

docker_logs=$(docker logs spark-master 2>&1)
spark_master_ip=$(echo ${docker_logs} | grep -oP 'Starting Spark master at spark://\K[0-9.]+:[0-9]+')

if [ $1 == "simple" ]; then
    docker cp -L app/spark.py spark-master:/opt/bitnami/spark/spark.py
    docker cp -L app/utils.py spark-master:/opt/bitnami/spark/utils.py
    docker-compose exec spark-master spark-submit --master spark://$spark_master_ip spark.py
elif [ $1 == "optimal" ]; then
    echo $1
else
    echo "incorrect type"
fi
