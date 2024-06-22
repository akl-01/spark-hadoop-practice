FROM bitnami/spark:latest

USER root
RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install numpy==1.26.4
RUN pip3 install catboost
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install psutil
RUN pip3 install pyspark
RUN pip3 install tqdm

USER 1001

COPY ./metric/ram_time_values.csv /output/ram_time_values.csv

CMD ["bin/spark-class", "org.apache.spark.deploy.master.Master"]