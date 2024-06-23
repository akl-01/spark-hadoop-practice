#!/bin/bash

target=metric

mkdir -p $target

pushd $target
touch ram_time_values.csv
chmod ugo+rwx ram_time_values.csv
popd