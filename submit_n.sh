#!/bin/bash
n=$1
temp=$(date +%d_%m-%H_%M_%S )
for ((i=1; i <= $n; i++));
do
 sbatch submit.sh $temp $i
 sleep 5
done