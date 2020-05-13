#!/bin/bash

set -e

# build_meta_data.sh --sub_task en-fr,en-de,de-fr --n_samples $n_samples --sub_task_data $sub_task_data

SUB_TASKS_DATA_PERCENT=${1-""}
N_SAMPLES=${2-'false'}


if [ $SUB_TASKS_DATA_PERCENT = "" ];then
    exit
fi

sub_tasks=""
data_percent=""

for task_data_percent in $(echo $SUB_TASKS_DATA_PERCENT | sed -e 's/\,/ /g'); do
    IFS=': ' read -r -a array <<< "$task_data_percent"
    sub_tasks=$sub_tasks,${array[0]}
    data_percent=$data_percent,${array[1]}
done

sub_tasks=$(echo $sub_tasks | cut -c2-)
data_percent=$(echo $data_percent | cut -c2-)
echo $sub_tasks
echo $data_percent