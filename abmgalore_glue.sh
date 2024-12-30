#!/bin/bash

tasks=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb" "wnli")
# tasks=("mnli" "qqp")

# Submit separate jobs for each task
for task in "${tasks[@]}"; do
    sbatch amgore_glue.sh $task
done

echo "All tasks submitted."
