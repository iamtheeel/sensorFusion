#!/bin/bash

labels=("apple" "ball" "bottle" "clip" "glove" "lid" "plate" "spoon" "tape spool")
for i in $(seq 0 8)
do
    echo "Label ${i}: ${labels[$i]}"
    grep "^${i}" *.txt | wc -l
done
