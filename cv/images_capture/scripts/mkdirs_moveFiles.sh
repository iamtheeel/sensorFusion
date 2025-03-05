#!/bin/sh

for i in $(seq 1 4);
do
    mkdir "sub${i}"
    mv *_${i}_*.jpg "sub${i}/"
done
