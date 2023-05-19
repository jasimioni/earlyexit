#!/bin/bash

for author in MOORE NIGEL ORUNADA VIEGAS
do
    for year in 2016 2017 2018 2019
    do
        echo ./TrainNets.py $author $year '32,64,128' '3,2,2'
        ./TrainNets.py $author $year '32,64,128' '3,2,2'
    done
done

