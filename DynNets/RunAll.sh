#!/bin/bash

for author in MOORE NIGEL ORUNADA VIEGAS
do
    for size in '' 32 32,64:
    do
        for year in 2016 2017 2018 2019
        do
            echo ./TrainNets.py $author $year "'$size'" "''"
            ./TrainNets.py $author $year "'$size'" "''"
        done
    done
done

