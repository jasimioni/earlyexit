#!/bin/bash

for x in *.log
do
    echo -n "$x "
    tail -n3 $x | head -n 1 | cut -f 5 -d ':' | sed -e 's/\./,/' | sed -e 's/ //g'
done
