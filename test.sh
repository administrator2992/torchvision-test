#!/bin/bash

for i in $(seq 1 10);
do
    echo "CloudLab12#$%" | sudo -S su -c "echo 3 > /proc/sys/vm/drop_caches"
    python3 test_img.py $i $1
done
