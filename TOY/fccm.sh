#!/bin/bash

for i in 0
do
	for j in $(seq 0 9)
	do
    	CUDA_VISIBLE_DEVICES=0 \
        python fccm.py \
        --seed $j \
        --alpha 2.5
	done
done
