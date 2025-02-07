#!/bin/bash

for i in 0
do
	for j in $(seq 1 9)
	do
    	CUDA_VISIBLE_DEVICES=0 \
        python fccm.py \
        --seed $j \
        --alpha 0 \
        --radius 0
	done
done
