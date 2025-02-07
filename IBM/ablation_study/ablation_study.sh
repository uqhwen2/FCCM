#!/bin/bash

for i in 0
do
	for j in $(seq 0 9)
	do
    	CUDA_VISIBLE_DEVICES=0 \
        python probcover.py \
        --seed $j \
        --alpha 0 \
        --radius 0
	done
done
