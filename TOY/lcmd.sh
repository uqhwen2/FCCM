#!/bin/bash

for i in 0
do
	for j in $(seq 0 9)
	do
    	CUDA_VISIBLE_DEVICES=0 \
        python bmdal_reg.py \
        --seed $j \
        --bmdal lcmd \
        --selwithtrain True
	done
done
