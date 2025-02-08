#!/bin/bash

for i in 0
do
	for j in $(seq 0.05 0.05 0.5)
	do
    	CUDA_VISIBLE_DEVICES=0 \
        python fccm_radius_tuning.py \
        --seed $i \
        --alpha 2.5 \
        --radius $j
	done
done
