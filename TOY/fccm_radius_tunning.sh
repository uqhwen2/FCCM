#!/bin/bash

for i in 0
do
	for j in $(seq 0.01 0.01 0.12)
	do
    	CUDA_VISIBLE_DEVICES=0 \
        python fccm_radius_tunning.py \
        --seed $i \
        --alpha 2.5 \
        --radius $j
	done
done
