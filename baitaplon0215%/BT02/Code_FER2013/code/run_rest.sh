#!/bin/bash
cd "$(dirname "$0")"
python3 train.py --name E5_small_sgd_onecycle --arch small    --aug 1 --epochs 12 --opt sgd --sched cosine --lr 0.05 --wd 5e-4 > ../runs/E5.log 2>&1
python3 train.py --name E6_medium_aug_onecycle --arch medium  --aug 1 --epochs 10 --sched cosine --lr 3e-3 --ls 0.1 > ../runs/E6.log 2>&1
echo ALLDONE > ../runs/ALLDONE
