#!/bin/bash
# Chạy toàn bộ thí nghiệm theo thứ tự (CPU 1 lõi). Log: ../runs/*.log
cd "$(dirname "$0")"; mkdir -p ../runs
python3 train.py --name E1_small_noaug      --arch small      --aug 0 --epochs 12                         > ../runs/E1.log 2>&1
python3 train.py --name E2_small_aug        --arch small      --aug 1 --epochs 12                         > ../runs/E2.log 2>&1
python3 train.py --name E3_small_aug_onecycle --arch small    --aug 1 --epochs 12 --sched cosine --lr 3e-3 > ../runs/E3.log 2>&1
python3 train.py --name E4_small_nobn_aug   --arch small_nobn --aug 1 --epochs 12                         > ../runs/E4.log 2>&1
python3 train.py --name E5_small_sgd_onecycle --arch small    --aug 1 --epochs 12 --opt sgd --sched cosine --lr 0.05 --wd 5e-4 > ../runs/E5.log 2>&1
python3 train.py --name E6_medium_aug_onecycle --arch medium  --aug 1 --epochs 10 --sched cosine --lr 3e-3 --ls 0.1 > ../runs/E6.log 2>&1
echo ALLDONE > ../runs/ALLDONE
