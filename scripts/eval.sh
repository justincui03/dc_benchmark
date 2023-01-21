#!/bin/bash
dataset=${dataset:-cifar10}
method=${method:-"DC"}
aug=${aug:-"random_aug"}
gpu=${gpu:-"auto"}
ipc=${ipc:-1}
model=${model:"convnet"}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

echo 'method:' $method 'dataset:' $dataset 'aug:' $aug
echo 'gpu:' $gpu

python evaluator/evaluator.py \
    --dataset $dataset \
    --aug $aug \
    --gpu $gpu \
    --ipc $ipc \
    --model $model
