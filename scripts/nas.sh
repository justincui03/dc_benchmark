#!/bin/bash
script_name=`basename "$0"`
id=${script_name%.*}
dataset=${dataset:-cifar10}
seed=${seed:-0}
gpu=${gpu:-"auto"}
dc_method=${dc_method:-"dc"}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

echo 'id:' $id 'seed:' $seed 'dataset:' $dataset
echo 'gpu:' $gpu
echo 'DC method: ' $dc_method

cd ../nasbench201/
python train_search.py \
    --dc_method $dc_method \
    --method darts \
    --dataset $dataset \
    --save $id --gpu $gpu --seed $seed \
    # --fast --expid_tag debug \

## bash darts-201.sh