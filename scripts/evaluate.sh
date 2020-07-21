#!/bin/bash

lgs=$1
# mettre le bon chemin
config_file=/content/config.json

../scripts/duplicate.sh $src_path $tgt_path $lgs $tgt_pair

# run eval
python train.py --config_file $config_file

#because the same dir : delete the rename file
../scripts/delete.sh $tgt_path $tgt_pair
