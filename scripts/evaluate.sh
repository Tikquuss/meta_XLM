#!/bin/bash

lgs=$1

tgt_path=$src_path

# creation of the dummy files (train) so that the experiment does not bug
touch /content/eval_Bafia_tmp/train.Bafi.pth

chmod +x ../scripts/duplicate.sh
chmod +x ../scripts/delete.sh

for lg in $(echo $lgs | sed -e 's/\,/ /g'); do
    ../scripts/duplicate.sh $src_path $tgt_path $lg $tgt_pair
    # run eval
    python train.py --config_file $config_file
    #because the same dir (tgt_path and src_path) : delete the rename file
    ../scripts/delete.sh $tgt_path $tgt_pair
done
