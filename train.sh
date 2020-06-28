#!/bin/bash

OUTPATH=/home/jupyter/models/africa/cluster4/data/XLM_all/processed
exp_id=maml
# If you don't have enough RAM or swap memory, leave these three parameters to True, otherwise you may get an error like this when evaluating 
# RuntimeError: copy_if failed to synchronize: cudaErrorAssert: device-side assert triggered
remove_long_sentences_train=True
remove_long_sentences_valid=True
remove_long_sentences_test=True
#--remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test

#################

epoch_size=6335

lgs=Guidar-Guiziga-Kapsiki_DC-Mofa-Mofu_Gudur



##############


mlm_steps=Bafia,Bulu,MKPAMAN_AMVOE_Ewondo,Bafia-Bulu,Bafia-MKPAMAN_AMVOE_Ewondo,Bulu-MKPAMAN_AMVOE_Ewondo

batch_size=4
max_epoch=100
dump_path=/home/jupyter/models/africa/cluster1
exp_name=mlm_tlm_BafiaBuluEwondo

python XLM/train.py --config_file $config_file