#!/bin/bash

python train.py \
  --exp_name xlm_en_fr \
  --dump_path ./dumped \
  --data_path $OUTPATH    \
  --lgs 'en-fr'           \
  --clm_steps ''           \
  --mlm_steps 'en,fr'       \
  \
  --emb_dim 512      \
  --n_layers 12        \
  --n_heads 16          \
  --dropout 0.1          \
  --attention_dropout 0.1  \
  --gelu_activation true    \
  \
  --batch_size 32   \
  --bptt 256         \
  --optimizer adam,lr=0.0001  \
  --epoch_size 300000     \
  --max_epoch 100000      \
  --validation_metrics _valid_mlm_ppl \
  --stopping_criterion _valid_mlm_ppl,25 \
  --fp16 false

  ## There are other parameters that are not specified here (see [here](https://github.com/facebookresearch/XLM/blob/master/train.py#L24-L198)).
