## I. Cross-lingual language model pretraining ([XLM](https://github.com/facebookresearch/XLM)) 

XLM supports multi-GPU and multi-node training, and contains code for:
- **Language model pretraining**:
    - **Causal Language Model** (CLM)
    - **Masked Language Model** (MLM)
    - **Translation Language Model** (TLM)
- **GLUE** fine-tuning
- **XNLI** fine-tuning
- **Supervised / Unsupervised MT** training:
    - Denoising auto-encoder
    - Parallel data training
    - Online back-translation

#### Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) (currently tested on version 0.4 and 1.0)
- [fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) (generate and apply BPE codes)
- [Moses](https://github.com/facebookresearch/XLM/tree/master/tools#tokenizers) (scripts to clean and tokenize text only - no installation required)
- [Apex](https://github.com/nvidia/apex#quick-start) (for fp16 training)

Look [facebookresearch/XLM](https://github.com/facebookresearch/XLM)

## II. Model-Agnostic Meta-Learning ([MAML](https://arxiv.org/abs/1911.02116))  

Look [maml](https://github.com/cbfinn/maml), [learn2learn](https://github.com/learnables/learn2learn)...  

Look [HowToTrainYourMAMLPytorch](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch) for a replication of the paper ["How to train your MAML"](https://arxiv.org/abs/1810.09502), along with a replication of the original ["Model Agnostic Meta Learning"](https://arxiv.org/abs/1703.03400) (MAML) paper.

## III. XLM + MAML  

### Pretrained models

##### todo

### Train your own meta_model

#### 1. Preparing the data

At this level, if you have pre-processed binary data in pth format (for example from XLM experimentation or improvised by yourself), please group them in a specific folder that you will mention as a parameter by calling the script train.py.  
If this is not the case, we assume that you have txt files available for preprocessing. Look at the following example for which we have three translation tasks: English-French, English-German and French-German.  

We have the following files available for preprocessing: 
- en-fr.en.txt and en-fr.fr.txt 
- en-de.txt and en-de.txt in the 
- fr-de.fr.txt and fr-de.de.txt 

All these files must be in the same folder.  
You can also (and optionally) have monolingual data available (en.txt, de.txt and fr.txt in the same folder).

```
PARA=True          # If parallel data is available and you need to preprocess it
MONO=True          # if you want to process monolingual data (if the monolingual data is unavailable and you 
                   # leave this parameter set to True, the parallel data will be used to build the monolingual data)
PARA_PATH=...      # folder containing the parallel data
MONO_PATH=...      # folder containing the monolingual data
SAME_VOCAB=True    # whether all languages should share the same vocabulary (leave to True)
nCodes=10000
shuf_n_samples=1000000
threads_for_tokenizer=16
test_size=10       # Percentage of test data (%)
val_size=10        # Percentage of valid data (%)

# tools paths
TOKENIZE=tools/tokenize_our.sh
LOWER_REMOVE_ACCENT=tools/lowercase_and_remove_accent.py
FASTBPE=tools/fastBPE/fast


OUTPATH=... # path where processed files will be stored
# create output path
mkdir -p $OUTPATH

chmod +x $FASTBPE
chmod +x build_meta_data.sh
chmod +x tools/mosesdecoder/scripts/tokenizer/*.perl

# The n_sample parameter is optional, and when it is not passed or when it exceeds the dataset size, the whole dataset is considered
n_samples=-1

# If you don't have any other data to fine-tune your model on a specific sub-task, specify the percentage of the sub-task metadata to consider or -1 to ignore it.

sub_task_data=10,10,-1

# transform (tokenize, lower and remove accent, loard code and vocab, apply BPE tokenization, binarize...) our data contained 
# in the text files into a pth file understandable by the framework : takes a lot of time with dataset size, nCodes and shuf_n_samples

sub_task=en-fr:10,en-de:-1,de-fr:1

./build_meta_data.sh $sub_task $n_samples 
```

After this you will have the following files in `$OUTPATH` :  

```
TODO
```

For fine-tune, in `$OUTPATH/fine_tune` :

```
TODO
```

#### 2. Pretrain a language model

```
python train.py

## main parameters
--exp_name mlm_enfrde                  # experiment name
--exp_id maml                          # Experiment ID
--dump_path ./dumped                   # where to store the experiment (the model will be stored in dump_path/exp_id/exp_name)

## data location / training objective
--data_path  $OUTPATH                   # data location 
--lgs 'en-fr|en-de|de-fr'               # considered languages
--clm_steps ''                          # CLM objective
--mlm_steps 'en,fr|en,de|de,fr'         # MLM objective

## transformer parameters
--emb_dim 1024                          # embeddings / model dimension
--n_layers 6                            # number of layers
--n_heads 8                             # number of heads
--dropout 0.1                           # dropout
--attention_dropout 0.1                 # attention dropout
--gelu_activation true                  # GELU instead of ReLU

## optimization
--batch_size 32                         # sequences per batch
--bptt 256                              # sequences length
--optimizer adam,lr=0.0001              # optimizer
--epoch_size 200000                     # number of sentences per epoch
--max_epoch 100                         # Maximum epoch size
--validation_metrics _valid_mlm_ppl     # validation metric (when to save the best model)
--stopping_criterion _valid_mlm_ppl,10  # end experiment if stopping criterion does not improve

## dataset
--train_n_samples -1                    # Just consider train_n_sample train data
--valid_n_samples -1                    # Just consider valid_n_sample validation data 
--test_n_samples -1                     # Just consider test_n_sample test data for
--remove_long_sentences_train True      # remove long sentences in train dataset
--remove_long_sentences_valid False     # remove long sentences in valid dataset
--remove_long_sentences_test False      # remove long sentences in test dataset

## There are other parameters that are not specified here (see train.py).
```

If parallel data is available for each task, the TLM objective can be used with `--mlm_steps 'en-fr|en-de|de-fr'`. To train with both the MLM and TLM objective for each task, you can use `--mlm_steps 'en,fr,en-fr|en,de,en-de|de,fr,de-fr'`. 

To [train with multiple GPUs](https://github.com/facebookresearch/XLM#how-can-i-run-experiments-on-multiple-gpus) use:
```
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
```

**Tips**: Even when the validation perplexity plateaus, keep training your model. The larger the batch size the better (so using multiple GPUs will improve performance). Tuning the learning rate (e.g. [0.0001, 0.0002]) should help.

#### 3. Train a (unsupervised/supervised) MT from a pretrained model

```
python train.py

## main parameters
--exp_name meta_MT_enfrde                                     # experiment name
--exp_id maml
--dump_path ./dumped/                                         # where to store the experiment
--reload_model '/dumped/mlm_enfrde/maml/best-valid_mlm_ppl.pth,/dumped/mlm_enfrde/maml/best-valid_mlm_ppl.pth'          
                                                              # model to reload for encoder,decoder

## data location / training objective
--data_path $OUTPATH                                          # data location
--lgs 'en-fr|en-de|de-fr'                                     # considered languages
--ae_steps 'en,fr|en,de|de-fr'                                # denoising auto-encoder training steps
--bt_steps 'en-fr-en,fr-en-fr|en-de-en,de-en-de|de-fr-de,fr-de-fr'    # back-translation steps
--word_shuffle 3                                              # noise for auto-encoding loss
--word_dropout 0.1                                            # noise for auto-encoding loss
--word_blank 0.1                                              # noise for auto-encoding loss
--lambda_ae '0:1,100000:0.1,300000:0'                         # scheduling on the auto-encoding coefficient

## transformer parameters
--encoder_only false                                          # use a decoder for MT
--emb_dim 1024                                                # embeddings / model dimension
--n_layers 6                                                  # number of layers
--n_heads 8                                                   # number of heads
--dropout 0.1                                                 # dropout
--attention_dropout 0.1                                       # attention dropout
--gelu_activation true                                        # GELU instead of ReLU

## optimization
--tokens_per_batch 2000                                       # use batches with a fixed number of words
--batch_size 32                                               # batch size (for back-translation)
--bptt 256                                                    # sequence length
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001  # optimizer
--epoch_size 200000                                           # number of sentences per epoch
--eval_bleu true                                              # also evaluate the BLEU score
--stopping_criterion 'valid todo todo todo,10'                 # validation metric (when to save the best model)
--validation_metrics 'valid todo todo todo'                    # end experiment if stopping criterion does not improve

## dataset
--train_n_samples -1                    # Just consider train_n_sample train data
--valid_n_samples -1                    # Just consider valid_n_sample validation data 
--test_n_samples -1                     # Just consider test_n_sample test data for
--remove_long_sentences_train True      # remove long sentences in train dataset
--remove_long_sentences_valid False     # remove long sentences in valid dataset
--remove_long_sentences_test False      # remove long sentences in test dataset

## There are other parameters that are not specified here (see train.py).
```
    
Above training is unsupervised. For a supervised nmt, add `--mt_steps en-fr,fr-en|en-de,de-en|de-fr,fr-de'` if parallel data is available.  

Here we have mentioned the objectives for each task. If you want to exclude a task in an objective, put a blank in its place. Suppose we want to exclude from `ae_steps 'en,fr|en,de|de-fr'` the task:
- en-de : `ae_steps 'en,fr||de-fr'` 
- de-fr : `ae_steps 'en,fr|en,de|'`

### Fine-tune the meta_model on a specific (sub) nmt task

At this point, if your fine-tuning data did not come from the previous pre-processing, you can just prepare your txt data and call the script build_meta_data.sh with the (sub) task in question. Since the codes and vocabulary must be preserved, we have prepared another script (build_fine_tune_data.sh) in which we directly apply BPE tokenization on dataset and binarize everything using preprocess.py based on the codes and vocabulary of the meta-model. So we have to call this script for each subtask like this :

```
PARA=True          # If parallel data is available and you need to preprocess it
MONO=True          # if you want to process monolingual data (if the monolingual data is unavailable and you 
                   # leave this parameter set to True, the parallel data will be used to build the monolingual data)
PARA_PATH=...      # folder containing the parallel data
MONO_PATH=...      # folder containing the monolingual data
CODE_VOCAB_MONO_PATH=... # File containing the codes and vocabularies from the previous meta-processing. 

test_size=10       # Percentage of test data (%)
val_size=10        # Percentage of valid data (%)

# tools paths
TOKENIZE=tools/tokenize.sh
LOWER_REMOVE_ACCENT=tools/lowercase_and_remove_accent.py
FASTBPE=tools/fastBPE/fast


OUTPATH=... # path where processed files will be stored
# create output path
mkdir -p $OUTPATH

chmod +x $FASTBPE
chmod +x build_meta_data.sh
chmod +x tools/mosesdecoder/scripts/tokenizer/*.perl

# The n_sample parameter is optional, and when it is not passed or when it exceeds the dataset size, the whole dataset is considered
n_samples=-1

# transform (tokenize, lower and remove accent, loard code and vocab, apply BPE tokenization, binarize...) our data contained 
# in the text files into a pth file understandable by the framework.

# Let's consider the sub-task en-fr.

./build_fine_tune_data.sh en-fr --n_samples $n_samples
```

Let's consider the sub-task en-fr.

```
python train.py

## main parameters
--exp_name meta_MT_enfr                                       # experiment name
--exp_id maml
--dump_path ./dumped/                                         # where to store the experiment
--reload_model '/dumped/meta_MT_enfrde/maml/todo.pth,/dumped/meta_MT_enfrde/maml/todo.pth'          
                                                             # model to reload for encoder,decoder

## data location / training objective
--data_path $OUTPATH                                          # data location
--lgs 'en-fr'                                                 # considered languages
--ae_steps 'en,fr'                                            # denoising auto-encoder training steps
--bt_steps 'en-fr-en,fr-en-fr'                                # back-translation steps
--word_shuffle 3                                              # noise for auto-encoding loss
--word_dropout 0.1                                            # noise for auto-encoding loss
--word_blank 0.1                                              # noise for auto-encoding loss
--lambda_ae '0:1,100000:0.1,300000:0'                         # scheduling on the auto-encoding coefficient

## transformer parameters
--encoder_only false                                          # use a decoder for MT
--emb_dim 1024                                                # embeddings / model dimension
--n_layers 6                                                  # number of layers
--n_heads 8                                                   # number of heads
--dropout 0.1                                                 # dropout
--attention_dropout 0.1                                       # attention dropout
--gelu_activation true                                        # GELU instead of ReLU

## optimization
--tokens_per_batch 2000                                       # use batches with a fixed number of words
--batch_size 32                                               # batch size (for back-translation)
--bptt 256                                                    # sequence length
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001  # optimizer
--epoch_size 200000                                           # number of sentences per epoch
--eval_bleu true                                              # also evaluate the BLEU score
--stopping_criterion 'valid_en-fr_mt_bleu,10'                 # validation metric (when to save the best model)
--validation_metrics 'valid_en-fr_mt_bleu'                    # end experiment if stopping criterion does not improve

## dataset
--train_n_samples -1                    # Just consider train_n_sample train data
--valid_n_samples -1                    # Just consider valid_n_sample validation data 
--test_n_samples -1                     # Just consider test_n_sample test data for
--remove_long_sentences_train True      # remove long sentences in train dataset
--remove_long_sentences_valid False     # remove long sentences in valid dataset
--remove_long_sentences_test False      # remove long sentences in test dataset
```

Above training is unsupervised. For a supervised nmt, add `--mt_steps en-fr,fr-en'` if parallel data is available.

## References

Please cite [[1]](https://arxiv.org/abs/1901.07291) and [[2]](https://arxiv.org/abs/1911.02116) if you found the resources in this repository useful.

### Cross-lingual Language Model Pretraining

[1] G. Lample *, A. Conneau * [*Cross-lingual Language Model Pretraining*](https://arxiv.org/abs/1901.07291) and [facebookresearch/XLM](https://github.com/facebookresearch/XLM)

\* Equal contribution. Order has been determined with a coin flip.

```
@article{lample2019cross,
  title={Cross-lingual Language Model Pretraining},
  author={Lample, Guillaume and Conneau, Alexis},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

### Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

[2] Chelsea Finn, Pieter Abbeel, Sergey Levine [*Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*](https://arxiv.org/abs/1911.02116) and [cbfinn/maml](https://github.com/cbfinn/maml)

```
@article{Chelsea et al.,
  title={Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks},
  author={Chelsea Finn, Pieter Abbeel, Sergey Levine},
  journal={Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, PMLR 70, 2017},
  year={2017}
}
```

## License

See the [LICENSE](LICENSE) file for more details.