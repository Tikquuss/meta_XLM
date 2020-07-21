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

### Pretrained models  

<table class="table table-striped">
    <caption><b>cross-lingual machine translation BLEU score</b></caption>
    <thead>
        <tr>
            <th scope="col">Translation_task</th>
            <th scope="col">Bafi-Bulu</th>
            <th scope="col">Bulu-Bafi</th>
            <th scope="col">Ghom-Limb</th>
            <th scope="col">Limb-Ghom</th>
            <th scope="col">Bafi-Ewon</th>
            <th scope="col">Ewon-Bafi</th>
            <th scope="col">Bulu-Ewon</th>
            <th scope="col">Ewon-Bulu</th>
            <th scope="col">Ghom-Ngie</th>
            <th scope="col">Ngie-Ghom</th>
            <th scope="col">Limb-Ngie</th>
            <th scope="col">Ngie-Limb</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th scope="row">single transformer</th>
            <td>08.64</td>
            <td>11.85</td>
            <td style="color:blue">18.31</td>
            <td style="color:blue">12.90</td> 
            <td>08.38</td>
            <td>13.68</td>
            <td>09.51</td>
            <td>11.24</td>
            <td>06.36</td>
            <td>07.56</td>
            <td>06.76</td>
            <td>11.29</td>   
        </tr>
        <tr>
            <th scope="row">XLMT_1</th>
            <td>14.91</td>
            <td>13.32</td>
            <td>15.40</td>
            <td>11.28 </td>
        </tr>
        <tr>
            <th scope="row">XLMT_23</th>
            <td><b>17.80</b></td>
            <td><b>28.42</b></td>
            <td><b>27.26</b></td>
            <td><b>24.82</b></td>
            <td><b>36.30</b></td>
            <td><b>13.71</b></td>
            <td><b>11.98</b></td>
            <td><b>18.43</b></td>
            <td><b>16.62</b></td>
            <td><b>08.55</b></td>
            <td><b>07.39</b></td>
            <td><b>13.48</b></td>
        </tr>
    </tbody>
  </table> 

  <table class="table table-striped">
    <thead>
        <tr>
            <th scope="col">Translation_task</th>
            <th scope="col">en-Bafi</th>
            <th scope="col">Bafi-en</th>
            <th scope="col">en-Bulu</th>
            <th scope="col">Bulu-en</th>
            <th scope="col">en-Ewon</th>
            <th scope="col">Ewon-en</th>
            <th scope="col">fr-Bafi</th>
            <th scope="col">Bafi-fr</th>
            <th scope="col">fr-Bulu</th>
            <th scope="col">Bulu-fr</th>
            <th scope="col">fr-Ewon</th>
            <th scope="col">Ewon-fr</th>
        </tr>  
    </thead>
    <tbody>
        <tr>
            <th scope="row">XLMT</th>
            <td><b>34.10</b></td>
            <td><b>30.03</b></td>
            <td><b>25.46</b></td>
            <td><b>31.82</b></td>
            <td><b>49.69</b></td>
            <td><b>43.85</b></td>
            <td><b>16.28</b></td>
            <td><b>23.84</b></td>
            <td><b>21.80</b></td>
            <td><b>30.02</b></td>
            <td><b>11.95</b></td>
            <td><b>27.84</b></td>  
        </tr>
    </tbody>
  </table> 
  

cluster1 = (<b>Bafi</b>a, <b>Bulu</b>, <b>Ghom</b>ala, <b>Limb</b>ium)  
cluster2 = (<b>Ghom</b>ala, <b>Limb</b>um, <b>Ngie</b>mboon)  
cluster3 = (<b>Bafi</b>a, <b>Bulu</b>, <b>Ewon</b>do)  
  
XLMT_1 = cross-lingual machine translation on cluster1 for translation tasks Bafi-Bulu, Bulu-Bafi, Ghom-Limb and Limb-Ghom  
XLMT_23 :  
    - cross-lingual machine translation on cluster2 for all translation tasks A-B with A, B ∈ cluster2  
    - cross-lingual machine translation on cluster3 for all translation tasks A-B with A, B ∈ cluster3

## II. Model-Agnostic Meta-Learning ([MAML](https://arxiv.org/abs/1911.02116))  

See [maml](https://github.com/cbfinn/maml), [learn2learn](https://github.com/learnables/learn2learn)...  

See [HowToTrainYourMAMLPytorch](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch) for a replication of the paper ["How to train your MAML"](https://arxiv.org/abs/1810.09502), along with a replication of the original ["Model Agnostic Meta Learning"](https://arxiv.org/abs/1703.03400) (MAML) paper.

## III. Train your own (meta-)model

#### 1. Preparing the data 

At this level, if you have pre-processed binary data in pth format (for example from XLM experimentation or improvised by yourself), group them in a specific folder that you will mention as a parameter by calling the script [train.py](XLM/train.py).  
If this is not the case, we assume that you have txt files available for preprocessing. Look at the following example for which we have three translation tasks: English-French, German-English and German-French (see this [notebooks](notebooks/enfrde.ipynb) for details on the following).

We have the following files available for preprocessing: 
- en-fr.en.txt and en-fr.fr.txt 
- de-en.de.txt and de-en.en.txt 
- de-fr.de.txt and de-fr.fr.txt 

All these files must be in the same folder (`PARA_PATH`).  
You can also (and optionally) have monolingual data available (en.txt, de.txt and fr.txt; in `MONO_PATH` folder).  
Parallel and monolingual data can all be in the same folder.

Note : Languages must be submitted in alphabetical order (de-en and not en-de, fr-ru and not ru-fr ...). If you submit them in any order you will have problems loading data during training, because when you run the [train.py](XLM/train.py) script the parameters like the language pair are put back in alphabetical order before being processed. Don't worry about this alphabetical order restriction, XLM for MT is naturally trained to translate sentences in both directions. See [translate.py](scripts/translate.py).

[OPUS collections](http://opus.nlpl.eu/) is a good source of dataset. We illustrate in the [opus.sh](scripts/opus.sh) script how to download the data from opus and convert it to a text file. 

Another source for other_languages-english data is [anki Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/). Simply download the .zip file, unzip to extract the other_language.txt file. This file usually contains data in the form of `sentence_en sentence_other_language other_information` on each line. See [anki.py](scripts/anki.py) and [anky.sh](scripts/anki.sh) in relation to a how to extract data from [anki](http://www.manythings.org/anki/). Example of how to download and extract de-en pair data.
```
cd meta_XLM
output_path=XLM/data/para
mkdir $output_path
chmod +x scripts/anki.sh
./anki.sh de,en deu-eng $output_path scripts/anki.py
#./anki.sh en,fr fra-eng $output_path scripts/anki.py
```
After that you will have in `data/para` following files : de-en.de.txt, de-en.en.txt, deu.txt, deu-eng.zip and _about.txt  

Move to the `XLM` folder in advance.  

Install the following dependencies ([fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) and [Moses](https://github.com/facebookresearch/XLM/tree/master/tools#tokenizers)) if you have not already done so. 
```
cd tools
git clone https://github.com/moses-smt/mosesdecoder
git clone https://github.com/glample/fastBPE && cd fastBPE && g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
```

Return to the `XLM` folder

```
languages=de,en,fr
chmod +x ../data.sh 
../data.sh $languages
```

If you stop the execution when processing is being done on a file please delete this erroneous file before continuing or restarting the processing, otherwise the processing will continue with this erroneous file and potential errors will certainly occur.  

After this you will have the following (necessary) files in `$OUTPATH` (and `$OUTPATH/fine_tune` depending on the parameter `$sub_task`):  

```
- monolingual data :
    - training data   : train.fr.pth, train.en.pth and train.de.pth
    - test data       : test.fr.pth, test.en.pth and test.de.pth
    - validation data : valid.fr.pth, valid.en.pth and valid.de.pth
- parallel data :
    - training data : 
        - train.en-fr.en.pth and train.en-fr.fr.pth 
        - train.de-en.en.pth and train.de-en.de.pth
        - train.de-fr.de.pth and train.de-fr.fr.pth 
    - test data :
        - test.en-fr.en.pth and test.en-fr.fr.pth 
        - test.de-en.en.pth and test.de-en.de.pth
        - test.de-fr.de.pth and test.de-fr.fr.pth 
    - validation data
        - valid.en-fr.en.pth and valid.en-fr.fr.pth 
        - valid.de-en.en.pth and valid.de-en.de.pth
        - valid.de-fr.de.pth and valid.de-fr.fr.pth 
 - code and vocab
```
To use the biblical corpus, run [bible.sh](bible.sh) instead of [data.sh](data.sh)

#### 2. Pretrain a language meta-model 

Install the following dependencie ([Apex](https://github.com/nvidia/apex#quick-start)) if you have not already done so.
```
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

Instead of passing all the parameters of train.py, put them in a json file and specify the path to this file in parameter (See [lm_template.json](configs/lm_template.json) file for more details).
```
config_file=/configs/lm_template.json
python train.py --config_file $config_file
```
When "mlm_steps":"...", train.py automatically uses the languages to have `"mlm_steps":"de,en,fr,de-en,de-fe,en-fr"` (Give a precise value to x if you don't want to do all MLM and TLM, example : `"mlm_steps":"en,fr,en-fr"`)

###### Description of some essential parameters

```
## main parameters
exp_name                     # experiment name
exp_id                       # Experiment ID
dump_path                    # where to store the experiment (the model will be stored in $dump_path/$exp_name/$exp_id)

## data location / training objective
data_path                    # data location 
lgs                          # considered languages/meta-tasks
clm_steps                    # CLM objective
mlm_steps                    # MLM objective

## transformer parameters
emb_dim                      # embeddings / model dimension
n_layers                     # number of layers
n_heads                      # number of heads
dropout                      # dropout
attention_dropout            # attention dropout
gelu_activation              # GELU instead of ReLU

## optimization
batch_size                   # sequences per batch
bptt                         # sequences length
optimizer                    # optimizer
epoch_size                   # number of sentences per epoch
max_epoch                    # Maximum epoch size
validation_metrics           # validation metric (when to save the best model)
stopping_criterion           # end experiment if stopping criterion does not improve

## dataset
#### These three parameters will always be rounded to an integer number of batches, so don't be surprised if you see different values than the ones provided.
train_n_samples              # Just consider train_n_sample train data
valid_n_samples              # Just consider valid_n_sample validation data 
test_n_samples               # Just consider test_n_sample test data for
#### If you don't have enough RAM/GPU or swap memory, leave these three parameters to True, otherwise you may get an error like this when evaluating :
###### RuntimeError: copy_if failed to synchronize: cudaErrorAssert: device-side assert triggered
remove_long_sentences_train True      
remove_long_sentences_valid False     
remove_long_sentences_test False      
```

###### There are other parameters that are not specified here (see [train.py](XLM/train.py))

If parallel data is available for each meta-task, the TLM objective can be used with `--mlm_steps 'en-fr|en-de|de-fr'`. To train with both the MLM and TLM objective for each meta-task, you can use `--mlm_steps 'en,fr,en-fr|en,de,en-de|de,fr,de-fr'`. 

To [train with multiple GPUs](https://github.com/facebookresearch/XLM#how-can-i-run-experiments-on-multiple-gpus) use:
```
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
```

**Tips**: Even when the validation perplexity plateaus, keep training your model. The larger the batch size the better (so using multiple GPUs will improve performance). Tuning the learning rate (e.g. [0.0001, 0.0002]) should help.

#### 3. Train a (unsupervised/supervised) MT from a pretrained meta-model 

See [mt_template.json](configs/mt_template.json) file for more details).
```
config_file=/configs/mt_template.json
python train.py --config_file $config_file
```
###### Description of some essential parameters  
The description made above remains valid here
```
ae_steps          # denoising auto-encoder training steps
bt_steps          # back-translation steps
word_shuffle      # noise for auto-encoding loss
word_dropout      # noise for auto-encoding loss
word_blank        # noise for auto-encoding loss
lambda_ae         # scheduling on the auto-encoding coefficient

## transformer parameters
encoder_only      # use a decoder for MT

## optimization
tokens_per_batch  # use batches with a fixed number of words
eval_bleu         # also evaluate the BLEU score
```
###### There are other parameters that are not specified here (see [train.py](XLM/train.py))

Above training is unsupervised. For a supervised nmt, add `--mt_steps 'en-fr,fr-en|en-de,de-en|de-fr,fr-de'` if parallel data is available.  

Here we have mentioned the objectives for each meta-task. If you want to exclude a meta-task in an objective, put a blank in its place. Suppose we want to exclude from `ae_steps 'en,fr|en,de|de-fr` the meta-task:
- de-en : `ae_steps 'en,fr||de-fr'` 
- de-fr : `ae_steps 'en,fr|de,en|'`

### 4. Fine-tune the meta-model on a specific (sub) nmt (meta) task (case of metalearning)

At this point, if your fine-tuning data did not come from the previous pre-processing, you can just prepare your txt data and call the script build_meta_data.sh with the (sub) task in question. Since the codes and vocabulary must be preserved, we have prepared another script ([build_fine_tune_data.sh](scripts/build_fine_tune_data.sh)) in which we directly apply BPE tokenization on dataset and binarize everything using preprocess.py based on the codes and vocabulary of the meta-model. So we have to call this script for each subtask like this :

```
goal=fine_tune 
CODE_VOCAB_PATH=/content/Bafi
chmod +x ../preprocess.sh 
../preprocess.sh $languages
```

Let's consider the sub-task en-fr.  
At this stage you can use one of the two previously trained meta-models: pre-formed meta-model or meta-MT formed from the pre-formed meta-model. But here we use the second model, because we intuitively believe that it will be more efficient than the first one.

Move to the `XLM` folder in advance.

See [mt_template.json](configs/mt_template.json) file for more details).
```
config_file=/configs/mt_template.json
python train.py --config_file $config_file
```

Above training is unsupervised. For a supervised nmt, add `--mt_steps 'en-fr,fr-en'` if parallel data is available.

### 5. How to evaluate a language model trained on a language L on another language L'.

```
goal=evaluation
CODE_VOCAB_PATH=/content/Bafi
chmod +x ../preprocess.sh 
../preprocess.sh $languages
```


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





