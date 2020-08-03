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

### 1. Preparing the data 

At this level, if you have pre-processed binary data in pth format (for example from XLM experimentation or improvised by yourself), group them in a specific folder that you will mention as a parameter by calling the script [train.py](XLM/train.py).  
If this is not the case, we assume that you have txt files available for preprocessing. Look at the following example for which we have three translation tasks: `English-French, German-English and German-French`.

We have the following files available for preprocessing: 
```
- en-fr.en.txt and en-fr.fr.txt 
- de-en.de.txt and de-en.en.txt 
- de-fr.de.txt and de-fr.fr.txt 
```
All these files must be in the same folder (`PARA_PATH`).  
You can also (only or optionally) have monolingual data available (`en.txt, de.txt and fr.txt`; in `MONO_PATH` folder).  
Parallel and monolingual data can all be in the same folder.

**Note** : Languages must be submitted in alphabetical order (`de-en and not en-de, fr-ru and not ru-fr...`). If you submit them in any order you will have problems loading data during training, because when you run the [train.py](XLM/train.py) script the parameters like the language pair are put back in alphabetical order before being processed. Don't worry about this alphabetical order restriction, XLM for MT is naturally trained to translate sentences in both directions. See [translate.py](scripts/translate.py).

[OPUS collections](http://opus.nlpl.eu/) is a good source of dataset. We illustrate in the [opus.sh](scripts/opus.sh) script how to download the data from opus and convert it to a text file. 

Another source for `other_languages-english` data is [anki Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/). Simply download the .zip file, unzip to extract the `other_language.txt` file. This file usually contains data in the form of `sentence_en sentence_other_language other_information` on each line. See [anki.py](scripts/anki.py) and [anky.sh](scripts/anki.sh) in relation to a how to extract data from [anki](http://www.manythings.org/anki/). Example of how to download and extract `de-en` pair data.
```
cd meta_XLM
output_path=XLM/data/para
mkdir $output_path
chmod +x ./scripts/anki.sh
./script/anki.sh de,en deu-eng $output_path scripts/anki.py
#./script/anki.sh en,fr fra-eng $output_path scripts/anki.py
```
After that you will have in `data/para` following files : `de-en.de.txt, de-en.en.txt, deu.txt, deu-eng.zip and _about.txt`  

Move to the `XLM` folder in advance.  
```
cd XLM
```
Install the following dependencies ([fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) and [Moses](https://github.com/facebookresearch/XLM/tree/master/tools#tokenizers)) if you have not already done so. 
```
git clone https://github.com/moses-smt/mosesdecoder tools/mosesdecoder
git clone https://github.com/glample/fastBPE tools/fastBPE && cd tools/fastBPE && g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
```
  
Changing parameters in [data.sh](data.sh).  
With too many BPE codes (depending on the size of the dataset) you may get this [error](https://github.com/glample/fastBPE/issues/7). Decrease the number of codes (e.g. you can dichotomously search for the appropriate/maximum number of codes that make the error disappear)

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
To use the biblical corpus, run [bible.sh](bible.sh) instead of [data.sh](data.sh). Here is the list of languages available (and to be specified as `$languages` value) in this case : 
- **Languages with data in the New and Old Testament** : `Francais, Anglais, Fulfulde_Adamaoua or Fulfulde_DC (formal name : Fulfulde), Bulu, KALATA_KO_SC_Gbaya or KALATA_KO_DC_Gbaya (formal name :  Gbaya), BIBALDA_TA_PELDETTA (formal name : MASSANA), Guiziga, Kapsiki_DC (formal name : Kapsiki), Tupurri`.
- **Languages with data in the New Testament only** : `Bafia, Ejagham, Ghomala, MKPAMAN_AMVOE_Ewondo (formal name : Ewondo), Ngiemboon, Dii, Vute, Limbum, Mofa, Mofu_Gudur, Doyayo, Guidar, Peere_Nt&Psalms, Samba_Leko, Du_na_sdik_na_wiini_Alaw`.  
It is specified in [bible.sh](bible.sh) that you must have in `csv_path` a folder named csvs. Here is the [drive link](https://drive.google.com/file/d/1NuSJ-NT_BsU1qopLu6avq6SzUEf6nVkk/view?usp=sharing) of its zipped version.  
Concerning training, specify the first four letters of each language (`Bafi` instead of `Bafia` for example), except `KALATA_KO_SC_Gbaya/KALATA_KO_DC_Gbaya which becomes Gbay (first letters of Gbaya), BIBALDA_TA_PELDETTA which becomes MASS (first letters of MASSANA), MKPAMAN_AMVOE_Ewondo which becomes Ewon (first letters of Ewondo), Francais and Anglais which becomes repectively fr and en`. Indeed, [bible.sh](bible.sh) uses these abbreviations to create the files and not the language names themselves.  
One last thing in the case of the biblical corpus is that when only one language is to be specified, it must be specified twice. For example: `languages=Bafia,Bafia` instead of `languages=Bafia`.

### 2. Pretrain a language (meta-)model 

Install the following dependencie ([Apex](https://github.com/nvidia/apex#quick-start)) if you have not already done so.
```
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

Instead of passing all the parameters of train.py, put them in a json file and specify the path to this file in parameter (See [lm_template.json](configs/lm_template.json) file for more details).
```
config_file=../configs/lm_template.json
python train.py --config_file $config_file
```
Once the training is finished you will see a file named `train.log` in the `$dump_path/$exp_name/$exp_id` folder information about the training. You will find in this same folder your checkpoints and best model.  
When `"mlm_steps":"..."`, train.py automatically uses the languages to have `"mlm_steps":"de,en,fr,de-en,de-fe,en-fr"` (give a precise value to mlm_steps if you don't want to do all MLM and TLM, example : `"mlm_steps":"en,fr,en-fr"`). This also applies to `"clm_steps":"..."` which deviates `"clm_steps":"de,en,fr"` in this case.    

Note :  
-`en` means MLM on `en`, and requires the following three files in `data_path`: `a.en.pth, a ∈ {train, test, valid} (monolingual data)`  
-`en-fr` means TLM on `en and fr`, and requires the following six files in `data_path`: `a.en-fr.b.pth, a ∈ {train, test, valid} and b ∈ {en, fr} (parallel data)`  
-`en,fr,en-fr` means MLM+TLM on `en, fr, en and fr`, and requires the following twelve files in `data_path`: `a.b.pth and a.en-fr.b.pth, a ∈ {train, test, valid} and b ∈ {en, fr}`  

To [train with multiple GPUs](https://github.com/facebookresearch/XLM#how-can-i-run-experiments-on-multiple-gpus) use:
```
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py --config_file $config_file
```

**Tips**: Even when the validation perplexity plateaus, keep training your model. The larger the batch size the better (so using multiple GPUs will improve performance). Tuning the learning rate (e.g. [0.0001, 0.0002]) should help.

In the case of <b>metalearning</b>, you just have to specify your meta-task separated by `|` in `lgs` and `objectives (clm_steps, mlm_steps, ae_steps, mt_steps, bt_steps and pc_steps)`.  
For example, if you only want to do metalearning (without doing XLM) in our case, you have to specify these parameters: `"lgs":"de-en|de-fr|en-fr"`, `"clm_steps":"...|...|..."` and/or `"mlm_steps":"...|...|..."`. These last two parameters, if specified as such, will become respectively `"clm_steps":"de,en|de,fr|en,fr"` and/or `"mlm_steps":"de,en,de-en|de,fr,de-fr|en,fr,en-fr"`.  
The passage of the three points follows the same logic as above. That is to say that if at the level of the meta-task `de-en`:  
	- we only want to do MLM (without TLM): `mlm_steps` becomes `"mlm_steps": "de,en|...|..."`  
	- we don't want to do anything: `mlm_steps` becomes `"mlm_steps":"|...|..."`.

It is also not allowed to specify a meta-task that has no objective. In our case, `"clm_steps":"...||..."` and/or `"mlm_steps":"...||..."` will generate an exception, in which case the meta-task `de-fr` (second task) has no objective.

If you want to do metalearning and XLM simultaneously : 
- `"lgs":"de-en-fr|de-en-fr|de-en-fr"` 
- Follow the same logic as described above for the other parameters.

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
remove_long_sentences_train # remove long sentences in train dataset      
remove_long_sentences_valid # remove long sentences in valid dataset  
remove_long_sentences_test  # remove long sentences in test dataset  
```

###### There are other parameters that are not specified here (see [train.py](XLM/train.py))

### 3. Train a (unsupervised/supervised) MT from a pretrained meta-model 

See [mt_template.json](configs/mt_template.json) file for more details.
```
config_file=../configs/mt_template.json
python train.py --config_file $config_file
```

When the `ae_steps` and `bt_steps` objects alone are specified, this is unsupervised machine translation, and only requires monolingual data. If the parallel data is available, give `mt_step` a value based on the language pairs for which the data is available.  
All comments made above about parameter passing and <b>metalearning</b> remain valid here : if you want to exclude a meta-task in an objective, put a blank in its place. Suppose, in the case of <b>metalearning</b>, we want to exclude from `"ae_steps":"en,fr|en,de|de-fr"` the meta-task:
- `de-en` : `ae_steps`  becomes `"ae_steps":"en,fr||de-fr"` 
- `de-fr` : `ae_steps`  becomes `"ae_steps":"en,fr|de,en|"`  

###### Description of some essential parameters  
The description made above remains valid here
```
## main parameters
reload_model     # model to reload for encoder,decoder
## data location / training objective
ae_steps          # denoising auto-encoder training steps
bt_steps          # back-translation steps
mt_steps          # parallel training steps
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


### 4. case of metalearning : optionally fine-tune the meta-model on a specific (sub) nmt (meta) task 

At this point, if your fine-tuning data did not come from the previous pre-processing, you can just prepare your txt data and call the script build_meta_data.sh with the (sub) task in question. Since the codes and vocabulary must be preserved, we have prepared another script ([build_fine_tune_data.sh](scripts/build_fine_tune_data.sh)) in which we directly apply BPE tokenization on dataset and binarize everything using preprocess.py based on the codes and vocabulary of the meta-model. So we have to call this script for each subtask like this :

```
languages = 
chmod +x ../ft_data.sh
../ft_data.sh $languages
```

At this stage, restart the training as in the previous section with :
- lgs="en-fr"
- reload_model = path to the folder where you stored the meta-model
- `bt_steps'':"..."`, `ae_steps'':"..."` and/or `mt_steps'':"..."` (replace the three bullet points with your specific objectives if any)  
You can use one of the two previously trained meta-models: pre-formed meta-model (MLM, TLM) or meta-MT formed from the pre-formed meta-model. 

### 5. How to evaluate a language model trained on a language L on another language L'.

###### Our

<table class="table table-striped">
    <caption><b></b></caption>
    <thead>
        <tr>
            <th scope="col">
                Trained on
                <br/>
                Evaluated on
            </th>
            <th scope="col">Bafi</th>
            <th scope="col">Bulu</th>
            <th scope="col">Ewon</th>
            <th scope="col">Ghom</th>
            <th scope="col">Limb</th>
            <th scope="col">Ngie</th>
            <th scope="col">Dii</th>
            <th scope="col">Doya</th>
            <th scope="col">Peer</th>
            <th scope="col">Samb</th>
            <th scope="col">Guid</th>
            <th scope="col">Guiz</th>
            <th scope="col">Kaps</th>
            <th scope="col">Mofa</th>
            <th scope="col">Mofu</th>
            <th scope="col">Du_n</th>
            <th scope="col">Ejag</th>
            <th scope="col">Fulf</th>
            <th scope="col">Gbay</th>
            <th scope="col">MASS</th>
            <th scope="col">Tupu</th>
            <th scope="col">Vute</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th scope="row">Bafi</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Bulu</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Ghom</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Limb</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Ngie</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Dii</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Doya</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Peer</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Samb</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Guid</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Guiz</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Kaps</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Mofa</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Mofu</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Du_n</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Ejagam</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Fulf</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Gbay</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">MASS</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Tupu</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Vute</th>
            <td>Bafi</td>
            <td>Bulu</td>
            <td>Ewon</td>
            <td>Ghom</td>
            <td>Limb</td>
            <td>Ngie</td>
            <td>Dii</td>
            <td>Doya</td>
            <td>Peer</td>
            <td>Samb</td>
            <td>Guid</td>
            <td>Guiz</td>
            <td>Kaps</td>
            <td>Mofa</td>
            <td>Mofu</td>
            <td>Du_n</td>
            <td>Ejag</td>
            <td>Fulf</td>
            <td>Gbay</td>
            <td>MASS</td>
            <td>Tupu</td>
            <td>Vute</td>
        </tr>
    </tbody>
</table>

###### Prerequisite
If you want to evaluate the LM on a language `lang`, you must first have a file named `lang.txt` in the `$src_path` directory of [eval_data.sh](eval_data.sh).  
For examplel if you want to use the biblical corpus, you can run [scripts/bible.py](scripts/bible.py) :
```
# folder containing the csvs folder
csv_path=
# folder in which the objective folders will be created (mono or para)
output_dir=
# monolingual one ("mono") or parallel one ("para")
data_type=mono
# list of languages to be considered in alphabetical order and separated by a comma
# case of one language
languages=lang,lang  
# case of many languages
languages=lang1,lang2,...   
old_only : use only old testament
#  use only new testament
new_only=True

python ../scripts/bible.py --csv_path $csv_path --output_dir $output_dir --data_type $data_type --languages $languages --new_only $new_only
```
See other parameters in [scripts/bible.py](scripts/bible.py)

###### Data pre-processing
Modify parameters in [eval_data.sh](eval_data.sh)
```
# languages to be evaluated
languages=lang1,lang2,... 
chmod +x ../eval_data.sh 
../eval_data.sh $languages
```

###### Evaluation 

We take the language to evaluate (say `Bulu`), replace the files `test.Bulu.pth` (which was created with the `VOCAB` and `CODE` of `Bafi`, the evaluating language) with `test.Bafi.pth` (since `Bafi` evaluates, the `train.py` script requires that the dataset has the (part of the) name of the `lgs`). Then we just run the evaluation, the results (acc and ppl) we get is the result of LM Bafia on the Bulu language.

```
# evaluating language
tgt_pair=
# folder containing the data to be evaluated (must match $tgt_path in eval_data.sh)
src_path=
# You have to change two parameters in the configuration file used to train the LM which evaluates ("data_path":"$src_path" and "eval_only": "True")
# You must also specify the "reload_model" parameter, otherwise the last checkpoint found will be loaded for evaluation.
config_file=../configs/lm_template.json 
# languages to be evaluated
eval_lang= 
chmod +x ../scripts/evaluate.sh
../scripts/evaluate.sh $eval_lang
```
When the evaluation is finished you will see a file named `eval.log` in the `$dump_path/$exp_name/$exp_id` folder containing the evaluation results.    
**Note** :The description given below is only valid when the LM evaluator has been trained on only one language (and therefore without TLM). But let's consider the case where the basic LM has been trained on `en-fr` and we want to evaluate it on `de` or `de-ru`. `$tgt_pair` deviates from `en-fr`, but `language` varies depending on whether the evaluation is going to be done on one language or two:  
- In the case of `de` : `lang=de-de`  
- in the case of `de-ru`: `lang=de-ru`.

## IV. References

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





