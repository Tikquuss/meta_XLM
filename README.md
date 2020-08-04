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

**Note** : Most of the bash scripts used in this repository were written on the windows operating system, and can generate this [error](https://prograide.com/pregunta/5588/configure--bin--sh--m-mauvais-interpreteur) on linux platforms.  
This problem can be corrected with the following command: 
```
filename=my_file.sh 
cat $filename | tr -d '\r' > $filename.new && rm $filename && mv $filename.new $filename 
```
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
    <caption><b>?</b></caption>
    <thead>
        <tr>
            <th scope="col">
                Evaluated on (rows)
                ---------
                <br/>
                Trained on (cols)
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
            <td>9641.862413/4.145078</td>
            <td>7023.006102/11.139896</td>
            <td>8488.757076/5.699482</td>
            <td>4836.354041/7.772021</td>
            <td>5154.110724/8.031088</td>
            <td>5205.630687/10.103627</td>
            <td>7735.052416/10.362694</td>
            <td>4938.250528/11.917098</td>
            <td>5952.549715/12.176166</td>
            <td>8069.986058/7.253886</td>
            <td>9331.144292/4.663212</td>
            <td>20763.651520/3.108808</td>
            <td><b><a>3968.884445/14.766839</a></b></td>
            <td>10056.528945/7.772021</td>
            <td>5691.886751/11.658031</td>
            <td>6354.060010/4.663212</td>
            <td>9525.391103/7.253886</td>
            <td>6127.726849/9.844560</td>
            <td>9046.982496/9.844560</td>
            <td>11365.703558/2.590674</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Bulu</th>
            <td>84.652506/41.191710</td>
            <td>Bulu</td>
            <td>740.771977/20.466321</td>
            <td><b><a>31.783430/59.067358</a></b></td>
            <td>1293.129827/13.471503</td>
            <td>115.662209/44.559585</td>
            <td>438.829552/22.020725</td>
            <td>59.983813/50.000000</td>
            <td>197.510486/31.606218</td>
            <td>139.914763/40.414508</td>
            <td>226.804356/32.642487</td>
            <td>334.615706/20.984456</td>
            <td>145.375169/36.787565</td>
            <td>528.222062/15.803109</td>
            <td>210.279107/28.756477</td>
            <td>247.222512/23.316062</td>
            <td>357.467149/26.424870</td>
            <td>290.799215/28.497409</td>
            <td>43.813012/54.145078</td>
            <td>267.563669/26.683938</td>
            <td>1362.146143/14.248705</td>
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
            <th scope="row">Ewon</th>
            <td>2151.431353/10.880829</td>
            <td><b><a>815.198358/12.435233</a></b></td>
            <td>Ewon</td>
            <td>6019.234940/3.886010</td>
            <td>1880.201237/2.331606</td>
            <td>3538.267341/7.512953</td>
            <td>2470.050441/6.735751</td>
            <td>1818.302513/11.917098</td>
            <td>2884.453057/11.139896</td>
            <td>4142.411503/3.626943</td>
            <td>2339.331947/13.212435</td>
            <td><b>1412.523008/13.212435</b></td>
            <td>2012.112662/11.917098</td>
            <td>1352.342604/6.994819</td>
            <td>2355.256775/3.626943</td>
            <td>1881.124257/4.404145</td>
            <td>1752.942742/8.031088</td>
            <td>3659.392301/2.849741</td>
            <td>3171.832886/8.808290</td>
            <td>3892.788686/6.994819</td>
            <td>2902.837358/6.735751</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Ghom</th>
            <td>12534.580387/4.663212</td>
            <td>6429.403387/9.585492</td>
            <td>12487.609719/3.886010</td>
            <td>Ghom</td>
            <td>12082.108668/2.590674</td>
            <td>10541.723001/9.067358</td>
            <td>10419.898378/10.362694</td>
            <td>8534.138193/10.880829</td>
            <td><b><a>4799.715045/13.471503</a></b></td>
            <td>7910.274852/8.808290</td>
            <td>14444.680356/3.886010</td>
            <td>6077.696992/11.398964</td>
            <td>9839.076908/8.031088</td>
            <td>17795.257890/1.813472</td>
            <td>5283.339702/11.398964</td>
            <td>12322.690724/2.849741</td>
            <td>9074.204567/8.290155</td>
            <td>11795.495703/6.476684</td>
            <td>11404.250560/3.626943</td>
            <td>6052.526703/9.844560</td>
            <td>6091.630087/11.398964</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Limb</th>
            <td>2548.509833/9.326425</td>
            <td>5867.537444/11.139896</td>
            <td>7638.680131/2.590674</td>
            <td><b><a>989.009397/10.880829</a></b></td>
            <td>Limb</td>
            <td>3907.341382/7.253886</td>
            <td>5057.720608/7.253886</td>
            <td>1860.932751/11.658031</td>
            <td>1860.932751/11.658031</td>
            <td>1629.312303/14.507772</td>
            <td>2068.833365/11.398964</td>
            <td>2472.515809/6.217617</td>
            <td>6228.619987/3.886010</td>
            <td>2898.169247/12.435233</td>
            <td><b>2507.417041/14.766839</b></td>
            <td>9058.923217/3.626943</td>
            <td>17742.041318/1.813472</td>
            <td>7047.464678/8.290155</td>
            <td>2401.919112/9.067358</td>
            <td>5616.579787/9.326425</td>
            <td>8866.336289/10.880829</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Ngie</th>
            <td>3839.537600/8.808290</td>
            <td>2258.625684/3.367876</td>
            <td>859.593223/7.772021</td>
            <td>5236.167128/6.476684</td>
            <td><b><a>642.637135/14.248705</a></b></td>
            <td>Ngie</td>
            <td>2179.338089/11.398964</td>
            <td>5322.743968/2.072539</td>
            <td>1306.552793/12.694301</td>
            <td>1923.131712/8.549223</td>
            <td>1033.530817/12.435233</td>
            <td>1309.807990/5.440415</td>
            <td>3468.478875/8.290155</td>
            <td>1099.209170/8.549223</td>
            <td>1308.837793/12.694301</td>
            <td>1619.173043/4.404145</td>
            <td>1536.168234/6.735751</td>
            <td>1243.568903/11.398964</td>
            <td>2721.658908/13.471503</td>
            <td>2657.650201/4.404145</td>
            <td>1108.130685/11.139896</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Dii</th>
            <td>1798.463676/12.176166</td>
            <td>3304.220186/12.694301</td>
            <td>3728.976216/9.067358</td>
            <td>1934.611307/5.699482</td>
            <td>2684.056032/12.953368</td>
            <td><b><a>966.387152/9.326425</a></b></td>
            <td>Dii</td>
            <td>2032.415258/12.435233</td>
            <td>1702.768217/8.808290</td>
            <td>4667.123531/4.663212</td>
            <td><b>1301.429756/15.803109</b></td>
            <td>1817.271287/10.621762</td>
            <td>1413.182329/9.067358</td>
            <td>2200.981269/9.067358</td>
            <td>2269.613026/6.994819</td>
            <td>2239.031973/13.212435</td>
            <td>3984.938295/6.735751</td>
            <td>3728.976216/9.067358</td>
            <td>2492.211612/9.585492</td>
            <td>1832.114609/9.067358</td>
            <td>2665.039369/10.103627</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Doya</th>
            <td>4893.198105/9.326425</td>
            <td>6287.481749/2.331606</td>
            <td>5808.410418/3.626943</td>
            <td>14406.605889/4.663212</td>
            <td>8473.045394/7.253886</td>
            <td>8920.099595/7.772021</td>
            <td>8191.981461/3.367876</td>
            <td>Doya</td>
            <td>4287.027967/13.471503</td>
            <td>5953.688449/11.658031</td>
            <td>14767.571446/3.108808</td>
            <td><b><a>3530.551890/13.730570</a></b></td>
            <td>10256.501949/7.512953</td>
            <td>3768.545361/10.880829</td>
            <td>11737.798567/4.922280</td>
            <td>14521.561276/1.813472</td>
            <td>6578.692985/7.772021</td>
            <td>9329.793306/7.772021</td>
            <td>3598.611434/12.694301</td>
            <td>6902.982200/12.176166</td>
            <td>10987.224415/5.958549</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Peer</th>
            <td>8908.379667/3.886010</td>
            <td>8652.128379/13.471503</td>
            <td>18627.804053/3.886010</td>
            <td>14583.766532/10.621762</td>
            <td>15608.195381/4.145078</td>
            <td>19464.813588/10.880829</td>
            <td><b><a>5231.032656/13.989637</a></b></td>
            <td>26042.041333/3.367876</td>
            <td>Peer</td>
            <td>11118.108068/4.922280</td>
            <td>18749.089653/4.922280</td>
            <td>6573.450465/11.917098</td>
            <td>14867.235979/9.585492</td>
            <td>10041.800720/6.476684</td>
            <td>14068.778453/4.663212</td>
            <td>17058.696000/1.813472</td>
            <td>8266.294649/10.362694</td>
            <td>7088.080623/13.730570</td>
            <td>14291.709022/3.886010</td>
            <td>10949.034999/12.176166</td>
            <td>14013.710629/7.253886</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Samb</th>
            <td>3288.687068/8.031088</td>
            <td>3404.860159/5.181347</td>
            <td><b>2231.246405/12.694301</b></td>
            <td>3077.913123/3.108808</td>
            <td>1694.117930/8.290155</td>
            <td><b><a>1449.211630/2.590674</a></b></td>
            <td>2040.977014/12.176166</td>
            <td>5026.008327/7.253886</td>
            <td>5620.717959/4.922280</td>
            <td>Samb</td>
            <td>2153.477184/12.176166</td>
            <td>2789.618463/11.658031</td>
            <td>2549.402110/10.621762</td>
            <td>7872.150457/2.590674</td>
            <td>3013.175369/8.808290</td>
            <td>5150.489427/1.813472</td>
            <td>2939.279236/7.772021</td>
            <td>3544.917114/4.663212</td>
            <td>2673.759202/12.694301</td>
            <td>1761.157273/6.217617</td>
            <td>2584.268630/3.626943</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Guid</th>
            <td>2059.881574/9.585492</td>
            <td>1904.170288/5.699482</td>
            <td>1233.402875/10.103627</td>
            <td><b>1428.437730/12.953368</b></td>
            <td>1322.252839/7.512953</td>
            <td>4067.960150/3.367876</td>
            <td>1285.030370/10.880829</td>
            <td>2256.136999/9.844560</td>
            <td>2378.502999/5.181347</td>
            <td>2839.932332/3.886010</td>
            <td>Guid</td>
            <td>2133.120878/5.699482</td>
            <td>2545.912923/2.590674</td>
            <td>1946.243250/3.108808</td>
            <td>1141.774751/8.549223</td>
            <td>1716.771722/3.367876</td>
            <td><b><a>1239.418475/10.103627</a></b></td>
            <td>1371.487257/8.290155</td>
            <td>2892.676739/7.512953</td>
            <td>2319.529896/1.813472</td>
            <td>1851.926954/7.772021</td>
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
            <td>975.002457/17.616580</td>
            <td>663.386340/20.725389</td>
            <td>245.019906/26.683938</td>
            <td>178.001163/25.647668</td>
            <td>304.888440/22.020725</td>
            <td><b><a>66.103359/27.720207</a></b></td>
            <td>267.606732/19.689119</td>
            <td>163.299261/21.243523</td>
            <td>501.440619/18.393782</td>
            <td>433.267198/15.544041</td>
            <td>426.763084/22.279793</td>
            <td>576.537985/14.507772</td>
            <td>Kaps</td>
            <td>585.658114/11.398964</td>
            <td>1104.268613/14.507772</td>
            <td>436.725091/19.430052</td>
            <td>391.324696/19.170984</td>
            <td>271.268658/20.466321</td>
            <td>370.848999/16.062176</td>
            <td>241.029656/23.316062</td>
            <td>522.839959/18.911917</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Mofa</th>
            <td>2226.573039/10.103627</td>
            <td><b>1821.428880/14.507772</b></td>
            <td>3587.808220/9.326425</td>
            <td>6150.006570/2.331606</td>
            <td>3669.169564/10.880829</td>
            <td>3304.108362/10.103627</td>
            <td>2714.833319/9.844560</td>
            <td>3929.381608/7.512953</td>
            <td>3179.650349/10.621762</td>
            <td>2533.267453/14.507772</td>
            <td>3952.469441/9.585492</td>
            <td>3527.106587/8.808290</td>
            <td>3345.078245/13.989637</td>
            <td>Mofa</td>
            <td>1925.508108/13.471503</td>
            <td><b><a>1178.287530/14.248705</a></b></td>
            <td>3936.148142/4.145078</td>
            <td>4849.457280/7.253886</td>
            <td>3292.250224/11.658031</td>
            <td>5178.524435/3.626943</td>
            <td>4471.619338/13.730570</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Mofu</th>
            <td>1221.517231/10.880829</td>
            <td>1509.403349/4.663212</td>
            <td><b><a>965.299050/11.398964</a></b></td>
            <td>2801.230161/3.886010</td>
            <td>1288.682714/4.663212</td>
            <td>1116.432968/11.139896</td>
            <td>1839.516041/6.994819</td>
            <td>1410.705031/10.103627</td>
            <td>1903.188068/10.880829</td>
            <td>2410.419127/3.886010</td>
            <td><b>1072.867240/11.917098</b></td>
            <td>1476.753929/9.067358</td>
            <td>2450.620591/3.626943</td>
            <td>1622.950336/5.181347</td>
            <td>Mofu</td>
            <td>1755.236333/5.440415</td>
            <td>1671.335252/5.181347</td>
            <td>1643.431276/4.404145</td>
            <td>2992.068854/4.145078</td>
            <td>2458.152754/5.440415</td>
            <td>2143.866379/6.476684</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Du_n</th>
            <td>1073.511749/13.212435</td>
            <td><b><a>951.447834/15.025907</a></b></td>
            <td>1726.115685/3.626943</td>
            <td>1535.436216/4.145078</td>
            <td>1119.924980/15.025907</td>
            <td>1370.296138/4.145078</td>
            <td>1138.137047/13.212435</td>
            <td>1824.375396/3.367876</td>
            <td>762.246813/15.544041</td>
            <td>1530.009027/6.476684</td>
            <td>740.315680/11.658031</td>
            <td>1280.492597/8.031088</td>
            <td>1126.285824/10.621762</td>
            <td>1015.113656/9.326425</td>
            <td>1163.049297/9.326425</td>
            <td>Du_n</td>
            <td>1042.802693/10.621762</td>
            <td>1110.908299/10.621762</td>
            <td>2061.871193/5.699482</td>
            <td>1678.421095/12.176166</td>
            <td>1755.093784/11.398964</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Ejag</th>
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
            <td>1184.516199/15.284974</td>
            <td>3135.833243/8.808290</td>
            <td>1748.882729/12.435233</td>
            <td><b><a>621.999487/15.803109</a></b></td>
            <td>2669.555407/10.880829</td>
            <td>1602.806312/15.803109</td>
            <td>1970.925740/9.585492</td>
            <td>1316.602911/8.290155</td>
            <td>1299.410165/13.989637</td>
            <td><b>1100.305846/17.875648</b></td>
            <td>635.394683/16.321244</td>
            <td>1438.762810/16.062176</td>
            <td>2170.088997/8.290155</td>
            <td>2458.666192/6.735751</td>
            <td>1706.137234/11.398964</td>
            <td>2166.289389/10.621762</td>
            <td>1396.348987/11.658031</td>
            <td>Fulf</td>
            <td>1423.229257/10.103627</td>
            <td>1672.188024/14.507772</td>
            <td>3135.833243/8.808290</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">Gbay</th>
            <td>219.140187/24.093264</td>
            <td>76.661343/29.792746</td>
            <td>114.322577/23.834197</td>
            <td>65.259857/24.870466</td>
            <td>160.053651/20.207254</td>
            <td><b><a>55.893237/19.430052</a></b></td>
            <td>206.477036/25.906736</td>
            <td>239.890494/20.466321</td>
            <td>129.491231/25.906736</td>
            <td>123.956888/30.569948</td>
            <td>77.795740/25.647668</td>
            <td>87.055673/30.829016</td>
            <td>123.998762/23.056995</td>
            <td>120.414121/25.647668</td>
            <td>104.907581/22.279793</td>
            <td><b>111.137699/31.347150</b></td>
            <td>88.120721/28.238342</td>
            <td>65.294247/26.943005</td>
            <td>Gbay</td>
            <td>128.657263/26.683938</td>
            <td>128.707253/24.870466</td>
            <td>Vute</td>
        </tr>
        <tr>
            <th scope="row">MASS</th>
            <td>2045.037056/11.398964</td>
            <td>4253.856094/9.326425</td>
            <td>2305.952094/11.398964</td>
            <td>563.963153/12.176166</td>
            <td>4880.308875/7.772021</td>
            <td>1607.562430/10.103627</td>
            <td>1762.240252/14.248705</td>
            <td>1404.948804/10.103627</td>
            <td>1597.505921/12.176166</td>
            <td><b>887.680799/15.803109</b></td>
            <td>1002.664319/12.176166</td>
            <td>1626.456117/14.766839</td>
            <td>1261.074608/13.212435</td>
            <td><b><a>13.212435/14.766839</a></b></td>
            <td>971.691727/13.730570</td>
            <td>2374.276617/13.471503</td>
            <td>3496.535632/6.476684</td>
            <td>2922.379583/5.699482</td>
            <td>1052.875047/11.398964</td>
            <td>MASS</td>
            <td>7345.553956/4.404145</td>
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
            <td>6645.796535/11.139896</a></td>
            <td>4925.596401/8.290155</td>
            <td>3741.521260/6.476684</td>
            <td>5361.760334/11.658031</td>
            <td>2579.121479/12.435233</td>
            <td>7014.681338/3.367876</td>
            <td>5295.572362/6.476684</td>
            <td>7638.941172/4.145078</td>
            <td>6666.381806/6.476684</td>
            <td>7053.783303/9.067358</td>
            <td>4638.151371/7.772021</td>
            <td><b><a>2443.262661/10.362694</a></b></td>
            <td>6183.810066/11.917098</td>
            <td>7720.488779/2.072539</td>
            <td>5085.297020/12.176166</td>
            <td>3228.098115/11.658031</td>
            <td><b>3278.564381/12.694301</b></td>
            <td>6983.135604/4.145078</td>
            <td>5252.519258/10.362694</td>
            <td>4835.933224/10.621762</td>
            <td>5773.691223/9.067358</td>
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





