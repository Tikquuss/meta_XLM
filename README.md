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

**Open the illustrative notebook in colab**[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tikquuss/meta_XLM/blob/master/notebooks/demo/tuto.ipynb)

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
Changing parameters ($PARA_PATH and $SRC) in [opus.sh](scripts/opus.sh).
```
chmod +x ./scripts/opus.sh
./scripts/opus.sh de-fr
```

Another source for `other_languages-english` data is [anki Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/). Simply download the .zip file, unzip to extract the `other_language.txt` file. This file usually contains data in the form of `sentence_en sentence_other_language other_information` on each line. See [anki.py](scripts/anki.py) and [anky.sh](scripts/anki.sh) in relation to a how to extract data from [anki](http://www.manythings.org/anki/). Example of how to download and extract `de-en` and `en-fr` pair data.
```
cd meta_XLM
output_path=/content/data/para
mkdir $output_path
chmod +x ./scripts/anki.sh
./scripts/anki.sh de,en deu-eng $output_path scripts/anki.py
./scripts/anki.sh en,fr fra-eng $output_path scripts/anki.py
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
  
Changing parameters in [data.sh](data.sh). Between lines 94 and 100 of [data.sh](data.sh), you have two options corresponding to two scripts to execute according to the distribution of the folders containing your data. Option 2 is chosen by default, kindly uncomment the lines corresponding to your option.  
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
If you pass a parameter by calling the script [train.py](XLM/train.py) (example: `python train.py --config_file $config_file --data_path my/data_path`), it will overwrite the one passed in `$config_file`.  
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
                Evaluated on (cols)
                ---------
                <br/>
                Trained on (rows)
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
            <td>17.369127/47.927461</td>
            <td>2210.859690/12.953368</td>
            <td>3536.959955/8.031088</td>
            <td><b>1250.237309/16.580311</b></td>
            <td>4708.072058/3.626943</td>
            <td>4083.258088/6.735751</td>
            <td>3210.763536/8.549223</td>
            <td>7450.920524/4.922280</td>
            <td>1542.992465/14.766839</td>
            <td>5732.719660/2.590674</td>
            <td>2286.665024/8.290155</td>
            <td>3379.404790/6.994819</td>
            <td>4936.428845/6.217617</td>
            <td>3153.084288/6.217617</td>
            <td>3791.381472/6.735751</td>
            <td>2993.925217/10.880829</td>
            <td>3662.573061/4.145078</td>
            <td>4992.132817/8.031088</td>
            <td>4790.053610/3.108808</td>
            <td>6147.900766/3.108808</td>
            <td>7872.794142/3.108808</td>
            <td>1768.544471/6.735751</td>
        </tr>
        <tr>
            <th scope="row">Bulu</th>
            <td>102.758157/23.575130</td>
            <td>18.977420/44.300518</td>
            <td>472.351963/20.984456</td>
            <td><b>42.786021/34.196891</b></td>
            <td>671.838047/19.430052</td>
            <td>194.339561/24.611399</td>
            <td>330.117482/24.093264</td>
            <td>103.418233/23.575130</td>
            <td>195.683811/25.388601</td>
            <td>116.266003/29.533679</td>
            <td>143.291890/22.538860</td>
            <td>308.746145/20.207254</td>
            <td>200.022902/22.797927</td>
            <td>243.380009/23.575130</td>
            <td>148.623956/25.647668</td>
            <td>201.760287/22.279793</td>
            <td>636.741453/12.176166</td>
            <td>529.934518/18.393782</td>
            <td>95.348245/32.124352</td>
            <td>513.211281/18.134715</td>
            <td>959.126270/15.544041</td>
            <td>219.722160/22.538860</td>
        </tr>
        <tr>
            <th scope="row">Ewon</th>
            <td>3597.458512/8.031088</td>
            <td><b>1083.923393/6.217617</b></td>
            <td>461.772550/11.398964</td>
            <td>6524.902804/3.108808</td>
            <td>2283.199946/8.031088</td>
            <td>4229.855086/7.253886</td>
            <td>2829.208898/9.326425</td>
            <td>4511.892131/6.217617</td>
            <td>2955.101998/10.880829</td>
            <td>3466.950061/10.362694</td>
            <td>2212.141539/11.938964</td>
            <td>2557.530000/9.067358</td>
            <td>2606.205549/12.176166</td>
            <td>1658.814733/8.290155</td>
            <td>1637.320893/10.880829</td>
            <td>1225.905459/14.507772</td>
            <td>1683.423820/11.917098</td>
            <td>2100.231348/12.435233</td>
            <td>4429.376168/5.181347</td>
            <td>6356.282741/3.367876</td>
            <td>3262.310291/10.103627</td>
            <td>5574.968837/2.072539</td>
        </tr>
        <tr>
            <th scope="row">Ghom</th>
            <td>10686.887171/10.103227</td>
            <td>7067.973327/10.362694</td>
            <td>10073.487104/7.512953</td>
            <td>60.445317/32.901554</td>
            <td>7838.029978/7.512953</td>
            <td>15789.399063/8.549223</td>
            <td><b>6966.481671/11.917098</b></td>
            <td>25182.897702/1.813472</td>
            <td>7754.804794/11.139896</td>
            <td>12269.033868/9.585492</td>
            <td>10930.420680/7.772021</td>
            <td>7020.830111/10.362694</td>
            <td>15490.173797/6.476684</td>
            <td>16005.435045/4.404145</td>
            <td>12029.289124/9.067358</td>
            <td>9606.282846/8.549223</td>
            <td>10292.913516/8.031088</td>
            <td>11086.757218/6.735751</td>
            <td>10572.099988/8.031088</td>
            <td>8849.561609/11.139896</td>
            <td>14309.365054/6.994819</td>
            <td>8502.434294/9.844560</td>
        </tr>
        <tr>
            <th scope="row">Limb</th>
            <td><b>194.450107/24.611399</b></td>
            <td>2279.104771/9.585492</td>
            <td>1834.474197/9.067358</td>
            <td>281.362618/22.538860</td>
            <td>16.954229/46.113990</td>
            <td>1096.512231/12.176166</td>
            <td>625.397045/16.321244</td>
            <td>279.383163/23.056995</td>
            <td>1133.796346/10.621762</td>
            <td>544.347122/15.544041</td>
            <td>1082.620378/10.103627</td>
            <td>350.860675/21.502591</td>
            <td>564.260425/13.989637</td>
            <td>884.156365/9.585492</td>
            <td>1034.851368/12.435233</td>
            <td>1143.924310/9.067358</td>
            <td>962.178892/18.652850</td>
            <td>1581.237710/6.994819</td>
            <td>382.452382/21.243523</td>
            <td>920.475416/12.435233</td>
            <td>2766.767861/11.658031</td>
            <td>454.221878/18.134715</td>
        </tr>
        <tr>
            <th scope="row">Ngie</th>
            <td>10216.071251/8.549223</td>
            <td>4059.395624/3.626943</td>
            <td><b>1599.223443/9.585492</b></td>
            <td>7074.498552/11.398964</td>
            <td>2728.321024/7.253886</td>
            <td>61.458772/28.497409</td>
            <td>3629.072977/15.025907</td>
            <td>15972.068840/3.367876</td>
            <td>5796.550442/5.440415</td>
            <td>4799.574152/12.435233</td>
            <td>3842.488266/12.176166</td>
            <td>4016.083956/4.404145</td>
            <td>8423.947264/10.103627</td>
            <td>3969.558394/5.440415</td>
            <td>5513.116134/13.212435</td>
            <td>3191.789160/9.326425</td>
            <td>2393.688703/11.658031</td>
            <td>2970.426306/7.512953</td>
            <td>6538.051891/12.435233</td>
            <td>6638.772351/7.512953</td>
            <td>6186.882656/3.626943</td>
            <td>5152.140872/11.139896</td>
        </tr>
        <tr>
            <th scope="row">Dii</th>
            <td>5182.794126/6.994819</td>
            <td>6733.458157/5.958549</td>
            <td>7798.881900/3.108808</td>
            <td>3478.568514/10.880829</td>
            <td>4046.004430/5.958549</td>
            <td><b>1846.013123/8.290155</b></td>
            <td>15.957556/47.409326</td>
            <td>4980.895859/8.808290</td>
            <td>3784.377099/9.067358</td>
            <td>2364.959386/11.917098</td>
            <td>2287.333437/10.362694</td>
            <td>3140.037268/9.067358</td>
            <td>2666.577183/10.621762</td>
            <td>4685.302321/4.404145</td>
            <td>2660.157570/9.585492</td>
            <td>3960.542742/6.994819</td>
            <td>3553.690731/5.440415</td>
            <td>4418.766412/8.808290</td>
            <td>2996.791963/12.435233</td>
            <td>2436.922431/11.917098</td>
            <td>5675.173968/3.626943</td>
            <td>5517.714729/4.404145</td>
        </tr>
        <tr>
            <th scope="row">Doya</th>
            <td>14774.250501/5.440415</td>
            <td>17497.512372/1.554404</td>
            <td>11197.204123/9.067358</td>
            <td>12160.212931/9.067358</td>
            <td>26979.755370/1.813472</td>
            <td>20725.215139/9.067358</td>
            <td><b>6340.275183/11.139896</b></td>
            <td>328.248913/17.616580</td>
            <td>11899.119529/9.326425</td>
            <td>11539.967728/6.735751</td>
            <td>32460.880304/3.886010</td>
            <td>8016.997905/15.284974</td>
            <td>23168.576004/8.549223</td>
            <td>7688.757049/11.139896</td>
            <td>21213.103917/10.362694</td>
            <td>34759.537685/1.554404</td>
            <td>12058.506793/7.772021</td>
            <td>10426.409290/12.953368</td>
            <td>18066.404052/3.367876</td>
            <td>24996.991516/8.290155</td>
            <td>17729.228412/11.917098</td>
            <td>14217.858853/10.103627</td>
        </tr>
        <tr>
            <th scope="row">Peer</th>
            <td>9497.634413/2.331606</td>
            <td>15078.061313/9.326425</td>
            <td>19724.476977/4.922280</td>
            <td>20257.047189/3.367876</td>
            <td>17278.707482/7.512953</td>
            <td>5710.808981/11.398964</td>
            <td>4599.615920/11.658031</td>
            <td>9570.910263/5.958549</td>
            <td>30.876676/45.077720</td>
            <td><b>4546.793096/12.176166</b></td>
            <td>23677.969381/3.367876</td>
            <td>10569.077651/3.626943</td>
            <td>15358.281918/2.849741</td>
            <td>9264.237259/4.922280</td>
            <td>6552.630357/9.844560</td>
            <td>11622.611420/5.958549</td>
            <td>7000.134267/9.326425</td>
            <td>5316.036949/10.621762</td>
            <td>10720.210603/4.145078</td>
            <td>9817.160188/11.917098</td>
            <td>16632.507263/2.072539</td>
            <td>6993.485527/11.917098</td>
        </tr>
        <tr>
            <th scope="row">Samb</th>
            <td>1928.707452/12.694301</td>
            <td>5918.285097/4.404145</td>
            <td>1973.781453/14.507772</td>
            <td>1053.132371/13.471503</td>
            <td>3171.203629/6.217617</td>
            <td><b>438.716025/22.020725</b></td>
            <td>2105.209423/9.844560</td>
            <td>2071.014475/12.953368</td>
            <td>3137.900012/10.362694</td>
            <td>16.459522/41.968912</td>
            <td>1727.379965/6.476684</td>
            <td>2044.243269/12.435233</td>
            <td>2364.525612/11.917098</td>
            <td>2162.773385/10.621762</td>
            <td>1917.373106/13.471503</td>
            <td>3474.932775/6.217617</td>
            <td>1783.717861/11.398964</td>
            <td>1688.410654/10.880829</td>
            <td>2253.322154/9.067358</td>
            <td>831.881901/15.284974</td>
            <td>2073.485379/8.031088</td>
            <td>788.133885/19.170984</td>
        </tr>
        <tr>
            <th scope="row">Guid</th>
            <td>36487.700988/1.295337</td>
            <td>11555.892244/7.512953</td>
            <td>20357.065390/1.813472</td>
            <td>7567.432971/12.435233</td>
            <td>6883.647150/9.067358</td>
            <td>17058.407512/1.554404</td>
            <td>9220.634635/11.917098</td>
            <td>12278.787596/11.139896</td>
            <td>8307.992774/13.730570</td>
            <td>12745.810095/11.139896</td>
            <td>525.233820/16.839378</td>
            <td><b>5429.635909/15.025907</b></td>
            <td>8492.285884/11.139896</td>
            <td>7138.581141/11.398964</td>
            <td>5942.657418/10.880829</td>
            <td>6343.277426/8.549223</td>
            <td>11630.904860/4.404145</td>
            <td>5977.433756/10.621762</td>
            <td>13099.041058/10.362694</td>
            <td>15062.670815/4.922280</td>
            <td>13468.424584/8.031088</td>
            <td>15256.479588/10.880829</td>
        </tr>
        <tr>
            <th scope="row">Guiz</th>
            <td>2144.511669/4.145078</td>
            <td>1522.129181/14.507772</td>
            <td>2965.967042/9.067358</td>
            <td><b>878.959253/14.507772</b></td>
            <td>5453.121159/7.772021</td>
            <td>1699.383243/13.212435</td>
            <td>2584.627046/9.326425</td>
            <td>1089.634733/12.953368</td>
            <td>1400.671748/12.176166</td>
            <td>1332.501257/13.730570</td>
            <td>1119.038280/15.025907</td>
            <td>9.027401/54.663212</td>
            <td>1883.347095/11.658031</td>
            <td>1644.761033/12.953368</td>
            <td>2651.363750/8.808290</td>
            <td>5004.245257/4.922280</td>
            <td>4734.173697/4.404145</td>
            <td>1997.489112/12.176166</td>
            <td>1148.487625/14.507772</td>
            <td>3902.374365/9.585492</td>
            <td>4908.730155/9.326425</td>
            <td>2361.128891/10.621762</td>
        </tr>
        <tr>
            <th scope="row">Kaps</th>
            <td>517.501704/25.129534</td>
            <td>335.001473/27.461140</td>
            <td>250.526636/30.310881</td>
            <td>85.488990/46.891192</td>
            <td>151.832484/33.419689</td>
            <td><b>34.036209/56.217617</b></td>
            <td>128.084275/40.155440</td>
            <td>59.883492/48.963731</td>
            <td>244.946246/34.974093</td>
            <td>186.697528/37.823834</td>
            <td>379.975968/28.756477</td>
            <td>344.038795/29.015544</td>
            <td>6.113031/63.212435</td>
            <td>290.767898/31.865285</td>
            <td>546.293058/16.321244</td>
            <td>233.746178/34.196891</td>
            <td>158.861471/38.860104</td>
            <td>99.329122/43.264249</td>
            <td>106.633701/38.082902</td>
            <td>97.828750/38.860104</td>
            <td>510.376473/23.575130</td>
            <td>96.740413/44.559585</td>
        </tr>
        <tr>
            <th scope="row">Mofa</th>
            <td>4927.084526/6.217617</td>
            <td>9139.163305/10.362694</td>
            <td>13235.337975/11.139896</td>
            <td>6278.059822/5.440415</td>
            <td>5670.239306/14.248705</td>
            <td>8565.298037/8.808290</td>
            <td>4646.559981/11.917098</td>
            <td>10017.129471/3.108808</td>
            <td>7035.663564/4.922280</td>
            <td>8854.486802/5.958549</td>
            <td>6218.773010/8.549223</td>
            <td>4786.792439/12.694301</td>
            <td>5610.114164/10.103627</td>
            <td>26.472098/40.414508</td>
            <td><b>3773.227930/14.248705</b></td>
            <td>6521.803186/8.031088</td>
            <td>6158.261749/11.139896</td>
            <td>14071.136103/4.663212</td>
            <td>5633.511624/5.699482</td>
            <td>17922.133788/4.663212</td>
            <td>18092.964913/9.326425</td>
            <td>5832.488229/8.031088</td>
        </tr>
        <tr>
            <th scope="row">Mofu</th>
            <td>6896.714450/2.331606</td>
            <td>3892.241450/3.886010</td>
            <td>8212.620520/4.145078</td>
            <td>7451.799342/1.554404</td>
            <td>5268.845970/2.072539</td>
            <td>6790.180462/2.849741</td>
            <td>7620.890096/3.108808</td>
            <td>5851.181318/2.590674</td>
            <td>7709.080601/3.108808</td>
            <td>5767.103133/2.849741</td>
            <td>7373.302125/4.145078</td>
            <td>3972.897440/6.735751</td>
            <td><b>3728.126003/3.367876</b></td>
            <td>6904.793047/3.626943</td>
            <td>29.978047/34.974093</td>
            <td>9979.261595/2.590674</td>
            <td>7432.213883/2.590674</td>
            <td>6282.462790/2.590674</td>
            <td>6292.372728/4.145078</td>
            <td>4848.987687/3.367876</td>
            <td>7994.401641/3.367876</td>
            <td>6979.542438/3.626943</td>
        </tr>
        <tr>
            <th scope="row">Du_n</th>
            <td>4858.473959/4.404145</td>
            <td>4188.857551/11.917098</td>
            <td>5441.583169/3.886010</td>
            <td>1921.185891/7.772021</td>
            <td>4245.741232/10.880829</td>
            <td>2110.274887/10.103627</td>
            <td>2291.977058/14.248705</td>
            <td>2538.044688/12.176166</td>
            <td>3494.286295/6.476684</td>
            <td>3991.204329/8.549223</td>
            <td><b>1371.068714/13.989637</b></td>
            <td>2492.821996/11.658031</td>
            <td>5500.742787/3.367876</td>
            <td>2085.347279/13.471503</td>
            <td>7551.767397/6.217617</td>
            <td>34.765539/39.378238</td>
            <td>4866.140413/4.922280</td>
            <td>3131.030431/12.176166</td>
            <td>4427.543974/3.626943</td>
            <td>12975.641264/7.253886</td>
            <td>6943.197466/10.103627</td>
            <td>2016.530646/13.730570</td>
        </tr>
        <tr>
            <th scope="row">Ejag</th>
            <td>2005.824978/3.108808</td>
            <td>3677.650854/12.176166</td>
            <td>3044.803279/8.290155</td>
            <td>8173.251592/2.072539</td>
            <td>3056.825824/11.658031</td>
            <td>2760.647365/5.181347</td>
            <td>3253.561319/7.253886</td>
            <td>2333.755169/10.621762</td>
            <td>2792.395764/10.880829</td>
            <td>1967.624702/12.953368</td>
            <td>6105.374003/8.549223</td>
            <td>2642.805862/5.440415</td>
            <td>2687.567523/4.145078</td>
            <td>3781.152158/2.072539</td>
            <td><b>1777.501383/13.471503</b></td>
            <td>5094.487098/3.886010</td>
            <td>22.832398/41.191710</td>
            <td>4295.533173/8.031088</td>
            <td>3452.388286/2.849741</td>
            <td>1972.054748/8.290155</td>
            <td>3034.048330/10.880829</td>
            <td>2051.433709/13.730570</td>
        </tr>
        <tr>
            <th scope="row">Fulf</th>
            <td>2547.059967/11.398964</td>
            <td>4009.713998/5.958549</td>
            <td>2193.495353/15.025907</td>
            <td>1225.203824/13.989637</td>
            <td>2923.054856/10.362694</td>
            <td>1378.334263/12.694301</td>
            <td>2643.777952/10.880829</td>
            <td>2431.245226/6.476684</td>
            <td>2677.153716/10.880829</td>
            <td>2142.497328/10.103627</td>
            <td><b>986.472753/14.766839</b></td>
            <td>1936.762652/14.766839</td>
            <td>2274.429624/8.549223</td>
            <td>2030.526779/14.248705</td>
            <td>1839.655237/6.217617</td>
            <td>3183.578812/10.621762</td>
            <td>3037.311058/12.953368</td>
            <td>11.256555/52.072539</td>
            <td>3393.564913/5.958549</td>
            <td>2355.052122/8.549223</td>
            <td>8150.425610/5.181347</td>
            <td>4552.548927/7.512953</td>
        </tr>
        <tr>
            <th scope="row">Gbay</th>
            <td>165.437943/29.015544</td>
            <td>142.299954/21.243523</td>
            <td>85.125389/29.274611</td>
            <td>32.616687/42.227979</td>
            <td>127.911490/26.424870</td>
            <td><b>25.557712/35.492228</b></td>
            <td>177.245780/24.611399</td>
            <td>266.144113/23.575130</td>
            <td>86.909139/33.419689</td>
            <td>88.042876/37.046632</td>
            <td>50.243443/34.196891</td>
            <td>142.398775/25.129534</td>
            <td>163.831572/30.569948</td>
            <td>122.191062/24.352332</td>
            <td>52.891848/33.160622</td>
            <td>138.606266/31.088083</td>
            <td>61.570856/32.642487</td>
            <td>38.891632/38.082902</td>
            <td>4.974619/64.766839</td>
            <td>112.223215/32.383420</td>
            <td>140.857772/31.088083</td>
            <td>45.327366/30.829016</td>
        </tr>
        <tr>
            <th scope="row">MASS</th>
            <td>1518.225541/9.326425</td>
            <td>2930.245451/9.585492</td>
            <td>1865.143568/15.284974</td>
            <td><b>718.377368/11.917098</b></td>
            <td>4266.411194/6.217617</td>
            <td>2526.400825/2.331606</td>
            <td>1104.573903/16.580311</td>
            <td>1771.220815/8.290155</td>
            <td>1719.550192/8.290155</td>
            <td>1098.494569/9.585492</td>
            <td>751.240769/17.875648</td>
            <td>2630.658431/8.031088</td>
            <td>1463.676914/12.176166</td>
            <td>790.144751/17.357513</td>
            <td>1067.233037/13.989637</td>
            <td>4222.852106/6.994819</td>
            <td>2946.203711/10.362694</td>
            <td>1670.892273/13.471503</td>
            <td>1393.513150/6.476684</td>
            <td>8.857192/56.735751</td>
            <td>3816.853567/4.922280</td>
            <td>884.293658/16.062176</td>
        </tr>
        <tr>
            <th scope="row">Tupu</th>
            <td>31.840418/43.782383</td>
            <td>110.652688/34.455959</td>
            <td>50.026892/43.523316</td>
            <td><b>9.946212/55.958549</b></td>
            <td>133.494420/33.419689</td>
            <td>11.815010/55.958549</td>
            <td>72.489724/39.378238</td>
            <td>16.914681/53.108808</td>
            <td>15.070417/60.362694</td>
            <td>13.179060/57.512953</td>
            <td>13.298034/51.554404</td>
            <td>18.848861/45.336788</td>
            <td>19.857303/47.668394</td>
            <td>47.110979/38.601036</td>
            <td>32.124252/44.300518</td>
            <td>53.378128/36.010363</td>
            <td>39.356560/39.378238</td>
            <td>45.747680/42.487047</td>
            <td>19.102613/52.590674</td>
            <td>46.843117/41.968912</td>
            <td>6.029574/56.994819</td>
            <td>27.510317/45.595855</td>
        </tr>
        <tr>
            <th scope="row">Vute</th>
            <td>8378.766511/6.735751</td>
            <td>12807.137451/3.367876</td>
            <td><b>5852.358145/9.067358</b></td>
            <td>29145.878656/2.072539</td>
            <td>10928.124624/2.072539</td>
            <td>11985.263481/8.808290</td>
            <td>10664.710576/6.476684</td>
            <td>20069.820087/2.072539</td>
            <td>8336.208578/8.290155</td>
            <td>11323.061966/8.031088</td>
            <td>12847.339796/10.362694</td>
            <td>8390.185701/4.663212</td>
            <td>10348.166574/9.844560</td>
            <td>7177.280932/7.253886</td>
            <td>6055.499936/11.917098</td>
            <td>8605.633787/10.103627</td>
            <td>8916.377856/6.994819</td>
            <td>18065.408520/3.886010</td>
            <td>7554.167755/7.772021</td>
            <td>11598.525536/6.476684</td>
            <td>7206.137402/12.953368</td>
            <td>79.077804/26.683938</td>
        </tr>
        <!--
        <tr>
            <th scope="row">foo</th>
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
        -->
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





