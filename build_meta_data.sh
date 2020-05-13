#!/bin/bash

# Usage: ./build_meta_data.sh $sub_task $n_samples $sub_task_data

set -e

# 1) if no parameters : stop
if [ $# = 0 ];then
  exit
fi

SUB_TASKS_DATA_PERCENT=${1-""}

N_SAMPLES=${2-'False'}
if [ $N_SAMPLES -le 0 ];then
    N_SAMPLES=False
fi

# 2) if no task : stop
if [ $SUB_TASKS_DATA_PERCENT = "" ];then
    exit
fi

sub_tasks=""
fine_tune_data_percent=""

for task_data_percent in $(echo $SUB_TASKS_DATA_PERCENT | sed -e 's/\,/ /g'); do
    IFS=': ' read -r -a array <<< "$task_data_percent"
    sub_tasks=$sub_tasks,${array[0]}
    fine_tune_data_percent=$fine_tune_data_percent,${array[1]}
done

# Remove the comma in front
sub_tasks=$(echo $sub_tasks | cut -c2-)
fine_tune_data_percent=$(echo $fine_tune_data_percent | cut -c2-)

# 4) if no task : stop
if [ $sub_tasks = "" ];then
    exit
fi

if [ $fine_tune_data_percent != "" ];then
    mkdir $OUTPATH/fine_tune
fi

# 5) if PARA = False && MONO = False : stop and report an error
if [ $PARA = "False" ] && [ $MONO = "False" ]; then
    echo "error"
    exit
# 6) if PARA = False && MONO = False : stop and report an error
elif [ $PARA = "True" ] && [ ! -d $PARA_PATH ]; then
    echo "error"
    exit
# 7) if MONO = True && PARA_PATH does not exist && MONO_PATH does not exist : stop and report an error
elif [ $MONO = "True" ] && [ ! -d $PARA_PATH ] && [ ! -d $MONO_PATH ]; then
    echo "error"
    exit
fi

# 7) Otherwise, it's okay, we keep going.
echo "params ok !"

#
# Tokenize and preprocess data
#
chmod +x $TOKENIZE

# usage : get_n_samples input_file n_samples output_file
get_n_samples() {
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NLINES=$(($NLINES+1));
    if [ $NLINES -le $2 ]; then
      cp $1 $3
    else
      NTAIL=$(($2/2));
      NHEAD=$(($2 - NTAIL));
      head -n $NHEAD $1 > $3;
      tail -n $NTAIL $1 >> $3;
    fi
}

#  para data 
# if PARA = True (then PARA_PATH must exist)
if [ $PARA = "True" ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        echo "*** Cleaning and tokenizing $pair data ... ***"
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            if [ ! -f $PARA_PATH/$pair.$lg.all ]; then
                if [ $N_SAMPLES = "False" ];then
                    cat $PARA_PATH/$pair.$lg.txt | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $PARA_PATH/$pair.$lg.all
                else
                    get_n_samples $PARA_PATH/$pair.$lg.txt $N_SAMPLES $PARA_PATH/samples.$pair.$lg
                    cat $PARA_PATH/samples.$pair.$lg | $TOKENIZE $lg $threads_for_tokenizer | python $LOWER_REMOVE_ACCENT > $PARA_PATH/$pair.$lg.all 
                    rm $PARA_PATH/${pair}/samples.$pair.$lg
                fi
                echo "*** Tokenized (+ lowercase + accent-removal) $pair.$lg data to $PARA_PATH/? ***"
            else
                #rm $PARA_PATH/$pair.$lg.all
                echo "file $PARA_PATH/$pair.$lg.all already exists" 
            fi
        done
    done
fi

# todo : 
get_seeded_random() {
    seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
};
forEach pair, data_percent in zip($sub_tasks, $fine_tune_data_percent):
    if good data_percent :
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            NLINES=`wc -l $PARA_PATH/$pair.$lg.all  | awk -F " " '{print $PARA_PATH/$pair.$lg.all}'`;
            NLINES=$(($NLINES+1));
            N_FINE_TUNE=$(((NLINES*$data_percent)/100))
            if [ $NLINES -le $N_FINE_TUNE ]; then
                # todo : exit
                echo "error"
            else
                NTAIL=$(($N_FINE_TUNE/2));
                NHEAD=$(($N_FINE_TUNE - NTAIL)); 
                NREST=$((NLINES - $NTAIL - $NHEAD));
                shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN                           > $2;
                shuf --random-source=<(get_seeded_random 42) $1 | head -$(($NTRAIN+$NVAL)) | tail -$NVAL  > $3;
                shuf --random-source=<(get_seeded_random 42) $1 | tail -$NTEST                            > $4;
                head -n $NHEAD $PARA_PATH/$pair.$lg.all > $OUTPATH/fine_tune/$pair.$lg.all;
                tail -n $NTAIL $PARA_PATH/$pair.$lg.all >> $OUTPATH/fine_tune/$pair.$lg.all;
            supprimer $PARA_PATH/$pair.$lg.all??
        done
    else :
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            rename $PARA_PATH/$pair.$lg.all?? en $PARA_PATH/$pair.$lg.all
        done

# mono data 
# if MONO = True &&  MONO_PATH exist
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            if [ ! -f $MONO_PATH/$lg.all ]; then
                if [ $N_SAMPLES = "False" ];then
                    cat $MONO_PATH/$lg.txt | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $MONO_PATH/$lg.all
                else
                    get_n_samples $MONO_PATH/$lg.txt $N_SAMPLES $MONO_PATH/samples.$lg
                    cat $MONO_PATH/samples.$lg | $TOKENIZE $lg $threads_for_tokenizer | python $LOWER_REMOVE_ACCENT > $MONO_PATH/$lg.all 
                    rm $MONO_PATH/samples.$lg
                fi
                echo "*** Tokenized (+ lowercase + accent-removal) $lg data to $MONO_PATH/? ***"
            else
                #rm $PARA_PATH/$pair.$lg.all
                echo "file $MONO_PATH/$lg.all already exists" 
            fi
        done
    done
fi

# meme todo que plus haut si necessaire

# Let's take the case $pair = "en-fr"
# At this point we have for this pair the following files:
# if PARA = True && PARA_PATH exists, in $PARA_PATH: en-en.en.all and en-en.fr.all
# if MONO = True && MONO_PATH exists, in $MONO_PATH: en.all and fr.all

#
# split into train / valid / test
#
split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NLINES=$(($NLINES+1));
    NTEST=$(((NLINES*$5)/100));
    NVAL=$(((NLINES*$6)/100));
    NTRAIN=$((NLINES - $NVAL - $NTEST));
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN                           > $2;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$(($NTRAIN+$NVAL)) | tail -$NVAL  > $3;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -$NTEST                            > $4;
}

# para 
# if PARA = True (then PARA_PATH must exist)
if [ $PARA = "True" ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            split_data $PARA_PATH/$pair.$lg.all $PARA_PATH/$pair.$lg.train $PARA_PATH/$pair.$lg.valid $PARA_PATH/$pair.$lg.test $test_size $val_size
        done
    done
fi

# mono
# if MONO = True &&  MONO_PATH exist
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            split_data $MONO_PATH/$lg.all $MONO_PATH/$lg.train $MONO_PATH/$lg.valid $MONO_PATH/$lg.test $test_size $val_size
        done
    done
fi

# Let's take the case $pair = "en-fr"
# At this point we have, in addition to the previous files, the following files:
# if PARA = True && PARA_PATH exists, in $PARA_PATH: en-fr.en.train and en-fr.fr.train, en-fr.en.valid and
#                                                    en-fr.fr.valid, en-fr.en.test and en-fr.fr.test
# if MONO = True && MONO_PATH exists, in $MONO_PATH: en.train and fr.train, en.valid and fr.valid, en.test et fr.test

#
# Now we create our training set for the BPE vocabulary, for instance by taking 100M sentences from each 
# monolingua corpora.
# 

echo "***build the training set for BPE tokenization ($nCodes codes)***"

# Je ne gère que le cas SAME_VOCAB = True pour l'instant

# para 
# if PARA = True (then PARA_PATH must exist)
if [ $PARA = "True" ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            shuf -r -n $shuf_n_samples $PARA_PATH/$pair.$lg.train >> $OUTPATH/bpe.train
        done
    done
fi

# mono
# if MONO = True &&  MONO_PATH exist
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            shuf -r -n $shuf_n_samples $MONO_PATH/$lg.train >> $OUTPATH/bpe.train
        done
    done
fi

echo "***Learn the BPE vocabulary on the training set : $OUTPATH/bpe.train***"
$FASTBPE learnbpe $nCodes $OUTPATH/bpe.train > $OUTPATH/codes

echo "***Get the post-BPE vocab***"
$FASTBPE applybpe $OUTPATH/train $OUTPATH/bpe.train $OUTPATH/codes 
cat $OUTPATH/train | $FASTBPE getvocab - > $OUTPATH/vocab 
echo "***Learn the $nCodes BPE code on the bpe.train file***" 

echo "***Apply BPE tokenization on the monolingual and parallel corpora, and binarize everything using preprocess.py.***"
# if PARA = True (then PARA_PATH must exist)
if [ $PARA = "True" ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            for split in train valid test; do
                $FASTBPE applybpe $OUTPATH/$pair.$lg.$split $PARA_PATH/$pair.$lg.$split $OUTPATH/codes
                python preprocess.py $OUTPATH/vocab $OUTPATH/$pair.$lg.$split
            done
        done
    done
fi

# mono
# if MONO = True &&  MONO_PATH exist
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            for split in train valid test; do
                $FASTBPE applybpe $OUTPATH/$split.$lg $MONO_PATH/$lg.$split $OUTPATH/codes
                # Add para data to mono data before preprocessing
                if [ $PARA = "True" ]; then
                    for lg_tmp in $(echo $pair | sed -e 's/\-/ /g'); do
                        for split_tmp in train valid test; do
                            # Add the contents of $OUTPATH/$pair.$lg_tmp.$split_tmp after $OUTPATH/$split.$lg
                            cat $OUTPATH/$pair.$lg_tmp.$split_tmp >> $OUTPATH/$split.$lg
                        done
                    done
                fi
                python preprocess.py $OUTPATH/vocab $OUTPATH/$split.$lg
            done
        done
    done
fi

# if MONO = True && MONO_PATH does not exist && PARA_PATH exists
if [ $MONO = "True" ] && [ ! -d $MONO_PATH ] && [ ! -d $PARA_PATH ]; then
    # We use our parallel data to construct the monolingual data 
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            for split in train valid test; do
                cp $OUTPATH/$pair.$lg.$split.pth $OUTPATH/$split.$lg.pth
            done
        done
    done
fi