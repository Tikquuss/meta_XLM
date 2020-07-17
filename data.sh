#!/bin/bash

# usage : data.sh $languages

# languages 
lgs=$1
       
# path containing the csv file
csv_path=/home/jupyter
# where to store the txt files
output_dir=/home/jupyter/data/xlm_cluster4

# If parallel data is available and you need to preprocess it
PARA=True
# If you want to process monolingual data (if the monolingual data is unavailable and you 
# leave this parameter set to True, the parallel data will be used to build the monolingual data)
MONO=True    
# folder containing the parallel data
PARA_PATH=$output_dir
# folder containing the monolingual data
MONO_PATH=$output_dir
# whether all languages should share the same vocabulary (leave to True)
SAME_VOCAB=True    
# Learn nCodes BPE code on the training data
nCodes=200
# Generating shuf_n_samples random permutations of training data to learn bpe
shuf_n_samples=1000 
# It is preferable and advisable that it be the powers of two...
threads_for_tokenizer=16 
# Percentage of data to use as test data (%)
test_size=10 
# Percentage of data to use as validation data (%)
val_size=10              

# tools paths
TOOLS_PATH=tools
TOKENIZE=$TOOLS_PATH/tokenizer_our.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast
PROCESSED_FILE=../scripts/build_meta_data_multixlm.sh

# path where processed files will be stored
OUTPATH=/home/jupyter/models/africa/cluster4/data/XLM_all/processed

# The n_sample parameter is optional, and when it is not passed or when it exceeds the dataset size, the whole dataset is considered
n_samples=-1
# If you don't have any other data to fine-tune your model on a specific sub-task, specify the percentage of the sub-task metadata to consider or -1 to ignore it.
#sub_tasks=en-fr:10,de-en:-1,de-fr:-1
#If you want the subtasks to be constructed from the pair combinations of your languages, put the three dots
sub_tasks=...


##############################################

function abrev() {
    if [[ $1 = "Francais" ]]; then
        result="fr"
    elif [[ $1 = "Anglais" ]]; then
        result="en"
    elif [[ $1 = "MKPAMAN_AMVOE_Ewondo" ]]; then
        result="Ewon"
    else
        length=${#1}
        if [[ $length -le 4 ]]; then
            result=$1
        else
            result=$(echo $1 | cut -c1-4)
        fi
    fi
}

if [ $sub_tasks="..." ]; then
    sub_tasks=""
	IFS=', ' read -r -a langs_array <<< "$languages"
	# todo : sort the array in alphebical oder
	array_length=${#langs_array[*]}
	for (( i=0; i<$array_length; ++i)); do 
		for (( j=$(($i+1)); j<$array_length; ++j)); do
            abrev ${langs_array[$i]} 
            a=$result
            abrev ${langs_array[$j]} 
            b=$result
        	sub_tasks=$sub_tasks,$a-$b:-1
		done
	done
	# Remove the comma in front
	sub_tasks=$(echo $sub_tasks | cut -c2-)
fi

echo $sub_tasks

# create output path
mkdir -p $OUTPATH
# avoid permission error
chmod +x $FASTBPE
chmod +x $TOOLS_PATH/mosesdecoder/scripts/tokenizer/*.perl

echo "======================="
echo "Extract texts files"
echo "======================="

data_type=para
python ../scripts/bible.py --csv_path $csv_path --output_dir $output_dir --data_type $data_type --languages $lgs

data_type=mono
python ../scripts/bible.py --csv_path $csv_path --output_dir $output_dir --data_type $data_type --languages $lgs

echo "======================="
echo "Processed"
echo "======================="

add_para_data_to_mono_data=False 
chmod +x ../scripts/build_meta_data_multixlm.sh
../scripts/build_meta_data_multixlm.sh $sub_tasks $n_samples $add_para_data_to_mono_data
# todo : rendre les choses dynamiques comme ceci
#chmod +x $PROCESSED_FILE
#$PROCESSED_FILE

echo "======================="
echo "End"
echo "======================="
