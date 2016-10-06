
#input from shell
usage() {
    echo "Usage:$0 model_dir mode [config]"
    exit 1
}

if [ $# -lt 1 ];then
    usage;
fi

model_dir=$1
mode=$2
config_file=$3
config_dir=configs

base_config_file=${config_dir}/base_config
if [ "${config_file}" = "" ]; then
    if [ "${mode}" = "--train_random" ]; then
	config_file=${config_dir}/config
    else 
	config_file=${model_dir}/config
    fi
fi

#. ./${base_config_file}
. ./${config_file} #./configs/config.txt

if [ "${mode}" = "--train_random" ]; then
    log_file=train.log
elif [ "${mode}" = "--test" ]; then
    log_file=test.log
else
    echo "Unknown mode"
    exit 1
fi




run(){
    params="
--mode=$mode
--train_dir=$train_dir
--train_file=$train_file
--valid_file=$valid_file
--test_file=$test_file
--hidden_size=$hidden_size
--batch_size=$batch_size
--log_file=$log_file
--num_layers=$num_layers
--keep_prob=$keep_prob
--max_step=$max_step
--max_to_keep=$max_to_keep
--source_data_dir=$source_data_dir
--steps_per_checkpoint=$steps_per_checkpoint
--steps_per_interruption=$steps_per_interruption
"
    echo $params
    python -B main.py $params
    wait
}

train_dir=$model_dir
mkdir -p $train_dir
run
wait
