#!/bin/bash

export ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export SVHN_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN'
export METADATA_FILENAME="${SVHN_DIR}/train_extra_metadata_split.pkl"


s_exec python ${ROOT_DIR}'/train.py' \
    --dataset_dir=${SVHN_DIR} \
    --metadata_filename=${METADATA_FILENAME} \
    --results_dir=${ROOT_DIR}/results \
    --cfg ${ROOT_DIR}/config/base_config.yml

# echo "Copying files to local hard drive..."
# cp -r $TMP_RESULTS_DIR $ROOT_DIR

# echo "Cleaning up data and results..."
# rm -r $TMP_DATA_DIR
# rm -r $TMP_RESULTS_DIR
