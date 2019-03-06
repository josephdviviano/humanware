#!/bin/bash

export ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export SVHN_DIR='/rap/jvb-000-aa/COURS2019/etudiants/user25/'
export DATA_DIR=$SVHN_DIR/train
export EXTRA_DIR=$SVHN_DIR/extra
export TMP_DATA_DIR=$DATA_DIR
export TMP_RESULTS_DIR=$DATA_DIR
export METADATA_FILENAME="${SVHN_DIR}/train_metadata.pkl"
export EXTRA_META_FILENAME="${SVHN_DIR}/extra_metadata.pkl"

mkdir -p ${TMP_DATA_DIR}
mkdir -p ${TMP_RESULTS_DIR}

if [ ! -f ${SVHN_DIR}'/train.tar.gz' ]; then

    echo "Downloading files for the training set!"
    wget -P ${SVHN_DIR} http://ufldl.stanford.edu/housenumbers/train.tar.gz
fi

if [ ! -d ${TMP_DATA_DIR} ]; then

    echo "Extracting Files to " ${TMP_DATA_DIR}
    cp ${DATA_DIR}'/train.tar.gz' ${TMP_DATA_DIR}
    tar -xzf ${TMP_DATA_DIR}'/train.tar.gz' -C ${TMP_DATA_DIR}
    echo "Extraction finished!"

else
    echo "Train files already present"
fi

s_exec python ${ROOT_DIR}'/train.py' \
    --dataset_dir=${TMP_DATA_DIR} \
    --metadata_filename=${METADATA_FILENAME} \
    --extra_dir=${EXTRA_DIR} \
    --extra_metadata_filename=${EXTRA_META_FILENAME} \
    --results_dir=${ROOT_DIR}/results \
    --cfg ${ROOT_DIR}/config/base_config.yml

# echo "Copying files to local hard drive..."
# cp -r $TMP_RESULTS_DIR $ROOT_DIR

# echo "Cleaning up data and results..."
# rm -r $TMP_DATA_DIR
# rm -r $TMP_RESULTS_DIR
