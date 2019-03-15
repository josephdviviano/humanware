#!/bin/bash

DATADIR="/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN"

python eval.py \
    --metadata_filename="${DATADIR}/test_sample_metadata_split.pkl" \
    --dataset_dir=${DATADIR} \
    --results_dir=/home/user25/

