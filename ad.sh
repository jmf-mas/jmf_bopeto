#!/bin/bash

datasets=("ciciot" "credit" "ecg" "ids" "kdd" "kitsune")
models=("ae" "alad" "dagmm" "dsebm" "svdd", "if", "lof", "ocsvm")
process_dataset_model() {
    detection="ad.py"
    dataset_name="--dataset"
    model_name="--model"
    dataset="$1"
    model="$2"
    echo "Processing dataset: $dataset with model: $model"
    python "$detection" "$dataset_name" "$dataset"
    python "$detection" "$dataset_name" "$dataset" "$model_name" "$model"
    echo "Finished processing dataset: $dataset with model: $model"
}

export -f process_dataset_model

parallel -j "$(nproc)" process_dataset_model ::: "${datasets[@]}" ::: "${models[@]}"
echo "Cleaning and OoD detection processed successfully"
