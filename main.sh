#!/bin/bash

datasets=("ciciot" "credit" "ecg" "ids" "kdd" "kitsune")
models=("ae" "alad" "dagmm" "dsebm" "svdd", "if", "lof", "ocsvm")

python_file1="cleaning.py"
python_file2="detection.py"

# Define a function to process each dataset and model
process_dataset_model() {
    dataset="$1"
    model="$2"
    echo "Processing dataset: $dataset with model: $model"
    python "$python_file1 --name" "$dataset"
    python "$python_file2 --name" "$dataset --model" "$model"
    echo "Finished processing dataset: $dataset with model: $model"
}

# Export the function so GNU Parallel can access it
export -f process_dataset_model

# Run the function in parallel for each dataset and model
parallel -j "$(nproc)" process_dataset_model ::: "${datasets[@]}" ::: "${models[@]}"
echo "All datasets and models processed successfully"
