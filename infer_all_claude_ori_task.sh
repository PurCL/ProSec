#!/bin/bash

# OUT_DIR=infer-ret-claude-haiku-inst-phi3m-ori-task
OUT_DIR=infer-ret-claude-haiku-inst-clm7b-ori-task
USER_NAME=${HF_USER:?Please set HF_USER environment variable (e.g., export HF_USER=your-hf-username)}

mkdir -p $OUT_DIR

function infer_one() {
    local_ds="--dataset_name $1"
    local_fout="--fout $OUT_DIR/infer-ret-$2-clm7b.jsonl"
    local_model='--model_name_or_path meta-llama/CodeLlama-7b-Instruct-hf'
    # python3 src/gen_inferences.py $local_ds --n 10 $local_fout $local_model --col_name original_prompt
    python3 src/gen_inferences.py $local_ds --n 20 $local_fout $local_model --col_name original_prompt
}


current_task=0
TOTAL_TASKS=25
for task in cwe22-javascript cwe78-c cwe78-cpp cwe78-java cwe79-java cwe89-python cwe119-javascript cwe119-cpp cwe119-c cwe200-java cwe295-java cwe338-cpp cwe338-python cwe338-c cwe338-javascript cwe352-java cwe377-c cwe377-cpp cwe502-java cwe502-python cwe611-java cwe676-cpp cwe676-c cwe78-python cwe79-javascript 
do
    current_task=$(($current_task + 1))
    echo "inferencing $task ($current_task/$TOTAL_TASKS)..."
    infer_one "$USER_NAME/secalign-$task-claude-haiku-inst-clustered" $task
done

