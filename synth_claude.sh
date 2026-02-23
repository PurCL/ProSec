#!/bin/bash

if [[ $2 = "java" ]]; then
    lang="java "
else
    lang=$2
fi

echo Start to process CWE-$1, $lang, $2
USER_NAME=${HF_USER:?Please set HF_USER environment variable (e.g., export HF_USER=your-hf-username)}
OUT_DIR=claude-syn-insts
INST_FILE_BEFORE_PROC=$OUT_DIR/vul-insts-cwe$1-$2-v2.jsonl
INST_DS_BEFORE_PROC=$USER_NAME/secalign-cwe$1-$2-claude-haiku-inst
CLUSTERED_DS_NAME=$USER_NAME/secalign-cwe$1-$2-claude-haiku-inst-clustered

# put your openAI api in 'my-api-key.txt'. It should only contain the API key string.
python3 src/gen_vul_insts_from_infinity_insts.py \
    --sys-prompt-in cwe-elicitors/CWE-$1-$2.txt \
    --fout  $INST_FILE_BEFORE_PROC \
    --from-idx 0 \
    --to-idx 10000 \
    --nproc 64 \
    --model_name haiku \
    --lang "$lang"

python3 src/upload_vul_insts_dataset.py \
    --ds_in $INST_FILE_BEFORE_PROC \
    --ds_out $INST_DS_BEFORE_PROC

export CUDA_VISIBLE_DEVICES=6,7
python3 src/cluster_dataset.py \
    --original_ds_name $INST_DS_BEFORE_PROC \
    --out_ds_name $CLUSTERED_DS_NAME &