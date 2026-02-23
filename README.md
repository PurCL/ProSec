# ProSec

Code repo for the paper [ProSec: Fortifying Code LLMs with Proactive Security Alignment](https://arxiv.org/abs/2411.12882).

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Pipeline](#pipeline)
  - [Step 1: Synthesize CWE-Inducing Instructions](#step-1-synthesize-cwe-inducing-instructions)
  - [Step 2: Generate Potentially Vulnerable Code](#step-2-generate-potentially-vulnerable-code)
  - [Step 3: Generate Benign Code](#step-3-generate-benign-code)
  - [Step 4: Scan Vulnerable Code with Purple Llama](#step-4-scan-vulnerable-code-with-purple-llama)
  - [Step 5: Mix Fixed Code with Benign Code](#step-5-mix-fixed-code-with-benign-code)
- [Influence Score](#influence-score)

## Overview

The pipeline follows these stages:

```
Synthesize CWE-Inducing Instructions
        |
        v
Generate Vulnerable Code  +  Generate Benign Code
        |                           |
        v                           |
Detect Vulnerabilities (Purple Llama)
        |                           |
        v                           |
Generate Fixes & Re-detect          |
        |                           |
        v                           v
        Mix Fixed Code with Benign Code
                    |
                    v
            Final Training Dataset
```

## Prerequisites

- Python 3
- The tested model must be hosted via [vLLM](https://github.com/vllm-project/vllm) with an OpenAI-compatible API endpoint.
- [PurCL's Purple Llama](https://github.com/PurCL/PurpleLlama) is included as a git submodule under `PurpleLlama/`. After cloning this repo, run `git submodule update --init --recursive` to fetch it.

## Data Synthesis Pipeline

### Step 1: Synthesize CWE-Inducing Instructions

Synthesize instructions for a single CWE-language pair:

```shell
./synth_claude.sh <CWE_ID> <LANG>
```

This generates instructions and clusters them to select 2000 per pair.

To synthesize for all CWE-language pairs at once:

```shell
./synth_all.sh
```

> **Note:** Set the `HF_USER` environment variable to your HuggingFace username before running any scripts (e.g., `export HF_USER=your-hf-username`). Make sure to `mkdir` the output directory before running the script.

### Step 2: Generate Potentially Vulnerable Code

Generate vulnerable code for all CWE-language pairs using the tested model:

```shell
./infer_all_claude.sh
```

> **Note:** Modify `src/gen_inferences.py` to specify the addresses of the hosted vLLM model.

### Step 3: Generate Benign Code

Generate normal (non-vulnerable) code with the original instructions:

```shell
./infer_all_claude_ori_task.sh
```

> **Note:** Host the tested model via vLLM and modify `src/gen_inferences.py` accordingly.

### Step 4: Scan Vulnerable Code with Purple Llama

This step detects vulnerabilities, generates fixes, and pairs them up. It uses scripts from both this repo and the `PurpleLlama/` submodule.

#### 4a. Merge inference results

Create a symlink from the output directory of `infer_all_claude` to the `PurpleLlama/` directory, then merge the inference results:

```shell
python3 PurpleLlama/prosec_scripts/merge_multiple_infer_rets.py
```

Also merge the benign inference results to produce `infer-ret-original.jsonl`.

> **Note:** You need to manually modify the merge script before running it.

#### 4b. Detect vulnerabilities

```shell
python3 PurpleLlama/prosec_scripts/detect_all.py
```

This produces `detection-ret.jsonl`.

#### 4c. Generate fix prompts

```shell
python3 src/gen_fix_inference_prompts.py \
    --fin detection-ret.jsonl \
    --fout-stats detection-ret.stats.json \
    --fout detection-ret.fix-prompt.jsonl
```

#### 4d. Generate fixed code

```shell
python3 src/gen_fix_inference.py \
    --prompts_in detection-ret.fix-prompt.jsonl \
    --fout detection-ret.fixed.jsonl
```

> **Note:** Host the tested model and modify `src/gen_fix_inference.py`.

#### 4e. Re-detect on fixed code

```shell
python3 PurpleLlama/prosec_scripts/detect_all_from_fixed.py
```

This produces `detection-ret-fixed.jsonl`.

#### 4f. Pair and upload

```shell
python3 src/collect_and_upload_fixed_batch.py \
    --detection_ret detection-ret.jsonl \
    --fixed_detected_ret detection-ret-fixed.jsonl \
    --ds_name <name-of-the-dataset> \
    --fout <intermediate-results>
```

This produces a fix-pair dataset (e.g., `purcl/fix-dataset`).

### Step 5: Mix Fixed Code with Benign Code

Concatenate multiple CWE-inducing instruction datasets:

```shell
python3 src/concat_dataset.py
```

> **Note:** You will need to manually modify this file. Suppose the output is `purcl/concat-dataset`.

Clean the benign data and mix with the fixed code:

```shell
python3 src/clean_benign_data.py --fin infer-ret-original.jsonl

python3 src/mix_and_upload_original_w_fixed_batch.py \
    --inst_ds_name purcl/concat-dataset \
    --fix_pair_ds_name purcl/fix-dataset \
    --infer_ori_in infer-ret-original-filtered.jsonl \
    --out_ds_name <output-dataset-name>
```

## Data Selection Pipeline

The `influence_score/` module provides tools for computing training dynamics and influence scores over synthesized datasets. These scores measure how individual training samples contribute to security alignment, enabling better data selection strategies. **More detailed instructions will be published soon.**

Key components:

| Module | Description |
|--------|-------------|
| `data_utils_refactored.py` | Entry point: prepares selection datasets from instruction, fix-pair, and benign data |
| `training_dynamics_refactored.py` | Collects log-probabilities and accuracy across training checkpoints |
| `sample_refactored.py` | Data selection strategies based on training dynamics correlations |
| `scores.py` | Computes sequence-level log-probabilities and normalized scores |
| `collator.py` | Data collator for pairwise training data |
| `collect_grad_reps.py` | Gradient representation collection using TRAK for influence estimation |
