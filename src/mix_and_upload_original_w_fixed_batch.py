import datasets
import argparse
import json
import multiprocessing
import openai
from tqdm import tqdm
import os
import utils
import numpy as np
# use tree-sitter to parse code
import os
import datasets
# use tree-sitter to parse code
import tree_sitter as ts
import ts_utils
from ts_utils import lang_parser_map



parser = argparse.ArgumentParser()
parser.add_argument("--inst_ds_name", type=str, required=True)
parser.add_argument("--fix_pair_ds_name", type=str, required=True)
parser.add_argument("--infer_ori_in", type=str, required=True)
parser.add_argument("--out_ds_name", type=str, required=True)
args = parser.parse_args()

ori_prompt_lang2impl = {}
for entry in tqdm(open(args.infer_ori_in, 'r'), desc="Loading original inference results"):
    entry = json.loads(entry)
    prompt = entry['prompt']
    lang = entry['lang']
    if (prompt, lang) not in ori_prompt_lang2impl:
        ori_prompt_lang2impl[(prompt, lang)] = []
    ori_prompt_lang2impl[(prompt, lang)].append(entry)

print("NOTE that now I assume the input benign data is already filtered!")
ori_prompt_lang2parsable_impl = {}
for (prompt, lang), impls in tqdm(ori_prompt_lang2impl.items(), desc="Filtering out unparsable implementations"):
    all_impls = []
    for impl in impls:
        all_impls.extend(impl['code_blocks'])
    
    parsable_impls = all_impls
    # parsable_impls = []
    # parser = lang_parser_map[impls[0]['lang']]
    # for impl in all_impls:
    #     parsed_code = parser.parse(bytes(impl, "utf8"))
    #     if not parsed_code.root_node.has_error:
    #         parsable_impls.append(impl)

    if len(parsable_impls) > 0:
        ori_prompt_lang2parsable_impl[(prompt, lang)] = {
            # 'cwe': impls[0]['cwe'],
            'lang': lang,
            'prompt': prompt,
            'impls': parsable_impls
        }

inst_ds = datasets.load_dataset(args.inst_ds_name)['train']

# ori_prompt_lang2inducing_prompt = {}
inducing_prompt2ori_prompt_lang = {}
for entry in tqdm(inst_ds):
    ori_prompt = entry['original_prompt']
    lang = entry['lang']
    inducing_prompt = entry['task']
    # assert (ori_prompt, lang) not in ori_prompt_lang2inducing_prompt
    assert inducing_prompt not in inducing_prompt2ori_prompt_lang
    # ori_prompt_lang2inducing_prompt[(ori_prompt, lang)] = inducing_prompt
    inducing_prompt2ori_prompt_lang[inducing_prompt] = (ori_prompt, lang)    

fix_pair_ds = datasets.load_dataset(args.fix_pair_ds_name)['train']
# add new column 'benign' to fix_pair_ds, and set all to False
fix_pair_ds = fix_pair_ds.add_column('benign', [False]*len(fix_pair_ds))

count = 0
additional_entries = []
np.random.seed(42)
for entry in tqdm(fix_pair_ds, desc="Generating additional entries"):
    inducing_prompt = entry['original_instruction']
    lang = entry['lang']
    if inducing_prompt not in inducing_prompt2ori_prompt_lang:
        continue
    ori_prompt, lang = inducing_prompt2ori_prompt_lang[inducing_prompt]
    if (ori_prompt, lang) not in ori_prompt_lang2parsable_impl:
        continue
    parsable_impls = ori_prompt_lang2parsable_impl[(ori_prompt, lang)]['impls']
    benign_code = np.random.choice(parsable_impls)
    count += len(parsable_impls)
    additional_entries.append({
        'lang': entry['lang'],
        'cwe': entry['cwe'],
        'original_instruction': ori_prompt,
        'original_code': entry['fixed_code'],
        'fixed_code': benign_code,
        'benign': True,
        'empty': ''
    })


print("In total, there are %d possible benign code snippets combinations" % count)

new_ds = datasets.Dataset.from_list(additional_entries)
all_ds = datasets.concatenate_datasets([fix_pair_ds, new_ds])
all_ds_shuffled = all_ds.shuffle(seed=42)
all_ds_shuffled.push_to_hub(args.out_ds_name, private=True)




print()