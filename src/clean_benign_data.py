import json
import os
import argparse
from tqdm import tqdm
import datasets
import utils
import numpy as np
# use tree-sitter to parse code
import tree_sitter as ts
import ts_utils
from ts_utils import lang_parser_map, lang2function_node_name

from fuzzywuzzy import fuzz

parser = argparse.ArgumentParser()
parser.add_argument("--fin", type=str, default="syn-rets/infer-ret-claude-haiku-inst-ministral-ori-task/infer-ret-all.jsonl")
parser.add_argument("--detected_in", type=str, default="syn-rets/infer-ret-claude-haiku-inst-ministral-ori-task/infer-ret-all.jsonl.detected.jsonl")
parser.add_argument("--fout", type=str, default="")

args = parser.parse_args()

if args.fout == "":
    args.fout = args.fin.replace(".jsonl", "-filtered.jsonl")

data_in = [json.loads(l) for l in tqdm(open(args.fin, "r"))]
if args.detected_in != "":
    data_detected = [json.loads(l) for l in tqdm(open(args.detected_in, "r"))]
else:
    data_detected = []

problematic = set()
for entry in tqdm(data_detected, desc="Collecting problematic entries"):
    if len(entry['detection_results']) > 0:
        problematic.add(entry['code'])

print()

def filter_data(entry):
    new_code_blocks = []
    problematic_code = 0    
    parser = lang_parser_map[entry['lang']]
    for code_blk in entry['code_blocks']:
        if code_blk in problematic:
            problematic_code += 1
            continue
        stripped_code_blk = code_blk.strip()
        if '...' in stripped_code_blk:
            continue
        if len(stripped_code_blk.split('\n')) < 5:
            continue
        parsed_code = parser.parse(bytes(code_blk, "utf8"))
        if parsed_code.root_node.has_error:
            continue
        if len(stripped_code_blk.split('\n')) > 10:
            new_code_blocks.append(code_blk)
            continue
        else:
            to_find = lang2function_node_name[entry['lang']]
            func_dec = ts_utils.find_first_recursively_opt(parsed_code.root_node, to_find)
            if func_dec is None:
                continue
            new_code_blocks.append(code_blk)
    entry['code_blocks'] = new_code_blocks
    entry['problematic_code'] = problematic_code
    return entry


fout = open(args.fout, "w")
import multiprocessing
pool = multiprocessing.Pool(32)

data_out = pool.imap_unordered(filter_data, data_in)
for entry in tqdm(data_out, desc="Writing filtered data", total=len(data_in)):
    fout.write(json.dumps(entry) + '\n')
fout.close()

exit(0)




# ori_prompt2impl = {}
# for entry in tqdm(data_in, desc="Loading original inference results"):    
#     prompt = entry['prompt']
#     if prompt not in ori_prompt2impl:
#         ori_prompt2impl[prompt] = []
#     ori_prompt2impl[prompt].append(entry)

# too_short = {}
# not_too_short = {}
# for prompt, impls in ori_prompt2impl.items():
#     lang = impls[0]['lang']
#     code_blks_all = []
#     for impl in impls:
#         code_blks_all.extend(impl['code_blocks'])
#     current_too_short = []
#     current_not_too_short = []
#     for code_blk in code_blks_all:
#         if len(code_blk.strip().split('\n')) < 5:
#             current_too_short.append(code_blk)
#         else:
#             current_not_too_short.append(code_blk)
#     if len(current_too_short) > 0:
#         too_short[(lang, prompt)] = current_too_short
#     if len(current_not_too_short) > 0:
#         not_too_short[(lang, prompt)] = current_not_too_short


# # dbg = [e for e in not_too_short.items() if e[0][0] == 'cpp']
# # dbg[1]

# prompt2parsable_not_too_short = {}
# for (lang, prompt), code_blks in tqdm(not_too_short.items(), desc="Filtering out unparsable implementations"):
#     all_impls = code_blks
#     parsable_impls = []
#     parser = lang_parser_map[lang]
#     for impl in all_impls:
#         parsed_code = parser.parse(bytes(impl, "utf8"))        
#         if not parsed_code.root_node.has_error:
#             if len(impl.strip().split('\n')) > 10:
#                 parsable_impls.append(impl)
#             else:
#                 to_find = lang2function_node_name[lang]
#                 func_dec = ts_utils.find_first_recursively_opt(parsed_code.root_node, to_find)
#                 if func_dec is not None:
#                     parsable_impls.append(impl)

#     if len(parsable_impls) > 0:
#         prompt2parsable_not_too_short[prompt] = {
#             'lang': lang,
#             'prompt': prompt,
#             'impls': parsable_impls
#         }

# all_possible_code = []
# for prompt, entry in prompt2parsable_not_too_short.items():
#     all_possible_code.extend(entry['impls'])

# all_code_sort_by_len = sorted(all_possible_code, key=lambda x: len(x.strip()))

# lang_cnt = {}
# for prompt, entry in prompt2parsable_not_too_short.items():
#     lang = entry['lang']
#     if lang not in lang_cnt:
#         lang_cnt[lang] = 0
#     lang_cnt[lang] += 1

# ori_lang_cnt = {}
# for prompt, entry in ori_prompt2impl.items():
#     lang = entry[0]['lang']
#     if lang not in ori_lang_cnt:
#         ori_lang_cnt[lang] = 0
#     ori_lang_cnt[lang] += 1

# not_too_short_lang_cnt = {}
# for (lang, prompt), code_blks in not_too_short.items():
#     if lang not in not_too_short_lang_cnt:
#         not_too_short_lang_cnt[lang] = 0
#     not_too_short_lang_cnt[lang] += 1

# dbg = 'function generateRandomInteger() {\n  return Math.floor(Math.random() * 10) + 1;\n}\n\nconsole.log(generateRandomInteger());'
# parser = lang_parser_map['javascript']
# root = parser.parse(bytes(dbg, "utf8")).root_node

# print()