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
from ts_utils import lang_parser_map

from fuzzywuzzy import fuzz
from codebleu import calc_codebleu


parser = argparse.ArgumentParser()
parser.add_argument("--detection_ret", type=str, default="rebuttal_data_codelmsec/syn_w_codelmsec-w-prompt.jsonl.detected.jsonl")
parser.add_argument("--fixed_detected_ret", type=str, default="rebuttal_data_codelmsec/syn_w_codelmsec-w-prompt.jsonl.detected.fixed.jsonl.detected.jsonl")
parser.add_argument('--fout', type=str, default="rebuttal_data_codelmsec/syn_w_codelmsec-w-prompt.fixed-new-feasible.jsonl")
# parser.add_argument("--ds_name", type=str, default="<hf_user>/secalign-phi3mini-inst-all-pair")
parser.add_argument("--ds_name", type=str, default="-")

args = parser.parse_args()


fixed_detected_ret = [
    json.loads(line) for line in tqdm(open(args.fixed_detected_ret, "r"), desc="Loading fixed detected results")
]

def pick_fixed_code(entry):
    code_detected_entries = entry['code_detected_entries']
    # first, filter out entries cannot be compiled
    compilable_entries = []
    lang = entry['lang']
    lang_parser = lang_parser_map[lang]
    for code_entry in code_detected_entries:
        code = code_entry['code']
        parsed_code = lang_parser.parse(bytes(code, "utf8"))
        if parsed_code.root_node.has_error:
            continue
        compilable_entries.append(code_entry)

    no_cwe_entries = []
    for code_entry in compilable_entries:
        if len(code_entry['detection_results']) == 0:
            no_cwe_entries.append(code_entry)
    
    if len(no_cwe_entries) > 0:
        return no_cwe_entries
    
    # if every one has cwe, pick ones without the expected cwe
    expected_cwe = entry['cwe']
    no_expected_cwe_entries = []
    for code_entry in compilable_entries:
        detection_rets = [d for d in code_entry['detection_results'] if d['cwe_id'] == expected_cwe]
        if len(detection_rets) == 0:
            no_expected_cwe_entries.append(code_entry)
        
    return no_expected_cwe_entries


fixed_detected_ret_w_feasible_fixes = []
for entry in tqdm(fixed_detected_ret, desc="Picking feasible fixes"):
    feasible_fixes = pick_fixed_code(entry)
    if len(feasible_fixes) > 0:
        entry['feasible_fixes'] = feasible_fixes
        fixed_detected_ret_w_feasible_fixes.append(entry)

with open(args.fout, "w") as f:
    for entry in fixed_detected_ret_w_feasible_fixes:
        f.write(json.dumps(entry) + "\n")

# fixed_detected_ret_w_feasible_fixes = [
#     json.loads(line) for line in tqdm(open(args.fout, "r"), desc="Loading fixed detected results with feasible fixes")
# ]
# # # # exit(0)
detection_results = [
    json.loads(line) for line in tqdm(open(args.detection_ret, "r"), desc="Loading detection results")
]

prompt_code2detection_ret_entry = {}
for entry in detection_results:
    code = entry["code"]
    prompt = entry["prompt"]
    prompt_code2detection_ret_entry[(prompt, code)] = entry


irr = []
prompt2entries = {}
for entry in tqdm(fixed_detected_ret_w_feasible_fixes, desc="Filtering out irrelevant entries"):
    prompt = entry["prompt"]
    code = entry["vul_code"]
    if (prompt, code) not in prompt_code2detection_ret_entry:
        print("Error: entry not found in detection results")
        break
    # skip irrelevant entries
    expected_cwe = entry["cwe"]
    current_detection_ret = prompt_code2detection_ret_entry[(prompt, code)][
        "detection_results"
    ]
    entry["vul_detection_results"] = current_detection_ret
    if expected_cwe not in [e["cwe_id"] for e in current_detection_ret]:
        irr.append(entry)
        continue
    if prompt not in prompt2entries:
        prompt2entries[prompt] = []
    prompt2entries[prompt].append(entry)
print(f"irrelevant entries: {len(irr)}")
print("Total number of entries: ", len(fixed_detected_ret_w_feasible_fixes))

prompt_entries_list = list(prompt2entries.items())


def cb_selection_for_one_entry(prompt_entries):
    def heuristic_filter(vul_code, fix_code):
        if len(fix_code.strip()) * 2 < len(vul_code.strip()):
            return False
        if "..." in fix_code or "unchange" in fix_code.lower() or 'no change' in fix_code.lower():
            return False
        return True
    ret = []
    entries = prompt_entries[1]
    for entry in entries:
        vul_code = entry["vul_code"]
        fix_code_all = [e["code"] for e in entry['feasible_fixes'] if heuristic_filter(vul_code, e["code"])]
        if len(fix_code_all) == 0:
            continue
        fix_codebleu_score = []
        for fix_code in fix_code_all:
            codeblue_score_all = calc_codebleu(
                references=[vul_code], 
                predictions=[fix_code],
                lang=entry["lang"])
            codebleu_score = codeblue_score_all['codebleu']
            fix_codebleu_score.append((codebleu_score, fix_code))
        fix_code_sorted_by_codebleu = sorted(fix_codebleu_score, key=lambda x: x[0], reverse=True)
        entry['selected_fix_code'] = fix_code_sorted_by_codebleu[0][1]        
        ret.append(entry)    
    return {
        "prompt": prompt_entries[0],
        "entries": ret[:3]
    }

final_entries_ds_format = []
import multiprocessing

pool = multiprocessing.Pool(64)
cb_selected_rets = pool.imap_unordered(cb_selection_for_one_entry, prompt_entries_list)
for ret in tqdm(cb_selected_rets, desc="Processing codebleu selection", total=len(prompt_entries_list)):
    prompt = ret['prompt']
    entries = ret['entries']
    for entry in entries:
        lang = entry["lang"]
        cwe = entry["cwe"]
        vul_code = entry["vul_code"]
        fixed_code = entry["selected_fix_code"]
        final_entries_ds_format.append({
            "lang": lang,
            "cwe": cwe,
            "original_instruction": prompt,
            "original_code": vul_code,
            'empty': "",
            "fixed_code": fixed_code
        })

# ################################################
# # use heuristic to filter out some entries
# ################################################

# potential_data_triples = []
# for prompt, entries in prompt2entries.items():
#     for e in entries:
#         vul_code = e["vul_code"]
#         for fix_entry in e["feasible_fixes"]:
#             fix_code = fix_entry["code"]
#             potential_data_triples.append(
#                 {
#                     "prompt": prompt,
#                     "vul_code": vul_code,
#                     "fix_code": fix_code,
#                     "vul_type": e["cwe"],
#                     "lang": e["lang"],
#                 }
#             )

# too_short = []
# not_too_short = []
# for triple in tqdm(potential_data_triples, desc="filtering out too short entries"):
#     prompt = triple["prompt"]
#     vul_code = triple["vul_code"]
#     fix_code = triple["fix_code"]
#     if len(fix_code.strip()) * 2 < len(vul_code.strip()):
#         too_short.append(triple)
#     else:
#         not_too_short.append(triple)


# unexpected_skip = []
# not_skipped = []
# for triple in tqdm(not_too_short, desc="filtering out unexpected skips"):
#     prompt = triple["prompt"]
#     vul_code = triple["vul_code"]
#     fix_code = triple["fix_code"]
#     if "..." in fix_code or "unchange" in fix_code or 'no change' in fix_code:
#         unexpected_skip.append(triple)
#     else:
#         not_skipped.append(triple)


# prompt2distinct_entries = {}
# for triple in tqdm(not_skipped, desc="dedup"):
#     prompt = triple["prompt"]
#     if prompt not in prompt2distinct_entries:
#         prompt2distinct_entries[prompt] = []
#     fix_code = triple["fix_code"]
#     if len(prompt2distinct_entries[prompt]) > 3:
#         continue
#     for existing in prompt2distinct_entries[prompt]:
#         if fuzz.ratio(fix_code, existing["fix_code"]) > 60:
#             break
#     else:
#         prompt2distinct_entries[prompt].append(triple)


# final_entries_ds_format = []
# np.random.seed(42)
# for prompt, entries in prompt2distinct_entries.items():
#     for entry in entries:
#         lang = entry["lang"]
#         cwe = entry["vul_type"]
#         vul_code = entry["vul_code"]
#         fixed_code = entry["fix_code"]
#         final_entries_ds_format.append({
#             "lang": lang,
#             "cwe": cwe,
#             "original_instruction": prompt,
#             "original_code": vul_code,
#             'empty': "",
#             "fixed_code": fixed_code
#         })

# #####################################################################################

ds = datasets.Dataset.from_list(final_entries_ds_format)
ds = ds.shuffle(seed=42)
ds.push_to_hub(args.ds_name, private=True)


exit(0)

################################
# previous
################################

prompt2entries = {}
for entry in fixed_detected_ret_w_feasible_fixes:
    prompt = entry['prompt']
    if prompt not in prompt2entries:
        prompt2entries[prompt] = []
    prompt2entries[prompt].append(entry)

final_entries = []
np.random.seed(42)
for prompt, entries in prompt2entries.items():
    if len(entries) > 3:
        entries = np.random.permutation(entries)[:3]
    final_entries.extend(entries)


final_entries_ds_format = []
np.random.seed(42)
for entry in final_entries:
    lang = entry['lang']
    cwe = entry['cwe']
    prompt = entry['prompt']
    vul_code = entry['vul_code']
    selected_fix = np.random.choice(entry['feasible_fixes'])
    fixed_code = selected_fix['code']
    final_entries_ds_format.append({
        "lang": lang,
        "cwe": cwe,
        "original_instruction": prompt,
        "original_code": vul_code,
        'empty': "",
        "fixed_code": fixed_code
    })
    

ds = datasets.Dataset.from_list(final_entries_ds_format)
ds = ds.shuffle(seed=42)
ds.push_to_hub(args.ds_name, private=True)

dbg_out = 'dbg-out-data/'
os.system(f"rm -rf {dbg_out}")
os.makedirs(dbg_out, exist_ok=True)
for i, entry in enumerate(ds):
    with open(f"{dbg_out}/{i}.{entry['lang']}", "w") as f:
        f.write("/* Instructions: \n %s \n*/\n" % entry['original_instruction'])
        f.write(entry['original_code'])
        f.write("\n// =========================\n")
        f.write(entry['fixed_code'])
        f.write("\n")
    if i > 100:
        break


print()

if 2+2<5:
    exit(0)

dbg_py = [e for e in tqdm(fixed_detected_ret_w_feasible_fixes) if e['lang'] == 'python']
dbg_py89 = [e for e in tqdm(dbg_py) if '89' in e['cwe']]

to_analyze  = dbg_py89
dbg_prompt2entries = {}
for entry in to_analyze:
    prompt = entry['prompt']
    if prompt not in dbg_prompt2entries:
        dbg_prompt2entries[prompt] = []
    dbg_prompt2entries[prompt].append(entry)


dbg_items = list(dbg_prompt2entries.items())
dbg_out = 'dbg-out/'
if os.path.exists(dbg_out):
    os.system(f"rm -rf {dbg_out}")
os.makedirs(dbg_out, exist_ok=True)
for i, (prompt, entries) in enumerate(dbg_items[:100]):
    current_out_dir = f"{dbg_out}/{i}"
    os.makedirs(current_out_dir, exist_ok=True)
    with open(f"{current_out_dir}/prompt.txt", "w") as f:
        f.write(prompt)
    for k, entry in enumerate(entries):
        vul_code = entry['vul_code']
        vul_code_fout = f"{current_out_dir}/{k}-vul.py"
        with open(vul_code_fout, "w") as f:
            f.write(vul_code)
        for fix_id, fix in enumerate(entry['feasible_fixes']):
            fixed_code = fix['code']
            detection_ret = fix['detection_results']
            fixed_code_fout = f"{current_out_dir}/{k}-vul-fix-{fix_id}.py"
            with open(fixed_code_fout, "w") as f:
                f.write(fixed_code)
                f.write("\n\n\n\"\"\"")
                f.write("Detection results:\n")
                f.write(json.dumps(detection_ret, indent=2))
                f.write("\n\n\n")
                f.write("\"\"\"")

    
    

# dbg
fix_prompt_in = [json.loads(l) for l in tqdm(open('infer-ret-detected/infer-ret-detected-all.phi3mini-inst.fix-prompt.jsonl', 'r'))]
interesting_entries = [e for e in fixed_detected_ret if e['lang'] == 'cpp' and e['cwe'] == 'CWE-119']

interesting_fix_prompt = [e for e in fix_prompt_in if e['lang'] == 'cpp' and e['cwe'] == 'CWE-119']
relevant_interesting_fix_prompt = []
for entry in fix_prompt_in:
    lang = entry['lang']
    cwe = entry['cwe']
    if lang == 'cpp' and cwe == 'CWE-119':
        detecterd_cwe_ids = [d['cwe_id'] for d in entry['detection_results']]
        if 'CWE-119' in detecterd_cwe_ids:
            relevant_interesting_fix_prompt.append(entry)

problematic_entries = []
for entry in tqdm(interesting_entries):
    potential_fixes = pick_fixed_code(entry)
    if len(potential_fixes) == 0:
        problematic_entries.append(entry)





########################################################################################
