import argparse
import json
from tqdm import tqdm
import glob
import os

parser = argparse.ArgumentParser(description='Generate fix instructions')
parser.add_argument('--detection-explanation', type=str, default='detection-rule-bypass-explanations.jsonl')
parser.add_argument('--cwe-explanation-dir', type=str, default='cwe-explanations')
parser.add_argument('--fin', type=str, help='Input file', required=True)
parser.add_argument('--fout-stats', type=str, help='Output file for statistics', default='')
parser.add_argument('--fout', type=str, help='Output file', default='')

args = parser.parse_args()

if args.fout_stats == '':
    # only replace the extension
    extension = '.jsonl'    
    args.fout_stats = args.fin[:-len(extension)] + '.stats.json'

if args.fout == '':
    extension = '.jsonl'
    args.fout = args.fin[:-len(extension)] + '.fix-prompt.jsonl'

cwe_explanation_files = glob.glob(args.cwe_explanation_dir + '/*.txt')
cwe_explanation = {}
for cwe_explanation_file in cwe_explanation_files:
    fname_wo_ext = os.path.splitext(os.path.basename(cwe_explanation_file))[0]
    _, cwe_id, lang = fname_wo_ext.split('-')
    cwe_id = 'CWE-' + cwe_id
    lang = lang.lower()
    cwe_explanation[(lang, cwe_id)] = open(cwe_explanation_file, 'r').read().strip()

detection_rule_explanations = [json.loads(line) for line in open(args.detection_explanation, 'r')]
lang_pattern_id2detection_rule_explanation = {}
for detection_rule_explanation in detection_rule_explanations:
    pattern_id = detection_rule_explanation['pattern_id']
    lang = detection_rule_explanation['language']
    if (lang, pattern_id) in lang_pattern_id2detection_rule_explanation:
        print(f'Warning: duplicate detection rule explanation for {lang}-{pattern_id}')
    lang_pattern_id2detection_rule_explanation[(lang, pattern_id)] = detection_rule_explanation
    

data_in = [json.loads(line) for line in tqdm(open(args.fin, 'r'))]

code_w_cwe = []
cwe_triggered_cnt = {}
relevant_cwe_triggered_cnt = {}
expected_cwe_cnt = {}

for entry in tqdm(data_in):
    lang = entry['lang']
    expected_cwe = entry['cwe']
    if (lang, expected_cwe) not in expected_cwe_cnt:
        expected_cwe_cnt[(lang, expected_cwe)] = 0
    expected_cwe_cnt[(lang, expected_cwe)] += 1
    if len(entry['detection_results']) == 0:
        continue
    code_w_cwe.append(entry)
    all_detection_entries = entry['detection_results']
    seen_rules = set()
    uniq_triggered_rules = []
    for detection_entry in all_detection_entries:
        pattern_id = detection_entry['pattern_id']
        if pattern_id in seen_rules:
            continue
        uniq_triggered_rules.append(detection_entry)
        seen_rules.add(pattern_id)
        cwe = detection_entry['cwe_id']
        if (lang, cwe) not in cwe_triggered_cnt:
            cwe_triggered_cnt[(lang, cwe)] = 0
        cwe_triggered_cnt[(lang, cwe)] += 1
        if cwe == expected_cwe:
            if (lang, cwe) not in relevant_cwe_triggered_cnt:
                relevant_cwe_triggered_cnt[(lang, cwe)] = 0
            relevant_cwe_triggered_cnt[(lang, cwe)] += 1
    entry['uniq_detection_results'] = uniq_triggered_rules

# statistics
lc2relevant_trigger_ratio = {}
for (lang, cwe) in expected_cwe_cnt:
    relevant_triggered = 0
    if (lang, cwe) in relevant_cwe_triggered_cnt:
        relevant_triggered = relevant_cwe_triggered_cnt[(lang, cwe)]
    expected = expected_cwe_cnt[(lang, cwe)]
    ratio = relevant_triggered / expected
    if lang not in lc2relevant_trigger_ratio:
        lc2relevant_trigger_ratio[lang] = {}
    lc2relevant_trigger_ratio[lang][cwe] = {
        'ratio': ratio,
        'relevant_triggered': relevant_triggered,
        'expected': expected
    }
    
# sort by ratio
lc2relevant_trigger_ratio_sorted = {}
for lang in lc2relevant_trigger_ratio:
    lc2relevant_trigger_ratio_sorted[lang] = sorted(lc2relevant_trigger_ratio[lang].items(), key=lambda x: x[1]['ratio'], reverse=True)

with open(args.fout_stats, 'w') as f:
    json.dump(lc2relevant_trigger_ratio_sorted, f, indent=2)


ret_entries = []
for vul_code_entry in tqdm(code_w_cwe):
    lang = vul_code_entry['lang']
    cwe = vul_code_entry['cwe']
    triggered_rule_explanations = []
    for triggered in vul_code_entry['uniq_detection_results']:
        cwe_id = triggered['cwe_id']
        pattern_id = triggered['pattern_id']
        analyzer = triggered['analyzer']
        if (lang, pattern_id) not in lang_pattern_id2detection_rule_explanation:
            # print(f'Warning: no explanation for {lang}-{pattern_id}')
            continue
        explanation = lang_pattern_id2detection_rule_explanation[(lang, pattern_id)]
        triggered_rule_explanations.append(explanation)

    if len(triggered_rule_explanations) == 0:
        # use cwe explanations instead
        triggered_cwe_explanations = []
        seen_cwe = set()
        for triggered in vul_code_entry['uniq_detection_results']:
            cwe_id = triggered['cwe_id']
            if cwe_id in seen_cwe:
                continue
            seen_cwe.add(cwe_id)            
            if (lang, cwe_id) not in cwe_explanation:
                if lang != 'cpp':
                    print(f'Warning: no explanation for {lang}-{cwe_id}')
                    continue
                if ('c', cwe_id) not in cwe_explanation:
                    print(f'Warning: no explanation for cpp-{cwe_id}')
                    continue
                explanation = cwe_explanation[('c', cwe_id)]
            else:
                explanation = cwe_explanation[(lang, cwe_id)]
            triggered_cwe_explanations.append(explanation)            
        explanation_text = "Here are the explanations for the CWEs:\n" + '\n'.join(triggered_cwe_explanations)
    else:
        explanation_text = "Here are the explanations for the triggered CWE detection rules, and their potential fixes:\n" 
        for i, explanation in enumerate(triggered_rule_explanations):
            explanation_text += "<Rule %d>\n"%i
            explanation_text += 'Triggered CWE: %s'%explanation['cwe_identifier'] + '\n'
            explanation_text += 'Explanation: %s'%explanation['detection_rule_description'] + '\n'
            explanation_text += 'Potential Fixes: %s'%str(explanation['better_practices']) + '\n'
            explanation_text += '</Rule %d>\n'%i


    system_prompt = """
You are a security expert helping developer fix potential CWEs in their code.

I will give you a snippet of code. The code triggers potential CWE detectors.
Here are the details for the triggered rules/CWEs:

Details:
%s

Your action are three-steps:

Step1: Analyze why the code triggeres the corresponding CWE detector.

Step2: For each triggered CWE detector, provide a potential fix plan based on the explanation.

Step3: Incorporate all the potential fixes into the code snippet.
Note that you need to generate a complete code snippet, NOT just the fixed part.
For example, do NOT skip lines that are not changed. Do NOT make irrelevant changes.

Wrap the fixed code in a code block.
That is,

```
...(fixed code)
```
    """%explanation_text

    vul_code = vul_code_entry['code']
    code_task = vul_code_entry['prompt']
    user_prompt = "Programming task: %s\n\nProblematic code:\n\n```\n%s\n```\n"%(code_task, vul_code)
    ret_entries.append({
        'fix_system_prompt': system_prompt,
        'fix_user_prompt': user_prompt,
        **vul_code_entry
    })

with open(args.fout, 'w') as f:
    for entry in ret_entries:
        f.write(json.dumps(entry) + '\n')
        



print()
