import sys
sys.path.append('../src')
from ts_utils import lang_parser_map

from datasets import (
    load_dataset,
    concatenate_datasets,
    Dataset, DatasetDict
)
import argparse
import json
import os
import torch
from tqdm import tqdm


# inspired by https://github.com/PurCL/SeCAlign-llama-factory/blob/main/src/llamafactory/data/processors/pairwise.py#L19
def encode_oneturn_phi3(instruction, reponse, tokenizer):
    "encode one turn of the conversation"
    SYSTEM_PROMPT="You are helpful coding assistant."
    prompt_ids = tokenizer.apply_chat_template(
        [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': instruction
            },
        ],
        # tokenize=False
    )
    label_ids = tokenizer.apply_chat_template(
        [
            {
                'role': 'assistant',
                'content': reponse
            }
        ],
        # tokenize=False
    )
    input_ids = prompt_ids[:-1] + label_ids
    label_ids = [-100] * len(prompt_ids[:-1]) + label_ids
    return input_ids, label_ids


def encode_oneturn_llama2(instruction, reponse, tokenizer):
    "encode one turn of the conversation"
    SYSTEM_PROMPT="You are helpful coding assistant."
    prompt_ids = tokenizer.apply_chat_template(
        [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': instruction
            },
        ],
        # tokenize=False
    )
    input_ids = tokenizer.apply_chat_template(
        [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': instruction
            },
            {
                'role': 'assistant',
                'content': reponse
            }
        ],
        # tokenize=False
    )
    label_ids = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    
    return input_ids, label_ids


if False:
    encode_oneturn = encode_oneturn_phi3
else:
    encode_oneturn = encode_oneturn_llama2


def test_encode_oneturn():
    from transformers import AutoTokenizer
    encode_oneturn_llama2(
        'Write a function that takes a list of integers and returns the sum of all the integers in the list.',
        'def sum_list(lst):\n    return sum(lst)',
        tokenizer=AutoTokenizer.from_pretrained('meta-llama/CodeLlama-7b-Instruct-hf')
    )


def preprocess_function(example, tokenizer):
    "preprocessing paired dataset (all)"
    
    instruction = example['original_instruction']
    chosen_response = example['fixed_code']
    rejected_response = example['original_code']
    benign = example['benign']
        
    chosen_input_ids, chosen_label_ids = encode_oneturn(instruction, chosen_response, tokenizer)
    rejected_input_ids, rejected_label_ids = encode_oneturn(instruction, rejected_response, tokenizer)
    
    return {
        'chosen_input_ids': chosen_input_ids,
        'chosen_labels': chosen_label_ids,
        'rejected_input_ids': rejected_input_ids,
        'rejected_labels': rejected_label_ids,
        'benign': benign
    }


def preprocess_function_batch(examples, tokenizer):
    "preprocessing paired dataset (all)"
    
    results = {
        'chosen_input_ids': [],
        # 'chosen_attention_mask': [],
        'chosen_labels': [],
        'rejected_input_ids': [],
        # 'rejected_attention_mask': [],
        'rejected_labels': [],
        'benign': []
    }
    
    for instruction, chosen_response, rejected_response, benign in \
        zip(
            examples['original_instruction'], 
            examples['fixed_code'], 
            examples['original_code'], 
            examples['benign']
        ):
        
        chosen_input_ids, chosen_label_ids = encode_oneturn(instruction, chosen_response, tokenizer)
        rejected_input_ids, rejected_label_ids = encode_oneturn(instruction, rejected_response, tokenizer)
        
        results['chosen_input_ids'].append(chosen_input_ids)
        results['chosen_labels'].append(chosen_label_ids)
        results['rejected_input_ids'].append(rejected_input_ids)
        results['rejected_labels'].append(rejected_label_ids)
        results['benign'].append(benign)
    
    
    # padding (FIXME: seems problematic with dataloader)
    max_length = max(
        max(len(x) for x in results['chosen_input_ids']),
        max(len(x) for x in results['rejected_input_ids'])
    )
    results['chosen_input_ids'] = torch.tensor(
        [[tokenizer.pad_token_id] * (max_length - len(x)) + x for x in results['chosen_input_ids']]
    )
    results['chosen_labels'] = torch.tensor(
        [[-100] * (max_length - len(x)) + x for x in results['chosen_labels']]
    )
    results['rejected_input_ids'] = torch.tensor(
        [[tokenizer.pad_token_id] * (max_length - len(x)) + x for x in results['rejected_input_ids']]
    )
    results['rejected_labels'] = torch.tensor(
        [[-100] * (max_length - len(x)) + x for x in results['rejected_labels']]
    )
    results['benign'] = torch.tensor(results['benign'])
    
    return results


def prepare_selection_dataset(
    language=None,
    cached=False
):
    """
    Prepare dataset for selection task.
    1. load benign
    2. load inst
    
    """
    
    # load benign
    if cached:
        ori_prompt2impl = json.load(open('../ori_prompt2impl.json', 'r'))
        ori_prompt2parsable_impl = json.load(open('../ori_prompt2parsable_impl.json', 'r'))
        
    else:
        ori_prompt2impl = {}
        for entry in tqdm(open('../infer-ret-ori-task-haiku.phi3m-inst.jsonl', 'r'), desc="Loading original inference results"):
            entry = json.loads(entry)
            prompt = entry['prompt']
            if prompt not in ori_prompt2impl:
                ori_prompt2impl[prompt] = []
            ori_prompt2impl[prompt].append(entry)
            
        ori_prompt2parsable_impl = {}
        for prompt, impls in tqdm(ori_prompt2impl.items(), desc="Filtering out unparsable implementations"):
            all_impls = []
            for impl in impls:
                all_impls.extend(impl['code_blocks'])
            # assert all([impls[0]['lang'] == impl['lang'] for impl in impls])
            # print([impl['lang'] for impl in impls])
            
            parsable_impls = []
            parser = lang_parser_map[impls[0]['lang']]  # FIXME: not necessarily the first one
            for impl in all_impls:
                parsed_code = parser.parse(bytes(impl, "utf8"))
                if not parsed_code.root_node.has_error:
                    parsable_impls.append(impl)

            if len(parsable_impls) > 0:
                ori_prompt2parsable_impl[prompt] = {
                    # 'cwe': impls[0]['cwe'],
                    # 'lang': impls[0]['lang'],
                    'cwe': [impl['cwe'] for impl in impls],
                    'lang': [impl['lang'] for impl in impls],
                    'prompt': prompt,
                    'impls': parsable_impls  # NOTE: all parsable implementations
                }
        
        with open('../ori_prompt2impl.json', 'w') as f:
            json.dump(ori_prompt2impl, f, indent=2)
        with open('../ori_prompt2parsable_impl.json', 'w') as f:
            json.dump(ori_prompt2parsable_impl, f, indent=2)
    
    # load inst
    inst_ds = load_dataset('PurCL/secalign-all-haiku-inst-clustered2k')['train']
    if language:
        inst_ds = inst_ds.filter(lambda x: x['lang'] == language)
    print(f'language: {language}, inst_ds subset size: {len(inst_ds)}')
    
    # pair instructions (benign and malicious) with bimap
    ori_prompt2inducing_prompt = {}
    inducing_prompt2ori_prompt = {}
    for entry in tqdm(inst_ds):
        ori_prompt = entry['original_prompt']
        inducing_prompt = entry['task']
        ori_prompt2inducing_prompt[ori_prompt] = inducing_prompt
        inducing_prompt2ori_prompt[inducing_prompt] = ori_prompt
    print('o2i size: {}, i2o size: {}'.format(len(ori_prompt2inducing_prompt), len(inducing_prompt2ori_prompt)))
    
    # with open('../ori_prompt2inducing_prompt.json', 'w') as f:
    #     json.dump(ori_prompt2inducing_prompt, f, indent=2)
    # with open('../inducing_prompt2ori_prompt.json', 'w') as f:
    #     json.dump(inducing_prompt2ori_prompt, f, indent=2)

    # load fix_pair_ds    
    fix_pair_ds = load_dataset('PurCL/secalign-claude-phi3mini-inst-all-pair')['train']
    if language:
        fix_pair_ds = fix_pair_ds.filter(lambda x: x['lang'] == language)
    fix_pair_ds = fix_pair_ds.add_column('benign', [False] * len(fix_pair_ds))
    print(f'language: {language}, fix_pair_ds subset size: {len(fix_pair_ds)}')
    
    # return all data (including all parsable benign implementations)
    # FIXME: benign implementations are differentiated by language
    additional_entries = []
    for entry in tqdm(fix_pair_ds, desc="Generating additional entries"):
        inducing_prompt = entry['original_instruction']
        if inducing_prompt not in inducing_prompt2ori_prompt:
            continue
        ori_prompt = inducing_prompt2ori_prompt[inducing_prompt]
        if ori_prompt not in ori_prompt2parsable_impl:
            continue
        parsable_impls = ori_prompt2parsable_impl[ori_prompt]['impls']
        lang_parsable_impls = ori_prompt2parsable_impl[ori_prompt]['lang']
        # benign_code = np.random.choice(parsable_impls)
        for benign_code, l in zip(parsable_impls, lang_parsable_impls):
            if language and l != language:
                continue
            additional_entries.append({
                'lang': entry['lang'],
                'cwe': entry['cwe'],
                'original_instruction': ori_prompt,
                'original_code': entry['fixed_code'],
                'fixed_code': benign_code,
                'benign': True
            })
            
    new_ds = Dataset.from_list(additional_entries)
    all_ds = concatenate_datasets([fix_pair_ds, new_ds])
    
    return all_ds


def prepare_selection_dataset_new(
    inst_dataset_name_or_path='PurCL/secalign-all-haiku-inst-clustered2k',
    dsec_name_or_path='PurCL/secalign-claude-phi3mini-inst-all-pair',
    dnorm_json_path='../infer-ret-ori-task-haiku.phi3m-inst.jsonl',
    language=None,
    cached=False,
    cache_dir='..',
    lang_cwe_distinct=False
):
    """
    Prepare dataset for selection task.
    
    What changed:
    1. previously not differentiated by language and cwe, currently differentiated
    """
    
    # load benign
    if cached:
        with open(os.path.join(cache_dir, 'ori_prompt2impl.json'), 'r') as f:
            ori_prompt2impl = json.load(f)
        with open(os.path.join(cache_dir, 'ori_prompt2parsable_impl.json'), 'r') as f:
            ori_prompt2parsable_impl = json.load(f)
        with open(os.path.join(cache_dir, 'ori_prompt_lang_cwe2parsable_impl.json'), 'r') as f:
            # decode tuple key
            ori_prompt_lang_cwe2parsable_impl = {eval(k): v for k, v in json.load(f).items()}
        with open(os.path.join(cache_dir, 'ori_prompt_lang_cwe2impl.json'), 'r') as f:
            # decode tuple key
            ori_prompt_lang_cwe2impl = {eval(k): v for k, v in json.load(f).items()}
    else:
        os.makedirs(cache_dir, exist_ok=True)
        
        ori_prompt2impl = {}
        ori_prompt_lang_cwe2impl = {}
        for entry in tqdm(open(dnorm_json_path, 'r'), desc="Loading original inference results"):
            entry = json.loads(entry)
            prompt = entry['prompt']
            if prompt not in ori_prompt2impl:
                ori_prompt2impl[prompt] = []
            ori_prompt2impl[prompt].append(entry)
            if (prompt, entry['lang'], entry['cwe']) not in ori_prompt_lang_cwe2impl:
                ori_prompt_lang_cwe2impl[(prompt, entry['lang'], entry['cwe'])] = []
            ori_prompt_lang_cwe2impl[(prompt, entry['lang'], entry['cwe'])].append(entry)
            
        ori_prompt2parsable_impl = {}
        ori_prompt_lang_cwe2parsable_impl = {}
        empty_block_count = 0
        for p_l_c, impls in tqdm(ori_prompt_lang_cwe2impl.items(), desc="Filtering out unparsable implementations"):
            
            if not impls:
                continue
                
            prompt, lang, cwe = p_l_c
            
            parsable_impls = []
            for impl in impls:
                parser = lang_parser_map[impl['lang']]
                for code_block in impl['code_blocks']:
                    parsable_impls.append(code_block)   # debug

            if len(parsable_impls) > 0:
                if prompt not in ori_prompt2parsable_impl:
                    ori_prompt2parsable_impl[prompt] = {
                        'cwe': [],
                        'lang': [],
                        'impls': []  # NOTE: all parsable implementations
                    }
                ori_prompt2parsable_impl[prompt]['cwe'].append(cwe)
                ori_prompt2parsable_impl[prompt]['lang'].append(lang)
                ori_prompt2parsable_impl[prompt]['impls'].extend(parsable_impls)
            
                ori_prompt_lang_cwe2parsable_impl[p_l_c] = {
                    'impls': parsable_impls
                }
        
        print(f'empty block count: {empty_block_count}')
        with open(os.path.join(cache_dir, 'ori_prompt2impl.json'), 'w') as f:
            json.dump(ori_prompt2impl, f, indent=2)
        with open(os.path.join(cache_dir, 'ori_prompt2parsable_impl.json'), 'w') as f:
            json.dump(ori_prompt2parsable_impl, f, indent=2)
        with open(os.path.join(cache_dir, 'ori_prompt_lang_cwe2parsable_impl.json'), 'w') as f:
            # encode tuple key
            json.dump({str(k): v for k, v in ori_prompt_lang_cwe2parsable_impl.items()},
                f, indent=2)
        with open(os.path.join(cache_dir, 'ori_prompt_lang_cwe2impl.json'), 'w') as f:
            # encode tuple key
            json.dump({str(k): v for k, v in ori_prompt_lang_cwe2impl.items()}, 
                f, indent=2)
    
    # load inst
    inst_ds = load_dataset(inst_dataset_name_or_path)['train']
    if language:
        inst_ds = inst_ds.filter(lambda x: x['lang'] == language)
    print(f'language: {language}, inst_ds subset size: {len(inst_ds)}')
    
    # pair instructions (benign and malicious) with bimap
    ori_prompt2inducing_prompt = {}
    inducing_prompt2ori_prompt = {}
    for entry in tqdm(inst_ds):
        ori_prompt = entry['original_prompt']
        inducing_prompt = entry['task']
        ori_prompt2inducing_prompt[ori_prompt] = inducing_prompt
        inducing_prompt2ori_prompt[inducing_prompt] = ori_prompt
    print('o2i size: {}, i2o size: {}'.format(len(ori_prompt2inducing_prompt), len(inducing_prompt2ori_prompt)))
    
    # load fix_pair_ds    
    fix_pair_ds = load_dataset(dsec_name_or_path)['train']
    if language:
        fix_pair_ds = fix_pair_ds.filter(lambda x: x['lang'] == language)
    fix_pair_ds = fix_pair_ds.add_column('benign', [False] * len(fix_pair_ds))
    print(f'language: {language}, fix_pair_ds subset size: {len(fix_pair_ds)}')
    
    if lang_cwe_distinct:
        additional_entries = []
        for entry in tqdm(fix_pair_ds, desc="Generating additional entries"):
            inducing_prompt = entry['original_instruction']
            if inducing_prompt not in inducing_prompt2ori_prompt:
                continue
            ori_prompt = inducing_prompt2ori_prompt[inducing_prompt]
            if (ori_prompt, entry['lang'], entry['cwe']) not in \
                ori_prompt_lang_cwe2parsable_impl:
                continue
            parsable_impls = ori_prompt_lang_cwe2parsable_impl[(ori_prompt, entry['lang'], entry['cwe'])]['impls']
            for benign_code in parsable_impls:
                additional_entries.append({
                    'lang': entry['lang'],
                    'cwe': entry['cwe'],
                    'original_instruction': ori_prompt,
                    'original_code': entry['fixed_code'],
                    'fixed_code': benign_code,
                    'benign': True
                })
        
        print('additional entries size:', len(additional_entries))
        torch.random.manual_seed(42)
        sampled_indices = torch.randperm(len(additional_entries))[:int(0.35 * len(additional_entries))]
        additional_entries = [additional_entries[i] for i in sampled_indices]
        print('sampled additional entries size:', len(additional_entries))
        
    else:
        additional_entries = []
        matched = 0
        for entry in tqdm(fix_pair_ds, desc="Generating additional entries"):
            inducing_prompt = entry['original_instruction']
            if inducing_prompt not in inducing_prompt2ori_prompt:
                continue
            ori_prompt = inducing_prompt2ori_prompt[inducing_prompt]
            if ori_prompt not in ori_prompt2parsable_impl:
                continue
            parsable_impls = ori_prompt2parsable_impl[ori_prompt]['impls']
            lang_parsable_impls = ori_prompt2parsable_impl[ori_prompt]['lang']
            cwe_parsable_impls = ori_prompt2parsable_impl[ori_prompt]['cwe']
            
            torch.random.manual_seed(42)
            sampled_indices = torch.randperm(len(parsable_impls))[:int(0.3 * len(parsable_impls))]
            parsable_impls = [parsable_impls[i] for i in sampled_indices]
            lang_parsable_impls = [lang_parsable_impls[i] for i in sampled_indices]
            cwe_parsable_impls = [cwe_parsable_impls[i] for i in sampled_indices]
            
            for benign_code, l, c in zip(
                parsable_impls, lang_parsable_impls, cwe_parsable_impls
            ):
                if language and l != language:
                    continue
                if entry['lang'] == l and entry['cwe'] == c:
                    matched += 1
                else:
                    continue
                additional_entries.append({
                    'lang': json.dumps({'l': entry['lang'], 'w': l}),
                    'cwe': json.dumps({'l': entry['cwe'], 'w': c}),
                    'original_instruction': ori_prompt,
                    'original_code': entry['fixed_code'],
                    'fixed_code': benign_code,
                    'benign': True
                })
        print('matched:', matched)
            
    new_ds = Dataset.from_list(additional_entries)
    all_ds = concatenate_datasets([fix_pair_ds, new_ds])
    
    return all_ds


def load_selection_dataset(
    dataset_path,
    cached=True,
):
    if cached:
        return Dataset.load_from_disk(dataset_path)
    else:
        # return prepare_selection_dataset(cached=False)
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inst_dataset_name_or_path', type=str, default='PurCL/secalign-all-haiku-inst-clustered2k')
    parser.add_argument('--dsec_name_or_path', type=str, default='prosecalign/secalign-haiku-new-fix-new-dsec-phi3m-all-pairs')
    parser.add_argument('--dnorm_json_path', type=str, default='<your-path>/infer-ret-ori-task-haiku.phi3m-inst-filtered.jsonl')
    parser.add_argument('--language', type=str, default=None)
    parser.add_argument('--cached', action='store_true')
    parser.add_argument('--cache_dir', type=str, default='.cache/phi3m')
    parser.add_argument('--lang_cwe_distinct', action='store_true', default=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    dataset = prepare_selection_dataset_new(
        inst_dataset_name_or_path=args.inst_dataset_name_or_path,
        dsec_name_or_path=args.dsec_name_or_path,
        dnorm_json_path=args.dnorm_json_path,
        language=args.language,
        cached=args.cached,
        cache_dir=args.cache_dir,
        lang_cwe_distinct=args.lang_cwe_distinct
    )
    
    dataset.save_to_disk(args.output_dir)
    print(f"Dataset saved to {args.output_dir}")
    print(dataset)

if __name__ == '__main__':
    main()
