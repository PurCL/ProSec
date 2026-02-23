import datasets
import argparse
import json
import multiprocessing
import openai
from tqdm import tqdm
import os
import utils
from transformers import AutoTokenizer
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompts_in",
    type=str,
    required=True,
    help="vul-inducing instructions dataset",
)
parser.add_argument("--clients_config", type=str, default="config/clients-config.yaml")
# parser.add_argument(
#     "--model_name_or_path", type=str, default="microsoft/Phi-3-mini-4k-instruct"
# )
parser.add_argument(
    "--model_id", type=str, default="phi3m"
)
# parser.add_argument("--model_name_or_path", type=str, default="meta-llama/CodeLlama-7b-Instruct-hf")
parser.add_argument("--n", type=int, default=20)
# parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--t", type=float, default=0.8)
parser.add_argument(
    "--fout",
    type=str,
    default="",
)
args = parser.parse_args()

if args.fout == "":
    args.fout = args.prompts_in.replace(".fix-prompt.jsonl", ".fixed.jsonl")

client_config = yaml.load(open(args.clients_config), Loader=yaml.SafeLoader)

clients = []
my_config = client_config[args.model_id]
model_name = my_config["model_name"]
for api_info in my_config["apis"]:
    addr = api_info["addr"]
    api_key = api_info["api_key"]
    clients.append(
        (
            openai.OpenAI(
                base_url=addr,
                api_key=api_key,
            ),
            model_name,
        )
    )

clients = clients * 2

tokenizer = AutoTokenizer.from_pretrained(model_name)


def request_one(prompt):
    global my_worker_id
    # idx = prompt['idx']
    idx = my_worker_id

    client, model_name = clients[idx % len(clients)]

    sys_prompt = prompt["fix_system_prompt"]
    user_prompt = prompt["fix_user_prompt"]
    if 'gemma' in model_name:
        messages = [
            {"role": "user", "content": sys_prompt + '\n\n' + user_prompt},
        ]
    else:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
    sys_prompt_tokens = tokenizer(sys_prompt, return_tensors="pt")["input_ids"]
    user_prompt_tokens = tokenizer(user_prompt, return_tensors="pt")["input_ids"]
    prompt_len_total = sys_prompt_tokens.size(1) + user_prompt_tokens.size(1)
    if prompt_len_total > 4096 - 2048 - 10:
        too_long = True
    else:
        too_long = False
    if too_long:
        print(f"Prompt too long for worker {my_worker_id}")
        return {
                "cwe": prompt["cwe"],
                "lang": prompt["lang"],
                "prompt": prompt["prompt"],
                "vul_code": prompt["code"],
                "code_blocks": ["too long"],
            }
    try:
        rsp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=2048,
            temperature=args.t,
            n=args.n,
        )
    except Exception as e:
        if too_long:
            print(f"Prompt too long for worker {my_worker_id}")
            return {
                "cwe": prompt["cwe"],
                "lang": prompt["lang"],
                "prompt": prompt["prompt"],
                "vul_code": prompt["code"],
                "code_blocks": ["too long"],
            }
        else:
            print("Error in worker %d" % my_worker_id)
            print("Bad prompt:")
            print(messages)
            print("Error message:")
            print(e)
            raise e

    rsp_strs = [c.message.content for c in rsp.choices]
    code_blocks = [
        utils.parse_code_block(rsp_str, lang="", only_capture_succ=True)
        for rsp_str in rsp_strs
    ]
    valid_code_blocks = [cb.strip() for cb in code_blocks if cb.strip() != ""]
    ret_entry = {
        "cwe": prompt["cwe"],
        "lang": prompt["lang"],
        "prompt": prompt["prompt"],
        "vul_code": prompt["code"],
        "code_blocks": valid_code_blocks,
    }
    return ret_entry


prompts = [json.loads(l) for l in tqdm(open(args.prompts_in), desc="loading prompts")]

query_prompts = []
lang_cwe_cnt = {}
uniq_prompts = {}
for prompt in prompts:
    lang = prompt["lang"]
    cwe = prompt["cwe"]
    prompt_str = prompt["prompt"]
    if prompt_str not in uniq_prompts:
        uniq_prompts[prompt_str] = 0
    uniq_prompts[prompt_str] += 1
    has_exp = [d for d in prompt["uniq_detection_results"] if d["cwe_id"] == cwe]
    if len(has_exp) == 0 and uniq_prompts[prompt_str] > 3:
        continue
    if uniq_prompts[prompt_str] > 6:
        continue
    if (lang, cwe) not in lang_cwe_cnt:
        lang_cwe_cnt[(lang, cwe)] = 0
    lang_cwe_cnt[(lang, cwe)] += 1
    # if lang_cwe_cnt[(lang, cwe)] >= 1000:
    #     continue
    query_prompts.append(prompt)


# **append** to fout
fout = open(args.fout, "a+")
fout.seek(0)
# count existing lines
lines = fout.readlines()
line_no = len(lines)
# move to end
fout.seek(0, 2)

query_prompts = query_prompts[line_no:]
print(f"Starting from line {line_no}")

worker_id = 0


def worker_init(worker_id):
    global my_worker_id
    with worker_id.get_lock():
        my_worker_id = worker_id.value
        worker_id.value += 1
    print(f"Worker {my_worker_id} started")


ctx = multiprocessing.get_context("spawn")
worker_id = ctx.Value("i", 0)
pool = multiprocessing.Pool(len(clients), worker_init, (worker_id,))
completions_all = pool.imap(
    request_one, tqdm(query_prompts, desc="Requesting", position=0)
)


code_snippets_all = []
for i, ret_entry in enumerate(
    tqdm(completions_all, total=len(query_prompts), desc="Writing", position=1)
):
    fout.write(json.dumps(ret_entry) + "\n")
    fout.flush()

fout.close()


print()

# # dbg
# fin = open(args.fout, "r").readlines()
# data = [json.loads(l) for l in fin]

# new_ret_data = []
# for data_entry, query in zip(data, query_prompts):
#     prompt_query = query['prompt']
#     prompt_ret = data_entry['prompt']
#     assert prompt_query == prompt_ret
#     new_ret_data.append({
#         'vul_code': query['code'],
#         **data_entry
#     })

# fout = open(args.fout, "w")
# for entry in new_ret_data:
#     fout.write(json.dumps(entry) + "\n")
# fout.close()
