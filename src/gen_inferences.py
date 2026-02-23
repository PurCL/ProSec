import datasets
import argparse
import json
import multiprocessing
import openai
from tqdm import tqdm
import os
from transformers import AutoTokenizer
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name", type=str, default="", help="vul-inducing instructions dataset"
)
parser.add_argument("--clients_config", type=str, default="config/clients-config.yaml")
# parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3-mini-4k-instruct")
parser.add_argument("--model_id", type=str, default="phi3m")
parser.add_argument("--n", type=int, default=20)
# parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--t", type=float, default=0.8)
parser.add_argument("--col_name", type=str, default="task")
parser.add_argument("--fout", type=str, default="out.jsonl")
args = parser.parse_args()

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

task_col_name = args.col_name
ds = datasets.load_dataset(args.dataset_name)["train"]

messages_list = []
for i, data in enumerate(ds):
    sample_data = data[task_col_name]
    if 'gemma' in args.model_id:
        # gemma does not support system messages
        current_message = [
            {"role": "user", "content": sample_data}
        ]
    else:
        current_message = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that helps developers implement the given task. Only return the code, don't include any other information,\n    such as a preamble or suffix.\n",
            },
            {"role": "user", "content": sample_data},
        ]
    messages_list.append(
        {
            "messages": current_message,
            "idx": i,
        }
    )


def request_one(prompt):
    global my_worker_id
    msg = prompt["messages"]
    # idx = prompt['idx']
    idx = my_worker_id
    client, model_name = clients[idx % len(clients)]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_tokens = tokenizer(
        msg[-1]["content"], return_tensors="pt", padding=False, truncation=False
    )
    prompt_len = len(prompt_tokens["input_ids"][0])
    if prompt_len > 2048:
        print(f"Prompt too long: {prompt_len}")
        return {"idx": idx, "prompt": msg[-1]["content"], "choices": []}

    try:
        completions = client.chat.completions.create(
            model=model_name,
            messages=msg,
            max_tokens=1024,
            temperature=args.t,
            n=args.n,
        )
    except Exception as e:
        if prompt_len > 2048:
            print(f"Prompt too long: {prompt_len}")
            return {"idx": idx, "prompt": msg[-1]["content"], "choices": []}
        else:
            print("Error in worker %d: %s" % (idx, e))
            print("Bad prompt:" + msg[-1]["content"])
            raise e

    return {"idx": idx, "prompt": msg[-1]["content"], "choices": completions.choices}


# **append** to fout
fout = open(args.fout, "a+")
fout.seek(0)
# count existing lines
lines = fout.readlines()
line_no = len(lines)
# move to end
fout.seek(0, 2)

messages_list = messages_list[line_no:]
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
    request_one, tqdm(messages_list, desc="Requesting", position=0)
)

code_snippets_all = []
for i, completions in enumerate(
    tqdm(completions_all, total=len(messages_list), desc="Writing", position=1)
):
    code_snippets = [c.message.content for c in completions["choices"]]
    code_snippets_all.append(code_snippets)
    fout.write(
        json.dumps({"prompt": completions["prompt"], "responses": code_snippets}) + "\n"
    )
    fout.flush()
fout.close()
