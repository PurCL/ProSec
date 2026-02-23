import argparse
import os
import json
from tqdm import tqdm
import openai
import datasets
import multiprocessing
import claude_utils
import pickle

parser = argparse.ArgumentParser(description="")
parser.add_argument("--ds_path", type=str, default="BAAI/Infinity-Instruct")
parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
parser.add_argument(
    "--sys-prompt-in", type=str, default="sys-prompt-cwe79-eliciter.txt"
)
parser.add_argument("--fout", type=str, default="tmp.jsonl")
parser.add_argument("--from-idx", type=int, default=0)
parser.add_argument("--to-idx", type=int, default=10000)
parser.add_argument("--nproc", type=int, default=4)
parser.add_argument("--lang", type=str, default="java")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

data_cache_fname = "data_cache-%s.pkl" % args.lang.strip()
if os.path.exists(data_cache_fname):
    with open(data_cache_fname, "rb") as f:
        data_cache = pickle.load(f)
else:
    data_cache = {}

api_key = open("my-api-key.txt", "r").read().strip()

ds = datasets.load_dataset(args.ds_path, "7M_domains")


def filter_by_lang(data, lang):
    conv = data["conversations"]
    if len(conv) == 0:
        return False
    if conv[0]["from"] != "human":
        return False
    if "```" in conv[0]["value"]:
        return False
    sample_data = conv[0]["value"]
    if lang in sample_data.lower():
        return True
    if lang == "cpp":
        if "c++" in sample_data.lower():
            return True
    return False


ds_filtered = ds.filter(
    lambda x: x["langdetect"] == "en"
    and "code" in x["source"].lower()
    and filter_by_lang(x, args.lang),
    num_proc=8,
)


to_idx = min(args.to_idx, len(ds_filtered["train"]))
print("There are total {} samples.".format(len(ds_filtered["train"])))
print("Selecting samples from {} to {}.".format(args.from_idx, to_idx))
ds_selected = ds_filtered.shuffle(seed=args.seed)["train"].select(
    range(args.from_idx, to_idx)
)


class ModelPrompter:
    def query(self, model_name, messages, system_prompt, temperature, max_tokens, n):
        raise NotImplementedError


class OpenAIModelPrompter(ModelPrompter):
    def __init__(self, api_key):
        self.api_key = api_key

    def query(
        self,
        rate_limiter,
        model_name,
        messages,
        system_prompt,
        temperature,
        max_tokens,
        n,
    ):
        client = openai.OpenAI(
            api_key=self.api_key,
        )
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
            ] + messages
        completions = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )
        return [c.message.content for c in completions.choices]


class ClaudeModelPrompter(ModelPrompter):
    def __init__(self):
        pass

    def query(
        self,
        rate_limiter,
        model_name,
        messages,
        system_prompt,
        temperature,
        max_tokens,
        n,
    ):
        rets = []
        for i in range(n):
            ret = claude_utils.query_claude(
                model_name=model_name,
                rate_limiter=rate_limiter,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
            ret_txt = ret["content"][0]["text"]
            rets.append(ret_txt)
        return rets


sys_prompt = open(args.sys_prompt_in, "r").read()

openai_model_prompter = OpenAIModelPrompter(api_key)
claude_model_prompter = ClaudeModelPrompter()
if "gpt" in args.model_name:
    prompter = openai_model_prompter
elif args.model_name in ["sonnet", "haiku"]:
    prompter = claude_model_prompter
else:
    raise ValueError("Unknown model name: {}".format(args.model_name))


def request_one(prompt):
    global my_rate_limiter
    global my_data_cache
    # thread_name = multiprocessing.current_process().name
    # print("Requesting from thread name: ", thread_name)
    msg = prompt["messages"]
    idx = prompt["ori_id"]
    sys_prompt = prompt["sys_prompt"]
    try:
        sample_data = msg[-1]["content"]
        question = """
Is the following question a [[LANG]] coding task? (That is, can it be answered by writing [[LANG]] code?)
<Question>
[[QUESTION]]
</Question>
Your answer: (Yes/No).
"""
        question = question.replace("[[LANG]]", args.lang)
        question = question.replace("[[QUESTION]]", sample_data)
        if question in my_data_cache:
            has_no_len = my_data_cache[question]
            if has_no_len >= 3:
                return {
                    "idx": idx,
                    "prompt": prompt["ori_inst"],
                    "choices": [],
                    "pre_filter": ["no identified by cache"],
                }
            else:
                pre_filter_question = ["yes identified by cache"]
        else:
            pre_filter_question = prompter.query(
                rate_limiter=my_rate_limiter,
                model_name=args.model_name,
                messages=[
                    # {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question},
                ],
                system_prompt=None,
                temperature=0.8,
                max_tokens=10,
                n=5,
            )
            has_no = [c for c in pre_filter_question if "no" in c.lower()]
            my_data_cache[question] = len(has_no)
            if len(has_no) >= 3:
                return {
                    "idx": idx,
                    "prompt": prompt["ori_inst"],
                    "choices": [],
                    "pre_filter": pre_filter_question,
                }

        completions = prompter.query(
            rate_limiter=my_rate_limiter,
            model_name=args.model_name,
            messages=msg,
            system_prompt=sys_prompt,
            temperature=0.8,
            max_tokens=512,
            n=5,
        )

    except Exception as e:
        print(e)
        return None
    return {
        "idx": idx,
        "prompt": prompt["ori_inst"],
        "choices": completions,
        "pre_filter": pre_filter_question,
    }


messages_list = []
for i, data in enumerate(ds_selected):
    conv = data["conversations"]
    if len(conv) == 0:
        continue
    if conv[0]["from"] != "human":
        continue
    sample_data = conv[0]["value"]
    # prompt = sys_prompt + "\n" + sample_data
    current_message = [
        {"role": "user", "content": sample_data},
        # {"role": "user", "content": prompt},
    ]
    messages_list.append(
        {
            "sys_prompt": sys_prompt,
            "messages": current_message,
            "ori_id": data["id"],
            "ori_inst": sample_data,
        }
    )

fout = open(args.fout, "w")

worker_id = multiprocessing.Value("i", 0)


def init_rate_limiter(share_data_cache,shared_list, shared_lock):
    global my_worker_id
    global my_rate_limiter
    global my_data_cache
    with worker_id.get_lock():
        my_data_cache = share_data_cache
        print("My data cache id: ", id(my_data_cache))
        my_worker_id = worker_id.value
        worker_id.value += 1
        my_rate_limiter = claude_utils.RateLimiter(1800, 60, shared_list, shared_lock)

    print(f"Worker {my_worker_id} started")


shared_list = multiprocessing.Manager().list()
shared_lock = multiprocessing.Manager().Lock()
shared_data_cache = multiprocessing.Manager().dict()

# use data_cache to update the shared_data_cache
for k, v in data_cache.items():
    shared_data_cache[k] = v

# completions_all = [request_one(msg) for msg in tqdm(messages_list, desc="Requesting", position=0)]
pool = multiprocessing.Pool(args.nproc, init_rate_limiter, (shared_data_cache, shared_list, shared_lock))
completions_all = pool.imap_unordered(
    request_one, tqdm(messages_list, desc="Requesting", position=0)
)

code_snippets_all = []
for i, completions in enumerate(
    tqdm(completions_all, total=len(messages_list), desc="Writing", position=1)
):
    if completions is None:
        continue
    idx = completions["idx"]
    code_snippets = completions["choices"]
    prompt = completions["prompt"]
    code_snippets_all.append(code_snippets)
    pre_filter_ans = completions["pre_filter"]
    fout.write(
        json.dumps(
            {"prompt": prompt, "responses": code_snippets, "pre_filter": pre_filter_ans}
        )
        + "\n"
    )
    # fout.write('\n')
    fout.flush()
    # write cache
    if i % 100 == 0:
        with open(data_cache_fname, "wb") as f:
            print("Data cache length: ", len(shared_data_cache))
            data_cache = {}
            for k, v in shared_data_cache.items():
                data_cache[k] = v
            pickle.dump(data_cache, f)
fout.close()

print()
