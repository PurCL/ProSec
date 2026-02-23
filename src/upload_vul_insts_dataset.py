import datasets
import json
import re
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--ds_out', type=str, default="")
parser.add_argument('--ds_in', type=str, default="", help="jsonl file output by gen_vul_insts_from_*.py")

args = parser.parse_args()

ds_in = open(args.ds_in, 'r')


data = [json.loads(line) for line in ds_in.readlines()]


json_str_pattern = re.compile(r'```json\n(.*?)\n```', re.DOTALL)

def manually_parse(json_str):
    json_str = json_str.strip()
    # remove the first and last curly braces
    json_str_content = json_str[1:-1]
    json_str_content = json_str_content.strip()
    if not json_str_content.startswith('"task": "'):
        return None
    json_str_content = json_str_content[len('"task": "'):].strip()
    # find the first occurrence of '"', which is the end of the task
    task_end = json_str_content.find('",')
    task = json_str_content[:task_end]
    json_str_content = json_str_content[task_end+2:].strip()
    # if not json_str_content.startswith('"implementation": "'):
    #     return None
    # json_str_content = json_str_content[len('"implementation": '):].strip()
    # impl_str = json_str_content[1:-1].strip()
    impl_str = ''
    return {
        "task": task,
        "implementation": impl_str
    }

        



ret_entries = []
for entry in data:
    # original_prompt = entry['prompt']['content']
    original_prompt = entry['prompt']
    if 'pre_filter' in entry:
        if len([e for e in entry['pre_filter'] if 'no' in e]) > 1:
            continue
    found = False
    for response in entry['responses']:
        try:
            # json_str = json_str_pattern.search(response).group(1)
            json_str = json_str_pattern.findall(response)[-1]
            try:
                json_obj = json.loads(json_str)
                if "implementation" not in json_obj:
                    json_obj['implementation'] = ''
            except:
                json_obj = manually_parse(json_str)                
            if type(json_obj['task']) != str or type(json_obj['implementation']) != str:
                continue
            task = json_obj['task']            
            reference_impl = json_obj['implementation']
            if ('validation' in task.lower() or 'sanitation' in task.lower()
                 or 'validate' in task.lower() or 'sanitize' in task.lower()):
                continue
            ret_entries.append({
                "task": task,
                "reference_impl": reference_impl,
                "original_prompt": original_prompt
            })
            found = True
            # break
        except:
            pass
    
print("Out of %d entries, %d are valid" % (len(data), len(ret_entries)))
ds = datasets.Dataset.from_list(ret_entries)

ds.push_to_hub(args.ds_out, private=True)




print()