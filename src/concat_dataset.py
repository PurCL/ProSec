import os
import datasets
from tqdm import tqdm

HF_USER = os.environ.get("HF_USER")
if not HF_USER:
    raise RuntimeError("Please set HF_USER environment variable (e.g., export HF_USER=your-hf-username)")

# ds_ins = [
#     '<hf_user>/secalign-java200-pair-w-benign',
#     '<hf_user>/secalign-java78-pair-w-benign',
#     '<hf_user>/secalign-java327-pair-w-benign',
#     '<hf_user>/secalign-cwe502-java-lang-aware-1k-pair-w-benign'
# ]

# ds_out = '<hf_user>/secalign-java-all-pair-w-benign'

# all_ds = [datasets.load_dataset(ds)['train'] for ds in ds_ins]

# all_ds = datasets.concatenate_datasets(all_ds)

# # shuffle
# all_ds = all_ds.shuffle(seed=42)

# all_ds.push_to_hub(ds_out, private=True)


# #####################################################################################################################

# cwe_lang = [
#     ["cwe502", "java"],
#     ["cwe78", "csharp"],
#     ["cwe676", "cpp"],
#     ["cwe119", "c"],
#     ["cwe78", "cpp"],
#     ["cwe200", "java"],
#     ["cwe377", "cpp"],
#     ["cwe338", "javascript"],
#     ["cwe338", "python"],
#     ["cwe352", "java"],
#     ["cwe89", "rust"],
#     ["cwe338", "csharp"],
#     ["cwe502", "python"],
#     ["cwe119", "javascript"],
#     ["cwe338", "cpp"],
#     ["cwe79", "java"],
#     ["cwe89", "python"],
#     ["cwe22", "php"],
#     ["cwe611", "java"],
#     ["cwe502", "csharp"],
#     ["cwe119", "cpp"],
#     ["cwe89", "csharp"],
#     ["cwe78", "java"],
#     ["cwe676", "c"],
#     ["cwe295", "java"],
#     ["cwe78", "c"],
#     ["cwe79", "javascript"],
#     ["cwe352", "csharp"],
#     ["cwe377", "c"],
#     ["cwe676", "rust"],
#     ["cwe78", "rust"],
#     ["cwe502", "php"],
#     ["cwe295", "rust"],
#     ["cwe611", "csharp"],
#     ["cwe78", "python"],
#     ["cwe200", "php"],
#     ["cwe338", "c"],
#     ["cwe22", "javascript"],
# ]

# ds_ins = [
#     ('PurCL/secalign-%s-%s-gpt4om-inst-clustered' % (cwe, lang), cwe, lang)
#     for cwe, lang in cwe_lang
# ]

#####################################################################################################################

cwe_lang = [
    ["cwe502", "java"],
    # ["cwe78", "csharp"],
    ["cwe676", "cpp"],
    ["cwe119", "c"],
    ["cwe78", "cpp"],
    ["cwe200", "java"],
    ["cwe377", "cpp"],
    ["cwe338", "javascript"],
    ["cwe338", "python"],
    ["cwe352", "java"],
    # ["cwe89", "rust"],
    # ["cwe338", "csharp"],
    ["cwe502", "python"],
    ["cwe119", "javascript"],
    ["cwe338", "cpp"],
    ["cwe79", "java"],
    ["cwe89", "python"],
    # ["cwe22", "php"],
    ["cwe611", "java"],
    # ["cwe502", "csharp"],
    ["cwe119", "cpp"],
    # ["cwe89", "csharp"],
    ["cwe78", "java"],
    ["cwe676", "c"],
    ["cwe295", "java"],
    ["cwe78", "c"],
    ["cwe79", "javascript"],
    # ["cwe352", "csharp"],
    ["cwe377", "c"],
    # ["cwe676", "rust"],
    # ["cwe78", "rust"],
    # ["cwe502", "php"],
    # ["cwe295", "rust"],
    # ["cwe611", "csharp"],
    ["cwe78", "python"],
    # ["cwe200", "php"],
    ["cwe338", "c"],
    ["cwe22", "javascript"],
]

ds_ins = [
    ('%s/secalign-%s-%s-claude-haiku-inst-clustered' % (HF_USER, cwe, lang), cwe, lang)
    for cwe, lang in cwe_lang
]


rets = []
for ds_in, cwe, lang in tqdm(ds_ins):
    try:
        ds = datasets.load_dataset(ds_in)['train']
    except:
        print(ds_in)
        continue
    for entry in ds:        
        entry['cwe'] = "CWE-%d"%(int(cwe[3:]))
        entry['lang'] = lang
        rets.append(entry)


print()

all_ds = datasets.Dataset.from_list(rets)
all_ds_shuffled = all_ds.shuffle(seed=42)
# all_ds_shuffled.push_to_hub('%s/secprompt-all-gpt4om-inst-clustered' % HF_USER, private=True)
# all_ds_shuffled.push_to_hub('%s/secalign-new-cpp-gpt4om-inst-clustered2k' % HF_USER, private=True)
# all_ds_shuffled.push_to_hub('%s/secalign-all-haiku-inst-clustered2k' % HF_USER, private=True)
all_ds_shuffled.push_to_hub('PurCL/secalign-all-haiku-inst-clustered2k', private=True)


# all_ds.push_to_hub('%s/secalign-all-gpt4om-inst-clustered-concated' % HF_USER, private=True)



print()

# ds = datasets.load_dataset('%s/secalign-all-haiku-inst-clustered2k' % HF_USER)
# ds.push_to_hub('PurCL/secalign-all-haiku-inst-clustered2k', private=True)