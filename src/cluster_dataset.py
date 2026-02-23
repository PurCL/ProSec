import datasets
import argparse
import json
import multiprocessing
import openai
from tqdm import tqdm
import os
from fuzzywuzzy import fuzz
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--original_ds_name", type=str, required=True)
parser.add_argument("--out_ds_name", type=str, required=True)
parser.add_argument("--num_clusters", type=int, default=2000)
parser.add_argument("--num_instances_per_cluster", type=int, default=1)

args = parser.parse_args()

ORIGINAL_DS = args.original_ds_name
OUT_DS = args.out_ds_name
COL_NAME = 'task'

ds = datasets.load_dataset(ORIGINAL_DS)

ds_train = ds['train']

instructions = ds_train[COL_NAME]

unique_instructions = sorted(set(instructions))

inst_embs = SentenceTransformer('BAAI/bge-m3').encode(unique_instructions)

print()

# # tsne
# from sklearn.manifold import TSNE

# tsne = TSNE(n_components=2, random_state=0)
# tsne_inst_embs = tsne.fit_transform(inst_embs)

# # visualize

# plt.close()
# plt.scatter(tsne_inst_embs[:, 0], tsne_inst_embs[:, 1])
# # for i, txt in enumerate(unique_instructions):
# #     plt.annotate(i, (tsne_inst_embs[i, 0], tsne_inst_embs[i, 1]))
# plt.savefig('tmp-tsne.png')


# clustering to 1k clusters
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(inst_embs)
labels = kmeans.labels_

instance_by_labels = {}
for i, l in enumerate(labels):
    if l not in instance_by_labels:
        instance_by_labels[l] = []
    instruction = unique_instructions[i]
    instance_by_labels[l].append(instruction)

selected_instances = []
# pick 10 instances from each cluster
for l in instance_by_labels:
    sample_num = min(args.num_instances_per_cluster, len(instance_by_labels[l]))
    sampled_instances = np.random.choice(instance_by_labels[l], sample_num, replace=False)
    selected_instances.extend([str(s) for s in sampled_instances])
    

selection_set = set(selected_instances)
seen_set = set()

selected_train_data = []
for entry in tqdm(ds_train):
    instruction = entry[COL_NAME]
    if instruction in selection_set and instruction not in seen_set:
        selected_train_data.append(entry)
        seen_set.add(instruction)

new_train_ds = datasets.Dataset.from_list(selected_train_data)
new_train_ds.shuffle(seed=42)
new_train_ds.push_to_hub(OUT_DS, private=True)


exit(0)
pairwise_fuzzy_ratio = []
for i in tqdm(range(len(unique_instructions))):
    for j in range(i+1, len(unique_instructions)):
        ratio = fuzz.ratio(unique_instructions[i], unique_instructions[j])
        pairwise_fuzzy_ratio.append((i, j, ratio))


large_ratio_pairs = [p for p in pairwise_fuzzy_ratio if p[2] > 80]
# hist of fuzzy ratio

plt.close()
plt.hist([p[2] for p in pairwise_fuzzy_ratio], bins=100)
plt.savefig('tmp-fuzzy-ratio-hist.png')

print()



# interesting_cases = []
# for entry in tqdm(ds['train']):
#     if '.*' in entry['original_code'] and '.*' not in entry['fixed_code']:
#         interesting_cases.append(entry)


# dbg_out = 'dbg-out/'
# os.system(f'rm -rf {dbg_out}')
# os.makedirs(dbg_out, exist_ok=True)
# for i, entry in enumerate(interesting_cases):
#     with open(f'{dbg_out}/case-{i}-ori.java', 'w') as f:
#         f.write(entry['original_code'])
#     with open(f'{dbg_out}/case-{i}-fix.java', 'w') as f:
#         f.write(entry['fixed_code'])