import argparse
from datasets import (
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    Dataset, DatasetDict
)
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy.stats as stats
import seaborn as sns
from tqdm import tqdm


def td_sample_plus(
    selection_dataset_path = 'selection_ds',
    d_all_dir = 'outputs',
    checkpoint_ids = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    length_norm=True,
    corr_method='spearman',
    corr_objects=['onof', 'ofif'],
    top_n=2,
    downsample_ratio=0.8,
    prefix='full'
):
    """
    Sample data based on training dynamics, especially margin correlation
    """
    
    def load_ckpt_data_from(
        ckpt_dir,
        checkpoint_ids = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    ):
        ckpt_data = []
        for checkpoint_id in checkpoint_ids:
            checkpoint_data = {}
            for rank in [0, 1, 2, 3, 4, 5, 6, 7]:
                with open(f'{ckpt_dir}/checkpoint-{checkpoint_id}_{rank}.json', 'r') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if k not in checkpoint_data:
                            checkpoint_data[k] = []
                        checkpoint_data[k].extend(v)
            ckpt_data.append(checkpoint_data)
        
        return ckpt_data

    def reshape_ckpt_data(
        ckpt_data,
        ln=True
    ):
        if ln:
            chosen_xy_key = 'chosen_ln_logps'
            rejected_xy_key = 'rejected_ln_logps'
            chosen_hue_key = 'chosen_acc'
            rejected_hue_key = 'rejected_acc'
        else:
            chosen_xy_key = 'chosen_logps'
            rejected_xy_key = 'rejected_logps'
            chosen_hue_key = 'chosen_acc'
            rejected_hue_key = 'rejected_acc'
        
        L_chosen = []
        L_rejected = []
        P_chosen = []
        P_rejected = []
        B = []
        for checkpoint_data in ckpt_data:
            L_chosen.append(checkpoint_data[chosen_xy_key])
            L_rejected.append(checkpoint_data[rejected_xy_key])
            P_chosen.append(checkpoint_data[chosen_hue_key])
            P_rejected.append(checkpoint_data[rejected_hue_key])
            B.append(checkpoint_data['benign'])
        
        L_chosen = np.array(L_chosen).transpose()
        L_rejected = np.array(L_rejected).transpose()
        P_chosen = np.array(P_chosen).transpose()
        P_rejected = np.array(P_rejected).transpose()
        B = np.array(B).transpose()
        
        return L_chosen, L_rejected

    
    ckpt_data_all = load_ckpt_data_from(d_all_dir, checkpoint_ids=checkpoint_ids)
    ds = load_from_disk(selection_dataset_path)

    # build (instruction-response, id) mapping
    res2ifv_idx = {}
    ifv_ids = []
    ifv_id2local = {}
    for idx, entry in tqdm(enumerate(ds), total=len(ds), desc='Building res2ifv_idx'):
        if entry['benign'] == False:
            res = entry['fixed_code']
            if res not in res2ifv_idx:
                res2ifv_idx[res] = idx
            else:
                print('duplicate response:', res)
            ifv_ids.append(idx)
            ifv_id2local[idx] = len(ifv_ids) - 1

    # map ifv idx to onf data
    ifv_idx2onf_cands = {}
    onf_idx2ifv_idx = {}
    onf_ids = []
    onf_id2local = {}
    for idx, entry in tqdm(enumerate(ds), total=len(ds), desc='Building ifv_idx2onf_cands'):
        if entry['benign'] == True:
            res = entry['original_code']    # here "original_code" is y_f
            if res not in res2ifv_idx:
                print('response not found:', idx)
                fv_idx = -1
            else:
                fv_idx = res2ifv_idx[res]
            if fv_idx not in ifv_idx2onf_cands:
                ifv_idx2onf_cands[fv_idx] = []
            ifv_idx2onf_cands[fv_idx].append(idx)
            onf_idx2ifv_idx[idx] = fv_idx
            onf_ids.append(idx)
            onf_id2local[idx] = len(onf_ids) - 1
    
    print('ifv_idx2onf_cands:', len(ifv_idx2onf_cands))
    print('onf_idx2ifv_idx:', len(onf_idx2ifv_idx))

    win_data, lose_data = reshape_ckpt_data(ckpt_data_all, ln=length_norm)
    if_data, iv_data = win_data[ifv_ids,:], lose_data[ifv_ids,:]
    on_data, of_data = win_data[onf_ids,:], lose_data[onf_ids,:]
    ifv_margin = if_data - iv_data
    onf_margin = on_data - of_data

    ## visualization for debugging    
    # from vis import pairwise_dataset_cartography
    # pairwise_dataset_cartography(
    #     x=np.std(ifv_margin, axis=1),
    #     y=np.mean(ifv_margin, axis=1),
    #     hues=np.mean(if_data, axis=1),
    #     out_filename='outputs_vis/ifv-cartography.png'
    # )
    # pairwise_dataset_cartography(
    #     x=np.std(onf_margin, axis=1),
    #     y=np.mean(onf_margin, axis=1),
    #     hues=np.mean(on_data, axis=1),
    #     out_filename='outputs_vis/onf-cartography.png'
    # )

    onf_paired_ids, ifv_paired_ids = \
        list(onf_idx2ifv_idx.keys()), list(onf_idx2ifv_idx.values())
    onof_margin = np.array([onf_margin[onf_id2local[idx]] for idx in onf_paired_ids])

    # paired if and of
    paired_of = [of_data[onf_id2local[idx]] for idx in onf_paired_ids]
    paired_if = [if_data[ifv_id2local[idx]] for idx in ifv_paired_ids]
    paired_if = np.array(paired_if)
    paired_of = np.array(paired_of)
    ofif_margin = paired_of - paired_if

    if corr_objects[1] == 'ofif':
        corr = pd.DataFrame(onof_margin).corrwith(
            pd.DataFrame(ofif_margin), axis=1, method=corr_method)
    elif corr_objects[1] == 'neg_if':
        neg_if = - paired_if
        corr = pd.DataFrame(onof_margin).corrwith(
            pd.DataFrame(neg_if), axis=1, method=corr_method)
    elif corr_objects[1] == 'neg_of_neg_if':
        neg_of_neg_if_margin = - paired_of - paired_if
        corr = pd.DataFrame(onof_margin).corrwith(
            pd.DataFrame(neg_of_neg_if_margin), axis=1, method=corr_method)
    elif corr_objects[1] == 'decrease':
        # create a dafaframe of 10 to 1 to compute corr
        decrease = np.array([list(range(10, 0, -1)) for _ in range(len(onof_margin))])
        corr = pd.DataFrame(onof_margin).corrwith(
            pd.DataFrame(decrease), axis=1, method=corr_method)
    elif corr_objects[0] == 'on' and corr_objects[1] == 'neg_if':
        paired_on = [on_data[onf_id2local[idx]] for idx in onf_paired_ids]
        neg_if = - paired_if
        corr = pd.DataFrame(paired_on).corrwith(
            pd.DataFrame(neg_if), axis=1, method=corr_method)
    elif corr_objects[0] == 'neg_of' and corr_objects[1] == 'neg_if':
        neg_of = - paired_of
        neg_if = - paired_if
        corr = pd.DataFrame(neg_of).corrwith(
            pd.DataFrame(neg_if), axis=1, method=corr_method)
    else:
        raise NotImplementedError(f'corr_objects[1] {corr_objects[1]} not implemented')
    
    # count the number of nan
    print('nan count:', corr.isna().sum())
    # fill in nan (TODO: check nan cases)
    corr.fillna(0, inplace=True)

    # select data based on corr
    # flag = 'min'
    flag = 'max'
    selected_ids = []
    selected_corr = []
    for ifv_idx, onf_cands in ifv_idx2onf_cands.items():
        cands_corr = corr[[onf_id2local[idx] for idx in onf_cands]]
        if flag == 'min':
            min_corr_idx = cands_corr.nsmallest(top_n).index
            selected_ids.extend(min_corr_idx)
            selected_corr.extend(cands_corr[min_corr_idx].tolist())
        elif flag == 'max':
            max_corr_idx = cands_corr.nlargest(top_n).index
            selected_ids.extend(max_corr_idx)
            selected_corr.extend(cands_corr[max_corr_idx].tolist())
    print('selected_ids:', len(selected_ids), selected_ids[:10])
    
    # downsample
    # if True:   # weighted random
    if False:   # weighted random
        # downsample_ratio = 0.75
        selected_ids = random.choices(
            selected_ids,
            weights=selected_corr,
            k=math.ceil(len(selected_ids) * downsample_ratio)
        )
        prefix = f'{prefix}-wds-{downsample_ratio}'
    else:   # corr
        # downsample_ratio = 0.3
        # sort by corr
        selected_ids = [x for _, x in sorted(zip(selected_corr, selected_ids), reverse=True)]
        # select the downsampled ratio
        selected_ids = selected_ids[:math.ceil(len(selected_ids) * downsample_ratio)]
        prefix = f'{prefix}-cds-{downsample_ratio}'
    
    # merge
    ifv_ds = ds.filter(lambda x: x['benign'] == False)
    onf_ds = ds.filter(lambda x: x['benign'] == True)
    onf_ds = onf_ds.select(selected_ids)
    ds = concatenate_datasets([ifv_ds, onf_ds])
    print(ds)
    ds.push_to_hub(
        f'prosecalign/{prefix}-{corr_method}-{corr_objects[0]}-{corr_objects[1]}-corr-{flag}-{top_n}',
        private=True
    )


def main():
    parser = argparse.ArgumentParser(description="Sample data based on training dynamics, especially margin correlation.")
    parser.add_argument('--selection_dataset_path', type=str, default='selection_ds', help='Path to the selection dataset.')
    parser.add_argument('--d_all_dir', type=str, default='outputs', help='Directory for all data.')
    parser.add_argument('--checkpoint_ids', type=int, nargs='+', default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], help='List of checkpoint IDs.')
    parser.add_argument('--length_norm', action='store_true', help='Enable length normalization.')
    parser.add_argument('--no-length_norm', action='store_false', dest='length_norm', help='Disable length normalization.')
    parser.set_defaults(length_norm=True)
    parser.add_argument('--corr_method', type=str, default='spearman', help='Correlation method.')
    parser.add_argument('--corr_objects', type=str, nargs=2, default=['onof', 'ofif'], help='Correlation objects.')
    parser.add_argument('--top_n', type=int, default=2, help='Top N samples to select.')
    parser.add_argument('--downsample_ratio', type=float, default=0.8, help='Downsample ratio.')
    parser.add_argument('--prefix', type=str, default='full', help='Prefix for the output dataset name.')

    args = parser.parse_args()

    td_sample_plus(
        selection_dataset_path=args.selection_dataset_path,
        d_all_dir=args.d_all_dir,
        checkpoint_ids=args.checkpoint_ids,
        length_norm=args.length_norm,
        corr_method=args.corr_method,
        corr_objects=args.corr_objects,
        top_n=args.top_n,
        downsample_ratio=args.downsample_ratio,
        prefix=args.prefix
    )

if __name__ == '__main__':
    main()