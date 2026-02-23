from transformers import (
    PreTrainedTokenizerBase,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq
)
from transformers.utils import PaddingStrategy
from typing import Union, Optional


class DataCollatorForPairWithPadding(DataCollatorForSeq2Seq):
    """
    Data collator that will dynamically pad the inputs to the maximum length in the batch, and will dynamically pad the
    labels to the maximum length of the labels in the batch.
    """

    def __call__(self, features):
        
        chosen_features = super().__call__(
            [
                {
                    'input_ids': x['chosen_input_ids'],
                    'labels': x['chosen_labels']
                } for x in features
            ]
        )
        rejected_features = super().__call__(
            [
                {
                    'input_ids': x['rejected_input_ids'],
                    'labels': x['rejected_labels']
                } for x in features
            ]
        )
        return {
            'chosen_input_ids': chosen_features['input_ids'],
            'chosen_labels': chosen_features['labels'],
            'rejected_input_ids': rejected_features['input_ids'],
            'rejected_labels': rejected_features['labels'],
            'benign': [x['benign'] for x in features]
        }


# debugging pipeline
"""
    batch_size=8
    column_names = ds.column_names
    processed_ds = ds.map(
        partial(preprocess_function_batch, tokenizer=tokenizer), 
        batched=True,
        remove_columns=column_names,
        num_proc=16
    )
    collator = DataCollatorForPairWithPadding(
        tokenizer=tokenizer, padding='longest', return_tensors='pt')
    dataloader = torch.utils.data.DataLoader(
        processed_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        # num_workers=4,
        num_workers=2,
        pin_memory=True
    )
    for i, batch in enumerate(tqdm(dataloader)):
        if not (batch['chosen_labels'][:, 0] == -100).all():
            print(batch['chosen_labels'])
            break
"""