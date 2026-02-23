import json
import torch
from tqdm import tqdm


stats_functions = {
    
}


def collect_probs(model, data):
    ...
    

    
def __get_probs(model, input_ids, labels):
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, 
            labels=labels
        )
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def __get_logps(model, input_ids, labels):
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, 
            labels=labels
        )
        logits = outputs.logits
        logps = torch.nn.functional.log_softmax(logits, dim=-1)
    return logps


def get_sequence_logps(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor
):
    """
    Get log-probabilities of each token in a sequence.
    """
    token_logps = __get_logps(model, input_ids, labels)
    batch_indices = torch.arange(labels.shape[0]).unsqueeze(1)  # Shape: [batch_size, 1]
    seq_indices = torch.arange(labels.shape[1]).unsqueeze(0)     # Shape: [1, seq_length]
    label_logps = token_logps[batch_indices, seq_indices, labels]

    # compute length-normalized log-probabilities
    mask = labels != -100
    masked_logps = label_logps * mask
    seq_logps = masked_logps.sum(dim=1)
    seq_len = mask.sum(dim=1)
    ln_seq_logps = seq_logps / seq_len
    
    # compute accuracy
    correct = (labels == torch.argmax(token_logps, dim=-1))
    masked_correct = correct * mask
    accuracy = masked_correct.sum(dim=1) / mask.sum(dim=1)
    
    return seq_logps, ln_seq_logps, seq_len, accuracy
    

def example_function(
    rank,
    checkpoint_id,
    processed_ds,
    batch_size,
    data_collator,
    model,
    device,
):
        # iterate over dataset
    stats = {
        'chosen_probs': [],
        'rejected_probs': [],
        'mean_chosen_probs': [],
        'mean_rejected_probs': [],
        'std_chosen_probs': [],
        'std_rejected_probs': [],
        'benign': []
    }
    
    dataloader = torch.utils.data.DataLoader(
        processed_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True
    )
    
    # for i, batch in enumerate(dataloader):
    for i, batch in enumerate(tqdm(dataloader, desc=f'Processing shard {rank}')):
        
        # if i > 500: # debug
        if i > 50: # debug
            break
        
        MAX_LENGTH = 2048   # NOTE: might by problematic
        chosen_input_ids = batch['chosen_input_ids'][:,:MAX_LENGTH].to(device)
        chosen_labels = batch['chosen_labels'][:,:MAX_LENGTH].to(device)
        rejected_input_ids = batch['rejected_input_ids'][:,:MAX_LENGTH].to(device)
        rejected_labels = batch['rejected_labels'][:,:MAX_LENGTH].to(device)
        
        with torch.no_grad():
        
            # compute probs
            chosen_probs = __get_probs(model, chosen_input_ids, chosen_labels)
            rejected_probs = __get_probs(model, rejected_input_ids, rejected_labels)
            
            # mask out -100
            chosen_mask = chosen_labels != -100
            chosen_batch_indices = torch.arange(chosen_labels.shape[0]).unsqueeze(1)  # Shape: [batch_size, 1]
            chosen_seq_indices = torch.arange(chosen_labels.shape[1]).unsqueeze(0)     # Shape: [1, seq_length]
            
            rejected_mask = rejected_labels != -100
            rejected_batch_indices = torch.arange(rejected_labels.shape[0]).unsqueeze(1)
            rejected_seq_indices = torch.arange(rejected_labels.shape[1]).unsqueeze(0)
            
            # label probs
            chosen_probs = chosen_probs[chosen_batch_indices, chosen_seq_indices, chosen_labels]
            rejected_probs = rejected_probs[rejected_batch_indices, rejected_seq_indices, rejected_labels]

            # # compute stats
            masked_chosen_probs = chosen_probs * chosen_mask
            mean_chosen_probs = masked_chosen_probs.sum(dim=1) / chosen_mask.sum(dim=1)
            masked_rejected_probs = rejected_probs * rejected_mask
            mean_rejected_probs = masked_rejected_probs.sum(dim=1) / rejected_mask.sum(dim=1)
            masked_chosen_square_diff = (chosen_probs - mean_chosen_probs.unsqueeze(1)) ** 2 * chosen_mask
            std_chosen_probs = torch.sqrt(masked_chosen_square_diff).sum(dim=1) / chosen_mask.sum(dim=1)
            masked_rejected_square_diff = (rejected_probs - mean_rejected_probs.unsqueeze(1)) ** 2 * rejected_mask
            std_rejected_probs = torch.sqrt(masked_rejected_square_diff).sum(dim=1) / rejected_mask.sum(dim=1)
            
            stats['mean_chosen_probs'].extend(mean_chosen_probs.cpu().numpy().tolist())
            stats['mean_rejected_probs'].extend(mean_rejected_probs.cpu().numpy().tolist())
            stats['std_chosen_probs'].extend(std_chosen_probs.cpu().numpy().tolist())
            stats['std_rejected_probs'].extend(std_rejected_probs.cpu().numpy().tolist())
            stats['benign'].extend(batch['benign'].cpu().numpy().tolist())
    
    # save stats
    with open(f'outputs/checkpoint-{checkpoint_id}_{rank}.pkl', 'w') as f:
        json.dump(stats, f, indent=2)