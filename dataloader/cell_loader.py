import torch

def collate_contrastive(batch):
    """
    Custom collate function for contrastive learning.
    
    Args:
        batch: List of tuples containing (inputs, targets)
        
    Returns:
        tuple: (inputs, targets)
    """
    inputs = torch.cat([i[0] for i in batch])
    targets = torch.cat([i[1] for i in batch])
    if len(batch[0]) == 3:
        targets2 = torch.cat([i[2] for i in batch])
        targets = [targets, targets2]
    return inputs, targets 