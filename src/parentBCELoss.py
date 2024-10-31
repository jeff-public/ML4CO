import torch
import torch.nn as nn

def parentBCELoss(y_pred, prob_true, batch_size):
    """
    Custom BCEWithLogitsLoss with an overall weight multiplier.
    
    Args:
    - y_pred: Raw logits output from the model.
    - y_true: True binary labels (0 or 1).
    - weight: Scalar weight to apply to the overall BCEWithLogitsLoss.
    
    Returns:
    - loss: Weighted BCEWithLogits loss.
    """
    # Initialize BCEWithLogitsLoss (no pos_weight, just regular BCE with logits)
    criterion = nn.BCELoss()
    
    prob_pred = y_pred.reshape((-1, batch_size)).prod(dim = 1)
    
    # Apply overall weight
    weighted_loss = criterion(prob_pred, prob_true)
    
    return weighted_loss
