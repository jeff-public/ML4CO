import torch
import torch.nn as nn

def weightedBCEWithLogitsLoss(y_pred, y_true, weight):
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
    criterion = nn.BCEWithLogitsLoss()
    
    # Calculate unweighted loss
    unweighted_loss = criterion(y_pred, y_true)
    
    # Apply overall weight
    weighted_loss = weight * unweighted_loss
    
    return weighted_loss
