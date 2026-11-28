import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    # Your code here
    y_scores = np.asarray(y_scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    pstv = y_scores[y_true==1]
    ngtv = y_scores[y_true==0]

    more=0
    equal=0
    less=0
    for p in pstv:
        for n in ngtv:
            if p>n:
                more+=1
            elif p==n:
                equal+=1
            else:
                less+=1
    return (more+0.5*equal)/(more+equal+less)
