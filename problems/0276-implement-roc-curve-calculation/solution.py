import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    # Your code here
    y_scores = np.asarray(y_scores,dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    thresholds = np.r_[np.unique(y_scores)[::-1],-np.inf]

    tprs = []
    fprs = []
    for threshold in thresholds:
        mask = y_scores>threshold
        y_pred = np.where(mask,1,0)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(y_pred)):
            if y_pred[i]==y_true[i]:
                if y_true[i]==0:
                    tn+=1
                else:
                    tp+=1
            else:
                if y_pred[i]==0:
                    fn+=1
                else:
                    fp+=1
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        tprs.append(tpr)
        fprs.append(fpr)
    return fprs,tprs

