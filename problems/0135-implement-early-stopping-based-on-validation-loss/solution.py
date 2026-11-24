from typing import Tuple
import numpy as np

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    # Your code here
    val_losses = np.asarray(val_losses, dtype=float)
    curr_patience = patience
    prev_loss = val_losses[0]
    stop_epoch = None
    best_epoch = 0
    for i in range(1,len(val_losses)):
        improvement = float(prev_loss-val_losses[i])
        if curr_patience <=0:
            stop_epoch = i-1
            break
        elif improvement <= min_delta:
            curr_patience -=1
        else:
            curr_patience = patience

        if improvement>=min_delta:
            prev_loss = val_losses[i]
            best_epoch = i

    if stop_epoch is None:
        return (len(val_losses)-1,best_epoch) 
    else:
        return (stop_epoch, best_epoch) 