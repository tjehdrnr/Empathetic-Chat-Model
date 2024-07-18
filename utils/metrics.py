import numpy as np
from typing import Dict

def complute_metrics(eval_pred) -> Dict:
    preds, labels = eval_pred

    for pred, label in zip(preds, labels):
        print(pred.argmax(-1))
        print(label)
    
    return {"none": None}