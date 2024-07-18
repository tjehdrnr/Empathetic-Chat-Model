import numpy as np
from typing import Dict

def complute_metrics(eval_pred) -> Dict:
    preds, labels = eval_pred

    for label in labels:
        print(label)