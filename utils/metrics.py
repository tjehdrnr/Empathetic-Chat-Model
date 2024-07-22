import torch
from typing import Dict
import numpy as np
import torch.nn.functional as F


def compute_metrics(eval_pred) -> Dict:
    logits, labels = eval_pred

    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    log_probs = F.log_softmax(logits, dim=-1)

    mask = labels != -100
    word_count = mask.sum().item()

    loss = F.nll_loss(
        log_probs.view(-1, log_probs.size(-1)),
        labels.view(-1),
        reduction="none"
    )

    loss = float(loss[mask.view(-1)].sum()) / word_count

    return {
        "ppl": np.exp(loss),
    }
    
    
    
    
    
    
# if __name__ == "__main__":
#     import torch
#     import numpy as np

#     # 랜덤 시드 설정
#     np.random.seed(42)
#     torch.manual_seed(42)

#     # 설정
#     batch_size = 4
#     vocab_size = 1000
#     max_seq_length = 50
#     min_seq_length = 10

#     # logits 생성 (배치 크기, 최대 시퀀스 길이, 어휘 크기)
#     logits = torch.randn(batch_size, max_seq_length, vocab_size)
#     # print(logits)

#     # labels 생성
#     labels = torch.full((batch_size, max_seq_length), fill_value=-100, dtype=torch.long)
#     # print(labels)

#     for i in range(batch_size):
#         seq_length = np.random.randint(min_seq_length, max_seq_length + 1)
#         labels[i, -seq_length:] = torch.randint(0, vocab_size, (seq_length,))

#     # 테스트용 eval_preds 튜플 생성
#     eval_preds = (logits, labels)
    
#     compute_metrics(eval_preds)

    


