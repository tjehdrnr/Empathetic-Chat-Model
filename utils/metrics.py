import torch
from typing import Dict
from torch.nn.functional import cross_entropy


def compute_metrics(eval_pred) -> Dict:
    logits, labels = eval_pred

    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    weight_mask = labels.view(-1) == -100
    loss_weight = torch.ones(weight_mask.size(0))
    loss_weight[weight_mask] = 0.

    loss = torch.nn.functional.nll_loss(
        log_probs.view(-1, log_probs.size(-1)),
        labels.view(-1),
        weights=loss_weight,
        reduction="sum",
    ).div(loss_weight.sum())

    perplexity = torch.exp(loss).item()

    return {
        'ppl': perplexity,
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

    


