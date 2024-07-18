import torch
import numpy as np

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Dict, List, Optional, Union


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded


@dataclass
class DataCollatorForLeftPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: bool = False
    return_tensors: Union[str, None] = None
    label_pad_token_id: int = -100
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None


    def __call__(self, features: List[Dict[str, Any]]) -> Dict:
        if self.return_tensors is None:
            self.return_tensors = "pt"
        
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        # run through tokenizer without labels to ensure no side effects.
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # we have to pad the labels manually as we cannot rely on 'tokenizer.pad'
        # and, we need them to be of the same length to return tensors.
        if labels is not None:
            if self.padding is False:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"]  = [np.concatenate([label, []] for label in labels)]
            else:
                max_padding = self.padding is True and self.max_length is not None
                max_label_length = max([len(label) for label in labels]) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )
                
                padding_side = self.tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label 
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                    np.array([self.label_pad_token_id] * (max_label_length) - len(label), dtype=np.int64),
                            ]
                        )
                        for label in labels
                    ]
        
        # reintroduce side effects via tokenizer that return respective datatypes for the 'return_tensors' argument.
        if batch.get("labels", None) is not None:
            if self.return_tensors == "pt":
                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            elif self.return_tensors == "tf":
                raise ValueError("This class does snot support tf tensor type.")
            else:
                batch["labels"] = np.array(batch["labels"], dtype=torch.int64)
        else:
            batch["labels"] = None


        return batch



# def main():
#     from data_loader import load_and_preprocess_data
#     from arguments import TrainArguments
#     from transformers import AutoTokenizer, AutoModelForCausalLM
    
#     config = TrainArguments.define_args()
#     tokenizer = AutoTokenizer.from_pretrained(config.base_model)
#     tokenizer.add_tokens(["<|unused|>"], special_tokens=True)
#     # tokenizer.add_special_tokens({"<|unused|>": len(tokenizer.vocab) + 1})
#     tokenizer.pad_token_id = 58944 # 58944
#     print(tokenizer.pad_token_id)

#     model = AutoModelForCausalLM.from_pretrained(config.base_model, device_map="cpu")
#     model.resize_token_embeddings(len(tokenizer))

#     train_batch_size = config.per_device_train_batch
#     valid_batch_size = config.per_device_valid_batch
#     train_data, valid_data = load_and_preprocess_data(config, tokenizer)

#     collator = DataCollatorForLeftPadding(
#         tokenizer,
#         model,
#         padding=True,
#         return_tensors="pt",
#         pad_to_multiple_of=8,
#     )

#     batches = []
#     # for i in range(0, len(train_data), train_batch_size):
#     #     batch = train_data[i:i+train_batch_size]
#     #     features = [{k: batch[k][j] for k in batch.keys()} for j in range(train_batch_size)]
#     #     batches.append(features)
#     for i in range(0, len(valid_data), valid_batch_size):
#         batch = train_data[i:i+valid_batch_size]
#         features = [{k: batch[k][j] for k in batch.keys()} for j in range(valid_batch_size)]
#         batches.append(features)
    
#     for batch in batches:
#         collator(batch)
    

# if __name__ == "__main__":
#     main()
