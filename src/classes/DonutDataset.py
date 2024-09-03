import json
from typing import Any, Tuple
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import DonutProcessor


class DonutDataset(Dataset):
    def __init__(
        self, dataset_name: str, processor: DonutProcessor, split: str = "train"
    ):
        super().__init__()
        self.dataset = load_dataset(dataset_name, split=split)
        self.processor = processor
        self.split = split
        self.sort_json_key = True
        self.added_tokens = []
        self.cache = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # check cache first
        if index in self.cache:
            return self.cache[index]
        
        item = self.dataset[index]

        # inputs
        pixel_values = self.processor(
            item["image"], random_padding=self.split == "train", return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.squeeze()

        # targets
        ground_truth = json.loads(item["ground_truth"])
        assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)

        gt_jsons = [ground_truth["gt_parse"]]

        target_sequence = [
            self.json2token(
                gt_json,
                sort_json_key=self.sort_json_key,
            )
            + self.processor.tokenizer.eos_token
            for gt_json in gt_jsons  # load json from list of json
        ]
        target_sequence = target_sequence[0]
    
        input_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.processor.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = (
            -100
        )  # model doesn't need to predict pad token

        # Cache the results
        self.cache[index] = (pixel_values, labels, target_sequence)
        
        return pixel_values, labels, target_sequence

    def json2token(
        self,
        obj: Any,
        sort_json_key: bool = True,
    ):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=False)
                else:
                    keys = obj.keys()
                for k in keys:
                    output += (
                        rf"<s_{k}>"
                        + self.json2token(obj[k], sort_json_key)
                        + rf"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj
