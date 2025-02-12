import logging
import os
from typing import Optional
from functools import partial
from datasets import load_dataset, load_from_disk,Dataset,DatasetDict
from omegaconf import DictConfig

from fusion_bench.utils import instantiate, timeit_context

import pandas as pd
from .datasets_preprocess import preprocess
from .perm_preprocessors import PERM_Preprocessor
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Union  # noqa: F401

from transformers import AutoTokenizer

log = logging.getLogger(__name__)
def _format_prompt(data_json: dict,append_label:bool=False,prompt_template:str="{}"):
    data = []
    for i in range(len(data_json)):
        inputs = data_json[i]['input']
        target = data_json[i]['target']
        prompt_body = prompt_template.format(inputs)
        if append_label:
            text = f"{prompt_body}{target}</s>"
            data.append({'text': text})
        else:
            text = f"{prompt_body}"
            data.append({'prompt': text,'label': target})
        
    df = pd.DataFrame(data=data)
    df = df.astype('str')
    data = Dataset.from_pandas(df)
    return data

def _load_perm_dataset(name, tokenizer,split_dir: str = None):
    if isinstance(tokenizer, (DictConfig, dict)):
        tokenizer = instantiate(tokenizer)
    dataset = {'train':{},'val':{},'test':{}}
    # preprocessor = partial(preprocess,
    # tokenizer=tokenizer,
    # tokenizer_kwawgs= {
    #         "padding": "max_length",
    #         "truncation": True,
    #         "return_tensors": "pt",
    #     }
    # )
    if tokenizer.pad_token is None:
        ## Temporary solution
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.pad_token = tokenizer.eos_token
    preprocessor = PERM_Preprocessor(
        template=None,
        tokenizer=tokenizer,
        tokenizer_kwargs={
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
            "max_length": 2048,
        },
    )

    for split in dataset:
        data_dir = os.path.join(split_dir,f"{split}.json")
        data_json = json.load(open(data_dir,'r'))
        data_json = data_json["data"]
        formatted_data = _format_prompt(data_json,append_label=False)
        ## Use a smaller test set for debugging
        if split == 'test':
            formatted_data = formatted_data.train_test_split(test_size=0.05,seed=42)['test']
        dataset[split] = formatted_data
    dataset = DatasetDict(dataset)
    dataset = dataset.map(
        preprocessor,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )
    return dataset


def load_perm_dataset(
    name,
    tokenizer,
    cache_dir: Optional[str] = "outputs/cache",
    split: Optional[str] = None,
    split_dir: str = "/work/10269/fcyin/algorithmic_composition/data/tokenized_v2/perm"
):
    with timeit_context(f"Loading {name} dataset"):
        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            cache_path = os.path.join(
                cache_dir, "algocomp-perm", f"_load_{name}_dataset_cached"
            )
            if os.path.exists(cache_path):
                dataset = load_from_disk(cache_path)
            else:
                data_dir = os.path.join(split_dir,name)
                if not os.path.exists(data_dir):
                    raise ValueError(f"Data directory not found at {data_dir}")
                
                dataset = _load_perm_dataset(name, tokenizer,data_dir)
                log.info(f"Saving {name} dataset to {cache_path}")
                dataset.save_to_disk(cache_path)
        else:
            dataset = _load_glue_dataset(name, tokenizer)

    if split is not None:
        return dataset[split]
    else:
        return dataset

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",model_max_length=4096)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_perm_dataset("perm", tokenizer,"outputs/cache")
