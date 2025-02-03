from typing import Any, Dict

from .datasets_preprocess import DatasetPreprocessor, preprocess


class PEN_Preprocessor(DatasetPreprocessor):
    def preprocess(self, sentence: str, label: int):
        input_text = self.template["prompt"].format(sentence=sentence)
        target_text = self.template["label"].format(label=label)
        
        return input_text, target_text

    def __call__(self, example: Dict[str, Any]):
        """
        Preprocess PEN dataset into a text-to-text format.
        """
        if isinstance(example["prompt"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["prompt"], example["label"]
            )
        else:
            # batched
            input_text, target_text = [], []
            for sentence, label in zip(example["prompt"], example["label"]):
                _input_text, _target_text = self.preprocess(sentence, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )