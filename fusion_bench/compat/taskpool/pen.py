from typing import Union

from omegaconf import DictConfig
from tqdm.autonotebook import tqdm
from fusion_bench.tasks import BaseTask
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from fusion_bench.tasks.PEN.pen_load_dataset import (
    load_pen_dataset,
)
from fusion_bench.tasks.PEN.pen_evaluation import (
    evaluate_accuracy
)
import functools
from fusion_bench.compat.taskpool import TaskPool
from fusion_bench.mixins import LightningFabricMixin
from transformers import LlamaForCausalLM
import torch
import os
import json
import logging
log = logging.getLogger(__name__)

PEN_TASKS = [
    "pen"
]
class PENTask(BaseTask):
    """
    A class to manage a pool of tasks for evaluation.
    This is the base class for version 0.1.x, deprecated.
    Use `fusion_bench.taskpool.BaseTaskPool` instead.

    Attributes:
        config (DictConfig): The configuration for the task pool.
        _all_task_names (List[str]): A list of all task names in the task pool.
    """

    _taskpool: "PenTaskPool" = None

    @property
    def taskpool(self):
        if self._taskpool is not None:
            return self._taskpool
        else:
            raise ValueError("Taskpool not set")

    @property
    def fabric(self):
        return self.taskpool.fabric

    @property
    def tokenizer(self):
        return self.taskpool.tokenizer

    @functools.cached_property
    def dataset(self):
        log.info(f'Loading dataset: "{self.config.dataset.name}"')

        dataset = load_pen_dataset(
            self.config.dataset.name, self.tokenizer, self.taskpool.config.cache_dir
        )
        return dataset

    @functools.cached_property
    def test_dataset(self, data_dir: str = None):
        return self.dataset[self.config.dataset.split]

    @property
    def test_loader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.taskpool.config.batch_size,
            num_workers=self.taskpool.config.num_workers,
            shuffle=False,
            collate_fn=default_data_collator,
        )
        loader = self.fabric.setup_dataloaders(loader)
        return loader

    @torch.no_grad()
    def evaluate(self, model):
        exact_acc, outputs = evaluate_accuracy(model, self.test_loader, self.tokenizer)
        result = {"accuracy": exact_acc}
        log.info(f'result for task "{self.config.name}": {result}')
        log.info(f"Writing outputs to {self.taskpool.config.output_dir}")
        eval_output_dir = os.path.join(self.taskpool.config.output_dir,"eval_outputs")
        os.makedirs(eval_output_dir, exist_ok=True)
        with open(os.path.join(eval_output_dir,f"outputs.json"),"w") as f:
            json.dump(outputs,f)
        return result


class PENTaskPool(LightningFabricMixin, TaskPool):
    """
    """

    _tokenizer_instance = None

    @property
    def tokenizer(self):
        """
        Returns the tokenizer. If it's not already initialized, it initializes it using the config's tokenizer.
        """
        if self._tokenizer_instance is None:
            self._tokenizer_instance = AutoTokenizer.from_pretrained(
                self.config.tokenizer
            )
        return self._tokenizer_instance

    def load_task(self, task_name_or_config: str | DictConfig):
        """
        Loads a task given a task name or config. If the task name is in `CLASSIFICATION_TASKS`, it creates a `FlanT5GLUETextGenerationClassificationTask`.
        If the task name is in `REGRESSION_TASKS`, it creates a `FlanT5GLUETextGenerationRegressionTask`. Otherwise, it raises a `ValueError`.
        """
        if isinstance(task_name_or_config, str):
            task_config = self.get_task_config(task_name_or_config)
        else:
            task_config = task_name_or_config

        if task_config.name in PEN_TASKS:
            task = PENTask(task_config)
            task._taskpool = self
            return task
        else:
            raise ValueError(f"Unknown task {task_config.name}")

    def evaluate(self, model: LlamaForCausalLM):
        """
        Evaluate the model on the FlanT5 GLUE text generation tasks.

        Args:
            model (T5ForConditionalGeneration): The model to evaluate.

        Returns:
            dict: A dictionary containing the evaluation results for each task.
        """
        
        report = {}
        report.update(super().evaluate(model))
        log.info(f"evaluation report: {report}")
        return report
