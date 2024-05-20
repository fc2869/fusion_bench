import logging
from copy import deepcopy
from typing import List, Mapping, Union

import torch
from torch import Tensor, nn

from ..utils.state_dict_arithmetic import state_dict_add, state_dict_avg, state_dict_mul
from ..utils.type import _StateDict
from .base_algorithm import ModelFusionAlgorithm
from ..modelpool import ModelPool, ListModelPool

log = logging.getLogger(__name__)


def simple_average(modules: List[Union[nn.Module, _StateDict]]):
    """
    Averages the parameters of a list of PyTorch modules or state dictionaries.

    This function takes a list of PyTorch modules or state dictionaries and returns a new module with the averaged parameters, or a new state dictionary with the averaged parameters.

    Args:
        modules (List[Union[nn.Module, _StateDict]]): A list of PyTorch modules or state dictionaries.

    Returns:
        module_or_state_dict (Union[nn.Module, _StateDict]): A new PyTorch module with the averaged parameters, or a new state dictionary with the averaged parameters.

    Examples:
        >>> import torch.nn as nn
        >>> model1 = nn.Linear(10, 10)
        >>> model2 = nn.Linear(10, 10)
        >>> averaged_model = simple_averageing([model1, model2])

        >>> state_dict1 = model1.state_dict()
        >>> state_dict2 = model2.state_dict()
        >>> averaged_state_dict = simple_averageing([state_dict1, state_dict2])
    """
    if isinstance(modules[0], nn.Module):
        new_module = deepcopy(modules[0])
        state_dict = state_dict_avg([module.state_dict() for module in modules])
        new_module.load_state_dict(state_dict)
        return new_module
    elif isinstance(modules[0], Mapping):
        return state_dict_avg(modules)


class SimpleAverageAlgorithm(ModelFusionAlgorithm):

    @torch.no_grad()
    def run(self, modelpool: ModelPool):
        """
        Fuse the models in the given model pool using simple averaging.

        This method iterates over the names of the models in the model pool, loads each model, and appends it to a list.
        It then returns the simple average of the models in the list.

        Args:
            modelpool: The pool of models to fuse.

        Returns:
            The fused model obtained by simple averaging.
        """
        if isinstance(modelpool, (list, tuple)) and all(
            isinstance(m, nn.Module) for m in modelpool
        ):
            modelpool = ListModelPool(modelpool)
        log.info(
            f"Fusing models using simple average on {len(modelpool.model_names)} models."
            f"models: {modelpool.model_names}"
        )
        sd: _StateDict = None
        forward_model = None

        for model_name in modelpool.model_names:
            model = modelpool.load_model(model_name)
            if sd is None:
                sd = model.state_dict(keep_vars=True)
                forward_model = model
            else:
                sd = state_dict_add(sd, model.state_dict(keep_vars=True))

        sd = state_dict_mul(sd, 1 / len(modelpool.model_names))
        forward_model.load_state_dict(sd)
        return forward_model
