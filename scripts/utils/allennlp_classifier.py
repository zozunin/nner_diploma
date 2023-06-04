from typing import List
from torch import nn
import torch

class TimeDistributed(nn.Module):
    """
    На вход получает данные размерности (batch_size, time_steps, [rest]) и Module,
    принимающий на вход данные размерности (batch_size, [rest])
    Модуль TimeDistributed меняет размер входных данных на (batch_size * time_steps, [rest]),
    применяет трансформации из Module и трансформирует размерность обратно.
    """

    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, *inputs, pass_through: List[str]=None, **kwargs):

        pass_through = pass_through or []

        reshaped_inputs = [
            self._reshape_tensor(input_tensor) for input_tensor in inputs
        ]

        some_input = None
        if inputs:
            some_input = inputs[-1]

        reshaped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key not in pass_through:
                if some_input is None:
                    some_input = value

                value = self._reshape_tensor(value)

            reshaped_kwargs[key] = value

        # применение модели к преобразованным данным
        reshaped_outputs = self._module(*reshaped_inputs, **reshaped_kwargs)

        if some_input is None:
            raise RuntimeError("No input tensor to time-distribute")

        # Приводим вывод к нужной размерности:
        # (batch_size, time_steps, **output_size)
        new_size = some_input.shape[:2] + reshaped_outputs.shape[1:]
        outputs = reshaped_outputs.contiguous().reshape(new_size)

        return outputs

    @staticmethod
    def _reshape_tensor(input_tensor):
        input_size = input_tensor.shape
        if len(input_size) <= 2:
            raise RuntimeError(f"No dimension to distribute: {input_size}")
        # Сведение batch_size and time_steps в единую ось, размерность:
        # (batch_size * time_steps, **input_size).
        squashed_shape = [-1] + list(input_size[2:])
        return input_tensor.reshape(*squashed_shape)