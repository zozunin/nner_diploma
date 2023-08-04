import torch.nn.functional as F
from torch import Tensor, cuda
from typing import Optional
import torch



class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message


def get_range_vector(size: int, device: int) -> Tensor:

    if device > -1:
        return cuda.LongTensor(
            size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)
    
def get_device_of(tensor: Tensor) -> int:

    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()
    

def flatten_and_batch_shift_indices(indices: Tensor,
                                    sequence_length: int) -> Tensor:
    """
    Вспомогательная функция для batched_index_select (см. далее)
    На вход функция получает "indices" размерности (batch_size, d_1, ..., d_n)
    приводит все к размерности: (batch_size, sequence_length, embedding_size)
    ```python
        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]
    ```
    # Parameters
    indices : `torch.LongTensor`, required.
    sequence_length : `int`, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.
    # Returns
    offset_indices : `torch.LongTensor`
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ConfigurationError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    
    offsets = (get_range_vector(indices.size(0), get_device_of(indices)) *
               sequence_length)
    for _ in range(len(indices.shape) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.reshape(-1)
    return offset_indices


def batched_index_select(
        target: Tensor,
        indices: torch.LongTensor,
        flattened_indices: Optional[torch.LongTensor]=None, ) -> Tensor:
    """
    На вход функция получает "indices" размерности (batch_size, d_1, ..., d_n). Они индексируются
    в размерность последовательности (dim 2). Размерность таргета: 
    (batch_size, sequence_length, embedding_size)
    Возвращает отобранные значения в таргете с опорой на полученные индексы,
    размера (batch_size, d_1, ..., d_n, embedding_size).
    
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    flattened_indices : `Optional[torch.Tensor]`, optional (default = `None`)
        An optional tensor representing the result of calling `flatten_and_batch_shift_indices`
        on `indices`. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.
    # Returns
    selected_targets : `torch.Tensor`
        A tensor with shape [indices.shape, target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices,
                                                            target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.reshape(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.shape) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.reshape(*selected_shape)
    return selected_targets


def get_lengths_from_binary_sequence_mask(
        mask: torch.BoolTensor) -> torch.LongTensor:
    """
    Вычисление длины последовательности в каждом батче с помощьью бинарной маски
    # Parameters
    mask : `torch.BoolTensor`, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.
    # Returns
    `torch.LongTensor`
        A torch.LongTensor of shape (batch_size,) representing the lengths
        of the sequences in the batch.
    """
    return mask.sum(-1)


def batched_span_select(target: Tensor,
                        spans: torch.LongTensor) -> Tensor:
    """
    На вход получает спэны размерности (batch_size, num_spans, 2), 
    индексируется в размерность последовательности (dim 2) таргета, который
    представлен размером: (batch_size, sequence_length, embedding_size)
    Возвращает сегментированные спэны в таргете с учетом полученных индексов:
    Эмбеддинг спэна размерности (batch_size, num_spans, max_batch_span_width, embedding_size)
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A 3 dimensional tensor of shape (batch_size, num_spans, 2) representing start and end
        indices (both inclusive) into the `sequence_length` dimension of the `target` tensor.
    # Returns
    span_embeddings : `torch.Tensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width, embedding_size)
        representing the embedded spans extracted from the batch flattened target tensor.
    span_mask: `torch.BoolTensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width) representing the mask on
        the returned span embeddings.
    """
    # both of shape (batch_size, num_spans, 1)
    span_starts, span_ends = spans.split(1, dim=-1)


    # shape (batch_size, num_spans, 1)
    # These span widths are off by 1, because the span ends are `inclusive`.
    span_widths = span_ends - span_starts

    # We need to know the maximum span width so we can
    # generate indices to extract the spans from the sequence tensor.
    # These indices will then get masked below, such that if the length
    # of a given span is smaller than the max, the rest of the values
    # are masked.
    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = get_range_vector(
                            max_batch_span_width, get_device_of(target)).reshape(1, 1, -1)
    # Shape: (batch_size, num_spans, max_batch_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.
    #
    # We're using <= here (and for the mask below) because the span ends are
    # inclusive, so we want to include indices which are equal to span_widths rather
    # than using it as a non-inclusive upper bound.
    span_mask = max_span_range_indices <= span_widths #(...).float()
    raw_span_indices = span_starts + max_span_range_indices
    # span_ends - max_span_range_indices

    # We also don't want to include span indices which greater than the sequence_length,
    # which happens because some spans near the end of the sequence
    # have a start index + max_batch_span_width > sequence_length, so we add this to the mask here.
    span_mask = (span_mask & (raw_span_indices < target.size(1)) &
                 (0 <= raw_span_indices)) # доп ограничение к оригиналу
    span_indices = raw_span_indices * span_mask 

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    # flatten & batch at once
    span_embeddings = batched_index_select(target, span_indices)

    return span_embeddings, span_mask


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Возвращает относительно маленькое значение данного типа данных, 
    что применяется во избежание ошибок с вычислением (деление на 0)
    Поддерживает только типы с плавающей точкой
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))
    

def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)

def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min


def max_value_of_dtype(dtype: torch.dtype):
    """
    Returns the maximum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).max


def weighted_sum(matrix: Tensor,
                 attention: Tensor) -> Tensor:
    """
    На вход получает матрицу векторов и множество весов для рядов в этой матрице
    - вектор внимания. После этого возвращается взвешеная сумма рядов в матрице.
    """
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.ndim == 2 and matrix.ndim == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1) # batch matrix-matrix product
    if attention.ndim == 3 and matrix.ndim == 3:
        return attention.bmm(matrix)
    if matrix.ndim - 1 < attention.ndim:
        expanded_size = list(matrix.shape)
        for i in range(attention.ndim - matrix.ndim + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def masked_softmax(
        vector: Tensor,
        mask: torch.BoolTensor,
        dim: int=-1,
        memory_efficient: bool=False, ) -> Tensor:
    """
    F.softmax(vector) не применяется, так как некоторые элементы вектора могут быть
    замаскированы. Поэтому функция применяет операцию softmax только на незамаскированной
    части вектора.
    """
    if mask is None:
        result = F.softmax(vector, dim=dim)
    else:
        while mask.ndim < vector.ndim:
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, 
            # we zero these out.
            result = F.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) +
                               tiny_value_of_dtype(result.dtype))
        else:
            masked_vector = vector.masked_fill(
                ~mask, min_value_of_dtype(vector.dtype))
            result = F.softmax(masked_vector, dim=dim)
    return result

def replace_masked_values(
    tensor: Tensor, mask: torch.BoolTensor, replace_with: float
) -> Tensor:
    """
    Replaces all masked values in `tensor` with `replace_with`.  `mask` must be broadcastable
    to the same shape as `tensor`. We require that `tensor.dim() == mask.dim()`, as otherwise we
    won't know which dimensions of the mask to unsqueeze.
    This just does `tensor.masked_fill()`, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    `tensor.masked_fill(~mask, replace_with)`.
    """
    if tensor.dim() != mask.dim():
        raise ConfigurationError(
            "tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim())
        )
    return tensor.masked_fill(~mask, replace_with)


