from .utils.allennlp_extractor import replace_masked_values
from .utils.allennlp_classifier import TimeDistributed
from typing import Dict, Any
from torch import nn
import torch

class NERTagger(nn.Module):

    '''
    Модуль классификации полученных отрезков на классы именованных сущностей

    Attributes
    ----------
    input_dim : 'int', required.
        Размерность получаемой последовательности.
    num_labels: 'int', required.
        Число классов.
    ff_dropout: 'float', optional (default = 0.4).
        Вероятность для Dropout в ff модуле.
    '''

    def __init__(self,
                 input_dim: int,
                 num_labels: int,
                 ff_dropout: float=0.4) -> None:
        super(NERTagger, self).__init__()

        self._num_labels = num_labels
        feed_forward = nn.Sequential(nn.Linear(input_dim, 256), 
                                    nn.ReLU(),
                                    nn.Dropout(ff_dropout))
        self._ner_scorer = torch.nn.Sequential(
                           TimeDistributed(feed_forward),
                           TimeDistributed(nn.Linear(256, self._num_labels-1)))
    def forward(self,
                spans: torch.IntTensor,
                span_mask: torch.IntTensor,
                span_embeddings: torch.IntTensor,
                previous_step_output: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:


        # Shape: (Batch size, Number of Spans, Span Embedding Size)
        # span_embeddings
        ner_scores = self._ner_scorer(span_embeddings)
        # Give large negative scores to masked-out elements.
        mask = span_mask.unsqueeze(-1)
        ner_scores = replace_masked_values(ner_scores, mask, -1e20)
        dummy_dims = [ner_scores.size(0), ner_scores.size(1), 1]
        dummy_scores = ner_scores.new_zeros(*dummy_dims)
        if previous_step_output is not None and "predicted_span" in previous_step_output and not self.training:
            dummy_scores.masked_fill_(previous_step_output["predicted_span"].bool().unsqueeze(-1), -1e20)
            dummy_scores.masked_fill_((1-previous_step_output["predicted_span"]).bool().unsqueeze(-1), 1e20)

        ner_scores = torch.cat((dummy_scores, ner_scores), -1)

        if previous_step_output is not None and "predicted_seq_span" in previous_step_output and not self.training:
            for row_idx, all_spans in enumerate(spans):
                pred_spans = previous_step_output["predicted_seq_span"][row_idx]
                pred_spans = all_spans.new_tensor(pred_spans)
                for col_idx, span in enumerate(all_spans):
                    if span_mask[row_idx][col_idx] == 0:
                        continue
                    bFind = False
                    for pred_span in pred_spans:
                        if span[0] == pred_span[0] and span[1] == pred_span[1]:
                            bFind = True
                            break
                    if bFind:
                        # if find, use the ner scores, set dummy to a big negative
                        ner_scores[row_idx, col_idx, 0] = -1e20
                    else:
                        # if not find, use the previous step, set dummy to a big positive
                        ner_scores[row_idx, col_idx, 0] = 1e20

        return ner_scores