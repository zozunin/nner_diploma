from ..utils.allennlp_extractor import batched_span_select, masked_softmax, weighted_sum, ConfigurationError
from .attentions import Biaffine, MultiHeadAttention
from .graph_network import AGGCN
from torch import nn
import torch


class SpanExtractor(nn.Module):
    # шаблонный класс
    """
    Many NLP models deal with representations of spans inside a sentence.
    SpanExtractors define methods for extracting and representing spans
    from a sentence.
    SpanExtractors take a sequence tensor of shape (batch_size, timesteps, embedding_dim)
    and indices of shape (batch_size, num_spans, 2) and return a tensor of
    shape (batch_size, num_spans, ...), forming some representation of the
    spans.
    """

    def get_input_dim(self) -> int:
        """
        Returns the expected final dimension of the 'sequence_tensor'.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the expected final dimension of the returned span representation.
        """
        raise NotImplementedError
    
    def forward(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            adj_matrix: torch.FloatTensor=None,
            sequence_mask: torch.BoolTensor=None,
            span_indices_mask: torch.BoolTensor=None, ):
        """
        Функция для извлечение спэнов, получения семантического эмбеддинга и конкатенации
        с эмбеддингом по длине
        
        Parameters
        ----------
        sequence_tensor : 'torch.FloatTensor', required.
            A tensor of shape (batch_size, sequence_length, embedding_size)
            representing an embedded sequence of words.
        span_indices : 'torch.LongTensor', required.
            A tensor of shape '(batch_size, num_spans, 2)', where the last
            dimension represents the inclusive start and end indices of the
            span to be extracted from the 'sequence_tensor'.
        sequence_mask : 'torch.BoolTensor', optional (default = 'None').
            A tensor of shape (batch_size, sequence_length) representing padded
            elements of the sequence.
        span_indices_mask : 'torch.BoolTensor', optional (default = 'None').
            A tensor of shape (batch_size, num_spans) representing the valid
            spans in the 'indices' tensor. This mask is optional because
            sometimes it's easier to worry about masking after calling this
            function, rather than passing a mask directly.
        
        Returns
        -------
        A tensor of shape '(batch_size, num_spans, embedded_span_size)',
        where 'embedded_span_size' depends on the way spans are represented.
        """
        # shape (batch_size, num_spans, embedding_dim)
        span_embeddings = self._embed_spans(sequence_tensor, span_indices,
                                            adj_matrix,
                                            sequence_mask, span_indices_mask)
        if self._span_width_embedding is not None:
            # width = end_index - start_index + 1 since 'SpanField' use inclusive indices.
            # But here we do not add 1 because we often initiate the span width
            # embedding matrix with 'num_width_embeddings = max_span_width'
            # shape (batch_size, num_spans)
            widths_minus_one = span_indices[..., 1] - span_indices[..., 0]

            # Embed the span widths and concatenate to the rest of the representations.
            span_width_embeddings = self._span_width_embedding(
                widths_minus_one)
            span_embeddings = torch.cat(
                [span_embeddings, span_width_embeddings], -1)

        if span_indices_mask is not None:
            # Here we are masking the spans which were originally passed in as padding.
            return span_embeddings * span_indices_mask.unsqueeze(-1)

        return span_embeddings
    



class WeightedPoolingSpanExtractor(SpanExtractor):
    """
    Представляет спэны с помощью взвешенного пулинга.
    Для расчета используется представление каждого слова, входящего 
    в отрезок, а также синтаксическое представление, если 
    выборано использование GCN
    
    Parameters
    ----------
    input_dim : 'int', required.
        Размерность полученной последовательности.
    reduced_dim : 'int', required.
        Размерность после снижения с помощью линейного слоя. 
        Стоит учитывать, что при использовании дополнительного эмбеддинга для кодирования ширины отрезка,
        общая длина представления - сумма reduced_dim и span_width_embedding_dim.
    num_width_embeddings : 'int', optional (default = None).
        Число сегментов для представления отрезка по ширине.
    span_width_embedding_dim : 'int', optional (default = None).
        Размерность эмбеддинга для кодирования информации о ширине отрезка.
    use_gcn : 'bool', optional (default = False).
        Выбор применения графовой сверточной сети по информации о синтаксических зависимостях.
    """

    def __init__(
            self,
            input_dim: int,
            reduced_dim: int,
            num_width_embeddings: int=None,
            span_width_embedding_dim: int=None,
            use_gcn: bool=False,) -> None:
        super().__init__()
        
        self._input_dim = input_dim
        self._num_width_embeddings = num_width_embeddings
        self._span_width_embedding = None
        self._use_gcn = use_gcn


        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = nn.Embedding(
                                         num_embeddings=num_width_embeddings,
                                         embedding_dim=span_width_embedding_dim)
        elif num_width_embeddings is not None or span_width_embedding_dim is not None:
            raise ConfigurationError(
                "To use a span width embedding representation, you must"
                "specify both num_width_embeddings and span_width_embedding_dim."
            )

        self.gcn_dim = 0
        if self._use_gcn:
            self.gcn_dim = 256
            self.graph_module = AGGCN(self._input_dim, self.gcn_dim,
                                        tree_prop= 1,
                                        tree_dropout=0.2, 
                                        aggcn_heads=4,
                                        aggcn_sublayer_first=2,
                                        aggcn_sublayer_second=4)

        # embedding_dim+1
        self.dim_reducer = nn.Linear(self._input_dim+self.gcn_dim, reduced_dim+1)

    def _embed_spans(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            adj_matrix: torch.FloatTensor=None,
            sequence_mask: torch.BoolTensor=None,
            span_indices_mask: torch.BoolTensor=None, ) -> torch.FloatTensor:

        if self._use_gcn:
            graph_output = self.graph_module(adj_matrix, sequence_tensor, sequence_mask)
            concat_tensor = torch.cat((sequence_tensor, graph_output), -1)
        else:
            concat_tensor = sequence_tensor

        reduced_tensor = self.dim_reducer(concat_tensor)
        
        concat_output, span_mask = batched_span_select(reduced_tensor,
                                                       span_indices)

        span_embeddings = concat_output[:, :, :, :-1]
        span_attention_logits = concat_output[:, :, :, -1]

        span_attention_weights = masked_softmax(span_attention_logits,
                                                span_mask)

        attended_text_embeddings = weighted_sum(span_embeddings,
                                                span_attention_weights)

        return attended_text_embeddings


class SelfAttentiveSpanExtractor(SpanExtractor):
    """
    Дополнительно контекстуализирует слова с помощью механизма самовнимания.
    Возможно дополнительное кодирование синтаксической информации с помощью GCN.
    Финальное представление каждого слова получается с помощью конкатенации
    эмбеддинга внимания, синтаксического и семантического из BERT, применения
    линейного слоя для снижения размерности. Представление каждого отрезка
    получается с помощью применения взвешенного пулинга.
    
    Parameters
    ----------
    input_dim : 'int', required.
        Размерность полученной последовательности.
    reduced_dim : 'int', required.
        Размерность после снижения с помощью линейного слоя. 
        Стоит учитывать, что при использовании дополнительного эмбеддинга для кодирования ширины отрезка,
        общая длина представления - сумма reduced_dim и span_width_embedding_dim.
    num_width_embeddings : 'int', optional (default = None).
        Число сегментов для представления отрезка по ширине.
    span_width_embedding_dim : 'int', optional (default = None).
        Размерность эмбеддинга для кодирования информации о ширине отрезка.
    num_heads: 'int', optional (default = 4).
        Число голов внимания.
    use_lstm: 'bool', optional (default = False).
        Требуется ли LSTM для снижения размерности конкатенированных представлений. 
    use_gcn : 'bool', optional (default = False).
        Выбор применения графовой сверточной сети по информации о синтаксических зависимостях.
    """

    def __init__(
            self,
            input_dim: int,
            reduced_dim: int,
            num_width_embeddings: int=None,
            span_width_embedding_dim: int=None,
            num_heads: int=4,
            use_lstm: bool=False,
            use_gcn: bool=False,) -> None:
        super().__init__()

        self._input_dim = input_dim
        self._num_width_embeddings = num_width_embeddings
        self._use_gcn = use_gcn
        self._span_width_embedding = None
        self._heads = num_heads
        self._use_lstm = use_lstm
        
        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = nn.Embedding(
                                         num_embeddings=num_width_embeddings,
                                         embedding_dim=span_width_embedding_dim)
        elif num_width_embeddings is not None or span_width_embedding_dim is not None:
            raise ConfigurationError(
                "To use a span width embedding representation, you must"
                "specify both num_width_embeddings and span_width_embedding_dim."
            )

        self.attn = MultiHeadAttention(self._input_dim, self._heads)
        self.attn_dropout = nn.Dropout(0.2)

        self._gcn_dim = 0
        if self._use_gcn:
            self._gcn_dim = 256
            self.graph_module = AGGCN(self._input_dim, self._gcn_dim,
                                        tree_prop= 1,
                                        tree_dropout=0.2, 
                                        aggcn_heads=4,
                                        aggcn_sublayer_first=2,
                                        aggcn_sublayer_second=4)

        if self._use_lstm :
            self.dim_reducer = nn.LSTM(self._input_dim+self._gcn_dim, reduced_dim+1,
                                    num_layers=1, bidirectional=False, batch_first=True)
        elif not self._use_lstm:
            self.dim_reducer = nn.Linear(self._input_dim+self._input_dim+self._gcn_dim, reduced_dim+1)


    def _embed_spans(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            adj_matrix: torch.FloatTensor=None,
            sequence_mask: torch.BoolTensor=None,
            span_indices_mask: torch.BoolTensor=None, ) -> torch.FloatTensor:
        
        attention_output, _ = self.attn(sequence_tensor)
        attention_output = self.attn_dropout(attention_output)
        
        # shape (batch_size, sequence_length, embedding_dim + 1)
        if self._use_lstm:
            concat_tensor = sequence_tensor + attention_output
        else:
            concat_tensor = torch.cat([sequence_tensor, attention_output],
                                    -1)
        if self._use_gcn:
            graph_output = self.graph_module(adj_matrix, sequence_tensor, sequence_mask)
            concat_tensor = torch.cat((concat_tensor, graph_output), -1)   

        if self._use_lstm:
            reduced_tensor, (_, _) = self.dim_reducer(concat_tensor)
        else:
            reduced_tensor = self.dim_reducer(concat_tensor)

        concat_output, span_mask = batched_span_select(reduced_tensor,
                                                       span_indices)

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = concat_output[:, :, :, :-1]
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_logits = concat_output[:, :, :, -1]

        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_weights = masked_softmax(span_attention_logits,
                                                span_mask)

        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim)
        attended_text_embeddings = weighted_sum(span_embeddings,
                                                span_attention_weights)

        return attended_text_embeddings


class BiaffineSpanExtractor(SpanExtractor):
    """
    Дополнительно к BERT кодирует каждое слово с помощью биаффинного механизма.
    Опционально использование модуля кодирования синтаксической информации GCN.
    В результате вектора зависимостей конкатенируются с семантическим представлением из BERT, с 
    помощью пулинга получается представление отрезка.
    
    Parameters
    ----------
    input_dim : 'int', required.
        Размерность получаемой последовательности.
    reduced_dim : 'int', required.
        Размерность после снижения с помощью линейного слоя. 
        Стоит учитывать, что при использовании дополнительного эмбеддинга для кодирования ширины отрезка,
        общая длина представления - сумма reduced_dim и span_width_embedding_dim.
    num_width_embeddings : 'int', optional (default = None).
        Число сегментов для представления отрезка по ширине.
    span_width_embedding_dim : 'int', optional (default = None).
        Размерность эмбеддинга для кодирования информации о ширине отрезка.
    use_gcn : 'bool', optional (default = False).
        Выбор применения графовой сверточной сети по информации о синтаксических зависимостях.
    """

    def __init__(
            self,
            input_dim: int,
            reduced_dim: int,
            num_width_embeddings: int=None,
            span_width_embedding_dim: int=None,
            use_gcn: bool=False,) -> None:
        super().__init__()

        self._input_dim = input_dim
        self._num_width_embeddings = num_width_embeddings
        self._use_gcn = use_gcn
        self._span_width_embedding = None
        
        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = nn.Embedding(
                                         num_embeddings=num_width_embeddings,
                                         embedding_dim=span_width_embedding_dim)
        elif num_width_embeddings is not None or span_width_embedding_dim is not None:
            raise ConfigurationError(
                "To use a span width embedding representation, you must"
                "specify both num_width_embeddings and span_width_embedding_dim."
            )
        self.lstm_dim = self._input_dim// 2 #768/2 обычно
        # batch_first - (batch, seq, 2*dim)
        self.bilstm = nn.LSTM(self._input_dim, self.lstm_dim, 
                         num_layers=1, bidirectional=True, batch_first=True)
        
        self.dep_vec_dim = 256        
        self.biaffine = Biaffine(self.lstm_dim, self.dep_vec_dim)

        self.gcn_dim = 0
        if self._use_gcn:
            self.gcn_dim = 256
            self.graph_module = AGGCN(self._input_dim, self.gcn_dim,
                                tree_prop= 1,
                                tree_dropout=0.2, 
                                aggcn_heads=4,
                                aggcn_sublayer_first=2,
                                aggcn_sublayer_second=4)

        # embedding_dim+1
        self.dim_reducer = nn.Linear(self._input_dim+self.dep_vec_dim
                                     +self.gcn_dim, reduced_dim+1)
    
    def _embed_spans(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            adj_matrix: torch.FloatTensor=None,
            sequence_mask:  torch.FloatTensor=None,
            span_indices_mask: torch.BoolTensor=None) -> torch.FloatTensor:
        
        output, _ = self.bilstm(sequence_tensor)
        h_forward = output[:, :, :self.lstm_dim]
        h_backward = output[:, :, self.lstm_dim:]

        dep_output = self.biaffine(h_forward, h_backward)
        if self._use_gcn:
            graph_output = self.graph_module(adj_matrix, sequence_tensor, sequence_mask)
            concat_tensor = torch.cat((sequence_tensor, dep_output, graph_output), -1)
        else:
            # batch x seq_len x (embedding_dim + 1) + dep_vec_dim
            concat_tensor = torch.cat((sequence_tensor, dep_output), -1)

        reduced_tensor = self.dim_reducer(concat_tensor)
        
        concat_output, span_mask = batched_span_select(reduced_tensor,
                                                       span_indices)

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = concat_output[:, :, :, :-1]
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_logits = concat_output[:, :, :, -1]


        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_weights = masked_softmax(span_attention_logits,
                                                span_mask)

        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim) # почему уменьшилась размерность

        attended_text_embeddings = weighted_sum(span_embeddings,
                                                span_attention_weights)
        return attended_text_embeddings



class SelfBiaffineSpanExtractor(SpanExtractor):

    """
    Для получения представления отрезка используется конкатенация представлений многоголового и биаффиного внимания.
    Опционально использование синтаксической информации (модуль GCN).
    В результате эмбеддинги конкатенируются с семантическим представлением из BERT и с 
    помощью пулинга получается представление отрезков.
    
    Parameters
    ----------
    input_dim : 'int', required.
        Размерность получаемой последовательности.
    reduced_dim : 'int', required.
        Размерность после снижения с помощью линейного слоя. 
        Стоит учитывать, что при использовании дополнительного эмбеддинга для кодирования ширины отрезка,
        общая длина представления - сумма reduced_dim и span_width_embedding_dim.
    num_width_embeddings : 'int', optional (default = None).
        Число сегментов для представления отрезка по ширине.
    span_width_embedding_dim : 'int', optional (default = None).
        Размерность эмбеддинга для кодирования информации о ширине отрезка.
    num_heads : 'int', optional (default = 2).
        Число голов для механизма внимания
    use_gcn : 'bool', optional (default = False).
        Выбор применения графовой сверточной сети по информации о синтаксических зависимостях.
    """

    def __init__(
            self,
            input_dim: int,
            reduced_dim: int,
            num_width_embeddings: int=None,
            span_width_embedding_dim: int=None,
            num_heads: int=2,
            use_gcn: bool=False,) -> None:
        super().__init__()

        self._input_dim = input_dim
        self._num_width_embeddings = num_width_embeddings
        self._use_gcn = use_gcn
        self._span_width_embedding = None
        self._heads = num_heads
        
        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = nn.Embedding(
                                         num_embeddings=num_width_embeddings,
                                         embedding_dim=span_width_embedding_dim)
        elif num_width_embeddings is not None or span_width_embedding_dim is not None:
            raise ConfigurationError(
                "To use a span width embedding representation, you must"
                "specify both num_width_embeddings and span_width_embedding_dim."
            )
        self.attn = MultiHeadAttention(self._input_dim, self._heads)

        self.lstm_dim = self._input_dim// 2 #768/2 обычно
        # batch_first - (batch, seq, 2*dim)
        self.bilstm = nn.LSTM(self._input_dim, self.lstm_dim, 
                         num_layers=1, bidirectional=True, batch_first=True)
        
        self.dep_vec_dim = 256        
        self.biaffine = Biaffine(self.lstm_dim, self.dep_vec_dim)

        self.gcn_dim = 0
        if self._use_gcn:
            self.gcn_dim = 256
            self.graph_module = AGGCN(self._input_dim, self.gcn_dim,
                                tree_prop= 1,
                                tree_dropout=0.2, 
                                aggcn_heads=4,
                                aggcn_sublayer_first=2,
                                aggcn_sublayer_second=4)

        self.dim_reducer = nn.Linear(self._input_dim+self.dep_vec_dim+
                                     self.gcn_dim+self._input_dim, reduced_dim+1)
    
    def _embed_spans(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            adj_matrix: torch.FloatTensor=None,
            sequence_mask:  torch.FloatTensor=None,
            span_indices_mask: torch.BoolTensor=None) -> torch.FloatTensor:
        
        attention_output, _ = self.attn(sequence_tensor)

        output, _ = self.bilstm(sequence_tensor)
        h_forward = output[:, :, :self.lstm_dim]
        h_backward = output[:, :, self.lstm_dim:]

        dep_output = self.biaffine(h_forward, h_backward)
        if self._use_gcn:
            graph_output = self.graph_module(adj_matrix, sequence_tensor, sequence_mask)
            concat_tensor = torch.cat((sequence_tensor, dep_output, graph_output, attention_output), -1)
        else:
            # batch x seq_len x (embedding_dim + 1) + dep_vec_dim
            concat_tensor = torch.cat((sequence_tensor, dep_output, attention_output), -1)

        reduced_tensor = self.dim_reducer(concat_tensor)
        
        concat_output, span_mask = batched_span_select(reduced_tensor,
                                                       span_indices)

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = concat_output[:, :, :, :-1]
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_logits = concat_output[:, :, :, -1]

        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_weights = masked_softmax(span_attention_logits,
                                                span_mask)

        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim) # почему уменьшилась размерность
        attended_text_embeddings = weighted_sum(span_embeddings,
                                                span_attention_weights)
        return attended_text_embeddings
