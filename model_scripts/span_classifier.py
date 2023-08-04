from .utils.allennlp_extractor import ConfigurationError
from .extractor_modules.extractors import *
from .classifier_module import NERTagger
from transformers import BertModel
import torch.nn.functional as F
from torch import nn
import torch

class NNERModel(nn.Module):
    '''
    Общий класс для span-based классификации сущностей, включающий генерацию отрезков и их
    последующую классификацию

    Attributes
    ----------
        num_labels : int, required.
            Число классов.
        device : str, required.
            'cuda' или 'cpu'
        dropout_rate : float, optional (default = 0.4).
            Dropout для модуля извлечения представлений токенов с помощью BERT.
        max_seq_len : int, optional (default = 256).
            Максимальная длина последовательности в токенах.
        num_heads : int, optional (default = None).
            Число голов в случае применения модули кодировки, включающего механизм внимания
            (в случае с SelfAttentiveSpanExtractor, SelfBiaffineSpanExtractor)
        max_span_len : int, optional (default = None).
            Максимальный размер (ширина) в словах отрезка.
        extractor_type : str, optional (default = 'biaffine').
            Тип модуля извлечения эмбеддингов отрезков.
            Варианты:
                'weightedpooling' - для просто получения отрезка из BERT-представлений посредством взвешенного пулинга.
                'biaffine' - получение представления отрезка с дополнительным определением представления по 
                             зависимостям с помощью биаффинного модуля. Семантическое из BERT и биаффинное представления
                             конкатенируются, применяется уменьшение размерности, после чего также с помощью взвешенного 
                             пулинга для каждого отрезка выводится представление.
                'selfbiaffine' - помимо BERT используется также получение эмбеддинга с помощью многоголового внимания
                                 и биаффинного.
                'lstmattention' - помимо BERT используется модуль многоголового внимания, для уменьшения размерности 
                                  применяется рекуррентный слой.
                'linearattention' - для уменьшения размерности используется линейный слой.
        mode : str, optional (default = 'classification').
            Режим модуля. Вариант 'extraction' применяется для получения отрезков и их представлений в рамках 
            эксперимента с few-shot обучением. Изменение для классификации не требуется.
        extractor_use_gcn : bool, optional (default = False).
            Требуется ли в модуле извлечения отрезков использовать дополнительное кодирование синтаксической информации
            с помощью графовых сверточных сетей.
    '''
    def __init__(self,
                 num_labels,
                 device,
                 dropout_rate=0.4,
                 max_seq_len=256,
                 num_heads = None,
                 max_span_len=None,
                 extractor_type='biaffine',
                 mode='classification',
                 extractor_use_gcn=False):
        
        super(NNERModel, self).__init__()
        
        self.max_span_len = max_span_len
        self.num_labels = num_labels
        self.extractor_type = extractor_type
        self.device = device
        self.mode = mode

        if num_heads == None and extractor_type in ['lstmattention', 'linearattention',
                                                        'selfbiaffine']:
            raise ConfigurationError(
                "To use attention mechanism you must specify num_heads."
            )

        # init span_indices
        if self.max_span_len == None or self.max_span_len > max_seq_len:
            self.max_span_len = 50
        elif self.max_span_len <= 0:
            self.max_span_len = 50
        self.max_span_len = int(self.max_span_len)

        self.triangle_mask = (torch.triu(
                            torch.ones(max_seq_len, max_seq_len), diagonal=0) - torch.triu(
                            torch.ones(max_seq_len, max_seq_len), diagonal=max_span_len)).bool()
        

        self.encoder = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
        encoder_hidden_state = self.encoder.config.hidden_size

        self.encoder_dropout = nn.Dropout(dropout_rate)

        reduced_dim = 512 # снижение размерности при извлечении отрезков
        width_dim = 128 # величина эмбеддинга по длине спэна

        if extractor_type=='weightedpooling':
            self.extractor = WeightedPoolingSpanExtractor(input_dim=encoder_hidden_state,
                                                          reduced_dim=reduced_dim,
                                                          num_width_embeddings = self.max_span_len,
                                                          span_width_embedding_dim=width_dim,
                                                          use_gcn=extractor_use_gcn)
        
        elif extractor_type in ['lstmattention', 'linearattention']:
            if extractor_type == 'lstmattention':
                lstm = True
                reduced_dim = encoder_hidden_state
            else:
                lstm = False
        
            self.extractor = SelfAttentiveSpanExtractor(input_dim=encoder_hidden_state,
                                                        reduced_dim=reduced_dim,
                                                        num_width_embeddings = self.max_span_len,
                                                        span_width_embedding_dim=width_dim,
                                                        num_heads=num_heads,
                                                        use_lstm=lstm,
                                                        use_gcn=extractor_use_gcn
                                                        )

        elif extractor_type=='biaffine':
            self.extractor = BiaffineSpanExtractor(input_dim=encoder_hidden_state,
                                                  reduced_dim=reduced_dim,
                                                  num_width_embeddings = self.max_span_len,
                                                  span_width_embedding_dim=width_dim,
                                                  use_gcn = extractor_use_gcn)
        
        elif extractor_type == 'selfbiaffine':
            self.extractor = SelfBiaffineSpanExtractor(input_dim=encoder_hidden_state,
                                                       reduced_dim=reduced_dim,
                                                       num_width_embeddings = self.max_span_len,
                                                       span_width_embedding_dim=width_dim,
                                                       num_heads = num_heads,
                                                       use_gcn = extractor_use_gcn)
        if mode == 'extraction':
            self.reducer = nn.Linear(reduced_dim+width_dim, 256)
        elif mode == 'classification':
            self.classifier = NERTagger(reduced_dim+width_dim, self.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        adj_matrix=None
        ):

        batch_size, max_seq_len = input_ids.size(0), input_ids.size(1)

        embedded_text_input = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.encoder_dropout(F.leaky_relu(embedded_text_input))

        span_indices = self.triangle_mask.nonzero().unsqueeze(0).expand(
                        batch_size, -1, -1).to(self.device)

        span_embeddings = self.extractor(sequence_tensor=embedded_text_input,
                                         span_indices=span_indices,
                                         adj_matrix=adj_matrix,
                                         sequence_mask=attention_mask)

        if self.mode == 'extraction':
            span_embeddings = self.reducer(span_embeddings)
            return span_embeddings, span_indices
        
        elif self.mode == 'classification':
            span_mask = torch.ones((span_embeddings.shape[0], span_embeddings.shape[1])).bool().to(self.device)
            output = self.classifier(spans=span_indices,
                                    span_mask=span_mask,
                                    span_embeddings=span_embeddings)
            return output, span_indices