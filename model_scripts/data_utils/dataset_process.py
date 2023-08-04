from natasha import NewsSyntaxParser, NewsEmbedding
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
from typing import List, Dict
from razdel import tokenize
from torch import Tensor
import pandas as pd
import numpy as np
import torch


class dataset(Dataset):
    '''
    Returns
    -------
    item : 'dict'
         * 'input_ids': токенизированное предложение, закодированное id токенов
         * 'mask': последовательность 0, 1, где 1 - позиции без паддинга
         * 'adj': матрица синтаксической смежности (для AGGCN)
         * 'address': указатель на id новости и номер предложения для получения из размеченных данных сущностей и их классов
    '''
    def __init__(self, texts, max_len=256):

        self.len = len(texts)
        self.data = texts
        self.max_len = max_len
        self.bert_tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased')
        emb = NewsEmbedding()
        self.syntax_parser = NewsSyntaxParser(emb)

    def __getitem__(self, index):
        
        # Данные с полями TextID, SentID и Contents
        temp_text = tokenize(self.data['Contents'][index])
                
        tokenized_text = [tok.text for tok in temp_text][:self.max_len]

        input_ids = [0] * self.max_len
        attention_mask = [0] * self.max_len
        for idx, word in enumerate(tokenized_text):
            # один токен - одно слово
            input_ids[idx] = self.bert_tokenizer.encode(word,add_special_tokens=False)[0]
            attention_mask[idx] = 1

        markup = self.syntax_parser(tokenized_text).tokens
        # конструирование матрицы синтаксических связей
        adj_matrix = np.zeros((self.max_len, self.max_len))
        for tok in markup:
            i = int(tok.id)-1
            j = int(tok.head_id)-1
            if i < 0 or j < 0:
                continue
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1
        
        # сохраняем путь до нужных данных, чтобы при валидации найти нужную часть датафрейма
        # без необходимости сохраняться все в тензоре одного размера
        address_true_spans = [self.data['TextID'][index],
                              self.data['SentID'][index]]
        
        item = {'input_ids': torch.as_tensor(input_ids),
                'mask': torch.as_tensor(attention_mask),
                'adj': torch.as_tensor(adj_matrix),
                'address': address_true_spans}        
        return item

    def __len__(self):
        return self.len
    

def get_ind_sequence(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Возвращает последовательность токенов в виде индексов начала и конца каждого
    '''
    temp_df = {'TextID':[], 'SentID':[], 'Indices':[]}
    for row in range(len(dataframe)):
        textid = dataframe['TextID'].iloc[row]
        sentid = dataframe['SentID'].iloc[row]
        content = dataframe['Contents'].iloc[row]

        saved_indices = [[tok.start, tok.stop] for tok in tokenize(content)]
        temp_df['TextID'].append(textid)
        temp_df['SentID'].append(sentid)
        temp_df['Indices'].append(saved_indices)
    
    df = pd.DataFrame(temp_df)
    return df

def get_batch_labels(batch_addresses: List, spans_dataset: pd.DataFrame, indices_dataset: pd.DataFrame, span_indices: Tensor,
                     labels2ids: Dict):
    '''
    Вспомогательная функция для получения последовательности верных меток для отрезков в предложении
    '''

    batch_size = len(batch_addresses[0])

    span_number = span_indices.shape[0]

    span_indices = span_indices.tolist()

    batch_labels = np.ones((batch_size, span_number))*-100

    for batch_idx, i in enumerate(zip(batch_addresses[0], batch_addresses[1].detach().tolist())):
        textid = i[0]
        sentid = i[1]

        labels_df = list(spans_dataset[spans_dataset['TextID']==textid][spans_dataset['SentID']==sentid]['Spans'])
        readable_spans = []
        for ent in labels_df:
            temp = ent[1:-1].split(', ')
            readable_spans.append([int(temp[0]), int(temp[1]),labels2ids.get(temp[2][1:-1], 0)])
        
        
        true_indices = indices_dataset[indices_dataset['TextID']==textid][indices_dataset['SentID']==sentid]['Indices'].iloc[0]
        max_ind = true_indices[-1][-1]
        for idx, spn in enumerate(span_indices):
            if spn[0] <= max_ind or spn[1] <= max_ind:
                batch_labels[batch_idx][idx] = 0
            else:
                break

        for span in readable_spans:
            temp_span_ids = []
            memory = []
            for idx, spn in enumerate(true_indices):
                if spn[0] in range(span[0], span[1]+1) and spn[1] in range(span[0], span[1]+1):
                    if temp_span_ids == []:
                        temp_span_ids = [idx, idx]
                        memory = [spn[0], spn[1]]
                    else:
                        temp_span_ids[-1] = idx
                        memory[-1] = spn[1]
            if temp_span_ids != []:
                try:
                    place = span_indices.index(temp_span_ids)
                    batch_labels[batch_idx][place] = span[2]
                except:
                    pass
                
    return torch.from_numpy(batch_labels).type(torch.int64)