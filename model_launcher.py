from model_scripts.data_utils.dataset_process import get_ind_sequence, dataset
from model_scripts.data_utils.label_helper import labels2ids, ids2labels
from model_scripts.task_trainers import GeneralTrainer
from model_scripts.span_classifier import NNERModel
from torch.utils.data import DataLoader
from torch import cuda
import pandas as pd
import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ModelLauncher:
    def __init__(self):
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        MAX_SEQ_LEN = 284
        MAX_SPAN_LEN = 20
        
        self.model = NNERModel(num_labels=len(ids2labels),
                        device=self.device,
                        max_seq_len=MAX_SEQ_LEN,
                        max_span_len=MAX_SPAN_LEN,
                        num_heads=None,
                        extractor_type='biaffine',
                        mode='classification',
                        extractor_use_gcn=True)
        _ = self.model.to(self.device)
        check_path = 'data/all_data/general_checkpoints/biaffine_gcn_checkpoint.pth.tar'
        checkpoint = torch.load(check_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.nner_trainer = GeneralTrainer(model=self.model,
                                      device = self.device,
                                      ids2labels=ids2labels,
                                      labels2ids=labels2ids,
                                      logger_path=None
                                        )
        triangle_mask = (torch.triu(
                            torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN), diagonal=0) - torch.triu(
                            torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN), diagonal=MAX_SPAN_LEN)).bool()
        self.span_indices = triangle_mask.nonzero().unsqueeze(0).expand(
                        1, -1, -1)[0].to(self.device)

    def form_data(self, text: str):
        data = pd.DataFrame({'TextID':[0], 'SentID': [0], 'Contents':[text]})
        self.data_inds = get_ind_sequence(data)
        data_ds = dataset(data, max_len=284)

        params = {'batch_size': 1,
                'shuffle': False,
                'num_workers': 0
                }
        self.data_loader = DataLoader(data_ds, **params)
        self.text = text

    def predict(self):
        return self.nner_trainer(self.data_loader, self.data_inds, mode='infer')[0]
    
    def get_prediction(self, preds):
        data_true_inds = self.data_inds.Indices.iloc[0]
        prediction_dict = {'Span':[], 'Class':[]}
        for ne_loc in range(len(self.span_indices)):
            if preds[ne_loc] != 0:
                token_loc = self.span_indices[ne_loc]
                lb_token = token_loc[0]
                lb_pos = data_true_inds[lb_token][0]

                rb_token = token_loc[1]
                rb_pos = data_true_inds[rb_token][1]

                prediction_dict['Span'].append(self.text[lb_pos:rb_pos])
                prediction_dict['Class'].append(ids2labels[preds[ne_loc]])
        
        return pd.DataFrame(prediction_dict)
    
    def __call__(self, text: str):
        self.form_data(text)
        raw_preds = self.predict()
        prediction_df = self.get_prediction(raw_preds)
        return prediction_df


