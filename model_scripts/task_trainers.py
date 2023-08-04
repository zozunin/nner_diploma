from .data_utils.dataset_process import get_batch_labels
from sklearn.neighbors import KNeighborsClassifier
from .data_utils.data_eval import Evaluator
from .loss_func import *
from tqdm import tqdm
from torch import nn
import logging
import random
import torch

class GeneralTrainer(nn.Module):

    """
    Трейнер для обучения и инференса на общих классах без few-shot экспериментов.

    Attributes
    ----------
    model : 'torch.nn.Module', required.
        Инициализированная модель для обучения/инференса
    device : str, required.
        'cuda' или 'cpu'
    ids2labels : dict, required.
        Маппинг индексов к меткам классов.
    labels2ids : dict, required.
        Маппинг меток классов к числовым обозначениям.
    val_mode : str, optional (default = 'comb').
        Режим оценивания работы модели. При инференсе не используется.
        Доступны варианты:
            'comb' - оценивает классификатор по всем классам, по умолчанию.
            'fewshot' - оценивает классификатор только по few-shot классам.
            'general' - оценивает классификатор только по классам, хорошо
                        представленным в обучении.
    logger_path : str, optional (default = None).
        При логировании в файл требуется указать путь до него.
    optimizer : 'torch.optim', optional (default = None).
        Оптимизатор. При инференсе не требуется указание.
    scheduler : torch.optim.lr_scheduler, optional (default = None).
        Планировщий. При инференсе не требуется указание.
    """

    def __init__(self, model, device, ids2labels, labels2ids, val_mode='comb', logger_path=None, optimizer=None, scheduler=None):

        super().__init__()
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.entropy = nn.CrossEntropyLoss(ignore_index=-100)

        self.num_classes = len(ids2labels)
        self.labels2ids = labels2ids    

        # define Logger
        logger = logging.getLogger('NNER Model')
        logger.setLevel(logging.INFO)

        format_logger = logging.Formatter(
            '%(message)s')
        anyhandler_logger = logging.StreamHandler()
        if logger_path:
            anyhandler_logger= logging.FileHandler(logger_path)
        anyhandler_logger.setFormatter(format_logger)
        logger.addHandler(anyhandler_logger)

        self.logger = logger

        self.evaluator = Evaluator(ids2labels, self.logger)

        if val_mode not in ['comb', 'fewshot', 'general']:
            val_mode='comb'
        self.val_mode = val_mode

    def forward(self, data_loader, data_indices, labels_dataset=None, mode='train'):
         
            '''
            Parameters
            ----------
            data_loader : torch DataLoader, required.
                Данные в батчах.
            data_indices : pandas.DataFrame, required.
                Относительные положения токенов в предложении, полученные из разделения с помощью natasha
                (один ряд в таблице соответствует одному тексту)
            labels_dataset : pandas.DataFrame, optional (default = None).
                Позиции именованных сущностей и их классы из тренировочной, валидационной и тестовой выборок.
            mode : str, optional (default = 'train').
                Предусмотрены "train" для обучения, "val" для валидации и "infer" для предсказания без оценивания.

            Returns
            -------
                float
                    Общую оценку по f-мере в случае режима train или val.
                torch.tensor
                    Предсказания в случае режима infer.

            '''      
            self.predictions = []
            self.true_labels = []
            epoch_f_score = 0
            epoch_loss = 0

            grad_regulator = torch.no_grad

            if mode == 'train':
                self.model.train()
                grad_regulator = torch.enable_grad
            
            else:
                self.model.eval()
                
            with grad_regulator():
                for idx, batch in tqdm(enumerate(data_loader)):
                        if mode == 'train':
                            self.optimizer.zero_grad()


                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['mask'].to(self.device)
                        adj = batch['adj'].to(self.device)
                        predicted, self.span_indices = self.model(input_ids=input_ids,
                                                                  attention_mask=attention_mask,
                                                                  adj_matrix=adj)
                        if mode != 'infer':
                            # get batch labels
                            batch_addresses = batch['address']
                            class_labels = get_batch_labels(batch_addresses,
                                                            labels_dataset,
                                                            data_indices,
                                                            self.span_indices[0],
                                                            self.labels2ids)
                            class_labels = class_labels.to(self.device)

                            self.true_labels += class_labels.tolist()

                        abs_labels = predicted.max(2).indices # для f_score
                        self.predictions += abs_labels.tolist()
                        
                        if mode == 'train':
                            # преобразование для вычисления ошибки
                            predicted = predicted.view(-1, self.num_classes)
                            class_labels = class_labels.view(-1)

                            active_tokens = class_labels.view(-1) != -100
                            class_labels = class_labels[active_tokens==1]
                            predicted = predicted[active_tokens==1]

                            ce_loss = self.entropy(predicted, class_labels)
                            loss = ce_loss

                            epoch_loss += loss.item()
                                
                        if mode == 'train':
                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()
            
                if mode == 'infer':
                    return self.predictions

                self.logger.info(f'RESULTS FOR MODE {mode.upper()}')
                if self.val_mode == 'comb':
                    epoch_f_score = self.evaluator.evaluate(self.true_labels, self.predictions, mode='all_ners')
                elif self.val_mode == 'fewshot':
                    epoch_f_score = self.evaluator.evaluate(self.true_labels, self.predictions, mode='fewshot')
                elif self.val_mode == 'general':
                    epoch_f_score = self.evaluator.evaluate(self.true_labels, self.predictions, mode='general')   
                
                if mode == 'train':
                    epoch_loss = epoch_loss / len(data_loader)
                    (f'Loss per epoch: {epoch_loss}')
                    self.logger.info(f'Loss per epoch: {epoch_loss}\n')

            return epoch_f_score
    


class FSTrainer(nn.Module):

    """
    Трейнер для few-shot подзадачи с  использованием Circle Loss
    Параметры для инициализации и вызова аналогичны GeneralTrainer

    Экспериментальная модель, под инференс не приспособлена
    """

    def __init__(self, model, device, ids2labels, labels2ids, logger = None, optimizer=None, scheduler=None):
        super().__init__()
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.circle_loss = CircleLoss(m=0.25, gamma=64)

        self.num_classes = len(ids2labels)
        self.labels2ids = labels2ids

        self.evaluator = Evaluator(ids2labels, logger)
        self.logger = logger
    
    def init_knn(self, X, y, k):
        self.nclassifier = KNeighborsClassifier(n_neighbors=k)
        self.nclassifier.fit(X, y)

    def forward(self, data_loader, data_indices, labels_dataset, mode='train'):

            epoch_loss = 0
            thres = 0.75
            # val stats
            tp=0
            fn=0
            fp=0

            grad_regulator = torch.no_grad
            spans = []
            labels = []
            self.preds = []
            self.true = []

            if mode == 'train':
                self.model.train()
                grad_regulator = torch.enable_grad
            
            else:
                self.model.eval()

            with grad_regulator():
                for idx, batch in tqdm(enumerate(data_loader)):
                        if mode == 'train':
                            self.optimizer.zero_grad()

                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['mask'].to(self.device)
                        adj = batch['adj'].to(self.device)

                        span_embeddings, span_indices = self.model(input_ids=input_ids,
                                                                   attention_mask=attention_mask,
                                                                   adj_matrix=adj)
                        
                        if mode in ['train', 'dev', 'get_spans', 'test']:
                            # get batch labels
                            batch_addresses = batch['address']
                            class_labels = get_batch_labels(batch_addresses,
                                                            labels_dataset,
                                                            data_indices,
                                                            span_indices[0],
                                                            self.labels2ids)
                            class_labels = class_labels.to(self.device)
                        
                        if mode in ['train', 'dev', 'get_spans']:
                            # блок отбора для круговой ошибки
                            chosen_labels = class_labels.view(-1).detach().clone()
                            # 0 - для спэнов, которые не являются искомыми сущностями
                            # -100 - для маскирования
                            chosen_labels[class_labels.view(-1) <= 0] = -100
                            negative_labels = class_labels.view(-1) == 0

                            negative_inds = negative_labels.nonzero().view(-1).tolist()
                            found_classes = len(class_labels[class_labels > 0])

                            if mode == 'train':
                                num = 20
                            elif mode == 'dev':
                                num = 2
                            elif mode == 'get_spans':
                                num = 1 # добавляем в каждом батче только по примеру трешовых классов

                            rand_negative_inds = random.sample(negative_inds, num)

                            for i in range(len(chosen_labels)):
                                if i in rand_negative_inds:
                                    chosen_labels[i] = 0
                            circle_labels = chosen_labels[chosen_labels != -100]
                            circle_spans = span_embeddings[0][chosen_labels != -100]

                        
                        if mode == 'train':
                            # circle loss
                            inp_sp, inp_sn = convert_label_to_similarity(circle_spans, circle_labels)
                            ccl_loss = self.circle_loss(inp_sp, inp_sn)
                            loss = ccl_loss

                            epoch_loss += loss.item()

                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()
                        
                        elif mode == 'dev':
                            for pred1 in range(len(circle_labels)):
                                for pred2 in range(len(circle_labels)):
                                    if pred1 == pred2:
                                        continue
                                    true_type = circle_labels[pred1] == circle_labels[pred2]
                                    
                                    pred_type = torch.sum(circle_spans[pred1]*circle_spans[pred2]) > thres

                                    if true_type and pred_type:
                                        tp += 1
                                    elif true_type and not pred_type:
                                        fn += 1
                                    elif not true_type and pred_type:
                                        fp += 1
                        
                        elif mode == 'get_spans':
                            spans += circle_spans.tolist()
                            labels += circle_labels.tolist()
                        elif mode == 'test':
                            batch_preds = self.nclassifier.predict(span_embeddings[0].tolist())
                            self.true += class_labels[0].tolist()
                            self.preds += batch_preds.tolist()
                
                if self.logger != None:
                    logging_prefix = self.logger.info
                else:
                    logging_prefix = print
                                
                if mode == 'train':
                    epoch_loss = epoch_loss / len(data_loader)
                    logging_prefix(f'Loss per epoch: {epoch_loss}')
                elif mode == 'dev':
                    recall = 0 if tp == 0 else tp / (tp+fn)
                    precision = 0 if tp == 0 else tp / (tp+fp)
                    f1 = 0 if precision == 0 else 2.0*recall*precision / (recall+precision)
                    logging_prefix("Mention Recall: {:.5f}%".format(recall * 100))
                    logging_prefix("Mention Precision: {:.5f}%".format(precision * 100))
                    logging_prefix("Mention F1: {:.5f}%".format(f1 * 100))
                elif mode == 'get_spans':
                    return spans, labels
                elif mode == 'test':
                    _ = self.evaluator.evaluate([self.true], [self.preds], mode='only_fs')
            
            return epoch_loss