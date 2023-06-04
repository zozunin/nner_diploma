class Evaluator:

    def __init__(self, ids2labels):
        if 0 in ids2labels:
            _ = ids2labels.pop(0)
        self.ids2labels = ids2labels
        self.num_types = len(self.ids2labels)

    def evaluate(self, y_true, y_pred, mode='general'):

        # y_true, y_pred передаются батчами/списками

        sub_tp, sub_fn, sub_fp = [0] * self.num_types, [0] * self.num_types, [0] * self.num_types

        for gold_example, pred_example in zip(y_true, y_pred):
            gold_ners = set([(idx, int(ent)) for idx, ent in enumerate(gold_example) if ent != 0])
            pred_ners = set([(idx, int(ent)) for idx, ent in enumerate(pred_example) if ent != 0])

            for i in range(self.num_types):

                if mode == 'general':
                    cond = i+1 not in [24, 28, 29]
                elif mode == 'fewshot':
                    cond = i+1 in [24, 28, 29]
                elif mode == 'only_fs':
                    cond = 1

                if cond:
                    sub_gm = set((idx, ent) for idx, ent in gold_ners if ent == i+1)
                    sub_pm = set((idx, ent) for idx, ent in pred_ners if ent == i+1)
                    sub_tp[i] += len(sub_gm & sub_pm)
                    sub_fn[i] += len(sub_gm - sub_pm)
                    sub_fp[i] += len(sub_pm - sub_gm)

        #logger.info("****************SUB NER TYPES********************")
        print("****************SUB NER TYPES********************")
        if mode == 'general':
            #logger.info("****************GENERAL********************")
            print("****************GENERAL********************")
        elif mode == 'fewshot':
            #logger.info("*******************FEWSHOT***********************")
            print("*******************FEWSHOT***********************")

        f1_scores_list = []
        for i in range(self.num_types):

            if mode == 'general':
                cond = i+1 not in [24, 28, 29]
            elif mode == 'fewshot':
                cond = i+1 in [24, 28, 29]
            elif mode == 'only_fs':
                    cond = 1

            if cond:
                sub_r = 0 if sub_tp[i] == 0 else float(sub_tp[i]) / (sub_tp[i] + sub_fn[i])
                sub_p = 0 if sub_tp[i] == 0 else float(sub_tp[i]) / (sub_tp[i] + sub_fp[i])
                sub_f1 = 0 if sub_p == 0 else 2.0 * sub_r * sub_p / (sub_r + sub_p)
                f1_scores_list.append(sub_f1)
                #logger.info("{} F1: {:.5f}%".format(self.ids2labels[i+1], sub_f1 * 100))
                #logger.info("{} Recall: {:.5f}%".format(self.ids2labels[i+1], sub_r * 100))
                #logger.info("{} Precision: {:.5f}%".format(self.ids2labels[i+1], sub_p * 100))
                print("{} F1: {:.5f}%".format(self.ids2labels[i+1], sub_f1 * 100))
                print("{} Recall: {:.5f}%".format(self.ids2labels[i+1], sub_r * 100))
                print("{} Precision: {:.5f}%".format(self.ids2labels[i+1], sub_p * 100))

        if mode == 'general':
            denom = self.num_types-3
        elif mode == 'fewshot':
            denom = 3
        elif mode == 'only_fs':
            denom = 3
        macro_f1 = sum([each for i, each in enumerate(f1_scores_list)]) \
            / float(denom)
        
        #logger.info("Macro F1: {:.5f}%".format(macro_f1 * 100))
        print("Macro F1: {:.5f}%".format(macro_f1 * 100))

        return macro_f1