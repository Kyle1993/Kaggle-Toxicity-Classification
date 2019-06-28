import os,sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import pickle
import gc
import time
from tqdm import tqdm,tqdm_notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader,TensorDataset
torch.cuda.set_device(0)

import re
gc.enable()
from nltk.tokenize import TweetTokenizer
from keras.preprocessing import text, sequence
import emoji
import collections

import shutil
package_dir = "../input/pp-bert"
sys.path.insert(0, package_dir)

from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam

import warnings
warnings.filterwarnings('ignore')

class JigsawEvaluator:

    def __init__(self, y_true, y_identity, power=-5, overall_model_weight=0.25):
        self.y = (y_true >= 0.5).astype(int)
        self.y_i = (y_identity >= 0.5).astype(int)
        self.n_subgroups = self.y_i.shape[1]
        self.power = power
        self.overall_model_weight = overall_model_weight

    @staticmethod
    def _compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def _compute_subgroup_auc(self, i, y_pred):
        mask = self.y_i[:, i] == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bpsn_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bnsp_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y != 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def compute_bias_metrics_for_model(self, y_pred):
        records = np.zeros((3, self.n_subgroups))
        for i in range(self.n_subgroups):
            records[0, i] = self._compute_subgroup_auc(i, y_pred)
            records[1, i] = self._compute_bpsn_auc(i, y_pred)
            records[2, i] = self._compute_bnsp_auc(i, y_pred)
        return records

    def _calculate_overall_auc(self, y_pred):
        return roc_auc_score(self.y, y_pred)

    def _power_mean(self, array):
        total = sum(np.power(array, self.power))
        return np.power(total / len(array), 1 / self.power)

    def get_final_metric(self, y_pred):
        bias_metrics = self.compute_bias_metrics_for_model(y_pred)
        bias_score = np.average([
            self._power_mean(bias_metrics[0]),
            self._power_mean(bias_metrics[1]),
            self._power_mean(bias_metrics[2])
        ])
        overall_score = self.overall_model_weight * self._calculate_overall_auc(y_pred)
        bias_score = (1 - self.overall_model_weight) * bias_score
        return overall_score + bias_score

def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -= 2
    all_tokens = []
    longer = 0
    for text in tqdm_notebook(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

# global variable
cv = 3
max_len = 220
train_num = 1804874
test_num = 97320
batch_size = 64

identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
aux_columns = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
bert_vocab_oath = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/vocab.txt'
test_csv_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
train_csv_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
processed_data_path = '../input/bert-get-data-lower/bert_preprocess_data.pkl'
config_path = '../input/toxicity-bert-config/bert_50config.json'
models_path = '../input/bert-train-2epoh-fork2'

# load data from pkl

with open('../input/toxicity-inference-pkl/kfold_3.pkl','rb') as f:
    kfold = pickle.load(f)

with open(processed_data_path,'rb') as f:
    processed_data = pickle.load(f)
x_train = processed_data['x_train']
y_train = processed_data['y_train']
weights = processed_data['weights']
y_true = processed_data['y_true']
y_identity = processed_data['y_identity']
loss_sacle = 1.0 / weights.mean()

# forget do this in 'get-data-lower'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)
x_test = processed_data['x_test']
x_test = convert_lines(x_test.fillna("DUMMY_VALUE"),max_len,tokenizer)


del processed_data
del tokenizer
gc.collect()
print('Data Loaded!')

# inference & get feature
bert_config = BertConfig(config_path)
# remeber to change feature_num
model = BertForSequenceClassification(bert_config, num_labels=7, feature_num=50)

validate = True

for fold in [1, ]:
    print('Fold{}:'.format(fold))

    validate_idx = kfold[fold][1]
    train_idx = kfold[fold][0]
    #     train_idx = list(range(nrows))[:int(nrows*0.8)]
    #     validate_idx = list(range(nrows))[int(nrows*0.8):]


    model.load_state_dict(torch.load(os.path.join(models_path, 'bert_fold{}.bin'.format(fold))))
    model.cuda()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    train_pred_fold = []
    test_pred_fold = []
    train_feature_fold = []
    test_feature_fold = []
    # on train_set
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.long), )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    for x in tqdm_notebook(train_loader):
        max_len = torch.max(torch.sum((x[0] != 0), 1))
        x = x[0][:, :max_len].long().cuda()
        pred, feature = model(x, attention_mask=(x > 0).cuda(), )
        train_feature_fold.append(feature)
        train_pred_fold.append(torch.sigmoid(pred[:, 0]))

    train_feature_fold = torch.cat(train_feature_fold, dim=0)
    train_feature_fold = train_feature_fold.cpu().numpy()
    train_pred_fold = torch.cat(train_pred_fold, dim=0)
    train_pred_fold = train_pred_fold.cpu().numpy()
    validate_pred_fold = train_pred_fold[validate_idx]

    if validate:
        train_pred_fold = train_pred_fold[train_idx]
        jigsawevaluator_train = JigsawEvaluator(y_true[train_idx], y_identity[train_idx])
        train_score = jigsawevaluator_train.get_final_metric(train_pred_fold)

        jigsawevaluator_validate = JigsawEvaluator(y_true[validate_idx], y_identity[validate_idx])
        validate_score = jigsawevaluator_validate.get_final_metric(validate_pred_fold)
        print('train_score:{:.6f}\tvalidate_scvore:{:.6f}'.format(train_score, validate_score))
        with open('log.txt', 'w') as f:
            f.write('train_score:{:.6f}\tvalidate_scvore:{:.6f}'.format(train_score, validate_score))

    validate_pred_fold = pd.DataFrame({'id': validate_idx, 'prediction': validate_pred_fold})
    validate_pred_fold.to_csv('bert_validate_pred_fold{}.csv'.format(fold))

    del train_dataset
    del train_loader
    gc.collect()

    # on test_set
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.long), )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for x in tqdm_notebook(test_loader):
        max_len = torch.max(torch.sum((x[0] != 0), 1))
        x = x[0][:, :max_len].long().cuda()
        pred, feature = model(x, attention_mask=(x > 0).cuda(), )
        test_feature_fold.append(feature)
        test_pred_fold.append(torch.sigmoid(pred[:, 0]))

    test_pred_fold = torch.cat(test_pred_fold, dim=0)
    test_pred_fold = test_pred_fold.cpu().numpy()
    test_feature_fold = torch.cat(test_feature_fold, dim=0)
    test_feature_fold = test_feature_fold.cpu().numpy()

    with open('bert_test_pred_fold{}.pkl'.format(fold), 'wb') as f:
        pickle.dump(test_pred_fold, f)
    with open('bert_feature_fold{}.pkl'.format(fold), 'wb') as f:
        pickle.dump({'train': train_feature_fold, 'test': test_feature_fold}, f)

    del test_dataset
    del test_loader
    gc.collect()


