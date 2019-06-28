import os,sys

# # Installing Nvidia Apex
# ! pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a

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

from apex import amp
from apex.optimizers.fused_adam import FusedAdam
import shutil
package_dir = "../input/pp-bert"
sys.path.insert(0, package_dir)

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

import warnings
warnings.filterwarnings('ignore')

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
    print(longer)
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


class LenMatchBatchSampler(data.BatchSampler):
    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx][0] == 0)
            count_zeros = int(count_zeros / 64)
            if len(buckets[count_zeros]) == 0:  buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, "produced an inccorect number of batches. expected %i, but yielded %i" %(len(self), yielded)

def trim_tensors(tsrs):
    max_len = torch.max(torch.sum( (tsrs[0] != 0  ), 1))
    tsrs[0] = tsrs[0][:, :max_len]
    return tsrs

# global variable
fold = 2
cv = 3
max_len = 220
train_num = 1804874
output_model_file = "bert_fold{}.bin".format(fold)
identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
aux_columns = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
WORK_DIR = "../working/"
train_csv_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
test_csv_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
processed_data_path = '../input/bert-get-data-lower/bert_preprocess_data.pkl'
bert_vocab_oath = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/vocab.txt'

# Translate model from tensorflow to pytorch
convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(BERT_MODEL_PATH + 'bert_model.ckpt',
                                                                  BERT_MODEL_PATH + 'bert_config.json',
                                                                  WORK_DIR + 'pytorch_model.bin')

shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'bert_config.json')

# load data
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

train_idx = kfold[fold][0]
validate_idx = kfold[fold][1]
# nrows = 1000
# train_idx = list(range(nrows))[:int(nrows*0.8)]
# validate_idx = list(range(nrows))[int(nrows*0.8):]

x_validate = x_train[validate_idx]
y_validate = y_train[validate_idx]
w_validate = weights[validate_idx]
x_train = x_train[train_idx]
y_train = y_train[train_idx]
w_train = weights[train_idx]


del processed_data
gc.collect()
print('Data Loaded!')

# train model

epochs = 2
batch_size = 32
lr = 1.8e-5
lr_decay = 0.95
feature_num = 50
target_num = len(aux_columns) + 2

model = BertForSequenceClassification.from_pretrained(WORK_DIR, cache_dir=None, num_labels=target_num,
                                                      feature_num=feature_num)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.012},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = []
# for n,p in list(model.named_parameters()):
#     params_set = {'params':p,}
#     if 'embeddings' in n:
#         params_set['lr'] = lr*lr_decay**11
#     elif 'layer' in n:
#         i = int(re.findall(r'\d+',n)[0])
#         params_set['lr'] = lr*lr_decay**(11-i)
#     else:
#         params_set['lr'] = lr

#     if any(nd in n for nd in no_decay):
#         params_set['weight_decay'] = 0.0
#     else:
#         params_set['weight_decay'] = 0.01

#     optimizer_grouped_parameters.append(params_set)

num_train_optimization_steps = int(epochs * len(x_train) / batch_size)
optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.05, t_total=num_train_optimization_steps, )
model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
# optimizer = FusedAdam(optimizer_grouped_parameters, lr=2e-5, bias_correction=False, max_grad_norm=1.0)

timestr = time.strftime('%m%d-%H%M')

log = ''
log += '{}\n'.format(timestr)
log += 'BERT:{}\n'
log += 'epochs:{}\n'.format(epochs)
log += 'lr:{}\n'.format(lr)

global_step = 0
train_record = []
model.train()
for epoch in range(epochs):
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float),
                                  torch.tensor(w_train, dtype=torch.float))

    # fast training
    ran_sampler = data.RandomSampler(train_dataset)
    len_sampler = LenMatchBatchSampler(ran_sampler, batch_size=batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=len_sampler)

    for i, batch in enumerate(tqdm_notebook(train_loader)):
        tsrs = trim_tensors(batch)
        x, y, w = tuple(t.cuda() for t in tsrs)
        x = x.long()
        y = y.float()
        w = w.float()
        pred, _ = model(x, attention_mask=(x > 0).cuda(), )

        loss1 = F.binary_cross_entropy_with_logits(pred[:, 0], y[:, 0], w)
        loss2 = F.binary_cross_entropy_with_logits(pred[:, 1:], y[:, 1:])
        loss = loss1 * loss_sacle + loss2

        train_record.append([global_step, loss.item()])
        global_step += 1

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

torch.save(model.cpu().state_dict(), output_model_file)
with open('train_record_fold{}.pkl'.format(fold), 'wb') as f:
    pickle.dump(train_record, f)




