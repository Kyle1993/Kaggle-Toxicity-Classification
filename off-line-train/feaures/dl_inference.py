import numpy as np
import pandas as pd
import os,sys
import time
import gc
import random
from tqdm import tqdm
from keras.preprocessing import sequence,text
import torch
from torch.utils.data import DataLoader
import pickle
from datetime import datetime

from global_variable import *
from dataset_helper import ToxicityTestDataset,SequenceBucketCollator
import models
import utils

with open(toxicity_embadding_path,'rb') as f:
    embedding = pickle.load(f)
embedding = embedding['embedding']

with open(processed_data_path,'rb') as f:
    data = pickle.load(f)

with open(tokenizer_path,'rb') as f:
    tokenizer = pickle.load(f)

with open(kfold_path,'rb') as f:
    kfold = pickle.load(f)

def load_config(log_path):
    config = {}
    with open(log_path,'r') as f:
        for line in f.readlines():
            if ':' in line and not('Fold' in line):
                k = line.split(':')[0].strip()
                if k in ['output_feature_num','num_targets','gpu','epoch_num','epoch_num','embedding_dim']:
                    v = int(line.split(':')[1].strip())
                elif k in ['model_name','text']:
                    v = line.split(':')[1].strip()
                else:
                    v = float(line.split(':')[1].strip())
                config[k] = v
    return config

start_time = datetime.now()

nrows = None
print('Loading Data...')
test = pd.read_csv(test_csv_path, nrows=nrows)
test_ids = test['id']

x_train = data['x_train']
x_test = data['x_test']

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

del data
del test
gc.collect()


print('Start Inference...')
model_floder = '/data2/jianglibin/toxicity/DL/Toxicity_BiLSTMSelfAttention_0623-1059'
print(model_floder)
config = load_config(os.path.join(model_floder,'log'))
config['embedding_size'] = embedding.shape
config['gpu'] = 1
config['batch_size'] = 256
assert config['model_name'] in model_floder

train_features = []
train_preds = []
test_features = []
test_preds = []
for fold in range(cv):
    print('Fold{}:'.format(fold))
    validate_idx = kfold[fold][1]
    # load model
    model = models.load_model(**config)
    model_dict = torch.load('{}/fold{}.pth'.format(model_floder,fold))
    model_dict['embedding.weight'] = torch.tensor(embedding)
    model.load_state_dict(model_dict)

    if config['gpu'] >= 0:
        model.cuda_(config['gpu'])
    model.eval()

    # on train set
    train_dataset = ToxicityTestDataset(x_train,np.zeros((train_num,)))
    train_dataloader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=False,num_workers=1,collate_fn=SequenceBucketCollator())

    train_feature_fold = []
    train_pred_fold = []

    with torch.no_grad():
        for x,_ in tqdm(train_dataloader):
            x = utils.totensor(x,config['gpu']).long()

            pred,feature = model(x)
            pred = pred[:,0]
            train_feature_fold.append(feature)
            train_pred_fold.append(pred)

    train_feature_fold = torch.cat(train_feature_fold,dim=0)
    train_features.append(train_feature_fold.cpu().numpy())

    train_pred_fold = torch.cat(train_pred_fold, dim=0)
    train_pred_fold = train_pred_fold.cpu().numpy()
    validate_pred_fold = train_pred_fold[validate_idx]
    train_preds.append(pd.DataFrame({'id':validate_idx,'prediction':validate_pred_fold}))

    # on test set
    test_dataset = ToxicityTestDataset(x_test,test_ids)
    test_dataloader = DataLoader(test_dataset,batch_size=config['batch_size'],shuffle=False,num_workers=1,collate_fn=SequenceBucketCollator())

    test_pred_fold = []
    test_feature_fold = []
    test_id_fold = []

    with torch.no_grad():
        for x,ids in tqdm(test_dataloader):
            x = utils.totensor(x,config['gpu']).long()

            pred,feature = model(x)
            pred = pred[:,0]

            test_id_fold.extend(ids)
            test_pred_fold.append(pred)
            test_feature_fold.append(feature)

    assert test_id_fold == test_ids.tolist()
    test_pred_fold = torch.cat(test_pred_fold,dim=0)
    test_preds.append(test_pred_fold.cpu().numpy())
    test_feature_fold = torch.cat(test_feature_fold,dim=0)
    test_features.append(test_feature_fold.cpu().numpy())

train_preds = pd.concat(train_preds,ignore_index=True)
train_preds = train_preds.sort_values(by='id')
assert train_preds['id'].values.tolist() == list(range(train_num))
train_preds.to_csv('{}/train_preds.csv'.format(model_floder))

test_pred = np.asarray(test_preds).mean(axis=0)
submit = pd.DataFrame({'id': test_ids, 'prediction': test_pred})
submit.set_index('id',drop=True,inplace=True)
submit.to_csv('{}/submission.csv'.format(model_floder))

train_features = np.asarray(train_features)
test_features = np.asarray(test_features)

with open('{}/features.pkl'.format(model_floder),'wb') as f:
    pickle.dump({'train':train_features,'test':test_features},f)

print('All Time Used:',datetime.now()-start_time)




















