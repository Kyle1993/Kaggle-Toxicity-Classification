import numpy as np
import pandas as pd
import os,sys
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import pickle
import gc

from global_variable import *
from dataset_helper import ToxicityDataset,SequenceBucketCollator
import models
import utils

def config2str(config):
    string = '\n'
    for k,v in sorted(config.items(),key=lambda x:x[0]):
        string += '{}:\t{}\n'.format(k,v)
    return string

with open(kfold_path,'rb') as f:
    kfold = pickle.load(f)

with open(toxicity_embadding_path,'rb') as f:
    embedding = pickle.load(f)
embedding = embedding['embedding']

with open(processed_data_path,'rb') as f:
    train_data = pickle.load(f)

with open(tokenizer_path,'rb') as f:
    tokenizer = pickle.load(f)

config = {'gpu':1,
          'lr':1e-3,
          'lr_decay':0.6,
          'output_feature_num':50,
          'num_targets':len(aux_columns)+2,
          'batch_size':128,
          'validate_batch_size':256,
          'epoch_num':5,
          'model_name':'Toxicity_LSTM2',
          }

nrows = None
train = pd.read_csv(train_csv_path,nrows=nrows,)
print('Data Loaded!')

x_train = train_data['x_train']
x_train = tokenizer.texts_to_sequences(x_train)

y_aux_train = train[['target']+aux_columns]
y_train = np.where(train['target'] >= 0.5, 1, 0)
y_train = np.concatenate([y_train[:, np.newaxis], y_aux_train],axis=1)

# Overall
weights = np.ones((train.shape[0],)) / 4
# Subgroup
weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
# Background Positive, Subgroup Negative
weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) + (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
# Background Negative, Subgroup Positive
weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) + (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
loss_sacle = 1.0 / weights.mean()

y_true = train['target'].values
y_identity = train[identity_columns].values

del train
del train_data
del y_aux_train
del tokenizer
gc.collect()

print('Start Training...')

timestr = time.strftime('%m%d-%H%M')
if not os.path.exists('DL/{}_{}'.format(config['model_name'], timestr)):
    os.mkdir('DL/{}_{}'.format(config['model_name'], timestr))

log = timestr
log += '\n'
log += config2str(config)
log += '\n'

print('Congig:')
print(log)

for fold in [0,1,2]:
    print('Fold{}:'.format(fold))
    print('Training ...'.format(fold))
    train_idx = kfold[fold][0]
    validate_idx = kfold[fold][1]

    model = models.load_model(embedding_matrix=embedding,embedding_size=embedding.shape,**config)
    if config['gpu'] >= 0:
        model.cuda_(config['gpu'])
    model.train()
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = LambdaLR(optimizer, lambda epoch: config['lr_decay'] ** epoch)

    # training
    for epoch in range(config['epoch_num']):
        scheduler.step()

        train_dataset = ToxicityDataset(x_train,y_train,weights,train_idx)
        train_dataloader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True,num_workers=1,collate_fn=SequenceBucketCollator())

        for batch in tqdm(train_dataloader):
            x = utils.totensor(batch[0], config['gpu']).long()
            y = utils.totensor(batch[1], config['gpu']).float()
            w = utils.totensor(batch[2], config['gpu']).float()

            pred,_ = model(x)

            loss1 = F.binary_cross_entropy(pred[:, 0], y[:, 0], w)
            loss2 = F.binary_cross_entropy(pred[:, 1:], y[:, 1:])
            loss = loss_sacle * loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    del train_dataset
    del train_dataloader
    gc.collect()

    # evaluating
    print('Evaluating...')
    model.eval()

    pred_train = []
    pred_validate = []
    with torch.no_grad():
        # on train_set
        jigsawevaluator_train = utils.JigsawEvaluator(y_true[train_idx], y_identity[train_idx])
        train_dataset = ToxicityDataset(x_train, y_train, weights, train_idx)
        train_dataloader = DataLoader(train_dataset, batch_size=config['validate_batch_size'], shuffle=False,collate_fn=SequenceBucketCollator())

        for batch in tqdm(train_dataloader):
            x = utils.totensor(batch[0],config['gpu']).long()
            pred,_ = model(x)
            pred_train.append(pred[:,0])

        pred_train = torch.cat(pred_train,dim=0)
        pred_train = pred_train.cpu().numpy()

        train_score = jigsawevaluator_train.get_final_metric(pred_train)

        # on validate_set
        jigsawevaluator_validate = utils.JigsawEvaluator(y_true[validate_idx], y_identity[validate_idx])
        validate_dataset = ToxicityDataset(x_train, y_train, weights, validate_idx)
        validate_dataloader = DataLoader(validate_dataset, batch_size=config['validate_batch_size'], shuffle=False,collate_fn=SequenceBucketCollator())

        for batch in tqdm(validate_dataloader):
            x = utils.totensor(batch[0], config['gpu']).long()
            pred, _ = model(x)
            pred_validate.append(pred[:, 0])

        pred_validate = torch.cat(pred_validate, dim=0)
        pred_validate = pred_validate.cpu().numpy()

        validate_score = jigsawevaluator_validate.get_final_metric(pred_validate)

    print('train scroe:{}\nvalidate score:{}\n'.format(train_score,validate_score))
    log += 'Fold{}\ttrain scroe:{:.6f}\tvalidate score:{:.6f}\n'.format(fold,train_score,validate_score)

    # save model
    model_dict = model.cpu().state_dict()
    del model_dict['embedding.weight']
    torch.save(model_dict, 'DL/{}_{}/{}_fold{}.pth'.format(config['model_name'], timestr, config['model_name'], fold))


    with open('DL/{}_{}/log'.format(config['model_name'], timestr),'w') as f:
        f.write(log)





















