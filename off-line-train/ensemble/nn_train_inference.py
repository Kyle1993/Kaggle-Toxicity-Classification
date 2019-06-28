import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import time
import numpy as np
import pandas as pd
import pickle
import gc
import sys, os
from tqdm import tqdm
from datetime import datetime

from global_variable import *
import utils
from models import Toxicity_NN

start_time = datetime.now()

data_train, label_train, data_test, test_id, weight, loss_scale, y_identity = utils.get_data(normalize=True, aux=True)
print('Features Loaded! Use Time:', datetime.now() - start_time)

with open(kfold_path, 'rb') as f:
    kfold = pickle.load(f)

feature_num = data_train.shape[2]
target_num = label_train.shape[1]

config = {'batch_size':512,
          'epoch_num':1,
          'gpu':0,
          'lr':1e-4,
          # 'lr_decay':0.1,
          'weight_decay':0.2}

print('Start Training & Inference...')

timestr = time.strftime('%m%d-%H%M')
if not os.path.exists('saved_models/NN_{}'.format(timestr)):
    os.mkdir('saved_models/NN_{}'.format(timestr))

log = timestr
log += '\n'
log += utils.config2str(config)
log += '\n'


train_preds = []
test_preds = []
cv_score = {'train':0,'validate':0}
for fold in range(cv):
    print('Fold{}:'.format(fold))
    train_idx = kfold[fold][0]
    validate_idx = kfold[fold][1]

    x_test = data_test[fold]

    x_validate = data_train[fold][validate_idx]
    y_validate = label_train[validate_idx]
    w_validate = weight[validate_idx]

    x_train = data_train[fold][train_idx]
    y_train = label_train[train_idx]
    w_train = weight[train_idx]

    model = Toxicity_NN(feature_num, target_num)
    model.cuda(config['gpu'])

    optimizer = Adam(model.parameters(), lr=config['lr'],weight_decay=config['weight_decay'])
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: config['lr_decay'] ** epoch)

    # train
    model.train()
    for epoch in range(config['epoch_num']):
        # scheduler.step()

        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float), torch.tensor(w_train, dtype=torch.float))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

        for i, (x, y, w) in enumerate(tqdm(train_loader)):
            x = utils.totensor(x, config['gpu']).float()
            y = utils.totensor(y, config['gpu']).float()
            w = utils.totensor(w, config['gpu']).float()

            pred = model(x)

            loss1 = F.binary_cross_entropy(pred[:, 0], y[:, 0], w)
            loss2 = F.binary_cross_entropy(pred[:, 1:], y[:, 1:])
            loss = loss_scale * loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluate
    model.eval()
    pred_train = []
    pred_validate = []
    pred_test = []

    with torch.no_grad():
        # on train set
        jigsawevaluator_train = utils.JigsawEvaluator(y_train[:,0], y_identity[train_idx])
        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float), )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)

        for x, in tqdm(train_loader):
            x = utils.totensor(x, config['gpu']).float()
            pred = model(x)
            pred_train.append(pred[:, 0])

        pred_train = torch.cat(pred_train, dim=0)
        pred_train = pred_train.cpu().numpy()
        train_score = jigsawevaluator_train.get_final_metric(pred_train)

        # on validate set
        jigsawevaluator_validate = utils.JigsawEvaluator(y_validate[:,0], y_identity[validate_idx])
        validate_dataset = TensorDataset(torch.tensor(x_validate, dtype=torch.float), )
        validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=config['batch_size'], shuffle=False)

        for x, in tqdm(validate_loader):
            x = utils.totensor(x, config['gpu']).float()
            pred = model(x)
            pred_validate.append(pred[:, 0])

        pred_validate = torch.cat(pred_validate, dim=0)
        pred_validate = pred_validate.cpu().numpy()
        validate_score = jigsawevaluator_validate.get_final_metric(pred_validate)

        # inference on test set
        test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float), )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

        for x, in tqdm(test_loader):
            x = utils.totensor(x, config['gpu']).float()
            pred = model(x)
            pred_test.append(pred[:, 0])

        pred_test = torch.cat(pred_test, dim=0)
        pred_test = pred_test.cpu().numpy()

        test_preds.append(pred_test)
        train_preds.append(pd.DataFrame({'id': validate_idx, 'prediction': pred_validate}))
        print('train score:{}\nvalidate score:{}'.format(train_score, validate_score))
        log += 'fold{}:\ttrain score:{:.6f}\tvalidate score:{:.6f}\n'.format(fold,train_score, validate_score)
        cv_score['train'] += train_score
        cv_score['validate'] += validate_score

    # save model
    torch.save(model.cpu().state_dict(), 'saved_models/NN_{}/fold{}.pth'.format(timestr,fold))

log += '\nmean train:{:.6f}\tmean validate:{:.6f}\n'.format(cv_score['train']/cv,cv_score['validate']/cv)
with open('saved_models/NN_{}/log'.format(timestr), 'w') as f:
    f.write(log)

test_preds = np.asarray(test_preds)
test_preds = test_preds.mean(axis=0)
test_preds = pd.DataFrame({'id': test_id, 'prediction': test_preds})
test_preds.set_index(keys='id',drop=True,inplace=True)
assert test_preds.shape[0] == test_num
test_preds.to_csv('saved_models/NN_{}/submission.csv'.format(timestr))

train_preds = pd.concat(train_preds,ignore_index=True)
train_preds = train_preds.sort_values(by='id')
assert train_preds['id'].values.tolist() == list(range(train_num))
train_preds.to_csv('saved_models/NN_{}/train_preds.csv'.format(timestr))





