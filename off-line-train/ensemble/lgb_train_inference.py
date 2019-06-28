import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import pickle
import gc
import sys,os
from datetime import datetime
import time

from global_variable import *
import utils

start_time = datetime.now()

data_train,label_train,data_test,test_id,weight,loss_scale,y_identity = utils.get_data(normalize=True)
print('Features Loaded!')
print('Time:',datetime.now()-start_time)


with open(kfold_path,'rb') as f:
    kfold = pickle.load(f)

params = {
    'boosting_type': 'gbdt',  # 训练方式
    'objective': 'binary',    # 损失函数
    'metric':'auc',
    'num_leaves':32,
    'learning_rate': 0.01,
    'feature_fraction': 0.2,
    'bagging_fraction': 0.8,
    # 'min_sum_hessian_in_leaf': 1,
    'min_data_in_leaf': 10000,
    # # "bagging_freq": 1,
    'lambda_l1': 0.2,
    'lambda_l2': 0.2,
    'verbose': -1,
    'boost_from_average':False,
}

print('Start Training & Inference...')

timestr = time.strftime('%m%d-%H%M')
if not os.path.exists('saved_models/LGB_{}'.format(timestr)):
    os.mkdir('saved_models/LGB_{}'.format(timestr))

log = timestr
log += '\n'
log += utils.config2str(params)
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

    # train
    lgb_train = lgb.Dataset(x_train, y_train,weight=w_train, free_raw_data=False)
    bst = lgb.train(params,lgb_train,num_boost_round=100,)
    bst.save_model('saved_models/LGB_{}/fold{}.pth'.format(timestr,fold), num_iteration=bst.best_iteration)

    # evaluate
    jigsawevaluator_train = utils.JigsawEvaluator(y_train, y_identity[train_idx])
    train_pred = bst.predict(x_train, num_iteration=bst.best_iteration)
    train_score = jigsawevaluator_train.get_final_metric(train_pred)

    jigsawevaluator_validate = utils.JigsawEvaluator(y_validate, y_identity[validate_idx])
    validate_pred = bst.predict(x_validate, num_iteration=bst.best_iteration)
    validate_score = jigsawevaluator_validate.get_final_metric(validate_pred)

    print('train score:{}\nvalidate score:{}'.format(train_score,validate_score))
    log += 'fold{}:\ttrain score:{:.6f}\tvalidate score:{:.6f}\n'.format(fold,train_score,validate_score)
    cv_score['train'] += train_score
    cv_score['validate'] += validate_score
    # inference
    test_pred = bst.predict(x_test, num_iteration=bst.best_iteration)

    test_preds.append(test_pred)
    train_preds.append(pd.DataFrame({'id':validate_idx,'prediction':validate_pred}))

log += '\nmean train:{:.6f}\tmean validate:{:.6f}\n'.format(cv_score['train']/cv,cv_score['validate']/cv)
with open('saved_models/LGB_{}/log'.format(timestr),'w') as f:
    f.write(log)

test_preds = np.asarray(test_preds)
test_preds = test_preds.mean(axis=0)
test_preds = pd.DataFrame({'id': test_id, 'prediction': test_preds})
test_preds.set_index(keys='id',drop=True,inplace=True)
assert test_preds.shape[0] == test_num
test_preds.to_csv('saved_models/LGB_{}/submission.csv'.format(timestr))

train_preds = pd.concat(train_preds,ignore_index=True)
train_preds = train_preds.sort_values(by='id')
assert train_preds['id'].values.tolist() == list(range(train_num))
train_preds.to_csv('saved_models/LGB_{}/train_preds.csv'.format(timestr))













