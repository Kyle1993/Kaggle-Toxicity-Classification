from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pickle
import time
import sys,os
from datetime import datetime


from global_variable import *
import utils


start_time = datetime.now()

data_train,label_train,data_test,test_id,weight,loss_scale,y_identity = utils.get_data(normalize=True,aux=False)
print('Features Loaded!')

print('Time:',datetime.now()-start_time)


with open(kfold_path,'rb') as f:
    kfold = pickle.load(f)

params = {
    'criterion':'entropy',
    'max_depth':5,
    'max_features':'auto',
    'min_samples_leaf':10000,
    'n_estimators':80,
    # 'n_jobs':2,
    # 'verbose':1,
}

timestr = time.strftime('%m%d-%H%M')
if not os.path.exists('saved_models/RF_{}'.format(timestr)):
    os.mkdir('saved_models/RF_{}'.format(timestr))

log = timestr
log += '\nRF\n'
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

    jigsawevaluator_train = utils.JigsawEvaluator(y_train, y_identity[train_idx])
    jigsawevaluator_validate = utils.JigsawEvaluator(y_validate, y_identity[validate_idx])

    # train
    clf = RandomForestClassifier(**params,)
    clf.fit(x_train,y_train,sample_weight=w_train)

    # on train set
    train_pred = clf.predict_proba(x_train,)
    train_pred = train_pred[:,1]
    train_score = jigsawevaluator_train.get_final_metric(train_pred)

    # on validate set
    validate_pred = clf.predict_proba(x_validate,)
    validate_pred = validate_pred[:,1]
    validate_score = jigsawevaluator_validate.get_final_metric(validate_pred)

    print('train score:{}\nvalidate score:{}'.format(train_score,validate_score))
    log += 'fold{}:\ttrain score:{:.6f}\tvalidate score:{:.6f}\n'.format(fold,train_score,validate_score)
    cv_score['train'] += train_score
    cv_score['validate'] += validate_score

    # on test set
    test_pred = clf.predict_proba(x_test,)
    test_pred = test_pred[:, 1]

    test_preds.append(test_pred)
    train_preds.append(pd.DataFrame({'id':validate_idx,'prediction':validate_pred}))

    # save
    # clf.n_jobs = 1
    with open('saved_models/RF_{}/fold{}.pth'.format(timestr,fold),'wb') as f:
        pickle.dump(clf,f)

log += '\nmean train:{:.6f}\tmean validate:{:.6f}\n'.format(cv_score['train']/cv,cv_score['validate']/cv)
with open('saved_models/RF_{}/log'.format(timestr),'w') as f:
    f.write(log)

test_preds = np.asarray(test_preds)
test_preds = test_preds.mean(axis=0)
test_preds = pd.DataFrame({'id': test_id, 'prediction': test_preds})
test_preds.set_index(keys='id',drop=True,inplace=True)
assert test_preds.shape[0] == test_num
test_preds.to_csv('saved_models/RF_{}/submission.csv'.format(timestr))

train_preds = pd.concat(train_preds,ignore_index=True)
train_preds = train_preds.sort_values(by='id')
assert train_preds['id'].values.tolist() == list(range(train_num))
train_preds.to_csv('saved_models/RF_{}/train_preds.csv'.format(timestr))

print('All Time:',datetime.now()-start_time)





