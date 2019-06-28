from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
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

data_train,label_train,data_test,test_id,weight,loss_scale,y_identity = utils.get_data(normalize=True,aux=False)
print('Features Loaded!')

print('Time:',datetime.now()-start_time)


with open(kfold_path,'rb') as f:
    kfold = pickle.load(f)

params = {
    'C':0.8,
    'kernel':'linear',
    'gamma':'auto',
    'probability':True,
    # 'verbose':True,
}

print('Start Training & Inference...')

timestr = time.strftime('%m%d-%H%M')
if not os.path.exists('saved_models/SVC_{}'.format(timestr)):
    os.mkdir('saved_models/SVC_{}'.format(timestr))

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

    # num_sample = 10000
    # train_idx = list(range(num_sample))[:int(num_sample*0.8)]
    # validate_idx = list(range(num_sample))[int(num_sample*0.8):]

    x_test = data_test[fold]

    x_validate = data_train[fold][validate_idx]
    y_validate = label_train[validate_idx]
    w_validate = weight[validate_idx]

    x_train = data_train[fold][train_idx]
    y_train = label_train[train_idx]
    w_train = weight[train_idx]

    # train
    n_estimators = 50
    clf = BaggingClassifier(SVC(**params),max_samples=1.0 / n_estimators, n_estimators=n_estimators,n_jobs=4)
    # clf = SVC(**params)

    clf.fit(x_train,y_train,sample_weight=w_train)

    # save
    with open('saved_models/SVC_{}/fold{}.pth'.format(timestr,fold),'wb') as f:
        pickle.dump(clf,f)

    # evaluate
    # on train set
    jigsawevaluator_train = utils.JigsawEvaluator(y_train, y_identity[train_idx])
    train_pred = clf.predict_proba(x_train,)
    train_pred = train_pred[:,1]
    train_score = jigsawevaluator_train.get_final_metric(train_pred)

    # on validate set
    jigsawevaluator_validate = utils.JigsawEvaluator(y_validate, y_identity[validate_idx])
    validate_pred = clf.predict_proba(x_validate,)
    validate_pred = validate_pred[:,1]
    validate_score = jigsawevaluator_validate.get_final_metric(validate_pred)

    # on test set
    test_pred = clf.predict_proba(x_test,)
    test_pred = test_pred[:,1]

    test_preds.append(test_pred)
    train_preds.append(pd.DataFrame({'id': validate_idx, 'prediction': validate_pred}))

    print('train score:{}\nvalidate score:{}'.format(train_score,validate_score))
    log += 'fold{}:\ttrain score:{:.6f}\tvalidate score:{:.6f}\n'.format(fold, train_score, validate_score)
    cv_score['train'] += train_score
    cv_score['validate'] += validate_score

    print('Fold Time:', datetime.now() - start_time)


log += '\nmean train:{:.6f}\tmean validate:{:.6f}\n'.format(cv_score['train']/cv,cv_score['validate']/cv)
with open('saved_models/SVC_{}/log'.format(timestr), 'w') as f:
    f.write(log)

test_preds = np.asarray(test_preds)
test_preds = test_preds.mean(axis=0)
test_preds = pd.DataFrame({'id': test_id, 'prediction': test_preds})
test_preds.set_index(keys='id',drop=True,inplace=True)
assert test_preds.shape[0] == test_num
test_preds.to_csv('saved_models/SVC_{}/submission.csv'.format(timestr))

train_preds = pd.concat(train_preds,ignore_index=True)
train_preds = train_preds.sort_values(by='id')
assert train_preds['id'].values.tolist() == list(range(train_num))
train_preds.to_csv('saved_models/SVC_{}/train_preds.csv'.format(timestr))

print('All Time:', datetime.now() - start_time)

