import pandas as pd
import numpy as np
import sys,os
import pickle
import time
import random

import utils
from global_variable import *


class Ensembler():
    def __init__(self,model_num,stop_round=10):
        self.model_num = model_num
        self.max_scroe = 0
        self.stop_round = stop_round
        self.sample_num = min(1,model_num)
        self.weight = np.ones((model_num,))
        # self.resolution = 0.5

    @staticmethod
    def softmax(x):
        return np.exp(x)/np.exp(x).sum()

    def sample(self,list,num):
        return [random.choice(list) for _ in range(num)]

    def evaluate(self,pred, y_train, y_identity,):
        jigsawevaluator = utils.JigsawEvaluator(y_train, y_identity)
        return jigsawevaluator.get_final_metric(pred)

    # random search, until score stop imporve for n round
    def fit(self, X, y_train, y_identity,):
        assert X.shape[1] == self.model_num
        jigsawevaluator = utils.JigsawEvaluator(y_train, y_identity)
        round = 0
        while round < self.stop_round:
            pre_weight = self.weight.copy()
            idx = self.sample(list(range(self.model_num)),self.sample_num)
            op = self.sample([1,-1],self.sample_num)
            for i in range(self.sample_num):
                self.weight[idx[i]] += op[i] * random.random()
            weights = self.softmax(self.weight) # make sure sum==1

            res = (weights * X).sum(axis=1)
            score = jigsawevaluator.get_final_metric(res)
            if score > self.max_scroe:
                self.max_scroe = score
                round = 0
                print('Inprove! Score:{:.6f}\tWeight:{}'.format(self.max_scroe,self.softmax(self.weight)))
            else:
                self.weight = pre_weight.copy()
                round += 1
        print('Ensemble Done! \nFinal Score:{}\nFinal Weight:{}\n'.format(self.max_scroe,self.softmax(self.weight)))
        return self.max_scroe

    def predict(self,X):
        assert X.shape[1] == self.model_num
        return (self.softmax(self.weight) * X).sum(axis=1)

    def save(self,path):
        dump_ = self.__dict__
        with open(path,'wb') as f:
            pickle.dump(dump_,f)

    @staticmethod
    def load(path):
        with open(path,'rb') as f:
            parameters = pickle.load(f)
        model = Ensembler(parameters['model_num'])
        model.weight = parameters['weight']
        model.stop_round = parameters['stop_round']
        model.max_scroe = parameters['max_scroe']
        model.sample_num = parameters['sample_num']
        model.resolution = parameters['resolution']

        return model

with open(kfold_path,'rb') as f:
    kfold = pickle.load(f)

test = pd.read_csv(test_csv_path)
train = pd.read_csv(train_csv_path)
test_id = test['id'].values.tolist()
train_id = list(range(train_num))

y_train = np.where(train['target'] >= 0.5, 1, 0)
y_identity = train[identity_columns].values

print('Data Loaded')

ensemble_list = ['saved_models/LGB_0626-1807',
                 'saved_models/NN_0625-2343',
                 'saved_models/RF_0626-1556',
                 'saved_models/XGB_0626-1837',
                 '/data2/jianglibin/toxicity/DL/BERT_v1',
                 '/data2/jianglibin/toxicity/DL/BERT_v2',
                 '/data2/jianglibin/toxicity/DL/Toxicity_BiLSTMSelfAttention_0623-1059',
                 '/data2/jianglibin/toxicity/DL/Toxicity_LSTM2_0623-1101',]

train_preds = []
test_preds = []

for el in ensemble_list:
    train_p = pd.read_csv(os.path.join(el,'train_preds.csv'))
    test_p = pd.read_csv(os.path.join(el,'submission.csv'))
    assert train_p['id'].values.tolist() == train_id
    assert test_p['id'].values.tolist() == test_id

    train_preds.append(train_p['prediction'].values)
    test_preds.append(test_p['prediction'].values)

train_preds = np.asarray(train_preds).transpose((1,0))
test_preds = np.asarray(test_preds).transpose((1,0))

timestr = time.strftime('%m%d-%H%M')
if not os.path.exists('saved_models/Ensemble_{}'.format(timestr)):
    os.mkdir('saved_models/Ensemble_{}'.format(timestr))

model = Ensembler(len(ensemble_list),stop_round=10)
train_final_score = model.fit(train_preds,y_train, y_identity)
save_path = 'saved_models/Ensemble_{}/ensembler.pth'.format(timestr)
model.save(save_path)

log = timestr
log += '\n'
for m in ensemble_list:
    log += '{}\n'.format(m)
log += '\ncv:{:.6f}\n'.format(train_final_score)

with open('saved_models/Ensemble_{}/log'.format(timestr),'w') as f:
    f.write(log)

test_pred = model.predict(test_preds)
# test_pred = test_preds.mean(axis=1)
test_pred = pd.DataFrame({'id': test_id, 'prediction': test_pred})
test_pred.set_index(keys='id',drop=True,inplace=True)
assert test_pred.shape[0] == test_num
test_pred.to_csv('saved_models/Ensemble_{}/submission.csv'.format(timestr))


