import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
import torch
from global_variable import *
import gc
import os,sys
from sklearn.feature_extraction.text import TfidfVectorizer


def config2str(config):
    string = '\n'
    for k,v in sorted(config.items(),key=lambda x:x[0]):
        string += '{}:\t{}\n'.format(k,v)
    return string

def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()

def totensor(data,gpu=-1):
    if isinstance(data, float):
        tensor = torch.FloatTensor([data])
    elif isinstance(data, int):
        tensor = torch.LongTensor([data])
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).float()
    elif isinstance(data, torch.Tensor):
        tensor = data.float()
    else:
        raise ('Error data type:{}'.format(type(data)))
    if gpu>=0:
        tensor = tensor.cuda(gpu)
    return tensor

def get_statistics_feature(text,toxicity_word,unknow):
    if isinstance(text,pd.Series):
        text = pd.DataFrame(text)
    text['comment_text'] = text['comment_text'].str.lower()
    text['split'] = text['comment_text'].apply(str.split)
    text['total_length'] = text['comment_text'].apply(len)
    text['capitals'] = text['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    text['caps_vs_length'] = text['capitals'] / text['total_length']
    text['num_exclamation_marks'] = text['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '!！'))
    text['num_question_marks'] = text['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '?？'))
    text['num_punctuation'] = text['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
    text['num_symbols'] = text['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '&$%'))
    text['num_star'] = text['comment_text'].apply(lambda comment:comment.count('*'))
    text['num_words'] = text['split'].apply(len)
    text['num_unique_words'] = text['split'].apply(lambda comment: len(set(comment)))
    text['num_smilies'] = text['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
    text['num_toxicity'] = text['split'].apply(lambda comment: len(list(filter(lambda w:w in toxicity_word, comment))))
    text['num_unknow'] = text['split'].apply(lambda comment: len(list(filter(lambda w:w in unknow, comment))))
    text['unique_vs_words'] = text['num_unique_words'] / text['num_words']
    text['toxicity_vs_words'] = text['num_toxicity'] / text['num_words']
    text['unknow_vs_words'] = text['num_unknow'] / text['num_words']


    feature_columns = ['total_length', 'capitals', 'caps_vs_length', 'num_exclamation_marks', 'num_question_marks', 'num_punctuation',
                       'num_symbols', 'num_star', 'num_words', 'num_unique_words', 'num_smilies', 'num_toxicity', 'num_unknow',
                       'unique_vs_words', 'toxicity_vs_words', 'unknow_vs_words']

    return text[feature_columns]

def generate_statistics_data():
    train = pd.read_csv(train_csv_path)
    test = pd.read_csv(test_csv_path)
    x_train = train['comment_text']
    x_test = test['comment_text']

    with open(toxicity_word_path, 'rb') as f:
        toxicity_word = pickle.load(f)['total']

    with open(toxicity_embadding_path, 'rb') as f:
        data = pickle.load(f)
    unknow_word = data['unknown_words']

    train_statistics_feature = get_statistics_feature(x_train,toxicity_word,unknow_word,)
    test_statistics_feature = get_statistics_feature(x_test,toxicity_word,unknow_word,)

    with open(statistics_features_path,'wb') as f:
        pickle.dump({'train':train_statistics_feature,'test':test_statistics_feature},f)

    normalize = {}
    for c in train_statistics_feature.columns:
        mean = train_statistics_feature[c].mean(skipna=True)
        std = train_statistics_feature[c].std(skipna=True)
        normalize[c + '_mean'] = mean
        normalize[c + '_std'] = std

    with open(normalize_path, 'wb') as f:
        pickle.dump(normalize, f)

def get_data(normalize=True,aux=False):
    train_features = []
    test_features = []

    train = pd.read_csv(train_csv_path)
    test = pd.read_csv(test_csv_path)

    # get y & test_id
    train_label = np.where(train['target'] >= 0.5, 1, 0)
    if aux:
        train_label_aux = train[['target'] + aux_columns]
        train_label = np.where(train['target'] >= 0.5, 1, 0)
        train_label = np.concatenate([train_label[:, np.newaxis], train_label_aux], axis=1)
    train_identity = train[identity_columns].values
    test_id = test['id']
    # Overall
    weights = np.ones((train.shape[0],)) / 4
    weights += (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    weights += (((train['target'].values >= 0.5).astype(bool).astype(np.int) + (
    train[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(bool).astype(np.int) / 4
    weights += (((train['target'].values < 0.5).astype(bool).astype(np.int) + (
    train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(bool).astype(np.int) / 4
    loss_sacle = 1.0 / weights.mean()

    with open(statistics_features_path,'rb') as f:
        statistics_features = pickle.load(f)
    train_statistics_feature = statistics_features['train']
    test_statistics_feature = statistics_features['test']

    del train
    del test
    gc.collect()

    if normalize:
        with open(normalize_path,'rb') as f:
            normalize_data = pickle.load(f)
        for c in train_statistics_feature.columns:
            mean = normalize_data[c+'_mean']
            std = normalize_data[c+'_std']
            train_statistics_feature[c] = train_statistics_feature[c].fillna(mean)
            train_statistics_feature[c] = (train_statistics_feature[c] - mean) / std
            test_statistics_feature[c] = test_statistics_feature[c].fillna(mean)
            test_statistics_feature[c] = (test_statistics_feature[c] - mean) /std

    train_statistics_feature = train_statistics_feature.values
    test_statistics_feature = test_statistics_feature.values
    train_statistics_feature = np.repeat(train_statistics_feature[np.newaxis, :, :], cv, axis=0)  # 3*l*d
    test_statistics_feature = np.repeat(test_statistics_feature[np.newaxis, :, :], cv, axis=0)

    train_features.append(train_statistics_feature)
    test_features.append(test_statistics_feature)

    # get dl features
    dl_models = ['/data2/jianglibin/toxicity/DL/BERT_v1',
                 '/data2/jianglibin/toxicity/DL/BERT_v2',
                 '/data2/jianglibin/toxicity/DL/Toxicity_BiLSTMSelfAttention_0623-1059',
                 '/data2/jianglibin/toxicity/DL/Toxicity_LSTM2_0623-1101',
                 ]
    for folder in dl_models:
        with open(os.path.join(folder, 'features.pkl'), 'rb') as f:
            dl_features = pickle.load(f)
        train_features.append(dl_features['train'])
        test_features.append(dl_features['test'])

    train_features = np.concatenate(train_features, axis=2)
    test_features = np.concatenate(test_features, axis=2)

    print('Feature Num:',train_features.shape[2])

    return train_features,train_label,test_features,test_id,weights,loss_sacle,train_identity


def check_valid():
    floders = ['/data2/jianglibin/toxicity/DL/Toxicity_BiLSTMSelfAttention_0621-2306',
               '/data2/jianglibin/toxicity/DL/Toxicity_LSTM2_0621-2304',
               '/data2/jianglibin/toxicity/DL/BERT_v1',
               '/data2/jianglibin/toxicity/DL/BERT_v2',
               # '/data2/jianglibin/toxicity/DL/Toxicity_BiLSTMSelfAttention_0618-0903',
               # '/data2/jianglibin/toxicity/DL/Toxicity_LSTM2_0618-0904']
                ]

    for models in floders:
        # check features.pkl
        with open(os.path.join(models, 'features.pkl'), 'rb') as f:
            features = pickle.load(f)
        assert features['train'].shape == (cv, train_num, 50)
        assert features['test'].shape == (cv, test_num, 50)

        # check submission
        test_pred = pd.read_csv(os.path.join(models, 'submission.csv'))
        assert test_pred[['id', 'prediction']].shape == (test_num, 2)

        # check train_preds.csv
        train_pred = pd.read_csv(os.path.join(models, 'train_preds.csv'))
        assert train_pred[['id', 'prediction']].shape == (train_num, 2)
        assert train_pred['id'].tolist() == list(range(train_num))

def merge_fold(folder = 'bert_2epoch_lower'):
    test = pd.read_csv('test.csv')
    test_id = test['id']

    train_pred = []
    test_pred = []
    train_features = []
    test_features = []
    for fold in range(cv):
        validate_pred = pd.read_csv('{}/validate_pred_fold{}.csv'.format(folder, fold), index_col=0)
        train_pred.append(validate_pred)
        with open('{}/test_pred_fold{}.pkl'.format(folder, fold), 'rb') as f:
            test_pred.append(pickle.load(f))
        with open('{}/feature_fold{}.pkl'.format(folder, fold), 'rb') as f:
            features = pickle.load(f)
        train_features.append(features['train'])
        test_features.append(features['test'])

    train_pred = pd.concat(train_pred, ignore_index=True)
    train_pred = train_pred.sort_values(by='id')
    assert train_pred['id'].values.tolist() == list(range(train_num))
    train_pred = train_pred[['id', 'prediction']]
    train_pred.to_csv('{}//train_preds.csv'.format(folder))

    test_pred = np.asarray(test_pred)
    test_pred = test_pred.mean(axis=0)
    test_pred = pd.DataFrame({'id': test_id, 'prediction': test_pred})
    test_pred.set_index(keys='id', drop=True, inplace=True)
    assert test_pred.shape[0] == test_num
    test_pred.to_csv('{}/submission.csv'.format(folder))

    train_features = np.asarray(train_features)
    test_features = np.asarray(test_features)

    with open('{}/features.pkl'.format(folder), 'wb') as f:
        pickle.dump({'train': train_features, 'test': test_features}, f)

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

if __name__ == '__main__':
    # generate_statistics_data()
    check_valid()
    # get_word_value()
