import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def load_model(model_name,num_targets, embedding_matrix=None, embedding_size=None, output_feature_num=50,**kw):
    if model_name == 'Toxicity_LSTM2':
        return Toxicity_LSTM2(num_targets,embedding_matrix,embedding_size,output_feature_num)
    elif model_name == 'Toxicity_BiLSTMSelfAttention':
        return Toxicity_BiLSTMSelfAttention(num_targets,embedding_matrix,embedding_size,output_feature_num)
    else:
        raise Exception("Error model name:{}".format(model_name))


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class Toxicity_LSTM2(nn.Module):
    def __init__(self, num_targets, embedding_matrix=None, embedding_size=None, output_feature_num=50, lstm_middle=128):
        super(Toxicity_LSTM2, self).__init__()

        self.embedding = nn.Embedding(*embedding_size)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.output_feature_num = output_feature_num
        self.lstm_middle = lstm_middle
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm1 = nn.LSTM(embedding_size[1], lstm_middle, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_middle * 2, lstm_middle, bidirectional=True, batch_first=True)

        self.fc1_1 = nn.Linear(lstm_middle*6, lstm_middle*6)
        self.fc1_2 = nn.Linear(lstm_middle*6, lstm_middle*6)

        self.fc2 = nn.Linear(lstm_middle*6, output_feature_num)
        self.fc3 = nn.Linear(output_feature_num, num_targets)

    def cuda_(self,gpu):
        self.cuda(gpu)
        self.embedding = self.embedding.cuda(gpu)

    def forward(self, x):

        x = self.embedding(x)
        x = self.embedding_dropout(x).float()

        h_lstm1, _ = self.lstm1(x)
        out, (hn,_) = self.lstm2(h_lstm1)

        hn = hn.transpose(0,1).contiguous().view(-1,2*self.lstm_middle)
        avg_pool = torch.mean(out, 1)
        max_pool, _ = torch.max(out, 1)

        out = torch.cat((hn, max_pool, avg_pool), 1)
        out1 = F.relu(self.fc1_1(out))
        out2 = F.relu(self.fc1_2(out))
        out = out + out1 + out2

        feature = self.fc2(out)
        out = torch.sigmoid(self.fc3(F.relu(feature)))

        return out,feature

    @property
    def name(self):
        return 'Toxicity_LSTM2'

def isnan_pytorch(tensor):
    return np.isnan(tensor.cpu().detach().numpy())


def isinf_pytorch(tensor):
    return np.isinf(tensor.cpu().detach().numpy())

class Toxicity_BiLSTMSelfAttention(nn.Module):
    def __init__(self, num_targets, embedding_matrix=None, embedding_size=None, output_feature_num=200,lstm_h=64,attn_d=64):
        super(Toxicity_BiLSTMSelfAttention, self).__init__()

        self.embedding = nn.Embedding(*embedding_size)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        self.output_feature_num = output_feature_num
        self.lstm = nn.LSTM(embedding_size[1],lstm_h,bidirectional=True,batch_first=True)

        self.d = lstm_h * 2
        self.n_head = 4
        self.d_q = attn_d
        self.d_k = attn_d
        self.d_v = attn_d
        self.temperature = np.power(self.d_k, 0.5)
        assert self.d_q == self.d_k

        self.fc_q = nn.Linear(self.d,self.d_q * self.n_head)
        self.fc_k = nn.Linear(self.d,self.d_k * self.n_head)
        self.fc_v = nn.Linear(self.d,self.d_v * self.n_head)

        self.fc = nn.Linear(self.n_head*self.d_v,128)

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, output_feature_num)
        self.fc3 = nn.Linear(output_feature_num,num_targets)

        self.layer_norm = nn.LayerNorm(self.d)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc_q.weight, mean=0, std=np.sqrt(2.0 / (self.d + self.d_q)))
        nn.init.normal_(self.fc_k.weight, mean=0, std=np.sqrt(2.0 / (self.d + self.d_k)))
        nn.init.normal_(self.fc_v.weight, mean=0, std=np.sqrt(2.0 / (self.d + self.d_v)))

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def cuda_(self,gpu):
        self.cuda(gpu)
        self.embedding = self.embedding.cuda(gpu)

    def forward(self, x,):
        bs = x.shape[0]
        l = x.shape[1]

        mask = (x == 0)  # (N, L,)
        # make sure attn wont be nan
        mask[:,-1] = 0
        mask = mask.unsqueeze(1).repeat(self.n_head,l,1)

        # embbeding
        x = self.embedding(x)
        x = self.embedding_dropout(x).float()

        # lstm
        x,_ = self.lstm(x)

        # self-attention
        q = self.fc_q(x).view(bs,l,self.n_head,self.d_q)
        k = self.fc_k(x).view(bs,l,self.n_head,self.d_k)
        v = self.fc_v(x).view(bs,l,self.n_head,self.d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(bs*self.n_head, l, self.d_q) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(bs*self.n_head, l, self.d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(bs*self.n_head, l, self.d_v) # (n*b) x lv x dv

        attn = torch.bmm(q,k.transpose(1,2))
        attn /= self.temperature
        attn = attn.masked_fill(mask, -np.inf)
        attn = F.softmax(attn,dim=2)

        x = torch.bmm(attn,v)

        x = x.view(self.n_head,bs,l,self.d_v)
        x = x.permute(1, 2, 0, 3).contiguous().view(bs, l, -1)  # b x lq x (n*dv)

        x = self.fc(x)

        # represent
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)
        x = torch.cat((max_pool, avg_pool), 1)

        x = F.relu(self.fc1(x))
        feature = self.fc2(x)
        assert not isnan_pytorch(feature).any()
        assert not isinf_pytorch(feature).any()

        out = torch.sigmoid(self.fc3(F.relu(feature)))

        return out,feature

class Toxicity_NN(nn.Module):
    def __init__(self,feature_num,target_num,hidden=1024):
        super(Toxicity_NN,self).__init__()
        self.fc1 = nn.Linear(feature_num,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,target_num)
        self.drop = nn.Dropout()

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


if __name__ == '__main__':

    batch_size = 2
    length = 5
    embedding = np.random.random(size=(1000,10))
    x = torch.from_numpy(np.random.randint(0,999,size=(batch_size,length)))
    x[0,:2] = 0
    # x[1,] = 0
    print(x)

    y = torch.from_numpy(np.random.randint(0,2,size=(batch_size,))).float()

    model = Toxicity_BiLSTMSelfAttention(5,embedding,embedding.shape,attn_d=8)
    # print(model.state_dict().keys())

    # print(model)

    pred,feature = model(x)
    print(pred.shape,feature.shape)
    print(np.isnan(pred.detach().numpy()))
    # loss = F.binary_cross_entropy(pred[:,0],y)
    # print(loss)