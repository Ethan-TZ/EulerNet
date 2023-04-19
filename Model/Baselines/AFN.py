import torch
from torch import nn
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN
import numpy as np
from Model.Layers.LR import LR
from Model.Layers.Embedding import Embedding

class AFN(BasicModel):
    def __init__(self, config : Config):
        super().__init__(config)
        hidden_size = config.hidden_size
        self.im_embedding = Embedding(config)

        field_num = len(config.feature_stastic) - 1
        self.weights = nn.Parameter(torch.randn(hidden_size, field_num) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1, hidden_size, 1))

        self.bn_log = nn.BatchNorm1d(num_features = field_num)
        self.bn_inter = nn.BatchNorm1d(num_features= hidden_size)    
        self.dnn = DNN(config , [hidden_size * config.embedding_size] + config.afn_dnn, autoMatch= False, drop_last= True, dp_rate=0.)
        self.f_dnn = DNN(config , config.pass_dnn)
        self.fc = nn.Linear(config.afn_dnn[-1] + config.pass_dnn[-1] , 1)

    def FeatureInteraction(self , feature , sparse_input):
        im_in = self.im_embedding(sparse_input)
        ff = self.f_dnn(im_in)
        feature = torch.abs(feature)
        feature = torch.clamp(feature, min = 1e-5)
        feature = torch.log(feature)
        feature = self.bn_log(feature)
        feature = self.weights @ feature
        feature = torch.exp(feature)
        feature = self.bn_inter(feature)
        feature = feature.view(feature.shape[0], -1)
        feature = self.dnn(feature)
        feature = torch.cat([feature, ff], dim = -1)
        self.logits = self.fc(feature)
        self.output = torch.sigmoid(self.logits)
        return self.output
