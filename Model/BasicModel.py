from abc import abstractmethod
from cmath import log
from curses.ascii import EM
from turtle import forward
import torch
from torch import embedding, nn
from Model.Layers.Embedding import Embedding
from abc import abstractmethod

class BasicModel(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.embedding_layer = Embedding(config)

    def forward(self , sparse_input, dense_input = None):
        self.dense_input = self.embedding_layer(sparse_input)
        predict = self.FeatureInteraction(self.dense_input , sparse_input)
        return predict
    
    @abstractmethod
    def FeatureInteraction(self , dense_input , sparse_input, *kwrds):
        pass