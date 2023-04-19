import torch
from torch import nn
from Model.BasicModel import BasicModel
from Model.Layers.DNN import DNN


class InteractionMachine(nn.Module):
    def __init__(self, embedding_dim, order=2, batch_norm=False):
        super(InteractionMachine, self).__init__()
        assert order < 6, "order={} is not supported.".format(order)
        self.order = order
        self.bn = nn.BatchNorm1d(embedding_dim * order) if batch_norm else None
        self.fc = nn.Linear(order * embedding_dim, 1)
        
    def second_order(self, p1, p2):
        return (p1.pow(2) - p2) / 2

    def third_order(self, p1, p2, p3):
        return (p1.pow(3) - 3 * p1 * p2 + 2 * p3) / 6

    def fourth_order(self, p1, p2, p3, p4):
        return (p1.pow(4) - 6 * p1.pow(2) * p2 + 3 * p2.pow(2)
                + 8 * p1 * p3 - 6 * p4) / 24

    def fifth_order(self, p1, p2, p3, p4, p5):
        return (p1.pow(5) - 10 * p1.pow(3) * p2 + 20 * p1.pow(2) * p3 - 30 * p1 * p4
                - 20 * p2 * p3 + 15 * p1 * p2.pow(2) + 24 * p5) / 120

    def forward(self, X):
        out = []
        Q = X
        if self.order >= 1:
            p1 = Q.sum(dim=1)
            out.append(p1)
            if self.order >= 2:
                Q = Q * X
                p2 = Q.sum(dim=1)
                out.append(self.second_order(p1, p2))
                if self.order >= 3:
                    Q = Q * X
                    p3 = Q.sum(dim=1)
                    out.append(self.third_order(p1, p2, p3))
                    if self.order >= 4:
                        Q = Q * X
                        p4 = Q.sum(dim=1)
                        out.append(self.fourth_order(p1, p2, p3, p4))
                        if self.order == 5:
                            Q = Q * X
                            p5 = Q.sum(dim=1)
                            out.append(self.fifth_order(p1, p2, p3, p4, p5))
        out = torch.cat(out, dim=-1)
        if self.bn is not None:
            out = self.bn(out)
        y = self.fc(out)
        return y

class DeepIM(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.im = InteractionMachine(config.embedding_size, config.order)
        self.dnn = DNN(config,config.mlp)
    
    def FeatureInteraction(self , feature , sparse_input, *kargs):
        self.logits = self.im(feature) + self.dnn(feature)
        self.output = torch.sigmoid(self.logits)
        return self.output
