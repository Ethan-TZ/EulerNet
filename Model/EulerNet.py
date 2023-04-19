from Model.BasicModel import BasicModel
from Utils import Config
import torch
from torch import nn

class EulerInteractionLayer(nn.Module):
    def __init__(self, config, inshape, outshape):
        super().__init__()
        self.inshape, self.outshape = inshape, outshape
        self.feature_dim = config.embedding_size
        self.apply_norm = config.apply_norm

        # Initial assignment of the order vectors, which significantly affects the training effectiveness of the model.
        # We empirically provide two effective initialization methods here.
        # How to better initialize is still a topic to be further explored.
        # Note: 👆 👆 👆
        if inshape == outshape:
            init_orders = torch.eye(inshape // self.feature_dim, outshape // self.feature_dim)
        else:
            init_orders = torch.softmax(torch.randn(inshape // self.feature_dim, outshape // self.feature_dim) / 0.01, dim = 0)
        
        self.inter_orders = nn.Parameter(init_orders)
        self.im = nn.Linear(inshape, outshape)
        nn.init.normal_(self.im.weight , mean = 0 , std = 0.01)

        self.bias_lam = nn.Parameter(torch.randn(1, self.feature_dim, outshape // self.feature_dim) * 0.01)
        self.bias_theta = nn.Parameter(torch.randn(1, self.feature_dim, outshape // self.feature_dim) * 0.01)

        self.drop_ex = nn.Dropout(config.drop_ex)
        self.drop_im = nn.Dropout(p = config.drop_im)
        self.norm_r = nn.LayerNorm([self.feature_dim])
        self.norm_p = nn.LayerNorm([self.feature_dim])
        

    def forward(self, complex_features):
        r, p = complex_features

        lam = r ** 2 + p ** 2 + 1e-8
        theta = torch.atan2(p, r)
        lam, theta = lam.reshape(lam.shape[0], -1, self.feature_dim), theta.reshape(theta.shape[0], -1, self.feature_dim)
        lam = 0.5 * torch.log(lam)
        lam, theta = torch.transpose(lam, -2, -1), torch.transpose(theta, -2, -1)
        lam, theta = self.drop_ex(lam), self.drop_ex(theta)
        lam, theta =  lam @ (self.inter_orders) + self.bias_lam,  theta @ (self.inter_orders) + self.bias_theta
        lam = torch.exp(lam)
        lam, theta = torch.transpose(lam, -2, -1), torch.transpose(theta, -2, -1)

        r, p = r.reshape(r.shape[0], -1), p.reshape(p.shape[0], -1)
        r, p = self.drop_im(r), self.drop_im(p)
        r, p = self.im(r), self.im(p)
        r, p = torch.relu(r), torch.relu(p)
        r, p = r.reshape(r.shape[0], -1, self.feature_dim), p.reshape(p.shape[0], -1, self.feature_dim)
        
        o_r, o_p =  r + lam * torch.cos(theta), p + lam * torch.sin(theta)
        o_r, o_p = o_r.reshape(o_r.shape[0], -1, self.feature_dim), o_p.reshape(o_p.shape[0], -1, self.feature_dim)
        if self.apply_norm:
            o_r, o_p = self.norm_r(o_r), self.norm_p(o_p)
        return o_r, o_p


class EulerNet(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        field_num = len(config.feature_stastic) - 1
        shape_list = [config.embedding_size * field_num] + [num_neurons * config.embedding_size for num_neurons in config.order_list]

        interaction_shapes = []
        for inshape, outshape in zip(shape_list[:-1], shape_list[1:]):
            interaction_shapes.append(EulerInteractionLayer(config, inshape, outshape))

        self.Euler_interaction_layers = nn.Sequential(*interaction_shapes)
        self.mu = nn.Parameter(torch.ones(1, field_num, 1))
        self.reg = nn.Linear(shape_list[-1], 1)
        nn.init.xavier_normal_(self.reg.weight)

    def FeatureInteraction(self , feature , sparse_input, *kargs):
        r, p = self.mu * torch.cos(feature), self.mu * torch.sin(feature)
        o_r, o_p = self.Euler_interaction_layers((r, p))
        o_r, o_p = o_r.reshape(o_r.shape[0], -1), o_p.reshape(o_p.shape[0], -1)
        re, im = self.reg(o_r), self.reg(o_p)
        self.logits = re + im
        self.output = torch.sigmoid(self.logits)
        return self.output
