import torch.nn as nn
from GCN import GCN
import torchvision.models as models
from torch.nn import Parameter
from GAT import GAT, GraphAttentionLayer
import torch
import torch.nn.functional as F
from math import sqrt
from util import *
from models.src_files.models.tresnet import TResnetL
from data import add_weight_decay



class GCNLayer(nn.Module):
    def __init__(self, num_input, num_hidden, adj, alpha=1e-2, dropout=0.1):
        super(GCNLayer, self).__init__()
        self.gc1 = GCN(num_input, num_hidden)
        self.gc2 = nn.Linear(num_hidden, num_input)
        # self.gc2 = GCN(num_hidden, num_input)
        self.ReLU = nn.LeakyReLU(alpha, inplace=True)
        self.adj = adj
        self.dropout = nn.Dropout(dropout)

    def forward(self, fea, a=0, b=0):
        # out = self.gc2(self.dropout(self.ReLU(self.gc1(fea.permute(1, 0, 2), self.adj))), self.adj).permute(1, 0, 2)
        out = self.gc2(self.dropout(self.ReLU(self.gc1(fea.permute(1, 0, 2), self.adj).permute(1, 0, 2))))
        
        return out

class GATGCN(nn.Module):
    def __init__(self, num_input, num_hidden, num_heads, adj, alpha=1e-2, dropout=0.1):
        super(GATGCN, self).__init__()
        self.attentions = [GraphAttentionLayer(num_input, num_hidden//num_heads, adj=adj, dropout= dropout, alpha=alpha, concat=False) for _ in
                           range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        self.gc = GCN(num_hidden, num_input)
        # self.gc2 = GCN(num_hidden, num_input)
        # self.ReLU = nn.LeakyReLU(alpha, inplace=True)
        self.adj = adj
        self.dropout = dropout
        self.elu = nn.ELU()

    def forward(self, fea):
        fea = fea.permute(1, 0, 2)
        fea = torch.cat([att(fea) for att in self.attentions], dim=2)
        fea = self.elu(fea)
        fea = F.dropout(fea, self.dropout, training=self.training)
        fea = self.gc(fea, self.adj)
        fea = fea.permute(1, 0, 2)
        return fea 
    
class FC(nn.Module):
    def __init__(self, num_input, num_hidden, alpha=1e-2, dropout=0.1):
        super(FC, self).__init__()
        self.gc1 = nn.Linear(num_input, num_hidden)
        self.gc2 = nn.Linear(num_hidden, num_input)
        # self.gc2 = GCN(num_hidden, num_input)
        self.ReLU = nn.LeakyReLU(alpha, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, fea):
        # out = self.gc2(self.dropout(self.ReLU(self.gc1(fea, self.adj))), self.adj)
        out = self.gc2(self.dropout(self.ReLU(self.gc1(fea))))
        return out
    
class TransformerReplaceFWByGATAndGCN(nn.Module):
    def __init__(self, num_hidden, dim_forward, num_head, num_layer, adj, dropout=0.1, last_layer=True, use_FC=False, use_GAT=False, self_att=False ,num_out=1, is_decoder=False):
        super(TransformerReplaceFWByGATAndGCN, self).__init__()
        self.num_layer = num_layer
        self.last_layer = last_layer
        self.self_att = self_att
        self.is_decoder = is_decoder
        self.dropout = dropout
        if not use_GAT:
            self.multheadatts = nn.ModuleList([nn.MultiheadAttention(num_hidden, num_head, dropout=dropout) for i in range(num_layer)])
        else:
            self.multheadatts = nn.ModuleList([GAT(num_hidden, num_hidden//num_head, num_hidden, adj, dropout, alpha=1e-2, n_heads=num_head, concat=True,softmax=True) for i in range(num_layer)])
        if self.is_decoder:
            self.decoderatt = nn.ModuleList([nn.MultiheadAttention(num_hidden, num_head, dropout=dropout) for i in range(num_layer)])
        if not use_FC:
            self.gatgcns = nn.ModuleList([GAT(num_hidden, dim_forward//num_head, num_hidden, adj, dropout, alpha=0.2, n_heads=num_head, softmax=False) for i in range(num_layer)])
        else:
            self.gatgcns = nn.ModuleList([FC(num_hidden, dim_forward, alpha=0.2, dropout=dropout) for i in range(num_layer)])
        
        if last_layer:
            self.w = torch.nn.Parameter(torch.Tensor(adj.shape[0], num_hidden, num_out), requires_grad=True)
            self.bias = torch.nn.Parameter(torch.Tensor(adj.shape[0], num_out), requires_grad=True)
            torch.nn.init.xavier_normal_(self.w)
            torch.nn.init.xavier_normal_(self.bias)
            self.last_dropout = nn.Dropout(dropout)
            #torch.nn.init.constant_(self.bias, 0)
            # self.last_dropout = nn.Dropout(dropout)
        # self.linear = nn.Linear(num_hidden, 1)
        self.lns1 = nn.LayerNorm(num_hidden, eps=1e-5)
        self.lns2 = nn.ModuleList([nn.LayerNorm(num_hidden, eps=1e-5) for _ in range(num_layer)])
        self.lns3 = nn.ModuleList([nn.LayerNorm(num_hidden, eps=1e-5) for _ in range(num_layer)])
        self.lns4 = nn.ModuleList([nn.LayerNorm(num_hidden, eps=1e-5) for _ in range(num_layer)]) 
        self.dropouts1 = nn.Dropout(dropout)
        self.dropouts2 = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layer)])
        self.dropouts3 = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layer)])
        self.dropouts4 = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layer)])
    def forward(self, fea, inp):
        f = fea
        fea = fea+self.dropouts1(f)
        fea = self.lns1(fea)
        for i in range(self.num_layer):
            if self.is_decoder:
                if i < 1: #i<1
                    f, _ = self.decoderatt[i](fea, inp, inp)
                    fea = fea+self.dropouts4[i](f)
                    fea = self.lns4[i](fea)
                else:
                    f, _ = self.multheadatts[i](fea, fea, fea)
                    fea = fea+self.dropouts2[i](f)
                    fea = self.lns2[i](fea)
            else:
                f, _ = self.multheadatts[i](fea, fea, fea)
                fea = fea+self.dropouts2[i](f)
                fea = self.lns2[i](fea)

            f = self.gatgcns[i](fea)
            fea = fea+self.dropouts3[i](f)
            fea = self.lns3[i](fea)
        out = fea

        if self.last_layer:
            # fea = self.linear(fea)
            fea = fea.permute(1, 0, 2)
            # out = fea
            fea = self.last_dropout(fea)
            out = torch.zeros(fea.shape[0], fea.shape[1], self.w.shape[2], device=fea.device, dtype=fea.dtype)
            for i in range(fea.shape[1]):
                out[:, i, :] = torch.matmul(fea[:, i, :], self.w[i, :, :])
            out += self.bias

        return out
    

class OutputLayer(nn.Module):
    def __init__(self, num_input, num_hidden, num_classes, adj_file=None, t=0.4, dropout=0.1):
        super(OutputLayer, self).__init__()
        self.numclasses = num_classes
        self.num_hidden = num_hidden
        self.A = Parameter(torch.from_numpy(gen_A(num_classes, t, adj_file)).float(), requires_grad=False)

        self.adj = Parameter(gen_adj(self.A), requires_grad=False)

        self.embed = nn.Embedding(num_classes, num_hidden)
        
        self.attlinear = nn.Linear(num_input, num_hidden)

        self.trans = TransformerReplaceFWByGATAndGCN(num_hidden, 2048, 8, 2, self.adj, dropout=dropout, is_decoder=True, use_FC=True, use_GAT=True)

        self.enco = TransformerReplaceFWByGATAndGCN(num_hidden, 2048, 8, 1, None, dropout=dropout, last_layer=False, use_FC=True)
        self.dropout = dropout
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, feature):

        inp = feature.flatten(2).permute(2, 0, 1)

        inp = self.attlinear(inp)
        inp = self.relu(inp)
        inp = self.enco(inp, inp)
        

        fea = self.embed.weight.unsqueeze(1).repeat(1, feature.shape[0], 1).permute(1, 0, 2)

        fea = fea.view(fea.shape[0], self.numclasses, -1)
        out = self.trans(fea.permute(1, 0, 2), inp)
        # return out
        return out.view(out.shape[0], -1)


class ResNet_With_OutputLayer(nn.Module):
    def __init__(self, model,num_output,num_classes,output_layer=None,bn_freeze=False):
        super(ResNet_With_OutputLayer, self).__init__()
        model = model
        self.bn_freeze=bn_freeze
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        self.output_layer=output_layer
        if output_layer is None:
            self.output_layer=nn.Sequential(
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten(),
                nn.Linear(num_output,num_classes)
            )
    def forward(self, feature):
        if self.bn_freeze:
            self.freeze_bn()
        #print(feature.shape)
        feature = self.features(feature)
        out = self.output_layer(feature)
        return out

    def freeze_bn(self):
        self.features.eval()


    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.output_layer.parameters(), 'lr': lr},
        ]


class TResNet_With_OutputLayer(nn.Module):
    def __init__(self, model,num_output,num_classes,output_layer=None,bn_freeze=False):
        super(TResNet_With_OutputLayer, self).__init__()
        model = model
        self.bn_freeze=bn_freeze
        self.features = model.body
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        self.output_layer=output_layer
        if output_layer is None:
            self.output_layer=nn.Sequential(
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten(),
                nn.Linear(num_output,num_classes)
            )
    def forward(self, feature):
        if self.bn_freeze:
            self.freeze_bn()
        #print(feature.shape)
        feature = self.features(feature)
        feature = self.output_layer(feature)
        return feature

    def freeze_bn(self):
        self.features.eval()


    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.output_layer.parameters(), 'lr': lr},
        ]    

def tresnet_with_outputlayer(num_classes, t,model='tresnet-l', pretrained=False, adj_file=None,output_layer=True,hidden_num=128,bn_freeze=False):
    if model=='tresnet-l':
        output_num=2048;
        model = TResnetL({'num_classes': num_classes})
        if pretrained:
            def load_model_weights(model, model_path):
                state = torch.load(model_path, map_location='cpu')
                for key in model.state_dict():
                    if 'num_batches_tracked' in key:
                        continue
                    p = model.state_dict()[key]
                    # if key in state['state_dict']:
                        # ip = state['state_dict'][key]
                    if key in state['model']:
                        ip = state['model'][key]
                        if p.shape == ip.shape:
                            p.data.copy_(ip.data)  # Copy the data of parameters
                        else:
                            print(
                                'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
                    else:
                        print('could not load layer: {}, not in checkpoint'.format(key))
                return model
            model = load_model_weights(model, 'models/tresnet_l_pretrain.pth')
    else:
        raise Exception('model must in [''tresnet-l'']')
    if output_layer:
        return TResNet_With_OutputLayer(model,output_num, num_classes,OutputLayer(output_num,hidden_num,num_classes, t=t, adj_file=adj_file,dropout=0.1),bn_freeze=bn_freeze)
    else:
        return TResNet_With_OutputLayer(model,output_num, num_classes,bn_freeze=bn_freeze)
    
