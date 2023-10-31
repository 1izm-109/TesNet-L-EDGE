import torch.nn as nn
# import dgl.function as fn
import torch.nn.functional as F
import math

# 定义节点的UDF apply_nodes  他是一个完全连接层
import torch.nn.parameter



class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features),requires_grad=True)
        # self.ea=nn.Parameter(torch.zeros(size=( 4, 4)),requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features),requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # nn.init.xavier_uniform_(self.weight.data, gain=1.414)   # xavier初始
        # nn.init.xavier_uniform_(self.ea.data, gain=1.414)   # xavier初始
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            # nn.init.xavier_uniform_(self.bias.data, gain=1.414)   # xavier初始

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)

#         N = adj.shape[0]
#         attention = adj.unsqueeze(0).repeat(input.shape[0], 1, 1)

#         attention_input = torch.cat([attention.view(attention.shape[0], N, N, -1),\
#                                      torch.diagonal(attention, 0, 1, 2).unsqueeze(2).repeat(1,1,N).view(attention.shape[0], N, N,-1),\
#                                      attention.transpose(1, 2).view(attention.shape[0], N, N, -1),\
#                                      torch.diagonal(attention, 0, 1, 2).unsqueeze(1).repeat(1, N, 1).view(attention.shape[0], N, N,-1)],\
#                                     dim=3).view(attention.shape[0], N, -1, 4)

#         attention_a = torch.matmul(attention_input, self.ea)
#         attention_a = F.softmax(attention_a, dim=3)
#         attention_a = attention_a[:, :, :, 0] +\
#             attention_a[:, :, :, 2].transpose(1, 2) +\
#             torch.diag_embed(attention_a[:, :, :, 1].sum(dim=-1)/N) +\
#             torch.diag_embed(attention_a[:, :, :, 3].transpose(1, 2).sum(dim=-1)/N)

#         attention = torch.mul(attention, attention_a)

        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class RightMultLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False,dropout=0):
        super(RightMultLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features),requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features),requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def reset_parameter(self):
        for s in self.parameters():
            stdv = 1. / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)

class AGCN(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(AGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features),requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features),requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self,adj):
        # support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def reset_parameter(self):
        for s in self.parameters():
            stdv = 1. / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)

class AGCN1(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(AGCN1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.Tensor(20, in_features),requires_grad=True)
        self.weight2 = nn.Parameter(torch.Tensor(20, out_features), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features),requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self,adj):
        support = torch.matmul(adj, self.weight1)
        output = torch.matmul(support.permute(1,0), self.weight2)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


class ConvGCN(nn.Module):
    def __init__(self, in_channel, out_channel,in_features,out_features, bias=False):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features),requires_grad=True)
        self.conv = nn.Conv2d(in_channel,out_channel,1)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features),requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.conv(input.view(input.shape[0]*20,in_channel,14,14)).view(input.shape[0],20,-1)
#         support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def reset_parameter(self):
        for s in self.parameters():
            stdv = 1. / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)

class GCNEmbd(nn.Module):
    def __init__(self,in_channel,numclasses):
        super(GCNEmbd, self).__init__()
        self.numclasses=numclasses
        self.gc = GCN(10,10)
        self.conv1=nn.Conv2d(in_channel,200,1,bias=False)
        self.conv2=nn.Conv2d(200,in_channel,1,bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.relu=nn.LeakyReLU()

    def forward(self, X, A):
        aembd = self.conv1(self.avg_pool(X))
        membd = self.conv1(self.max_pool(X))
#         print(membd.shape)
        aembd = self.gc(aembd.view(aembd.shape[0],20,-1),A)
        membd = self.gc(membd.view(membd.shape[0],20,-1),A)
        membd = self.relu(membd)
        aembd = self.relu(aembd)
        embd = self.conv2(membd.view(membd.shape[0],200,1,1))+self.conv2(aembd.view(aembd.shape[0],200,1,1))
        return F.sigmoid(embd)
    
class GCNEmbd1(nn.Module):
    def __init__(self,in_channel,common_channel,size):
        super(GCNEmbd1, self).__init__()
        self.common_channel=common_channel
        self.gc = GCN(size*size*common_channel//20,size*size*common_channel//20//4)
        self.gc2 = GCN(size*size*common_channel//20//4,size*size*common_channel//20)
        self.conv1=nn.Conv2d(in_channel,common_channel,1,bias=False)
        self.conv2=nn.Conv2d(common_channel,in_channel,1,bias=False)
        self.relu=nn.LeakyReLU(0.2)

    def forward(self, X, A):
        embd = self.conv1(X)
        embd = self.gc(embd.view(embd.shape[0],20,-1),A)
        embd = self.relu(embd)
        embd = self.gc2(embd.view(embd.shape[0],20,-1),A)
        embd = self.relu(embd)
        embd = self.conv2(embd.view(embd.shape[0],self.common_channel,X.shape[2],X.shape[3]))
        return F.sigmoid(embd)
    
class GCNEmbd2(nn.Module):
    def __init__(self,in_channel,numclasses,size):
        super(GCNEmbd2, self).__init__()
        self.size=size
        self.gc2 = GCN(size*size,size*size)
        self.conv3=nn.Conv2d(in_channel,numclasses,1)
        self.relu=nn.LeakyReLU(0.2)
        self.conv4=nn.Conv2d(numclasses,1,1)
        
        self.numclasses=numclasses
        self.gc = GCN(1,1)
        self.conv1=nn.Conv2d(in_channel,numclasses,1)
        self.conv2=nn.Conv2d(numclasses,in_channel,1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.relu=nn.LeakyReLU(0.2)


    def forward(self, X, A):
        
        aembd = self.conv1(self.avg_pool(X))
        membd = self.conv1(self.max_pool(X))
#         print(membd.shape)
        aembd = self.gc(aembd.view(aembd.shape[0],self.numclasses,-1),A)
        membd = self.gc(membd.view(membd.shape[0],self.numclasses,-1),A)
        membd = self.relu(membd)
        aembd = self.relu(aembd)
        embd = self.conv2(membd.view(membd.shape[0],self.numclasses,1,1))+self.conv2(aembd.view(aembd.shape[0],self.numclasses,1,1))
        X=F.sigmoid(embd)*X
        
        embd = self.conv3(X)
        embd = self.gc2(embd.view(embd.shape[0],20,-1),A)
        embd = self.relu(embd)
        embd =self.conv4(embd.view(embd.shape[0],20,self.size,self.size))
#         embd = self.gc2(embd.view(embd.shape[0],20,-1),A)
#         embd = self.relu(embd)
#         embd = self.conv2(embd.view(embd.shape[0],self.common_channel,X.shape[2],X.shape[3]))
        return F.sigmoid(embd)*X

class GCNEmbd3(nn.Module):
    def __init__(self,in_channel,numclasses):
        super(GCNEmbd3, self).__init__()
        self.numclasses=numclasses
        self.in_channel=in_channel
        self.conv1=nn.Conv2d(in_channel,numclasses,1,bias=False)
        # self.conv2=nn.Conv2d(in_channel // 16,in_channel,1,bias=False)

        self.conv3 = nn.Conv2d(numclasses,in_channel // 16 , 1, bias=False)
        # self.conv21 = nn.Conv2d(numclasses*8, in_channel // 16, 1, bias=False)
        self.gc1 = AGCN(numclasses, in_channel)
        self.gc2 = AGCN(numclasses, in_channel)
        self.relu=nn.ReLU()
        self.mpooling=nn.AdaptiveMaxPool2d(1)
        self.apooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, X, A):
        # membd = self.relu(self.conv1(self.mpooling(X)))
        # aembd = self.relu(self.conv1(self.apooling(X)))
#         print(X.shape)
        membd = self.relu(torch.matmul(self.mpooling(X).view(X.shape[0],-1), self.gc1(A).permute(1,0)))
        aembd = self.relu(torch.matmul(self.apooling(X).view(X.shape[0],-1), self.gc1(A).permute(1,0)))

        membd = torch.matmul(membd.view(membd.shape[0],-1), self.gc2(A))
        aembd = torch.matmul(aembd.view(aembd.shape[0],-1), self.gc2(A))

        # membd = self.relu(self.conv3(membd.view(membd.shape[0],self.numclasses,1,1)))
        # aembd = self.relu(self.conv3(aembd.view(aembd.shape[0],self.numclasses,1,1)))

        # membd = self.conv2(membd)
        # aembd = self.conv2(aembd)
        return F.sigmoid(membd+aembd).view(membd.shape[0],self.in_channel,1,1)
    
class GCNEmbd4(nn.Module):
    def __init__(self,in_channel,numclasses):
        super(GCNEmbd4, self).__init__()
        self.numclasses=numclasses
        self.in_channel=in_channel
        self.conv1=nn.Conv2d(in_channel,in_channel // 16,1,bias=False)
        self.conv2=nn.Conv2d(in_channel // 16,in_channel,1,bias=False)

        self.conv3 = nn.Conv2d(numclasses,in_channel // 16 , 1, bias=False)
        # self.conv21 = nn.Conv2d(numclasses*8, in_channel // 16, 1, bias=False)
        self.gc1 = AGCN1(in_channel // 16, in_channel)
        self.gc2 = AGCN1(in_channel // 16,in_channel)
        self.relu=nn.ReLU()
        self.mpooling=nn.AdaptiveMaxPool2d(1)
        self.apooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, X, A):

        membd = self.relu(torch.matmul(self.mpooling(X).view(X.shape[0],-1), self.gc1(A).permute(1,0)))
        aembd = self.relu(torch.matmul(self.apooling(X).view(X.shape[0],-1), self.gc1(A).permute(1,0)))

        membd = torch.matmul(membd.view(membd.shape[0],-1), self.gc2(A))
        aembd = torch.matmul(aembd.view(aembd.shape[0],-1), self.gc2(A))

        return F.sigmoid(membd+aembd).view(membd.shape[0],self.in_channel,1,1)
    
class GCNEmbd31(nn.Module):
    def __init__(self,in_channel,numclasses):
        super(GCNEmbd31, self).__init__()
        self.numclasses=numclasses
        self.in_channel=in_channel
        self.conv1=nn.Conv2d(in_channel,in_channel // 16,1,bias=False)
        # self.conv2=nn.Conv2d(in_channel // 16,in_channel,1,bias=False)

        self.conv3 = nn.Conv2d(numclasses,in_channel // 16 , 1, bias=False)
        # self.conv21 = nn.Conv2d(numclasses*8, in_channel // 16, 1, bias=False)
        self.gc1 = AGCN1(in_channel, in_channel//16)
        self.gc2 = AGCN(numclasses, in_channel)
        self.relu=nn.ReLU()
        self.mpooling=nn.AdaptiveMaxPool2d(1)
        self.apooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, X, A):
        membd = self.relu(self.conv1(self.mpooling(X)))
        aembd = self.relu(self.conv1(self.apooling(X)))
        membd = torch.matmul(membd.view(membd.shape[0],-1), self.gc1(A).permute(1,0))
        aembd = torch.matmul(aembd.view(aembd.shape[0],-1), self.gc1(A).permute(1,0))
        return F.sigmoid(membd+aembd).view(membd.shape[0],self.in_channel,1,1)
    
class GCNEmbd32(nn.Module):
    def __init__(self,in_channel,numclasses):
        super(GCNEmbd32, self).__init__()
        self.numclasses=numclasses
        self.in_channel=in_channel
        self.conv1=nn.Conv2d(in_channel,in_channel // 16,1,bias=False)
        self.conv2=nn.Conv2d(in_channel // 16,in_channel,1,bias=False)

        self.conv3 = nn.Conv2d(numclasses,in_channel // 16 , 1, bias=False)
        # self.conv21 = nn.Conv2d(numclasses*8, in_channel // 16, 1, bias=False)
        self.gc11 = AGCN(numclasses, in_channel//2)
        self.gc12 = AGCN(in_channel//2, in_channel)
        self.gc21 = AGCN(numclasses, in_channel//2)
        self.gc22 = AGCN(in_channel//2, in_channel)
        self.relu=nn.ReLU()
        self.lrelu=nn.LeakyReLU(0.2)
        self.mpooling=nn.AdaptiveMaxPool2d(1)
        self.apooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, X, A):
#         membd = self.mpooling(X)
#         aembd = self.apooling(X)
#         membd = torch.matmul( self.gc1(A).permute(1,0),membd.view(membd.shape[0],-1))
#         aembd = torch.matmul(self.gc1(A).permute(1,0),aembd.view(aembd.shape[0],-1))
#         membd = self.relu(self.conv1(self.mpooling(X)))
#         aembd = self.relu(self.conv1(self.apooling(X)))
        X1 = self.gc11(A)
        X1 = self.lrelu(X1)
        X1 = self.gc12(X1)
        membd = self.relu(torch.matmul(self.mpooling(X).view(X.shape[0],-1), X1.permute(1,0)))
        aembd = self.relu(torch.matmul(self.apooling(X).view(X.shape[0],-1), X1.permute(1,0)))
        
        X1 = self.gc21(A)
        X1 = self.lrelu(X1)
        X1 = self.gc22(X1)
        membd = torch.matmul(membd.view(membd.shape[0],-1), X1)
        aembd = torch.matmul(aembd.view(aembd.shape[0],-1), X1)
        
#         membd = self.relu(torch.matmul(membd.view(membd.shape[0],-1),self.gc2(A).permute(1,0)))
#         aembd = self.relu(torch.matmul(aembd.view(aembd.shape[0],-1),self.gc2(A).permute(1,0)))
        
#         membd = self.conv2(membd.view(membd.shape[0],self.in_channel//16,1,1))
#         aembd = self.conv2(aembd.view(aembd.shape[0],self.in_channel//16,1,1))
        
        return F.sigmoid(membd+aembd).view(membd.shape[0],self.in_channel,1,1)
    
class GCNEmbd33(nn.Module):
    def __init__(self,in_channel,numclasses):
        super(GCNEmbd33, self).__init__()
        self.numclasses = numclasses
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, numclasses, 1, bias=False)
        # self.conv2=nn.Conv2d(in_channel // 16,in_channel,1,bias=False)

        self.conv3 = nn.Conv2d(numclasses, in_channel // 16, 1, bias=False)
        # self.conv21 = nn.Conv2d(numclasses*8, in_channel // 16, 1, bias=False)
        self.gc11 = AGCN(numclasses, in_channel//2)
        self.gc12 = AGCN(in_channel//2, in_channel)
        self.gc13 = AGCN(numclasses, in_channel//2)
        self.gc14 = AGCN(in_channel//2, in_channel)
        self.gc21 = AGCN(numclasses, in_channel//2)
        self.gc22 = AGCN(in_channel//2, in_channel)
        self.gc23 = AGCN(numclasses, in_channel//2)
        self.gc24 = AGCN(in_channel//2, in_channel)
        self.relu = nn.ReLU()
        self.mpooling = nn.AdaptiveMaxPool2d(1)
        self.apooling = nn.AdaptiveAvgPool2d(1)


    def forward(self, X, A):
        Y = self.gc11(A)
        Y = self.relu(Y)
        Y = self.gc12(Y).permute(1, 0)
        membd = self.relu(torch.matmul(self.mpooling(X).view(X.shape[0], -1), Y))
        Y = self.gc11(A)
        Y = self.relu(Y)
        Y = self.gc12(Y).permute(1, 0)
        aembd = self.relu(torch.matmul(self.apooling(X).view(X.shape[0], -1), Y))

        Y = self.gc21(A)
        Y = self.relu(Y)
        Y = self.gc22(Y)
        membd = torch.matmul(membd.view(membd.shape[0], -1), Y)
        Y = self.gc21(A)
        Y = self.relu(Y)
        Y = self.gc22(Y)
        aembd = torch.matmul(aembd.view(aembd.shape[0], -1), Y)
        return F.sigmoid(membd + aembd).view(membd.shape[0], self.in_channel, 1, 1)
    
class GCNEmbd34(nn.Module):
    def __init__(self,in_channel,numclasses):
        super(GCNEmbd34, self).__init__()
        self.numclasses = numclasses
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, numclasses, 1, bias=False)
        # self.conv2=nn.Conv2d(in_channel // 16,in_channel,1,bias=False)

        self.conv3 = nn.Conv2d(numclasses, in_channel // 16, 1, bias=False)
        # self.conv21 = nn.Conv2d(numclasses*8, in_channel // 16, 1, bias=False)
        self.gc11 = AGCN(numclasses, in_channel//2)
        self.gc12 = AGCN(in_channel//2, in_channel)
        self.gc13 = AGCN(numclasses, in_channel//2)
        self.gc14 = AGCN(in_channel//2, in_channel)
        self.gc21 = AGCN(numclasses, in_channel//2)
        self.gc22 = AGCN(in_channel//2, in_channel)
        self.gc23 = AGCN(numclasses, in_channel//2)
        self.gc24 = AGCN(in_channel//2, in_channel)
        self.relu = nn.ReLU()
        self.mpooling = nn.AdaptiveMaxPool2d(1)
        self.apooling = nn.AdaptiveAvgPool2d(1)


    def forward(self, X, A):
        Y = self.gc11(A)
        Y = self.relu(Y)
        Y = self.gc12(Y).permute(1, 0)
        membd = torch.matmul(self.mpooling(X).view(X.shape[0], -1), Y)
        Y = self.gc13(A)
        Y = self.relu(Y)
        Y = self.gc14(Y).permute(1, 0)
        aembd = torch.matmul(self.apooling(X).view(X.shape[0], -1), Y)

        return membd+aembd
    
    
class GCNEmbd35(nn.Module):
    def __init__(self,in_channel,numclasses):
        super(GCNEmbd35, self).__init__()
        self.numclasses=numclasses
        self.in_channel=in_channel
        self.conv1=nn.Conv2d(in_channel,numclasses,1,bias=False)
        # self.conv2=nn.Conv2d(in_channel // 16,in_channel,1,bias=False)

        self.conv3 = nn.Conv2d(numclasses,in_channel // 16 , 1, bias=False)
        # self.conv21 = nn.Conv2d(numclasses*8, in_channel // 16, 1, bias=False)
        # self.conv21 = nn.Conv2d(numclasses*8, in_channel // 16, 1, bias=False)
        self.gc1 = AGCN(numclasses, in_channel)
        self.gc2 = AGCN(numclasses, in_channel)
#         self.gc3 = GCN(in_channel, numclasses)
#         self.gc4 = GCN(in_channel, numclasses)
        self.relu=nn.ReLU()
        self.mpooling=nn.AdaptiveMaxPool2d(1)
        self.apooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, X, A):
        # membd = self.relu(self.conv1(self.mpooling(X)))
        # aembd = self.relu(self.conv1(self.apooling(X)))
#         print(X.shape)
        membd = self.relu(torch.matmul(self.mpooling(X).view(X.shape[0],-1), self.gc1(A).permute(1,0)))
        aembd = self.relu(torch.matmul(self.apooling(X).view(X.shape[0],-1), self.gc1(A).permute(1,0)))
#         Y1=self.gc3(self.relu(Y1),A)
        
        membd = torch.matmul(membd.view(membd.shape[0],-1), self.gc2(A.t()))
        aembd = torch.matmul(aembd.view(aembd.shape[0],-1), self.gc2(A.t()))
#         Y2=self.gc4(self.relu(Y2),A)
        # membd = self.relu(self.conv3(membd.view(membd.shape[0],self.numclasses,1,1)))
        # aembd = self.relu(self.conv3(aembd.view(aembd.shape[0],self.numclasses,1,1)))

        # membd = self.conv2(membd)
        # aembd = self.conv2(aembd)
        return F.sigmoid(membd+aembd).view(membd.shape[0],self.in_channel,1,1)
    

