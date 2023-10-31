import torch.nn as nn
# import dgl.function as fn
import torch.nn.functional as F
import torch


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力
    """
    def __init__(self, in_features, out_features, dropout, adj, alpha=1e-2, concat=True, edge_att=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   # 节点表示向量的输入特征维
        self.out_features = out_features   # 节点表示向量的输出特征维
        self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu激活的参数
        self.concat = concat   # 如果为true, 再进行elu激
        self.edge_att = edge_att
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)   # xavier初始
        self.Wk = nn.Parameter(torch.zeros(size=(in_features, out_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.Wk.data, gain=1.414)   # xavier初始
        self.Wv = nn.Parameter(torch.zeros(size=(in_features, out_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.Wv.data, gain=1.414)   # xavier初始
        
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # xavier初始

        
        self.adj = adj
        # 定义leakyrelu激活函
        self.leakyrelu = nn.LeakyReLU(self.alpha, inplace=True)
        #self.elu = nn.ELU()
        

    def forward(self, inp, k, v):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个
        adj: 图的邻接矩阵  [N, N] 非零即一，数据结构基本知
        """
        h = inp

        h = torch.matmul(inp, self.W)   # [N, out_features]
        k = torch.matmul(k, self.Wk)
        v = torch.matmul(v, self.Wv)
        

        N = h.size()[1]    # N 图的节点
        attention = torch.cat([h.repeat(1,1, N).view(h.shape[0], N *N, -1), k.repeat(1,N, 1)], dim=1).view(h.shape[0],N, -1, 2* self.out_features)
        # attention = torch.cat([h.repeat(1,1, N).view(h.shape[0], N *N, -1), h.repeat(1,N, 1)], dim=1).view(h.shape[0],N, -1, 2* self.out_features)
        #print(a_input.shape)
        attention = self.leakyrelu(torch.matmul(attention, self.a).squeeze(3))
        #print(e.shape)
        # attention = e
        

        

#         attention_a = attention_a[:,:,:,0]+attention_a[:,:,:,2].transpose(1,2)+torch.diag_embed(attention_a[:,:,:,1].sum(dim=-1)/N)+torch.diag_embed(attention_a[:,:,:,3].transpose(1,2).sum(dim=-1)/N)
        
        
#         attention = torch.mul(attention, attention_a)

        # zero_vec = -1e12 * torch.ones_like(attention)  # 将没有连接的边置为负无穷

        # attention = torch.where(self.adj > 0, attention, zero_vec)  # [N, N]
        
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑
        attention = F.softmax(attention, dim=-1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重
        
       
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        
        if self.adj != None:

            h = torch.matmul((attention + self.adj)/2, v)  # [N, N].[N, out_features] => [N, out_features]

        else:
            h = torch.matmul(attention, h)
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return self.leakyrelu(h)
            #return self.elu(h)
        else:
            return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, o_feat, adj, dropout, alpha=1e-2, n_heads=0, softmax=True, concat=True, trans=None):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.adj = adj
        self.n_heads = n_heads
        # 定义multi-head的图注意力层
        self.W = nn.Parameter(torch.zeros(size=(n_hid * n_heads, n_feat)), requires_grad=True)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        #self.gas = [GraphAttentionLayer(n_hid * n_heads, n_feat//n_heads, adj=adj, dropout= dropout, alpha=alpha, concat=False) for _ in
      #                     range(n_heads)]
       # for i, attention in enumerate(self.gas):
          #  self.add_module('gas_{}'.format(i), attention)
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, adj=adj, dropout= dropout, alpha=alpha, concat=concat) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        # self.out = GraphAttentionLayer(n_hid * n_heads, o_feat, adj=adj, dropout=dropout, alpha=alpha, concat=False)#使用注意力
        # self.out = nn.Linear(n_hid * n_heads, o_feat, bias=False)#使用线形
        # self.out = nn.Linear(n_hid * n_heads, o_feat)#使用线形
        self.softmax = softmax
        #self.ln = nn.LayerNorm(n_feat)

    def forward(self, x, k=None, v=None):
        x = x.permute(1, 0, 2)
        if not k is None:
            k = k.permute(1, 0, 2)
            v = v.permute(1, 0, 2)
        else:
            k = x
            v = x
        # x = torch.matmul(x, self.W1)
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, k, v) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼
        # x = torch.cat([self.attentions[i](x[:, :, x.shape[2]//self.n_heads*i:x.shape[2]//self.n_heads*(i+1)])
               # for i in range(len(self.attentions))], dim=2)  # 将每个head得到的表示进行拼
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        #x = torch.cat([att(x, x, x) for att in self.gas], dim=2)
        x = torch.matmul(x, self.W)
        # x += self.bias
        x = x.permute(1, 0, 2)
        if self.softmax:
            return F.softmax(x, dim=0), 0#softmax
        else:
            return x, 0
