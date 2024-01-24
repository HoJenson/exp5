import copy
import math

import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.functional import F

from settings import *

class Embeddings(nn.Module):
    """词嵌入层"""
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.embeding = nn.Embedding(vocab_size, d_model)
        # embedding维度
        self.d_model = d_model

    def forward(self, x):
        # 将embedding分布从N(0,1/embedding_size) 拉回 N(0,1)
        return self.embeding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5300):
        """
        :param d_model: embedding的维度
        :param dropout: dropout率
        :param max_len: 最大句子长度
        """
        super(PositionalEncoding, self).__init__()
        self.droupout = nn.Dropout(p=dropout)
        # [max_len, d_model]
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # 下面利用transformer给出的公式计算
        # 这里代码的计算方式与公式中给出的是不同的，是等价的，这样计算是为了避免中间的数值计算结果超出float的范围。
        position = torch.arange(0, max_len, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=DEVICE) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加批次维度，[1, max_len, embedding_dim]
        pe = pe.unsqueeze(0)
        # Adds a persistent buffer to the module/向模块添加持久缓冲区。
        # 反向传播不会更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个批次中语句所有词向量与位置编码相加
        # 注意，位置编码不参与训练，因此设置requires_grad=False
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.droupout(x)
    
# h个attention 同时计算
def attention(query, key, value, mask=None, dropout=None):
    """
    h代表每个多少个头部, 这里使矩阵之间的乘机操作, 对所有的head进行同时计算
    :param query: [batch_size, h, sequence_len_1, d_model]
    :param key: [batch_size, h, sequence_len_2, d_model]
    :param value: [batch_size, h, sequence_len_2, d_model]
    """
    d_k = query.size(-1)
    # 多维矩阵相乘需保证最后两维匹配
    # (batch_size, h, sequence_len_1, dk) * (batch_size, h, dk, sequence_len_2)
    # scores --> (batch_size, h, sequence_len_1, sequence_len_2)
    # scores 中每一行代表着长度为sequence_len的句子中每个单词与其他单词的相似度
    scores = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
    # 如果有 mask 先进行 mask 操作
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # softmax (对每一行)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # (batch_size, h, sequence_len_1, sequence_len_2) * (batch_size, h, sequence_len_2, dk)
    #  res--> (batch_size, h, sequence_len_1,dk)
    return torch.matmul(p_attn, value), p_attn

# 将一个模块clone N份
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeaderAttention(nn.Module):
    """多头注意力"""
    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: head的数目
        :param d_model: embedding的维度
        :param dropout:dropout操作时置0比率, 默认是0.1
        """
        super(MultiHeaderAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # 得到每个头获得的分割词向量维度d_k
        self.h = h  # 传入头数h
        
        # 创建linear层，通过nn的Linear实例化，它的内部变换矩阵是d_model x d_model，
        # 创建四个是因为在多头注意力中，Q,K,V各需要一个，最后拼接的矩阵还需要一个
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以为None
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 前向逻辑函数，它输入参数有四个，前三个就是注意力机制需要的Q,K,V
        # 最后一个是注意力机制中可能需要的mask掩码张量，默认是None
        if mask is not None:
            # Same mask applied to all h heads.
            # 使用unsqueeze扩展维度，代表多头中的第n头
            mask = mask.unsqueeze(1)
        # 接着，我们获得一个batch_size的变量，他是query尺寸的第1个数字，代表有多少条样本
        nbatches = query.size(0)

        # 首先利用zip将输入QKV与三个线性层组到一起，然后利用for循环，将输入QKV分别传到线性层中，做完线性变换后，开始为每个头分割输入，
        # 这里使用view方法对线性变换的结构进行维度重塑，多加了一个维度h代表头，这样就意味着每个头可以获得一部分词特征组成的句子，
        # 对第二维和第三维进行转置操作，为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系。
        # 对于query来说 [nbatch,length,d_model]*[d_model,d_model] --> [nbatch,length,d_model] --> [nbatch,length,h,dk] -> [nbatch,h,length,dk]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 得到每个头的输入后，接下来就是将他们传入到attention中，这里直接调用我们之前实现的attention函数，同时也将mask和dropout传入其中
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，我们需要将其转换为输入的形状以方便后续的计算，因此这里开始进行第一步处理环节的逆操作，
        # 先对第二和第三维进行转置(batch_size, h, sequence_len_1,dk)-->(batch_size, sequence_len_1, h, dk)，
        # 然后使用contiguous方法。这个方法的作用就是能够让转置后的张量应用view方法，否则将无法直接使用，
        # 下一步就是使用view重塑形状，变成和输入形状相同，后两个维度h,dk 合并为 dmodel。
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # 最后使用线性层列表中的最后一个线性变换得到最终的多头注意力结构的输出
        return self.linears[-1](x) # batch_size, sequence_len, embed_dim
    
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 按照最后一个维度计算均值和方差 (最后一个维度是embed_dim)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (torch.sqrt(std ** 2 + self.eps)) + self.b_2
    
class SublayerConnection(nn.Module):
    # 实现子层链接 每个子层的最终输出都是 LayerNorm(x+Sublayer(x))
    # 论文中黄色的 Add&Norm模块
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 原paper的方案
        # sublayer_out = sublayer(x)
        # x_norm = self.norm(x + self.dropout(sublayer_out))
        # 稍加调整的版本 (把x从norm中拿出来 加速收敛)
        sublayer_out = sublayer(self.norm(x))
        sublayer_out = self.dropout(sublayer_out)
        x_norm = x + self.norm(sublayer_out)
        return x_norm
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        # 一个多头注意力 --> Add&Norm --> 一个前向网络 -->Add&Norm (四个)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # attention + Add&Norm
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # feedforward + Add&Norm
        return self.sublayer[1](x, self.feed_forward)   

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 调用时回将编码器层传进来，叠加在一起
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        :param size: 词嵌入的维度大小,也代表解码器的尺寸 (二者相等)
        :param self_attn: 多头[自]注意力对象（Q=K=V）
        :param src_attn: 多头注意力对象，（Q!=K=V）
        :param feed_forward: 前馈全连接层对象
        :param dropout: dropout比率
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # clone了三个，Decoder有三块
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 进行源数据遮掩，抑制信息泄露
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 遮蔽掉对结果没有意义的padding
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
class Decoder(nn.Module):
    # 初始化函数的参数有两个，第一个就是解码器层layer，第二个是解码器层的个数N
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 调用时回将编码器层传进来，叠加在一起
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        :param x: 目标数据的嵌入表示
        :param memory: 编码器层的输出
        :param src_mask: 源数据数据的掩码张量
        :param tgt_mask: 目标数据的掩码张量
        :return:  x通过每一个层的处理，得出最后的结果，再进行一次规范化返回
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        :param encoder: 编码器对象
        :param decoder: 解码器对象
        :param src_embed: 源数据嵌入
        :param tgt_embed: 目标数据嵌入
        :param generator: 输出部分的单词生成器对象
        """
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 在函数中，将source source_mask传入编码函数，得到结果后与source_mask target 和target_mask一同传给解码函数
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

class Generator(nn.Module):
    """模型输出 linaer+softmax"""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # 损失函数 nn.KLDivLoss / KL散度，又叫做相对熵，算的是两个分布之间的距离，越相似则越接近零。
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        # 如果我们想要修改tensor的数值，但是又不希望被autograd记录，那么我么可以对tensor.data进行操作
        true_dist = x.data.clone()
        # [个人理解！] 减去UNK 和target的idx 之后取均匀分布 smoothing/(n-2)
        true_dist.fill_(self.smoothing / (self.size - 2))
        # 将target的idx在one-hot中位置改成大小为confidence 0.9 (1-smoothing)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 将UNK这一列置0
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    """损失函数"""
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    # 使得类实例对象可以像调用普通函数一样被调用
    # 相当于 计算loss -> backward -> optimizer.step(). 之后记得梯度清零
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()
    
class NoamOpt:
    """
    优化器
    Optim wrapper that implements rate
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

# 对模型进行初始化
def make_model(src_vocab, tgt_vocab, N=LAYERS, d_model=D_MODEL, d_ff=D_FF, h=H_NUM, dropout=DROPOUT):
    c = copy.deepcopy
    # 实例化Attention对象
    attn = MultiHeaderAttention(h, d_model).to(DEVICE)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout).to(DEVICE)
    # 实例化Transformer模型对象
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)).to(DEVICE)

    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)
