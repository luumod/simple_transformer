import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
import copy
import os
from pyitcast.transformer_utils import Batch
from pyitcast.transformer_utils import get_std_opt
from pyitcast.transformer_utils import LabelSmoothing
from pyitcast.transformer_utils import SimpleLossCompute
from pyitcast.transformer_utils import run_epoch
from pyitcast.transformer_utils import greedy_decode

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        '''
        :param d_model: 词嵌入的维度，每个词被映射为向量的维度
        :param vocab:  词表的大小
        '''
        super(Embeddings,self).__init__()
        # 定义Embedding层
        self.lut = nn.Embedding(vocab,d_model)
        # 将参数传入类中
        self.d_model = d_model

    def forward(self, x):
        '''
        :param x: 源数据（未嵌入表示）
        '''
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        '''
        :param d_model: 词嵌入的维度
        :param dropout: 置0比率
        :param max_len: 每个句子的最大长度（相当于第二维）
        '''
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        # 实例化dropout层
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码矩阵，大小是 max_len*d_model
        pe = torch.zeros(max_len, d_model)
        # 初始化一个绝对位置矩阵，从一维扩展到二维 max_len * 1
        position = torch.arange(0, max_len).unsqueeze(1)
        # 定义变换矩阵：1*d_model/2
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # 将前面定义的位置矩阵的奇数和偶数列分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将二维张量扩充为三维张量 1*max_len*d_model
        pe = pe.unsqueeze(0)
        # 将位置编码矩阵注册为模型的buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: 词嵌入层的输出：文本序列的词嵌入表示: 3*4*512
        '''
        # 把三维张量的第二维，也就是句子的最大长度这一维，切片到与输入的x的第二维相同，即x.size(1)
        # InputEmb(x) = sqrt(d_model) * E(x) + PE
        # 将词义信息x与位置信息pe融合
        x = x + Variable(self.pe[:, :x.size(1), :self.d_model], requires_grad=False)
        return self.dropout(x)

def attention(query, key, value, mask=None,dropout=None):
    '''
    :param query:
    :param key:
    :param value:
    :param mask: 掩码张量
    :param dropout: 传入的实例化对象
    :return: 3*4*512的注意力表示 和 3*4*4的注意力张量
    '''
    # 提取query的最后一个维度：也就是词嵌入维度
    d_k = query.size(-1) # 512
    # 注意力计算公式： 3*4*512 * 3*512*4 => 3*4*4
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k) # 2*8*4*64 * 2*8*64*4 -> 2*8*4*4
    # 判断是否使用掩码张量
    if mask is not None:
        # 将无效的位置的得分置为极小值
        scores = scores.masked_fill(mask==0,-1e9)

    # 对scores的最后一个维度进行softmax操作： 3*4*4
    p_attn = F.softmax(scores,dim=-1)

    # 判断是否使用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 完成p_attn与value的乘法，并且返回query的注意力表示和注意力张量
    # 3*4*4 * 3*4*512 -> 3*4*512
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    '''
    :param module: 要克隆的目标网络层
    :param N: module要克隆几次
    :return:
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        '''
        :param head: 几个头的参数
        :param embedding_dim: 词嵌入的维度
        :param dropout: 置零比率
        '''
        super(MultiHeadAttention, self).__init__()
        # 要保证多头的head需要整除词嵌入的维度
        assert embedding_dim % head == 0
        # 得到每个头获得的词向量的维度
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim
        # 获得线性层：4个：Q、K、V、输出线性层
        self.linears = clones(nn.Linear(embedding_dim,embedding_dim), 4)
        # 初始化注意力张量
        self.attn = None
        # 初始化dropout对象
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 首先判断是否使用掩码张量
        if mask is not None:
            # 将掩码张量维度扩充,代表多头中的第n个头
            mask = mask.unsqueeze(1)
        # 获得批次，即句子（样本）的个数
        batch_size = query.size(0)
        # 使用zip将网络层和输入数据连接在一起，模型的输出利用view和transpose进行维度和形状的改变
        query, key, value = [model(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2)
                             for model, x in zip(self.linears, (query,key,value))]
        # query, key, value： 【批次, 头数, 长度, 单词】
        # 将每个头的输出传入到注意力层
        x, self.attn = attention(query, key, value, mask=mask,dropout=self.dropout)
        # 得到每个头的计算结果是4维张量，需要进行形状上的改变，转置回来1，2的维度，transpose -> contiguous -> view
        # 【批次, 长度, 词嵌入维度】
        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.head * self.d_k)
        # 最后将x输入到最后一个线性层中进行处理
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        '''
        :param d_model: 词嵌入的维度，同时也是两个线性层的输入维度和输出维度
        :param d_ff: 线性层的维度
        :param dropout: 置零比率
        '''
        super(PositionwiseFeedForward, self).__init__()
        # 定义两层全连接线性层
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        :param x: 代表来自上一层的输出: 第一层 -> relu -> dropout -> 第二层
        '''
        return self.w2(self.dropout(F.relu(self.w1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        '''
        :param features: 词嵌入的维度
        :param eps: 一个足够小的整数，用于保证分母不为零，防止除零
        '''
        super(LayerNorm, self).__init__()
        # 初始化两个参数张量 a2, b2，便于计算。用nn.Parameter进行封装，代表他们是模型的参数
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        '''
        :param x: 上一层的输出
        '''
        # 对x进行最后一个维度上的求均值及标准差操作，保持维度
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 规范化公式
        return self.a2 * (x - mean) / (std + self.eps) + self.b2

class SublayerConnection(nn.Module):
    def __init__(self, features, dropout=0.1):
        '''
        :param features: 词嵌入的维度
        :param dropout: 置零比率
        '''
        super(SublayerConnection, self).__init__()
        # 实例化一个规范化层对象
        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(p=dropout)
        self.features = features

    def forward(self, x, sublayer):
        '''
        :param x: 上一层传入的张量
        :param sublayer:  子层连接中的子层函数
        '''
        # norm -> sublayer -> dropout -> residual
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, features, self_attn, feed_forward, dropout):
        '''
        :param features: 词嵌入维度
        :param self_attn: 多头子注意力子层的实例化对象
        :param feed_forward: 前馈全连接层实例化对象
        :param dropout: 置零比率
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.features = features
        self.dropout = nn.Dropout(p=dropout)
        # 编码器中有两个子层连接结构
        self.sublayers = clones(SublayerConnection(features, dropout), 2)

    def forward(self, x, mask):
        '''
        :param x: 源数据的嵌入表示 或 上一层的输出
        :param mask: 掩码张量
        '''
        # 首先第一个子层连接结构：多头自注意力机制
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask=mask))
        # 再经过第二个子层连接结构：前馈全连接网络层
        return self.sublayers[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        '''
        :param layer: 编码器层
        :param N: 编码器层的数量
        '''
        super(Encoder, self).__init__()
        # 首先克隆N个编码器层
        self.layers = clones(layer, N)
        # 初始化一个规范化层，放在编码器的后面
        self.norm = LayerNorm(layer.features)

    def forward(self, x, mask):
        '''
        :param x: 源数据的嵌入表示
        :param mask: 掩码张量
        '''
        # 让x依次经过N个编码器层的处理，最后规范化
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, features, self_attn, src_attn, feed_forward, dropout):
        '''
        :param features: 嵌入词维度
        :param self_attn: 多头自注意力机制对象
        :param src_attn: 常规注意力机制对象
        :param feed_forward: 前馈全连接网络层
        :param dropout: 置零比率
        '''
        super(DecoderLayer, self).__init__()
        self.features = features
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        # 克隆三个子层连接对象：多头自注意力 + 常规注意力 + 前馈全连接
        self.sublayers = clones(SublayerConnection(features,dropout),3)

    def forward(self, x, memory, src_mask, target_mask):
        '''
        :param x: 目标数据的嵌入表示 或者 上一层输出
        :param memory: 编码器的最终输出张量
        :param src_mask: 源数据的掩码张量（用于第二层）
        :param target_mask: 目标数据的掩码张量（用于第一层）
        '''
        m = memory
        # 首先让x经历第一层：多头自注意力机制层target_mask：遮掩未来信息
        x = self.sublayers[0](x, lambda x: self.self_attn(x,x,x,target_mask))
        # 其次让x经历第二层：常规注意力机制子层 Q!=K=V src_mask：遮掩无用信息
        x = self.sublayers[1](x, lambda x:self.src_attn(x,m,m,src_mask))
        # 最后让x经历第三层：前馈全连接层
        return self.sublayers[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        '''
        :param layer: 解码器层的对象
        :param N:  解码器中所包含的层数
        '''
        super(Decoder,self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.features)

    def forward(self, x, memory, src_mask, target_mask):
        '''
        :param x: 目标数据的嵌入表示
        :param memory: 编码器的输出张量
        :param src_mask: 源数据的掩码张量（用于第二层）
        :param target_mask: 目标数据的掩码张量（用于第一层）
        :return:
        '''
        for layer in self.layers:
            x = layer(x, memory, src_mask, target_mask)
        return self.norm(x)

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        '''
        :param d_model: 词嵌入维度
        :param vocab_size: 词表大小
        '''
        super(Generator, self).__init__()
        # 定义一个线性层，完成网络输出维度的变换 512 -> 1000
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        '''
        :param x: 经解码器后的输出张量
        '''
        # 线性层 -> softmax
        return F.log_softmax(self.project(x), dim=-1)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, target_embed, generator):
        '''
        :param encoder: 编码器对象
        :param decoder:  解码器对象
        :param src_embed: 源数据的嵌入函数
        :param target_embed: 目标数据的嵌入函数
        :param generator: 输出部分的类别生成器
        '''
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.generator = generator

    def forward(self, source, target, src_mask, target_mask):
        '''
        :param source: 源数据（并没有词嵌入）
        :param target: 目标数据（并没有词嵌入）
        :param src_mask:  源数据掩码张量
        :param target_mask: 目标数据掩码张量
        '''
        return self.decode(self.encode(source, src_mask), src_mask, target, target_mask)

    def encode(self, source, src_mask):
        '''
        :param source: 源数据（未嵌入表示）
        :param src_mask: 源数据的掩码张量
        '''
        return self.encoder(self.src_embed(source), src_mask)

    def decode(self, memory, src_mask, target, target_mask):
        '''
        :param memory: 编码器的最终输出张量
        :param src_mask: 源数据的掩码张量
        :param target: 目标数据（未嵌入表示）
        :param target_mask: 目标数据的掩码张量
        '''
        return self.decoder(self.target_embed(target), memory, src_mask, target_mask)


def make_model(source_vocab, target_vocab, N=6,d_model=512,d_ff=2048,head=8,dropout=0.2):
    '''
    :param source_vocab: 源数据的词汇总数
    :param target_vocab: 目标数据的词汇总数
    :param N: 整个编码器和解码器堆叠的层数
    :param d_model: 词嵌入的维度
    :param d_ff: 前馈全连接层中变换矩阵的维度
    :param head: 多头注意力机制中的头数
    :param dropout: 置零比率
    '''
    c = copy.deepcopy
    # 实例化：多头注意力类
    attn = MultiHeadAttention(head,d_model,dropout)
    # 实例化：前馈全连接层
    ff = PositionwiseFeedForward(d_model,d_ff,dropout)
    # 实例化：位置编码器
    position = PositionalEncoding(d_model,dropout)
    # 实例化：最终模型：
    # 编码器：两层
    # 解码器：三层
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout), N),
        Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )
    # 初始化模型中的参数，如果参数维度大于1，则初始化为服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def data_generator(V, batch_size, num_batch):
    '''
    :param V: 随机生成数据的数字的范围 [1 - V)
    :param batch_size:
    :param num_batch:
    :return:
    '''
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(1,V,size=(batch_size, 10),dtype=np.int64))
        # 第一列全为1，作为起始标志列
        data[:, 0] = 1
        # copy任务
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        yield Batch(source, target)

def run(model, loss, epochs=30):
    '''
    :param model: 要训练的模型
    :param loss: 使用的损失计算方法
    :param epochs: 训练的轮次数
    '''
    for epoch in range(epochs):
        # 训练模式：更新参数
        model.train()
        run_epoch(data_generator(V, 15, 20), model, loss)

        # 验证模型
        model.eval()
        run_epoch(data_generator(V, 8, 5), model, loss)

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    print('训练完成')
    torch.save(model.state_dict(), './models/transformer_model.pt')
    # 测试
    model.eval()
    # 假定的输入数据
    source = Variable(torch.LongTensor([[1,3,2,4,5,7,6,8,9,10]]))
    # 全1不遮掩
    source_mask = Variable(torch.ones(1,1,10))

    res = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(res)

if __name__ == '__main__':
    V = 11
    # 实例化模型
    model = make_model(V, V, N=2)
    # 模型优化器
    model_optimizer = get_std_opt(model)
    # 标签平滑对象
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    # 获得利用标签平滑的结果得到的损失计算方法
    loss = SimpleLossCompute(model.generator, criterion, model_optimizer)
    run(model, loss)