import math
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.legacy import data
from pyitcast.transformer import TransformerModel
from torchtext.legacy import datasets


TEXT = data.Field(tokenize=get_tokenizer('basic_english'),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

# 导入wikitext2数据
train_txt, val_txt, test_txt = datasets.WikiText2.splits(TEXT)

TEXT.build_vocab(train_txt)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 构建批次数量的函数
def batchify(data, batch_size):
     '''
     :param data: 之前得到的文本数据
     :param batch_size: 批次样本数量
     '''
     # 将单词映射为对应的连续数字
     data = TEXT.numericalize([data.examples[0].text])
     # 经过多少batch_size后能够取得所有数据:
     nbatch = data.size(0) // batch_size
     # 利用narrow对数据进行切割: n*1
     data = data.narrow(0,0,nbatch * batch_size)
     # 改变data的形状: m*batch_size
     data = data.view(batch_size, -1).t().contiguous()
     return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size) # nbatch * batch_size
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

# 句子的最大长度
bptt = 35
def get_batch(source, i):
    '''
    :param source: 数据
    :param i: 批次数量
    '''
    # 确定句子的长度值
    seq_len = min(bptt, len(source) - 1 - i)
    # 首先得到源数据
    data = source[i:i+seq_len]
    # 得到目标数据
    target = source[i+1:i+seq_len+1].view(-1)
    return data, target

# source = test_data
# i = 1
# x,y = get_batch(source, i)
# print(x)
# print(y)

# 获得词汇表中的词汇的数量
ntokens = len(TEXT.vocab.stoi)

# 词嵌入维度
emsize = 200

# 前馈全连接层的节点数
nhid = 200

# 编码器层的层数
nlayers = 2

# 多头注意力机制的头数
nhead = 2

# 置零比率
dropout = 0.2

# 模型实例化
model = TransformerModel(ntokens,emsize,nhead,nhid,nlayers,dropout).to(device)

# 设置损失函数
criterion = nn.CrossEntropyLoss()

# 设置学习率
lr = 5.0

# 优化器
optim = torch.optim.SGD(model.parameters(), lr=lr)

# 学习率调整器
scheduler = torch.optim.lr_scheduler.StepLR(optim, 1.0, 0.95)

# 训练模型
def train():
    model.train()
    total_loss = 0
    start_time = time.time()
    # 200个批次打印一次日志
    log_interval = 200
    # 遍历训练数据
    for batch, i in enumerate(range(0,train_data.size(0) - 1,bptt)):
        data, targets = get_batch(train_data, i)
        optim.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.view(-1, ntokens), targets)
        loss.backward()
        # 梯度规范化，防止梯度爆炸或者消失
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
        optim.step()
        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            # 计算平均损失
            cur_loss = total_loss / log_interval
            # 计算耗时（单位秒）
            elapsed = time.time() - start_time
            # 打印日志
            print('| epoch {:3d} | {:5d}/{:5d} batches |'
                  'lr {:02.2f} | ms/batch {:5.2f} |'
                  'loss {:5.2f} | ppl {:8.2f}'.format(epoch,batch, len(train_data) // bptt,
                                                      scheduler.get_lr()[0],elapsed * 1000 / log_interval,
                                                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    '''
    :param eval_model: 每轮训练后的模型
    :param data_source: 验证集或者测试集数据
    '''
    eval_model.eval()
    total_loss = 0
    with torch.no_grad():
        # 遍历验证数据
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            outputs = eval_model(data)
            outputs_flat = outputs.view(-1, ntokens)
            total_loss += criterion(outputs_flat, targets).item()
    return total_loss

best_val_loss = float('inf')

# 训练轮次
epochs = 3

# 定义最佳模型，初始为空
best_model = None

for epoch in range(epochs):
    # 获取当前轮次的开始时间
    epoch_start_time = time.time()
    # 直接训练
    train()
    # 调用评估函数得到损失
    val_loss = evaluate(model, val_data)
    print('-' * 90)
    print('| end of epoch {:3d} | time {:5.2f} | valid_loss {:5.2f} | '
          .format(epoch, (time.time() - epoch_start_time), val_loss))
    print('-' * 90)
    # 获取最佳损失及模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
    # 每个轮次后调整优化器学习率
    scheduler.step()

test_loss = evaluate(best_model, test_data)
print('-' * 90)
print('| End of training | test loss: {:5.2f}'.format(test_loss))
print('-' * 90)

if not os.path.isdir('./models'):
    os.mkdir('./models')

torch.save(best_model.state_dict(), './models/WikiText2.pt')
print('模型保存成功！')