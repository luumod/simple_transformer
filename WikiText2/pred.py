import torch
from torchtext.data.utils import get_tokenizer
from torchtext.legacy import data,datasets
from pyitcast.transformer import TransformerModel

TEXT = data.Field(tokenize=get_tokenizer('basic_english'),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_text(model, start_str, max_len=100):
    model.eval()
    tokens = TEXT.numericalize([TEXT.tokenize(start_str)]).to(device)
    generated = []
    with torch.no_grad():
        for _ in range(max_len):
            output = model(tokens)
            next_token = output.argmax(dim=-1)[-1].item()
            generated.append(next_token)
            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)])
    return ' '.join([TEXT.vocab.itos[t] for t in generated])

if __name__ == '__main__':
    ntokens = len(TEXT.vocab.stoi)  # 词汇表大小必须一致
    emsize = 200  # 必须与训练参数一致
    nhead = 2  # 多头注意力头数
    nhid = 200  # 前馈层维度
    nlayers = 2  # 编码器层数
    dropout = 0.2

    model = TransformerModel(ntokens,emsize,nhead,nhid,nlayers,dropout).to(device)
    model.load_state_dict(torch.load("./models/WikiText2.pt"))
    # 示例：输入开头生成文本
    print(generate_text(model, "I want"))
    print(generate_text(model, "The book is really"))
    print(generate_text(model, "How are"))
    print(generate_text(model, "The future of AI"))