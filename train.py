import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = 'cpu'

# ———————————————————————————————————————————定义数据集———————————————————————————————————————————
# 手动输入了两对中文→英语的句子
# 每个字的索引手动编码
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

# 训练集
sentences = [
    # 中文和英语的单词个数不要求相同
    # enc_input                dec_input           dec_output
    ['我 有 一 个 好 朋 友 P', 'S I have a good friend .', 'I have a good friend . E'],
    ['我 有 零 个 女 朋 友 P', 'S I have zero girl friend .', 'I have zero girl friend . E'],
    ['我 有 一 个 男 朋 友 P', 'S I have a boy friend .', 'I have a boy friend . E']
]

# 测试集（希望transformer能达到的效果）
# 输入："我 有 一 个 女 朋 友"
# 输出："i have a girlfriend"

# 中文和英语的单词要分开建立词库
# Padding Should be Zero
src_vocab = {'P': 0, '我': 1, '有': 2, '一': 3,
             '个': 4, '好': 5, '朋': 6, '友': 7, '零': 8, '女': 9, '男': 10}
src_idx2word = {i: w for i, w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'I': 1, 'have': 2, 'a': 3, 'good': 4,
             'friend': 5, 'zero': 6, 'girl': 7,  'boy': 8, 'S': 9, 'E': 10, '.': 11}
tgt_idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 8  # （源句子的长度）enc_input max sequence length
tgt_len = 7  # dec_input(=dec_output) max sequence length

print("src_idx2word: {}".format(src_idx2word))
print("tgt_idx2word: {}".format(tgt_idx2word))



def make_data(sentences):
    """
    把单词序列转化为数字序列
    :param sentences: 样本list，[[enc_input1, dec_input1, dec_output1],[enc_input2, dec_input2, dec_output2]]
    :return: 
    """
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [src_vocab[n] for n in sentences[i][0].split()]
        dec_input = [tgt_vocab[n] for n in sentences[i][1].split()]
        dec_output = [tgt_vocab[n] for n in sentences[i][2].split()]

        #[[1, 2, 3, 4, 5, 6, 7, 0], [1, 2, 8, 4, 9, 6, 7, 0], [1, 2, 3, 4, 10, 6, 7, 0]]
        enc_inputs.append(enc_input)
        #[[9, 1, 2, 3, 4, 5, 11], [9, 1, 2, 6, 7, 5, 11], [9, 1, 2, 3, 8, 5, 11]]
        dec_inputs.append(dec_input)
        #[[1, 2, 3, 4, 5, 11, 10], [1, 2, 6, 7, 5, 11, 10], [1, 2, 3, 8, 5, 11, 10]]
        dec_outputs.append(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


class MyDataset(Data.Dataset):
    """
    自定义DataLoader
    """
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataset, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
    
    def __len__(self):
        return self.enc_inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
loader = Data.DataLoader(MyDataset(enc_inputs, dec_inputs, dec_outputs), batch_size=2, shuffle=True)


# ———————————————————————————————————————————定义模型———————————————————————————————————————————
vocab_params ={
	'src_vocab_size': src_vocab_size,   # 输入词表大小
	'tgt_vocab_size': tgt_vocab_size    # 输出词表大小
}

enc_dec_params ={
	'n_layers': 6   # Encoder和Decoder的层数
}

attn_params = {
	'n_heads': 8,    # 多头注意力的头数,
	'd_k': 64,       # 注意力中Q和K的嵌入维度，Q和K的嵌入维度必须相等，不然Q和K.T无法矩阵相乘
	'd_v': 64        # 注意力中V的嵌入维度，可以跟d_k不同，为了方便一般取相同
}

common_net_params = {
	'd_model': 512,     # token Embedding和position Encoding的嵌入维度
	'd_ff': 2048,       # 前馈层中隐藏层的维度
	'dropout': 0.1      # 置零比率，nn.Dropout(p=dropout)
}

# 定义模型
model = Transformer(vocab_params, 
					enc_dec_params, 
					attn_params,
					common_net_params).to(device)

# ———————————————————————————————————————————训练———————————————————————————————————————————
# nn.CrossEntropyLoss()为交叉熵损失函数，用于解决多分类问题，也可以用于解决二分类问题。在nn.CrossEntropyLoss()其内部已经包含了softmax层
# BCELoss是Binary CrossEntropyLoss的缩写，nn.BCELoss()为二元交叉熵损失函数，只能解决二分类问题，在使用nn.BCELoss()作为损失函数时，需要在该层前面加上Sigmoid函数，一般使用nn.Sigmoid()即可
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
epochs = 20

for epoch in range(epochs):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        """
        enc_inputs:[batch_size, src_len]
        dec_inputs:[batch_size, tgt_len]
        dec_outputs:[batch_size, tgt_len]
        """
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        # outputs:[batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        # dec_outputs.view(-1):[batch_size*tgt_len]
        loss = criterion(outputs, dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ———————————————————————————————————————————预测———————————————————————————————————————————
def greedy_decoder(model, enc_input, start_symbol):
    """
    为简单起见，当 K=1 时，贪婪解码器即为 Beam 搜索。这在推理中是必要的，因为我们不知道目标序列输入。因此，我们尝试逐个生成目标输入单词，然后将其馈送到 Transformer 模型中。起始参考链接：http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol.
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    # decoder的第一个输入为start_symbol的标识符，去tgt_vocab取对应的序号，构造初始输入
    dec_input = torch.tensor([[tgt_vocab[start_symbol]]]).type_as(enc_input.data)
    terminal = False
    # 存储本次预测出的下一个词，初始是一个空值
    next_word = torch.zeros(1,0).type_as(enc_input.data)
    while not terminal:
        # 预测阶段：dec_input序列会一点点变长（每次增加一个新预测出来的单词）
        dec_input = torch.cat([dec_input.to(device), next_word.to(device)], -1)
        # dec_outputs: [1, tgt_len, d_model]
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        # projected: [1, tgt_len, tgt_vocab_size]
        projected = model.projection(dec_outputs)
        # 取tgt_vocab_size中最大的值，并取返回值中的index部分
        prob = projected.max(dim=-1, keepdim=False)[1]
        # 增量更新（我们希望重复单词预测结果是一致的）
        # 我们在预测时会选择性忽略重复的预测的词，只摘取最新预测的单次拼接到输入序列中
        # 拿出当前预测的单词(数字），我们用x'_t对应的输出z_t去预测下一个单次的概率， 不用z_1,z_2..z_{t-1}
        next_word = prob.data[-1][-1]
        if next_word == tgt_vocab["E"]:
            terminal = True
        next_word = torch.tensor([[next_word]], dtype=dec_input.dtype)
        print("—————————————————————————")
        print(next_word)
        print(dec_input)
        
    return dec_input

# 预测阶段
# 测试集
sentences = [
    # [enc_input, dec_input, dec_output]
    ['我 有 一 个 女 朋 友 P', '', '']
]
enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
test_loader = Data.DataLoader(
    MyDataset(enc_inputs, dec_inputs, dec_outputs), 2, True)
enc_inputs, _, _ = next(iter(test_loader))

print("利用训练好的Transformer模型将中文句子'我 有 一 个 女 朋 友' 翻译成英文句子: ")
greedy_dec_predict = greedy_decoder(model, enc_inputs.to(device), start_symbol="S")
print([src_idx2word[t.item()] for t in enc_inputs.squeeze(0)], '->',
      [tgt_idx2word[n.item()] for n in greedy_dec_predict.squeeze(0)])





































