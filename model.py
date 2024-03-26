import pandas as pd
import numpy as np
import os
import pickle
import spacy
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score, classification_report

dep_path = 'E:\\using_csv\\Suicide_Detection.csv'
emo_path = 'E:\\using_csv\\isear.csv'
dep_f = pd.read_csv(dep_path)
emo_f = pd.read_csv(emo_path)
dep_texts, dep_labels = dep_f['text'][0:3000], dep_f['class'][0:3000]
emo_texts, emo_labels = emo_f['text'][0:3000], emo_f['label'][0:3000]

sp_nlp = spacy.load('en_core_web_sm') #英文预料词库, sm/md/lg, 分词/词意标注

def load_word_vec(path, word2idx): #word2idx:字典,记录单词在idx2word中的下标
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.strip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32') #list转化成dict
            except:
                print('WARNING: corrupted word vector of {} when being loaded from GloVe.'.format(tokens[0]))
    return word_vec

def build_embedding_matrix(word2idx, embed_dim, type): #embed_dim 词向量维度
    embedding_matrix_file_name = 'E:\\using_csv\\{0}_{1}_embedding_matrix.pk1'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix...:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
        #print(embedding_matrix)
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim)) #均匀分布的区域中随机采样

        fname = 'E:\\using_csv\\glove.42B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                #word没被找到，index为0
                embedding_matrix[i] = vec
        wb = open(embedding_matrix_file_name, 'wb')
        pickle.dump(embedding_matrix, wb )
    return embedding_matrix

class Tokenizer(object):  #分词器
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx #补全字符
            self.idx += 1
            self.word2idx['UNK'] = self.idx #未在词表中的词
            self.idx2word[self.idx] = 'UNK'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):  #适配文本
        words = []

        for x in text:
            x = x.lower().strip()
            xx = sp_nlp(x)
            words = words + [str(y) for y in xx]

        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower().strip()
        words = sp_nlp(text)
        words = [str(x) for x in words]
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]

        if len(sequence) == 0:
            sequence = [0]
        return sequence


if os.path.exists("suicide" + '_word2idx.pk1'):
    print("loading suicide tokenizer...")
    with open("suicide" + '_word2idx.pk1', 'rb') as f:
        print(f)
        word2idx = pickle.load(f)
        tokenizer = Tokenizer(word2idx=word2idx)
else:
    tokenizer = Tokenizer()
    tokenizer.fit_on_text(dep_texts)
    with open("suicide" + '_word2idx.pk1', 'wb') as f: #open file to write when first
        pickle.dump(tokenizer.word2idx, f)

label2int_dep = {'suicide':0, 'non-suicide':1}
label2int_emo = {'anger': 0, 'disgust': 1, 'fear': 2, 'guilt': 3, 'joy': 4, 'sadness':5, 'shame':6 }

def read_data(texts, labels, max_seq_len = -1, flag = True):
    all_data = []
    label2int = {}

    for idx, i in enumerate(set(labels)):
        label2int[i] = idx
    for i in tqdm(range(0, len(texts))):
        context = texts[i].lower().strip()
        context_indices = tokenizer.text_to_sequence(context)

        label = int(label2int[labels[i]])
        if max_seq_len > 0:
            if len(context_indices) < max_seq_len:
                context_indices = context_indices + [0] * (max_seq_len - len(context_indices))
            else:
                context_indices = context_indices[:max_seq_len]
        all_data.append([context_indices, label])
    return all_data

dep_data = read_data(dep_texts, dep_labels, max_seq_len=30, flag = True)
emo_data = read_data(emo_texts, emo_labels, max_seq_len=30, flag = False)

class MyDataSet(Dataset):
    def __init__(self, texts, labels):
        super(MyDataSet, self).__init__()
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

dep_new_texts = []
dep_new_labels = []
for i in dep_data:
    xx = []
    for j in i[0]:
        if type(j) == int:
            xx.append(j)
        else:
            xx.append(j[0])
    dep_new_texts.append(xx)
    dep_new_labels.append(i[1])

dep_tensor_texts = [torch.tensor(i) for i in dep_new_texts]
dep_tensor_labels = [torch.tensor(i) for i in dep_new_labels]

dep_dataset = MyDataSet(dep_tensor_texts, dep_tensor_labels)
dep_train_loader = DataLoader(dep_dataset, batch_size=32, shuffle=True) #return iterator

emo_new_texts = []
emo_new_labels = []
for i in emo_data:
    xx = []
    for j in i[0]:
        if type(j) == int:
            xx.append(j)
        else:
            xx.append(j[0])
    emo_new_texts.append(xx)
    emo_new_labels.append(i[1])

emo_tensor_texts = [torch.tensor(i) for i in emo_new_texts]
emo_tensor_labels = [torch.tensor(i) for i in emo_new_labels]

emo_dataset = MyDataSet(emo_tensor_texts, emo_tensor_labels)
emo_train_loader = DataLoader(emo_dataset, batch_size=32, shuffle=True)


class MTL(nn.Module):
  def __init__(self,w):
    super().__init__()
    self.emb = nn.Embedding.from_pretrained(w) #加载预训练好的词向量进行嵌入
    self.label_encoder = nn.Linear(in_features=300,out_features=125,bias=True)
    self.cr_encoder = nn.GRU(input_size=300,hidden_size=125,batch_first=True)
    self.coattn_0 = nn.MultiheadAttention(embed_dim=125,num_heads=1, batch_first=True)
    self.coattn_1 = nn.MultiheadAttention(embed_dim=125,num_heads=1, batch_first=True)
    self.d_cls = nn.Linear(in_features=125,out_features=2,bias=True) #二分类
    self.e_cls = nn.Linear(in_features=125,out_features=7,bias=True) #七分类

  def forward(self, dx, ex):
    dx = self.emb(dx).float()
    ex = self.emb(ex).float()

    t_emo_l = torch.rand((7,300), requires_grad=True, device='cuda')

    _, vd = self.cr_encoder(dx)
    _, ve = self.cr_encoder(ex)
    vl = self.label_encoder(t_emo_l)

    vd = vd.view(vd.size()[1],vd.size()[0],-1)
    ve = ve.view(ve.size()[1],ve.size()[0],-1)

    vl = torch.stack([vl] * vd.size()[0], dim=0)

    vdl,_ = self.coattn_0(vd, vl, vl)
    vel,_ = self.coattn_1(ve, vl, vl)

    y_d = self.d_cls(vdl)
    y_e = self.e_cls(vel)

    y_d = y_d.view(y_d.size()[0],-1)
    y_e = y_e.view(y_e.size()[0],-1)

    return y_d, y_e

embedding_matrix = build_embedding_matrix(tokenizer.word2idx, 300, "suicide")
embs = np.array(embedding_matrix)
pt_embs = torch.tensor(embs)
mtl = MTL(pt_embs)

optimizer = torch.optim.AdamW(mtl.parameters(),lr = 1e-3, eps = 1e-8)  #mtl.parameters()待优化
loss_ = torch.nn.CrossEntropyLoss()  #交叉熵损失

epoches = 200

#启动GPU
mtl.cuda()
print("start training ...")

# 训练模式
mtl.train()
for e in range(epoches):

    # 记录损失函数
    loss_rec = []

    # 将抑郁症以及情绪分类的训练集，同时输入到模型中进行训练
    for idx, (b_dep, b_emo) in enumerate(zip(dep_train_loader, emo_train_loader)): #(index, (dep_data, emo_data))
        # x - batch size, sequence len
        dep_x = b_dep[0].cuda()
        dep_y = b_dep[1].cuda()

        emo_x = b_emo[0].cuda()
        emo_y = b_emo[1].cuda()

        y_d, y_e = mtl(dep_x, emo_x)

        optimizer.zero_grad()
        #计算抑郁症检测任务的损失函数，以及情绪分类任务的损失函数
        dep_loss = loss_(y_d, dep_y)
        emo_loss = loss_(y_e, emo_y)

        loss = dep_loss + 0.001 * emo_loss

        loss_rec.append(loss.cpu().item())

        loss.backward()
        optimizer.step()


#停止训练
mtl.eval()

dep_test_loader = DataLoader(dep_dataset, batch_size=32, shuffle=False)

results = []

with torch.no_grad():
    for idx, (b_dep, b_emo) in enumerate(zip(dep_test_loader, emo_train_loader)):
            # x - batch size, sequence len
            dep_x = b_dep[0].cuda()
            dep_y = b_dep[1].cuda()

            emo_x = b_emo[0].cuda()
            emo_y = b_emo[1].cuda()

            y_d, y_e = mtl(dep_x, emo_x)

            results.append(y_d.detach().cpu())


v_results = torch.vstack(results)
preds = v_results.argmax(1).numpy().tolist()

num_dep_labels = [label2int_dep[i] for i in dep_labels]

print(accuracy_score(num_dep_labels, preds))
print(precision_score(num_dep_labels, preds))
print(recall_score(num_dep_labels, preds))
print(f1_score(num_dep_labels, preds))
print(classification_report(num_dep_labels, preds))

test_texts, test_labels = dep_f['text'][3000:5000].values.tolist(), dep_f['class'][3000:5000].values.tolist()

test_data = read_data(test_texts, test_labels, max_seq_len=30)

test_new_texts = []
test_new_labels = []

for i in test_data:
    xx = []
    for j in i[0]:
        if type(j) == int:
            xx.append(j)
        else:
            xx.append(j[0])
    test_new_texts.append(xx)
    test_new_labels.append(i[1])

test_tensor_texts = [torch.tensor(i) for i in test_new_texts]
test_tensor_labels = [torch.tensor(i) for i in test_new_labels]

test_dataset = MyDataSet(test_tensor_texts, test_tensor_labels)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

eemo_tensor_texts = [torch.tensor(i) for i in emo_new_texts[0:2000]]
eemo_tensor_labels = [torch.tensor(i) for i in emo_new_labels[0:2000]]

eemo_dataset = MyDataSet(eemo_tensor_texts, eemo_tensor_labels)
emo_test_loader = DataLoader(eemo_dataset, batch_size=32, shuffle=False)

mtl.eval()

rresults = []

with torch.no_grad():
    for idx, (b_dep, b_emo) in enumerate(zip(test_loader, emo_test_loader)):
            # x - batch size, sequence len
            dep_x = b_dep[0].cuda()
            dep_y = b_dep[1].cuda()

            emo_x = b_emo[0].cuda()
            emo_y = b_emo[1].cuda()

            y_d, y_e = mtl(dep_x, emo_x)

            rresults.append(y_d.detach().cpu())

vv_results = torch.vstack(rresults)
ppreds = vv_results.argmax(1).numpy().tolist()


nnum_dep_labels = [label2int_dep[i] for i in test_labels]

print(accuracy_score(nnum_dep_labels, ppreds))
print(precision_score(nnum_dep_labels, ppreds))
print(recall_score(nnum_dep_labels, ppreds))
print(f1_score(nnum_dep_labels, ppreds))
print(classification_report(nnum_dep_labels, ppreds))

y = loss_rec
x = list(range(1,len(y)+1))

plt.plot(x,y,color = 'r',label="Total-Loss")#s-:方形
plt.xlabel("Epoch")#横坐标名字
plt.ylabel("Loss")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()