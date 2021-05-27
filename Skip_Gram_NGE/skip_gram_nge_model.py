import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.w_embeddings = nn.Embedding(vocab_size, embed_size)
        self.v_embeddings = nn.Embedding(vocab_size, embed_size)
        self._init_emb()

    def _init_emb(self):
        initrange = 0.5 / self.embed_size
        self.w_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_w, pos_v, neg_v):
        # emb_w = self.w_embeddings(torch.LongTensor(pos_w).cuda())  # 转为tensor 大小 [ mini_batch_size * emb_dimension ]
        # emb_v = self.v_embeddings(torch.LongTensor(pos_v).cuda())  # 转为tensor 大小 [ mini_batch_size * emb_dimension ]
        emb_w = self.w_embeddings(torch.LongTensor(pos_w))  # 转为tensor 大小 [ mini_batch_size * emb_dimension ]
        emb_v = self.v_embeddings(torch.LongTensor(pos_v))  # 转为tensor 大小 [ mini_batch_size * emb_dimension ]

        # neg_emb_v = self.v_embeddings(
        #     torch.LongTensor(neg_v).cuda())  # 转换后大小 [ negative_sampling_number * mini_batch_size * emb_dimension ]
        neg_emb_v = self.v_embeddings(
            torch.LongTensor(neg_v))  # 转换后大小 [ negative_sampling_number * mini_batch_size * emb_dimension ]
        # torch.mul() 是对应元素相乘，矩阵形状不变
        # score shape=[ mini_batch_size, emb_dimension ]
        # 我们希望 score 值越大越好，说明词向量越相近
        score = torch.mul(emb_w, emb_v)
        # 将sorce按行求和，shape=[mini_batch_size]
        score = torch.sum(score, dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = F.logsigmoid(score)

        # neg_score shape=[batch_size, negative_sampling_number, 1]
        neg_score = torch.bmm(neg_emb_v, emb_w.unsqueeze(2))
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        # 因为是负采样，我们希望neg_score越小越好，又因为加了负号，所以希望neg_score越大越好
        neg_score = F.logsigmoid(-1 * neg_score)
        # L = log sigmoid (Xw.T * θv) + ∑neg(v) [log sigmoid (-Xw.T * θneg(v))]
        # 这里score和neg_score的和都加了负号，所以loss越小越好
        loss = - torch.sum(score) - torch.sum(neg_score)
        return loss

    def save_embedding(self, id2word, file_name):
        embedding_1 = self.w_embeddings.weight.data.cpu().numpy()
        embedding_2 = self.v_embeddings.weight.data.cpu().numpy()
        embedding = (embedding_1 + embedding_2) / 2
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.embed_size))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


if __name__ == '__main__':
    model = SkipGramModel(100, 10)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    pos_w = [0, 0, 1, 1, 1]
    pos_v = [1, 2, 0, 2, 3]
    neg_v = [[23, 42, 32], [32, 24, 53], [32, 24, 53], [32, 24, 53], [32, 24, 53]]
    model.forward(pos_w, pos_v, neg_v)
    model.save_embedding(id2word, '../results/test.txt')
