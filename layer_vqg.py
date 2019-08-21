import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from beam_search import CaptionGenerator

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        d = torch.zeros(adj.size()).cuda()
        d.copy_(torch.sum(adj, 2).unsqueeze(2))
        output = torch.matmul(adj/d, support)
        # output = torch.stack([torch.spmm(norm[i], support[i]) for i in range(36)], 0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class conditional_GCN_1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, emb, dropout):
        super(conditional_GCN_1, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.embedding = nn.Linear(nhid, emb)
        self.fliter_layer = nn.Linear(emb, 1)
        self.dropout = dropout

    def fliter(self, mix_f):
        x = F.relu(self.embedding(mix_f))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.softmax(self.fliter_layer(x), 1).squeeze()
        topk_15 = torch.topk(x, 15, dim=1)[0][:,-1].unsqueeze(1).repeat(1, x.size(1))
        mask = torch.zeros(x.size()).cuda()
        x = torch.where(x > topk_15, x, mask)
        return x.unsqueeze(2)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # weight = self.fliter(x)
        # mask_ = torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).cuda()
        # adj_new = weight.repeat(1, 1, x.size(1)).mul(mask_) - mask_ + adj

        x = self.gc2(x, adj)
        return torch.mean(x, 1), adj


class conditional_GCN_2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, emb, dropout):
        super(conditional_GCN_2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.embedding = nn.Linear(nhid, emb)
        self.fliter_layer = nn.Linear(emb, 1)
        self.dropout = dropout

    def fliter(self, mix_f):
        x = F.relu(self.embedding(mix_f))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.softmax(self.fliter_layer(x), 1).squeeze()
        topk_15 = torch.topk(x, 15, dim=1)[0][:,-1].unsqueeze(1).repeat(1, x.size(1))
        mask = torch.zeros(x.size()).cuda()
        x = torch.where(x > topk_15, x, mask)
        return x.unsqueeze(2)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        weight = self.fliter(x)
        mask_ = torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).cuda()
        adj_new = weight.repeat(1, 1, x.size(1)).mul(mask_) - mask_ + adj

        x = self.gc2(x, adj_new)
        return torch.mean(x, 1), adj_new


class no_finding_area(nn.Module):
    def __init__(self, in_feature, hidden_feature, dropout):
        super(no_finding_area, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.dropout = dropout
        self.embedding = nn.Linear(in_feature, hidden_feature)
        self.fliter = nn.Linear(hidden_feature, 1)

    def forward(self, mix_f):
        x = F.relu(self.embedding(mix_f))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.sigmoid(self.fliter(x))
        mask = torch.zeros(x.size()).cuda()
        x = torch.where(x > 0.5, x, mask) # reserve nodes which score > 0.5
        return x.view(-1, 1)

class no_finding_area_top(nn.Module):
    def __init__(self, in_feature, hidden_feature, dropout):
        super(no_finding_area_top, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.dropout = dropout
        self.embedding = nn.Linear(in_feature, hidden_feature)
        self.fliter = nn.Linear(hidden_feature, 1)

    def forward(self, mix_f):
        x = F.relu(self.embedding(mix_f))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.sigmoid(self.fliter(x)).view(-1, 1296)
        topk_100 = torch.topk(x, 100, dim=1)[0][:, -1].unsqueeze(1).repeat(1, x.size(1))
        mask = torch.zeros(x.size()).cuda()
        x = torch.where(x > topk_100, x, mask)
        return x





class Abstract_Gen_Model(nn.Module):
    def __init__(self, vocab, opt):
        super(Abstract_Gen_Model, self).__init__()
        self.vocab = vocab
        self.start = vocab.get('START') if 'START' in vocab else None
        self.end = vocab.get('EOS')
        self.unk = vocab.get('UNK')
        self.classifier = nn.Linear(opt['dim_h'], len(self.vocab),  bias=False)
        self.embedder = nn.Embedding(len(self.vocab), opt['dim_embedding'])
        self.opt = opt
        if opt['share_weight']:
            assert opt['dim_embedding'] == opt['dim_h'], 'If share_weight is set, dim_embedding == dim_h required!'
            self.embedder.weight = self.classifier.weight # make sure the embeddings are from the final
        # initilization
        torch.nn.init.uniform(self.embedder.weight, -0.25, 0.25)


class question_gen(Abstract_Gen_Model):

    def __init__(self, vocab, opt):
        super(question_gen, self).__init__(vocab, opt)
        self.rnn = nn.LSTM(opt['dim_embedding'] + opt['dim_v'], opt['dim_h'], num_layers=opt['nb_layers'],
                           batch_first=True)

    def forward(self, v_feat, a_feat, questions):

        # prepare the data
        batch_size = questions.size(0)
        max_length = questions.size(1)
        new_ids, lengths, inv_ids = process_lengths_sort(questions.cpu().data, include_inv=True)
        new_ids = Variable(new_ids).detach()
        inv_ids = Variable(inv_ids).detach()

        padding_size = questions.size(1) - lengths[0]
        questions = questions.index_select(0, new_ids)
        v_feat = v_feat.index_select(0, new_ids)
        a_feat = a_feat.index_select(0, new_ids)
        embeddings = self.embedder(questions)
        v_feat = v_feat.unsqueeze(1).expand(batch_size, max_length, self.opt['dim_v'])
        embeddings = torch.cat([embeddings, v_feat], 2)  # each time step, input image and word embedding
        a_feat = a_feat.unsqueeze(1)
        _, hidden_feat = self.rnn(a_feat)
        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True)  # add additional image feature
        feats, _ = self.rnn(packed_embeddings, hidden_feat)

        pred = self.classifier(feats[0])
        pred = pad_packed_sequence([pred, feats[1]], batch_first=True)
        pred = pred[0].index_select(0, inv_ids)
        if padding_size > 0:
            pred = torch.cat(
                [pred, Variable(torch.zeros(batch_size, padding_size, pred.size(2)).type_as(pred.data)).detach()], 1)
        return pred


    def generate(self, v_feat, a_feat):
        batch_size = v_feat.size(0)
        max_length = self.opt['nseq'] if 'nseq' in self.opt else 20
        # x = Variable(torch.ones(1, batch_size,).type(torch.LongTensor) * self.start, volatile=True).cuda() # <start>
        output = Variable(torch.zeros(max_length, batch_size).type(torch.LongTensor)).cuda()
        scores = torch.zeros(batch_size)
        flag = torch.ones(batch_size)
        input_x = a_feat.unsqueeze(1)
        _, hidden_feat = self.rnn(input_x)  # initialize the LSTM
        x = Variable(torch.ones(batch_size, 1, ).type(torch.LongTensor) * self.start,
                     requires_grad=False).cuda()  # <start>
        v_feat = v_feat.unsqueeze(1)
        input_x = torch.cat([self.embedder(x), v_feat], 2)
        for i in range(max_length):
            output_feature, hidden_feat = self.rnn(input_x, hidden_feat)
            output_t = self.classifier(output_feature.view(batch_size, output_feature.size(2)))
            output_t = F.log_softmax(output_t)
            logprob, x = output_t.max(1)
            output[i] = x
            scores += logprob.cpu().data * flag
            flag[x.cpu().eq(self.end).data] = 0
            if flag.sum() == 0:
                break
            input_x = torch.cat([self.embedder(x.view(-1, 1)), v_feat], 2)
        return output.transpose(0, 1)



class question_gen_Base(Abstract_Gen_Model):
    def __init__(self, vocab, vocab_i2t, opt):
        super(question_gen_Base, self).__init__(vocab, opt)
        self.rnn = nn.LSTM(opt['dim_embedding'] + opt['dim_v'], opt['dim_h'], num_layers=opt['nb_layers'], batch_first=True)
        # self.rnn_d = nn.LSTM(opt['dim_embedding'] + opt['dim_v'], opt['dim_h'], num_layers=opt['nb_layers'], batch_first=True)
        self.vocab_i2t = vocab_i2t


    def forward(self, v_feat, a_feat, questions):
        # image feature tranform
        batch_size = questions.size(0)
        max_length = questions.size(1)
        new_ids, lengths, inv_ids = process_lengths_sort(questions.cpu().data, include_inv=True)
        new_ids = Variable(new_ids).detach()
        inv_ids = Variable(inv_ids).detach()
        # manually set the first length to MAX_LENGTH
        padding_size = questions.size(1) - lengths[0] - 1
        questions = questions.index_select(0, new_ids)[:, :-1]
        v_feat = v_feat.index_select(0, new_ids)
        a_feat = a_feat.index_select(0, new_ids)

        embeddings = self.embedder(questions)
        va_feat = torch.cat([a_feat, v_feat], 1).unsqueeze(1)
        embeddings = torch.cat([embeddings, v_feat.unsqueeze(1).expand(batch_size, max_length-1, self.opt['dim_v'])], 2)
        _, hidden_feat = self.rnn(va_feat)
        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True) # add additional image feature
        feats, _ = self.rnn(packed_embeddings, hidden_feat)

        pred = self.classifier(feats[0])
        pred = pad_packed_sequence(PackedSequence(pred, feats[1]), batch_first=True)

        # pred2 = pad_packed_sequence(feats, batch_first=True)
        # pred2 = self.classifier(pred2[0])
        pred = pred[0].index_select(0, inv_ids)
        if padding_size > 0: # to make sure the sizes of different patches are matchable
            pred = torch.cat([pred, Variable(torch.zeros(batch_size, padding_size, pred.size(2)).type_as(pred.data)).detach()], 1)
        return pred



    def generate(self, v_feat, a_feat):
        batch_size = v_feat.size(0)
        max_length = 15
        #x = Variable(torch.ones(1, batch_size,).type(torch.LongTensor) * self.start, volatile=True).cuda() # <start>
        output = Variable(torch.zeros(max_length, batch_size).type(torch.LongTensor)).cuda()
        scores = torch.zeros(batch_size)
        flag = torch.ones(batch_size)
        input_x = torch.cat([a_feat, v_feat], 1).unsqueeze(1)
        # input_x = va_feat.unsqueeze(1)
        _, hidden_feat = self.rnn(input_x) # initialize the LSTM
        x = Variable(torch.ones(batch_size, 1, ).type(torch.LongTensor) * self.start, requires_grad=False).cuda() # <start>
        embeddings = self.embedder(x)
        input_x = torch.cat([embeddings, v_feat.unsqueeze(1)], 2)
        for i in range(max_length):
            output_feature, hidden_feat = self.rnn(input_x, hidden_feat)
            output_t = self.classifier(output_feature.view(batch_size, output_feature.size(2)))
            output_t = F.log_softmax(output_t)
            logprob, x = output_t.max(1)
            output[i] = x
            scores += logprob.cpu().data * flag
            flag[x.cpu().eq(self.end).data] = 0
            if flag.sum() == 0:
                break
            input_x = self.embedder(x.view(-1, 1))
            input_x = torch.cat([input_x, v_feat.unsqueeze(1)], 2)
        return output.transpose(0, 1)



    def beam_search(self, v_feat, a_feat, beam_size=3, max_caption_length=15, length_normalization_factor=0.0,
                    include_unknown=False, sentence_word=False):
        batch_size = v_feat.size(0)
        assert batch_size == 1, 'Currently, the beam search only support batch_size == 1'
        input_x = torch.cat([a_feat, v_feat], 1).unsqueeze(1)
        _, hidden_feat = self.rnn(input_x)  # initialize the LSTM
        x = Variable(torch.ones(batch_size, 1, ).type(torch.LongTensor) * self.start, requires_grad=False).cuda()  # <start>
        embeddings = self.embedder(x)
        input_x = torch.cat([embeddings, v_feat.unsqueeze(1)], 2)
        cap_gen = CaptionGenerator(embedder=self.embedder,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.end,
                                   include_unknown=include_unknown,
                                   unk_id=self.unk,
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor,
                                   batch_first=True)
        sentences, score = cap_gen.beam_search(input_x, v_feat.unsqueeze(1), hidden_feat)
        sentence = sentences[0]
        sentence_word_ = [str(self.vocab_i2t[int(idx.cpu().numpy())]) for idx in sentence]
        while 'EOS' in sentence_word_:
            sentence_word_.remove('EOS')
        sentence = [' '.join(sentence_word_) + '.']
        if sentence_word:
            return sentence, sentence_word_
        else:
            return sentence


def process_lengths_sort(input, include_inv = False, cuda=True):
    # the input sequence should be in [batch x word]
    max_length = input.size(1) - 1 # remove additional START
    lengths = list(max_length - input.eq(0).sum(1).squeeze())
    lengths = [(i, lengths[i]) for i in range(len(lengths))]
    lengths.sort(key=lambda p:p[1], reverse=True)
    feat_id = [lengths[i][0] for i in range(len(lengths))]
    lengths = [min(max_length, lengths[i][1] + 1) for i in range(len(lengths))] # add additional word for EOS
    if include_inv:
        inv_id = torch.LongTensor(len(lengths))
        for i, i_id in enumerate(feat_id):
            inv_id[i_id] = i
        if cuda:
            return torch.LongTensor(feat_id).cuda(), lengths, torch.LongTensor(inv_id).cuda()
        else:
            return torch.LongTensor(feat_id), lengths, torch.LongTensor(inv_id)
    else:
        if cuda:
            return torch.LongTensor(feat_id).cuda(), lengths
        else:
            return torch.LongTensor(feat_id), lengths


