#    Copyright 2018 AimBrain Ltd.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this fi le except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import torch
from torch.autograd import Variable


def batch_to_cuda(batch, volatile=False):
    # moves dataset batch on GPU

    q = Variable(batch[0],  requires_grad=False).cuda()
    a = Variable(batch[1],  requires_grad=False).cuda()
    img_feat = Variable(batch[2],  requires_grad=False).cuda()
    adj_mat = Variable(batch[3],  requires_grad=False).cuda()
    #n_votes = Variable(batch[2], volatile=volatile, requires_grad=False).cuda()
    #i = Variable(batch[4], volatile=volatile, requires_grad=False).cuda()
    #k = Variable(batch[5], volatile=volatile, requires_grad=False).cuda()
    #qlen = list(batch[6])

    return q, a, img_feat, adj_mat


def save(model_gcn, model_vqg, optimizer, ep, epoch_loss, epoch_acc, dir, name):
    # saves model and optimizer state

    tbs = {
        'epoch': ep + 1,
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'state_dict_gcn': model_gcn.state_dict(),
        'state_dict_vqg': model_vqg.state_dict(),
        'optimizer': optimizer.state_dict()
        }
    torch.save(tbs, os.path.join(dir, name + '.pth.tar'))

def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]
    return dict_to


def total_vqa_score(output_batch, n_votes_batch):
    # computes the total vqa score as assessed by the challenge

    vqa_score = 0
    _, oix = output_batch.data.max(1)
    for i, pred in enumerate(oix):
        count = n_votes_batch[i,pred]
        vqa_score += min(count.data.cpu().numpy()[0]/3, 1)
    return vqa_score


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    num_word = targets.size(1)
    correct_zero = targets.eq(0).view(-1).float().sum()-batch_size
    _, ind = scores.topk(k, 2, True, True)
    correct = ind.eq(targets.unsqueeze(2).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return (correct_total.item()-correct_zero.item()) * (100.0 / (batch_size*num_word-correct_zero.item()))


def find_area(answer, clses):
    # find the area of the image related to the answer

    n_answer = answer.size(0)
    sim_all = torch.zeros(clses.size(0), n_answer*2)

    for i in range(n_answer):
        w1 = answer[i].repeat(clses.size(0), 1).float()
        # w2 = torch.chunk(clses.float(), 2, 1)
        for j in range(2):
            w2 = clses[:, j*300:(j+1)*300].float()
            sim_all[:, 2*i+j] = get_sim(w1, w2)
    sim_max, _ = torch.max(sim_all, 1)
    _, idx = torch.max(sim_max, 0)

    return idx

        # for j in range(clses.size(0)):
        #     w2 = clses[j]
        #     sim = get_sim(w1, w2)
        #     if  sim > sim_max:
        #         sim_max = sim


def get_sim(answer, cls):
    assert answer.size() == cls.size()
    a = cosine_sim(answer, cls)
    return cosine_sim(answer, cls)


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps))


def idx2question(idx, idx_ref, question_vocab):
    gen_sentences = []
    ref_sentences = []

    for i in range(idx.shape[0]):
        words = [str(question_vocab.get(int(index))) for index in idx[i].tolist()]
        while 'EOS' in words:
            words.remove('EOS')
        sentence = [' '.join(words) + '.']
        gen_sentences.append(sentence)

    gen = {i:s for i,s in enumerate(gen_sentences)}

    for i in range(idx_ref.shape[0]):
        words = [str(question_vocab.get(int(index))) for index in idx_ref[i].tolist()]
        while 'EOS' in words:
            words.remove('EOS')
        while 'START' in words:
            words.remove('START')
        sentence = [' '.join(words) + '.']
        ref_sentences.append(sentence)

    ref = {i:s for i,s in enumerate(ref_sentences)}

    return gen, ref


def ref2question(idx_ref, question_vocab):
    ref_sentences = []


    for i in range(idx_ref.shape[0]):
        words = [str(question_vocab.get(int(index))) for index in idx_ref[i].tolist()]
        while 'EOS' in words:
            words.remove('EOS')
        while 'START' in words:
            words.remove('START')
        sentence = [' '.join(words) + '.']
        ref_sentences.append(sentence)

    ref = {i:s for i,s in enumerate(ref_sentences)}

    return ref


def save_the_best(bleu, cider, meteor, rouge, bleu_best, cider_best, meteor_best, rouge_best):
    flag = 0
    for i,score in enumerate(bleu):
        if score >= bleu_best[i]:
            flag+=1
    if flag == 4:
        print('a better bleu !')
        bleu_best = bleu

    if cider >= cider_best:
        flag += 1
        cider_best = cider
    if meteor >= meteor_best:
        flag += 1
        meteor_best = meteor
    if rouge >= rouge_best:
        flag += 1
        rouge_best = rouge
    if flag >= 4:
        choice = True
    else:
        choice = False

    return bleu_best, cider_best, meteor_best, rouge_best, choice


