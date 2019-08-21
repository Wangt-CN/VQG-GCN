from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import json
import numpy as np
import yaml
import argparse
from tqdm import tqdm
try:
    import cPickle as pickle
except:
    import pickle as pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from dataset_vqg import collate_fn
from sparse_graph_model import Model
from ZS_VQA.VQG import *
import utils
import main
from layer_vqg import question_gen_Base as question_gen
from layer_vqg import conditional_GCN_2 as conditional_GCN


def test(args):
    """
        Creates a result.json for predictions on
        the test set
    """
    # Check that the model path is accurate
    if args.model_path and os.path.isfile(args.model_path):
        print('Resuming from checkpoint %s' % (args.model_path))
    else:
        raise SystemExit('Need to provide model path.')

    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, script requires CUDA')

    print('Loading data')
    supp_train = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/data/vqa2_zsa_trainsupp.json'))
    dataset = VQA_Dataset_Test(args.data_dir, args.emb, train=False)
    loader = DataLoader(dataset, batch_size=args.bsize,
                        shuffle=False, num_workers=0,
                        collate_fn=collate_fn)

    # Print data and model parameters
    print('Parameters:\n\t'
          'vocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d'
          '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words, args.emb,
                                                    dataset.feat_dim,
                                                    args.hid,
                                                    dataset.n_answers))

    # Define model
    question_vocab = pickle.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/data/train_q_dict.p', 'rb'))

    model_gcn = conditional_GCN(nfeat=options['gcn']['nfeat'],
                                nhid=options['gcn']['nhid'],
                                nclass=options['gcn']['nclass'],
                                dropout=options['gcn']['dropout'])

    model_vqg = question_gen(vocab=question_vocab['wtoi'], vocab_i2t=question_vocab['itow'], opt=options['vqg'])

    # move to CUDA
    model_gcn = model_gcn.cuda()
    model_vqg = model_vqg.cuda()

    # Restore pre-trained model
    ckpt = torch.load(args.model_path)
    model_gcn.load_state_dict(ckpt['state_dict_gcn'])
    model_vqg.load_state_dict(ckpt['state_dict_vqg'])
    model_gcn.train(False)
    model_vqg.train(False)

    answer = []
    ref_q = []
    new_trainset = []
    answer_bs3 = {}
    answer_bs5 = {}
    flag = 0
    for step, next_batch in tqdm(enumerate(loader)):
        # Batch preparation
        target_q, an_feat, img_feat, adj_mat = \
            utils.batch_to_cuda(next_batch, volatile=True)

        # get predictions
        feat_gcn = model_gcn(img_feat, adj_mat)
        output = model_vqg.generate(feat_gcn, an_feat)

        for ii in range(feat_gcn.size(0)):

            output_bs3_, sentence_word= model_vqg.beam_search(feat_gcn[ii].unsqueeze(0),
                                                an_feat[ii].unsqueeze(0), sentence_word=True)

            answer_bs3[flag] = output_bs3_

            new_pair = supp_train[flag]
            new_pair['question'] = output_bs3_[0]
            new_pair['question_toked_UNK'] = sentence_word
            question_toked = [0]*16
            question_tokedd = [question_vocab['wtoi'].get('START')] + [question_vocab['wtoi'].get(word) for word in sentence_word]
            question_toked[:len(question_tokedd)] = question_tokedd
            new_pair['question_wids'] = question_toked
            new_pair['seq_length'] = len(sentence_word)
            new_trainset.append(new_pair)
            flag += 1

    json.dump(new_trainset, open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/data/zsa_trainingset.json', 'w'))







if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Conditional Graph Convolutions for VQA')
    parser.add_argument('--train', action='store_true',
                        help='set this to training mode.')
    parser.add_argument('--trainval', action='store_true',
                        help='set this to train+val mode.')
    parser.add_argument('--eval', action='store_true',
                        help='set this to evaluation mode.')
    parser.add_argument('--test', action='store_true',
                        help='set this to test mode.')
    parser.add_argument('--lr', metavar='', type=float,
                        default=1e-4, help='initial learning rate')
    parser.add_argument('--ep', metavar='', type=int,
                        default=40, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int,
                        default=64, help='batch size.')
    parser.add_argument('--hid', metavar='', type=int,
                        default=1024, help='hidden dimension')
    parser.add_argument('--emb', metavar='', type=int, default=300,
                        help='question embedding dimension')
    parser.add_argument('--neighbourhood_size', metavar='', type=int, default=16,
                        help='number of graph neighbours to consider')
    parser.add_argument('--data_dir', metavar='', type=str, default='/mnt/data/xiaojinhui/wangtan_MM/vqa-project/data',
                        help='path to data directory')
    parser.add_argument('--save_dir', metavar='', type=str, default='/mnt/data/xiaojinhui/wangtan_MM/vqa-project/model')
    parser.add_argument('--path_opt', metavar='', type=str, default='')
    parser.add_argument('--name', metavar='', type=str,
                        default='model', help='model name')
    parser.add_argument('--dropout', metavar='', type=float, default=0.5,
                        help='probability of dropping out FC nodes during training')
    parser.add_argument('--model_path', metavar='', type=str, default='/mnt/data/xiaojinhui/wangtan_MM/vqa-project/model/model_10_v1.pth.tar'
                        ,help='trained model path.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))

    options = {
        'gcn': {
            'nfeat': 2652,
            'nhid': 1024,
            'nclass': 1024,
            'dropout': 0.5
        },
        'vqg': {
            'share_weight': 0,
            'dim_embedding': 300,
            'dim_h': 2048,
            'dim_v': 1024,
            'nb_layers': 1
        },
    }

    # if args.path_opt is not None:
    #     with open(args.path_opt, 'r') as handle:
    #         options_yaml = yaml.load(handle)
    #     options = utils.update_values(options, options_yaml)


    if args.test:
        test(args)
    if not args.train and not args.eval and not args.trainval and not args.test:
        parser.print_help()