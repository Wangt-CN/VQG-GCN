from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from torch.nn.utils import clip_grad_norm_
import os
import json
import numpy as np
import yaml
import argparse
import time
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
from dataset_vqg import *
import utils
import main
from layer_vqg import question_gen_Base as question_gen
from layer_vqg import conditional_GCN_2 as conditional_GCN
import layer_vqg

os.environ["CUDA_VISIBLE_DEVICES"] = "1"




def train(args):
    """
        Train a VQG model using the training set and validate on val set.
    """


    # Load the VQA training set
    print('Loading data...')
    dataset = VQA_Dataset(args.data_dir, args.emb)
    loader = DataLoader(dataset, batch_size=args.bsize,
                        shuffle=True, num_workers=0, collate_fn=collate_fn)

    # Load the VQA validation set
    dataset_test = VQA_Dataset(args.data_dir, args.emb, train=False)
    loader_val = DataLoader(dataset_test,
                                  batch_size=args.bsize,
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=collate_fn)

    n_batches = len(dataset) // args.bsize
    question_vocab = pickle.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/data/train_q_dict.p', 'rb'))

    # Print data and model parameters
    print('Parameters:\n\t'
          'vocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d'
          '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.n_answers, args.emb,
                                                    dataset.feat_dim,
                                                    args.hid,
                                                    dataset.n_answers))

    print('Initializing model')
    model_gcn = conditional_GCN(nfeat=options['gcn']['nfeat'],
                                nhid=options['gcn']['nhid'],
                                nclass=options['gcn']['nclass'],
                                emb = options['gcn']['fliter_emb'],
                                dropout=options['gcn']['dropout'])
    # model_gcn_nofinding = layer_vqg.conditional_GCN_1(nfeat=options['gcn']['nfeat'],
    #                             nhid=options['gcn']['nhid'],
    #                             nclass=options['gcn']['nclass'],
    #                             emb = options['gcn']['fliter_emb'],
    #                             dropout=options['gcn']['dropout'])
    model_vqg = question_gen(vocab=question_vocab['wtoi'], vocab_i2t=question_vocab['itow'], opt=options['vqg'])
    # no_finding = layer_vqg.no_finding_area_top(in_feature=2652, hidden_feature=512, dropout=options['gcn']['dropout'])



    criterion = nn.CrossEntropyLoss()

    # Move it to GPU
    model_gcn = model_gcn.cuda()
    model_vqg = model_vqg.cuda()
    # model_gcn_nofinding = model_gcn_nofinding.cuda()
    # no_finding = no_finding.cuda()
    criterion = criterion.cuda()



    # Define the optimiser
    optimizer = torch.optim.Adam([
        {'params': model_gcn.parameters()},
        {'params': model_vqg.parameters()}], lr=args.lr)

    # Continue training from saved model
    start_ep = 0
    if args.model_path and os.path.isfile(args.model_path):
        print('Resuming from checkpoint %s' % (args.model_path))
        ckpt = torch.load(args.model_path)
        start_ep = ckpt['epoch']
        model_gcn.load_state_dict(ckpt['state_dict_gcn'])
        model_vqg.load_state_dict(ckpt['state_dict_vqg'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # Update the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

    # Learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=[30], gamma=0.5)
    scheduler.last_epoch = start_ep - 1

    # Train iterations
    print('Start training.')
    bleu_best= [0.0, 0.0, 0.0, 0.0]
    cider_best = 0.0
    meteor_best = 0.0
    rouge_best = 0.0
    for ep in range(start_ep, start_ep + args.ep):

        adjust_learning_rate(optimizer, ep)
        scheduler.step()
        ep_loss = 0.0
        ep_top3 = 0.0
        ep_top1 = 0.0
        ave_loss = 0.0
        ave_top3 = 0.0
        ave_top1 = 0.0
        iter_time_all = 0.0

        for step, next_batch in enumerate(loader):

            model_gcn.train()
            model_vqg.train()
            # Move batch to cuda
            target_q, an_feat, img_feat, adj_mat = \
                utils.batch_to_cuda(next_batch, volatile=True)

            # forward pass
            torch.cuda.synchronize()
            start = time.time()

            # img_feat_no = torch.mul(img_feat[:,:,None,:], img_feat[:,None,:,:]).view(-1, 2652)
            # adj_mat = no_finding(img_feat_no).view(-1,36,36)
            # adj_mat += torch.eye(36).cuda()
            # adj_mat = torch.clamp(adj_mat, max=1)
            feat_gcn, adj_new = model_gcn(img_feat, adj_mat)
            # feat_gcn, adj_new = model_gcn_nofinding(img_feat, adj_mat)
            output = model_vqg(feat_gcn, an_feat, target_q)
            # for i in range(256):
            #     dataset.drawarea[i]['adj'] = adj_new[i].detach().cpu().numpy().tolist()
            #     dataset.drawarea[i]['adj_diag'] = np.diag(adj_new[i].detach().cpu().numpy()).tolist()
            #
            # json.dump(dataset.drawarea, open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/draw/new_adj_t.json', 'w'))
            # output_bs = model_vqg.beam_search(feat_gcn[0].unsqueeze(0),
            #                                   an_feat[0].unsqueeze(0))
            target_q = target_q[:, 1:].contiguous()

            loss = criterion(output.view(output.size(0)*output.size(1), output.size(2)), target_q.view(target_q.size(0)*target_q.size(1)))



            # Compute batch accu

            top1 = utils.accuracy(output, target_q, 1)
            top3 = utils.accuracy(output, target_q, 3)

            ep_top1 += top1
            ep_top3 += top3
            ep_loss += loss.item()
            ave_top1 += top1
            ave_top3 += top3
            ave_loss += loss.item()


            # This is a 40 step average
            if step % 40 == 0 and step != 0:
                print('  Epoch %02d(%03d/%03d), ave loss: %.7f, top1: %.2f%%, top3: %.2f%%, iter time: %.4fs' %
                      (ep + 1, step, n_batches, ave_loss / 40,
                       ave_top1/40, ave_top3 / 40, iter_time_all/40))

                ave_top1 = 0
                ave_top3 = 0
                ave_loss = 0
                iter_time_all = 0

            # Compute gradient and do optimisation step
            optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_(model_gcn.parameters(), 2.)
            # clip_grad_norm_(model_gcn.parameters(), 2.)
            # clip_grad_norm_(no_finding.parameters(), 2.)
            optimizer.step()

            end = time.time()
            iter_time = end - start
            iter_time_all += iter_time



            # save model and compute validation accuracy every 400 steps
            if step == 0:
                with torch.no_grad():
                    epoch_loss = ep_loss / n_batches
                    epoch_top1 = ep_top1 / n_batches
                    epoch_top3 = ep_top3 / n_batches

                    # compute validation accuracy over a small subset of the validation set
                    model_gcn.train(False)
                    model_vqg.train(False)
                    model_gcn.eval()
                    model_vqg.eval()

                    output_all = []
                    output_all_bs = {}
                    ref_all = []

                    flag_val = 0

                    for valstep, val_batch in tqdm(enumerate(loader_val)):
                        # test_batch = next(loader_test)
                        target_q, an_feat, img_feat, adj_mat = \
                            utils.batch_to_cuda(val_batch, volatile=True)
                        # img_feat_no = torch.mul(img_feat[:, :, None, :], img_feat[:, None, :, :]).view(-1, 2652)
                        # adj_mat = no_finding(img_feat_no).view(-1, 36, 36)
                        # adj_mat += torch.eye(36).cuda()
                        # adj_mat = torch.clamp(adj_mat, max=1)
                        # feat_gcn, _ = model_gcn_nofinding(img_feat, adj_mat)
                        feat_gcn, adj_new = model_gcn(img_feat, adj_mat)
                        output = model_vqg.generate(feat_gcn, an_feat)

                        for j in range(feat_gcn.size(0)):
                            output_bs = model_vqg.beam_search(feat_gcn[j].unsqueeze(0),
                                                              an_feat[j].unsqueeze(0))
                            output_all_bs[flag_val] = output_bs
                            flag_val += 1


                        output_all.append(output.cpu().numpy())
                        ref_all.append(target_q[:, :-1].cpu().numpy())



                    gen, ref = utils.idx2question(np.concatenate(output_all, 0), np.concatenate(ref_all, 0), question_vocab['itow'])
                    print(gen.values()[:10])

                    # save the best
                    bleu, cider, meteor, rouge = main.main(ref, gen)
                    bleu_best, cider_best, meteor_best, rouge_best, choice = utils.save_the_best(bleu, cider, meteor, rouge,
                                                                                                 bleu_best, cider_best,
                                                                                                 meteor_best, rouge_best)
                    if choice:
                        utils.save(model_gcn, model_vqg, optimizer, ep, epoch_loss, epoch_top1,
                                   dir=args.save_dir, name=args.name + '_' + str(ep + 1))

                    print('use beam search...')
                    bleu, cider, meteor, rouge = main.main(ref, output_all_bs)
                    bleu_best, cider_best, meteor_best, rouge_best, choice = utils.save_the_best(bleu, cider, meteor, rouge,
                                                                                                 bleu_best, cider_best,
                                                                                                 meteor_best, rouge_best)
                    if choice:
                        utils.save(model_gcn, model_vqg, optimizer, ep, epoch_loss, epoch_top1,
                                   dir=args.save_dir, name=args.name + '_' + str(ep + 1))


                    print('the best bleu: %s, cider: %.6s, meteor: %.6s, rouge: %.6s'
                          % (bleu_best, cider_best, meteor_best, rouge_best))
                    print(output_all_bs.values()[:10])



                model_gcn.train(True)
                model_vqg.train(True)




def test(args):
    """
        model testing
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
                                emb = options['gcn']['fliter_emb'],
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
    answer_bs3 = {}
    answer_bs5 = {}
    flag = 0
    bs3 = []
    for step, next_batch in tqdm(enumerate(loader)):
        # Batch preparation
        target_q, an_feat, img_feat, adj_mat = \
            utils.batch_to_cuda(next_batch, volatile=True)

        # get predictions
        feat_gcn, adj_new = model_gcn(img_feat, adj_mat)
        output = model_vqg.generate(feat_gcn, an_feat)
        # for i in range(256):
        #     dataset.drawarea[i]['adj'] = adj_new[i].detach().cpu().numpy().tolist()
        #     dataset.drawarea[i]['adj_diag'] = np.diag(adj_new[i].detach().cpu().numpy()).tolist()
        #
        # json.dump(dataset.drawarea, open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/draw/new_adj_fram.json','w'))

        for ii in range(feat_gcn.size(0)):

            output_bs3_ = model_vqg.beam_search(feat_gcn[ii].unsqueeze(0),
                                                an_feat[ii].unsqueeze(0))
            output_bs5_ = model_vqg.beam_search(feat_gcn[ii].unsqueeze(0),
                                                an_feat[ii].unsqueeze(0), 5)
            answer_bs3[flag] = output_bs3_
            answer_bs5[flag] = output_bs5_
            flag += 1
            bs3.append(output_bs3_)


        answer.append(output.cpu().numpy())
        ref_q.append(target_q[:, :-1].cpu().numpy())

    json.dump(bs3, open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/draw/test_all_question.json', 'w'))

    gen, ref = utils.idx2question(np.concatenate(answer, 0), np.concatenate(ref_q, 0), question_vocab['itow'])
    print(gen.values()[:10])

    bleu, cider, meteor, rouge = main.main(ref, gen)
    bleu3, cider3, meteor3, rouge3 = main.main(ref, answer_bs3)
    bleu5, cider5, meteor5, rouge5 = main.main(ref, answer_bs5)


    print('Testing done')




def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    parser.add_argument('--data_dir', metavar='', type=str, default='',
                        help='path to data directory')
    parser.add_argument('--save_dir', metavar='', type=str, default='')
    parser.add_argument('--path_opt', metavar='', type=str, default='')
    parser.add_argument('--name', metavar='', type=str,
                        default='model', help='model name')
    parser.add_argument('--dropout', metavar='', type=float, default=0.5,
                        help='probability of dropping out FC nodes during training')
    parser.add_argument('--model_path', metavar='', type=str, default=''
                        ,help='trained model path.')
    #/mnt/data/xiaojinhui/wangtan_MM/vqg_fliter_softmax/model/model_13.pth_v.tar
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))

    options = {
        'gcn': {
            'nfeat': 2652,
            'nhid': 1024,
            'nclass': 1024,
            'fliter_emb': 512,
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


    if args.train:
        train(args)
    if args.test:
        test(args)
    if not args.train and not args.eval and not args.trainval and not args.test:
        parser.print_help()



