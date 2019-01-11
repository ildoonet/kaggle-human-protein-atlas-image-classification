import itertools
import sys

import pandas as pd
import pickle
import random
import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.nn import functional as F, BCELoss, MultiLabelMarginLoss, DataParallel, BCEWithLogitsLoss

from theconf import Config as C

# setup parameters for xgboost
from tqdm import tqdm

from common import LABELS, num_class, threshold_search, save_pred, LABELS_HPA, grouper
from data import get_dataset, get_dataloaders
from metric import get_f1_threshold, f1_loss, FocalLoss

if __name__ == '__main__':
    ensemble_181228_m3 = [
        'densenet121_hpa',
        'densenet169_hpa',
        'inceptionv4_lr0.0001',
    ]

    ensemble_181228 = [
        'densenet121_hpa',
        'densenet169_hpa',
        'inceptionv4_lr0.0001',
        'inception_hpa',
        'resnet50_hpa',
        'resnet34_hpa'
    ]

    ensemble_181229 = [
        'densenet121_fold0',
        'densenet121_fold1',
        'densenet121_fold2',
        'densenet121_fold3',
        'densenet121_fold4',
        'densenet169_fold0',
        'densenet169_fold1',
        'densenet169_fold2',
        'densenet169_fold3',
        'densenet169_fold4',

        'inceptionv4_fold0',
        'inceptionv4_fold1',
        'inceptionv4_fold2',
        'inceptionv4_fold3',
        'inceptionv4_fold4',

        'inception_fold0',
        'inception_fold1',
        'inception_fold2',
        'inception_fold3',
        'inception_fold4',
    ]

    ensemble_190103 = ensemble_181229 + [
        'pnasnet_fold0_lr0.00005',
        'pnasnet_fold1_lr0.00005',
        'pnasnet_fold2_lr0.00005',
        'pnasnet_fold3_lr0.00005',
        'pnasnet_fold4_lr0.00005',
        'senet_fold0_lr0.00005',
        'senet_fold1_lr0.00005',
        'senet_fold2_lr0.00005',
        'senet_fold3_lr0.00005',
        'senet_fold4_lr0.00005',
        'densenet121_fold0_lr0.0001_bce',
        'densenet121_fold1_lr0.0001_bce',
        'densenet121_fold2_lr0.0001_bce',
        'densenet121_fold3_lr0.0001_bce',
        'densenet121_fold4_lr0.0001_bce',

        'densenet169_fold1_lr0.0001_bce',
        'densenet169_fold2_lr0.0001_bce',
        'densenet169_fold3_lr0.0001_bce',
        'densenet169_fold4_lr0.0001_bce',
        'nasnet_fold1_lr0.0001_relu_bce',
        'inceptionv4_fold0_bce',
        'inceptionv4_fold1_bce',
        'inceptionv4_fold2_bce',
        'inceptionv4_fold3_bce',
        'inceptionv4_fold4_bce',
    ]

    model_selection = [
        'densenet121_fold0',
        'densenet121_fold3',
        'densenet121_fold4',
        'densenet121_fold2_lr0.0001_bce',

        'densenet169_fold0',
        'densenet169_fold3',
        'densenet169_fold4',
        'densenet169_fold2_lr0.0001_bce',   # <--- best one

        'inceptionv4_fold0',
        'inceptionv4_fold1',
        'inceptionv4_fold3',

        'pnasnet_fold2_lr0.00005',
        'senet_fold0_lr0.00005',

        'nasnet_fold0_lr0.0001_relu_bce',
        'nasnet_fold2_lr0.0001_relu_bce',
    ]

    models = model_selection
    test_pkl_s = 'asset_v3/%s.aug.pkl'
    train_pkl_s = 'asset_v3/%s.aug.train.pkl'
    valid_pkl_s = 'asset_v3/%s.aug.valid.pkl'
    key = 'feature'

    C.get()['cv_fold'] = True
    C.get()['eval'] = False
    C.get()['extdata'] = True
    C.get()['batch'] = 1
    _, _, ids_cvalid, ids_test = get_dataset()
    with open('./split/sampled_names', 'r') as text_file:
        tr = text_file.read().split(',')
    with open('./split/sampled_ext_names', 'r') as text_file:
        ext_n = text_file.read().split(',')
    ids_train = tr + ext_n
    dataset = pd.read_csv(LABELS).set_index('Id')
    dataset_hpa = pd.read_csv(LABELS_HPA).set_index('Id')

    train_preds = []
    valid_preds = []    # model x data x label(28)
    test_preds = []
    train_logit = []
    valid_logit = []
    test_logit = []
    for i, model_name in enumerate(models):
        print('read... ', model_name)
        with open(train_pkl_s % model_name, 'rb') as f:
            d_train = pickle.load(f)
        with open(valid_pkl_s % model_name, 'rb') as f:
            d_valid = pickle.load(f)
        with open(test_pkl_s % model_name, 'rb') as f:
            d_test = pickle.load(f)

        def get_merged_feat(feats):
            if feats[0].shape[0] == 1:
                feats = np.concatenate(feats, axis=0)
            else:
                feats = np.stack(feats)
            feats = feats.reshape(feats.shape[0], -1)
            return feats

        train_preds.append(get_merged_feat(d_train[key]))
        valid_preds.append(get_merged_feat(d_valid[key]))
        test_preds.append(get_merged_feat(d_test[key]))

        train_logit.append(np.expand_dims(np.concatenate(d_train['prediction'], axis=0), 1))
        valid_logit.append(np.expand_dims(np.concatenate(d_valid['prediction'], axis=0), 1))
        test_logit.append(np.expand_dims(np.concatenate(d_test['prediction'], axis=0), 1))

    train_merged = np.concatenate(train_preds, axis=1)
    valid_merged = np.concatenate(valid_preds, axis=1)
    test_merged = np.concatenate(test_preds, axis=1)

    train_logit = np.concatenate(train_logit, axis=1)
    valid_logit = np.concatenate(valid_logit, axis=1)
    test_logit = np.concatenate(test_logit, axis=1)

    del train_preds, valid_preds, test_preds

    def get_ohs(id_list, preds_list):
        assert len(id_list) == len(preds_list)
        ohs = []
        for uuid, x in zip(id_list, preds_list):
            try:
                lb = [int(i) for i in dataset.loc[uuid]['Target'].split()]
            except KeyError:
                lb = [int(i) for i in dataset_hpa.loc[uuid]['Target'].split()]

            lb_onehot = np.eye(num_class(), dtype=np.float)[lb].sum(axis=0)
            ohs.append(lb_onehot)
        ohs = np.array(ohs, dtype=np.float)
        return ohs

    train_ohs = get_ohs(ids_train, train_merged)
    valid_ohs = get_ohs(ids_cvalid, valid_merged)

    del ids_train, ids_cvalid

    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=200)
    train_index, test_index = next(mskf.split(list(range(len(valid_ohs))), valid_ohs))
    valid_t = valid_merged[train_index]
    valid_v = valid_merged[test_index]
    valid_ohs_t = valid_ohs[train_index]
    valid_ohs_v = valid_ohs[test_index]
    valid_logit_t = valid_logit[train_index]
    valid_logit_v = valid_logit[test_index]

    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=200)
    train_index, test_index = next(mskf.split(list(range(len(train_ohs))), train_ohs))
    train_t = train_merged[train_index]
    train_v = train_merged[test_index]
    train_ohs_t = train_ohs[train_index]
    train_ohs_v = train_ohs[test_index]
    train_logit_t = train_logit[train_index]
    train_logit_v = train_logit[test_index]

    tv_merged = np.concatenate((train_t, valid_t))
    tv_ohs = np.concatenate((train_ohs_t, valid_ohs_t))
    tv_logit_merged = np.concatenate((train_logit_t, valid_logit_t))

    # all_merged = np.concatenate((train_merged, valid_merged))
    # all_ohs = np.concatenate((train_ohs, valid_ohs))

    # train w for linear combination
    bs = 128
    input_size = train_merged.shape[1]
    generator = tqdm(range(10000))

    inp_v, logit_v, lb_v = torch.Tensor(valid_v), torch.Tensor(valid_logit_v), torch.Tensor(valid_ohs_v)
    inp_v = inp_v.cuda()
    lb_v = lb_v.cuda()
    logit_v = logit_v.cuda()

    inp_train_v, logit_train_v, lb_train_v = torch.Tensor(train_v), torch.Tensor(train_logit_v), torch.Tensor(train_ohs_v)
    inp_train_v = inp_train_v.cuda()
    lb_train_v = lb_train_v.cuda()
    logit_train_v = logit_train_v.cuda()

    class EnsembleNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            num_feat = 1024
            self.dropch = torch.nn.Dropout2d(0.5)
            self.net = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(input_size, num_feat),
                torch.nn.BatchNorm1d(num_feat),
                torch.nn.ReLU6(),
            )

            self.net2 = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(num_feat, num_feat),
                torch.nn.BatchNorm1d(num_feat),
                torch.nn.ReLU6(),
                torch.nn.Linear(num_feat, num_feat),
                torch.nn.BatchNorm1d(num_feat),
                torch.nn.ReLU6(),

                # torch.nn.Sigmoid()
            )

            self.out = torch.nn.Linear(num_feat, num_class())

        def forward(self, feats, logits, in_train=False):
            logits = self.dropch(logits)
            logits = torch.mean(logits, dim=1)  # naive average
            logits = torch.clamp(logits, 0.0001, 0.9999)
            logits = torch.log(logits / (1 - logits))

            # --- fc ---
            feats = feats.view(feats.size(0), -1)
            fc_out = self.net(feats)
            fc_out = self.net2(fc_out)
            fc_out = self.out(fc_out)

            out = logits + fc_out
            # out = fc_out
            # out = torch.clamp(out, 0.0, 1.0)
            # print(logits.max(), logits.min(), fc_out.max(), fc_out.min())

            # if not in_train:
            out = torch.sigmoid(out)

            return out

    net = EnsembleNet()
    # net = DataParallel(net)
    net.cuda()

    def new_optimizer():
        # return torch.optim.SGD(list(net.parameters()), lr=0.00001, weight_decay=1e-5, momentum=0.95, nesterov=False)
        return torch.optim.Adam(list(net.parameters()), lr=0.00001, weight_decay=1e-4, amsgrad=False)
        # return torch.optim.RMSprop(net.parameters(), lr=0.00001)

    optimizer = new_optimizer()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=True)

    def eval_batch(data_all, logit_all, in_train=False):
        out_list = []
        for batch, logit in zip(grouper(data_all, bs), grouper(logit_all, bs)):
            batch = [b if isinstance(b, torch.Tensor) else torch.from_numpy(b) for b in batch if b is not None]
            logit = [b if isinstance(b, torch.Tensor) else torch.from_numpy(b) for b in logit if b is not None]
            out_batch = net(torch.stack(batch, dim=0).cuda(), torch.stack(logit, dim=0).cuda(), in_train)
            out_list.append(out_batch)
        out = torch.cat(out_list, dim=0)
        return out

    # loss_fn = MultiLabelMarginLoss()
    # loss_fn = FocalLoss()
    # loss_fn = BCELoss()
    # loss_fn = BCEWithLogitsLoss()
    loss_fn = f1_loss

    ma_loss_t = ma_loss_v = ma_loss_train_v = None
    best_loss_v = 100.0
    best_model = None
    not_improved_cnt = 0
    shuffled_idxs = []
    for step in generator:
        if len(shuffled_idxs) < bs:
            shuffled_idxs = list(range(len(valid_t)))
            random.shuffle(shuffled_idxs)
        idx = [shuffled_idxs.pop() for _ in range(bs)]          # train+valid
        inp, logit, lb = torch.Tensor(valid_t[idx]), torch.Tensor(valid_logit_t[idx]), torch.Tensor(valid_ohs_t[idx])
        inp = inp.cuda()
        lb = lb.cuda()
        logit = logit.cuda()

        net.train()
        out = net(inp, logit, in_train=True)

        loss = loss_fn(out, lb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step(best_loss_v)

        if ma_loss_t:
            ma_loss_t = ma_loss_t * 0.9 + loss.item() * 0.1
        else:
            ma_loss_t = loss.item()

        del loss, out
        net.eval()
        # --- test
        out = eval_batch(inp_v, logit_v, in_train=True)
        loss = loss_fn(out, lb_v)
        if ma_loss_v:
            ma_loss_v = ma_loss_v * 0.9 + loss.item() * 0.1
        else:
            ma_loss_v = loss.item()

        if loss.item() < best_loss_v:
            best_loss_v = loss.item()
            not_improved_cnt = 0
            best_model = net.state_dict()
        else:
            not_improved_cnt += 1

        del loss, out
        out = eval_batch(inp_train_v, logit_train_v, in_train=True)
        loss = loss_fn(out, lb_train_v)
        if ma_loss_train_v:
            ma_loss_train_v = ma_loss_train_v * 0.9 + loss.item() * 0.1
        else:
            ma_loss_train_v = loss.item()

        generator.set_description('loss=%0.4f loss(valid)=%.4f loss(train_v)=%.4f not_improved_cnt=%d' % (ma_loss_t, ma_loss_v, ma_loss_train_v, not_improved_cnt))

        if not_improved_cnt > 200:
            break

        # if optimizer.param_groups[0]['lr'] < 0.00001:
        #     break
        if step % 1000 == 0:
            print()

        del loss, out
    generator.close()
    del generator

    print()
    print(best_loss_v)

    print('--- retrain ---')
    net = EnsembleNet()
    net.cuda()
    optimizer = new_optimizer()
    generator = tqdm(range(step))
    shuffled_idxs = []
    for _ in generator:
        if len(shuffled_idxs) < bs:
            shuffled_idxs = list(range(len(valid_merged)))
            random.shuffle(shuffled_idxs)
        idx = [shuffled_idxs.pop() for _ in range(bs)]          # train+valid
        inp, logit, lb = torch.Tensor(valid_merged[idx]), torch.Tensor(valid_logit[idx]), torch.Tensor(valid_ohs[idx])
        inp = inp.cuda()
        logit = logit.cuda()
        lb = lb.cuda()

        net.train()
        out = net(inp, logit, in_train=True)

        loss = loss_fn(out, lb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ma_loss_t = ma_loss_t * 0.9 + loss.item() * 0.1
        generator.set_description('loss=%0.4f' % ma_loss_t)

        del loss, out

    generator.close()
    del generator
    print()
    # net.load_state_dict(best_model)

    net.eval()

    # weighted_inp = np.mean(train_v, axis=1)
    # best_th = threshold_search(weighted_inp, train_ohs_v)
    # __best_threshold = best_th
    # f1_best = get_f1_threshold(weighted_inp, train_ohs_v, __best_threshold)
    # print(__best_threshold)
    # print('f1_best(naive train_v)=%.4f' % f1_best)
    #
    # weighted_inp = np.mean(valid_merged, axis=1)
    # best_th = threshold_search(weighted_inp, valid_ohs)
    # __best_threshold = best_th
    # f1_best = get_f1_threshold(weighted_inp, valid_ohs, __best_threshold)
    # print(__best_threshold)
    # print('f1_best(naive valid)=%.4f' % f1_best)
    #
    # weighted_inp = np.mean(valid_v, axis=1)
    # best_th = threshold_search(weighted_inp, valid_ohs_v)
    # __best_threshold = best_th
    # __best_threshold_naive = best_th
    # f1_best = get_f1_threshold(weighted_inp, valid_ohs_v, __best_threshold)
    # print(__best_threshold)
    # print('f1_best(naive valid_v)=%.4f' % f1_best)

    out = eval_batch(train_v, train_logit_v)
    out = out.detach().cpu().numpy()

    best_th = threshold_search(out, train_ohs_v)
    __best_threshold = best_th
    f1_best = get_f1_threshold(out, train_ohs_v, __best_threshold)

    print(__best_threshold)
    print('f1_best(train_v)=', f1_best)

    out = eval_batch(valid_merged, valid_logit)
    out = out.detach().cpu().numpy()

    f1_best = get_f1_threshold(out, valid_ohs, __best_threshold)

    print(__best_threshold)
    print('f1_best(valid)=', f1_best)

    best_th = threshold_search(out, valid_ohs)
    f1_best = get_f1_threshold(out, valid_ohs, best_th)

    print(best_th)
    print('f1_best(valid, th searched)=', f1_best)

    out_list = []
    for batch, logit in zip(grouper(test_merged, bs), grouper(test_logit, bs)):
        batch = [b if isinstance(b, torch.Tensor) else torch.from_numpy(b) for b in batch if b is not None]
        logit = [b if isinstance(b, torch.Tensor) else torch.from_numpy(b) for b in logit if b is not None]
        out_batch = net(torch.stack(batch, dim=0).cuda(), torch.stack(logit, dim=0).cuda())
        out_list.append(out_batch.detach().cpu().numpy())

    out = np.concatenate(out_list, axis=0)

    output = 'asset/ensemble_nn4.csv'
    save_pred(ids_test, out, th=__best_threshold, fname=output)

    print('done-')
