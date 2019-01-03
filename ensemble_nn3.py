import sys

import pandas as pd
import pickle
import random
import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.nn import functional as F, BCELoss, MultiLabelMarginLoss

from theconf import Config as C

# setup parameters for xgboost
from tqdm import tqdm

from common import LABELS, num_class, threshold_search, save_pred
from data import get_dataset, get_dataloaders
from metric import get_f1_threshold, f1_loss

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
        'inception_fold0',
        'inception_fold1',
        'inception_fold2',
        'inception_fold3',
        'inception_fold4',
        'inceptionv4_fold0',
        'inceptionv4_fold1',
        'inceptionv4_fold2',
        'inceptionv4_fold3',
        'inceptionv4_fold4',
        'densenet121_fold0',
        'densenet121_fold1',
        'densenet121_fold2',
        'densenet121_fold3',
        'densenet121_fold4',
        'densenet169_fold0',
        'densenet169_fold1',
        'densenet169_fold2',
        'densenet169_fold3',
        'densenet169_fold4'
    ]

    models = ensemble_181229
    test_pkl_s = 'asset/%s.aug.pkl'
    valid_pkl_s = 'asset/%s.aug.valid.pkl'
    key = 'prediction'

    C.get()['cv_fold'] = True
    C.get()['eval'] = False
    C.get()['extdata'] = True
    C.get()['batch'] = 1
    _, _, ids_cvalid, ids_test = get_dataset()
    dataset = pd.read_csv(LABELS).set_index('Id')

    valid_preds = []    # model x data x label(28)
    test_preds = []
    for model_name in models:
        with open(valid_pkl_s % model_name, 'rb') as f:
            d_valid = pickle.load(f)
        with open(test_pkl_s % model_name, 'rb') as f:
            d_test = pickle.load(f)
        valid_preds.append(np.expand_dims(np.concatenate(d_valid[key], axis=0), 1))
        test_preds.append(np.expand_dims(np.concatenate(d_test[key], axis=0), 1))

    valid_merged = np.concatenate(valid_preds, axis=1)
    test_merged = np.concatenate(test_preds, axis=1)

    valid_ohs = []
    for uuid, val_x in zip(ids_cvalid, valid_merged):
        lb = [int(x) for x in dataset.loc[uuid]['Target'].split()]
        lb_onehot = np.eye(num_class(), dtype=np.float)[lb].sum(axis=0)
        valid_ohs.append(lb_onehot)
    valid_ohs = np.array(valid_ohs, dtype=np.float)

    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=200)
    train_index, test_index = next(mskf.split(list(range(len(valid_ohs))), valid_ohs))
    valid_t = valid_merged[train_index]
    valid_v = valid_merged[test_index]
    valid_ohs_t = valid_ohs[train_index]
    valid_ohs_v = valid_ohs[test_index]

    # train w for linear combination
    bs = 256
    generator = tqdm(range(200000))

    inp_v, lb_v = torch.Tensor(valid_v), torch.Tensor(valid_ohs_v)

    ma_loss_t = ma_loss_v = 1.0
    best_loss_v = 1.0
    not_improved_cnt = 0

    num_feat = 512
    net = torch.nn.Sequential(
        torch.nn.Linear(num_class() * len(models), num_feat),
        torch.nn.BatchNorm1d(num_feat),     # num_class() * len(models)
        torch.nn.LeakyReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_feat, num_feat),
        torch.nn.BatchNorm1d(num_feat),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(num_feat, num_class()),
        # torch.nn.Sigmoid()
    )
    # loss_fn = BCELoss()
    loss_fn = MultiLabelMarginLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4, amsgrad=True)
    for _ in generator:
        idx = random.sample(range(len(valid_t)), bs)
        inp, lb = torch.Tensor(valid_t[idx]), torch.Tensor(valid_ohs_t[idx])

        net.train()
        inp_sum = torch.sum(inp, dim=1)
        out = net(inp.view(inp.size(0), -1))
        out = out + inp_sum
        out = torch.sigmoid(out)

        loss = loss_fn(out, lb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ma_loss_t = ma_loss_t * 0.9 + loss.item() * 0.1

        net.eval()
        inp_sum = torch.sum(inp_v, dim=1)
        out = net(inp_v.view(inp_v.size(0), -1))
        out = out + inp_sum
        out = torch.sigmoid(out)

        loss = loss_fn(out, lb_v)
        ma_loss_v = ma_loss_v * 0.9 + loss.item() * 0.1
        generator.set_description('loss=%0.4f loss(valid)=%.4f' % (ma_loss_t, ma_loss_v))

        if loss.item() < best_loss_v:
            best_loss_v = loss.item()
            not_improved_cnt = 0
        else:
            not_improved_cnt += 1

        if not_improved_cnt > 100:
            break

    print()
    print(best_loss_v)

    net.eval()

    inp = torch.Tensor(valid_merged)
    inp_sum = torch.sum(inp, dim=1)
    out = net(inp.view(inp.size(0), -1))
    out = out + inp_sum
    out = torch.sigmoid(out)
    out = out.detach().numpy()

    best_th = threshold_search(out, valid_ohs)
    __best_threshold = best_th
    f1_best = get_f1_threshold(out, valid_ohs, __best_threshold)
    print(__best_threshold)
    print('f1_best(all, naive valid)=%.4f' % f1_best)

    inp = torch.Tensor(valid_t)
    inp_sum = torch.sum(inp, dim=1)
    out = net(inp.view(inp.size(0), -1))
    out = out + inp_sum
    out = torch.sigmoid(out)
    out = out.detach().numpy()

    best_th = threshold_search(out, valid_ohs_t)
    __best_threshold = best_th
    f1_best = get_f1_threshold(out, valid_ohs_t, __best_threshold)
    print(__best_threshold)
    print('f1_best(naive valid)=%.4f' % f1_best)

    inp = torch.Tensor(valid_v)
    inp_sum = torch.sum(inp, dim=1)
    out = net(inp.view(inp.size(0), -1))
    out = out + inp_sum
    out = torch.sigmoid(out)
    out = out.detach().numpy()

    best_th = threshold_search(out, valid_ohs_v)
    __best_threshold = best_th
    f1_best = get_f1_threshold(out, valid_ohs_v, __best_threshold)

    print(__best_threshold)
    print('f1_best(valid)=', f1_best)

    inp = torch.Tensor(valid_merged)
    inp_sum = torch.sum(inp, dim=1)
    out = net(inp.view(inp.size(0), -1))
    out = out + inp_sum
    out = torch.sigmoid(out)
    out = out.detach().numpy()

    best_th = threshold_search(out, valid_ohs)
    __best_threshold = best_th
    f1_best = get_f1_threshold(out, valid_ohs, __best_threshold)

    print(__best_threshold)
    print('f1_best=', f1_best)

    inp = torch.Tensor(test_merged)
    inp_sum = torch.sum(inp, dim=1)
    out = net(inp.view(inp.size(0), -1))
    out = out + inp_sum
    out = torch.sigmoid(out)
    out = out.detach().numpy()

    output = 'asset/ensemble_nn3.csv'
    save_pred(ids_test, out, th=__best_threshold, fname=output)
