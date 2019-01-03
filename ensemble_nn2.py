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
    ]

    models = ensemble_190103
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
    bs = 512
    w = torch.zeros((1, len(models), num_class()), requires_grad=True)
    generator = tqdm(range(200000))
    # optimizer = torch.optim.SGD([w], lr=1, momentum=0.99, weight_decay=1e-5, nesterov=True)
    optimizer = torch.optim.Adam([w], lr=0.1, weight_decay=1e-5, amsgrad=True)
    loss_fn = BCELoss()
    # loss_fn = f1_loss
    # loss_fn = MultiLabelMarginLoss()

    inp_v, lb_v = torch.Tensor(valid_v), torch.Tensor(valid_ohs_v)
    weighted_inp = torch.mean(inp_v, dim=1)
    loss = loss_fn(weighted_inp, lb_v)
    print('val loss=%.4f' % loss)

    ma_loss_t = ma_loss_v = 1.0
    best_loss_v = 1000.0
    best_w = None
    not_improved_cnt = 0
    overfit_cnt = 0
    for _ in generator:
        # w_norm = F.softmax(w, dim=0)
        w_norm = torch.sigmoid(w)
        # w_norm = w

        idx = random.sample(range(len(valid_t)), bs)
        inp, lb = torch.Tensor(valid_t[idx]), torch.Tensor(valid_ohs_t[idx])
        weighted_inp = torch.sum(inp * w_norm, dim=1)
        weighted_inp = torch.clamp(weighted_inp, 0.0, 1.0)
        loss = loss_fn(weighted_inp, lb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ma_loss_t = ma_loss_t * 0.9 + loss.item() * 0.1

        weighted_inp = torch.sum(inp_v * w_norm, dim=1)
        weighted_inp = torch.clamp(weighted_inp, 0.0, 1.0)
        loss = loss_fn(weighted_inp, lb_v)
        ma_loss_v = ma_loss_v * 0.9 + loss.item() * 0.1
        generator.set_description('loss=%0.4f loss(valid)=%.4f' % (ma_loss_t, ma_loss_v))

        if loss.item() < best_loss_v:
            best_loss_v = loss.item()
            best_w = w.squeeze().detach().numpy()
            not_improved_cnt = 0
        else:
            not_improved_cnt += 1

        if loss.item() > ma_loss_t:
            overfit_cnt += 1
        else:
            overfit_cnt = 0

        if not_improved_cnt > 1000:
            break
        # if overfit_cnt > 100:
        #     break

    def sigmoid(x, derivative=False):
        return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))

    def tanh(x):
        return 2 / (1 + np.exp(-2 * x) - 1)
    best_w = sigmoid(best_w)
    print()
    with np.printoptions(precision=2, suppress=True):
        print(best_loss_v)
        for l in best_w:
            for e in l:
                print('%.2f' % e, end=' ')
            print()
        print()

        print(np.sum(best_w, axis=0))

    weighted_inp = np.mean(valid_merged, axis=1)
    best_th = threshold_search(weighted_inp, valid_ohs)
    __best_threshold = best_th
    f1_best = get_f1_threshold(weighted_inp, valid_ohs, __best_threshold)
    print(__best_threshold)
    print('f1_best(all, naive valid)=%.4f' % f1_best)

    weighted_inp = np.mean(valid_v, axis=1)
    best_th = threshold_search(weighted_inp, valid_ohs_v)
    __best_threshold = best_th
    f1_best = get_f1_threshold(weighted_inp, valid_ohs_v, __best_threshold)
    print(__best_threshold)
    print('f1_best(naive valid)=%.4f' % f1_best)

    weighted_valid = valid_v * best_w
    weighted_valid = np.sum(weighted_valid, axis=1)
    weighted_valid = np.clip(weighted_valid, 0.0, 1.0)

    best_th = threshold_search(weighted_valid, valid_ohs_v)
    __best_threshold = best_th
    f1_best = get_f1_threshold(weighted_valid, valid_ohs_v, __best_threshold)

    print(__best_threshold)
    print('f1_best(valid)=', f1_best)

    weighted_valid = valid_merged * best_w
    weighted_valid = np.sum(weighted_valid, axis=1)
    weighted_valid = np.clip(weighted_valid, 0.0, 1.0)

    best_th = threshold_search(weighted_valid, valid_ohs)
    __best_threshold_prev = __best_threshold
    __best_threshold = best_th
    f1_best = get_f1_threshold(weighted_valid, valid_ohs, __best_threshold)

    print(__best_threshold)
    print('f1_best=', f1_best)
    f1_best = get_f1_threshold(weighted_valid, valid_ohs, __best_threshold - 0.1)
    print('f1_best(-0.1)=', f1_best)
    f1_best = get_f1_threshold(weighted_valid, valid_ohs, __best_threshold + 0.1)
    print('f1_best(+0.1)=', f1_best)

    f1_best = get_f1_threshold(weighted_valid, valid_ohs, __best_threshold_prev)
    print(__best_threshold_prev)
    print('f1_best(prev)=', f1_best)

    weighted_test = test_merged * best_w
    weighted_test = np.sum(weighted_test, axis=1)
    weighted_test = np.clip(weighted_test, 0.0, 1.0)

    output = 'asset/ensemble_nn2.csv'
    save_pred(ids_test, weighted_test, th=__best_threshold, fname=output)
    output = 'asset/ensemble_nn2_thval.csv'
    save_pred(ids_test, weighted_test, th=__best_threshold_prev, fname=output)

    output = 'asset/ensemble_nn2_th-1.csv'
    save_pred(ids_test, weighted_test, th=__best_threshold - 0.1, fname=output)
    output = 'asset/ensemble_nn2_th+1.csv'
    save_pred(ids_test, weighted_test, th=__best_threshold + 0.1, fname=output)
