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

from common import LABELS, num_class, threshold_search, save_pred, LABELS_HPA
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
        'densenet169_fold4'     # 20
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
        'nasnet_fold1_lr0.0001_relu_bce',    # 20
        'inceptionv4_fold0_bce',
        'inceptionv4_fold1_bce',
        'inceptionv4_fold2_bce',
        'inceptionv4_fold3_bce',
        'inceptionv4_fold4_bce',
    ]

    ensemble_190108_best = [
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
        'densenet169_fold4',  # 20
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

    models = ensemble_190103        # 0.601
    models = ensemble_190108_best   # 0.591
    models = ['densenet169_fold2_lr0.0001_bce']
    test_pkl_s = 'asset_v3/%s.aug.pkl'
    train_pkl_s = 'asset_v3/%s.aug.train.pkl'
    valid_pkl_s = 'asset_v3/%s.aug.valid.pkl'
    key = 'prediction'

    print('models', len(models))

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

    train_preds = []
    valid_preds = []    # model x data x label(28)
    test_preds = []
    prev_labels = []
    for model_name in models:
        with open(train_pkl_s % model_name, 'rb') as f:
            d_train = pickle.load(f)
        with open(valid_pkl_s % model_name, 'rb') as f:
            d_valid = pickle.load(f)
        with open(test_pkl_s % model_name, 'rb') as f:
            d_test = pickle.load(f)
        train_preds.append(np.expand_dims(np.concatenate(d_train[key], axis=0), 1))
        valid_preds.append(np.expand_dims(np.concatenate(d_valid[key], axis=0), 1))
        test_preds.append(np.expand_dims(np.concatenate(d_test[key], axis=0), 1))

    train_merged = np.concatenate(train_preds, axis=1)
    valid_merged = np.concatenate(valid_preds, axis=1)
    test_merged = np.concatenate(test_preds, axis=1)

    train_ohs = get_ohs(ids_train, train_merged)
    valid_ohs = get_ohs(ids_cvalid, valid_merged)

    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=200)
    train_index, test_index = next(mskf.split(list(range(len(valid_ohs))), valid_ohs))
    valid_t = valid_merged[train_index]
    valid_v = valid_merged[test_index]
    valid_ohs_t = valid_ohs[train_index]
    valid_ohs_v = valid_ohs[test_index]

    print('valid_t', np.sum(valid_ohs_t, axis=0))
    print('valid_v', np.sum(valid_ohs_v, axis=0))

    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=200)
    train_index, test_index = next(mskf.split(list(range(len(train_ohs))), train_ohs))
    train_t = train_merged[train_index]
    train_v = train_merged[test_index]
    train_ohs_t = train_ohs[train_index]
    train_ohs_v = train_ohs[test_index]

    print('train_t', np.sum(train_ohs_t, axis=0))
    print('train_v', np.sum(train_ohs_v, axis=0))

    tv_merged = np.concatenate((train_t, valid_t))
    tv_ohs = np.concatenate((train_ohs_t, valid_ohs_t))

    all_merged = np.concatenate((train_merged, valid_merged))
    all_ohs = np.concatenate((train_ohs, valid_ohs))

    # train w for linear combination
    bs = 512
    generator = tqdm(range(200000))

    # inp_v, lb_v = torch.Tensor(valid_v), torch.Tensor(valid_ohs_v)
    inp_v, lb_v = torch.Tensor(train_v), torch.Tensor(train_ohs_v)
    inp_v = inp_v.cuda()
    lb_v = lb_v.cuda()
    print('valid_v shape=', inp_v.shape)

    class EnsembleNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            num_feat = 2048
            self.dropch = torch.nn.Dropout2d(0.5)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(num_class() * len(models), num_feat),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(num_feat, num_class())
                # torch.nn.Sigmoid()
            )
            self.w = torch.zeros((1, len(models), num_class()), requires_grad=True, device="cuda")
            self.w_model = torch.zeros((1, len(models), 1), requires_grad=True, device="cuda")

            self.net2 = torch.nn.Sequential(
                torch.nn.Linear(3 * num_class(), num_class()),
            )
            self.w2 = torch.zeros((1, 3, 1), requires_grad=True, device="cuda")

        def forward(self, inp):
            inp = self.dropch(inp)

            # --- average output for skip conn ---
            # inp_logit = torch.log(inp / (1 - inp))
            # inp_sum = torch.sum(inp, dim=1)   # naive sum
            inp_sum = torch.mean(inp, dim=1)   # naive average
            # inp_sum = torch.sum(inp * torch.sigmoid(self.w_model), dim=1)  # model attention
            # inp_sum = torch.mean(inp * torch.sigmoid(self.w_model), dim=1)  # model attention

            # --- fc ---
            inp_flt = inp.view(inp.size(0), -1)
            fc_out = self.net(inp_flt)
            # fc_out = torch.sigmoid(fc_out - 0.5)

            # --- attention ---
            weighted_inp = torch.sum(inp * torch.sigmoid(self.w), dim=1)
            # weighted_inp = torch.clamp(weighted_inp, 0.0, 1.0)

            # --- out
            out = fc_out + inp_sum
            # out = torch.sigmoid(out)
            # out = torch.sigmoid(out)
            out = torch.clamp(out, 0.0, 1.0)

            # --- out2
            # out = out + weighted_inp
            # out = torch.clamp(out, 0.0, 1.0)

            # --- out3
            # out = torch.stack([inp_sum, fc_out, weighted_inp], dim=1)
            # out = torch.sum(out * torch.sigmoid(self.w2), dim=1)
            # out = out.view(out.size(0), -1)
            # out = self.net2(out) + inp_sum
            # out = torch.sigmoid(2.0 * (out - 0.5))

            return out

    net = EnsembleNet()
    net.cuda()
    optimizer = torch.optim.Adam(list(net.parameters()) + [net.w, net.w_model, net.w2], lr=0.0001, weight_decay=1e-5, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=True)

    # loss_fn = MultiLabelMarginLoss()
    # loss_fn = FocalLoss()
    loss_fn = BCELoss()
    # loss_fn = f1_loss

    ma_loss_t = ma_loss_v = 1.0
    best_loss_v = 1.0
    best_model = None
    not_improved_cnt = 0
    shuffled_idxs = []
    for step in generator:
        if len(shuffled_idxs) < bs:
            shuffled_idxs = list(range(len(valid_t)))
            random.shuffle(shuffled_idxs)
        # idx = random.sample(range(len(train_merged)), bs)       # train
        # inp, lb = torch.Tensor(train_merged[idx]), torch.Tensor(train_ohs[idx])
        idx = [shuffled_idxs.pop() for _ in range(bs)]          # train+valid
        inp, lb = torch.Tensor(valid_t[idx]), torch.Tensor(valid_ohs_t[idx])
        # idx = random.sample(range(len(valid_t)), bs)          # valid_train
        # inp, lb = torch.Tensor(valid_t[idx]), torch.Tensor(valid_ohs_t[idx])
        inp = inp.cuda()
        lb = lb.cuda()

        net.train()
        out = net(inp)

        loss = loss_fn(out, lb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step(best_loss_v)

        ma_loss_t = ma_loss_t * 0.9 + loss.item() * 0.1

        net.eval()
        out = net(inp_v)
        loss = loss_fn(out, lb_v)

        ma_loss_v = ma_loss_v * 0.9 + loss.item() * 0.1
        generator.set_description('loss=%0.4f loss(valid)=%.4f' % (ma_loss_t, ma_loss_v))

        if loss.item() < best_loss_v:
            best_loss_v = loss.item()
            not_improved_cnt = 0
            best_model = net.state_dict()
        else:
            not_improved_cnt += 1

        if not_improved_cnt > 500:
            break

        # if optimizer.param_groups[0]['lr'] < 0.00001:
        #     break
    print()
    print(best_loss_v)

    print('--- retrain ---')
    net = EnsembleNet()
    net.cuda()
    optimizer = torch.optim.Adam(list(net.parameters()) + [net.w, net.w_model, net.w2], lr=0.001, weight_decay=1e-5, amsgrad=True)

    shuffled_idxs = []
    for _ in range(step):
        if len(shuffled_idxs) < bs:
            shuffled_idxs = list(range(len(valid_merged)))
            random.shuffle(shuffled_idxs)
        idx = [shuffled_idxs.pop() for _ in range(bs)]
        inp, lb = torch.Tensor(valid_merged[idx]), torch.Tensor(valid_ohs[idx])
        inp = inp.cuda()
        lb = lb.cuda()

        net.train()
        out = net(inp)

        loss = loss_fn(out, lb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ma_loss_t = ma_loss_t * 0.9 + loss.item() * 0.1

    print()
    # net.load_state_dict(best_model)

    net.eval()
    th_flat = True

    weighted_inp = np.mean(train_merged, axis=1)
    best_th = threshold_search(weighted_inp, train_ohs, flat=th_flat)
    __best_threshold = best_th
    f1_best = get_f1_threshold(weighted_inp, train_ohs, __best_threshold)
    print(__best_threshold)
    print('f1_best(naive train)=%.4f' % f1_best)

    weighted_inp = np.mean(train_v, axis=1)
    best_th = threshold_search(weighted_inp, train_ohs_v, flat=th_flat)
    __best_threshold = best_th
    f1_best = get_f1_threshold(weighted_inp, train_ohs_v, __best_threshold)
    print(__best_threshold)
    print('f1_best(naive train_v)=%.4f' % f1_best)

    weighted_inp = np.mean(valid_merged, axis=1)
    f1_best = get_f1_threshold(weighted_inp, valid_ohs, __best_threshold)
    print('f1_best(naive valid)=%.4f' % f1_best)

    best_th = threshold_search(weighted_inp, valid_ohs, flat=th_flat)
    __best_threshold = best_th
    f1_best = get_f1_threshold(weighted_inp, valid_ohs, __best_threshold)
    print(__best_threshold)
    print('f1_best(naive valid, th searched)=%.4f' % f1_best)

    weighted_inp = np.mean(train_v, axis=1)
    f1_best = get_f1_threshold(weighted_inp, train_ohs_v, __best_threshold)
    print('f1_best(naive train_v)=%.4f' % f1_best)

    print('-----')

    inp = torch.Tensor(train_merged)
    inp = inp.cuda()
    out = net(inp)
    out = out.detach().cpu().numpy()

    best_th = threshold_search(out, train_ohs, flat=th_flat)
    __best_threshold = best_th
    f1_best = get_f1_threshold(out, train_ohs, __best_threshold)

    print(__best_threshold)
    print('f1_best(train_all)=', f1_best)

    inp = torch.Tensor(train_v)
    inp = inp.cuda()
    out = net(inp)
    out = out.detach().cpu().numpy()

    # best_th = threshold_search(out, train_ohs_v, flat=th_flat)
    # __best_threshold = best_th
    f1_best = get_f1_threshold(out, train_ohs_v, __best_threshold)

    # print(__best_threshold)
    print('f1_best(train_v)=', f1_best)

    inp = torch.Tensor(valid_merged)
    inp = inp.cuda()
    out = net(inp)
    out = out.detach().cpu().numpy()

    best_th = threshold_search(out, valid_ohs, flat=th_flat)
    # __best_threshold = best_th
    f1_best = get_f1_threshold(out, valid_ohs, __best_threshold)

    # print(__best_threshold)
    print('f1_best(valid)=', f1_best)

    inp = torch.Tensor(train_v)
    inp = inp.cuda()
    out = net(inp)
    out = out.detach().cpu().numpy()

    best_th = threshold_search(out, train_ohs_v, flat=th_flat)
    f1_best = get_f1_threshold(out, train_ohs_v, __best_threshold)

    # print(__best_threshold)
    print('f1_best(train_v)=', f1_best)

    inp = torch.Tensor(test_merged)
    inp = inp.cuda()
    out = net(inp)
    out = out.detach().cpu().numpy()

    output = 'asset/ensemble_nn3.csv'
    save_pred(ids_test, out, th=__best_threshold, fname=output)

    weighted_inp = np.mean(valid_merged, axis=1)
    best_th_naive = threshold_search(weighted_inp, valid_ohs, flat=th_flat)

    out = np.mean(test_merged, axis=1)

    output = 'asset/ensemble_nn3_naive.csv'
    save_pred(ids_test, out, th=best_th_naive, fname=output)
