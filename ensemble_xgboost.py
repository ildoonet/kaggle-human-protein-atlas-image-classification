import sys

import pandas as pd
import pickle
import numpy as np
import xgboost as xgb

from theconf import Config as C

# setup parameters for xgboost
from common import LABELS, num_class, threshold_search, save_pred
from data import get_dataset, get_dataloaders
from metric import get_f1_threshold

param = {}
# use softmax multi-class classification
# param['objective'] = 'multi:softmax'
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.05
# param['max_depth'] = 1
param['silent'] = 1
param['nthread'] = 8
param['base_score'] = 0.5
param['booster'] = 'gblinear'
param['lambda'] = 0.00001
param['alpha'] = 0.00005
param['feature_selector'] = 'shuffle'
# param['n_estimators'] = 200
param['subsample'] = 0.8
param['colsample_bynode'] = 0.85
param['num_class'] = num_class()


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

    models = ensemble_181228
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
        valid_preds.append(np.concatenate(d_valid[key], axis=0))
        test_preds.append(np.concatenate(d_test[key], axis=0))

    valid_merged = np.concatenate(valid_preds, axis=1)
    test_merged = np.concatenate(test_preds, axis=1)

    valid_expand = []
    valid_lbs = []
    valid_ohs = []
    for uuid, val_x in zip(ids_cvalid, valid_merged):
        for x in dataset.loc[uuid]['Target'].split():
            lb = int(x)
            valid_expand.append(val_x)
            valid_lbs.append(lb)
        lb = [int(x) for x in dataset.loc[uuid]['Target'].split()]
        lb_onehot = np.eye(num_class(), dtype=np.float)[lb].sum(axis=0)
        valid_ohs.append(lb_onehot)
    valid_expand = np.array(valid_expand, dtype=np.float)
    valid_lbs = np.array(valid_lbs, dtype=np.int)
    valid_ohs = np.array(valid_ohs, dtype=np.float)

    # TODO
    split_idx = 1900
    valid_expand_t = []
    valid_lbs_t = []
    valid_expand_v = []
    valid_lbs_v = []
    for uuid, val_x in zip(ids_cvalid, valid_merged):
        for x in dataset.loc[uuid]['Target'].split():
            lb = int(x)
            if len(valid_expand_t) < split_idx:
                valid_expand_t.append(val_x)
                valid_lbs_t.append(lb)
            else:
                valid_expand_v.append(val_x)
                valid_lbs_v.append(lb)
        lb = [int(x) for x in dataset.loc[uuid]['Target'].split()]
        lb_onehot = np.eye(num_class(), dtype=np.float)[lb].sum(axis=0)
    valid_expand_t = np.array(valid_expand_t, dtype=np.float)
    valid_expand_v = np.array(valid_expand_v, dtype=np.float)
    valid_lbs_t = np.array(valid_lbs_t, dtype=np.int)
    valid_lbs_v = np.array(valid_lbs_v, dtype=np.int)

    # TODO
    # valid_lbs = valid_lbs[:len(valid_merged)]
    valid_ohs = valid_ohs[:len(valid_merged)]

    xg_train = xgb.DMatrix(valid_expand, label=valid_lbs)
    xg_train2 = xgb.DMatrix(valid_merged)
    xg_test = xgb.DMatrix(test_merged)

    num_round = 128
    early_stop = 30
    watchlist = [(xg_train, 'train')]
    cvr = xgb.cv(
        dtrain=xg_train, params=param, nfold=5, stratified=True,
        num_boost_round=num_round, early_stopping_rounds=early_stop, metrics="merror", seed=100
    )
    print(cvr['train-merror-mean'].tail(1))
    print(cvr['test-merror-mean'].tail(1))
    print('-----')

    print('----- train on subset')
    xg_subset_t = xgb.DMatrix(valid_expand_t, label=valid_lbs_t)
    xg_subset_v = xgb.DMatrix(valid_expand_v, label=valid_lbs_v)
    bst = xgb.train(
        param, xg_subset_t, num_round,
        [(xg_subset_t, 'train'), (xg_subset_v, 'valid')], early_stopping_rounds=early_stop
    )
    xgpred_val = bst.predict(xgb.DMatrix(valid_merged[:split_idx]))
    best_th = threshold_search(xgpred_val, valid_ohs[:split_idx])
    __best_threshold = best_th
    f1_best = get_f1_threshold(xgpred_val, valid_ohs[:split_idx], __best_threshold)
    print(__best_threshold)
    print('f1_best=', f1_best)

    sys.exit(0)

    # print('----------- train')
    # bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=early_stop)
    #
    # xgpred_val = bst.predict(xg_train2)
    # best_th = threshold_search(xgpred_val, valid_ohs)
    # __best_threshold = best_th
    # f1_best = get_f1_threshold(xgpred_val, valid_ohs, __best_threshold)
    # print(__best_threshold)
    # print('f1_best=', f1_best)
    #
    # f1_best = get_f1_threshold(xgpred_val, valid_ohs, 0.5)
    # print('f1_best(@0.5)=', f1_best)

    xgpred_tst = bst.predict(xg_test)

    output = 'asset/ensemble_xgboost.csv'
    save_pred(ids_test, xgpred_tst, th=__best_threshold, fname=output)

    pass
