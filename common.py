import itertools
import os

import pickle
import numpy as np
import pandas as pd
import scipy.optimize as opt

from metric import get_f1_threshold, get_f1_threshold_soft, get_precision_soft

PATH = '/data/public/rw/kaggle-human-protein-atlas'
TRAIN = os.path.join(PATH, 'train')
TEST = os.path.join(PATH, 'test')
LABELS = os.path.join(PATH, 'train.csv')
LABELS_HPA = os.path.join('/data/public/rw/kaggle-human-protein-atlas/hpa_v18', 'HPAv18RBGY_wodpl.csv')
SAMPLE = os.path.join(PATH, 'sample_submission.csv')
test_aug_sz = 16

name_label_dict = {
    0: 'Nucleoplasm',
    1: 'Nuclear membrane',
    2: 'Nucleoli',
    3: 'Nucleoli fibrillar center',
    4: 'Nuclear speckles',
    5: 'Nuclear bodies',
    6: 'Endoplasmic reticulum',
    7: 'Golgi apparatus',
    8: 'Peroxisomes',
    9: 'Endosomes',
    10: 'Lysosomes',
    11: 'Intermediate filaments',
    12: 'Actin filaments',
    13: 'Focal adhesion sites',
    14: 'Microtubules',
    15: 'Microtubule ends',
    16: 'Cytokinetic bridge',
    17: 'Mitotic spindle',
    18: 'Microtubule organizing center',
    19: 'Centrosome',
    20: 'Lipid droplets',
    21: 'Plasma membrane',
    22: 'Cell junctions',
    23: 'Mitochondria',
    24: 'Aggresome',
    25: 'Cytosol',
    26: 'Cytoplasmic bodies',
    27: 'Rods & rings'}


def num_class():
    return len(name_label_dict)


def save_pred(ids, pred, feat=None, th=0.0, fname='asset/submission.csv', valid_pred=None, train_pred=None):
    pred_list = []
    for line in pred:
        if len(line) != len(name_label_dict):
            line = line[0]
        # select best one if no output
        non_zeros = np.nonzero(np.array(line) >= th)[0]
        if len(non_zeros) > 0:
            s = ' '.join(list([str(i) for i in non_zeros]))
        else:
            s = str(np.argmax(line))
        pred_list.append(s)

    df = pd.DataFrame({'Id': ids, 'Predicted': pred_list})
    df.sort_values(by='Id').to_csv(fname, header=True, index=False)

    # save predictions
    fname_pred = fname.replace('.csv', '.pkl')
    with open(fname_pred, 'wb') as f:
        pickle.dump({
            'threshold': th,
            'prediction': pred,
            'feature': feat,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    if valid_pred:
        fname_pred = fname.replace('.csv', '.valid.pkl')
        with open(fname_pred, 'wb') as f:
            pickle.dump({
                'threshold': th,
                'prediction': valid_pred['prediction'],
                'feature': valid_pred['feature'],
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    if train_pred:
        fname_pred = fname.replace('.csv', '.train.pkl')
        with open(fname_pred, 'wb') as f:
            pickle.dump({
                'threshold': th,
                'prediction': train_pred['prediction'],
                'feature': train_pred['feature'],
            }, f, protocol=pickle.HIGHEST_PROTOCOL)


def threshold_search(preds, ys, flat=True):
    if isinstance(preds, list):
        preds = np.concatenate(preds, axis=0)
        ys = np.concatenate(ys, axis=0)
    # grid search
    max_f1 = -1.0
    max_th = 0.5
    for delta in range(30, 60):
        th = delta / 100.
        f1 = get_f1_threshold(preds, ys, th)
        if max_f1 <= f1:
            max_f1 = f1
            max_th = th
    found = max_th

    if flat:
        return found

    from itertools import product
    delta = [-0.05, 0.0, 0.05]
    delta = [-0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.20, 0.25, 0.0]

    max_th = [max_th] * num_class()
    for cls in range(num_class()):
        before_changed = max_th.copy()
        for d in delta:
            new_th = before_changed.copy()
            if new_th[cls] + d <= 0.1:
                continue

            new_th[cls] += d
            f1 = get_f1_threshold(preds, ys, new_th)
            if max_f1 <= f1:
                max_f1 = f1
                max_th = new_th
    return max_th


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)
