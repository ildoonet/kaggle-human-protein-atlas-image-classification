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
test_aug_bs = 6
test_aug_sz = 12

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


def save_pred(ids, pred, th=0.0, fname='asset/submission.csv', valid_pred=None):
    pred_list = []
    for line in pred:
        if len(line) != len(name_label_dict):
            line = line[0]
        # TODO : select best one if no output
        if len(np.nonzero(np.array(line) >= th)[0]) > 0:
            s = ' '.join(list([str(i) for i in np.nonzero(np.array(line) >= th)[0]]))
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
            'prediction': pred
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    if valid_pred:
        fname_pred = fname.replace('.csv', '.valid.pkl')
        with open(fname_pred, 'wb') as f:
            pickle.dump({
                'threshold': th,
                'prediction': valid_pred
            }, f, protocol=pickle.HIGHEST_PROTOCOL)


def threshold_search(preds, ys):
    if isinstance(preds, list):
        preds = np.concatenate(preds, axis=0)
        ys = np.concatenate(ys, axis=0)
    ths = np.zeros(1) + 0.5

    # maximize f1
    error_f = lambda p: np.concatenate((
        1 - get_f1_threshold_soft(preds, ys, p),
        0 * (p-0.5)
    ), axis=None)
    found, success = opt.leastsq(error_f, ths)
    return found

    #
    ths = np.zeros(num_class()) + found[0]

    # maximize f1
    error_f = lambda p: np.concatenate((
        1 - get_f1_threshold_soft(preds, ys, p),
        1e-6 * (p - found[0])
    ), axis=None)
    founds, success = opt.leastsq(error_f, ths)

    # founds = founds - 2./100.

    return founds
