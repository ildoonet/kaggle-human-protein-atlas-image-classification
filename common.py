import os

import pickle
import numpy as np
import pandas as pd

PATH = '/data/public/rw/kaggle-human-protein-atlas'
TRAIN = os.path.join(PATH, 'train')
TEST = os.path.join(PATH, 'test')
LABELS = os.path.join(PATH, 'train.csv')
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


def save_pred(ids, pred, th=0.0, fname='asset/submission.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line[0] > th)[0]]))
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
