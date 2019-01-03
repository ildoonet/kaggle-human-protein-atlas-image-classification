import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from common import LABELS, num_class, LABELS_HPA
from data import get_dataset
from theconf import Config as C

if __name__ == '__main__':
    # k-folds, https://github.com/trent-b/iterative-stratification
    C.get()['cv_fold'] = -1

    tr_list, _, _, _ = get_dataset(False)
    labels = pd.read_csv(LABELS).set_index('Id')

    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=0)

    X = np.array(tr_list)

    def one_hot(uuid):
        lb = [int(x) for x in labels.loc[uuid]['Target'].split()]
        lb = np.eye(num_class(), dtype=np.float)[lb].sum(axis=0)
        return lb

    y = [one_hot(uuid) for uuid in tr_list]

    for fold, (train_index, test_index) in enumerate(mskf.split(X, y)):
        tr = ','.join([tr_list[i] for i in train_index])
        vl = ','.join([tr_list[i] for i in test_index])
        with open('./split/tr_names_fold%d' % fold, 'w') as f:
            f.write(tr)
        with open('./split/val_names_fold%d' % fold, 'w') as f:
            f.write(vl)

    print('done for original data')

    labels = pd.read_csv(LABELS_HPA).set_index('Id')

    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=0)

    tr_list = list(labels.index)
    X = np.array(tr_list)
    y = [one_hot(uuid) for uuid in tr_list]

    for fold, (train_index, test_index) in enumerate(mskf.split(X, y)):
        tr = ','.join([tr_list[i] for i in train_index])
        vl = ','.join([tr_list[i] for i in test_index])
        with open('./split/tr_ext_names_fold%d' % fold, 'w') as f:
            f.write(tr)
        with open('./split/val_ext_names_fold%d' % fold, 'w') as f:
            f.write(vl)
    print('done for external data(hpa v18)')
