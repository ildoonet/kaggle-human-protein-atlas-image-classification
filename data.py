import itertools
import os
import numpy as np
import cv2
import random
import torch
import torch.utils.data
import pandas as pd
from imgaug import augmenters as iaa
from theconf import Config as C

from common import PATH, name_label_dict, LABELS, TEST, TRAIN, SAMPLE, test_aug_sz, LABELS_HPA


class Oversampling:
    def __init__(self, path):
        self.train_labels = pd.read_csv(path).set_index('Id')
        self.train_labels['Target'] = [[int(i) for i in s.split()]
                                       for s in self.train_labels['Target']]
        # set the minimum number of duplicates for each class
        self.multi = [1, 1, 1, 1, 1, 1, 1, 1, 8, 8,
                      8, 1, 1, 1, 1, 8, 1, 2, 1, 1,
                      4, 1, 1, 1, 2, 1, 2, 8]
        # TODO : different oversampling? https://www.kaggle.com/wordroid/inceptionresnetv2-resize256-f1loss-lb0-419

    def get(self, image_id):
        labels = self.train_labels.loc[image_id, 'Target'] if image_id \
                                                              in self.train_labels.index else []
        m = 1
        for l in labels:
            if m < self.multi[l]: m = self.multi[l]
        return m


def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0, 0)):
    """
    Reference : https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
    """
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout


def get_dataset(oversample=True):
    cv_fold = C.get()['cv_fold']

    if cv_fold < 0:
        with open(os.path.join('./split/tr_names.txt'), 'r') as text_file:
            tr_n = text_file.read().split(',')
        with open(os.path.join('./split/val_names.txt'), 'r') as text_file:
            val_n = text_file.read().split(',')
        cval_n = val_n
    else:
        with open(os.path.join('./split/tr_names_fold%d' % cv_fold), 'r') as text_file:
            tr_n = text_file.read().split(',')
        with open(os.path.join('./split/val_names_fold%d' % cv_fold), 'r') as text_file:
            val_n = text_file.read().split(',')
        with open(os.path.join('./split/val_names.txt'), 'r') as text_file:
            cval_n = text_file.read().split(',')

    # test_names = sorted({f[:36] for f in os.listdir(TEST)})
    with open(SAMPLE, 'r') as text_file:
        test_names = [x.split(',')[0] for x in text_file.readlines()[1:]]

    # print(len(tr_n), len(val_n), len(test_names))
    if oversample:
        s = Oversampling(os.path.join(PATH, LABELS))
        tr_n = [idx for idx in tr_n for _ in range(s.get(idx))]

    return tr_n, val_n, cval_n, test_names


class KaggleDataset(torch.utils.data.Dataset):
    def __init__(self, setname, data_list, aug=False):
        super().__init__()

        self.setname = setname
        self.list = data_list
        self.labels = pd.read_csv(LABELS).set_index('Id')
        self.default_path = TEST if self.setname == 'tests' else TRAIN
        if C.get()['highres']:
            self.default_path = self.default_path.replace('train', 'train_full_size').replace('test', 'test_full_size')
            self.resize = True
            self.ext = '.tif'
        else:
            self.resize = False
            self.ext = '.png'
        self.aug = aug

    def __len__(self):
        tta = 1
        if self.aug:
            tta = test_aug_sz
        return len(self.list) * tta

    def __getitem__(self, item):
        if not self.aug:
            uuid = self.list[item]
        else:
            uuid = self.list[item // test_aug_sz]

        colors = ['red', 'green', 'blue', 'yellow']
        flags = cv2.IMREAD_GRAYSCALE
        img = [cv2.imread(os.path.join(self.default_path, uuid + '_' + color + self.ext), flags) for color in colors]
        if self.resize:
            img = [cv2.resize(x, (1024, 1024)) for x in img]

        img = np.stack(img, axis=-1)

        # TODO : data augmentation zoom/shear/brightness
        if 'train' in self.setname:
            augment_img = iaa.Sequential([
                iaa.OneOf([
                    iaa.Affine(rotate=0),
                    iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270),
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                ])
            ], random_order=True)
            img = augment_img.augment_image(img)

            # cutout
            if C.get()['cutout_p'] > 0.0:
                img = cutout(C.get()['cutout_size'], C.get()['cutout_p'], False)(img)

            # TODO : channel drop(except green)?
            # d_ch = random.choice([0, 2, 3])
            # img[:, :, d_ch] = 0

        if self.aug:
            # teat-time aug. : tta
            tta_list = list(itertools.product(
                [iaa.Affine(rotate=0), iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270)],
                [iaa.Fliplr(0.0), iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)])]
            ))
            tta_idx = item % len(tta_list)
            img = tta_list[tta_idx][0].augment_image(img)
            img = tta_list[tta_idx][1].augment_image(img)

        img = img.astype(np.float32)
        img /= 255.  # TODO : different normalization?
        img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(img)

        if self.setname == 'tests':
            lb = np.zeros(len(name_label_dict), dtype=np.int)
        else:
            lb = [int(x) for x in self.labels.loc[uuid]['Target'].split()]
            lb = np.eye(len(name_label_dict), dtype=np.float)[lb].sum(axis=0)
        return img, lb


class HPADataset(KaggleDataset):
    def __init__(self, setname, data_list):
        csv = pd.read_csv(LABELS_HPA).set_index('Id')
        super().__init__(setname, data_list, aug=False)
        if C.get()['highres']:
            self.default_path = '/data/public/rw/kaggle-human-protein-atlas/hpa_v18/images_2048'
        else:
            self.default_path = '/data/public/rw/kaggle-human-protein-atlas/hpa_v18/images'
        self.labels = csv
        self.ext = '.png'


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return sum([len(x) for x in self.datasets])

    def __getitem__(self, item):
        for dataset in self.datasets:
            if item < len(dataset):
                return dataset[item]
            item -= len(dataset)
        raise Exception(item)

    def set_aug(self, aug):
        for d in self.datasets:
            d.aug = aug


def get_dataloaders(tests_aug=False):
    tr, vl, cvl, ts = get_dataset()
    if C.get()['extdata']:
        if C.get()['cv_fold'] >= 0:
            with open(os.path.join('./split/tr_ext_names_fold%d' % C.get()['cv_fold']), 'r') as text_file:
                tr_n = text_file.read().split(',')
            with open(os.path.join('./split/val_ext_names_fold%d' % C.get()['cv_fold']), 'r') as text_file:
                val_n = text_file.read().split(',')
            ds_train = CombinedDataset(KaggleDataset('train', tr), HPADataset('train_hpa_v18', tr_n))
            ds_valid = CombinedDataset(KaggleDataset('valid', tr), HPADataset('valid_hpa_v18', val_n))
        else:
            with open(os.path.join('./split/tr_ext_names_fold0'), 'r') as text_file:
                tr_n = text_file.read().split(',')
            with open(os.path.join('./split/val_ext_names_fold0'), 'r') as text_file:
                val_n = text_file.read().split(',')
            tr_n += val_n
            ds_train = CombinedDataset(KaggleDataset('train', tr), HPADataset('train_hpa_v18', tr_n))
            ds_valid = KaggleDataset('valid', tr)
    else:
        ds_train = KaggleDataset('train', tr)
        ds_valid = KaggleDataset('valid', vl, aug=False)
    ds_cvalid = KaggleDataset('cvalid', cvl, aug=False)
    ds_tests = KaggleDataset('tests', ts, aug=tests_aug)
    print('data size=', len(ds_train), len(ds_valid), len(ds_cvalid), len(ds_tests))

    d_train = torch.utils.data.DataLoader(ds_train, C.get()['batch'], pin_memory=True, num_workers=16 if C.get()['highres'] else 128, shuffle=True, drop_last=True)
    d_valid = torch.utils.data.DataLoader(ds_valid, C.get()['batch'], pin_memory=True, num_workers=4, shuffle=False, drop_last=True)
    d_cvalid = torch.utils.data.DataLoader(ds_cvalid, C.get()['batch'], pin_memory=True, num_workers=4, shuffle=False, drop_last=True)
    d_tests = torch.utils.data.DataLoader(ds_tests, test_aug_sz if tests_aug else 1, pin_memory=True, num_workers=16, shuffle=False)

    return d_train, d_valid, d_cvalid, d_tests


def get_dataloaders_eval(tta=True):
    tr, vl, cvl, ts = get_dataset()
    with open('./split/sampled_names', 'r') as text_file:
        tr = text_file.read().split(',')
    with open('./split/sampled_ext_names', 'r') as text_file:
        ext_n = text_file.read().split(',')
    ds_train = CombinedDataset(KaggleDataset('train', tr), HPADataset('train_hpa_v18', ext_n))
    ds_train.set_aug(tta)
    ds_cvalid = KaggleDataset('cvalid', cvl, aug=tta)
    ds_tests = KaggleDataset('tests', ts, aug=tta)

    d_train = torch.utils.data.DataLoader(ds_train, test_aug_sz if tta else 1, pin_memory=True, num_workers=16, shuffle=False)
    d_valid = torch.utils.data.DataLoader(ds_cvalid, test_aug_sz if tta else 1, pin_memory=True, num_workers=16, shuffle=False)
    d_tests = torch.utils.data.DataLoader(ds_tests, test_aug_sz if tta else 1, pin_memory=True, num_workers=16, shuffle=False)
    return d_train, d_valid, d_tests
