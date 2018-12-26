import itertools
import os
import numpy as np
import cv2
import torch
import torch.utils.data
import pandas as pd
from imgaug import augmenters as iaa
from theconf import Config as C

from common import PATH, name_label_dict, LABELS, TEST, TRAIN, SAMPLE, test_aug_bs, test_aug_sz


class Oversampling:
    def __init__(self, path):
        self.train_labels = pd.read_csv(path).set_index('Id')
        self.train_labels['Target'] = [[int(i) for i in s.split()]
                                       for s in self.train_labels['Target']]
        # set the minimum number of duplicates for each class
        self.multi = [1, 1, 1, 1, 1, 1, 1, 1,
                      8, 8, 8, 1, 1, 1, 1, 8,
                      1, 2, 1, 1, 4, 1, 1, 1,
                      2, 1, 2, 8]
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


def get_dataset():
    # TODO : HPA external data
    cv_fold = C.get()['cv_fold']

    with open(os.path.join('./split/tr_names.txt'), 'r') as text_file:
        tr_n = text_file.read().split(',')
    with open(os.path.join('./split/val_names.txt'), 'r') as text_file:
        val_n = text_file.read().split(',')
    # test_names = sorted({f[:36] for f in os.listdir(TEST)})
    with open(SAMPLE, 'r') as text_file:
        test_names = [x.split(',')[0] for x in text_file.readlines()[1:]]

    print(len(tr_n), len(val_n), len(test_names))
    s = Oversampling(os.path.join(PATH, LABELS))
    tr_n = [idx for idx in tr_n for _ in range(s.get(idx))]

    return tr_n, val_n, test_names


class KaggleDataset(torch.utils.data.Dataset):
    def __init__(self, setname, data_list, aug=False):
        super().__init__()

        self.setname = setname
        self.list = data_list
        self.labels = pd.read_csv(LABELS).set_index('Id')
        self.default_path = TEST if self.setname == 'tests' else TRAIN
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
        img = [cv2.imread(os.path.join(self.default_path, uuid + '_' + color + '.png'), flags).astype(np.float32) for color in colors]
        img = np.stack(img, axis=-1)
        img /= 255.

        # TODO : data augmentation zoom/shear/brightness
        if self.setname == 'train':
            augment_img = iaa.Sequential([
                iaa.OneOf([
                    iaa.Affine(rotate=0),
                    iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270),
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                ])], random_order=True)
            img = augment_img.augment_image(img)

            # cutout
            if C.get()['cutout_p'] > 0.0:
                img = cutout(C.get()['cutout_size'], C.get()['cutout_p'], False)(img)

        if self.aug:
            # teat-time aug. : tta
            tta_list = list(itertools.product(
                [iaa.Affine(rotate=0), iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270)],
                [iaa.Fliplr(0.0), iaa.Fliplr(1.0), iaa.Flipud(1.0)]))
            tta_idx = item % len(tta_list)
            img = tta_list[tta_idx][0].augment_image(img)
            img = tta_list[tta_idx][1].augment_image(img)

        img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(img)

        if self.setname == 'tests':
            lb = np.zeros(len(name_label_dict), dtype=np.int)
        else:
            lb = [int(x) for x in self.labels.loc[uuid]['Target'].split()]
            lb = np.eye(len(name_label_dict), dtype=np.float)[lb].sum(axis=0)
        return img, lb


def get_dataloaders(tests_aug=False):
    tr, vl, ts = get_dataset()
    ds_train = KaggleDataset('train', tr)
    ds_valid = KaggleDataset('valid', vl, aug=False)
    ds_tests = KaggleDataset('tests', ts, aug=tests_aug)

    d_train = torch.utils.data.DataLoader(ds_train, C.get()['batch'], pin_memory=True, num_workers=32, shuffle=True, drop_last=True)
    d_valid = torch.utils.data.DataLoader(ds_valid, 64, pin_memory=True, num_workers=4, shuffle=False, drop_last=True)
    d_tests = torch.utils.data.DataLoader(ds_tests, test_aug_sz if tests_aug else 1, pin_memory=True, num_workers=4, shuffle=False)

    return d_train, d_valid, d_tests
