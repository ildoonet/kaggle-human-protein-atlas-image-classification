import numpy as np
import torch
from theconf import Config as C, ConfigArgumentParser
from torch.nn import BCELoss, MultiLabelMarginLoss
from tqdm import tqdm
from imgaug import augmenters as iaa

from common import save_pred, test_aug_sz, num_class, threshold_search
from data import get_dataloaders, get_dataset
from metric import FocalLoss, acc, get_f1_threshold, get_f1, f1_loss, stats_by_class
from models.densenet import Densenet121, Densenet161, Densenet169, Densenet201
from models.etc import PNasnet, Nasnet, Polynet, SENet154
from models.inception import InceptionV3, InceptionV4
from models.resnet import Resnet34, Resnet50, Resnet101, Resnet152
from tensorboardX import SummaryWriter

from models.vgg import Vgg16

__best_threshold = 0.2      # TODO : different threshold for each classes
__f1_ths = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]


def run_epoch(model, it_data, optimizer=None, title='', aug=False, bt_update=True):
    global __best_threshold, __f1_ths
    losses = []
    f1s = [[] for _ in range(len(__f1_ths))]
    t = tqdm(it_data)
    preds = []
    ys = []

    # TODO : BCE? FocalLoss?
    # loss_f = FocalLoss()
    if C.get()['loss'] == 'f1':
        loss_f = f1_loss
    elif C.get()['loss'] == 'bce':
        loss_f = BCELoss(reduction='mean')
    elif C.get()['loss'] == 'margin':
        loss_f = MultiLabelMarginLoss()
    else:
        raise Exception('invalid loss=%s' % C.get()['loss'])

    for cnt, (x, y) in enumerate(t):
        pred_y = model(x.cuda())
        if not aug:
            if len(pred_y.shape) < 2:
                pred_y = pred_y.unsqueeze(0)
            pred_y = pred_y.cuda().float()
        else:
            means = []
            targs = []
            for i in range(0, len(x), test_aug_sz):
                mean_y = torch.mean(pred_y[i:i+test_aug_sz], dim=0, keepdim=True)
                means.append(mean_y.squeeze())
                targs.append(y[i])
            pred_y = torch.stack(means, dim=0)
            y = torch.stack(targs, dim=0)

        if C.get()['loss'] == 'margin':
            y = y.cuda().long()
        else:
            y = y.cuda().float()
        loss = loss_f(pred_y, y)

        lr_curr = 0.0
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_curr = optimizer.param_groups[0]['lr']

        losses.append(loss.item())
        preds.append(pred_y.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())

        if title != 'test' and cnt % 20 == 0:
            preds_concat = np.concatenate(preds, axis=0)
            ys_concat = np.concatenate(ys, axis=0)
            for i, th in enumerate(__f1_ths):
                f1 = get_f1_threshold(preds_concat, ys_concat, th)
                f1s[i] = f1

        desc = ['[%s]' % title]
        if title == 'test':
            if isinstance(__best_threshold, np.ndarray):
                bt_str = ','.join(['%.1f' % t for t in __best_threshold])
                desc.append(' best_th=%s' % bt_str)
            else:
                desc.append(' best_th=%.3f' % __best_threshold)
        else:
            desc.append('loss=%.4f' % np.mean(losses))
            f1_desc = ' '.join(['%.3f@%.2f' % (f1, th) for th, f1 in zip(__f1_ths, f1s)])
            desc.append('f1(%s)' % f1_desc)

        if 'train' in title:
            desc.append(' lr=%.5f' % lr_curr)
        desc = ' '.join(desc)
        t.set_description(desc)

        del pred_y, loss

    if title == 'valid' and bt_update:
        __best_threshold = __f1_ths[np.argmax(f1s)]

    if title != 'test':
        preds_concat = np.concatenate(preds, axis=0)
        ys_concat = np.concatenate(ys, axis=0)
        for i, th in enumerate(__f1_ths):
            f1 = get_f1_threshold(preds_concat, ys_concat, th)
            f1s[i] = f1
        stats = stats_by_class(preds_concat, ys_concat)
    else:
        stats = []

    return {
        'loss': np.mean(losses),
        'prediction': preds,
        'labels': ys,
        'f1_scores': f1s,
        'stats': stats
    }


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--extdata', type=int, default=1)
    # parser.add_argument('--dump', type=str, default=None, help='config dump filepath')
    parsed_args = parser.parse_args()
    print(C.get_instance())

    writer = SummaryWriter('asset/log2/%s' % C.get()['name'])

    _, _, _, ids_test = get_dataset()
    d_train, d_valid, d_cvalid, d_tests = get_dataloaders(tests_aug=C.get()['eval'])

    models = {
        'resnet34': Resnet34,
        'resnet50': Resnet50,
        'resnet101': Resnet101,
        'resnet152': Resnet152,
        'inception_v3': InceptionV3,
        'inception_v4': InceptionV4,
        'vgg16': Vgg16,
        'densenet121': Densenet121,
        'densenet161': Densenet161,
        'densenet169': Densenet169,
        'densenet201': Densenet201,
        'pnasnet': PNasnet,
        'nasnet': Nasnet,
        'polynet': Polynet,
        'senet154': SENet154,
    }

    model = models[C.get()['model']](True)
    model = torch.nn.DataParallel(model)
    model.cuda()

    if C.get()['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=C.get()['lr'], betas=(0.9, 0.999), amsgrad=True, weight_decay=C.get()['optimizer_reg']
        )
    elif C.get()['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=C.get()['lr'], momentum=0.9, weight_decay=1e-5, nesterov=True
        )
    else:
        raise Exception(C.get()['optimizer'])
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    # TODO : CosineAnnealingLR?
    prev_train_loss = 9999999
    best_valid_loss = 9999999
    best_valid_epoch = 0

    epoch = 1

    if not C.get()['eval']:
        if C.get()['load']:
            print('load...')
            checkpoint = torch.load('asset/%s.pt' % C.get()['name'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch'] + 1

        for curr_epoch in range(epoch, C.get()['epoch'] + 1):
            # ----- train -----
            lr_schedule.step(metrics=prev_train_loss)
            model.train()
            train_result = run_epoch(model, d_train, optimizer=optimizer, title='train@%03d' % curr_epoch)
            prev_train_loss = train_result['loss']
            writer.add_scalar('loss/train', prev_train_loss, curr_epoch)
            writer.add_scalar('f1_best/train', np.max(train_result['f1_scores']), curr_epoch)
            p, r = train_result['stats']
            for class_idx in range(num_class()):
                writer.add_scalar('precision_train/class%d' % class_idx, p[class_idx], curr_epoch)
                writer.add_scalar('recall_train/class%d' % class_idx, r[class_idx], curr_epoch)

            del train_result

            # ----- eval on valid/test -----
            model.eval()
            if curr_epoch % 10 == 0:
                valid_result = run_epoch(model, d_valid, title='valid', aug=False)
                valid_loss = valid_result['loss']

                writer.add_scalar('loss/valid-in-fold', valid_loss, curr_epoch)
                writer.add_scalar('f1_best/valid-in-fold', np.max(valid_result['f1_scores']), curr_epoch)

                # cross-valid-set and test-set
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    best_valid_epoch = curr_epoch

                    # run on cross-valid set
                    valid_result = run_epoch(model, d_cvalid, title='cvalid', aug=False)
                    valid_loss = valid_result['loss']
                    writer.add_scalar('loss/valid', valid_loss, curr_epoch)
                    writer.add_scalar('f1_best/valid', np.max(valid_result['f1_scores']), curr_epoch)
                    p, r = valid_result['stats']
                    for class_idx in range(num_class()):
                        writer.add_scalar('precision_valid/class%d' % class_idx, p[class_idx], curr_epoch)
                        writer.add_scalar('recall_valid/class%d' % class_idx, r[class_idx], curr_epoch)

                    # run on test set
                    preds_test = run_epoch(model, d_tests, title='test', aug=False)['prediction']
                    save_pred(ids_test, preds_test, th=__best_threshold, fname='asset/%s.csv' % C.get()['name'], valid_pred=valid_result['prediction'])

                    torch.save({
                        'epoch': curr_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, 'asset/%s.best.pt' % C.get()['name'])

                torch.save({
                    'epoch': curr_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'asset/%s.pt' % C.get()['name'])

        print('test result @ epoch=%d' % best_valid_epoch)
    else:
        print('load...')
        checkpoint = torch.load('asset/%s.best.pt' % C.get()['name'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print('epoch@%d' % checkpoint['epoch'])

        # only for evaluation
        model.eval()
        cvalid_result = run_epoch(model, d_cvalid, title='valid', aug=False)

        # TODO : threshold search
        best_th = threshold_search(cvalid_result['prediction'], cvalid_result['labels'])
        __best_threshold = best_th
        print('best_th=', ' '.join(['%.3f' % x for x in __best_threshold]))

        preds_concat = np.concatenate(cvalid_result['prediction'], axis=0)
        ys_concat = np.concatenate(cvalid_result['labels'], axis=0)
        f1_best = get_f1_threshold(preds_concat, ys_concat, __best_threshold)
        print('f1_best=', f1_best)

        preds_test = run_epoch(model, d_tests, title='test', aug=True)['prediction']
        save_pred(ids_test, preds_test, th=__best_threshold, fname='asset/%s.aug.csv' % C.get()['name'], valid_pred=cvalid_result['prediction'])

        valid_result = run_epoch(model, d_valid, title='valid-in-fold', aug=False, bt_update=False)
        preds_concat = np.concatenate(valid_result['prediction'], axis=0)
        ys_concat = np.concatenate(valid_result['labels'], axis=0)
        f1_best2 = get_f1_threshold(preds_concat, ys_concat, __best_threshold)

        print(__best_threshold)
        # print('best_th=', ' '.join(['%.3f' % x for x in __best_threshold]))
        print('f1_best(valid)=', f1_best2)
        print('f1_best(cvalid)=', f1_best)
        print(valid_result['loss'], max(valid_result['f1_scores']), valid_result['f1_scores'])
        print(cvalid_result['loss'], max(cvalid_result['f1_scores']), cvalid_result['f1_scores'])
