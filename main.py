import numpy as np
import torch
from theconf import Config as C, ConfigArgumentParser
from tqdm import tqdm
from imgaug import augmenters as iaa

from common import save_pred, test_aug_sz
from data import get_dataloaders, get_dataset
from metric import FocalLoss, acc, get_f1_threshold, get_f1, f1_loss, stats_by_class
from models.densenet import Densenet121, Densenet161, Densenet169, Densenet201
from models.inception import InceptionV3
from models.resnet import Resnet34, Resnet50, Resnet101, Resnet152
from tensorboardX import SummaryWriter

from models.vgg import Vgg16

__best_threshold = 0.2      # TODO : different threshold for each classes
__f1_ths = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def run_epoch(model, it_data, optimizer=None, title='', aug=False):
    global __best_threshold, __f1_ths
    losses = []
    f1s = [[] for _ in range(len(__f1_ths))]
    t = tqdm(it_data)
    preds = []
    stats = []

    # loss_f = FocalLoss()
    loss_f = f1_loss

    for x, y in t:
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

        y = y.cuda().float()
        loss = loss_f(pred_y, y)

        lr_curr = 0.0
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_curr = optimizer.param_groups[0]['lr']

        losses.append(loss.item())
        if title == 'test':
            preds.append(pred_y.detach().cpu().numpy())
        else:
            for i, th in enumerate(__f1_ths):
                f1 = get_f1_threshold(pred_y, y, th)
                f1s[i].append(f1.item())
            stats_by_class(pred_y, y)

        desc = ['[%s]' % title]
        if title == 'test':
            desc.append(' best_th=%.3f' % __best_threshold)
        else:
            desc.append('loss=%.4f' % np.mean(losses))
            f1_desc = ' '.join(['%.3f@%.2f' % (np.mean(f1), th) for th, f1 in zip(__f1_ths, f1s)])
            desc.append('f1(%s)' % f1_desc)

        if 'train' in title:
            desc.append(' lr=%.5f' % lr_curr)
        desc = ' '.join(desc)
        t.set_description(desc)

    if title == 'valid':
        __best_threshold = __f1_ths[np.argmax([np.mean(f1) for f1 in f1s])]

    return {
        'loss': np.mean(losses),
        'prediction': preds,
        'f1_scores': f1s
    }


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--eval', type=bool, default=False)
    # parser.add_argument('--dump', type=str, default=None, help='config dump filepath')
    parsed_args = parser.parse_args()
    print(C.get_instance())

    writer = SummaryWriter('asset/log/%s' % C.get()['name'])

    _, _, ids_test = get_dataset()
    d_train, d_valid, d_tests = get_dataloaders(tests_aug=C.get()['eval'])

    models = {
        'resnet34': Resnet34,
        'resnet50': Resnet50,
        'resnet101': Resnet101,
        'resnet152': Resnet152,
        'inception_v3': InceptionV3,
        'vgg16': Vgg16,
        'densenet121': Densenet121,
        'densenet161': Densenet161,
        'densenet169': Densenet169,
        'densenet201': Densenet201,
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
            epoch = checkpoint['epoch']

        for curr_epoch in range(epoch, C.get()['epoch'] + 1):
            # ----- train -----
            lr_schedule.step(metrics=prev_train_loss)
            model.train()
            train_result = run_epoch(model, d_train, optimizer=optimizer, title='train@%03d' % curr_epoch)
            prev_train_loss = train_result['loss']
            writer.add_scalars('loss', {
                'train': prev_train_loss,
            }, curr_epoch)

            # ----- eval on valid/test -----
            model.eval()
            if curr_epoch % 10 == 0:
                valid_result = run_epoch(model, d_valid, title='valid', aug=False)
                valid_loss = valid_result['loss']
                writer.add_scalars('loss', {
                    'train': prev_train_loss,
                    'valid': valid_loss
                }, curr_epoch)
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    best_valid_epoch = curr_epoch

                    # run on test set
                    # TODO : test-time augmentation
                    preds_test = run_epoch(model, d_tests, title='test', aug=False)['prediction']
                    save_pred(ids_test, preds_test, th=__best_threshold, fname='asset/%s.csv' % C.get()['name'])

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
        valid_result = run_epoch(model, d_valid, title='valid', aug=False)

        preds_test = run_epoch(model, d_tests, title='test', aug=True)['prediction']
        save_pred(ids_test, preds_test, th=__best_threshold, fname='asset/%s.aug.csv' % C.get()['name'])
