import csv
import os
import sys
import json
import torch
import random
import argparse
import datetime
import numpy as np
import torch.optim as optim

from tempfile import mkdtemp
from collections import defaultdict

from utils import Logger, save_best_model, return_predict_fn, return_criterion
from tally import TALLY
from config import dataset_defaults
from initialize import initialize_dataloader

parser = argparse.ArgumentParser(description='TALLY')

# General
parser.add_argument('--dataset', type=str, default='iwildcam',
                    help="Name of dataset, choose from PACS, VLCS, officehome, domainnet, terraincognita, iwildcam")
parser.add_argument('--experiment', type=str, default='.',
                    help='experiment name, set as . for automatic naming.')
parser.add_argument('--data-dir', type=str, default='./data',
                    help='path to data dir')
parser.add_argument("--save_dir", default='result', type=str)
parser.add_argument("--ckpt", default=None, type=str)
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument("--b", action='store_false')
parser.add_argument("--a", action='store_true')
parser.add_argument('--seed', type=int, default=-1)

# Computation
parser.add_argument("--warmup_epochs", default=7, type=int)
parser.add_argument('--iter_num', type=int, default=500)
parser.add_argument("--print_loss_iters", default=100, type=int)
parser.add_argument("--split", default="sub", type=str)
parser.add_argument("--eval_batch_size", default=256, type=int)
parser.add_argument("--c", default=512, type=int)
parser.add_argument("--h", default=28, type=int)
parser.add_argument("--w", default=28, type=int)
parser.add_argument("--mix_alpha", default=0.5, type=float)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

args_dict = args.__dict__
args_dict.update(dataset_defaults[args.dataset])
args = argparse.Namespace(**args_dict)

if args.seed == -1:
    args.seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

runId = datetime.datetime.now().isoformat().replace(':', '_')

args.experiment = f"{args.dataset}_{args.seed}" \
    if args.experiment == '.' else args.experiment

directory_name = '../runs/{}'.format(args.experiment)
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
runPath = mkdtemp(prefix=runId, dir=directory_name)

# logging setup
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:' + runPath)
print(args)
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
torch.save(args, '{}/args.rar'.format(runPath))

train_loader, tally_loader, tv_loaders = initialize_dataloader(args, device)
train_data = train_loader.dataset
val_loader, test_loader = tv_loaders['val'], tv_loaders['test']

model = TALLY(train_data.class_num, args.mix_alpha).to(device)

assert args.optimiser in ['SGD', 'Adam', 'AdamW'], "Invalid choice of optimiser, choose between 'Adam' and 'SGD'"
opt = getattr(optim, args.optimiser)
optimiserC = opt(model.parameters(), **args.optimiser_args)

predict_fn = return_predict_fn(args.dataset)
criterion = return_criterion(args, train_data, False).to(device)


def warmup(train_loader, train_data, epoch, agg):
    running_loss = 0
    total_iters = len(train_loader)
    criterion = return_criterion(args, train_data, False).to(device)
    print('\n====> Epoch: {:03d} (Warmup) '.format(epoch))

    for i, data in enumerate(train_loader):
        model.train()
        x, y, g, idx = data[0].to(device), data[1].to(device), data[2].to(device), data[3]

        y_hat, norm, sig, mu = model.forward(x)

        optimiserC.zero_grad()
        loss = criterion(y_hat, y)
        loss.backward()
        optimiserC.step()

        train_data.update_statistics(norm.detach().cpu(), sig.detach().cpu(), mu.detach().cpu(), idx, idx)

        running_loss += loss.item()
        # print statistics
        if (i + 1) % args.print_loss_iters == 0:
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print('iter {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_loss_iters))
            running_loss = 0.0


def train(train_loader, train_data, epoch, agg):
    running_loss = 0
    total_iters = len(train_loader)
    criterion = return_criterion(args, train_data, True).to(device)
    print('\n====> Epoch: {:03d} '.format(epoch))

    for i, data in enumerate(train_loader):
        model.train()
        x, y, g, idx, n, s, m = data[0].to(device), data[1].to(device), data[2].to(device), data[3], \
                                data[4].to(device), data[5].to(device), data[6].to(device)
        x1, x2 = x[:args.batch_size], x[args.batch_size:]
        y = y[:args.batch_size]
        g = g[args.batch_size:]
        idx1, idx2 = idx[:args.batch_size], idx[args.batch_size:]
        n1, n2 = n[:args.batch_size], n[args.batch_size:]
        s1, s2 = s[:args.batch_size], s[args.batch_size:]
        m1, m2 = m[:args.batch_size], m[args.batch_size:]

        y_hat, norm, sig, mu = model.forward_train_aug(x1, n1, x2, s2, m2)

        optimiserC.zero_grad()
        loss = criterion(y_hat, y)
        loss.backward()
        optimiserC.step()

        train_data.update_statistics(norm.detach().cpu(), sig.detach().cpu(), mu.detach().cpu(), idx1, idx2)

        running_loss += loss.item()
        # print statistics
        if (i + 1) % args.print_loss_iters == 0:
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print('iter {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_loss_iters))
            running_loss = 0.0

    #test(val_loader, agg, loader_type='val')
    if args.split in ("id", "ood"):
        test(test_loader, agg, loader_type='val')
    else:
        test(val_loader, agg, loader_type='val')
    save_best_model(model, runPath, agg, args)


def test(test_loader, agg, loader_type='test', verbose=True, save_ypred=False):
    model.eval()
    yhats, ys, metas = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # get the inputs
            x, y = batch[0].to(device), batch[1].to(device)
            y_hat = model(x)
            ys.append(y)
            yhats.append(y_hat)
            metas.append(batch[2])

        ypreds, ys, metas = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(metas)

        if save_ypred:
            save_name = f"{args.dataset}_split:{loader_type}_seed:" \
                        f"{args.seed}_epoch:best_pred.csv"
            with open(f"{runPath}/{save_name}", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(ypreds.unsqueeze(1).cpu().tolist())
        if isinstance(test_loader.dataset, torch.utils.data.dataset.Subset):
            test_val = test_loader.dataset.dataset.eval(ypreds.cpu(), ys.cpu(), metas)
        else:
            test_val = test_loader.dataset.eval(ypreds.cpu(), ys.cpu(), metas)
        agg[f'{loader_type}_stat'].append(test_val[args.selection_metric])
        if verbose:
            print(f"=============== {loader_type} ===============\n{test_val}")


if __name__ == '__main__':
    print("=" * 30 + f"Training: TALLY" + "=" * 30)

    agg = defaultdict(list)
    agg['val_stat'] = [0.]
    agg['test_stat'] = [0.]

    if args.ckpt is not None:
        print("Saving Representation...")
        model.load_state_dict(torch.load(args.ckpt + '/model.rar'))
        if args.split in ("id", "ood"):
            test(test_loader, agg, loader_type='test', save_ypred=True)
        else:
            test(val_loader, agg, loader_type='val', save_ypred=True)

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            warmup(train_loader, train_data, epoch, agg)
        else:
            train(tally_loader, train_data, epoch, agg)
    model.load_state_dict(torch.load(runPath + '/model.rar'))
    print('Finished training! Loading best model...')
    for split, loader in tv_loaders.items():
        test(loader, agg, loader_type=split, save_ypred=True)





