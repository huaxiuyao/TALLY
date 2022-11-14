import os
import sys
import shutil
import torch
from criterions import CrossEntropyLoss


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    filepath = filepath
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)


def save_best_model(model, runPath, agg, args):
    if agg['val_stat'][-1] >= max(agg['val_stat'][:-1]):
        save_model(model, f'{runPath}/model.rar')
        save_vars(agg, f'{runPath}/losses.rar')


def return_criterion(args, dataset, train):
    return CrossEntropyLoss(dataset.get_cls_num_list(),  args.b and train)


def single_class_predict_fn(yhat):
    _, predicted = torch.max(yhat.data, 1)
    return predicted


def return_predict_fn(dataset):
    return {
        'iwildcam': single_class_predict_fn,
        'domainnet': single_class_predict_fn,
        'officehome': single_class_predict_fn,
        'VLCS': single_class_predict_fn,
        'PACS': single_class_predict_fn,
        'terraincognita': single_class_predict_fn
    }[dataset]
