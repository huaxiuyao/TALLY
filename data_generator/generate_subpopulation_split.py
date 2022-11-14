import argparse
import os
import glob
import random

import numpy as np

parser = argparse.ArgumentParser(description='Data Split')

parser.add_argument('--dataset', type=str, default='officehome')
parser.add_argument('--data-dir', type=str, default='../data',
                    help='path to data dir')
parser.add_argument('--imb_type', type=str, default='exp')
parser.add_argument('--imb_factor', type=float, default=0.005)
parser.add_argument('--img_max_train', type=int, default=6000)
parser.add_argument('--img_val', type=int, default=50)
parser.add_argument('--img_test', type=int, default=100)
args = parser.parse_args()

data_path = os.path.join(args.data_dir, args.dataset)

if args.dataset == 'officehome':
    domains = ["art", "clipart", "product", "real"]
elif args.dataset == 'domainnet':
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
elif args.dataset == 'PACS':
    domains = ["art_painting", "cartoon", "photo", "sketch"]
elif args.dataset == 'VLCS':
    domains = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]
else:
    raise NotImplementedError

classes = [i.split('/')[-1] for i in list(glob.iglob(os.path.join(data_path) + '/' + domains[0] + '/*'))]


def get_random_sum(total, leave, info, c):
    flag = False
    random_dict = {}
    while flag == False:
        random_dict = {}
        current_sum = 0
        flag = True
        for i, d in enumerate(domains):
            if i != len(domains) - 1:
                random_value = total - current_sum - len(domains) + i + 1
                if random_value > info[d][c] - leave:
                    random_value = info[d][c] - leave
                value = random.randint(1, random_value)
                random_dict[d] = value
                current_sum += value
            else:
                random_dict[d] = total - current_sum
                if random_dict[d] > info[d][c] - leave:
                    flag = False
    return random_dict


def collect():
    info = {}
    for d in domains:
        info_domain = {}
        for i, c in enumerate(classes):
            if not os.path.exists(os.path.join(data_path, d, c)):
                info_domain[c] = 0
                continue
            img_paths = list(glob.iglob(os.path.join(data_path, d, c) + f'/*'))
            info_domain[c] = len(img_paths)
        info[d] = info_domain
    info["all"] = {}

    for d in domains:
        for i, c in enumerate(classes):
            if info["all"].get(c) == None:
                info["all"][c] = info[d][c]
            else:
                info["all"][c] += info[d][c]
    print(sum(list(info["all"].values())))
    return info


def get_img_num_per_cls_dmn(args, info):
    num_per_class = list(info["all"].values())
    idx = np.argsort(np.array(num_per_class))[::-1]
    num_per_class.sort(reverse=True)
    cls_num = len(classes)
    dmn_num = len(domains)

    img_max = args.img_max_train
    img_val = args.img_val
    img_test = args.img_test
    img_num_per_cls = []
    if args.imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (args.imb_factor ** (cls_idx / (cls_num - 1.0)))
            if num > num_per_class[cls_idx] - img_val * dmn_num - img_test * dmn_num:
                num = num_per_class[cls_idx] - img_val * dmn_num - img_test * dmn_num
            if num < dmn_num:
                num = dmn_num
            img_num_per_cls.append(int(num))
    elif args.imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * args.imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    img_num_per_cls = np.array(img_num_per_cls)
    tmp = np.ones_like(img_num_per_cls)
    tmp[idx] = img_num_per_cls
    img_num_per_cls = tmp.tolist()

    img_num_per_cls_dmn_train = {}
    img_num_per_cls_dmn_val = {}
    img_num_per_cls_dmn_test = {}

    flag = False
    while flag == False:
        for i, c in enumerate(classes):
            img_num_per_cls_dmn_train[c] = get_random_sum(img_num_per_cls[i], img_val + img_test, info, c)
            img_num_per_cls_dmn_val[c] = {}
            img_num_per_cls_dmn_test[c] = {}
            for d in domains:
                img_num_per_cls_dmn_val[c][d] = img_val
                img_num_per_cls_dmn_test[c][d] = img_test

        flag = True

        for i, c in enumerate(classes):
            for d in domains:
                if info[d][c] < img_num_per_cls_dmn_train[c][d] + img_num_per_cls_dmn_val[c][d] + \
                        img_num_per_cls_dmn_test[c][d]:
                    print(d, c, info[d][c], img_num_per_cls_dmn_train[c][d], img_num_per_cls_dmn_val[c][d], img_num_per_cls_dmn_test[c][d])
                    flag = False

        domain_sample_num = {}
        for d in domains:
            domain_sample_num[d] = 0
            for i, c in enumerate(classes):
                domain_sample_num[d] += img_num_per_cls_dmn_train[c][d]
        domain_sample_num = list(domain_sample_num.values())
        if max(domain_sample_num) / min(domain_sample_num) < 2:
            flag = False

    return img_num_per_cls_dmn_train, img_num_per_cls_dmn_val, img_num_per_cls_dmn_test


def long_tail_split(img_num_per_cls_dmn_train, img_num_per_cls_dmn_val, img_num_per_cls_dmn_test):
    train_num = {}
    val_num = {}
    test_num = {}
    for d in domains:
        train_num[d] = {}
        val_num[d] = {}
        test_num[d] = {}

    for i, c in enumerate(classes):
        for j, d in enumerate(domains):
            train_num[d][c] = img_num_per_cls_dmn_train[c][d]
            val_num[d][c] = img_num_per_cls_dmn_val[c][d]
            test_num[d][c] = img_num_per_cls_dmn_test[c][d]

    for d in domains:
        train_split_path = os.path.join(data_path, d + "_train_sub.txt")
        val_split_path = os.path.join(data_path, d + "_val_sub.txt")
        test_split_path = os.path.join(data_path, d + "_test_sub.txt")
        with open(train_split_path, 'w')as train_file, open(val_split_path, 'w')as val_file, open(test_split_path, 'w')as test_file:
            for i, c in enumerate(classes):
                img_paths = list(glob.iglob(os.path.join(data_path, d, c) + f'/*'))
                random.shuffle(img_paths)
                for j, img_path in enumerate(img_paths):
                    if j < train_num[d][c]:
                        train_file.write(img_path.split('/', 3)[3] + f" {i}\n")
                    elif train_num[d][c] <= j < train_num[d][c] + val_num[d][c]:
                        val_file.write(img_path.split('/', 3)[3] + f" {i}\n")
                    elif train_num[d][c] + val_num[d][c] <= j < train_num[d][c] + val_num[d][c] + test_num[d][c]:
                        test_file.write(img_path.split('/', 3)[3] + f" {i}\n")


if __name__ == '__main__':
    info = collect()
    img_num_per_cls_dmn_train, img_num_per_cls_dmn_val, img_num_per_cls_dmn_test = get_img_num_per_cls_dmn(args, info)
    long_tail_split(img_num_per_cls_dmn_train, img_num_per_cls_dmn_val, img_num_per_cls_dmn_test)