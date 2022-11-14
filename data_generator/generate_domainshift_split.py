import argparse
import os
import glob
import random

import numpy as np

parser = argparse.ArgumentParser(description='Data Split')

parser.add_argument('--dataset', type=str, default='terraincognita')
parser.add_argument('--data-dir', type=str, default='../data',
                    help='path to data dir')
parser.add_argument('--subset', action='store_true')
parser.add_argument('--imb_type', type=str, default='exp')
parser.add_argument('--imb_factor', type=float, default=0.005)
parser.add_argument('--img_max_train', type=int, default=6000)
args = parser.parse_args()

data_path = os.path.join(args.data_dir, args.dataset)

if args.dataset == 'terraincognita':
    domains = ["location_38", "location_46", "location_100", "location_43", "location_33", "location_88",
               "location_120", "location_115", "location_61", "location_78", "location_105", "location_0",
               "location_7", "location_125", "location_51", "location_130", "location_28", "location_108",
               "location_40", "location_90"]
    train_domains = ["location_38", "location_46", "location_100", "location_43", "location_33",
                     "location_88", "location_61", "location_0", "location_115", "location_78"]
    val_domains = ["location_130", "location_51", "location_108", "location_40", "location_7"]
    test_domains = ["location_120", "location_105", "location_90", "location_125", "location_28"]
elif args.dataset == 'officehome':
    domains = ["art", "clipart", "product", "real"]
    train_domains = ["art", "clipart", "product"]
    test_domains = ["real"]
elif args.dataset == 'domainnet':
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    train_domains = ["infograph", "painting", "quickdraw", "real", "sketch"]
    test_domains = ["clipart"]
elif args.dataset == 'PACS':
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    train_domains = ["art_painting", "cartoon", "photo"]
    test_domains = ["sketch"]
elif args.dataset == 'VLCS':
    domains = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]
    train_domains = ["LabelMe", "SUN09", "VOC2007"]
    test_domains = ["Caltech101"]

if args.dataset == 'terraincognita':
    classes = [
        "bird", "bobcat", "cat", "coyote", "dog", "empty", "rabbit",
        "raccoon", "squirrel", "car", "skunk", "rodent", "deer"]
else:
    classes = [i.split('/')[-1] for i in list(glob.iglob(os.path.join(data_path) + '/' + domains[0] + '/*'))]


def get_random_sum(total, leave, info, c):
    random_dict = {}
    current_sum = 0
    for i, d in enumerate(train_domains):
        if i != len(train_domains) - 1:
            random_value = total - current_sum - len(train_domains) + i + 1
            if random_value > info[d][c] - leave:
                random_value = info[d][c] - leave
            value = random.randint(0, random_value)
            random_dict[d] = value
            current_sum += value
        else:
            random_dict[d] = total - current_sum
    return random_dict


def collect():
    info = {}
    for d in train_domains:
        info_domain = {}
        for i, c in enumerate(classes):
            img_paths = list(glob.iglob(os.path.join(data_path, d, c) + f'/*'))
            info_domain[c] = len(img_paths)
        info[d] = info_domain

    for d in test_domains:
        info_domain = {}
        for i, c in enumerate(classes):
            img_paths = list(glob.iglob(os.path.join(data_path, d, c) + f'/*'))
            info_domain[c] = len(img_paths)
        info[d] = info_domain

    info["all"] = {}

    for d in train_domains:
        for i, c in enumerate(classes):
            if info["all"].get(c) == None:
                info["all"][c] = info[d][c]
            else:
                info["all"][c] += info[d][c]
    return info


def get_img_num_per_cls_dmn(args, info):
    num_per_class = list(info["all"].values())
    idx = np.argsort(np.array(num_per_class))[::-1]
    num_per_class.sort(reverse=True)
    cls_num = len(classes)

    img_max = args.img_max_train
    img_num_per_cls = []
    if args.imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (args.imb_factor ** (cls_idx / (cls_num - 1.0)))
            if num > num_per_class[cls_idx]:
                num = num_per_class[cls_idx]
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
    for i, c in enumerate(classes):
        flag = False
        while flag == False:
            img_num_per_cls_dmn_train[c] = get_random_sum(img_num_per_cls[i], 1, info, c)
            flag = True
            for d in train_domains:
                if info[d][c] < img_num_per_cls_dmn_train[c][d]:
                    flag = False

    return img_num_per_cls_dmn_train


def long_tail_split(img_num_per_cls_dmn_train):
    train_num = {}
    for d in domains:
        train_num[d] = {}

    for i, c in enumerate(classes):
        for j, d in enumerate(train_domains):
            train_num[d][c] = img_num_per_cls_dmn_train[c][d]

    for d in train_domains:
        train_split_path = os.path.join(data_path, d + f"_train_{test_domains[0]}.txt")
        val_split_path = os.path.join(data_path, d + f"_val_{test_domains[0]}.txt")
        with open(train_split_path, 'w')as train_file, open(val_split_path, 'w')as val_file:
            for i, c in enumerate(classes):
                img_paths = list(glob.iglob(os.path.join(data_path, d, c) + f'/*'))
                for j, img_path in enumerate(img_paths):
                    if j < train_num[d][c]:
                        train_file.write(img_path.split('/', 3)[3] + f" {i}\n")

    for d in test_domains:
        train_split_path = os.path.join(data_path, d + f"_train_{test_domains[0]}.txt")
        val_split_path = os.path.join(data_path, d + f"_val_{test_domains[0]}.txt")
        with open(train_split_path, 'w')as train_file, open(val_split_path, 'w')as val_file:
            for i, c in enumerate(classes):
                img_paths = list(glob.iglob(os.path.join(data_path, d, c) + f'/*'))
                for j, img_path in enumerate(img_paths):
                    val_file.write(img_path.split('/', 3)[3] + f" {i}\n")


def simple_split():
    for d in domains:
        train_split_path = os.path.join(data_path, d + "_train_ood.txt")
        val_split_path = os.path.join(data_path, d + "_val_ood.txt")
        test_split_path = os.path.join(data_path, d + "_test_ood.txt")
        with open(train_split_path, 'w')as train_file, open(val_split_path, 'w')as val_file, open(test_split_path,
                                                                                                  'w')as test_file:
            # with open(val_split_path, 'w')as val_file, open(test_split_path,  'w')as test_file:
            for i, c in enumerate(classes):
                img_paths = list(glob.iglob(os.path.join(data_path, d, c) + f'/*'))
                for j, img_path in enumerate(img_paths):
                    if d in train_domains:
                        train_file.write(img_path.split('/', 3)[3] + f" {i}\n")
                    if d in val_domains:
                        val_file.write(img_path.split('/', 3)[3] + f" {i}\n")
                    elif d in test_domains:
                        test_file.write(img_path.split('/', 3)[3] + f" {i}\n")

if __name__ == '__main__':
    if args.subset:
        info = collect()
    # print(info)
        img_num_per_cls_dmn_train = get_img_num_per_cls_dmn(args, info)
        long_tail_split(img_num_per_cls_dmn_train)
    else:  # for TerraIncoginta (naturally imbalance)
        simple_split()
