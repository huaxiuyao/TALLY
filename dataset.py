import os

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from wilds.common.metrics.all_metrics import Accuracy, Recall, F1

ImageFile.LOAD_TRUNCATED_IMAGES = True


class IWildCam(Dataset):

    class_num = 186

    def __init__(self, args, train_data, domain_idx=0):
        domains = train_data.metadata_array[:, domain_idx]
        train_data._input_array = [train_data.dataset._input_array[i] for i in train_data.indices]

        self.metadata_array = train_data.metadata_array
        self.y_array = train_data.y_array
        self.data = train_data._input_array

        self.collate = train_data.collate
        self.metadata_fields = train_data.metadata_fields
        self.data_dir = train_data.data_dir
        if 'iwildcam' in str(self.data_dir):
            self.data_dir = f'{self.data_dir}/train'
        self.transform = train_data.transform

        self.data = train_data._input_array
        self.labels = self.y_array
        self.domains = domains
        self.pairs = self.domains * self.class_num + self.labels

        self.unique_domains, self.domains_inverse = torch.unique(self.domains, return_inverse=True)
        self.unique_labels, self.labels_inverse = torch.unique(self.labels, return_inverse=True)
        self.unique_pairs = torch.unique(self.pairs)

        self.num_domains = len(self.unique_domains)
        self.num_classes = len(self.unique_labels)

        self.labels_indices = [torch.nonzero(self.labels == cls).squeeze(-1) for cls in self.unique_labels]
        self.domains_indices = [torch.nonzero(self.domains == loc).squeeze(-1) for loc in self.unique_domains]
        print("Each domain numbers", [len(d) for d in self.domains_indices])

        self.classes_norm = torch.randn(self.num_classes, args.c, args.h, args.w)
        self.domains_sigma = torch.randn(self.num_domains, args.c, 1, 1)
        self.domains_mean = torch.randn(self.num_domains, args.c, 1, 1)

    def get_input(self, idx):
        """Returns x for a given idx."""
        img_path = f'{self.data_dir}/{self.data[idx]}'
        img = Image.open(img_path)
        if isinstance(self.data_dir, str) and not ('iwildcam' in self.data_dir):
            img = img.convert('RGB')
        return img

    def get_balance_pair(self, batch_size):
        idx1, idx2 = [], []

        for _ in range(batch_size):
            class_idx = np.random.choice(len(self.unique_labels), 1)[0]
            feat_idx = np.random.choice(len(self.labels_indices[class_idx]), 1)[0]
            idx1.append(self.labels_indices[class_idx][feat_idx])

            domain_idx = \
                np.random.choice(len(self.unique_domains), 1)[0]
            feat_idx = \
                np.random.choice(len(self.domains_indices[domain_idx]), 1)[0]
            idx2.append(self.domains_indices[domain_idx][feat_idx])

        idx = idx1 + idx2

        return idx

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.class_num):
            cls_num_list.append((self.labels == i).sum().item())
        return cls_num_list

    def update_statistics(self, norm, sig, mu, idx1, idx2):
        y_unique = self.labels_inverse[idx1].unique()
        d_unique = self.domains_inverse[idx2].unique()

        classes_feat_update = [
            norm[self.labels_inverse[idx1] == cls].mean(dim=0).unsqueeze(dim=0) for cls in
            y_unique]
        classes_feat_update = torch.cat(classes_feat_update)

        domains_mean_update = [
            mu[self.domains_inverse[idx2] == loc].mean(dim=0).unsqueeze(dim=0) for loc in
            d_unique]
        domains_mean_update = torch.cat(domains_mean_update)

        domains_sigma_update = [
            sig[self.domains_inverse[idx2] == loc].mean(dim=0).unsqueeze(dim=0) for loc in
            d_unique]
        domains_sigma_update = torch.cat(domains_sigma_update)

        self.classes_norm[y_unique] = 0.8 * self.classes_norm[y_unique] + 0.2 * classes_feat_update
        self.domains_mean[d_unique] = 0.8 * self.domains_mean[d_unique] + 0.2 * domains_mean_update
        self.domains_sigma[d_unique] = 0.8 * self.domains_sigma[d_unique] + 0.2 * domains_sigma_update

    def __getitem__(self, idx):
        label = self.labels[idx]
        domain = self.domains[idx]
        img = self.transform(self.get_input(idx))
        norm = self.classes_norm[self.labels_inverse[idx]]
        sigma = self.domains_sigma[self.domains_inverse[idx]]
        mean = self.domains_mean[self.domains_inverse[idx]]
        return img, label, domain, idx, norm, sigma, mean

    def __len__(self):
        return len(self.labels)

    def eval(self, ypreds, ys, meta):
        acc_per_class = np.zeros(len(self.unique_labels))
        for i, c in enumerate(self.unique_labels):
            correct = np.zeros(len(self.domains[ys == c].unique()))
            num = np.zeros(len(self.domains[ys == c].unique()))
            domain_unique = self.domains[ys == c].unique()
            for j, d in enumerate(domain_unique):
                correct[j] = ((ypreds == c) & (ys == c) & (meta == d)).sum()
                num[j] = ((ys == c) & (meta == d)).sum()
            acc_per_class[i] = (correct / num).mean()

        acc_per_domain = np.zeros(len(self.unique_domains))
        for i, d in enumerate(self.unique_domains):
            correct = np.zeros(len(self.labels[meta == d].unique()))
            num = np.zeros(len(self.labels[meta == d].unique()))
            class_unique = self.labels[meta == d].unique()
            for j, c in enumerate(class_unique):
                correct[j] = ((ypreds == c) & (ys == c) & (meta == d)).sum()
                num[j] = ((ys == c) & (meta == d)).sum()
            acc_per_domain[i] = (correct / num).mean()

        correct_per_pair = np.zeros(len(self.unique_pairs))
        num_per_pair = np.zeros(len(self.unique_pairs))
        for i, p in enumerate(self.unique_pairs):
            correct_per_pair[i] = (ypreds == ys)[(meta * self.class_num + ys) == p].sum()
            num_per_pair[i] = ((meta * self.class_num + ys) == p).sum()

        metrics = [
            Accuracy(prediction_fn=None),
            Recall(prediction_fn=None, average='macro'),
            F1(prediction_fn=None, average='macro'),
        ]

        test_val = \
            {'Average acc': metrics[0].compute(ypreds, ys, False),
             'Recall macro': metrics[1].compute(ypreds, ys, False),
             'F1-macro': metrics[2].compute(ypreds, ys, False),
             'Average Class acc': acc_per_class.mean(),
             'Worst Class acc': acc_per_class.min(),
             'Average Domain acc': acc_per_domain.mean(),
             'Worst Domain acc': acc_per_domain.min(),
             'Average Pair acc': (correct_per_pair / num_per_pair).mean(),
             'Worst Pair acc': (correct_per_pair / num_per_pair).min(),
             }
        return test_val


class DomainNet(Dataset):
    """
    It consists of 345 classes in 6 domains:
    clipart, infograph, painting, quickdraw, real, sketch
    """

    class_num = 345
    domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    def __init__(self, args, data_dir="data", split="train", transform=None):
        super().__init__()
        self.labels = []
        self.img_paths = []
        self.domains = []
        for i, domain in enumerate(self.domain_names):
            labels_file = os.path.join(data_dir, "domainnet", f"{domain}_{split}_{args.split}.txt")
            img_dir = os.path.join(data_dir, "domainnet")
            with open(labels_file) as f:
                content = [line.rstrip().split(" ") for line in f]
            self.img_paths = self.img_paths + [os.path.join(img_dir, x[0]) for x in content]
            self.labels = self.labels + [int(x[1]) for x in content]
            self.domains = self.domains + [i for x in content]

        self.labels = torch.Tensor(self.labels).long()
        self.domains = torch.Tensor(self.domains).long()
        self.pairs = self.domains * self.class_num + self.labels

        self.unique_domains, self.domains_inverse = torch.unique(self.domains, return_inverse=True)
        self.unique_labels, self.labels_inverse = torch.unique(self.labels, return_inverse=True)
        self.unique_pairs = torch.unique(self.pairs)

        self.num_domains = len(self.unique_domains)
        self.num_classes = len(self.unique_labels)

        self.domains_indices = [torch.nonzero(self.domains == loc).squeeze(-1) for loc in self.domains.unique()]
        self.labels_indices = [torch.nonzero(self.labels == cls).squeeze(-1) for cls in self.labels.unique()]

        self.transform = transform

        self.classes_norm = torch.randn(self.num_classes, args.c, args.h, args.w)
        self.domains_sigma = torch.randn(self.num_domains, args.c, 1, 1)
        self.domains_mean = torch.randn(self.num_domains, args.c, 1, 1)

    def get_balance_pair(self, batch_size):
        idx1, idx2 = [], []

        for _ in range(batch_size):
            class_idx = np.random.choice(len(self.unique_labels), 1)[0]
            feat_idx = np.random.choice(len(self.labels_indices[class_idx]), 1)[0]
            idx1.append(self.labels_indices[class_idx][feat_idx])

            domain_idx = \
                np.random.choice(len(self.unique_domains), 1)[0]
            feat_idx = \
                np.random.choice(len(self.domains_indices[domain_idx]), 1)[0]
            idx2.append(self.domains_indices[domain_idx][feat_idx])

        idx = idx1 + idx2

        return idx

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.class_num):
            cls_num_list.append((self.labels == i).sum().item())
        return cls_num_list

    def update_statistics(self, norm, sig, mu, idx1, idx2):
        y_unique = self.labels_inverse[idx1].unique()
        d_unique = self.domains_inverse[idx2].unique()

        classes_feat_update = [
            norm[self.labels_inverse[idx1] == cls].mean(dim=0).unsqueeze(dim=0) for cls in
            y_unique]
        classes_feat_update = torch.cat(classes_feat_update)

        domains_mean_update = [
            mu[self.domains_inverse[idx2] == loc].mean(dim=0).unsqueeze(dim=0) for loc in
            d_unique]
        domains_mean_update = torch.cat(domains_mean_update)

        domains_sigma_update = [
            sig[self.domains_inverse[idx2] == loc].mean(dim=0).unsqueeze(dim=0) for loc in
            d_unique]
        domains_sigma_update = torch.cat(domains_sigma_update)

        self.classes_norm[y_unique] = 0.8 * self.classes_norm[y_unique] + 0.2 * classes_feat_update
        self.domains_mean[d_unique] = 0.8 * self.domains_mean[d_unique] + 0.2 * domains_mean_update
        self.domains_sigma[d_unique] = 0.8 * self.domains_sigma[d_unique] + 0.2 * domains_sigma_update

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        domain = self.domains[idx]
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        norm = self.classes_norm[self.labels_inverse[idx]]
        sigma = self.domains_sigma[self.domains_inverse[idx]]
        mean = self.domains_mean[self.domains_inverse[idx]]
        return img, label, domain, idx, norm, sigma, mean

    def eval(self, ypreds, ys, meta):
        acc_per_class = np.zeros(len(self.unique_labels))
        for i, c in enumerate(self.unique_labels):
            correct = np.zeros(len(self.domains[ys == c].unique()))
            num = np.zeros(len(self.domains[ys == c].unique()))
            domain_unique = self.domains[ys == c].unique()
            for j, d in enumerate(domain_unique):
                correct[j] = ((ypreds == c) & (ys == c) & (meta == d)).sum()
                num[j] = ((ys == c) & (meta == d)).sum()
            acc_per_class[i] = (correct / num).mean()

        acc_per_domain = np.zeros(len(self.unique_domains))
        for i, d in enumerate(self.unique_domains):
            correct = np.zeros(len(self.labels[meta == d].unique()))
            num = np.zeros(len(self.labels[meta == d].unique()))
            class_unique = self.labels[meta == d].unique()
            for j, c in enumerate(class_unique):
                correct[j] = ((ypreds == c) & (ys == c) & (meta == d)).sum()
                num[j] = ((ys == c) & (meta == d)).sum()
            acc_per_domain[i] = (correct / num).mean()

        correct_per_pair = np.zeros(len(self.unique_pairs))
        num_per_pair = np.zeros(len(self.unique_pairs))
        for i, p in enumerate(self.unique_pairs):
            correct_per_pair[i] = (ypreds == ys)[(meta * self.class_num + ys) == p].sum()
            num_per_pair[i] = ((meta * self.class_num + ys) == p).sum()

        metrics = [
            Accuracy(prediction_fn=None),
            Recall(prediction_fn=None, average='macro'),
            F1(prediction_fn=None, average='macro'),
        ]

        test_val = \
            {'Average acc': metrics[0].compute(ypreds, ys, False),
             'Recall macro': metrics[1].compute(ypreds, ys, False),
             'F1-macro': metrics[2].compute(ypreds, ys, False),
             'Average Class acc': acc_per_class.mean(),
             'Worst Class acc': acc_per_class.min(),
             'Average Domain acc': acc_per_domain.mean(),
             'Worst Domain acc': acc_per_domain.min(),
             'Average Pair acc': (correct_per_pair / num_per_pair).mean(),
             'Worst Pair acc': (correct_per_pair / num_per_pair).min(),
            }
        return test_val


class VLCS(Dataset):
    """
    It consists of 5 classes in 4 domains:
    Caltech101, LabelMe, SUN09, VOC2007
    """

    class_num = 5
    domain_names = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]

    def __init__(self, args, data_dir="data", split="train", transform=None):
        super().__init__()
        self.labels = []
        self.img_paths = []
        self.domains = []
        for i, domain in enumerate(self.domain_names):
            labels_file = os.path.join(data_dir, "VLCS", f"{domain}_{split}_{args.split}.txt")
            img_dir = os.path.join(data_dir, "VLCS")
            with open(labels_file) as f:
                content = [line.rstrip().split(" ") for line in f]
            self.img_paths = self.img_paths + [os.path.join(img_dir, x[0]) for x in content]
            self.labels = self.labels + [int(x[1]) for x in content]
            self.domains = self.domains + [i for x in content]

        self.labels = torch.Tensor(self.labels).long()
        self.domains = torch.Tensor(self.domains).long()
        self.pairs = self.domains * self.class_num + self.labels

        self.unique_domains, self.domains_inverse = torch.unique(self.domains, return_inverse=True)
        self.unique_labels, self.labels_inverse = torch.unique(self.labels, return_inverse=True)
        self.unique_pairs = torch.unique(self.pairs)

        self.num_domains = len(self.unique_domains)
        self.num_classes = len(self.unique_labels)

        self.domains_indices = [torch.nonzero(self.domains == loc).squeeze(-1) for loc in self.domains.unique()]
        self.labels_indices = [torch.nonzero(self.labels == cls).squeeze(-1) for cls in self.labels.unique()]

        self.transform = transform

        self.classes_norm = torch.randn(self.num_classes, args.c, args.h, args.w)
        self.domains_sigma = torch.randn(self.num_domains, args.c, 1, 1)
        self.domains_mean = torch.randn(self.num_domains, args.c, 1, 1)

    def get_balance_pair(self, batch_size):
        idx1, idx2 = [], []

        for _ in range(batch_size):
            class_idx = np.random.choice(len(self.unique_labels), 1)[0]
            feat_idx = np.random.choice(len(self.labels_indices[class_idx]), 1)[0]
            idx1.append(self.labels_indices[class_idx][feat_idx])

            domain_idx = \
                np.random.choice(len(self.unique_domains), 1)[0]
            feat_idx = \
                np.random.choice(len(self.domains_indices[domain_idx]), 1)[0]
            idx2.append(self.domains_indices[domain_idx][feat_idx])

        idx = idx1 + idx2

        return idx

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.class_num):
            cls_num_list.append((self.labels == i).sum().item())
        return cls_num_list

    def update_statistics(self, norm, sig, mu, idx1, idx2):
        y_unique = self.labels_inverse[idx1].unique()
        d_unique = self.domains_inverse[idx2].unique()

        classes_feat_update = [
            norm[self.labels_inverse[idx1] == cls].mean(dim=0).unsqueeze(dim=0) for cls in
            y_unique]
        classes_feat_update = torch.cat(classes_feat_update)

        domains_mean_update = [
            mu[self.domains_inverse[idx2] == loc].mean(dim=0).unsqueeze(dim=0) for loc in
            d_unique]
        domains_mean_update = torch.cat(domains_mean_update)

        domains_sigma_update = [
            sig[self.domains_inverse[idx2] == loc].mean(dim=0).unsqueeze(dim=0) for loc in
            d_unique]
        domains_sigma_update = torch.cat(domains_sigma_update)

        self.classes_norm[y_unique] = 0.8 * self.classes_norm[y_unique] + 0.2 * classes_feat_update
        self.domains_mean[d_unique] = 0.8 * self.domains_mean[d_unique] + 0.2 * domains_mean_update
        self.domains_sigma[d_unique] = 0.8 * self.domains_sigma[d_unique] + 0.2 * domains_sigma_update

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        domain = self.domains[idx]
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        norm = self.classes_norm[self.labels_inverse[idx]]
        sigma = self.domains_sigma[self.domains_inverse[idx]]
        mean = self.domains_mean[self.domains_inverse[idx]]
        return img, label, domain, idx, norm, sigma, mean

    def eval(self, ypreds, ys, meta):
        acc_per_class = np.zeros(len(self.unique_labels))
        for i, c in enumerate(self.unique_labels):
            correct = np.zeros(len(self.domains[ys == c].unique()))
            num = np.zeros(len(self.domains[ys == c].unique()))
            domain_unique = self.domains[ys == c].unique()
            for j, d in enumerate(domain_unique):
                correct[j] = ((ypreds == c) & (ys == c) & (meta == d)).sum()
                num[j] = ((ys == c) & (meta == d)).sum()
            acc_per_class[i] = (correct / num).mean()

        acc_per_domain = np.zeros(len(self.unique_domains))
        for i, d in enumerate(self.unique_domains):
            correct = np.zeros(len(self.labels[meta == d].unique()))
            num = np.zeros(len(self.labels[meta == d].unique()))
            class_unique = self.labels[meta == d].unique()
            for j, c in enumerate(class_unique):
                correct[j] = ((ypreds == c) & (ys == c) & (meta == d)).sum()
                num[j] = ((ys == c) & (meta == d)).sum()
            acc_per_domain[i] = (correct / num).mean()

        correct_per_pair = np.zeros(len(self.unique_pairs))
        num_per_pair = np.zeros(len(self.unique_pairs))
        for i, p in enumerate(self.unique_pairs):
            correct_per_pair[i] = (ypreds == ys)[(meta * self.class_num + ys) == p].sum()
            num_per_pair[i] = ((meta * self.class_num + ys) == p).sum()

        metrics = [
            Accuracy(prediction_fn=None),
            Recall(prediction_fn=None, average='macro'),
            F1(prediction_fn=None, average='macro'),
        ]

        test_val = \
            {'Average acc': metrics[0].compute(ypreds, ys, False),
             'Recall macro': metrics[1].compute(ypreds, ys, False),
             'F1-macro': metrics[2].compute(ypreds, ys, False),
             'Average Class acc': acc_per_class.mean(),
             'Worst Class acc': acc_per_class.min(),
             'Average Domain acc': acc_per_domain.mean(),
             'Worst Domain acc': acc_per_domain.min(),
             'Average Pair acc': (correct_per_pair / num_per_pair).mean(),
             'Worst Pair acc': (correct_per_pair / num_per_pair).min(),
             }
        return test_val


class PACS(Dataset):
    """
    It consists of 7 classes in 4 domains:
    art_painting, cartoon, photo, sketch
    """

    class_num = 7
    domain_names = ["art_painting", "cartoon", "photo", "sketch"]

    def __init__(self, args, data_dir="data", split="train", transform=None):
        """
        Arguments:
            root: The dataset must be located at ```<root>/domainnet```
            domain: One of the 6 domains
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        super().__init__()
        self.labels = []
        self.img_paths = []
        self.domains = []
        for i, domain in enumerate(self.domain_names):
            # if i == 3 and split == "train":
            #    continue
            labels_file = os.path.join(data_dir, "PACS", f"{domain}_{split}_{args.split}.txt")
            img_dir = os.path.join(data_dir, "PACS")
            with open(labels_file) as f:
                content = [line.rstrip().split(" ") for line in f]
            self.img_paths = self.img_paths + [os.path.join(img_dir, x[0]) for x in content]
            self.labels = self.labels + [int(x[1]) for x in content]
            self.domains = self.domains + [i for x in content]

        self.labels = torch.Tensor(self.labels).long()
        self.domains = torch.Tensor(self.domains).long()
        self.pairs = self.domains * self.class_num + self.labels

        self.unique_domains, self.domains_inverse = torch.unique(self.domains, return_inverse=True)
        self.unique_labels, self.labels_inverse = torch.unique(self.labels, return_inverse=True)
        self.unique_pairs = torch.unique(self.pairs)

        self.num_domains = len(self.unique_domains)
        self.num_classes = len(self.unique_labels)

        self.domains_indices = [torch.nonzero(self.domains == loc).squeeze(-1) for loc in self.domains.unique()]
        self.labels_indices = [torch.nonzero(self.labels == cls).squeeze(-1) for cls in self.labels.unique()]

        self.transform = transform

        self.classes_norm = torch.randn(self.num_classes, args.c, args.h, args.w)
        self.domains_sigma = torch.randn(self.num_domains, args.c, 1, 1)
        self.domains_mean = torch.randn(self.num_domains, args.c, 1, 1)

    def get_balance_pair(self, batch_size):
        idx1, idx2 = [], []

        for _ in range(batch_size):
            class_idx = np.random.choice(len(self.unique_labels), 1)[0]
            feat_idx = np.random.choice(len(self.labels_indices[class_idx]), 1)[0]
            idx1.append(self.labels_indices[class_idx][feat_idx])

            domain_idx = \
                np.random.choice(len(self.unique_domains), 1)[0]
            feat_idx = \
                np.random.choice(len(self.domains_indices[domain_idx]), 1)[0]
            idx2.append(self.domains_indices[domain_idx][feat_idx])

        idx = idx1 + idx2

        return idx

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.class_num):
            cls_num_list.append((self.labels == i).sum().item())
        return cls_num_list

    def update_statistics(self, norm, sig, mu, idx1, idx2):
        y_unique = self.labels_inverse[idx1].unique()
        d_unique = self.domains_inverse[idx2].unique()

        classes_feat_update = [
            norm[self.labels_inverse[idx1] == cls].mean(dim=0).unsqueeze(dim=0) for cls in
            y_unique]
        classes_feat_update = torch.cat(classes_feat_update)

        domains_mean_update = [
            mu[self.domains_inverse[idx2] == loc].mean(dim=0).unsqueeze(dim=0) for loc in
            d_unique]
        domains_mean_update = torch.cat(domains_mean_update)

        domains_sigma_update = [
            sig[self.domains_inverse[idx2] == loc].mean(dim=0).unsqueeze(dim=0) for loc in
            d_unique]
        domains_sigma_update = torch.cat(domains_sigma_update)

        self.classes_norm[y_unique] = 0.8 * self.classes_norm[y_unique] + 0.2 * classes_feat_update
        self.domains_mean[d_unique] = 0.8 * self.domains_mean[d_unique] + 0.2 * domains_mean_update
        self.domains_sigma[d_unique] = 0.8 * self.domains_sigma[d_unique] + 0.2 * domains_sigma_update

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        domain = self.domains[idx]
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        norm = self.classes_norm[self.labels_inverse[idx]]
        sigma = self.domains_sigma[self.domains_inverse[idx]]
        mean = self.domains_mean[self.domains_inverse[idx]]
        return img, label, domain, idx, norm, sigma, mean

    def eval(self, ypreds, ys, meta):
        acc_per_class = np.zeros(len(self.unique_labels))
        for i, c in enumerate(self.unique_labels):
            correct = np.zeros(len(self.domains[ys == c].unique()))
            num = np.zeros(len(self.domains[ys == c].unique()))
            domain_unique = self.domains[ys == c].unique()
            for j, d in enumerate(domain_unique):
                correct[j] = ((ypreds == c) & (ys == c) & (meta == d)).sum()
                num[j] = ((ys == c) & (meta == d)).sum()
            acc_per_class[i] = (correct / num).mean()

        acc_per_domain = np.zeros(len(self.unique_domains))
        for i, d in enumerate(self.unique_domains):
            correct = np.zeros(len(self.labels[meta == d].unique()))
            num = np.zeros(len(self.labels[meta == d].unique()))
            class_unique = self.labels[meta == d].unique()
            for j, c in enumerate(class_unique):
                correct[j] = ((ypreds == c) & (ys == c) & (meta == d)).sum()
                num[j] = ((ys == c) & (meta == d)).sum()
            acc_per_domain[i] = (correct / num).mean()

        correct_per_pair = np.zeros(len(self.unique_pairs))
        num_per_pair = np.zeros(len(self.unique_pairs))
        for i, p in enumerate(self.unique_pairs):
            correct_per_pair[i] = (ypreds == ys)[(meta * self.class_num + ys) == p].sum()
            num_per_pair[i] = ((meta * self.class_num + ys) == p).sum()

        metrics = [
            Accuracy(prediction_fn=None),
            Recall(prediction_fn=None, average='macro'),
            F1(prediction_fn=None, average='macro'),
        ]

        test_val = \
            {'Average acc': metrics[0].compute(ypreds, ys, False),
             'Recall macro': metrics[1].compute(ypreds, ys, False),
             'F1-macro': metrics[2].compute(ypreds, ys, False),
             'Average Class acc': acc_per_class.mean(),
             'Worst Class acc': acc_per_class.min(),
             'Average Domain acc': acc_per_domain.mean(),
             'Worst Domain acc': acc_per_domain.min(),
             'Average Pair acc': (correct_per_pair / num_per_pair).mean(),
             'Worst Pair acc': (correct_per_pair / num_per_pair).min(),
             }
        return test_val


class OfficeHome(Dataset):
    """
    It consists of 65 classes in 4 domains:
    art, clipart, product, real
    """

    class_num = 65
    domain_names = ["art", "clipart", "product", "real"]

    def __init__(self, args, data_dir="data", split="train", transform=None):
        """
        Arguments:
            root: The dataset must be located at ```<root>/domainnet```
            domain: One of the 6 domains
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        super().__init__()
        self.labels = []
        self.img_paths = []
        self.domains = []
        for i, domain in enumerate(self.domain_names):
            # if i == 3 and split == "train":
            #    continue
            labels_file = os.path.join(data_dir, "officehome", f"{domain}_{split}_{args.split}.txt")
            img_dir = os.path.join(data_dir, "officehome")
            with open(labels_file) as f:
                content = [line.rstrip().split(" ") for line in f]
            self.img_paths = self.img_paths + [os.path.join(img_dir, x[0]) for x in content]
            self.labels = self.labels + [int(x[1]) for x in content]
            self.domains = self.domains + [i for x in content]

        self.labels = torch.Tensor(self.labels).long()
        self.domains = torch.Tensor(self.domains).long()
        self.pairs = self.domains * self.class_num + self.labels

        self.unique_domains, self.domains_inverse = torch.unique(self.domains, return_inverse=True)
        self.unique_labels, self.labels_inverse = torch.unique(self.labels, return_inverse=True)
        self.unique_pairs = torch.unique(self.pairs)

        self.num_domains = len(self.unique_domains)
        self.num_classes = len(self.unique_labels)

        self.domains_indices = [torch.nonzero(self.domains == loc).squeeze(-1) for loc in self.domains.unique()]
        self.labels_indices = [torch.nonzero(self.labels == cls).squeeze(-1) for cls in self.labels.unique()]

        self.transform = transform

        self.classes_norm = torch.randn(self.num_classes, args.c, args.h, args.w)
        self.domains_sigma = torch.randn(self.num_domains, args.c, 1, 1)
        self.domains_mean = torch.randn(self.num_domains, args.c, 1, 1)

    def get_balance_pair(self, batch_size):
        idx1, idx2 = [], []

        for _ in range(batch_size):
            class_idx = np.random.choice(len(self.unique_labels), 1)[0]
            feat_idx = np.random.choice(len(self.labels_indices[class_idx]), 1)[0]
            idx1.append(self.labels_indices[class_idx][feat_idx])

            domain_idx = \
                np.random.choice(len(self.unique_domains), 1)[0]
            feat_idx = \
                np.random.choice(len(self.domains_indices[domain_idx]), 1)[0]
            idx2.append(self.domains_indices[domain_idx][feat_idx])

        idx = idx1 + idx2

        return idx

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.class_num):
            cls_num_list.append((self.labels == i).sum().item())
        return cls_num_list

    def update_statistics(self, norm, sig, mu, idx1, idx2):
        y_unique = self.labels_inverse[idx1].unique()
        d_unique = self.domains_inverse[idx2].unique()

        classes_feat_update = [
            norm[self.labels_inverse[idx1] == cls].mean(dim=0).unsqueeze(dim=0) for cls in
            y_unique]
        classes_feat_update = torch.cat(classes_feat_update)

        domains_mean_update = [
            mu[self.domains_inverse[idx2] == loc].mean(dim=0).unsqueeze(dim=0) for loc in
            d_unique]
        domains_mean_update = torch.cat(domains_mean_update)

        domains_sigma_update = [
            sig[self.domains_inverse[idx2] == loc].mean(dim=0).unsqueeze(dim=0) for loc in
            d_unique]
        domains_sigma_update = torch.cat(domains_sigma_update)

        self.classes_norm[y_unique] = 0.8 * self.classes_norm[y_unique] + 0.2 * classes_feat_update
        self.domains_mean[d_unique] = 0.8 * self.domains_mean[d_unique] + 0.2 * domains_mean_update
        self.domains_sigma[d_unique] = 0.8 * self.domains_sigma[d_unique] + 0.2 * domains_sigma_update

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        domain = self.domains[idx]
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        norm = self.classes_norm[self.labels_inverse[idx]]
        sigma = self.domains_sigma[self.domains_inverse[idx]]
        mean = self.domains_mean[self.domains_inverse[idx]]
        return img, label, domain, idx, norm, sigma, mean

    def eval(self, ypreds, ys, meta):
        acc_per_class = np.zeros(len(self.unique_labels))
        for i, c in enumerate(self.unique_labels):
            correct = np.zeros(len(self.domains[ys == c].unique()))
            num = np.zeros(len(self.domains[ys == c].unique()))
            domain_unique = self.domains[ys == c].unique()
            for j, d in enumerate(domain_unique):
                correct[j] = ((ypreds == c) & (ys == c) & (meta == d)).sum()
                num[j] = ((ys == c) & (meta == d)).sum()
            acc_per_class[i] = (correct / num).mean()

        acc_per_domain = np.zeros(len(self.unique_domains))
        for i, d in enumerate(self.unique_domains):
            correct = np.zeros(len(self.labels[meta == d].unique()))
            num = np.zeros(len(self.labels[meta == d].unique()))
            class_unique = self.labels[meta == d].unique()
            for j, c in enumerate(class_unique):
                correct[j] = ((ypreds == c) & (ys == c) & (meta == d)).sum()
                num[j] = ((ys == c) & (meta == d)).sum()
            acc_per_domain[i] = (correct / num).mean()

        correct_per_pair = np.zeros(len(self.unique_pairs))
        num_per_pair = np.zeros(len(self.unique_pairs))
        for i, p in enumerate(self.unique_pairs):
            correct_per_pair[i] = (ypreds == ys)[(meta * self.class_num + ys) == p].sum()
            num_per_pair[i] = ((meta * self.class_num + ys) == p).sum()

        metrics = [
            Accuracy(prediction_fn=None),
            Recall(prediction_fn=None, average='macro'),
            F1(prediction_fn=None, average='macro'),
        ]

        test_val = \
            {'Average acc': metrics[0].compute(ypreds, ys, False),
             'Recall macro': metrics[1].compute(ypreds, ys, False),
             'F1-macro': metrics[2].compute(ypreds, ys, False),
             'Average Class acc': acc_per_class.mean(),
             'Worst Class acc': acc_per_class.min(),
             'Average Domain acc': acc_per_domain.mean(),
             'Worst Domain acc': acc_per_domain.min(),
             'Average Pair acc': (correct_per_pair / num_per_pair).mean(),
             'Worst Pair acc': (correct_per_pair / num_per_pair).min(),
             }
        return test_val


class TerraIncognita(Dataset):
    """
    A large dataset used in "Moment Matching for Multi-Source Domain Adaptation".
    It consists of 345 classes in 6 domains:
    clipart, infograph, painting, quickdraw, real, sketch
    """

    domain_names = ["location_38", "location_46", "location_100", "location_43", "location_88",
                    "location_33", "location_61", "location_0", "location_115", "location_78",
                    "location_130", "location_28", "location_108", "location_40", "location_90",
                    "location_120", "location_105", "location_7", "location_125", "location_51"]
    class_num = 10

    def __init__(self, args, data_dir="data", split="train", transform=None):
        """
        Arguments:
            root: The dataset must be located at ```<root>/domainnet```
            domain: One of the 6 domains
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        super().__init__()
        self.labels = []
        self.img_paths = []
        self.domains = []
        for i, domain in enumerate(self.domain_names):
            # if i == 3 and split == "train":
            #    continue
            labels_file = os.path.join(data_dir, "terra_incognita", f"{domain}_{split}_{args.split}.txt")
            img_dir = os.path.join(data_dir, "terra_incognita")
            with open(labels_file) as f:
                content = [line.rstrip().split(" ") for line in f]
            self.img_paths = self.img_paths + [os.path.join(img_dir, x[0]) for x in content]
            self.labels = self.labels + [int(x[1]) for x in content]
            self.domains = self.domains + [i for x in content]

        self.labels = torch.Tensor(self.labels).long()
        self.domains = torch.Tensor(self.domains).long()
        self.pairs = self.domains * self.class_num + self.labels

        self.unique_domains, self.domains_inverse = torch.unique(self.domains, return_inverse=True)
        self.unique_labels, self.labels_inverse = torch.unique(self.labels, return_inverse=True)
        self.unique_pairs = torch.unique(self.pairs)

        self.num_domains = len(self.unique_domains)
        self.num_classes = len(self.unique_labels)

        self.domains_indices = [torch.nonzero(self.domains == loc).squeeze(-1) for loc in self.domains.unique()]
        self.labels_indices = [torch.nonzero(self.labels == cls).squeeze(-1) for cls in self.labels.unique()]

        self.transform = transform

        self.classes_norm = torch.randn(self.num_classes, args.c, args.h, args.w)
        self.domains_sigma = torch.randn(self.num_domains, args.c, 1, 1)
        self.domains_mean = torch.randn(self.num_domains, args.c, 1, 1)

    def get_balance_pair(self, batch_size):
        idx1, idx2 = [], []

        for _ in range(batch_size):
            class_idx = np.random.choice(len(self.unique_labels), 1)[0]
            feat_idx = np.random.choice(len(self.labels_indices[class_idx]), 1)[0]
            idx1.append(self.labels_indices[class_idx][feat_idx])

            domain_idx = \
                np.random.choice(len(self.unique_domains), 1)[0]
            feat_idx = \
                np.random.choice(len(self.domains_indices[domain_idx]), 1)[0]
            idx2.append(self.domains_indices[domain_idx][feat_idx])

        idx = idx1 + idx2

        return idx

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.class_num):
            cls_num_list.append((self.labels == i).sum().item())
        return cls_num_list

    def update_statistics(self, norm, sig, mu, idx1, idx2):
        y_unique = self.labels_inverse[idx1].unique()
        d_unique = self.domains_inverse[idx2].unique()

        classes_feat_update = [
            norm[self.labels_inverse[idx1] == cls].mean(dim=0).unsqueeze(dim=0) for cls in
            y_unique]
        classes_feat_update = torch.cat(classes_feat_update)

        domains_mean_update = [
            mu[self.domains_inverse[idx2] == loc].mean(dim=0).unsqueeze(dim=0) for loc in
            d_unique]
        domains_mean_update = torch.cat(domains_mean_update)

        domains_sigma_update = [
            sig[self.domains_inverse[idx2] == loc].mean(dim=0).unsqueeze(dim=0) for loc in
            d_unique]
        domains_sigma_update = torch.cat(domains_sigma_update)

        self.classes_norm[y_unique] = 0.8 * self.classes_norm[y_unique] + 0.2 * classes_feat_update
        self.domains_mean[d_unique] = 0.8 * self.domains_mean[d_unique] + 0.2 * domains_mean_update
        self.domains_sigma[d_unique] = 0.8 * self.domains_sigma[d_unique] + 0.2 * domains_sigma_update

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        domain = self.domains[idx]
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        norm = self.classes_norm[self.labels_inverse[idx]]
        sigma = self.domains_sigma[self.domains_inverse[idx]]
        mean = self.domains_mean[self.domains_inverse[idx]]
        return img, label, domain, idx, norm, sigma, mean

    def eval(self, ypreds, ys, meta):
        acc_per_class = np.zeros(len(self.unique_labels))
        for i, c in enumerate(self.unique_labels):
            correct = np.zeros(len(self.domains[ys == c].unique()))
            num = np.zeros(len(self.domains[ys == c].unique()))
            domain_unique = self.domains[ys == c].unique()
            for j, d in enumerate(domain_unique):
                correct[j] = ((ypreds == c) & (ys == c) & (meta == d)).sum()
                num[j] = ((ys == c) & (meta == d)).sum()
            acc_per_class[i] = (correct / num).mean()

        acc_per_domain = np.zeros(len(self.unique_domains))
        for i, d in enumerate(self.unique_domains):
            correct = np.zeros(len(self.labels[meta == d].unique()))
            num = np.zeros(len(self.labels[meta == d].unique()))
            class_unique = self.labels[meta == d].unique()
            for j, c in enumerate(class_unique):
                correct[j] = ((ypreds == c) & (ys == c) & (meta == d)).sum()
                num[j] = ((ys == c) & (meta == d)).sum()
            acc_per_domain[i] = (correct / num).mean()

        correct_per_pair = np.zeros(len(self.unique_pairs))
        num_per_pair = np.zeros(len(self.unique_pairs))
        for i, p in enumerate(self.unique_pairs):
            correct_per_pair[i] = (ypreds == ys)[(meta * self.class_num + ys) == p].sum()
            num_per_pair[i] = ((meta * self.class_num + ys) == p).sum()

        metrics = [
            Accuracy(prediction_fn=None),
            Recall(prediction_fn=None, average='macro'),
            F1(prediction_fn=None, average='macro'),
        ]

        test_val = \
            {'Average acc': metrics[0].compute(ypreds, ys, False),
             'Recall macro': metrics[1].compute(ypreds, ys, False),
             'F1-macro': metrics[2].compute(ypreds, ys, False),
             'Average Class acc': acc_per_class.mean(),
             'Worst Class acc': acc_per_class.min(),
             'Average Domain acc': acc_per_domain.mean(),
             'Worst Domain acc': acc_per_domain.min(),
             'Average Pair acc': (correct_per_pair / num_per_pair).mean(),
             'Worst Pair acc': (correct_per_pair / num_per_pair).min(),
             }
        return test_val




