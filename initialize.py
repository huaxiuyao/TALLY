import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from wilds.datasets.iwildcam_dataset import IWildCamDataset
from wilds.common.data_loaders import get_eval_loader

from dataset import IWildCam, PACS, VLCS, OfficeHome, DomainNet, TerraIncognita
from tally_sampler import TallySampler


def initialize_dataloader(args, device):
    if args.dataset == "iwildcam":
        dataset = IWildCamDataset(root_dir=args.data_dir, download=True)
        # get all train data
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        train_data = dataset.get_subset('train', transform=transform)
        train_sets = IWildCam(args, train_data, domain_idx=0)
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=transform)
        # get the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}

        train_loader = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)

        sampler = TallySampler(train_sets, batch_size=args.batch_size, iter_num=args.iter_num)
        tally_loader = DataLoader(train_sets, batch_size=2 * args.batch_size, shuffle=False, sampler=sampler, **kwargs)

        tv_loaders = {}
        for split, dataset in datasets.items():
            tv_loaders[split] = get_eval_loader('standard', dataset, batch_size=32)
        return train_loader, tally_loader, tv_loaders
    elif args.dataset == "PACS":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        train_data = PACS(args, args.data_dir, split="train", transform=transform)
        val_data = PACS(args, args.data_dir, split="val", transform=transform)
        test_data = PACS(args, args.data_dir, split="test", transform=transform)

        # get the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}

        train_loaders = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,  **kwargs)

        sampler = TallySampler(train_data, batch_size=args.batch_size, iter_num=args.iter_num)
        tally_loader = DataLoader(train_data, batch_size=2 * args.batch_size, shuffle=False, sampler=sampler, **kwargs)

        tv_loaders = {}
        tv_loaders['val'] = DataLoader(dataset=val_data, batch_size=args.eval_batch_size, shuffle=False)
        tv_loaders['test'] = DataLoader(dataset=test_data, batch_size=args.eval_batch_size, shuffle=False)
        return train_loaders, tally_loader, tv_loaders
    elif args.dataset == "VLCS":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        train_data = VLCS(args, args.data_dir, split="train", transform=transform)
        val_data = VLCS(args, args.data_dir, split="val", transform=transform)
        test_data = VLCS(args, args.data_dir, split="test", transform=transform)

        # get the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}

        train_loaders = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

        sampler = TallySampler(train_data, batch_size=args.batch_size, iter_num=args.iter_num)
        tally_loader = DataLoader(train_data, batch_size=2 * args.batch_size, shuffle=False, sampler=sampler, **kwargs)

        tv_loaders = {}
        tv_loaders['val'] = DataLoader(dataset=val_data, batch_size=args.eval_batch_size, shuffle=False)
        tv_loaders['test'] = DataLoader(dataset=test_data, batch_size=args.eval_batch_size, shuffle=False)
        return train_loaders, tally_loader, tv_loaders
    elif args.dataset == "officehome":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        train_data = OfficeHome(args, args.data_dir, split="train", transform=transform)
        val_data = OfficeHome(args, args.data_dir, split="val", transform=transform)
        test_data = OfficeHome(args, args.data_dir, split="test", transform=transform)

        # get the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}

        train_loaders = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

        sampler = TallySampler(train_data, batch_size=args.batch_size, iter_num=args.iter_num)
        tally_loader = DataLoader(train_data, batch_size=2 * args.batch_size, shuffle=False, sampler=sampler, **kwargs)

        tv_loaders = {}
        tv_loaders['val'] = DataLoader(dataset=val_data, batch_size=args.eval_batch_size, shuffle=False)
        tv_loaders['test'] = DataLoader(dataset=test_data, batch_size=args.eval_batch_size, shuffle=False)
        return train_loaders, tally_loader, tv_loaders
    elif args.dataset == "domainnet":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        train_data = DomainNet(args, args.data_dir, split="train", transform=transform)
        val_data = DomainNet(args, args.data_dir, split="val", transform=transform)
        test_data = DomainNet(args, args.data_dir, split="test", transform=transform)

        # get the loaders
        kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}

        train_loaders = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

        sampler = TallySampler(train_data, batch_size=args.batch_size, iter_num=args.iter_num)
        tally_loader = DataLoader(train_data, batch_size=2 * args.batch_size, shuffle=False, sampler=sampler, **kwargs)

        tv_loaders = {}
        tv_loaders['val'] = DataLoader(dataset=val_data, batch_size=args.eval_batch_size, shuffle=False)
        tv_loaders['test'] = DataLoader(dataset=test_data, batch_size=args.eval_batch_size, shuffle=False)
        return train_loaders, tally_loader, tv_loaders
    elif args.dataset == "terraincognita":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        train_data = TerraIncognita(args, args.data_dir, split="train", transform=transform)
        val_data = TerraIncognita(args, args.data_dir, split="val", transform=transform)
        test_data = TerraIncognita(args, args.data_dir, split="test", transform=transform)

        # get the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}

        train_loaders = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

        sampler = TallySampler(train_data, batch_size=args.batch_size, iter_num=args.iter_num)
        tally_loader = DataLoader(train_data, batch_size=2 * args.batch_size, shuffle=False, sampler=sampler, **kwargs)

        tv_loaders = {}
        tv_loaders['val'] = DataLoader(dataset=val_data, batch_size=args.eval_batch_size, shuffle=False)
        tv_loaders['test'] = DataLoader(dataset=test_data, batch_size=args.eval_batch_size, shuffle=False)
        return train_loaders, tally_loader, tv_loaders
