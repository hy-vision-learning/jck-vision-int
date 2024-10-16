import torch
import torchvision
import numpy as np

from sklearn.model_selection import train_test_split

import logging


class DataPreProcessor:
    def __init__(self, parallel=0):
        self.__logger = logging.getLogger('main')
        self.__parallel = parallel
        
        self.data_mean, self.data_std = self.__data_mean_std()
        
        self.__trainset_original = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=None)
        self.__trainset_augment = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=None)
        self.__testset = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=None)
        
        self.__logger.debug('data preprocessor init')
        
    def __data_mean_std(self):
        data = torchvision.datasets.CIFAR100("./data", train=True, download=True)
        x = np.concatenate([np.asarray(data[i][0]) for i in range(len(data))])

        mean = np.mean(x, axis=(0, 1))/255
        std = np.std(x, axis=(0, 1))/255

        mean = mean.tolist()
        std = std.tolist()
        self.__logger.debug(f'data mean: {mean}\tdata std: {std}')
        return mean, std
    
    def transform_data(self, train_original, train_augment, test):
        self.__trainset_original.transform = train_original
        self.__trainset_augment.transform = train_augment
        self.__testset.transform = test
        self.__logger.debug(f'set transform')
        
        
    def split_data(self, p=0.2):
        targets = self.__trainset_original.targets

        train_idx, val_idx = [], []
        for cls in range(100):
            cls_idx = [i for i, t in enumerate(targets) if t == cls]
            cls_train_idx, cls_val_idx = train_test_split(cls_idx, test_size=p, random_state=42)
            train_idx.extend(cls_train_idx)
            val_idx.extend(cls_val_idx)

        train_dataset_original = torch.utils.data.Subset(self.__trainset_original, train_idx)
        val_dataset_original = torch.utils.data.Subset(self.__trainset_original, val_idx)
        train_dataset_augment = torch.utils.data.Subset(self.__trainset_augment, train_idx)
        val_dataset_augment = torch.utils.data.Subset(self.__trainset_augment, val_idx)

        self.__train_dataset = torch.utils.data.ConcatDataset([train_dataset_original, train_dataset_augment])
        self.__val_dataset = torch.utils.data.ConcatDataset([val_dataset_original, val_dataset_augment])

        self.__logger.debug(f'data split - train size: {len(self.__train_dataset)}\tvalidation size: {len(self.__val_dataset)}')

    def get_data_loader(self, batch_size, num_worker=0):
        if self.__parallel == 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=self.__train_dataset, shuffle=True)
            batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=False)
            trainloader = torch.utils.data.DataLoader(self.__train_dataset, batch_sampler=batch_sampler_train, num_workers=num_worker, pin_memory=True)
            
            validation_sampler = torch.utils.data.distributed.DistributedSampler(dataset=self.__val_dataset, shuffle=True)
            batch_sampler_validation = torch.utils.data.BatchSampler(validation_sampler, batch_size, drop_last=False)
            validationloader = torch.utils.data.DataLoader(self.__val_dataset, batch_sampler=batch_sampler_validation, num_workers=num_worker, pin_memory=True)
            
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset=self.__testset, shuffle=True)
            batch_sampler_test = torch.utils.data.BatchSampler(test_sampler, batch_size, drop_last=False)
            testloader = torch.utils.data.DataLoader(self.__testset, batch_sampler=batch_sampler_test, num_workers=num_worker, pin_memory=True)
        else:
            trainloader = torch.utils.data.DataLoader(self.__train_dataset, batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)
            validationloader = torch.utils.data.DataLoader(self.__val_dataset, batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)
            testloader = torch.utils.data.DataLoader(self.__testset, batch_size*2,pin_memory=True, num_workers=0)
        
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.testloader = testloader
