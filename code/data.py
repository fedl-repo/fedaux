import torch, torchvision
import numpy as np
import os, pickle


def get_cifar10(path):
  transforms = torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                          ])
  train_data = torchvision.datasets.CIFAR10(root=path+"CIFAR", train=True, download=True, transform=transforms)
  test_data = torchvision.datasets.CIFAR10(root=path+"CIFAR", train=False, download=True, transform=transforms)

  return train_data, test_data


def get_cifar100(path):
  transforms = torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                          ])
  train_data = torchvision.datasets.CIFAR100(root=path+"CIFAR100", train=True, download=True, transform=transforms)
  test_data = torchvision.datasets.CIFAR100(root=path+"CIFAR100", train=False, download=True, transform=transforms)

  return train_data, test_data


def get_cifar100_distill(path):
  transforms = torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                          ])
  train_data = torchvision.datasets.CIFAR100(root=path+"CIFAR100", train=True, download=True, transform=transforms)
  test_data = torchvision.datasets.CIFAR100(root=path+"CIFAR100", train=False, download=True, transform=transforms)

  return torch.utils.data.ConcatDataset([train_data, test_data])



def get_stl10(path):
  transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                               ])

  data = torchvision.datasets.STL10(root=path+"STL10", split='unlabeled', folds=None, 
                             transform=transforms,
                                    download=True)
  return data





def get_data(dataset, path):
  return {"cifar10" : get_cifar10, "stl10" : get_stl10,"cifar100" : get_cifar100, "cifar100_distill" : get_cifar100_distill}[dataset](path)


def get_loaders(train_data, test_data, n_clients=10, alpha=0, batch_size=128, n_data=None, num_workers=0, seed=0):

  subset_idcs = split_dirichlet(train_data.targets, n_clients, n_data, alpha, seed=seed)
  client_data = [torch.utils.data.Subset(train_data, subset_idcs[i]) for i in range(n_clients)]


  client_loaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers) for subset in client_data]
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, num_workers=num_workers)

  return client_loaders, test_loader




def split_dirichlet(labels, n_clients, n_data, alpha, double_stochstic=True, seed=0):
    '''Splits data among the clients according to a dirichlet distribution with parameter alpha'''

    np.random.seed(seed)

    if isinstance(labels, torch.Tensor):
      labels = labels.numpy()
    n_classes = np.max(labels)+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    if double_stochstic:
      label_distribution = make_double_stochstic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels)==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    print_split(client_idcs, labels)
  
    return client_idcs


def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0 
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    return x



def print_split(idcs, labels):
  n_labels = np.max(labels) + 1 
  print("Data split:")
  splits = []
  for i, idccs in enumerate(idcs):
    split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
    splits += [split]
    if len(idcs) < 30 or i < 10 or i>len(idcs)-10:
      print(" - Client {}: {:55} -> sum={}".format(i,str(split), np.sum(split)), flush=True)
    elif i==len(idcs)-10:
      print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)

  print(" - Total:     {}".format(np.stack(splits, axis=0).sum(axis=0)))
  print()




class IdxSubset(torch.utils.data.Dataset):

    def __init__(self, dataset, indices, return_index):
        self.dataset = dataset
        self.indices = indices
        self.return_index = return_index

    def __getitem__(self, idx):
        if self.return_index:
          return self.dataset[self.indices[idx]], idx
        else:
          return self.dataset[self.indices[idx]]#, idx

    def __len__(self):
        return len(self.indices)







