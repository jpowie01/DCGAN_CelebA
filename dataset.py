import torch
import torchvision
import torchvision.transforms as transforms


def get(batch_size, dataset_directory, dataloader_workers):
    # Prepare dataset for training
    train_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=dataset_directory, train=True,
                                                 download=True, transform=train_transformation)

    # Split training dataset into real training and validation
    training_sampler = torch.utils.data.SubsetRandomSampler(range(len(train_dataset)))

    # Prepare Data Loaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=training_sampler,
                                               pin_memory=True, num_workers=dataloader_workers)

    return train_dataset, train_loader
