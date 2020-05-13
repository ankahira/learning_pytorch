#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms

from torchvision.models import resnet50
import matplotlib.pyplot as plt

from torch.utils import checkpoint


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


# Load Data
def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # first we get the training dataset from touchvision
    training_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # We then pass this dataset to our dataloader with specifications such as batch size
    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=256, shuffle=True, num_workers=2)

    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2)

    return training_data_loader, test_data_loader


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset():
    """ Partitioning MNIST """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    training_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    size = dist.get_world_size()
    bsz = 1024 // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(training_data, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True)

    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    partition = DataPartitioner(test_data, partition_sizes)
    partition = partition.use(dist.get_rank())
    test_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True)

    return train_set, test_set


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def train(training_data_loader, model, loss_function, optimizer):
    model.train()
    training_loss = 0.0
    correct = 0
    total = 0
    for iteration, data in enumerate(training_data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Move both inputs and labels to the GPU.
        inputs, labels = inputs.cuda(rank), labels.cuda(rank)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        average_gradients(model)
        optimizer.step()

        training_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        avg_loss = training_loss / (iteration + 1)
        avg_acc = 100. * correct / total
        # print('Rank ', dist.get_rank(), 'Training Loss: %.3f |  Training Acc: %.3f%% ' % (avg_loss, avg_acc))

    return avg_loss, avg_acc


# Create a test function

def test(test_data_loader, model, loss_function):
    # Set the model to Evaluation Mode
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for iteration, data in enumerate(test_data_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(rank), labels.cuda(rank)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            avg_loss = test_loss / (iteration + 1)
            avg_acc = 100. * correct / total
            # print('Testing Loss: %.3f | Testing Acc: %.3f%%' % (avg_loss, avg_acc))
    return avg_loss, avg_acc


def run(rank, size):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    training_data_loader, test_data_loader = partition_dataset()
    model = resnet50(num_classes=10)
    model = model.cuda(rank)
    # Create a loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Training loop
    training_losses = []
    training_accuracies = []
    testing_losses = []
    testing_accuracies = []
    print("Training")
    start = time.time()
    for epoch in range(100):
        #  Move the model to the GPU
        model.cuda(rank)
        training_loss, training_acc = train(training_data_loader, model, loss_function, optimizer)
        training_losses.append(training_loss)
        training_accuracies.append(training_acc)
        test_loss, test_accuracy = test(test_data_loader, model, loss_function)
        testing_losses.append(test_loss)
        testing_accuracies.append(test_accuracy)
        print('Rank ', dist.get_rank(), 'Epoch: {} | Training Loss: {:.3f} | Test Loss: {:.3f} | Test Accuracy: {}'.format(epoch + 1,
                                                                                                 training_loss,
                                                                                                 test_loss,
                                                                                                 test_accuracy
                                                                                                 ))

        checkpoint.checkpoint_sequential()
    end = time.time()
    print("Elapsed time", end - start)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(training_losses, label='train')
    plt.plot(testing_losses, label='test')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy [%]')
    plt.plot(training_accuracies, label='train')
    plt.plot(testing_accuracies, label='test')
    plt.legend()
    plt.savefig('cifar10_single_gpu.png')
    plt.close()


def init_processes(rank, size, fn, backend='nccl'):
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()