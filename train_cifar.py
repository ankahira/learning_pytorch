import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import time

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data.distributed

from torchvision.models import resnet50
import matplotlib.pyplot as plt

# Set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


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


# Create a function to train a single epoch
def train(training_data_loader, model, loss_function, optimizer):
    model.train()
    training_loss = 0.0
    correct = 0
    total = 0
    for iteration, data in enumerate(training_data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Move both inputs and labels to the GPU.
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        avg_loss = training_loss / (iteration + 1)
        avg_acc = 100. * correct / total
        print('Training Loss: %.3f |  Training Acc: %.3f%% ' % (avg_loss, avg_acc))

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
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            avg_loss = test_loss / (iteration + 1)
            avg_acc = 100. * correct / total
            print('Testing Loss: %.3f | Testing Acc: %.3f%%' % (avg_loss, avg_acc))
    return avg_loss, avg_acc


def main():
    # Load the data
    training_data_loader, test_data_loader = load_data()
    # Create the model
    # model = Net()
    model = resnet50(num_classes=10)

    # Create a loss function
    loss_function = nn.CrossEntropyLoss()

    # Create an optimiser
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Training loop
    training_losses = []
    training_accuracies = []
    testing_losses = []
    testing_accuracies = []
    print("Training")
    start = time.time()
    for epoch in range(100):
        #  Move the model to the GPU
        model.to(device)
        training_loss, training_acc = train(training_data_loader, model, loss_function, optimizer)
        training_losses.append(training_loss)
        training_accuracies.append(training_acc)
        test_loss, test_accuracy = test(test_data_loader, model, loss_function)
        testing_losses.append(test_loss)
        testing_accuracies.append(test_accuracy)
        print('Epoch: {} | Training Loss: {:.3f} | Test Loss: {:.3f} | Test Accuracy: {}'.format(epoch + 1,
                                                                                                 training_loss,
                                                                                                 test_loss,
                                                                                                 test_accuracy
                                                                                                 ))

    end = time.time()
    print("Elapsed time", end-start)

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


if __name__ == '__main__':
    main()
