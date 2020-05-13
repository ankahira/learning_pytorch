import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3,  padding=(1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=(1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=(1, 1))

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=(1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 512)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = F.softmax(x, dim=1)
        return x

